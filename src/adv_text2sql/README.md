# Autonomous Text2SQL Pipeline

Автоматический pipeline: даёшь PostgreSQL-URL → за ~30 минут получаешь
inference-сервис, обученный под твою конкретную БД.

## Архитектура

```
┌──────────────┐   ┌─────────────────┐   ┌──────────────┐   ┌─────────────┐
│  PROFILE     │ → │  GENERATE       │ → │  TRAIN       │ → │  SERVE      │
│              │   │                 │   │              │   │             │
│ DB → JSON    │   │ profile → Q-SQL │   │ JSONL → LoRA │   │ vLLM/Docker │
└──────────────┘   └─────────────────┘   └──────────────┘   └─────────────┘
                                                                     ↓
                                                            ┌─────────────┐
                                                            │  HEAL       │
                                                            │             │
                                                            │ failed→fix  │
                                                            └─────────────┘
```

Каждая стадия — отдельный CLI-subcommand, читает артефакт предыдущей,
пишет следующий. Стадии **идемпотентны** и **независимы** — можно
останавливать/возобновлять/дебажить каждую отдельно.

## Папки

| Module | Что делает | Главные классы |
|---|---|---|
| `profiler/` | Извлекает schema, relationships, column stats, samples из БД | `DBConnector`, `SchemaExtractor`, `StatsCollector`, `SampleCollector`, `Profile` |
| `synth/` | Генерирует Q-SQL пары (шаблоны + LLM), валидирует, дедуплицирует | `TemplateSyntheticGenerator`, `LLMSyntheticGenerator`, `Validator`, `LLMJudge` |
| `training/` | Собирает chat-format JSONL для FT, генерирует Colab notebook | `DatasetBuilder`, `generate_finetune_notebook` |
| `serve/` | vLLM Docker-образ, Python SDK для клиента, query logger | `Text2SQLClient`, `QueryLogger` |
| `heal/` | Сбор failed-запросов, корректирующая регенерация SQL через GPT-4 | `HealCollector`, `HealSQLGenerator` |
| `cli/` | Top-level Typer-приложение | `app` (entry point `text2sql`) |

## Поток артефактов

```
data/profiles/<db_id>/profile.json              ← PROFILE
                ↓
data/synthetic/<db_id>_raw_*.jsonl              ← GENERATE (raw)
                ↓
data/synthetic/<db_id>_*_validated.jsonl        ← GENERATE (после Validator)
                ↓
data/finetune/<db_id>/{train,val}.jsonl         ← BUILD-DATASET
                ↓
notebooks/auto_train_<db_id>.ipynb              ← TRAIN (template)
                ↓ [запускается в Colab → ~1 час]
models/<db_id>/adapter/                         ← LoRA веса
                ↓
docker-compose up vllm                          ← SERVE
                ↓
data/logs/{queries,failed}.jsonl                ← logging
                ↓
data/heal/<db_id>_corrections.jsonl             ← HEAL (опц.)
```

## Быстрый старт

```bash
# Одной командой (если есть SSH-туннель + LLM_API_KEY):
uv run text2sql init --db-url postgresql+psycopg://user:pass@localhost:5444/card_games

# По стадиям:
uv run text2sql profile --db-url postgresql+psycopg://.../card_games
uv run text2sql generate --profile data/profiles/card_games/profile.json --count auto --generator both
uv run text2sql build-dataset --profile data/profiles/card_games/profile.json \
    --synthetic data/synthetic/card_games_raw_both_validated.jsonl
uv run text2sql train --db-id card_games   # генерит Colab notebook

# После FT в Colab → adapter скачан в models/card_games/adapter/:
cd src/adv_text2sql/serve/
ADAPTER_DIR=../../../models/card_games/adapter docker-compose up -d

# REPL клиент:
uv run text2sql serve --profile data/profiles/card_games/profile.json \
    --model-url http://localhost:8000/v1 --model-name card_games \
    --db-url postgresql+psycopg://.../card_games
```

## Главный инвариант: train == inference

`schema_str`, `relationships_str`, `column_stats_str` в `profile.json` должны
**byte-to-byte** совпадать с тем, что `Text2SQLGenerator.build()` (на инференсе)
кладёт в `self.db_schema` и т.д. — иначе модель учится на одном формате, а
видит другой → деградация accuracy.

Это гарантируется тем, что:
1. `profiler/schema_extractor.py` — pure functions, переиспользуемые
   `Text2SQLGenerator._get_db_schema_light` (через копию логики)
2. `Text2SQLGenerator.load_from_profile()` берёт строки **прямо из** JSON,
   ничего не пересчитывая
3. Тест `tests/test_profiler.py::test_schema_str_invariant` проверяет это
   на реальной БД, а `tests/test_training.py::test_dataset_builder_invariant_with_text2sql_generator`
   проверяет инвариант на unit-уровне без БД

## Адаптивный размер датасета

`Profile.target_synthetic_count()` вычисляет нужное число синтетических
примеров на основе сложности схемы:

```
complexity_score = log(#tables + 1) × sqrt(#columns) × (1 + #fk / #tables)
target_count = max(200, complexity_score × 50)
```

| БД | Tables | Columns | FK | Score | Target |
|---|---|---|---|---|---|
| Sakila | 16 | 90 | 22 | ~12 | 600 |
| card_games | 6 | 113 | 2 | ~9 | 450 |
| financial | 8 | 58 | 8 | ~10 | 500 |

Если в реальности после execution-фильтра остаётся <40% → CLI с
`--auto-resize` удваивает target и регенерит.

## Эксперимент: какая модель лучше для генерации?

Запускаем `text2sql generate --generator llm --llm-model X` с разными X:

| X | Цена | Pass rate | Final BIRD acc |
|---|---|---|---|
| `gpt-4o` | ~$3 / 500 | ? | ? |
| `claude-haiku` | ~$0.5 / 500 | ? | ? |
| `qwen-coder-32b` (local) | $0 | ? | ? |
| `qwen-coder-7b` (local) | $0 | ? | ? |

Результаты — в `experiments/generator_comparison.md`. Это центральный
исследовательский контрибьюшен: нужно ли платить за GPT-4 для синтетики,
или достаточно локальной 7B-модели?

## Тестирование

```bash
uv run python -m pytest tests/ -v -m "not integration"
# 29 unit-тестов, ~1 сек

uv run python -m pytest tests/ -v -m integration
# 3 integration-теста; требуют SSH-туннель + DB_USER/DB_PASS env
```
