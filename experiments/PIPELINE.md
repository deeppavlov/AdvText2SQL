# Пайплайн эксперимента v2 — один скрипт на этап

Единый источник правды — `experiments/registry.json`. Имя эксперимента = имя
папки на всех платформах. Меняешь реестр → меняется везде.

## Соглашение о путях
```
Локально:  data/exp/<name>/{raw_<lang>.jsonl, validated.jsonl, train.jsonl, val.jsonl, eval_predictions.json}
На Drive:  MyDrive/text2sql/<name>/{train.jsonl, val.jsonl, adapter/, eval_predictions.json}
           MyDrive/text2sql/{registry.json, bird_large.json}   ← загрузить вручную
```

## Этапы

| # | Платформа | Команда | Что делает |
|---|---|---|---|
| 1 | локально | `uv run --env-file .env python scripts/exp_generate.py` | генерит + валидирует синтетику (RU/EN) для всех экспериментов |
| 2 | локально | `uv run --env-file .env python scripts/exp_build.py` | собирает train.jsonl/val.jsonl |
| — | вручную | загрузить `data/exp/*` + `registry.json` + `bird_large.json` на Drive | |
| 3 | Colab | `notebooks/train_all_experiments.ipynb` | обучает LoRA-адаптеры (все эксперименты, один сеанс) |
| 4 | Colab | `notebooks/eval_all_experiments.ipynb` | инференс baseline + все адаптеры на BIRD (с evidence) |
| — | вручную | скачать `eval_predictions.json` в `data/exp/<name>/` | |
| 5 | локально | `uv run --env-file .env python scripts/compare_generators.py` | execution accuracy → `generator_comparison.md` |

Каждый этап-скрипт принимает `--only <name>` для одного эксперимента.

## Что исправлено по сравнению с run1 (русская синтетика)
- **Язык**: `--language ru|en` — синтетика на языке теста (фикс главной причины).
- **Evidence**: инференс подаёт BIRD-подсказку (снят структурный потолок ~20%).
- **max_seq_length=12288**: нет обрезки промпта (был баг с 4096).
- **batch=1, packing=False**: ответ не теряется, нет коллапса в повторы.
- **Единый реестр**: согласованные имена папок/файлов на всех платформах.

## Как добавить эксперимент
Допиши объект в `experiments` в `registry.json`:
```json
{"name": "haiku_en", "llm_model": "claude-haiku", "languages": ["en"], "count_per_lang": 500}
```
Все скрипты и ноутбуки подхватят его автоматически.
