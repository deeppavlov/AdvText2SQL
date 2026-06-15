# Горизонт-1 — операционный плейбук (делай по порядку)

Это пошаговый сценарий: от подготовки окружения до сбора цифр для отчёта.
Выполняй фазы последовательно. Всё запускать из корня репозитория:
`/Users/vlad/Desktop/uni/maga/2_quarter_project/AdvText2SQL`

Легенда ресурсов: 💻 локально · 🔌 SSH-туннель+БД · 🐳 Docker · ☁️ Colab/GPU · 💰 LLM-API

---

## Фаза 0. Подготовка окружения 💻

0.1. Проверь, что заполнен `.env` (он уже есть): `DB_USER`, `DB_PASS`,
`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL_NAME`.

0.2. Поставь зависимости (вкл. опц. группы serve/synth):
```bash
uv sync --group serve --group synth
```

0.3. Прогон preflight (без БД — просто env+deps):
```bash
uv run python scripts/preflight.py
```
Если ругается на SSH-туннель — это норм на Фазе 0, туннель поднимем в Фазе 3/4.

**Готово, если:** preflight зелёный по env и deps.

---

## Фаза 1. Санити-проверка сборки 💻 (≈3 мин)

Перед реальными прогонами убедись, что код здоров.

> ⚠️ `ruff check .` покажет ~51 предобразующую ошибку в чужих файлах
> (бенчмарки/ноутбуки/старые скрипты) — это не код Г1. Линтим только Г1:
```bash
ruff check \
  src/adv_text2sql/training/seq_len.py src/adv_text2sql/training/dataset_builder.py \
  src/adv_text2sql/cli/main.py src/adv_text2sql/serve/metrics_exporter.py \
  src/adv_text2sql/profiler/sample_collector.py src/adv_text2sql/profiler/cli.py \
  scripts/_registry.py scripts/exp_generate.py scripts/exp_build.py tests/

uv run python -m pytest tests/ -m "not integration" -q
```
**Готово, если:** `All checks passed!` и `48 passed, 3 deselected`.

> Полная проверка отдельных фич (max_seq_len, exporter, harness) — в
> [VERIFY_HORIZON1.md](VERIFY_HORIZON1.md), уровень A. Здесь не дублируем.

---

## Фаза 2. Деплой observability (пункт 1.1) 🐳

Цель: увидеть продуктовые метрики качества в Grafana.

2.1. Подготовь источник метрик. Если `data/logs/queries.jsonl` ещё пуст —
налей тестовых строк (или сгенерируй реальный трафик через MCP-сервер позже):
```bash
mkdir -p data/logs
printf '%s\n' \
  '{"status":"success","latency_ms":120.5,"db_id":"card_games"}' \
  '{"status":"error_execute","latency_ms":80.0,"db_id":"card_games"}' \
  '{"status":"success","latency_ms":300.0,"db_id":"card_games"}' \
  >> data/logs/queries.jsonl
```

2.2. Подними observability-сервисы (без vLLM — ему нужен GPU):
```bash
cd src/adv_text2sql/serve
LOG_DIR=$PWD/../../../data/logs \
  docker compose --profile observability up prometheus metrics-exporter grafana
```

2.3. Открой и проверь:
- **Grafana** → http://localhost:3000 → дашборд **«Text2SQL — Quality & Serving»**.
  Панели success-rate / requests-min / latency наполняются из `queries.jsonl`.
  Панель «vLLM requests» пустая — это ожидаемо (vLLM не поднят).
- **Prometheus** → http://localhost:9090/targets → job `text2sql_quality` = `UP`.
- **Exporter** → http://localhost:9105/metrics → строки `text2sql_*`.

2.4. Останов:
```bash
docker compose down            # из той же папки src/adv_text2sql/serve
```

**Что записать для отчёта:** скриншот дашборда + значение success-rate. Это
демонстрация, что pipeline наблюдаем.

> Боевой вариант (с vLLM): когда есть GPU-машина и обученный адаптер —
> `ADAPTER_DIR=<путь> docker compose --profile observability up` поднимет всё,
> включая vLLM; тогда заполнятся и инфра-панели.

---

## Фаза 3. Масштаб на крупной БД (пункт 1.3) 🔌

Цель: прогнать профайлер на самой большой схеме сервера и зафиксировать,
где упирается масштаб.

3.1. Подними SSH-туннель (в отдельном окне терминала, не закрывай):
```bash
ssh -N -L 5444:10.11.1.6:5444 <user>@lnsigo.mipt.ru -p2278
```

3.2. Найди самую крупную БД. Профилируй несколько кандидатов в schema-only
режиме (быстро, без статистики и DISTINCT) и сравни `complexity_score`:
```bash
for DB in card_games financial formula_1 european_football_2 codebase_community; do
  echo "=== $DB ===";
  uv run --env-file .env text2sql profile \
    --db-url postgresql+psycopg://$DB_USER:$DB_PASS@localhost:5444/$DB \
    --skip-stats --skip-low-cardinality 2>/dev/null | grep -E "tables,|complexity";
done
```
Выбери БД с максимумом таблиц/колонок и `complexity_score`. Назовём её `<BIG>`.

3.3. Замерь время на «дорогом» прогоне. Сначала smoke (дешёвый шаг выключен):
```bash
time uv run --env-file .env text2sql profile \
  --db-url postgresql+psycopg://$DB_USER:$DB_PASS@localhost:5444/<BIG> \
  --skip-low-cardinality
```
Затем полный (для сравнения):
```bash
time uv run --env-file .env text2sql profile \
  --db-url postgresql+psycopg://$DB_USER:$DB_PASS@localhost:5444/<BIG>
```

3.4. Проверь подбор контекста под эту схему:
```bash
uv run text2sql train --db-id <BIG> --max-seq-len auto
```
Смотри строку `~N токенов → max_seq_len=...`. Если увидишь
`overflow / усечь column_stats` — это и есть найденный предел масштаба.

**Что записать:** число таблиц/колонок, `complexity_score`, время smoke vs полный,
оценку токенов промпта и выбранный `max_seq_len` (или факт overflow).

---

## Фаза 4. Эксперимент adaptive vs fixed (пункт 1.2) 🔌💰☁️

Цель: сравнить точность фиксированного бюджета синтетики против адаптивного.
Детали дизайна — [experiments/TARGET_COUNT_VALIDATION.md](experiments/TARGET_COUNT_VALIDATION.md).

Предусловие: туннель из Фазы 3 поднят; `LLM_*` env заданы.

4.1. Генерация + валидация обоих вариантов (отдельный registry, основной не трогаем):
```bash
uv run --env-file .env python scripts/exp_generate.py \
  --registry experiments/registry_target_count.json
```
Создаст `data/exp/tc_fixed_1000/validated.jsonl` (~1000 пар) и
`data/exp/tc_auto/validated.jsonl` (~1391 пара).

4.2. Сборка train/val для обоих:
```bash
uv run --env-file .env python scripts/exp_build.py \
  --registry experiments/registry_target_count.json
```
Создаст `data/exp/<name>/{train,val}.jsonl`.

4.3. Сгенерируй ноутбуки и залей данные на Drive. Для каждого эксперимента:
```bash
# fixed
uv run text2sql train --db-id tc_fixed_1000 \
  --profile data/profiles/card_games/profile.json --max-seq-len auto
# auto
uv run text2sql train --db-id tc_auto \
  --profile data/profiles/card_games/profile.json --max-seq-len auto
```
Затем вручную: залей `data/exp/tc_fixed_1000/{train,val}.jsonl` на Drive в
`MyDrive/text2sql_finetune/tc_fixed_1000/`, и аналогично для `tc_auto`.

4.4. ☁️ Обучи обе модели: открой `notebooks/auto_train_tc_fixed_1000.ipynb` и
`notebooks/auto_train_tc_auto.ipynb` в Colab (Runtime: GPU A100) → Run all.
Скачай по адаптеру на каждую (~1 час на модель).

4.5. Eval каждой модели на BIRD subset. Нужен поднятый vLLM с соответствующим
адаптером (🐳 на GPU-машине: `ADAPTER_DIR=<adapter> docker compose up vllm`),
затем:
```bash
uv run text2sql eval --profile data/profiles/card_games/profile.json \
  --gold data/bird_large.json --model-url http://localhost:8000/v1 \
  --model-name <adapter-name> \
  --db-url postgresql+psycopg://$DB_USER:$DB_PASS@localhost:5444/card_games
```
Отчёт пишется в `experiments/`.

**Что записать:** execution accuracy у `tc_fixed_1000` (1000 пар) vs `tc_auto`
(~1391 пара) + sample-efficiency = accuracy / #пар. Вывод: даёт ли адаптивный
объём выигрыш, или фиксированный не хуже при меньших затратах.

---

## Фаза 5. Свод результатов

Собери в один отчёт (например, дополни `experiments/final_results.md`):
1. **1.1** — скриншот Grafana + success-rate (наблюдаемость работает).
2. **1.3** — таблица: таблицы/колонки, complexity, время профайла, токены промпта,
   max_seq_len/overflow (масштабируемость измерена).
3. **1.2** — таблица: вариант → #пар → accuracy → sample-efficiency (валидация
   адаптивного target_count).

---

## Шпаргалка: минимальный путь без GPU

Если сейчас нет GPU/Colab, реально провести и задокументировать **Фазы 0–3**
полностью (настройка, санити, observability, масштаб профайлера) — это уже 2 из
3 пунктов Г1. Фаза 4 (обучение+eval) ставится в очередь под доступ к Colab.

## Типичные проблемы
- `connection refused :5444` → не поднят SSH-туннель (Фаза 3.1).
- preflight ругается на LLM_* → проверь `.env` и что запускаешь с `--env-file .env`.
- Grafana пустая → `data/logs/queries.jsonl` пуст или не примонтирован `LOG_DIR`.
- `docker compose` поднимает vLLM и падает без GPU → запускай сервисы поимённо:
  `... up prometheus metrics-exporter grafana` (как в 2.2).
