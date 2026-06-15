# Проверка Горизонта-1 — пошаговая инструкция

Два уровня. **Уровень A** — без внешних ресурсов (~5 минут, всё на твоём ноуте).
**Уровень B** — с БД (SSH-туннель), Docker и Colab.

Все команды запускать из корня репозитория:
`/Users/vlad/Desktop/uni/maga/2_quarter_project/AdvText2SQL`

---

## Уровень A — без внешних ресурсов

### A1. Линт + все unit-тесты

> ⚠️ **Не используй `ruff check .`** — в репозитории есть ~51 предобразующая
> lint-ошибка в файлах, не относящихся к Горизонту-1 (бенчмарки, ноутбуки,
> старые скрипты, mcp-сервер). Они были до этой работы. Линт проверяем **только
> по коду Горизонта-1**:

```bash
ruff check \
  src/adv_text2sql/training/seq_len.py \
  src/adv_text2sql/training/dataset_builder.py \
  src/adv_text2sql/cli/main.py \
  src/adv_text2sql/serve/metrics_exporter.py \
  src/adv_text2sql/profiler/sample_collector.py \
  src/adv_text2sql/profiler/cli.py \
  scripts/_registry.py scripts/exp_generate.py scripts/exp_build.py \
  tests/test_seq_len.py tests/test_metrics_exporter.py \
  tests/test_sample_collector.py tests/test_registry.py

uv run python -m pytest tests/ -m "not integration" -q
```
**Ожидается:** `All checks passed!` и `48 passed, 3 deselected`.
(3 deselected — это integration-тесты, требующие БД; они на уровне B.)

### A2. Динамический `max_seq_len` (общий фикс + было: немой дефолт 4096)
```bash
uv run text2sql train --db-id card_games --max-seq-len auto
grep -o "MAX_SEQ_LEN = [0-9]*" notebooks/auto_train_card_games.ipynb | sort -u
```
**Ожидается:**
```
max_seq_len: ~6382 токенов → max_seq_len=8192
...
MAX_SEQ_LEN = 8192
```
Смысл: для card_games промпт ~6382 токена → подобрался bucket 8192 (раньше
команда молча ставила 4096 и обрезала бы промпт). Проверь, что значение в
ноутбуке совпало с рекомендованным.

Дополнительно — поведение на «огромной» схеме (overflow) без БД:
```bash
uv run python -c "
from adv_text2sql.profiler.profile import Profile
from adv_text2sql.training.seq_len import recommend_max_seq_len
p = Profile(db_id='big', db_uri_redacted='x', tables=['t'],
            column_info={'t':[{'name':'c'}]},
            relationships={'foreign_keys':[],'primary_keys':{}},
            schema_str='', relationships_str='', column_stats_str='x'*60000)
r = recommend_max_seq_len(p)
print(r.max_seq_len, r.overflow, '|', r.message)
"
```
**Ожидается:** `16384 True | ...усеч...` — то есть на схеме, чей промпт пробивает
потолок, флаг `overflow=True` и предупреждение про усечение `column_stats`.

### A3. Exporter продуктовых метрик (1.1) — standalone, без Docker
```bash
mkdir -p /tmp/t2sql_logs
printf '%s\n' \
  '{"status":"success","latency_ms":120.5}' \
  '{"status":"error_execute","latency_ms":80.0}' \
  '{"status":"success","latency_ms":300.0}' \
  > /tmp/t2sql_logs/queries.jsonl

uv pip install prometheus-client -q
uv run python src/adv_text2sql/serve/metrics_exporter.py \
    --log-file /tmp/t2sql_logs/queries.jsonl --port 9137 --poll-interval 1 &
sleep 3
curl -s http://localhost:9137/metrics | grep -E '^text2sql_' | grep -v '#'
kill %1
```
**Ожидается** (ключевое):
```
text2sql_queries_total{status="success"} 2.0
text2sql_queries_total{status="error_execute"} 1.0
text2sql_failed_total 1.0
text2sql_success_rate 0.6666666666666666
```
2 успеха / 1 ошибка → success_rate ≈ 0.667, latency-гистограмма заполнена.

### A4. Harness adaptive vs fixed (1.2) — резолв count
```bash
uv run python -c "
import sys; sys.path.insert(0,'scripts')
from _registry import load_registry
from adv_text2sql.profiler.profile import Profile
reg = load_registry('experiments/registry_target_count.json')
p = Profile.load_json(reg.profile_path)
print('complexity =', p.complexity_score, '| total target =', p.target_synthetic_count())
for e in reg.experiments:
    tag = 'adaptive' if e.is_adaptive else 'fixed'
    print(f'  {e.name}: {e.resolved_count_per_lang(p)}/язык [{tag}]')
"
```
**Ожидается:**
```
complexity = 27.82 | total target = 1391
  tc_fixed_1000: 500/язык [fixed]
  tc_auto: 695/язык [adaptive]
```
`auto` делит общий бюджет 1391 между двумя языками → 695/язык.

### A5. Новый CLI-флаг профайлера (1.3)
```bash
uv run text2sql profile --help | grep -A1 skip-low
```
**Ожидается:** строка `--skip-low-cardinality  Skip per-column DISTINCT pass ...`.

---

## Уровень B — с внешними ресурсами

### B1. Observability-стек в Docker (1.1)
Поднимаем только observability-сервисы (без vLLM — ему нужен GPU):
```bash
cd src/adv_text2sql/serve
LOG_DIR=$PWD/../../../data/logs \
  docker compose --profile observability up prometheus metrics-exporter grafana
```
Затем:
- **Grafana:** http://localhost:3000 → дашборд **«Text2SQL — Quality & Serving»**
  (anonymous Admin, логин не нужен). Продуктовые панели (success-rate,
  requests/min, latency) наполнятся из реального `data/logs/queries.jsonl`.
  Панель «vLLM requests» будет пустой — это норм, vLLM не поднят.
- **Prometheus targets:** http://localhost:9090/targets → job `text2sql_quality`
  должен быть `UP` (job `vllm` будет `DOWN` без vLLM — ожидаемо).
- **Exporter напрямую:** http://localhost:9105/metrics.

Если `data/logs/queries.jsonl` пуст — сначала сгенерируй трафик через MCP-сервер
(`src/adv_text2sql/mcp_servers/text2sql_tool/main.py`) или просто скопируй туда
тестовые строки как в A3. Останов: `Ctrl-C`, затем `docker compose down`.

### B2. Масштабируемость профайлера (1.3)
Нужен SSH-туннель к BIRD-серверу:
```bash
ssh -N -L 5444:10.11.1.6:5444 <user>@lnsigo.mipt.ru -p2278   # в отдельном окне
```
1. Найди самую крупную БД на сервере (по числу таблиц/колонок) — это и есть
   «enterprise»-прокси.
2. **Smoke** (быстро, дорогой шаг выключен):
   ```bash
   time uv run --env-file .env text2sql profile \
       --db-url postgresql+psycopg://$DB_USER:$DB_PASS@localhost:5444/<db> \
       --skip-low-cardinality
   ```
3. **Полный** прогон (для сравнения времени):
   ```bash
   time uv run --env-file .env text2sql profile \
       --db-url postgresql+psycopg://$DB_USER:$DB_PASS@localhost:5444/<db>
   ```
**На что смотреть:** общее время и строка `complexity_score → adaptive
target_count`. Затем проверь размер промпта/выбор контекста:
```bash
uv run text2sql train --db-id <db> --max-seq-len auto   # напечатает ~N токенов → max_seq_len
```
Если увидишь `overflow` / `усечь column_stats` — это и есть искомый
scalability-предел (фиксируй для отчёта).

### B3. Эксперимент adaptive vs fixed end-to-end (1.2)
Полная инструкция — в [experiments/TARGET_COUNT_VALIDATION.md](experiments/TARGET_COUNT_VALIDATION.md).
Кратко (нужны SSH-туннель + `LLM_*`/`DB_*` env + Colab A100):
```bash
# генерация + валидация (отдельный registry, основной не трогаем)
uv run --env-file .env python scripts/exp_generate.py \
    --registry experiments/registry_target_count.json
# сборка train/val
uv run --env-file .env python scripts/exp_build.py \
    --registry experiments/registry_target_count.json
# → обучи обе модели в Colab, затем eval каждой:
uv run text2sql eval --profile data/profiles/card_games/profile.json \
    --gold data/bird_large.json --model-url http://localhost:8000/v1 \
    --model-name <adapter> --db-url postgresql+psycopg://...@localhost:5444/card_games
```
**Результат:** сравнить execution accuracy и sample-efficiency (accuracy / #пар)
у `tc_fixed_1000` (1000 пар) против `tc_auto` (~1391 пар).

---

## Итоговая таблица «что проверяет что»

| Шаг | Пункт Г1 | Ресурсы |
|---|---|---|
| A1 | все | — |
| A2 | общий фикс max_seq_len | — |
| A3 | 1.1 exporter | — |
| A4 | 1.2 harness | — |
| A5 | 1.3 флаг | — |
| B1 | 1.1 Grafana/Prometheus | Docker |
| B2 | 1.3 масштаб | SSH + БД |
| B3 | 1.2 эксперимент | SSH + БД + LLM-API + Colab |
