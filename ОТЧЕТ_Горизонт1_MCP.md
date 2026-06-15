# Отчёт о работе и результатах

**Проект:** AutoText2SQL · ветка `feat/autonomous-pipeline`
**Объём работ:** реализация Горизонта-1 дорожной карты + диагностика и починка
MCP-сервера + полная верификация.

---

## 1. Контекст

После составления дорожной карты проекта (`roadmap.html`) был выбран к
реализации **Горизонт-1** — «довести и провалидировать построенное». Разведка
кодовой базы показала, что серверинг и адаптивный `target_count` уже частично
реализованы, поэтому работа свелась к hardening, observability и
экспериментальной инфраструктуре, а не к разработке с нуля. Отдельно по запросу
проведена проверка готовности MCP-сервера, в ходе которой вскрылись и были
устранены блокирующие баги.

---

## 2. Что сделано — Горизонт-1

### 2.1. Динамический `max_seq_len` (общий критический фикс)
- Новый модуль [`training/seq_len.py`](src/adv_text2sql/training/seq_len.py):
  оценка длины системного промпта в токенах + подбор `max_seq_length` под
  конкретную схему; флаг `overflow` при превышении потолка (16384).
- Источник промпта вынесен в единую `build_system_prompt()`
  ([`dataset_builder.py`](src/adv_text2sql/training/dataset_builder.py)) —
  оценка, обучение и инференс берут промпт из одной функции (инвариант
  train↔inference сохранён).
- Подключено в CLI `train` и `init`
  ([`cli/main.py`](src/adv_text2sql/cli/main.py)).
- **Побочно устранён скрытый баг:** `train`/`init` раньше молча использовали
  дефолт `max_seq_len=4096` (хотя эксперименты шли на 12288) → промпт мог
  обрезаться. Теперь значение подбирается автоматически.
- **Результат:** для card_games → `~6382 токена → max_seq_len=8192`
  (проверено, значение попало в сгенерированный ноутбук).

### 2.2. Observability (пункт 1.1)
- Новый exporter [`serve/metrics_exporter.py`](src/adv_text2sql/serve/metrics_exporter.py):
  читает `queries.jsonl` инкрементально, отдаёт продуктовые метрики
  (`text2sql_success_rate`, `..._queries_total{status}`, `..._failed_total`,
  гистограмму latency).
- Prometheus scrape-job + Grafana provisioning (datasource + дашборд
  «Text2SQL — Quality & Serving») + сервисы `metrics-exporter`/`grafana` в
  `docker-compose` под профилем `observability`.
- **Результат:** exporter проверен локально — `success_rate=0.667` на тестовых
  данных (2 success / 1 fail), гистограмма latency заполняется.

### 2.3. Масштабируемость профайлера (пункт 1.3)
- В [`sample_collector.py`](src/adv_text2sql/profiler/sample_collector.py):
  `SELECT DISTINCT ... LIMIT threshold+1` вместо полного скана уникальных
  значений (early-exit на high-cardinality колонках — критично на больших
  таблицах, результат идентичен).
- Новый флаг `--skip-low-cardinality` (по аналогии с `--skip-stats`) для
  быстрого профилирования крупных схем.

### 2.4. Эксперимент adaptive vs fixed (пункт 1.2)
- В [`scripts/_registry.py`](scripts/_registry.py): поддержка
  `count_per_lang: "auto"` с делением адаптивного бюджета между языками.
- Флаг `--registry` для `exp_generate.py`/`exp_build.py` (отдельные наборы
  экспериментов, не трогая основной реестр).
- Конфиг [`experiments/registry_target_count.json`](experiments/registry_target_count.json)
  + методика [`experiments/TARGET_COUNT_VALIDATION.md`](experiments/TARGET_COUNT_VALIDATION.md).
- **Результат:** для card_games `complexity_score=27.82 → total=1391`,
  fixed=500/язык, auto=695/язык — резолвер проверен.

---

## 3. MCP-сервер: диагностика и починка

При проверке готовности обнаружено, что MCP-сервер был **реализован, но не
работал** — три интеграционных бага (обёртка рассинхронизирована с API
`Text2SQLGenerator`, юнит-тестами не покрыта):

| # | Баг | Фикс |
|---|---|---|
| 1 | Импорт отсутствующего модуля `generate_tool_description` → `ModuleNotFoundError` на старте | Статическое схемо-осведомлённое описание tool'а (без LLM-вызова на бутстрапе) |
| 2 | `query(..., check_ambiguity=, check_sql_query=)` — аргументов нет | Вызов `query(user_query_text)` |
| 3 | Разбор ответа по несуществующей вложенной структуре, SQL не исполнялся | Переписан под реальный контракт `{"status","query"}` + исполнение через `execute_safe()` |

Все правки — в [`main.py`](src/adv_text2sql/mcp_servers/text2sql_tool/main.py),
ruff чистый.

---

## 4. Результаты верификации (прогнано вживую)

### 4.1. Юнит-тесты и линт
- **48 unit-тестов проходят** (+9 новых: seq_len, metrics_exporter,
  sample_collector, registry), ~1 сек.
- `ruff` по всему коду Г1 + MCP — чисто. (NB: `ruff check .` по всему репо даёт
  51 предобразующую ошибку в legacy-файлах — бенчмарки/ноутбуки/старые скрипты,
  к этой работе не относятся.)

### 4.2. MCP-сервер — полный end-to-end (с туннелём + chat-моделью)
| Этап | Результат |
|---|---|
| Boot / HTTP `:8000/mcp/` | ✅ `Uvicorn running` |
| Discovery | ✅ `TOOLS: ['text2sql']` |
| NL → SQL | ✅ `SELECT COUNT(*) AS total_cards FROM "cards";` |
| Исполнение на БД | ✅ **56 822 карты**; топ-5 художников (John Avon 1128, Kev Walker 997, …) |
| Логирование | ✅ `queries.jsonl: status=success` |

**Подтверждение инварианта:** результаты инференса (John Avon 1128, Kev Walker
997) побайтово совпали со статистикой в `profile.json` — наглядное доказательство
train↔serve консистентности.

**Вердикт по MCP: полностью готов.** Эксплуатационные условия (не код):
chat-совместимая модель в `.env` (текущий `gpt-5.1-codex-mini` несовместим с
`/v1/chat/completions` → `404`) и поднятый SSH-туннель на `localhost:5444`.

---

## 5. Артефакты

**Новый код:** `training/seq_len.py`, `serve/metrics_exporter.py`,
`serve/grafana/*`, тесты `test_seq_len.py`, `test_metrics_exporter.py`,
`test_sample_collector.py`, `test_registry.py`.

**Изменённый код:** `training/dataset_builder.py`, `cli/main.py`,
`profiler/sample_collector.py`, `profiler/cli.py`, `serve/prometheus.yml`,
`serve/docker-compose.yml`, `scripts/_registry.py`, `scripts/exp_generate.py`,
`scripts/exp_build.py`, `mcp_servers/text2sql_tool/main.py`, `pyproject.toml`.

**Документация:** `roadmap.html` (карта), `VERIFY_HORIZON1.md` (проверка кода),
`RUNBOOK_HORIZON1.md` (операционный плейбук), `VERIFY_MCP.md` (проверка MCP),
`experiments/TARGET_COUNT_VALIDATION.md`, `experiments/registry_target_count.json`.

---

## 6. Что остаётся (внешние ресурсы / следующие шаги)

1. **Эксперимент 1.2 end-to-end** — генерация (LLM-API) → обучение (Colab A100) →
   eval; сравнить accuracy и sample-efficiency fixed vs auto.
2. **Прогон 1.3 на крупной схеме** — выбрать максимальную БД на сервере,
   замерить время/токены/overflow.
3. **Observability в Docker** — поднять стек, снять скриншот дашборда с боевым
   трафиком через MCP.
4. **Закрепить конфиг MCP** — сменить `LLM_MODEL_NAME` в `.env` на
   chat-совместимую модель.
5. **Unit-тест на MCP-обёртку** (мок `query()`/`execute_safe()`), чтобы рассинхрон
   с API генератора ловился автоматически.

---

## 7. Итог

Горизонт-1 реализован на уровне кода и инфраструктуры (4 пункта), покрыт
тестами и документацией. MCP-сервер из «реализован, но не запускается» приведён
в состояние **полностью рабочего**, что подтверждено живым end-to-end прогоном с
реальными данными из БД. Эксперименты, требующие GPU/LLM-бюджета, подготовлены
как конфиг + раннер и поставлены в очередь под доступ к ресурсам.
