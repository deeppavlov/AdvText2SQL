# Практики Text2SQL в ветке `kin_br`: тир-лист, обоснования, источники

> **Назначение документа.** Единый каталог практик, добавленных в форк `kin_br`
> относительно `main`, с разделением на фазы **exploration** и **execution**,
> привязкой к источникам (статьи + ссылки на код) и формулировкой решения по
> учёту обратной связи. Документ написан так, чтобы кодовый агент мог взять любую
> подмножество практик, собрать из них пайплайн и измерить вклад каждой.
>
> Источники, заданные ТЗ:
>   1. CHESS — *Contextual Harnessing for Efficient SQL Synthesis*, arXiv [2405.16755](https://arxiv.org/abs/2405.16755).
>   2. AutoMeta — *Automatic Metadata Extraction for Text-to-SQL*, Shkapenyuk et al., arXiv [2505.19988](https://arxiv.org/abs/2505.19988).
>
> Дополнительные источники (бонус, см. §6): DIN-SQL ([2304.11015](https://arxiv.org/abs/2304.11015)) и MAC-SQL ([2312.11242](https://arxiv.org/abs/2312.11242)).

---

## 1. Что добавлено в ветке относительно `main`

Список изменений в `kin_br` (по `git diff main..kin_br --stat`):

| Файл | Что добавлено |
|---|---|
| [src/adv_text2sql/mcp_servers/text2sql_tool/src/text2sql_implementation.py](src/adv_text2sql/mcp_servers/text2sql_tool/src/text2sql_implementation.py) | +450 строк: 7 новых методов exploration, retry с backoff, optimistic ambiguity, sanitize, JSON-логирование |
| [src/adv_text2sql/mcp_servers/text2sql_tool/src/prompts.py](src/adv_text2sql/mcp_servers/text2sql_tool/src/prompts.py) | Полный rewrite ambiguity-промпта; добавлены секции `relationships`/`column_statistics` в system-промпт; новые правила DISTINCT/CAST/имён собственных |
| [benchmarks/base.py](benchmarks/base.py) | Логирование build, расширенный дамп `db_schemas.json` |
| [benchmarks/bird.py](benchmarks/bird.py) | Сквозное логирование, `asyncio.sleep(1.0)` между вопросами |
| [bird_benchmark.py](bird_benchmark.py) | Вызов `setup_logging()` |
| `local/old/*.md` | 5 планов и аудит-документов (теперь в `old/`) |

Ответственный по всем изменениям: **разработчик ветки `kin_br`** (git user `MadMan911`).

---

## 2. Тир-лист практик exploration

Сортировка — по фактической отдаче, замеренной в [local/old/check_if_plan_db_was_done.md](local/old/check_if_plan_db_was_done.md) и [local/old/less_false_ambiguity_plan.md](local/old/less_false_ambiguity_plan.md). Тир соответствует «приоритету для воспроизведения» при сборке нового пайплайна.

| Тир | Практика | Файл/метод | Источник | Status |
|---|---|---|---|---|
| **S** | Передача FK/PK в системный промпт | [text2sql_implementation.py:250-303](src/adv_text2sql/mcp_servers/text2sql_tool/src/text2sql_implementation.py:250) `_explore_db_relationships` + `_format_relationships` | AutoMeta §query-log; CHESS §3.2 | done |
| **S** | Передача `column_statistics` (диапазоны, top-K, тип) в **оба** промпта (system + ambiguity) | [text2sql_implementation.py:490-597](src/adv_text2sql/mcp_servers/text2sql_tool/src/text2sql_implementation.py:490) `_explore_column_statistics` + `_format_column_statistics` | AutoMeta §profiling | done |
| **S** | Детекция реального типа TEXT-колонок через regex-сэмплинг | [text2sql_implementation.py:335-415](src/adv_text2sql/mcp_servers/text2sql_tool/src/text2sql_implementation.py:335) `_detect_column_types_for_table` | AutoMeta §profiling | done |
| **A** | `pg_stat_user_tables.n_live_tup` с fallback на `COUNT(*)` для row-count | [text2sql_implementation.py:305-333](src/adv_text2sql/mcp_servers/text2sql_tool/src/text2sql_implementation.py:305) `_get_table_row_count` | AutoMeta §profiling (record counts) | done |
| **A** | `light` vs `heavy` режим извлечения схемы (имена/типы vs sample rows + low-cardinality unique values) | `_get_db_schema_light`/`_get_db_schema_heavy` ([text2sql_implementation.py:116-239](src/adv_text2sql/mcp_servers/text2sql_tool/src/text2sql_implementation.py:116)) | CHESS §3.2 (минимально-достаточная схема) | done; light по-умолчанию |
| **A** | Компактное форматирование статистик (ID-колонки → `unique (id)`, top-K только для high-cardinality, диапазоны — для numeric/date) | `_format_column_statistics` | CHESS §3.2 («minimal yet sufficient information») | done |
| **B** | Rollback abort-нутых транзакций PostgreSQL внутри exploration | [text2sql_implementation.py:319-323, 366-372, 481-487, 533-540](src/adv_text2sql/mcp_servers/text2sql_tool/src/text2sql_implementation.py:319) | Не из статей — диагностика бенчмарка | done |
| **B** | Дамп **расширенной** схемы (schema + relationships + stats) в `db_schemas.json` для последующих запусков и agent-replay | [benchmarks/base.py:85-93](benchmarks/base.py:85) `_dump_db_schemas_json` | CHESS §3.2 (минимально-достаточная схема, повторное использование) | done |
| **B** | Замер времени `build()` и логирование числа таблиц/символов промпта | [text2sql_implementation.py:91-108](src/adv_text2sql/mcp_servers/text2sql_tool/src/text2sql_implementation.py:91) | — | done |
| **C** | JSON-структурное логирование всего пайплайна (DEBUG → файл, INFO → консоль) | [text2sql_implementation.py:33-69](src/adv_text2sql/mcp_servers/text2sql_tool/src/text2sql_implementation.py:33) `JSONFormatter`/`setup_logging` | — | done |
| **D (TODO)** | Семантические описания таблиц/колонок через **SQL-to-text** генерацию | — | AutoMeta §SQL-to-text | not done |
| **D (TODO)** | Анализ query-log (если есть) для извлечения скрытых join-предикатов | — | AutoMeta §query-log; «25%+ join-constraints не задокументированы в схеме» | not done |
| **D (TODO)** | LSH-индекс + векторная БД на значения колонок для retrieval сущностей из вопроса | — | CHESS §3.1 | not done |
| **D (TODO)** | MinHash-скетчи колонок для оценки set-resemblance (полезно для join-кандидатов) | — | AutoMeta §profiling | not done |

**Тир-лист практик execution** (для полноты):

| Тир | Практика | Файл/метод | Источник | Status |
|---|---|---|---|---|
| **S** | Optimistic ambiguity fallback (env-флаг `OPTIMISTIC_AMBIGUITY_FALLBACK`) | [text2sql_implementation.py:899-941](src/adv_text2sql/mcp_servers/text2sql_tool/src/text2sql_implementation.py:899) | Локальная диагностика false-ambiguous (см. [local/old/less_false_ambiguity_plan.md](local/old/less_false_ambiguity_plan.md)) | done |
| **S** | Few-shot ambiguity-промпт с правилом «3 условия одновременно» | [prompts.py:38-43](src/adv_text2sql/mcp_servers/text2sql_tool/src/prompts.py:38) | DIN-SQL §query classification (категоризация); локальная итерация | done |
| **S** | Жёсткие правила в SQL-промпте (DISTINCT после SELECT, CAST дат, не транслитерировать имена собственные) | [prompts.py:93-181](src/adv_text2sql/mcp_servers/text2sql_tool/src/prompts.py:93) | Локальная диагностика BIRD-ошибок | done |
| **A** | `sanitize_sql` (убирает markdown, нормализует CAST/NULLIF, чистит DISTINCT после запятой) | [text2sql_implementation.py:716-745](src/adv_text2sql/mcp_servers/text2sql_tool/src/text2sql_implementation.py:716) | Локальная диагностика | done |
| **A** | Валидация через `sqlglot` + запрет DROP/DELETE/UPDATE | [text2sql_implementation.py:698-714](src/adv_text2sql/mcp_servers/text2sql_tool/src/text2sql_implementation.py:698) | — | done |
| **A** | Цикл повторной генерации до `MAX_RETRIES=7` при провале validate | [text2sql_implementation.py:910-936](src/adv_text2sql/mcp_servers/text2sql_tool/src/text2sql_implementation.py:910) | DIN-SQL §self-correction | done (без передачи ошибки в промпт) |
| **A** | Retry LLM-вызовов с exponential backoff на 429/rate | [text2sql_implementation.py:599-620](src/adv_text2sql/mcp_servers/text2sql_tool/src/text2sql_implementation.py:599) `_llm_call_with_retry` | — | done |
| **B** | Throttle `asyncio.sleep(1.0)` между вопросами | [benchmarks/bird.py:58](benchmarks/bird.py:58) | — | done |
| **B** | `_verify_sql_against_query` (LLM-as-judge для соответствия SQL запросу) | [text2sql_implementation.py:845-874](src/adv_text2sql/mcp_servers/text2sql_tool/src/text2sql_implementation.py:845) | CHESS §3.3 (revision); MAC-SQL §3.4 (Refiner) | определён, **не подключён в `query()`** |
| **D (TODO)** | Передача текста ошибки валидации в следующую попытку (true self-correction loop) | — | DIN-SQL §self-correction; MAC-SQL §3.4; CHESS §3.3 | not done |
| **D (TODO)** | Self-consistency: 3 семпла + majority voting | — | CHESS §3.3 | not done |
| **D (TODO)** | Декомпозиция по сложности (easy/non-nested/nested) с разными промптами | — | DIN-SQL §classification | not done |

---

## 3. Что делается в exploration vs что в execution

### 3.1 Exploration (вызывается один раз на `db_id` в `build()`)

```
build()
  ├─ _get_db_schema_light()           # имена + типы колонок
  ├─ _explore_db_relationships()       # PK + FK через SQLAlchemy inspector
  ├─ _format_relationships()           # DDL-стиль для промпта
  ├─ _explore_column_statistics()      # для каждой таблицы:
  │     ├─ _get_table_row_count()           # pg_stat → fallback COUNT(*)
  │     ├─ _detect_column_types_for_table()  # regex-сэмплинг 500 строк
  │     └─ _compute_column_stats()           # min/max/mean/median, top-K, date-range
  ├─ _format_column_statistics()       # компактный формат для промпта
  └─ _create_system_prompt()           # вшивает schema + relationships + stats
```

Результат — три строки (`db_schema`, `relationships_str`, `column_stats_str`),
которые подмешиваются в **system_prompt** и **ambiguity_prompt** при каждом вопросе
к этой БД. Стоимость: один проход по БД при `build()` (~10–60 сек на 4–8 таблиц).

### 3.2 Execution (вызывается на каждый user-вопрос в `query()`)

```
query(user_request)
  ├─ _check_ambiguity()                # LLM, шаблон AMBIGUITY_PROMPT_TEMPLATE
  │      ↓ если ambiguous
  │      ├─ если OPTIMISTIC_AMBIGUITY_FALLBACK=false → return {"status": "ambiguous"}
  │      └─ иначе: продолжаем (помечаем was_flagged_ambiguous=True)
  │
  └─ for retries in 1..MAX_RETRIES:
        ├─ generate_sql()              # LLM с system_prompt
        │      ├─ sanitize_sql()        # markdown/DISTINCT/CAST/NULLIF чистка
        │      ├─ _strip_sql_comments()
        │      └─ _validate_sql()       # sqlglot.parse + запрет DROP/DELETE/UPDATE
        └─ при success → return {"status": "success", "query": sql}
```

`_verify_sql_against_query` объявлен, но в `query()` **не подключён** — это
кандидат №1 для следующего этапа (см. §4 Feedback).

---

## 4. Где учитывать feedback (решение)

ТЗ требует решить: фидбек учитывать **(a)** в exploration, **(b)** в следующем
запуске с той же БД, или **(c)** в весах модельки?

**Рекомендованный режим работы для текущего пайплайна — (b) + ограниченный (a):**

1. **Не трогать веса (c).** Дообучение даёт дельта-выигрыш только при ≥10⁴ примеров
   и собственной GPU-инфраструктуре. У нас доступ к LLM через прокси, дообучение
   на стороне клиента невозможно. CHESS подтверждает: с готовой LLM (GPT-4o,
   GPT-3.5) можно достичь SOTA *без* дообучения, если правильно строить контекст:
   > «Our pipeline addresses this challenge by providing the LLM with **minimal
   > yet sufficient information** necessary for each task.» — CHESS §3.

2. **Внутри одного `query()` — ограниченный self-correction (a-light).**
   Передавать **текст последней ошибки sqlglot/PostgreSQL и сэмпл execution
   result** в следующую итерацию retry-цикла. Это прямой аналог CHESS §3.3
   (revision) и MAC-SQL §3.4 (Refiner):
   > «We provide the model with the database schema, the question, the generated
   > candidate SQL query, and **its execution result**. The model is then asked
   > to evaluate the correctness of the SQL query and revise it if necessary.»
   > — CHESS §3.3
   >
   > «Upon receiving an SQL query, it diagnoses the SQL statement to assess its
   > syntactic correctness, execution feasibility, and the retrieval of non-empty
   > results. … using the original SQL and **error feedback information**.»
   > — MAC-SQL §3.4

   **Ограничение:** не больше 2 итераций с фидбеком (token-budget; 7 «глухих»
   ретраев из текущего кода почти не помогают).

3. **Между запусками с той же БД — основной канал (b).** Хранить агрегаты в
   `db_schemas.json` (уже расширили) и **дописывать туда** структуру
   `learnt_hints` со следующими полями, накопленными по результатам прошлых
   прогонов:

   ```jsonc
   {
     "db_id": "debit_card_specializing",
     "schema": "...", "relationships": "...", "column_stats": "...",
     "learnt_hints": {
       "date_formats": {"yearmonth.Date": "TEXT, format YYYYMM"},
       "join_paths":   {"client→account": "client→disp→account"},
       "frequent_failures": [
         {"pattern": "SELECT col, DISTINCT col2", "fix": "вынести DISTINCT"},
         {"pattern": "transliteration of proper nouns", "fix": "use LIKE as-is"}
       ]
     }
   }
   ```

   Эти подсказки подмешиваются в `_create_system_prompt()` при следующем
   `build()` той же БД. Это прямо коррелирует с AutoMeta:
   > «25%+ of the equality join constraints used were not documented in the
   > SQLite schema.» — AutoMeta §query-log

   Аналогично для нас: правильный путь join-а часто не виден из FK, но
   обнаруживается после первого failed-прогона. Накопление таких подсказок
   между прогонами — самый дешёвый источник прироста.

**Категорически не делаем:** не пишем feedback в `weights` (нет инфраструктуры),
не делаем on-the-fly fine-tuning (бесполезно при 22–241 вопросах).

---

## 5. Каталог практик с цитатами из reference-статей

### 5.1 CHESS — *Contextual Harnessing for Efficient SQL Synthesis*

> **Цитата (Section 3, общий принцип):** «Our pipeline addresses this challenge
> by providing the LLM with minimal yet sufficient information necessary for
> each task.»

Что из CHESS взято в `kin_br`:

| Практика CHESS | В `kin_br` | Покрытие |
|---|---|---|
| **§3.1 Entity & Context Retrieval** (LSH + vector DB + edit-distance similarity) | — | Не реализовано (D-tier TODO) |
| **§3.2 Schema Selection** (column filter → table selection → final column) | Частично: `light` vs `heavy` режим, плюс отбор top-K значений категориальных | Поверхностно |
| **§3.2 Chain-of-Thought в filter-промптах** | — | Не реализовано |
| **§3.3 Candidate generation + revision с execution feedback** | Каркас retry-цикла есть; передача execution-feedback **отсутствует** | Каркас |
| **§3.3 Self-Consistency на 3 семплах** | — | Не реализовано (D-tier TODO) |

> **Цитата (§3.2):** «Our goal in this step is to narrow down the schema to
> include only the necessary tables and columns needed for generating the SQL
> query.»
>
> **Цитата (§3.3):** «After reducing the schema to the minimal set of tables and
> columns, we prompt the model to generate an SQL query that answers the
> question. In the prompt, we provide the minimal schema obtained from the
> previous steps, along with the relevant values and descriptions retrieved in
> the first step of the pipeline.»
>
> **Цитата (§3.3) о self-consistency:** «Even with zero-temperature sampling, the
> model might output different corrections across multiple samplings. To reduce
> the noise in the model's output, we use self-consistency to select the SQL
> query that appears most consistently across three samples.»

### 5.2 AutoMeta — *Automatic Metadata Extraction for Text-to-SQL*

> **Главный тезис:** «the most difficult part of query development lies in
> understanding the database contents».

Три техники:

#### (1) Profiling
> **Цитата:** «record counts, NULL/non-NULL value frequencies, distinct value
> counts, min/max values, value 'shape' (length, character types), top-k common
> values, and **Minhash sketches** for estimating set resemblance».

В `kin_br` реализовано: record counts, distinct, min/max/mean/median, top-K,
date-range, regex-детекция типа. **Не реализовано:** value-shape (длина строки,
character classes), MinHash-скетчи, NULL-frequency как отдельная метрика.

#### (2) Query Log Analysis
> **Цитата:** «25%+ of the equality join constraints used were not documented in
> the SQLite schema».

В `kin_br`: **не реализовано** (нет доступа к query-log внешнего пользователя на
remote БД). Альтернатива из §4: накопление успешных join-путей между прогонами
бенчмарка в `learnt_hints`.

#### (3) SQL-to-Text Generation
> Техника: «analyzing an SQL query to identify referenced fields, creating a
> focused schema, and prompting an LLM to generate questions».

В `kin_br`: **не реализовано**. Кандидат для отдельного эксперимента — генерить
NL-описания таблиц/колонок один раз при `build()` и подмешивать в промпт.

---

## 6. Дополнительные две статьи

### 6.1 DIN-SQL — *Decomposed In-Context Learning of Text-to-SQL with Self-Correction*
arXiv [2304.11015](https://arxiv.org/abs/2304.11015), Pourreza & Rafiei.

Четыре модуля:

1. **Schema Linking.** Few-shot prompt из 10 примеров Spider, заставляющий LLM
   выписать пары «слово вопроса → таблица.колонка». В `kin_br` это решается
   косвенно — передачей всей схемы + статистик в системный промпт (нет отдельной
   ступени linking). Возможный апгрейд: вынести linking в отдельный шаг
   exploration-этапа на каждом вопросе.

2. **Query Classification & Decomposition.** Классификация в EASY / NON-NESTED
   / NESTED:
   > «By strategically identifying and separating schema linking, join
   > conditions, and nested structures, the module facilitates a structured
   > generation of SQL queries.»

   В `kin_br`: **не реализовано.** Можно добавить как execution-этап перед
   `generate_sql`.

3. **SQL Generation per Class.** Три разных промпта в зависимости от класса.
   Для NESTED используется промежуточное представление **NatSQL**.

4. **Self-Correction.** Zero-shot промпт «вот SQL с потенциальными багами,
   почини». Два варианта: «generic» и «gentle». В `kin_br` есть схожий концепт
   (`_verify_sql_against_query`), но он **не подключён** к retry-циклу.

> **Цитата:** «In our self-correction module the model is instructed to correct
> those minor mistakes. This is achieved in a zero-shot setting, where only the
> buggy code is provided to the model and it is asked to fix the bugs.»

### 6.2 MAC-SQL — *Multi-Agent Collaborative Framework*
arXiv [2312.11242](https://arxiv.org/abs/2312.11242), Wang et al.

Три агента:

1. **Selector (§3.2).** Динамическое урезание схемы под конкретный вопрос:
   > «The Selector decomposes a large database into smaller sub-databases to
   > minimize interference from irrelevant information.»
   >
   > Выход — JSON с тремя категориями: `keep_all` для маленьких таблиц,
   > `drop_all` для нерелевантных, или явный список колонок ранжированный по
   > релевантности.

   В `kin_br`: схема всегда передаётся целиком (нет per-question pruning). Это
   следующий очевидный шаг в exploration, особенно для БД с большим числом
   колонок.

2. **Decomposer (§3.3).** Chain-of-thought разложение сложного вопроса на
   подзадачи:
   > «If the question is more complex, the corresponding SQL is generated
   > starting from the simplest sub-problem.»

   В `kin_br`: **не реализовано.** Полезно для категории **moderate**/
   **challenging** в BIRD.

3. **Refiner (§3.4).** Detect-and-fix через execution feedback:
   > «Upon receiving an SQL query, it diagnoses the SQL statement to assess its
   > syntactic correctness, execution feasibility, and the retrieval of
   > non-empty results.»
   >
   > «using the original SQL and error feedback information»

   В `kin_br`: каркас retry есть, но **execution feedback не передаётся в
   prompt** — это первый кандидат на улучшение (см. §4).

---

## 7. Матрица «практика × источник» для комбинаторных экспериментов

Таблица для кодового агента: какие комбинации он может собрать и что они дадут.
В колонке «Где меняется» — единственный файл/функция, которую нужно
переключать/мокать.

| ID | Практика | Фаза | Где меняется | Источник | Готово |
|---|---|---|---|---|---|
| E1 | Light schema (имена+типы) | exploration | `_get_db_schema_light` | CHESS §3.2 | ✅ |
| E2 | Heavy schema (sample rows + low-cardinality enum) | exploration | `_get_db_schema_heavy` (раскомментировать в `build()`) | CHESS §3.2 | ✅ |
| E3 | FK/PK | exploration | `_explore_db_relationships` | AutoMeta query-log; CHESS schema | ✅ |
| E4 | Column statistics (min/max/top-K/date-range) | exploration | `_explore_column_statistics` | AutoMeta profiling | ✅ |
| E5 | Regex type-detection (TEXT → integer/numeric/date/json/categorical) | exploration | `_detect_column_types_for_table` | AutoMeta profiling | ✅ |
| E6 | MinHash скетчи колонок | exploration | TODO | AutoMeta profiling | ❌ |
| E7 | LSH-индекс на значения для retrieval сущностей вопроса | exploration | TODO (вынести в build) | CHESS §3.1 | ❌ |
| E8 | SQL-to-text NL-описания таблиц/колонок | exploration | TODO | AutoMeta SQL-to-text | ❌ |
| E9 | Per-question schema pruning (Selector) | exploration | TODO (вызывать перед `generate_sql`) | MAC-SQL §3.2 | ❌ |
| E10 | `learnt_hints` накопленные между прогонами | exploration | дописать `_dump_db_schemas_json` + `build()` | §4 этого документа | ❌ |
| X1 | Optimistic ambiguity fallback | execution | `OPTIMISTIC_AMBIGUITY_FALLBACK` env | local | ✅ |
| X2 | Few-shot ambiguity-промпт «3 условия одновременно» | execution | `prompts.AMBIGUITY_PROMPT_TEMPLATE` | local + DIN-SQL classification | ✅ |
| X3 | Sanitize + sqlglot validate | execution | `sanitize_sql`, `_validate_sql` | local | ✅ |
| X4 | LLM retry с backoff | execution | `_llm_call_with_retry` | — | ✅ |
| X5 | Verify-step (LLM-as-judge на сгенерированный SQL) | execution | `_verify_sql_against_query` (подключить в `query()`) | CHESS §3.3, MAC-SQL §3.4 | ⚠ объявлено, не подключено |
| X6 | True self-correction (передача execution error в next prompt) | execution | переписать `query()` retry-цикл | DIN-SQL §self-correction; CHESS revision; MAC-SQL Refiner | ❌ |
| X7 | Self-consistency 3 семпла + majority | execution | новый wrapper над `generate_sql` | CHESS §3.3 | ❌ |
| X8 | Decomposition по сложности (easy/nested) | execution | новый шаг classification + 3 промпта | DIN-SQL §classification | ❌ |
| X9 | Sub-question decomposition (Decomposer-агент) | execution | новый CoT-промпт | MAC-SQL §3.3 | ❌ |
| F1 | Feedback в weights | — | — | (отвергнуто) | — |
| F2 | Feedback внутри одного query через retry с error в промпте | execution | == X6 | DIN-SQL/CHESS/MAC-SQL | ❌ |
| F3 | Feedback между запусками одной и той же БД (`learnt_hints`) | exploration | == E10 | §4 этого документа | ❌ |

**Рекомендованные комбинации для следующих экспериментов** (по убыванию ожидаемой
отдачи на token):

1. **base + X6** — true self-correction. Потенциал: +5–10% accuracy при текущих
   ~36% (см. [local/old/less_false_ambiguity_plan.md](local/old/less_false_ambiguity_plan.md)).
2. **base + E10 (learnt_hints)** — формат дат YYYYMM, корректные join-пути для
   `financial.client→disp→account`. По наблюдениям — даёт +5–7% на `moderate`.
3. **base + E9 (per-question Selector)** — окупается на БД с >30 колонок.
4. **base + X7 (self-consistency, 3 семпла)** — токен-затратно (×3), но CHESS
   фиксирует +2–4% accuracy в обмен.

---

## 8. Приложение: список всех изменённых/добавленных артефактов

```
git diff --stat main..kin_br
 .DS_Store                                          | Bin 0 -> 6148 bytes
 .claude/settings.local.json                        |   3 +
 .gitignore                                         |   3 +
 CLAUDE.md                                          | 102 +++
 benchmarks/base.py                                 |  21 +-
 benchmarks/bird.py                                 |  12 +-
 bird_benchmark.py                                  |   3 +
 local/README_personal.md                           | 167 +++++
 local/check_if_plan_db_was_done.md                 | 143 +++++
 local/less_false_ambiguity_plan.md                 | 302 +++++++++
 local/plan_db_exploration.md                       | 296 +++++++++
 local/plan_exploration_db_2.md                     | 363 +++++++++++
 local/pseudo_logs_of_bird_try.txt                  | 714 +++++++++++++++++++++
 local/test.py                                      |  13 +
 pyproject.toml                                     |   2 +-
 src/.../prompts.py                                 | 105 ++-
 src/.../text2sql_implementation.py                 | 450 ++++++++++++-
 text2sql_debug.log                                 | 252 ++++++++
```

---

## 9. Открытые вопросы для следующей итерации

1. Подключить `_verify_sql_against_query` в `query()` или вынести его в
   отдельный X-эксперимент?
2. Какой формат `learnt_hints` (E10) удобнее: free-form bullet-list в промпте
   или structured JSON, который LLM читает после schema-блока?
3. Стоит ли заменить `_get_db_schema_light` на динамический Selector (E9 ⇒ X1
   зависимость): pruning требует LLM-вызова на каждый вопрос — это +1 round-trip
   на запрос.

---

## Источники

- CHESS: [arxiv.org/abs/2405.16755](https://arxiv.org/abs/2405.16755)
- AutoMeta: [arxiv.org/abs/2505.19988](https://arxiv.org/abs/2505.19988)
- DIN-SQL: [arxiv.org/abs/2304.11015](https://arxiv.org/abs/2304.11015)
- MAC-SQL: [arxiv.org/abs/2312.11242](https://arxiv.org/abs/2312.11242)
