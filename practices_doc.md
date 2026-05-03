# Практики Text2SQL: единая таблица

> Объединение трёх таблиц (тир-лист exploration, тир-лист execution, матрица
> комбинаторных экспериментов) в одну, со сквозной нумерацией. Дубликаты схлопнуты.
>
> Колонки:
> - **№** — сквозной номер практики.
> - **Практика** — короткое описание; в скобках — ID из исходной матрицы (Eₙ/Xₙ/Fₙ).
> - **Фаза** — `exploration` (раз на `db_id` в `build()`) / `execution` (на каждый вопрос в `query()`).
> - **Файл / метод (ветка)** — путь и метод; в скобках — ветка с реализацией для `done`-практик.
> - **Источник** — конкретные параграфы статей, где практика описана. Тег **`[shared N]`** означает, что одна и та же практика покрывается ≥ 2 статьями (см. §«Кросс-цитированные практики» ниже — там разобрано, *чем* каждая статья её закрывает).
> - **Почему полезно** — обоснование пользы со ссылкой на источник (цитата или тезис).
> - **Status** — `done` / `partial` / `not done` / `rejected`.
> - **Ответственный**, **Дедлайн** — заполняются по мере планирования.
>
> Источники: **CHESS** ([2405.16755](https://arxiv.org/abs/2405.16755)), **AutoMeta** ([2505.19988](https://arxiv.org/abs/2505.19988)), **DIN-SQL** ([2304.11015](https://arxiv.org/abs/2304.11015)), **MAC-SQL** ([2312.11242](https://arxiv.org/abs/2312.11242)).

| № | Практика | Фаза | Файл / метод (ветка) | Источник | Почему полезно | Status | Ответственный | Дедлайн |
|---|---|---|---|---|---|---|---|---|
| 1 | FK / PK relationships в системном промпте (**E3**) | exploration | `text2sql_implementation.py:250-303` → `_explore_db_relationships` + `_format_relationships` (ветка `kin_br`) | AutoMeta §query-log · CHESS §3.2 **[shared 2]** | AutoMeta: **«25%+ of the equality join constraints used were not documented in the SQLite schema»** — без явных FK/PK LLM угадывает join-ы и промахивается. CHESS включает связи как часть «минимально-достаточной» схемы. | done | — | — |
| 2 | Column statistics (min/max/top-K/диапазон дат) в обоих промптах (**E4**) | exploration | `text2sql_implementation.py:490-597` → `_explore_column_statistics` + `_format_column_statistics` (ветка `kin_br`) | AutoMeta §profiling | AutoMeta: **«the most difficult part of query development lies in understanding the database contents»**. record counts, distinct, min/max, top-K дают LLM понимание распределения значений → корректные литералы, агрегации, форматы сравнений. | done | — | — |
| 3 | Regex-детекция реального типа TEXT-колонок (**E5**) | exploration | `text2sql_implementation.py:335-415` → `_detect_column_types_for_table` (ветка `kin_br`) | AutoMeta §profiling (value 'shape') | AutoMeta требует «**value 'shape' (length, character types)**». Вся БД хранит TEXT — без этого LLM не знает, нужен ли CAST к INTEGER/DATE; regex-сэмплинг 500 строк это закрывает. | done | — | — |
| 4 | `pg_stat_user_tables.n_live_tup` + fallback на `COUNT(*)` для row-count | exploration | `text2sql_implementation.py:305-333` → `_get_table_row_count` (ветка `kin_br`) | AutoMeta §profiling (record counts) | AutoMeta перечисляет «**record counts**» первой метрикой профайлинга. Знание размера таблицы помогает LLM решать, нужен ли LIMIT, подзапросы, можно ли позволить себе DISTINCT. `pg_stat` — почти бесплатное приближение. | done | — | — |
| 5 | Light-режим извлечения схемы — имена + типы (**E1**) | exploration | `text2sql_implementation.py:116-239` → `_get_db_schema_light` (ветка `kin_br`, по умолчанию) | CHESS §3.2 | CHESS: **«Our pipeline addresses this challenge by providing the LLM with minimal yet sufficient information necessary for each task»**. Light экономит токены и снижает шум промпта. | done | — | — |
| 6 | Heavy-режим — sample rows + low-cardinality enums (**E2**) | exploration | `text2sql_implementation.py:116-239` → `_get_db_schema_heavy` (ветка `kin_br`, выкл. по умолчанию) | CHESS §3.2 | CHESS §3.2 включает в схему «**relevant values and descriptions**». Образцы значений помогают LLM понимать формат данных (бренды/категории), полезно для сильных моделей с запасом токенов. | done | — | — |
| 7 | Компактное форматирование статистик (ID → `unique (id)`, top-K только для high-cardinality) | exploration | `text2sql_implementation.py` → `_format_column_statistics` (ветка `kin_br`) | CHESS §3.2 | Прямая реализация принципа CHESS: **«minimal yet sufficient»** — печатать диапазоны только там, где они информативны. Меньше токенов, меньше отвлечений на нерелевантные факты. | done | — | — |
| 8 | Rollback abort-нутых PostgreSQL-транзакций внутри exploration | exploration | `text2sql_implementation.py:319-323, 366-372, 481-487, 533-540` (ветка `kin_br`) | локальная диагностика | Закрывает cascade `InFailedSqlTransaction`, который в первой версии давал 0% accuracy: одна failed-транзакция блокировала всё дальнейшее exploration. Не из статей — обнаружено в логах. | done | — | — |
| 9 | Дамп расширенной схемы (schema + relationships + stats) в `db_schemas.json` | exploration | `benchmarks/base.py:85-93` → `_dump_db_schemas_json` (ветка `kin_br`) | CHESS §3.2 | Делает результат exploration воспроизводимым: можно перезапускать execution-фазу без похода в БД. Базис для пункта **26 (learnt_hints)** — без дампа некуда дописывать накопленные подсказки. | done | — | — |
| 10 | Замер времени `build()` + лог числа таблиц / символов промпта | exploration | `text2sql_implementation.py:91-108` (ветка `kin_br`) | — | Ловит деградации (БД с 100+ таблицами, разрастание промпта); даёт данные для выбора, какие практики из этого списка можно позволить себе одновременно по бюджету. | done | — | — |
| 11 | JSON-структурное логирование пайплайна | exploration + execution | `text2sql_implementation.py:33-69` → `JSONFormatter`, `setup_logging` (ветка `kin_br`) | — | Позволяет постфактум парсить ход пайплайна (jq/pandas), считать % ambiguous, retry-distribution; без этого нельзя делать диагностические доки и решать, какую практику оставлять / выключать. | done | — | — |
| 12 | Optimistic ambiguity fallback — env `OPTIMISTIC_AMBIGUITY_FALLBACK` (**X1**) | execution | `text2sql_implementation.py:899-941` → `query()` (ветка `kin_br`) | локальная диагностика | Baseline ambiguity-промпт ложно помечал 72% запросов BIRD как ambiguous → 0% accuracy. Fallback продолжать генерировать SQL даже при ambiguous-флаге дал прирост с 13% до 36% (см. `local/old/less_false_ambiguity_plan.md` в `kin_br`). | done | — | — |
| 13 | Few-shot ambiguity-промпт «3 условия одновременно» (**X2**) | execution | `prompts.py:38-43` → `AMBIGUITY_PROMPT_TEMPLATE` (ветка `kin_br`) | DIN-SQL §classification + локальная итерация | DIN-SQL: классификация запросов в категории + примеры даёт «**structured generation of SQL queries**». У нас — три явных условия + 8 правил «всегда OK» + примеры из тех же доменов, что бенчмарк. | done | — | — |
| 14 | Жёсткие правила в SQL-промпте (DISTINCT после SELECT, CAST дат, не транслитерировать имена) | execution | `prompts.py:93-181` → `SQL_PROMPT_TEMPLATE`, `SYSTEM_PROMPT_TEMPLATE` (ветка `kin_br`) | локальная диагностика | Каждое правило закрывает измеренный класс ошибок: транслитерация имён → 1 запрос, DISTINCT после запятой → 1 запрос, неявный CAST дат → 2 запроса. Превентивные правила в промпте дешевле, чем post-fix sanitize. | done | — | — |
| 15 | `sanitize_sql` — markdown / DISTINCT / CAST / NULLIF чистка (**X3** часть 1) | execution | `text2sql_implementation.py:716-745` → `sanitize_sql` (ветка `kin_br`) | локальная диагностика | LLM возвращает markdown-обёртки, SQLite-style CAST вместо PostgreSQL, DISTINCT после запятой. Нормализация до валидации экономит retry-итерации (а значит токены) на форматных ошибках. | done | — | — |
| 16 | Валидация через `sqlglot` + запрет DROP/DELETE/UPDATE (**X3** часть 2) | execution | `text2sql_implementation.py:698-714` → `_validate_sql` (ветка `kin_br`) | — | Парсинг до execution отсеивает синтаксические ошибки без обращения к БД (быстро + дёшево); запрет mutation-операций — обязательный safety-guard для read-only БД. | done | — | — |
| 17 | Retry-цикл до `MAX_RETRIES=7` при провале validate | execution | `text2sql_implementation.py:910-936` → `query()` (ветка `kin_br`) | DIN-SQL §self-correction · CHESS §3.3 · MAC-SQL §3.4 **[shared 3]** | DIN-SQL: «**In our self-correction module the model is instructed to correct those minor mistakes**». Все три статьи описывают итеративный regenerate-loop как стандартный приём; у нас сейчас цикл «глухой» — это **каркас** для пункта **27** (true self-correction). | partial | — | — |
| 18 | LLM-вызовы с exponential backoff на 429 / rate-limit (**X4**) | execution | `text2sql_implementation.py:599-620` → `_llm_call_with_retry` (ветка `kin_br`) | — | На 200+ вопросах rate-limit неизбежен; backoff с jitter позволяет завершить весь прогон без падения и без потери уже потраченных токенов. | done | — | — |
| 19 | Throttle `asyncio.sleep(1.0)` между вопросами | execution | `benchmarks/bird.py:58` (ветка `kin_br`) | — | Дополнительная защита от burst-нагрузок поверх backoff'а. Стоимость низкая (200 × 1 сек ≈ 3.3 мин), стабильность пайплайна выше — страховка пункта **18**. | done | — | — |
| 20 | `_verify_sql_against_query` — LLM-as-judge на сгенерированный SQL (**X5**) | execution | `text2sql_implementation.py:845-874` → `_verify_sql_against_query` (ветка `kin_br`, **не подключён в `query()`**) | CHESS §3.3 (revision) · MAC-SQL §3.4 (Refiner) **[shared 2]** | CHESS: «**The model is then asked to evaluate the correctness of the SQL query and revise it if necessary**». Доп. LLM-проход отсеивает грубые семантические ошибки, которые `sqlglot` поймать не может (например `SUM(amount)` вместо `SUM(price)`). | partial | — | — |
| 21 | MinHash-скетчи колонок (**E6**) | exploration | TODO | AutoMeta §profiling | AutoMeta: «**Minhash sketches for estimating set resemblance**». За O(k) оценивает долю общих значений двух колонок → быстро предлагает join-кандидатов, которых нет в FK (закрывает половину пункта **24**). | not done | — | — |
| 22 | LSH-индекс + векторная БД на значения колонок для retrieval сущностей вопроса (**E7**) | exploration | TODO | CHESS §3.1 (Entity & Context Retrieval) | CHESS §3.1: для текстовых колонок LSH + embeddings находят **точные значения**, упомянутые в вопросе, и подмешивают их в промпт. Решает класс ошибок типа «What's the French name of the set that contains 'Tendo Ice Bridge'?». | not done | — | — |
| 23 | SQL-to-text NL-описания таблиц / колонок (**E8**) | exploration | TODO | AutoMeta §SQL-to-text | AutoMeta: «**analyzing an SQL query to identify referenced fields, creating a focused schema, and prompting an LLM to generate questions**» — обратная задача: один раз генерируем NL-описания при `build()`, помогает LLM выбирать правильную колонку среди похожих по имени. | not done | — | — |
| 24 | Анализ query-log для извлечения скрытых join-предикатов | exploration | TODO (для нашего стенда query-log внешних пользователей недоступен — альтернатива в **26**) | AutoMeta §query-log | AutoMeta: **«25%+ of the equality join constraints used were not documented in the SQLite schema»**. Query-log хранит реально используемые join-пути, которые в FK не описаны — прямой источник «скрытого знания о БД». | not done | — | — |
| 25 | Per-question schema pruning — Selector-агент (**E9**) | exploration | TODO (вызывать перед `generate_sql`) | MAC-SQL §3.2 (Selector) · CHESS §3.2 (Schema Selection) **[shared 2]** | MAC-SQL: «**The Selector decomposes a large database into smaller sub-databases to minimize interference from irrelevant information**». CHESS §3.2 делает то же по сути — отбор минимально-достаточного подмножества колонок/таблиц под каждый вопрос. Для БД с >30 колонок (наш `cards` — 72) уменьшает шум промпта. Цена: +1 LLM round-trip / запрос. | not done | — | — |
| 26 | `learnt_hints` — накопление подсказок между прогонами одной БД (**E10 = F3**) | exploration | TODO (дописать `_dump_db_schemas_json` и `build()`) | AutoMeta §query-log (адаптировано — см. §«Кросс-цитированные практики») | После прогона BIRD знаем формат дат (`yearmonth.Date` = `YYYYMM`), реальные join-пути (`client → disp → account`), частые SQL-паттерны-ошибки. Дозапись в `db_schemas.json` и подмешивание в промпт следующего запуска — самый дешёвый источник прироста (наш аналог query-log из пункта **24**). | not done | — | — |
| 27 | True self-correction — передача текста ошибки в next prompt (**X6 = F2**) | execution | TODO (переписать retry-цикл `query()`) | DIN-SQL §self-correction · CHESS §3.3 (revision) · MAC-SQL §3.4 (Refiner) **[shared 3]** | MAC-SQL: «**using the original SQL and error feedback information**»; CHESS: «**we provide the model with the database schema, the question, the generated candidate SQL query, and its execution result**». Превращает 7 «слепых» retry в осмысленный self-correction-loop. **Главный кандидат на +5–10% accuracy.** | not done | — | — |
| 28 | Self-consistency — 3 семпла + majority voting (**X7**) | execution | TODO (новый wrapper над `generate_sql`) | CHESS §3.3 | CHESS: «**we use self-consistency to select the SQL query that appears most consistently across three samples**». Снижает шум при ненулевой температуре; CHESS фиксирует +2–4% accuracy за 3× стоимость токенов. | not done | — | — |
| 29 | Декомпозиция по сложности (easy / non-nested / nested) с разными промптами (**X8**) | execution | TODO (новый шаг classification + 3 специализированных промпта) | DIN-SQL §classification | DIN-SQL: «**By strategically identifying and separating schema linking, join conditions, and nested structures, the module facilitates a structured generation**». Особенно помогает на NESTED (BIRD `challenging`); сейчас один универсальный промпт «лечит» все сложности усреднённо. | not done | — | — |
| 30 | Sub-question decomposition — Decomposer-агент, CoT (**X9**) | execution | TODO (новый CoT-промпт) | MAC-SQL §3.3 | MAC-SQL: «**If the question is more complex, the corresponding SQL is generated starting from the simplest sub-problem**». CoT-разложение помогает на `moderate`/`challenging`, где требуется композиция нескольких подзапросов; одношаговая генерация теряет промежуточные шаги. | not done | — | — |
| 31 | Feedback в веса модели (**F1**) | — | — | — | Дообучение даёт выигрыш только при ≥10⁴ примеров и собственной GPU-инфраструктуре; LLM у нас через прокси, fine-tuning невозможен. CHESS подтверждает: с готовой LLM можно достичь SOTA без дообучения, если правильно строить контекст («**minimal yet sufficient information**»). Все ресурсы — на пункты **17–30**. | rejected | — | — |

---

## Привязки фич к статьям

### Кросс-цитированные практики (≥ 2 источников)

Пять практик описаны в нескольких статьях по-разному, но решают одну задачу.
Здесь разбор, *чем именно* каждая статья её закрывает — это нужно, чтобы при
имплементации можно было выбирать подходящую формулировку, и чтобы агент видел,
где «одна и та же» практика прячется под разными именами.

#### № 1 — FK / PK relationships в промпте · `[shared 2]`

| Статья | Раздел | Чем покрывает |
|---|---|---|
| **AutoMeta** | §query-log | Фиксирует *проблему*: «25%+ of the equality join constraints used were not documented in the SQLite schema». То есть просто FK из схемы недостаточно — без них pipeline теряет четверть реальных join-предикатов. |
| **CHESS** | §3.2 (Schema Selection) | Включает FK/PK как обязательную часть «**minimal yet sufficient**» контекста. Формулировка: схема без связей не передаётся LLM никогда. |

Расхождение: AutoMeta мотивирует *«FK мало, нужна доразметка»*; CHESS отвечает на более простой вопрос — *«какая часть DDL должна попасть в промпт»* (ответ: FK обязательно).

#### № 17 — Retry-каркас (regenerate-on-failure) · `[shared 3]`

| Статья | Раздел | Чем покрывает |
|---|---|---|
| **DIN-SQL** | §self-correction | Канонический термин — «self-correction module»; цикл «генерация → проверка → перегенерация», zero-shot, два варианта (generic/gentle). |
| **CHESS** | §3.3 | Тот же цикл, но в терминах «candidate generation + revision»; добавляет execution result в контекст ревизии. |
| **MAC-SQL** | §3.4 (Refiner) | То же, но как *отдельный агент* — Refiner получает SQL + diagnostics и решает: оставить, переписать или попросить регенерировать. |

Расхождение: DIN-SQL описывает retry как *шаг в линейном пайплайне*; CHESS — как *подмодуль шага генерации*; MAC-SQL — как *отдельный агент* в multi-agent-системе. Реализация у нас сейчас ближе всего к DIN-SQL — линейный цикл в `query()`.

#### № 20 — LLM-as-judge (Verify) · `[shared 2]`

| Статья | Раздел | Чем покрывает |
|---|---|---|
| **CHESS** | §3.3 (revision) | «The model is then asked to evaluate the correctness of the SQL query and revise it if necessary». Verify здесь — *первая половина* revision-шага (вторая — фактическая ревизия). |
| **MAC-SQL** | §3.4 (Refiner) | Refiner-агент диагностирует «syntactic correctness, execution feasibility, and the retrieval of non-empty results» — фактически тот же verify, но как самостоятельный агент. |

Наш `_verify_sql_against_query` пока ближе к CHESS-формулировке (один LLM-вызов после генерации), но **не подключён в `query()`** — это то, что отделяет нас от любой из двух рецептур.

#### № 25 — Per-question schema pruning · `[shared 2]`

| Статья | Раздел | Чем покрывает |
|---|---|---|
| **MAC-SQL** | §3.2 (Selector) | Самое явное определение: «decomposes a large database into smaller sub-databases». Селектор работает per-question и выдаёт три класса таблиц (keep_all / drop_all / explicit subset). |
| **CHESS** | §3.2 (Schema Selection) | По сути то же самое: «narrow down the schema to include only the necessary tables and columns». Реализован через каскад column-filter → table-selection → final-column. |

Различие техническое: MAC-SQL — один LLM-вызов с JSON-выходом; CHESS — каскад из трёх под-промптов. Если решим внедрять, MAC-SQL-форма дешевле (1 round-trip), CHESS-форма точнее.

#### № 27 — True self-correction (передача ошибки в prompt) · `[shared 3]`

| Статья | Раздел | Чем покрывает |
|---|---|---|
| **DIN-SQL** | §self-correction | Передаёт LLM «buggy code», просит починить. *Не передаёт текст ошибки* — zero-shot фикс на основе кода. |
| **CHESS** | §3.3 (revision) | Идёт дальше: «we provide the model with the database schema, the question, the generated candidate SQL query, and **its execution result**». Передаёт результат выполнения, в т.ч. ошибки. |
| **MAC-SQL** | §3.4 (Refiner) | Самая полная формулировка: «using the original SQL and **error feedback information**». Прямо про error feedback. |

Различие важное при выборе реализации:
- DIN-SQL-вариант — самый дешёвый (не нужно ходить в БД между retry).
- CHESS/MAC-SQL — нужно выполнять SQL до retry, чтобы получить execution result/error. Но именно эта пара даёт тот самый «+5–10% accuracy».

---

### Полная матрица практика × статья

Компактная сводка для агента: где (в какой статье) описана каждая практика.
Ячейка = раздел в статье; пусто = статья практику не затрагивает.

| № | Практика | CHESS | AutoMeta | DIN-SQL | MAC-SQL | Локальное / др. |
|---|---|---|---|---|---|---|
| 1  | FK / PK в промпте                                  | §3.2 | §query-log | — | — | — |
| 2  | Column statistics                                  | — | §profiling | — | — | — |
| 3  | Regex type-detection (value 'shape')               | — | §profiling | — | — | — |
| 4  | pg_stat row-count                                  | — | §profiling | — | — | — |
| 5  | Light schema                                       | §3.2 | — | — | — | — |
| 6  | Heavy schema                                       | §3.2 | — | — | — | — |
| 7  | Compact stats formatting                           | §3.2 | — | — | — | — |
| 8  | PG transaction rollback                            | — | — | — | — | local diagnostics |
| 9  | Dump `db_schemas.json`                             | §3.2 | — | — | — | — |
| 10 | `build()` timing & log                             | — | — | — | — | — |
| 11 | JSON structured logging                            | — | — | — | — | — |
| 12 | Optimistic ambiguity fallback                      | — | — | — | — | local diagnostics |
| 13 | Few-shot ambiguity prompt                          | — | — | §classification | — | local iteration |
| 14 | Strict SQL prompt rules                            | — | — | — | — | local diagnostics |
| 15 | `sanitize_sql`                                     | — | — | — | — | local diagnostics |
| 16 | sqlglot validate + forbid mutations                | — | — | — | — | — |
| 17 | Retry-каркас MAX_RETRIES                           | §3.3 | — | §self-correction | §3.4 | — |
| 18 | LLM backoff                                        | — | — | — | — | — |
| 19 | `asyncio.sleep` throttle                           | — | — | — | — | — |
| 20 | `_verify_sql_against_query` (LLM-as-judge)         | §3.3 | — | — | §3.4 | — |
| 21 | MinHash sketches                                   | — | §profiling | — | — | — |
| 22 | LSH retrieval of column values                     | §3.1 | — | — | — | — |
| 23 | SQL-to-text NL-описания                            | — | §SQL-to-text | — | — | — |
| 24 | Query-log analysis                                 | — | §query-log | — | — | — |
| 25 | Per-question schema pruning                        | §3.2 | — | — | §3.2 | — |
| 26 | `learnt_hints` между прогонами                     | — | §query-log (адаптация) | — | — | local extension |
| 27 | True self-correction (error feedback)              | §3.3 | — | §self-correction | §3.4 | — |
| 28 | Self-consistency                                   | §3.3 | — | — | — | — |
| 29 | Decomposition по сложности                         | — | — | §classification | — | — |
| 30 | Sub-question decomposition (CoT Decomposer)        | — | — | — | §3.3 | — |
| 31 | Feedback в веса (отвергнуто)                       | — | — | — | — | — |

**Сводка по статьям:**

| Статья | Сколько практик | Какие |
|---|---|---|
| **CHESS** | 11 | 1, 5, 6, 7, 9, 17, 20, 22, 25, 27, 28 |
| **AutoMeta** | 8 | 1, 2, 3, 4, 21, 23, 24, 26 |
| **DIN-SQL** | 4 | 13, 17, 27, 29 |
| **MAC-SQL** | 4 | 17, 20, 25, 27, 30 |
| **Локальное** | 7 | 8, 10, 11, 12, 13, 14, 15 (плюс инфраструктурные 16, 18, 19) |

CHESS и AutoMeta — главные «доноры» практик: между ними поделена exploration-фаза.
DIN-SQL и MAC-SQL добавляются в основном в execution-цикле (retry / verify /
decomposition), и именно там лежат все три «**[shared]**»-практики высокого тира.

---

## Ablation experiment (practices #2, #4, #6, #8, #10)

Каждая из пяти практик exploration-фазы вынесена в отдельную ветку от `main`,
чтобы измерить её индивидуальный вклад в accuracy BIRD и Ambrosia.

**Методология:**

- **Изоляция:** *leave-one-in*. Каждая ветка = `main` + только эта практика +
  минимально необходимые зависимости. Co-applied честно перечислены ниже.
- **Baseline:** ветка `main` на момент создания ablation-веток
  (commit `c3d2cfc`). На ней практики #2/#4/#6/#8/#10 отсутствуют.
- **Прогон:** один прогон BIRD-small + Ambrosia-small на каждую ветку через
  `scripts/run_ablation.sh <branch>`. Результат — accuracy, печатаемый
  бенчмарками. Заполняется по факту запуска.
- **Шум:** 1 прогон при temperature > 0 даёт ±5–10% разброс. Цифры рассчитаны
  как ориентир, не как статистически значимые показатели.
- **Ожидаемые «нулевые» эффекты:** ветки `feat/08_*` и `feat/10_*` —
  служебные практики (rollback-guard и observability-логи). На чистом `main`
  им нечего ловить / некуда влиять, accuracy ≈ baseline. Это **валидный**
  результат: подтверждает, что они работают «в комбинации», а не самостоятельно.

**Результаты (заполняются после прогона):**

| Branch | Practice | Co-applied (deps) | Base commit | BIRD acc | Ambrosia acc |
|---|---|---|---|---|---|
| `main` (baseline) | — | — | `c3d2cfc` | TBD | TBD |
| `feat/02_column_statistics` | #2 column stats | #4, #8 | `75175bd` | **31.82%** (simple 66.67% / moderate 0.00% / challenging 33.33%; false-ambiguous 36.36%) | **58.33%** (24 q, 9 gold-ambig, 7 predicted-ambig, 3 correctly flagged) |
| `feat/04_pg_stat_rowcount` | #4 row-count + min-formatter | #8 | `c3d2cfc` | TBD | TBD |
| `feat/06_heavy_schema` | #6 heavy schema | #8 | `c3d2cfc` | TBD | TBD |
| `feat/08_pg_rollback` | #8 PG rollback guards | — | `c3d2cfc` | TBD | TBD |
| `feat/10_build_timing` | #10 build() timing + log | — | `c3d2cfc` | TBD | TBD |

**Как заполнить:** `./scripts/run_ablation.sh <branch-name>` запустит BIRD +
Ambrosia на указанной ветке и положит логи в `ablation_results/<branch>/`.
Финальная цифра для BIRD печатается как `Overall accuracy: X.XX%`,
для Ambrosia — `Accuracy: X.XX%`. Перепиши их в таблицу выше.
