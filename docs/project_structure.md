# Структура проекта

Организация файлов выполнена 23.10.2026, дерево обновлено 04.11.2026. Подробности исследований —
[01_gpt_research.md](01_gpt_research.md), следующий этап —
[02_feature_selector.md](02_feature_selector.md).

```text
AdvText2SQL/
├── защита/                             # Презентации и исторические отчёты; исключена из Git
│   ├── README.md                       # Оглавление материалов и ссылки на актуальные итоги
│   ├── AutoText2SQL.pdf / presentation.pdf # Две презентации
│   └── *.md                            # Шесть исторических материалов: слайды, текст и отчёты
├── docs/                               # Краткие итоги и журнал этапов Roadmap
│   ├── README.md                       # Индекс этапов и правило ведения документации
│   ├── 01_gpt_research.md               # GPT baseline/гибриды и сравнение с Mistral
│   ├── 02_feature_selector.md           # План и журнал выбора фичей и испытаний классификатора
│   ├── 03_deepseek_ablation.md          # Протокол и ручной запуск одиночных фичей DeepSeek
│   ├── 04_deepseek_combinations.md      # Три этапа комбинаций, строгая оценка и команды запуска
│   ├── 05_configuration_dataset.md     # Phase 0.2: utility, маски, групповые splits и CV
│   ├── 06_rule_selector.md             # Phase 1: правила, интеграция и отрицательный результат test
│   ├── 07_multilabel_dataset.md        # Phase 2, шаг 1: слабая разметка полезности и маски
│   ├── 08_multilabel_classifier.md     # Phase 2: обученная голова, ограничения и ручной DeepSeek test
│   ├── 09_classifier_improvement.md    # Совместные маски: train CV и отрицательный результат dev
│   ├── 10_static_hybrid_search.md      # Семантика всех фичей, ограничения и план поиска комбинаций
│   ├── 11_static_hybrid_experiments.md # Поэтапный запуск поиска, cohorts, команды и контроль результатов
│   ├── 12_results_and_data_quality.md # Итоги additions, аудит SQL/схем и отдельный защищённый v3
│   ├── 13_static_hybrid_confirmation.md # Ограниченные 15 масок, фиксация финалистов и повторные train/dev
│   ├── 14_static_hybrid_connection_fix.md # Отказ SSH/preflight, retry и исправление диагностики monitor
│   ├── 15_prompt_optimization_plan.md  # Обзор TextGrad/MIPROv2/GEPA, место в Roadmap и протокол будущего пилота
│   ├── 16_transport_recovery.md        # Защита от сбоев API, происхождение и ручное восстановление 9 ID
│   ├── 17_static_hybrid_dev.md         # Подтверждение пяти статических профилей на dev, три повтора
│   ├── 18_full_static_hybrid.md        # Четыре профиля × два полных согласованных набора, ручной запуск
│   └── project_structure.md            # Этот файл: дерево и назначение папок
├── research_results/                   # Все архивные и новые результаты исследований
│   ├── deepseek_direct_ablation_large_20261024/ # Прямой DeepSeek: baseline и одиночные фичи
│   ├── deepseek_full_static_20261103/  # Отдельная full-серия: experiment.json, full_report, full/<профиль>/<бенчмарк>
│   ├── rule_selector_deepseek_20261028/ # Замороженная политика, offline replay, сравнение и решения
│   ├── multilabel_classifier_deepseek_20261029/ # Head, encoder/scaler manifest, обучение и оценка меток
│   ├── multilabel_deepseek_test_20261029/ # Live test 100: bird 55 и ambrosia_227 45; root comparison.md
│   ├── multilabel_structured_dev_20261029/ # Офлайн-эксперимент трёх масок; test не оценивался
│   ├── static_hybrid_audit_20261029/   # Парные одиночные эффекты и 20 предложенных, не запущенных масок
│   ├── deepseek_combinations_large_20261026/ # Новые комбинации, контроли и выбор конфигураций
│   │   ├── search_v2_20261029/         # Продолжение поиска: отдельный protocol, train/dev/full и summary
│   │   ├── quality_audit_20261029/     # Пересчёт ответов, TSV-snapshot и 16 несоответствий Ambrosia
│   │   ├── search_v3_20261029/         # Создаётся при ручной подготовке/запуске нового защищённого протокола
│   │   ├── final_search_20261030/      # Исходная ограниченная серия; остановка до LLM на preflight
│   │   ├── final_search_20261030_connectionfix/ # Исходные результаты и логи, сохранены перед восстановлением
│   │   └── final_search_20261030_transportfix/  # Завершённые train/dev и transport recovery
│   │       └── static_hybrid_conclusion.json   # Выбранный 14+32, ограничения вывода и хеши свидетельств
│   ├── deepseek_relationships_large_20261026/ # Проверка FEAT_1 на опоре 14+32+35
│   │   └── maintenance/                # Исходные снимки и контроль целостности технических исправлений
│   ├── README.md                       # Формат хранения и пояснение переноса старых путей
│   ├── ablation_results_large/         # Большие одиночные абляции и baseline Mistral
│   │   ├── feat_baseline/              # Исходный большой baseline Mistral
│   │   ├── feat_N_bird/                # BIRD с одной включённой фичей
│   │   └── feat_N_ambrosia/            # Ambrosia с одной включённой фичей
│   ├── ablation_results_small/         # Данные малых абляций и поиска комбинаций
│   │   └── feat_best_hybrid_large/     # Важно: здесь лежит best hybrid НА БОЛЬШИХ наборах
│   ├── runs/                           # Новые воспроизводимые запуски и диагностики
│   │   ├── gpt54mini_baseline_large_20261016/
│   │   │   ├── bird/                   # GPT baseline: 99/241, 41,08%
│   │   │   ├── ambrosia/               # Неудачные попытки с полными 240; диагностика пустых БД
│   │   │   └── ambrosia_227/           # GPT baseline: 160/227, 70,48%, TP=0
│   │   ├── gpt54mini_hybrid_large_20261016/
│   │   │   ├── bird/                   # Исходный 14-фичевый гибрид: 93/241, 38,59%
│   │   │   └── ambrosia_227/           # Гибрид: 166/227, 73,13%, F1=57,93%
│   │   ├── gpt54mini_hybrid_bird_no32_33_50_20261023/
│   │   │   └── bird/                   # Диагностическая выборка 50: 22/50, не полный benchmark
│   │   ├── gpt54mini_hybrid_bird_no32_33_full_20261023/
│   │   │   └── bird/                   # Полный адаптированный гибрид: 113/241, 46,89%
│   │   ├── gpt54mini_main_prompt_ambrosia227_20261016/
│   │   │   └── ambrosia_227/           # Контроль с точным промптом main: 46,70%
│   │   ├── ambrosia_ambiguity_main_20261016/ # Те же 12: точный промпт main
│   │   ├── bird_small_openrouter_deepseek_20261016/ # Ранний smoke-test, 22 вопроса
│   │   └── feature_audit_20261023/      # Полный аудит 33 флагов и таблица больших абляций
│   ├── legacy_notes/                   # Прежние планы local/old; только исторические заметки
│   ├── legacy_snapshots/               # Прежние all_*_results.json и db_schemas.json из корня
│   └── organization_20261023/          # Контроль целостности и журнал этой уборки
├── data/                               # Исходные benchmark-вопросы и производные данные обучения
│   ├── bird_small.json                 # 22 вопроса, подмножество bird_large
│   ├── bird_large.json                 # 241 вопрос BIRD
│   ├── ambrosia_small.json             # 24 вопроса, подмножество ambrosia_large
│   ├── ambrosia_large.json             # 240 исходных вопросов
│   ├── ambrosia_large_available_227.json          # Явно зафиксированная доступная выборка
│   ├── ambrosia_large_available_227.manifest.json # ID, исключения и hashes
│   ├── feature_labels*.json            # Исторические labels; требуют аудита до обучения
│   ├── configuration_utility/          # Новый корректный датасет выбора конфигураций
│   │   ├── README.md                   # Назначение и команды воспроизведения
│   │   └── deepseek_20261028/           # 468 строк, inputs, manifest, splits, CV и summary
│   ├── multilabel_utility/             # Слабые метки полезности отдельных фичей DeepSeek
│   │   ├── README.md                   # Формат, маски и воспроизведение
│   │   └── deepseek_20261029/           # 468 строк, 16 выходов, отдельные условные пары и прежние splits
│   ├── ambrosia_train.json             # Исходные train-данные Ambrosia
│   ├── train_queries.json              # Исходные примеры проекта
│   └── launch_db/                      # Инструкции и утилиты размещения БД
├── src/adv_text2sql/                    # Основная библиотека и MCP/Text2SQL реализация
│   └── mcp_servers/text2sql_tool/src/
│       ├── text2sql_implementation.py   # Генератор, извлечение схем, FEAT_N, обработка SQL
│       ├── feature_selector.py          # Выбор 35 / 14+32+35 по правилам без API
│       ├── selective_generator.py       # Генератор с изоляцией флагов между конфигурациями
│       ├── multilabel_classifier.py      # Замороженный encoder, 6 признаков, masked loss и 16 выходов
│       ├── multilabel_generator.py       # Применение масок, ленивый build и изоляция состояния вопроса
│       └── prompts.py                  # Шаблоны промптов
├── benchmarks/                         # BIRD/Ambrosia pipeline, оценщики и учёт токенов
├── scripts/                            # Поддерживаемые инструменты текущих исследований
│   ├── run_feature_ablation.py          # Последовательные абляции, контроль ошибок, сводные таблицы
│   ├── run_feature_combinations.py      # Шесть запусков: 2+35, затем правила и детектор
│   ├── run_hybrid_relationships.py      # Два запуска: PK/FK-подсказки к выбранному гибриду
│   ├── research_metrics.py              # Офлайн-оценка Ambrosia: error всегда неверен
│   ├── atomic_artifacts.py              # Атомарная запись JSON с повторами при Windows-блокировках
│   ├── ablation_maintenance.py           # Проверенные I/O-версии и восстановление технических checkpoint-ошибок
│   ├── llm_endpoints.py                # Прямой DeepSeek/OpenRouter: запросы, баланс и валюты
│   ├── run_openrouter_baseline.py       # Baseline / hybrid / hybrid_no32_33; checkpoint и metadata
│   ├── feature_presets.py              # Фиксированные наборы флагов
│   ├── baseline_connection.py          # Таймауты PostgreSQL и повторы подготовки БД
│   ├── research_paths.py               # Новые пути и совместимость старых metadata/команд
│   ├── ambrosia_subset.py              # Формирование выборки 227
│   ├── bird_subset.py                  # Выборка 50 заблокированных вопросов и сравнение
│   ├── compare_hybrid_runs.py          # Сравнение новых запусков с baseline и Mistral
│   ├── audit_feature_results.py        # Аудит всех больших одиночных абляций без LLM
│   ├── audit_static_hybrid.py          # Проверка bundle, парные эффекты и экспорт плана без SSH/API
│   ├── static_hybrid_search.py         # Ручная очередь этапов, frozen controls/cohorts и сравнения
│   ├── static_hybrid_runtime.py        # Один прогон: checkpoint, usage и прежний SQL-генератор
│   ├── compare_static_hybrid_control.py # Офлайн-сравнение со свежим H без изменения frozen-протокола
│   ├── sql_output_contract.py         # Пустые/текстовые/не-query ответы → ошибка, отдельно от FEAT_16
│   ├── audit_question_schemas.py      # Офлайн-сопоставление gold SQL со snapshot и отдельный пересчёт
│   ├── guarded_hybrid_search.py       # V3: явная политика схем, новые controls и отдельный протокол
│   ├── guarded_hybrid_runtime.py      # Защита ответов и live-проверка каталога до платных вызовов
│   ├── static_hybrid_campaign.py      # 15 масок, fresh controls, фиксация train-финалистов и повторные train/dev
│   ├── resilient_hybrid_campaign.py   # Отдельная версия подключения; папка диагностики создаётся до child
│   ├── resilient_hybrid_runtime.py    # Read-only каталог через одно соединение на БД, retry и прогресс preflight
│   ├── transport_safe_campaign.py     # Перенос с provenance и отдельный ручной этап recovery
│   ├── transport_safe_runtime.py      # Первая ошибка API прерывает вопрос до checkpoint, deadline 180 с
│   ├── full_static_hybrid.py          # Четыре замороженных профиля, full-очередь и описательный отчёт
│   ├── full_static_hybrid_runtime.py  # Full-обёртка с прежними SQL/transport-правилами и schema preflight
│   ├── build_feature_dataset.py        # Старый builder labels; требуется доработка перед обучением
│   ├── build_configuration_dataset.py  # Новый офлайн-builder utility и валидация bundle
│   ├── configuration_splits.py         # Компоненты БД/семейств, train/dev/test и train-only CV
│   ├── evaluate_rule_selector.py        # Выбор правил на dev и offline replay отложенных ответов
│   ├── build_multilabel_dataset.py      # Офлайн-сборка полезности 33 флагов и экспорт 16 допустимых выходов
│   ├── multilabel_targets.py            # Политика исключений, utility и масок; без обучения
│   ├── prepare_classifier_encoder.py     # Загрузка в локальный cache, закрепление revision/SHA
│   ├── train_multilabel_classifier.py     # Обучение головы на train/dev и отдельная оценка меток
│   ├── classifier_benchmark.py           # Сравнение live-ответов с контролями на тех же ID
│   ├── experiment_structured_classifier.py # Joint quality loss, четыре train CV-варианта, dev replay
│   ├── prompt_versions.py              # Чтение точного промпта из main без переключения ветки
│   ├── diagnose_ambrosia.py             # Read-only диагностика пустых серверных БД
│   ├── check_openrouter.py             # Короткая платная проверка API, только при явном запуске
│   └── watch_baseline.ps1              # Наблюдение за конкретным процессом и остановка при сбое
├── local/                              # Shell-запуски с SSH
│   ├── run_full_static_hybrid.ps1 / .sh # Ручной запуск восьми full-прогонов; offline prepare/status/dry-run
│   ├── run_feature_ablation.ps1         # PowerShell-вход: выбирает Git Bash вместо WSL
│   ├── run_deepseek_ablation.ps1        # Та же серия с локальным .env.deepseek
│   ├── run_deepseek_combinations.ps1     # Ручной запуск трёх этапов комбинаций через SSH
│   ├── run_deepseek_relationships.ps1    # Ручной запуск опыта со связями через SSH
│   ├── run_multilabel_deepseek.ps1       # Preflight + ручной DeepSeek прогон выбранных классификатором фичей
│   ├── run_feature_ablation.sh          # Серия абляций через общий SSH launcher
│   ├── run_openrouter_baseline.sh       # Основной текущий запуск: туннель + Python runner
│   ├── start_tunnel.sh                 # Наблюдение/переподключение SSH
│   ├── run_single.sh                   # Рабочий исторический запуск произвольных FEAT через .env
│   ├── ablation_full.sh                # Исторические малые абляции
│   ├── ablation_large_bird.sh          # Исторические большие BIRD-абляции
│   ├── ablation_large_ambrosia.sh      # Исторические большие Ambrosia-абляции
│   └── run_baseline_large.sh           # Исторический baseline через .env
├── tests/                              # Офлайн-проверки наборов, путей, таймаутов, повторов
├── hints/                              # Старый cache FEAT_26; не смешивать train/test и модели
├── nl_descriptions/                    # Cache описаний FEAT_23; учитывать версию схемы/модели
├── Roadmap.md                          # Исходный архитектурный план динамического выбора фичей
├── practices_doc.md                    # Полный каталог практик; актуальные итоги — в docs
├── README.md                           # Исходная инструкция проекта из main, сохранена
├── CLAUDE.md                           # Существующая инструкция по работе с репозиторием
├── bird_benchmark.py                   # Исходная точка запуска из main, сохранена
├── ambrosia_benchmark.py               # Исходная точка запуска из main, сохранена
├── bird_evaluate_only.py               # Исходная переоценка; путь задаётся RESULTS_FILE
├── .env / .env.openrouter / .env.deepseek # Локальные секреты и подключения; не публиковать
├── .env.example                        # Исходный пример настроек из main
├── pyproject.toml / uv.lock            # Окружение и зависимости из main
├── requirements-classifier.txt         # Дополнительные PyTorch/Transformers зависимости
├── .cache/classifier/                  # Игнорируемые веса энкодера и download/pip-cache
├── .gitattributes                      # LF для bash-скриптов
├── .gitignore                          # Существующие исключения Git
├── .claude/                            # Локальные настройки инструмента, сохранены
└── .venv/                              # Локальное окружение, не исходный код проекта
```

## Файлы внутри одного запуска

- `metadata.json` — статус, модель, флаги, hashes, покрытие, токены и снимки расхода.
- `query_results.json` — checkpoint с ответами по question_id.
- `report.json` — итоговые метрики; у BIRD — также результаты по вопросам.
- `raw_responses.jsonl` — полные запросы/ответы; локальный журнал, исключён из Git.
- `bird.log` / `ambrosia.log` — ход работы; локальные файлы, исключены из Git.
- `all_gold_results.json` / `all_predicted_results.json` — исполненные результаты BIRD.
- `comparison.json` / `comparison.md` / `ANALYSIS.md` — дополнительные сравнения конкретных запусков.

Журналы `attempt_history.jsonl`, `question_usage.jsonl`, `schema_preflight.jsonl`,
`invalid_outputs.jsonl`, `transport_events.jsonl`, служебные `suite_status.json`,
`monitor_stop.json` и снимки `key_usage_snapshot.json` также остаются локально
в папках прогонов `research_results/`; правила заданы в `.gitignore`.
`metadata.json`, итоговые отчёты, манифесты и ответы датасета сохраняются для Git.
Личные `plan.md`, `docs/security_review.md` и материалы `защита/` не публикуются.

## Проверка и ручные запуски

Из корня проекта, **сначала dry-run** (нет API/БД):

```powershell
.venv\Scripts\python.exe scripts/run_openrouter_baseline.py --configuration hybrid --benchmark ambrosia --dry-run
.venv\Scripts\python.exe scripts/run_openrouter_baseline.py --configuration hybrid_no32_33 --benchmark bird --dry-run
```

Ручной запуск с SSH; платные вызовы начинаются только при выполнении команды:

```powershell
bash local/run_openrouter_baseline.sh --configuration hybrid --benchmark ambrosia
bash local/run_openrouter_baseline.sh --configuration hybrid_no32_33 --benchmark bird
```

Завершённый запуск с совпадающей конфигурацией пропускается. Новый эксперимент
записывайте в новую папку через `--output research_results/runs/имя_эксперимента`.
При одинаковой папке и другой конфигурации runner откажется смешивать ответы.
Старые пути вида `runs/...` принимаются Python runner как совместимые aliases,
но новые документы и команды используют `research_results/runs/...`.

Офлайн-сравнение полного BIRD:

```powershell
.venv\Scripts\python.exe scripts/compare_hybrid_runs.py --run-dir research_results/runs/gpt54mini_hybrid_bird_no32_33_full_20261023
```

Файлы, существующие в `main`, при уборке не редактировались и не удалялись.
Поэтому исходные README и точки запуска могут описывать старое расположение
выходных файлов. Современный путь запуска и каталог результатов описаны здесь.
