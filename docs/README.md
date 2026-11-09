# Документация по этапам

Здесь ведём краткий журнал результатов: один файл на законченный этап или
самостоятельный экспериментальный блок. Полные логи и ответы находятся в
[`research_results/`](../research_results/README.md).

PowerShell-скрипты `.ps1` используются только локально и исключены из Git;
в репозитории сохраняются Bash/Python-запуски. Для воспроизведения протоколов,
закрепляющих хеши `.ps1`, нужны исходные локальные скрипты, см.
[структуру проекта](project_structure.md).

| Этап | Файл | Состояние |
| --- | --- | --- |
| Phase 0: GPT baseline, гибриды, сравнение с Mistral | [01_gpt_research.md](01_gpt_research.md) | Результаты зафиксированы |
| Phase 1–2: выбор фичей и классификатор | [02_feature_selector.md](02_feature_selector.md) | Rule-based selector и multilabel проверены; статический гибрид пока сильнее |
| DeepSeek Flash: baseline и отдельные фичи | [03_deepseek_ablation.md](03_deepseek_ablation.md) | Все 68 запусков завершены |
| DeepSeek Flash: комбинации и PK/FK-подсказки | [04_deepseek_combinations.md](04_deepseek_combinations.md) | Все 8 запусков завершены; опора 14+32+35, экономичный контроль 35 |
| Phase 0.2: полезность конфигураций и групповой split | [05_configuration_dataset.md](05_configuration_dataset.md) | 468 записей, 60 БД, train/dev/test и CV готовы |
| Phase 1: селектор по правилам | [06_rule_selector.md](06_rule_selector.md) | Реализован и проверен офлайн; на test уступил статическому hybrid |
| Phase 2, шаг 1: multilabel-разметка | [07_multilabel_dataset.md](07_multilabel_dataset.md) | Датасет готов: 468 вопросов, 16 допустимых выходов, маски и 2 340 условных наблюдений |
| Phase 2, шаги 2–3: multilabel-классификатор | [08_multilabel_classifier.md](08_multilabel_classifier.md) | DeepSeek test 100 завершён: BIRD 18/55, Ambrosia 37/45; ниже статического гибрида |
| Улучшение классификатора: совместные маски | [09_classifier_improvement.md](09_classifier_improvement.md) | Четыре варианта на train CV; совместный loss не дал преимущества на dev, test не оценивался |
| Семантика фичей и поиск статического гибрида | [10_static_hybrid_search.md](10_static_hybrid_search.md) | Исходный план: 36 номеров, зависимости и 20 предложенных масок; последующие эксперименты — в отчётах 11–18 |
| Поиск статического гибрида | [11_static_hybrid_experiments.md](11_static_hybrid_experiments.md) | Core, повтор H и additions завершены; актуальный разбор данных и дальнейший протокол — в отчёте 12 |
| Итоги additions, невалидные ответы и соответствие схем | [12_results_and_data_quality.md](12_results_and_data_quality.md) | Исторический аудит: 40 невалидных ответов, 16 несоответствий Ambrosia; исправленный протокол и итоговые профили — в отчётах 13–18 |
| Подтверждение статического гибрида | [13_static_hybrid_confirmation.md](13_static_hybrid_confirmation.md) | Screening, три train- и три dev-повтора завершены; итоговый разбор и фиксация выбранного профиля — в отчёте 17 |
| Исправление соединения и preflight | [14_static_hybrid_connection_fix.md](14_static_hybrid_connection_fix.md) | Отдельная connectionfix-версия: ограниченные повторы каталога и сохранение исходной ошибки monitor |
| Phase 1.5: оптимизация промптов | [15_prompt_optimization_plan.md](15_prompt_optimization_plan.md) | Обзор и исходный план; ограниченная серия, dev и full завершены, текущие результаты — в отчётах 20–24 |
| Транспорт API и восстановление вопросов | [16_transport_recovery.md](16_transport_recovery.md) | Recovery и train/r3 завершены; compact лучший по среднему train, 14+32 — по Ambrosia; все кандидаты выше H в 3/3 повторах, интервалы включают ноль; результаты dev — в отчёте 17 |
| Статический гибрид: dev | [17_static_hybrid_dev.md](17_static_hybrid_dev.md) | Подтверждение завершено: лучший наблюдаемый 14+32 — 60,16%, compact 60,10%, H 59,49%; устойчивый выигрыш над H не доказан; conclusion.json зафиксирован |
| Статический гибрид: полные наборы | [18_full_static_hybrid.md](18_full_static_hybrid.md) | Все восемь запусков завершены: compact BIRD 51,45% и среднее 64,59%; 14+32 Ambrosia 80,09%; полный разбор ошибок и расходов |
| Phase 1.5: zero-shot GEPA-пилот | [19_prompt_optimization_pilot.md](19_prompt_optimization_pilot.md) | Только ограниченный SQL-пилот: seed 17/24, два новых промпта 16/24; сохраняем seed. Вывод на весь этап не переносится |
| Phase 1.5: сравнение оптимизаторов | [20_prompt_optimizer_comparison.md](20_prompt_optimizer_comparison.md) | Завершены 18/24 поисков GEPA/MIPROv2/TextGrad; затем согласованный shortlist и отдельные dev/full. Шесть ambiguity-поисков не выполнялись |
| BIRD: промпты на compact | [21_bird_compact_prompt_transfer.md](21_bird_compact_prompt_transfer.md) | Завершены 3 повтора / 880 оценок; MIPRO few-shot 45,45% против исходных 38,79%, +6,67 п.п.; 9,73 млн токенов |
| Ambrosia: найденные промпты на dev | [22_ambrosia_prompt_dev.md](22_ambrosia_prompt_dev.md) | Завершены 552 оценки: сохраняем исходный 14+32, 80,43%; MIPRO 76,81% / 78,26%, TextGrad 73,91%. Улучшение inner не подтвердилось; 1,20 млн токенов |
| BIRD: полный контроль SQL-промпта | [23_bird_prompt_full.md](23_bird_prompt_full.md) | Завершены 482 оценки: MIPRO seed 43 без demos — 132/241 (54,77%) против 126/241 (52,28%), +2,49 п.п., 17/11 парных исходов; 3,22 млн токенов; один ретроспективный full |
| Ambrosia: свежий полный контроль | [24_ambrosia_prompt_full.md](24_ambrosia_prompt_full.md) | Завершён: 171/211 (81,04%), F1 60,78%, recall 49,21%; 14+32 с исходными промптами, 498 448 токенов. Сведены итоговые опоры BIRD/Ambrosia, дальше train utility |
| Навигация по проекту | [project_structure.md](project_structure.md) | Актуальная структура |

Презентации, текст выступления и исторические отчёты собраны отдельно в
[`защита/`](../защита/README.md).

Архитектурный план — [`Roadmap.md`](../Roadmap.md). Каталог практик —
[`practices_doc.md`](../practices_doc.md). Старые выводы в этих документах не
заменяют измеренные результаты этапов выше.

Для каждого следующего этапа добавляем файл со следующим свободным номером и запись в эту
таблицу. Внутри достаточно указать цель, конфигурацию/выборку, результат,
сравнение, ограничения и следующий шаг. План и выполненный результат отмечаем
раздельно; метрики сопровождаем размером набора и ссылкой на исходный отчёт.

Сравнения оформляем таблицами: указываем модель/конфигурацию, размер выборки,
метрики и при необходимости изменение относительно baseline в процентных
пунктах. Выводы и ограничения кратко поясняем под таблицей. Полные прогоны,
диагностические подвыборки и составные оценки явно различаем.
