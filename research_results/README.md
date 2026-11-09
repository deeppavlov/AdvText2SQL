# Результаты исследований

Единое место хранения исходных ответов, логов, абляций и диагностик.
Краткие выводы по этапам находятся в [`docs/`](../docs/README.md).
Текстовые логи и JSONL-журналы запросов/попыток/диагностики пишутся локально
в подпапки соответствующих прогонов и исключены через [`.gitignore`](../.gitignore).
В Git сохраняем итоговые отчёты, metadata, манифесты и ответы для датасетов.
JSONL-датасеты и решения селектора не относятся к журналам и остаются доступными
для версионирования. Для проверки протоколов, закрепляющих хеши сырых журналов,
после чистого клонирования понадобятся локальные артефакты из архива экспериментов.
Презентации и исторические Markdown-отчёты перенесены в
[`защита/`](../защита/README.md). В папках абляций остаются исходные данные прогонов.

| Папка | Содержимое |
| --- | --- |
| `ambrosia_prompt_full_20261107/` | Завершён свежий full: 171/211 (81,04%), F1 60,78%, recall 49,21%; 14+32, исходные промпты, 498 448 токенов, pending пусты; [итоги](../docs/24_ambrosia_prompt_full.md) |
| `prompt_stage_summary_20261107.json` | Итоговая совместимая опора следующего этапа: BIRD compact + SQL seed 43 (132/241), Ambrosia 14+32 с исходным промптом (171/211); candidate identity/source SHA, лимиты интерпретации, deployed=false |
| `bird_prompt_full_20261107/` | Завершены 482 оценки: новый SQL seed 43 — 132/241 (54,77%) против исходных 126/241 (52,28%), +2,49 п.п.; 17 исправлено / 11 ухудшено, 3,22 млн токенов; conclusion и integrity review; [результаты](../docs/23_bird_prompt_full.md) |
| `ambrosia_prompt_dev_20261107/` | Завершены 552 оценки: исходный 14+32 — 80,43%, MIPRO — 76,81% / 78,26%, TextGrad — 73,91%; сохраняем исходный, conclusion и SHA закреплены; 1,20 млн токенов; [итоги](../docs/22_ambrosia_prompt_dev.md) |
| `bird_compact_prompt_transfer_20261107/` | Завершены три compact-dev повтора / 880 оценок: MIPRO few-shot 45,45% против исходных 38,79%, +6,67 п.п.; 9,73 млн токенов; [итоги](../docs/21_bird_compact_prompt_transfer.md) |
| `prompt_optimization_comparison_20261105/` | Выполнены 18/24 исходных поисков; MIPRO zero-shot Ambrosia: 95% / 90% / 83,33%, среднее 89,44% против 83,33% на 60 inner-вопросах. Остальные методы Ambrosia имеют один повтор; shortlist dev подготовлен отдельно; [результаты](../docs/20_prompt_optimizer_comparison.md) |
| `prompt_optimization_20261105/` | Завершённый первый GEPA-пилот: seed 17/24, 194 269 токенов, два ложных отклонения guard; dev не запускался; [разбор](../docs/19_prompt_optimization_pilot.md) |
| `prompt_optimization_20261105_guardfix/` | Исправленный поиск завершён: seed 17/24, два новых промпта 16/24, 399 905 токенов; pilot_conclusion.json фиксирует сохранение исходного промпта и отсутствие необходимости dev-подтверждения; [итоги](../docs/19_prompt_optimization_pilot.md#итог-исправленного-поиска) |
| `ablation_results_large/` | Исторический Mistral baseline и одиночные большие абляции |
| `ablation_results_small/` | Малые абляции и исторические эксперименты; `feat_best_hybrid_large/` внутри содержит **large** best hybrid |
| `runs/` | Запуски GPT/DeepSeek, пробы промптов, отчёты, checkpoints, полные ответы и аудит фичей |
| `deepseek_flash_ablation_large_20261024/` | Создаётся при ручном запуске новой серии: baseline и отдельные фичи; [протокол](../docs/03_deepseek_ablation.md) |
| `deepseek_direct_ablation_large_20261024/` | Отдельная серия через прямой API DeepSeek и `.env.deepseek`; создаётся при ручном запуске |
| `deepseek_combinations_large_20261026/` | Три этапа комбинаций на BIRD/Ambrosia, строгий подсчёт ошибок и сравнение с прежними контролями; [протокол](../docs/04_deepseek_combinations.md) |
| `deepseek_combinations_large_20261026/search_v2_20261029/` | Продолжение поиска: 20 новых гипотез по этапам и отдельный повтор H, grouped train/dev/full, новые summary; [запуск](../docs/11_static_hybrid_experiments.md) |
| `deepseek_combinations_large_20261026/quality_audit_20261029/` | Аудит всех 468 вопросов, 16 несоответствий Ambrosia, невалидные ответы и отдельные пересчёты; [выводы и v3](../docs/12_results_and_data_quality.md) |
| `deepseek_combinations_large_20261026/search_v3_20261029/` | Создаётся при ручной подготовке/запуске нового протокола с проверкой ответов и реального каталога; по умолчанию несоответствия блокируют запуск |
| `deepseek_combinations_large_20261026/final_search_20261030/` | Ограниченная серия: 15 масок, свежие контроли, фиксация финалистов и три повтора train/dev; [команды и объём](../docs/13_static_hybrid_confirmation.md) |
| `deepseek_combinations_large_20261026/final_search_20261030_connectionfix/` | Продолжение с теми же масками и cohorts в отдельной версии подключения: retries preflight, диагностические папки до child; [исправление](../docs/14_static_hybrid_connection_fix.md) |
| `deepseek_combinations_large_20261026/final_search_20261030_transportfix/` | Ограниченный поиск завершён, recovery сохранён; static_hybrid_conclusion.json фиксирует 14+32 как лучший наблюдаемый dev-профиль без доказанного выигрыша над H; [итоги](../docs/17_static_hybrid_dev.md) |
| `deepseek_full_static_20261103/` | Завершены восемь full-запусков: BIRD 241 / Ambrosia 211; compact лучший BIRD/среднее, 14+32 лучшая Ambrosia; [результаты, ошибки и расходы](../docs/18_full_static_hybrid.md) |
| `deepseek_relationships_large_20261026/` | FEAT_1 к выбранному гибриду 14+32+35; два новых запуска с прежними контролями; [протокол](../docs/04_deepseek_combinations.md) |
| `rule_selector_deepseek_20261028/` | Офлайн-селектор: зафиксированный выбор по dev, сравнение train/dev/test и решения по вопросам; [отчёт](../docs/06_rule_selector.md) |
| `multilabel_classifier_deepseek_20261029/` | Обученная линейная голова, замороженный encoder manifest, scaler и оценка меток; [протокол и ручной DeepSeek](../docs/08_multilabel_classifier.md) |
| `multilabel_deepseek_test_20261029/` | Завершённый test 100 с выбранными фичами: BIRD 18/55, Ambrosia 37/45; [сравнение с контролями](multilabel_deepseek_test_20261029/comparison.md) |
| `multilabel_structured_dev_20261029/` | Офлайн-эксперимент совместного выбора 35 / 14+35 / 14+32+35, четыре train CV-варианта и dev replay; [анализ](../docs/09_classifier_improvement.md) |
| `static_hybrid_audit_20261029/` | Офлайн-аудит одиночных эффектов и предложение 20 новых масок; новые прогоны не запускались; [семантика и план](../docs/10_static_hybrid_search.md) |
| `legacy_notes/` | Старые планы и заметки из `local/old`; это исторический контекст, не актуальные инструкции |
| `legacy_snapshots/` | Прежние корневые JSON-снимки схем и результатов; не опорные метрики новых прогонов |
| `organization_20261023/` | Проверка целостности переноса и список удалённых служебных файлов |

## Перенос 23.10.2026

Папки `ablation_results_large`, `ablation_results_small`, `runs` перенесены сюда
из корня. Содержимое исходных результатов не переписывалось. Поэтому старые
metadata, команды и абсолютные ссылки внутри исторических файлов могут
содержать прежние пути. Для чтения добавляйте `research_results/` перед старым
корневым путём; активные скрипты уже обновлены.

Старые `run.py`/`reevaluate.py` внутри DeepSeek-прогона — сохранённые снимки
разовой процедуры, не поддерживаемые точки запуска. Используйте инструменты
из `scripts/` и `local/`. Не запускайте весь архив рекурсивно как код или тесты.

Исторические логи могут содержать DB URI. Не публикуйте архив целиком без
отдельной проверки содержимого. `.env` и `.env.openrouter` остаются в корне,
секреты в документацию не переносятся.

Новые результаты запускайте в отдельную папку и сохраняйте configuration,
выборку, checkpoint, report и provenance. Завершённые исходные прогоны не
редактируйте ради новых выводов; выводы записывайте в документы этапа.
