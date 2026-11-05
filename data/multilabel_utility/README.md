# Multilabel-разметка полезности фичей

`deepseek_20261029/` — производные данные из 76 сохранённых прогонов DeepSeek:
baseline и 33 одиночных флага на двух benchmark, плюс восемь запусков комбинаций.
В наборе 468 вопросов и 16 допустимых выходов классификатора.

Полный протокол, причины исключений и ограничения —
[docs/07_multilabel_dataset.md](../../docs/07_multilabel_dataset.md).

```powershell
.venv\Scripts\python.exe scripts/build_multilabel_dataset.py
.venv\Scripts\python.exe scripts/build_multilabel_dataset.py --verify-only
```

| Файл | Назначение |
| --- | --- |
| `records.jsonl` | Входы, аудит, наблюдения всех флагов, полезность и маски |
| `examples.jsonl` | ID, inputs и целевые векторы; без БД, SQL и исходных ответов |
| `conditional_labels.jsonl` | Добавление фичи в конкретном контексте; отдельно от одиночных targets |
| `manifest.json` | Порядок 16 выходов, модель, источники, политика меток и SHA-256 |
| `splits.json`, `cv_folds.json`, `ambrosia_family_holdout.json` | Неизменённые групповые разделения Phase 0.2 |
| `summary.json`, `summary.md` | Число меток, классы, исключения и распределения |

Модельные признаки — только `inputs`: question, evidence, schema_stats.
ID нужен для сопоставления с split, а не для обучения как текстовый признак.
Вектор выходов упорядочен по `manifest.target_features`.

`benefit_targets=1` означает наблюдаемое улучшение относительно all-false baseline;
0 — отсутствие улучшения в этом парном проходе. `loss_mask=false` означает
недоступную или исключённую метку, соответствующий target остаётся null.
`preference_targets` использует только различия правильности +1/−1; ничьи
исключены отдельной `preference_mask`.

Это слабая наблюдательная разметка, не оптимальная маска комбинации и не
доказанный причинный эффект фичи. Test уже просмотрен на предыдущем этапе.
Обучения, вызовов LLM, установки зависимостей и подключений к БД при сборке нет.
