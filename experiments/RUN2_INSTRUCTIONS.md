# Финальный прогон Эксп. 2 — полные bi-датасеты

После дедлайн-демо: тренируем на 100% датасета вместо 25%.

## Что уже готово (не трогаем)
| | Статус |
|---|---|
| Реестр `experiments/registry.json` | 5 bi-экспериментов |
| Сгенерённая+валидированная синтетика `data/exp/*_bi/validated.jsonl` | ✓ есть для всех 5 |
| Собранные `train.jsonl`/`val.jsonl` локально | ✓ есть для всех 5 |
| Перевод BIRD на русский `data/bird_large_ru.json` | ✓ есть |

## Что меняется по сравнению с демо-прогоном
- `SAMPLE_FRAC` в train-ноутбуке: **0.25 → 1.0** (весь датасет)
- Добавлена resume-логика: уже обученные/проинференсенные эксперименты пропускаются

## Тайминг
| Этап | На модель | На 5 моделей |
|---|---|---|
| Обучение (full 800-900 примеров, A100) | ~40 мин | ~3.5 часа |
| Инференс (31 BIRD-вопрос) | ~3 мин | ~15 мин |
| Локальное сравнение | мгновенно | мгновенно |

---

## План по шагам

### Шаг 0 — синхронизация Drive
Если что-то менялось локально в `data/exp/*/train.jsonl` — перезалей на Drive. Скорее всего ничего не менялось, train.jsonl те же что вчера.

### Шаг 1 — загрузка обновлённых ноутбуков
В `MyDrive/text2sql/` перезаписать:
- `train_all_experiments.ipynb` (SAMPLE_FRAC=1.0, resume)
- `eval_all_experiments.ipynb` (resume)

### Шаг 2 — обучение в Colab
1. Открыть `train_all_experiments.ipynb` в Colab (A100).
2. Запустить ячейку 0 (install) → **перезапустить сеанс**.
3. Запустить ячейки 1, 2 (импорты + функция).
4. Запустить ячейки 3–7 (по одной на эксперимент).
   - Если адаптер для `gpt41_bi` (1/4-вариант) уже лежит на Drive — он пропустится. Хочешь его перетренировать на полных данных:
     - Либо удали папку `MyDrive/text2sql/gpt41_bi/adapter/` перед запуском.
     - Либо в ячейке 1 поставь `SKIP_IF_EXISTS = False` только для этого прогона.

### Шаг 3 — инференс в Colab
1. Открыть `eval_all_experiments.ipynb` (можно в том же сеансе после обучения).
2. Запустить ячейку 0 → перезапуск (если новый сеанс).
3. Запустить ячейки 1–3 (импорты, данные, промпт).
4. Запустить ячейки 4–9: `baseline`, потом 5 адаптеров.
   - Старые predictions с 1/4-датасета пропустятся. Чтобы перезапустить — удали `eval_predictions.json` или `SKIP_IF_EXISTS = False`.

### Шаг 4 — скачивание + локальное сравнение
Скачать `eval_predictions.json` из каждой папки на Drive в `data/exp/<имя>/`:
```
MyDrive/text2sql/baseline/eval_predictions.json     → data/exp/baseline/eval_predictions.json
MyDrive/text2sql/gpt41_bi/eval_predictions.json     → data/exp/gpt41_bi/eval_predictions.json
MyDrive/text2sql/codexmini_bi/eval_predictions.json → data/exp/codexmini_bi/eval_predictions.json
MyDrive/text2sql/gpt41nano_bi/eval_predictions.json → data/exp/gpt41nano_bi/eval_predictions.json
MyDrive/text2sql/gpt4omini_bi/eval_predictions.json → data/exp/gpt4omini_bi/eval_predictions.json
MyDrive/text2sql/gpt41mini_bi/eval_predictions.json → data/exp/gpt41mini_bi/eval_predictions.json
```
Локально:
```bash
nc -z localhost 5444 && echo OK
uv run --env-file .env python scripts/compare_generators.py
```
Получишь финальную таблицу accuracy в `experiments/generator_comparison.md`.

### Шаг 5 — обновить итоговый отчёт
Открой `experiments/final_results.md` и замени плейсхолдеры в колонке «Эксп. 2: RU+EN-синтетика» на реальные числа из `generator_comparison.md`.

---

## Если хочешь сэкономить время Colab

- **Тренируй только те, что считаешь нужными**: вызови `train_one('gpt41_bi')` напрямую в любой ячейке. Реестр всё равно подхватится, но обучить можно выборочно.
- **Прогон самых интересных**: gpt-4.1 + gpt-4o-mini (по гипотезе они лучшие). Если результаты хорошие — прогонишь остальные потом.
- **Если у тебя меньше 15 единиц A100**: оставь только 2-3 модели. Полный прогон 5 × 40 мин = ~30 единиц.

## Контрольная проверка после обучения
В логах **не должно быть** строки `Tried to allocate XX.XX GiB` — это OOM. Если есть:
- проверь что `SAMPLE_FRAC=1.0` и `eval_strategy='no'` (мой фикс),
- проверь, не запущены ли две сессии Colab (по 16 ГБ съедают и взрываются).

## Что показывать после прогона
В `experiments/final_results.md` главная таблица станет такой:

| Подход | Train data | Accuracy |
|---|---|---|
| без FT | — | 19.4% |
| Эксп. 0 (full BIRD FT) | ~10k человеческих | 38.7% |
| **Эксп. 2 / gpt-4.1 / bi** | 1000 автогенерированных | **<реальное число>** |
| Эксп. 2 / gpt-4o-mini / bi | 1000 автогенерированных | <число> |
| ... | ... | ... |

Главный месседж сохраняется: автономный пайплайн на синтетике сопоставим с classical full-BIRD FT.
