# Валидация адаптивного `target_count` (Горизонт-1.2)

**Вопрос:** даёт ли адаптивный объём синтетики
`target_count = max(200, complexity_score * 50)` выигрыш над фиксированным —
по точности и/или по sample-efficiency?

## Дизайн

Генератор фиксируем (лучший: `gpt-4.1`, RU+EN), варьируем только объём:

| Эксп. | `count_per_lang` | Всего пар (card_games) |
|---|---|---|
| `tc_fixed_1000` | 500 | 1000 |
| `tc_auto` | `"auto"` | ≈1391 (complexity≈27.8 → 695/язык) |

`"auto"` резолвится из `profile.target_synthetic_count()`, поделённого между
языками (см. `scripts/_registry.py::Experiment.resolved_count_per_lang`) — иначе
двуязычный эксп. сгенерировал бы бюджет ×2 и сравнение было бы нечестным.

## Метрики

- **Execution accuracy** на BIRD card_games subset (`text2sql eval`).
- **Sample-efficiency** = accuracy / число обучающих пар. Ключевая метрика:
  адаптивный подход полезен, если держит точность при меньших затратах или
  растёт быстрее на сложных БД.

## Как запустить

```bash
# 1. Генерация + валидация (нужен SSH-туннель + LLM_* + DB_* env)
uv run --env-file .env python scripts/exp_generate.py \
    --registry experiments/registry_target_count.json

# 2. Сборка train/val
uv run --env-file .env python scripts/exp_build.py \
    --registry experiments/registry_target_count.json

# 3. Обучение (Colab A100) — по одному ноутбуку на эксп., затем:
# 4. Eval каждой модели
uv run text2sql eval --profile data/profiles/card_games/profile.json \
    --gold data/bird_large.json --model-url http://localhost:8000/v1 \
    --model-name <adapter> --db-url postgresql+psycopg://...@localhost:5444/card_games
```

## Расширение на разброс по сложности

Одна БД не валидирует *адаптивность* — нужен контраст. Чтобы повторить пару
`fixed` vs `auto` на БД другой сложности (низкой/высокой), скопируй этот
registry, поменяй `db_id` + `profile_path` на другую БД с сервера и прогони те
же 3 шага. Адаптивный подход оправдан, если на сложной БД `auto` обгоняет
фиксированный бюджет, а на простой — не хуже при меньшем объёме.
