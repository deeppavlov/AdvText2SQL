# Эксперимент: сравнение моделей-генераторов синтетических Q-SQL пар

**Гипотеза**: качество fine-tuned Text2SQL-модели существенно зависит от того,
какая модель использовалась для генерации обучающего набора синтетических
Q-SQL пар. Дороже не всегда лучше.

**Цель**: эмпирически выбрать минимально-достаточную модель-генератор для
будущих запусков `text2sql generate`.

## Методология

1. **Фиксированная целевая БД**: `card_games` (BIRD-train, 6 таблиц, 113 колонок).
   Выбор: средняя сложность, разнообразие типов колонок, есть валидационный
   subset из BIRD-eval.

2. **Фиксированный профайл**: один раз создаётся `data/profiles/card_games/profile.json`,
   используется для всех 4 ранов.

3. **Фиксированный target_count**: 500 примеров.

4. **Переменная**: модель-генератор:
   - `gpt-4o` (OpenAI, сильная универсальная)
   - `claude-haiku` (Anthropic, средняя дешёвая)
   - `qwen-coder-32b` (локально, через vLLM или OpenRouter)
   - `qwen-coder-7b` (локально, базовая модель которую мы файнтюним)

5. **Одинаковая валидация**: `Validator` с execute-фильтром +
   `LLMJudge` с GPT-4 (как single source of truth).

6. **Одинаковая FT-конфигурация**: r=16, alpha=32, lr=2e-4, 1 epoch, batch=8.
   Базовая модель — Qwen2.5-Coder-7B-Instruct-bnb-4bit.

7. **Eval**: held-out subset из BIRD-eval по схеме `card_games` (~50 вопросов).
   Метрика — execution accuracy.

## Запуск

```bash
# Для каждого генератора:
for GEN in gpt-4o claude-haiku qwen-coder-32b qwen-coder-7b; do
    LLM_MODEL_NAME=$GEN uv run text2sql generate \
        --profile data/profiles/card_games/profile.json \
        --count 500 --generator llm --llm-model $GEN \
        --judge --out data/synthetic/exp_${GEN}/

    uv run text2sql build-dataset \
        --profile data/profiles/card_games/profile.json \
        --synthetic data/synthetic/exp_${GEN}/card_games_raw_llm_validated.jsonl \
        --out data/finetune/exp_${GEN}/

    uv run text2sql train --db-id card_games \
        --output-dir notebooks/exp_${GEN}/
    # → запускаем сгенерированный notebook в Colab вручную
done
```

## Результаты (заполняется по мере экспериментов)

### Стадия GENERATE

| Генератор | Стоимость 500 пар | Pass rate (parse+execute) | После judge | Дедуп Q | Финальные примеров |
|---|---|---|---|---|---|
| gpt-4o          | $___ | ___ % | ___ % | ___ | ___ |
| claude-haiku    | $___ | ___ % | ___ % | ___ | ___ |
| qwen-coder-32b  | ~$0  | ___ % | ___ % | ___ | ___ |
| qwen-coder-7b   | ~$0  | ___ % | ___ % | ___ | ___ |

### Стадия TRAIN

| Генератор | Train loss (1 epoch) | Val loss | Время FT |
|---|---|---|---|
| gpt-4o          | ___ | ___ | ___ мин |
| claude-haiku    | ___ | ___ | ___ мин |
| qwen-coder-32b  | ___ | ___ | ___ мин |
| qwen-coder-7b   | ___ | ___ | ___ мин |

### Стадия EVAL (на BIRD card_games subset)

| Генератор | Execution accuracy | Δ vs baseline (без FT) | Простые | Moderate | Challenging |
|---|---|---|---|---|---|
| baseline (no FT) | 18.18% | — | 22.22% | 0% | 33.33% |
| gpt-4o           | ___ % | +___ pp | ___ % | ___ % | ___ % |
| claude-haiku     | ___ % | +___ pp | ___ % | ___ % | ___ % |
| qwen-coder-32b   | ___ % | +___ pp | ___ % | ___ % | ___ % |
| qwen-coder-7b    | ___ % | +___ pp | ___ % | ___ % | ___ % |

## Качественный анализ

### Какие категории вопросов лучше пишет каждый генератор?

(заполнить после прогона: разнообразие, сложность, естественность языка)

- `gpt-4o`: ___
- `claude-haiku`: ___
- `qwen-coder-32b`: ___
- `qwen-coder-7b`: ___

### Типичные ошибки в SQL

(заполнить после прогона: галлюцинации имён колонок, неверные JOIN, etc.)

- `gpt-4o`: ___
- `claude-haiku`: ___
- `qwen-coder-32b`: ___
- `qwen-coder-7b`: ___

## Вывод (заполнить после)

**Рекомендация для производства**: ___

**Объяснение**: ___

**Ограничения эксперимента**: только одна БД, фиксированный target_count.
Для генерализации стоит повторить на 2-3 БД разной сложности.
