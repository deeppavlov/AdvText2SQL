# Сравнение генераторов синтетики

Каждый генератор создал синтетический Q-SQL датасет для `card_games`, на котором дообучен LoRA-адаптер поверх Qwen2.5-Coder-7B. Метрика — execution accuracy на BIRD card_games subset (реальные вопросы).

| Генератор | Accuracy | Correct/Total | Errors |
|---|---|---|---|
| gpt-4.1-nano [ru+en] | **35.5%** | 11/31 | 11 |
| Qwen-Coder-7B (без FT) | **32.3%** | 10/31 | 8 |
| gpt-4.1 [ru+en] | **32.3%** | 10/31 | 9 |
| gpt-5.1-codex-mini [ru+en] | **25.8%** | 8/31 | 14 |
| gpt-4o-mini [ru+en] | **25.8%** | 8/31 | 10 |

## Accuracy по сложности

| Генератор | challenging | moderate | simple |
|---|---|---|---|
| gpt-4.1-nano [ru+en] | 30.0 | 36.4 | 40.0 |
| Qwen-Coder-7B (без FT) | 20.0 | 36.4 | 40.0 |
| gpt-4.1 [ru+en] | 20.0 | 27.3 | 50.0 |
| gpt-5.1-codex-mini [ru+en] | 20.0 | 9.1 | 50.0 |
| gpt-4o-mini [ru+en] | 20.0 | 18.2 | 40.0 |
