# Данные выбора конфигураций

Текущая версия — [`deepseek_20261028/`](deepseek_20261028/summary.md),
протокол — [`docs/05_configuration_dataset.md`](../../docs/05_configuration_dataset.md).

Сборка и проверка из корня проекта:

```powershell
.venv\Scripts\python.exe scripts/build_configuration_dataset.py
.venv\Scripts\python.exe scripts/build_configuration_dataset.py --verify-only
```

Модельными признаками служит только `inputs`: question, evidence, schema_stats.
Остальные поля records — идентификаторы, targets и аудит. Для безопасного чтения
входов есть отдельный `inputs.jsonl`; его ID не является текстовым признаком.

Utility отличается от правильности. Ничьи не являются наблюдаемым преимуществом
по качеству, пропуски сохраняются как null, ошибки генерации — как неуспех.
Основные кандидаты: 35 и 14+32+35; baseline и вариант с 1 сохранены для анализа.
Точные затраты отдельных вопросов неизвестны и помечены маской.

Train/dev/test разделены по БД и известным семействам. Test ретроспективный:
наборы уже участвовали в выборе конфигураций. Данные не являются независимым
финальным тестом или готовыми оптимальными multilabel-метками всех фич.
