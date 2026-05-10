# Synthetic Text2SQL examples

Папка содержит сгенерированные мной (Claude-в-сессии) пары `(question, SQL)` для дообучения. **Не правится руками** — генерируется батчами и проходит execution-фильтр.

## Workflow

```
1. Я в чате генерирую пакет (50-200 примеров)  →  data/synthetic/raw_<timestamp>.jsonl
2. Запускаешь execution-filter на удалённой PG  →  data/synthetic/<timestamp>.jsonl  (выживают только работающие)
3. build_finetune_dataset.py читает все *.jsonl кроме raw_* и мерджит в train.jsonl
```

## Формат строки

Один JSON-объект на строку (jsonl). Те же поля, что в BIRD/Ambrosia, плюс `task`:

### SQL-генерация (BIRD-аналог)

```json
{
  "question_id": "syn_0001",
  "db_id": "card_games",
  "question": "Сколько карт в наборе с кодом 'LEA'?",
  "SQL": "SELECT COUNT(*) FROM cards c JOIN sets s ON c.set_id = s.id WHERE s.code = 'LEA';",
  "evidence": "",
  "task": "sql"
}
```

### Ambiguity-классификация (Ambrosia-аналог)

```json
{
  "question_id": "syn_amb_0001",
  "db_id": "vague_2cols_compensation",
  "question": "Покажи лучших сотрудников по компенсации",
  "SQL": "ambiguous",
  "ambig_question": "Лучших — по какой метрике (зарплата / бонусы / total)? Период?",
  "task": "ambiguity"
}
```

Для **non**-ambiguous примеров на ambiguity-задаче кладём конкретный SELECT в `SQL` — модель учится отвечать "OK".

## Hold-out

Скрипт `build_finetune_dataset.py` автоматически выкидывает строки, у которых `db_id ∈ bird_small ∪ ambrosia_small`. Так что синтетику можно генерировать без оглядки — фильтр сработает на этапе сборки.

## Запуск execution-filter

```bash
uv run --env-file .env scripts/filter_synthetic_by_execution.py \
    data/synthetic/raw_2026-05-08.jsonl \
    data/synthetic/2026-05-08.jsonl
```

Принимает входной jsonl, прогоняет каждый SQL на `localhost:5444`, оставляет только те, что вернули непустой результат без ошибок. Падающие — логируются в `*.rejected.jsonl` рядом, чтобы можно было посмотреть глазами и поправить.

## Учебный нюанс: почему raw_* отдельным префиксом

Соглашение: `raw_*.jsonl` — то, что я только что сгенерировал, **без проверки**. После execution-фильтра файл переписывается под именем без `raw_`. `build_finetune_dataset.py` читает только последние — это защита от случая «забыл прогнать фильтр и затащил галлюцинации в train».
