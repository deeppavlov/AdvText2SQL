# /feature — Добавление новой FEAT_N практики

## Чеклист для добавления фичи

При добавлении новой практики с номером N всегда модифицируются эти файлы:

### 1. `src/adv_text2sql/mcp_servers/text2sql_tool/src/text2sql_implementation.py`

Добавить флаг в блок объявлений (после других FEAT_* строк):
```python
FEAT_N = os.getenv("FEAT_N", "false").lower() == "true"  # #N короткое описание
```
Значение по умолчанию: `"true"` для стабильных фич, `"false"` для экспериментальных.

### 2. `src/adv_text2sql/mcp_servers/text2sql_tool/src/prompts.py`

Если фича требует нового промпта — добавить шаблон:
```python
NEW_PROMPT_TEMPLATE = """..."""
```

### 3. `local/ablation_full.sh`

Добавить в блок `unset`:
```bash
unset ... FEAT_N ...
```
Добавить в блок `printf`:
```bash
printf 'FEAT_N=false\n' >> "$tmp_env"
```
Добавить `run_feature` строку (с base deps):
```bash
run_feature "FEAT_N: описание" "feat_N_slug" $BASE FEAT_N=true
```
Добавить `feat_N_slug` в summary-цикл в конце файла.

### 4. `local/run_single.sh`

Добавить в `unset` и `printf` блоки (аналогично ablation_full.sh).

### 5. `CLAUDE.md`

Добавить строку в таблицу флагов:
```markdown
| `FEAT_N` | true/false | #N | Краткое описание эффекта |
```

### 6. `practices_doc.md`

Обновить статус практики #N с `not done` → `done`.
Добавить в колонку "Файл / метод" ссылку на реализацию.

## Типичные места интеграции

- **Exploration-фича** → метод вызывается в `build()` после сбора схемы
- **Execution-фича** → метод вызывается в `query()` или `generate_sql()`
- **Prompt-фича** → переключение шаблона внутри существующего метода

## Нумерация

Следующие свободные номера: **#21, #22, #23, #25, #26, #28, #30, #36**.
Номера 31, 37 зарезервированы (rejected / too complex).
