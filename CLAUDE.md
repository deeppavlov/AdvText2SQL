# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AdvText2SQL is a text-to-SQL benchmarking framework for evaluating student-written tools against BIRD and Ambrosia benchmarks. Students implement a `Text2SQLGenerator` and run it against remote PostgreSQL databases.

## Commands

### Setup
```bash
pip install uv
cp .env.example .env  # Fill in DB_USER, DB_PASS, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_NAME
# Start SSH tunnel to remote DB (required before running benchmarks):
ssh -N -L 5444:10.11.1.6:5444 user_name@lnsigo.mipt.ru -p2278
```

### Running Benchmarks
```bash
uv run --env-file .env bird_benchmark.py        # SQL generation test
uv run --env-file .env ambrosia_benchmark.py    # Ambiguity detection test
uv run --env-file .env bird_evaluate_only.py    # Re-evaluate saved results (no LLM calls)
```

### Linting & Tests
```bash
ruff check .
pytest
```

## Architecture

### Benchmark Pipeline

`BenchmarkBIRD` / `BenchmarkAmbrosia` (in `benchmarks/`) runs three phases:

1. **build** — creates one `Text2SQLGenerator` instance per unique `db_id`, extracts schema, initializes system prompt
2. **predict** — iterates dataset questions, calls `tool.query(question)` → `ToolResponse`
3. **evaluate** — executes gold + predicted SQL on remote DB, compares result sets

Each LLM client is wrapped in `TokenTrackingClient` (`benchmarks/token_tracking.py`) for token aggregation.

### Tool Interface (what students implement)

`Text2SQLGenerator` in `src/adv_text2sql/mcp_servers/text2sql_tool/src/text2sql_implementation.py`:

```python
class Text2SQLGenerator:
    def __init__(self, db_uri: str, llm_client): ...
    def build(self): ...  # Extract schema, build system prompt
    def query(self, user_request: str) -> dict: ...
    # Must return: {"status": "success", "query": sql}
    #           or {"status": "ambiguous"}
    #           or {"status": "error"}
```

### Key Data Flow

```
query(user_request)
  → _check_ambiguity()        # LLM decides: "OK" or ambiguous
  → if ambiguous: return {"status": "ambiguous"}
  → for up to MAX_RETRIES:
      generate_sql()          # LLM generates SQL
      sanitize_sql()          # Fix known LLM output patterns
      _validate_sql()         # sqlglot parse + forbid DROP/DELETE/UPDATE
    → return {"status": "success", "query": sql}
```

### SQL Validation & Sanitization

`evaluate_bird.py` transpiles SQLite → PostgreSQL using sqlglot before execution. The generator also uses sqlglot to validate and sanitize LLM output:
- Wraps `SELECT DISTINCT ... ORDER BY` in subquery (PostgreSQL requirement)
- Forbids DROP, DELETE, UPDATE
- Converts SQLite-style casts

### Prompts

All LLM prompts are in `prompts.py` and written **in Russian** (the target domain is Russian banking analytics). There are four templates: system, SQL generation, ambiguity detection, and SQL verification.

### Schema Extraction

Two modes in `text2sql_implementation.py`:
- `_get_db_schema_light()` — table/column names + types only (fast, low token cost)
- `_get_db_schema_heavy()` — adds sample rows, unique values, cardinality (high token cost)

### MCP Server

`src/adv_text2sql/mcp_servers/text2sql_tool/main.py` wraps the generator as a FastMCP tool for use as an MCP server. This is separate from the benchmark pipeline.

## Important Constraints

- Remote DB is always accessed via SSH tunnel on `localhost:5444`
- All column data in the DB is TEXT type — explicit casting required for numeric/date comparisons
- Only SELECT queries are permitted (enforced by `_validate_sql()`)
- Student branches should be created off `main` and never pushed directly to `main`
- `MAX_RETRIES` defaults to 7; overridable via env var

## Baseline Accuracy (reference)
- BIRD: ~20–35% with gpt-3.5-turbo
- Ambrosia: ~50–65% ambiguity detection accuracy

## Autonomous Text2SQL Pipeline (ветка `feat/autonomous-pipeline`)

Параллельно с benchmark-фреймворком построен **автономный pipeline**: даёшь
PostgreSQL-URL → за ~30 минут получаешь fine-tuned модель и inference-сервис
под эту конкретную БД.

```bash
uv run python scripts/preflight.py        # проверка окружения
uv run text2sql --help                    # 7 команд pipeline
uv run text2sql init --db-url ...         # full orchestration: profile → generate → train
```

### Стадии (CLI subcommands)

| Команда | Что делает | Артефакт |
|---|---|---|
| `text2sql profile` | Извлекает schema/relationships/stats/samples из БД | `data/profiles/<db_id>/profile.json` |
| `text2sql generate` | Генерирует Q-SQL пары (template + LLM) + валидирует | `data/synthetic/*_validated.jsonl` |
| `text2sql build-dataset` | Собирает chat-format train.jsonl/val.jsonl | `data/finetune/<db_id>/` |
| `text2sql train` | Генерирует Colab notebook для FT | `notebooks/auto_train_<db_id>.ipynb` |
| `text2sql serve` | REPL-клиент поверх vLLM | (interactive) |
| `text2sql heal` | Собирает failed → корректирует через GPT-4 | `data/heal/<db_id>_corrections.jsonl` |
| `text2sql init` | Orchestrator: profile → generate → build → train | (multi) |

### Главный инвариант: train == inference

`schema_str` / `relationships_str` / `column_stats_str` в `profile.json` —
**byte-to-byte** идентичны выходу `Text2SQLGenerator.build()`. Это позволяет
`Text2SQLGenerator.load_from_profile()` загружаться за <50ms вместо 30+ сек.

Проверяется тестами:
- `tests/test_profiler.py::test_schema_str_invariant` (с реальной БД)
- `tests/test_training.py::test_dataset_builder_invariant_with_text2sql_generator` (unit)

### Модули в `src/adv_text2sql/`

- `profiler/` — извлечение метаданных БД
- `synth/` — генерация и валидация синтетических Q-SQL пар
- `training/` — chat-format dataset + Colab notebook template
- `serve/` — vLLM Docker + Python SDK + query logger
- `heal/` — self-healing loop через failed-queries
- `cli/` — top-level Typer entry point (`text2sql`)

Подробности: `src/adv_text2sql/README.md`.

### Запуск unit-тестов

```bash
uv run python -m pytest tests/ -m "not integration"   # 29 unit-тестов, ~1 сек
uv run python -m pytest tests/ -m integration          # требует SSH + DB env
```
