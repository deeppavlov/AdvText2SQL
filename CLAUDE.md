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

## Ablation Feature Flags

Feature flags gate the schema/stats enrichment in `Text2SQLGenerator.build()`. They are read at **module import time** — not per-call — so they must be set before any `uv run`.

FEAT_N maps 1:1 to Practice #N in `practices_doc.md`. Flags are read at **module import time** — must be set before `uv run`.

| Flag | Default | Practice | Effect when `true` |
|------|---------|----------|--------------------|
| `FEAT_1`  | `true`  | #1  | FK/PK relationships in prompt |
| `FEAT_2`  | `true`  | #2  | Column statistics (min/max/top-K) |
| `FEAT_3`  | `true`  | #3  | Regex type detection for TEXT columns (requires FEAT_2) |
| `FEAT_4`  | `true`  | #4  | `pg_stat` row count (requires FEAT_2) |
| `FEAT_5`  | `true`  | #5  | Light schema (table/column names + types) |
| `FEAT_6`  | `false` | #6  | Heavy schema — sample rows + enums (off by default) |
| `FEAT_7`  | `true`  | #7  | Compact stats formatting (requires FEAT_2) |
| `FEAT_8`  | `true`  | #8  | Rollback guards on failed PG transactions |
| `FEAT_9`  | `true`  | #9  | Dump `db_schemas.json` after build phase |
| `FEAT_10` | `true`  | #10 | `build()` timing + prompt-size log |
| `FEAT_11` | `true`  | #11 | JSON structured logging to file |
| `FEAT_12` | `true`  | #12 | Optimistic ambiguity fallback — generate SQL even when flagged ambiguous |
| `FEAT_13` | `true`  | #13 | Few-shot examples in ambiguity prompt |
| `FEAT_14` | `true`  | #14 | Strict SQL rules in generation prompt |
| `FEAT_15` | `true`  | #15 | `sanitize_sql` clean-up (markdown, DISTINCT, CAST fixes) |
| `FEAT_16` | `true`  | #16 | sqlglot validate + forbid DROP/DELETE/UPDATE |
| `FEAT_17` | `true`  | #17 | Retry loop up to `MAX_RETRIES` |
| `FEAT_18` | `true`  | #18 | Exponential backoff on 429 rate-limit |
| `FEAT_19` | `true`  | #19 | `asyncio.sleep` throttle between benchmark questions |
| `FEAT_20` | `true`  | #20 | LLM-as-judge verify in retry loop |
| `FEAT_27` | `true`  | #27 | True self-correction — pass error text into retry prompt |
| `FEAT_29` | `true`  | #29 | Complexity-based prompt selection (simple/complex) |
| `FEAT_32` | `false` | #32 | AmbiSQL Stage 1: taxonomy detection (A1-A6) |
| `FEAT_33` | `false` | #33 | AmbiSQL Stage 2a: clarification questions (requires FEAT_32) |
| `FEAT_34` | `false` | #34 | AmbiSQL Stage 2b: query rewriting / auto-resolve (requires FEAT_33) |
| `FEAT_35` | `true`  | #35 | TSV schema format (30–40% fewer tokens vs JSON) |

**Critical**: flags default to `"true"` unless noted. Any flag absent from the environment is treated as enabled. To run ablation, all flags must be explicitly set. Exceptions: FEAT_32/33/34 default `false` (AmbiSQL pipeline is experimental).

### Running isolated ablation (leave-one-in)

On Windows/Git Bash, shell `export` does not reliably cross the MSYS2→uv.exe boundary. Use a temp env file:

```bash
# One feature at a time:
bash local/run_single.sh FEAT_5 ambrosia   # ambrosia | bird | both

# Full 14-feature ablation (BIRD + Ambrosia per feature, ~6 hours):
bash local/ablation_full.sh
```

Results are saved to `ablation_results/<feat_name>/` (bird.log, ambrosia.log, summary.txt).

### Additional env vars

- `MAX_RETRIES` — SQL generation retries (default: 7)
- `LLM_MIN_INTERVAL` — seconds between questions (default: 3; ablation scripts use 12)

## Baseline Accuracy (reference)
- BIRD: ~20–35% with gpt-3.5-turbo
- Ambrosia: ~50–65% ambiguity detection accuracy
