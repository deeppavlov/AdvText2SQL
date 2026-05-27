import json as json_module
import logging
import os
import pathlib
import re
import time
from collections import Counter
from datetime import datetime, timezone
from textwrap import dedent
from typing import Any

from autogen_core.models import SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, Result
from sqlglot import Dialects, exp, parse_one
from sqlglot.errors import ParseError

from .prompts import (
    AMBIGUITY_PROMPT_TEMPLATE,
    AMBIGUITY_PROMPT_SIMPLE,
    SQL_PROMPT_TEMPLATE,
    SQL_PROMPT_BASIC,
    SQL_PROMPT_COMPLEX_TEMPLATE,
    SQL_PROMPT_SIMPLE_TEMPLATE,
    SQL_RETRY_PROMPT_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE,
    VERIFICATION_PROMPT_TEMPLATE,
    TAXONOMY_DETECTION_PROMPT,
    CLARIFICATION_PROMPT,
    AUTO_RESOLVE_PROMPT,
    SCHEMA_PRUNING_PROMPT,
    TABLE_DESCRIPTION_PROMPT,
    DECOMPOSITION_PROMPT,
)
from .utils import print_result

# Load environment variables for the main block
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 7))

# Ablation feature flags — FEAT_N maps 1:1 to Practice #N in practices_doc.md
# Default true = full feature set when not running ablation
FEAT_1  = os.getenv("FEAT_1",  "true").lower()  == "true"  # #1  FK/PK relationships in prompt
FEAT_2  = os.getenv("FEAT_2",  "true").lower()  == "true"  # #2  column statistics (min/max/top-K)
FEAT_3  = os.getenv("FEAT_3",  "true").lower()  == "true"  # #3  regex type detection for TEXT cols
FEAT_4  = os.getenv("FEAT_4",  "true").lower()  == "true"  # #4  pg_stat row count (vs fallback COUNT)
FEAT_5  = os.getenv("FEAT_5",  "true").lower()  == "true"  # #5  light schema (names + types)
FEAT_6  = os.getenv("FEAT_6",  "false").lower() == "true"  # #6  heavy schema (samples + enums; off by default)
FEAT_7  = os.getenv("FEAT_7",  "true").lower()  == "true"  # #7  compact stats formatting
FEAT_8  = os.getenv("FEAT_8",  "true").lower()  == "true"  # #8  PG rollback guards on failed transactions
FEAT_9  = os.getenv("FEAT_9",  "true").lower()  == "true"  # #9  dump db_schemas.json after build
FEAT_10 = os.getenv("FEAT_10", "true").lower()  == "true"  # #10 build() timing + prompt size log
FEAT_11 = os.getenv("FEAT_11", "true").lower()  == "true"  # #11 JSON structured logging to file
FEAT_12 = os.getenv("FEAT_12", "true").lower()  == "true"  # #12 optimistic ambiguity fallback
FEAT_13 = os.getenv("FEAT_13", "true").lower()  == "true"  # #13 few-shot ambiguity prompt with rules
FEAT_14 = os.getenv("FEAT_14", "true").lower()  == "true"  # #14 strict SQL rules in generation prompt
FEAT_15 = os.getenv("FEAT_15", "true").lower()  == "true"  # #15 sanitize_sql clean-up
FEAT_16 = os.getenv("FEAT_16", "true").lower()  == "true"  # #16 sqlglot validate + forbid mutations
FEAT_17 = os.getenv("FEAT_17", "true").lower()  == "true"  # #17 retry loop up to MAX_RETRIES
FEAT_18 = os.getenv("FEAT_18", "true").lower()  == "true"  # #18 LLM exponential backoff on 429
FEAT_19 = os.getenv("FEAT_19", "true").lower()  == "true"  # #19 asyncio.sleep throttle between questions
FEAT_20 = os.getenv("FEAT_20", "true").lower()  == "true"  # #20 LLM-as-judge verify in retry loop
FEAT_27 = os.getenv("FEAT_27", "true").lower()  == "true"  # #27 true self-correction (error feedback in retry)
FEAT_29 = os.getenv("FEAT_29", "true").lower()  == "true"  # #29 complexity-based prompt selection
FEAT_32 = os.getenv("FEAT_32", "false").lower() == "true"  # #32 AmbiSQL Stage 1: taxonomy detection (A1-A6)
FEAT_33 = os.getenv("FEAT_33", "false").lower() == "true"  # #33 AmbiSQL Stage 2a: clarification questions
FEAT_34 = os.getenv("FEAT_34", "false").lower() == "true"  # #34 AmbiSQL Stage 2b: query rewriting
FEAT_35 = os.getenv("FEAT_35", "true").lower()  == "true"  # #35 TSV schema format (token-efficient)
FEAT_36 = os.getenv("FEAT_36", "true").lower()  == "true"  # #36 smart LIMIT 101 injection (N+1 pagination)
FEAT_25 = os.getenv("FEAT_25", "false").lower() == "true"  # #25 per-question schema pruning (1 extra LLM call)
FEAT_26 = os.getenv("FEAT_26", "false").lower() == "true"  # #26 learnt_hints — persist successful patterns
FEAT_28 = os.getenv("FEAT_28", "false").lower() == "true"  # #28 self-consistency — 3 samples + majority vote
FEAT_21 = os.getenv("FEAT_21", "false").lower() == "true"  # #21 MinHash column similarity — implicit join hints
FEAT_23 = os.getenv("FEAT_23", "false").lower() == "true"  # #23 NL table descriptions via LLM (cached)
FEAT_30 = os.getenv("FEAT_30", "false").lower() == "true"  # #30 CoT sub-question decomposition


logger = logging.getLogger("text2sql_tool")


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured log output to file."""
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json_module.dumps(log_entry, ensure_ascii=False)


def setup_logging(log_file: str = "text2sql_debug.log"):
    """Configure logging: console (INFO) + JSON file (DEBUG)."""
    root_logger = logging.getLogger("text2sql_tool")
    if root_logger.handlers:
        return  # Already configured
    root_logger.setLevel(logging.DEBUG)

    # Console handler — compact INFO
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    root_logger.addHandler(console)

    # File handler — JSON DEBUG (FEAT_11)
    if FEAT_11:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)


class Text2SQLGenerator:
    def __init__(self, db_uri: str, llm_client: OpenAIChatCompletionClient):
        """
        Initializes the Text2SQL generator with a database URI.

        Args:
            db_uri (str): SQL database URI.
            llm_client (openai.AsyncOpenAI): An initialized asynchronous OpenAI client.
        """
        self.db_uri = db_uri
        self.engine: Engine = create_engine(
            db_uri,
            pool_pre_ping=True,
            pool_size=3,
            max_overflow=2,
            pool_timeout=30,
        )

        self.llm_client = llm_client

        logger.info("Initialized Text2SQLGenerator")

    async def build(self):
        start = time.time()
        self._db_id = self.db_uri.rsplit("/", 1)[-1]
        logger.info(f"Build started for {self.db_uri}")
        logger.info(
            f"Feature flags: "
            f"FEAT_1={FEAT_1} FEAT_2={FEAT_2} FEAT_3={FEAT_3} FEAT_4={FEAT_4} "
            f"FEAT_5={FEAT_5} FEAT_6={FEAT_6} FEAT_7={FEAT_7} FEAT_8={FEAT_8} "
            f"FEAT_9={FEAT_9} FEAT_10={FEAT_10} FEAT_11={FEAT_11} FEAT_12={FEAT_12} "
            f"FEAT_13={FEAT_13} FEAT_14={FEAT_14} FEAT_15={FEAT_15} FEAT_16={FEAT_16} "
            f"FEAT_17={FEAT_17} FEAT_18={FEAT_18} FEAT_19={FEAT_19} FEAT_20={FEAT_20} "
            f"FEAT_27={FEAT_27} FEAT_29={FEAT_29} "
            f"FEAT_32={FEAT_32} FEAT_33={FEAT_33} FEAT_34={FEAT_34} FEAT_35={FEAT_35} "
            f"FEAT_36={FEAT_36} FEAT_25={FEAT_25} FEAT_26={FEAT_26} FEAT_28={FEAT_28} "
            f"FEAT_21={FEAT_21} FEAT_23={FEAT_23} FEAT_30={FEAT_30}"
        )

        if FEAT_35:
            self.db_schema = self._get_db_schema_tsv()
        elif FEAT_6:
            self.db_schema = self._get_db_schema_heavy()
        else:
            self.db_schema = self._get_db_schema_light()  # FEAT_5 or baseline fallback

        if FEAT_1:
            self.db_relationships = self._explore_db_relationships()
            self.relationships_str = self._format_relationships()
        else:
            self.db_relationships = {"foreign_keys": [], "primary_keys": {}}
            self.relationships_str = ""

        if FEAT_2:
            self.column_stats = self._explore_column_statistics(
                use_regex=FEAT_3, use_pg_stat=FEAT_4
            )
            self.column_stats_str = (
                self._format_column_statistics(self.column_stats) if FEAT_7
                else self._format_column_statistics_verbose(self.column_stats)
            )
        else:
            self.column_stats = {}
            self.column_stats_str = ""

        self.system_prompt = self._create_system_prompt()

        # FEAT_21: MinHash column similarity — append implicit join hints to system prompt
        if FEAT_21:
            similar_cols = self._find_similar_columns()
            if similar_cols:
                self.system_prompt += f"\n\n### Возможные неявные связи колонок (MinHash):\n{similar_cols}"
                logger.info(f"FEAT_21: added MinHash column hints")

        # FEAT_23: NL table descriptions — append to system prompt (cached per db_id)
        if FEAT_23:
            descs = await self._generate_table_descriptions()
            if descs:
                desc_str = "\n".join(f"- {t}: {d}" for t, d in descs.items())
                self.system_prompt += f"\n\n### Описания таблиц:\n{desc_str}"
                logger.info(f"FEAT_23: added NL descriptions for {len(descs)} tables")

        # FEAT_26: append learnt hints from previous runs to system prompt
        if FEAT_26:
            hints = self._load_learnt_hints()
            if hints:
                hints_text = "\n".join(
                    f"- Q: {h['question']}\n  SQL: {h['sql']}" for h in hints[-10:]
                )
                self.system_prompt += f"\n\n## Learnt hints from previous runs\n{hints_text}"
                logger.info(f"FEAT_26: loaded {len(hints)} learnt hints for {self._db_id}")

        if FEAT_10:
            elapsed = time.time() - start
            logger.info(
                f"Build completed in {elapsed:.1f}s — "
                f"stats for {len(self.column_stats)} tables, "
                f"system prompt {len(self.system_prompt)} chars"
            )

    async def _update_db_schema(self, db_uri):
        self.db_uri = db_uri
        self.engine.dispose()
        self.engine = create_engine(db_uri, pool_pre_ping=True)
        await self.build()

    def _get_db_schema_heavy(
        self,
        sample_rows_limit: int = 3,
        get_unique_values: bool = True,
        unique_threshold: int = 10,
        sample_tables_with_more_rows: dict | None = None,
    ) -> str:
        """
        Extract db_schema with a few more features (they all spend tokens).

        Get PostgreSQL schema info with:
        - table definitions (columns + types)
        - sample rows
        - unique values for low-cardinality columns (optional)
        """

        # print(self.db_uri)
        if sample_tables_with_more_rows is None:
            sample_tables_with_more_rows = {}

        inspector = inspect(self.engine)
        schema_parts: list[str] = []
        unique_values_parts: list[str] = []

        with self.engine.connect() as conn:
            conn.execute(text("SET LOCAL statement_timeout = '10s'"))
            tables = inspector.get_table_names()

            for table in tables:
                schema_parts.append(f"-- Table: {table}")

                # 1. Columns
                columns = inspector.get_columns(table)
                schema_parts.append(f"CREATE TABLE {table} (")
                for col in columns:
                    schema_parts.append(f"    {col['name']} {col['type']},")
                schema_parts.append(");")

                # 2. Sample rows
                limit = sample_tables_with_more_rows.get(table, sample_rows_limit)
                try:
                    result = conn.execute(
                        text(f'SELECT * FROM "{table}" LIMIT :limit'),
                        {"limit": limit},
                    )
                    rows = result.fetchall()

                    if rows:
                        col_names = result.keys()
                        schema_parts.append("/*")
                        schema_parts.append("\t".join(col_names))
                        for row in rows:
                            schema_parts.append("\t".join(map(str, row)))
                        schema_parts.append("*/")

                except Exception:
                    logger.exception(f"Could not fetch sample rows for table {table}")

                schema_parts.append("\n")

                # 3. Unique values per column (low-cardinality only)
                if not get_unique_values:
                    continue

                for col in columns:
                    col_name = col["name"]

                    try:
                        res = conn.execute(
                            text(
                                f'''
                                SELECT DISTINCT "{col_name}"
                                FROM "{table}"
                                WHERE "{col_name}" IS NOT NULL
                                '''
                            )
                        )
                        values = [r[0] for r in res.fetchall()]

                        if 0 < len(values) <= unique_threshold:
                            formatted = ", ".join(repr(v) for v in values)
                            unique_values_parts.append(
                                f'Possible values for "{table}.{col_name}": [{formatted}]'
                            )

                    except Exception:
                        # silently skip problematic columns (JSON, arrays, etc.)
                        continue

        if unique_values_parts:
            schema_parts.append("\n### Low-cardinality value hints")
            schema_parts.extend(unique_values_parts)

        return "\n".join(schema_parts).strip()

    def _get_db_schema_light(self) -> str:
        """
        Lightweight schema extraction for LLM prompts.

        Includes:
        - table names
        - column names
        - column types
        """
        inspector = inspect(self.engine)
        schema_parts = []

        # Default schema for PostgreSQL
        # print(self.db_uri)
        tables = inspector.get_table_names(schema="public")
        if not tables:
            print("this db failed: ", self.db_uri)
            raise RuntimeError("No tables found in public schema")

        for table in tables:
            schema_parts.append(f"TABLE {table}")

            columns = inspector.get_columns(table, schema="public")
            for col in columns:
                col_type = str(col["type"])
                schema_parts.append(f"  - {col['name']} ({col_type})")

            schema_parts.append("")

        return "\n".join(schema_parts).strip()

    def _get_db_schema_tsv(self) -> str:
        """FEAT_35: TSV schema — table/column/type, one row per column (~30% fewer tokens than light schema)."""
        inspector = inspect(self.engine)
        tables = inspector.get_table_names(schema="public")
        if not tables:
            import sys as _sys
            print(f"[ERROR] No tables in public schema for: {self.db_uri}", file=_sys.stderr, flush=True)
            raise RuntimeError(f"No tables found in public schema for {self.db_uri}")

        rows = ["table\tcolumn\ttype"]
        for table in tables:
            columns = inspector.get_columns(table, schema="public")
            for col in columns:
                rows.append(f"{table}\t{col['name']}\t{str(col['type'])}")

        return "\n".join(rows)

    def _create_system_prompt(self) -> str:
        """Создает системный промпт с описанием схемы БД"""
        return SYSTEM_PROMPT_TEMPLATE.format(
            db_schema=self.db_schema,
            db_relationships=self.relationships_str,
            column_statistics=self.column_stats_str,
            sql_dialect="PostgreSQL",
        )

    def _explore_db_relationships(self) -> dict:
        """Извлекает FK и PK через SQLAlchemy inspector."""
        inspector = inspect(self.engine)
        tables = inspector.get_table_names(schema="public")

        foreign_keys = []
        primary_keys = {}

        for table in tables:
            try:
                pk = inspector.get_pk_constraint(table, schema="public")
                if pk and pk.get("constrained_columns"):
                    primary_keys[table] = pk["constrained_columns"]
            except Exception:
                logger.exception(f"Could not get PK for table {table}")

            try:
                for fk in inspector.get_foreign_keys(table, schema="public"):
                    foreign_keys.append({
                        "from_table": table,
                        "from_cols": fk["constrained_columns"],
                        "to_table": fk["referred_table"],
                        "to_cols": fk["referred_columns"],
                    })
            except Exception:
                logger.exception(f"Could not get FK for table {table}")

        logger.info(f"Relationships: {len(primary_keys)} PKs, {len(foreign_keys)} FKs")
        return {"foreign_keys": foreign_keys, "primary_keys": primary_keys}

    def _format_relationships(self) -> str:
        """Форматирует FK/PK в DDL-стиль для промпта."""
        parts = []

        pks = self.db_relationships.get("primary_keys", {})
        if pks:
            parts.append("-- Primary Keys:")
            for table, cols in pks.items():
                parts.append(f"-- {table}: {', '.join(cols)}")
            parts.append("")

        fks = self.db_relationships.get("foreign_keys", [])
        if fks:
            parts.append("-- Foreign Keys:")
            for fk in fks:
                from_str = ", ".join(
                    f'{fk["from_table"]}.{c}' for c in fk["from_cols"]
                )
                to_str = ", ".join(
                    f'{fk["to_table"]}.{c}' for c in fk["to_cols"]
                )
                parts.append(f"-- {from_str} -> {to_str}")

        return "\n".join(parts) if parts else "-- No relationships found"

    def _get_table_row_count(self, conn, table: str) -> int:
        """Получает количество строк: сначала из статистики, fallback на COUNT(*)."""
        try:
            res = conn.execute(
                text(
                    "SELECT n_live_tup FROM pg_stat_user_tables "
                    "WHERE relname = :table_name"
                ),
                {"table_name": table},
            )
            row = res.fetchone()
            if row and row[0] > 0:
                return int(row[0])
        except Exception:
            if FEAT_8:
                try:
                    conn.rollback()
                except Exception:
                    pass

        try:
            res = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"'))
            return int(res.scalar())
        except Exception:
            logger.exception(f"COUNT(*) failed for table {table}")
            if FEAT_8:
                try:
                    conn.rollback()
                except Exception:
                    pass
            return 0

    def _detect_column_types_for_table(
        self, conn, table: str, columns: list[dict], sample_size: int = 500
    ) -> dict[str, dict]:
        """
        Определяет реальный тип данных для каждой колонки через sampling.
        Возвращает {col_name: {detected_type, distinct_count, nonnull_count}}.
        """
        if not columns:
            return {}

        select_parts = []
        for col in columns:
            name = col["name"]
            safe = name.replace('"', '""')
            select_parts.extend([
                f'COUNT(*) FILTER (WHERE "{safe}" IS NOT NULL) AS "{safe}__nonnull"',
                f"COUNT(*) FILTER (WHERE \"{safe}\"::text ~ '^-?[0-9]+$') AS \"{safe}__integer\"",
                f"COUNT(*) FILTER (WHERE \"{safe}\"::text ~ '^-?[0-9]+(\\.[0-9]+)?$') AS \"{safe}__numeric\"",
                f"COUNT(*) FILTER (WHERE \"{safe}\"::text ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}') AS \"{safe}__date\"",
                f"COUNT(*) FILTER (WHERE \"{safe}\"::text ~ '^[\\[{{]') AS \"{safe}__json\"",
                f'COUNT(DISTINCT "{safe}") AS "{safe}__distinct"',
            ])

        query = (
            f"SELECT {', '.join(select_parts)} "
            f'FROM (SELECT * FROM "{table}" ORDER BY RANDOM() LIMIT {sample_size}) s'
        )

        try:
            result = conn.execute(text(query))
            row = result.fetchone()
        except Exception:
            logger.exception(f"Column type detection failed for table {table}")
            if FEAT_8:
                try:
                    conn.rollback()
                except Exception:
                    pass
            return {}

        if not row:
            return {}

        row_dict = dict(row._mapping)
        detected = {}

        for col in columns:
            name = col["name"]
            nonnull = row_dict.get(f"{name}__nonnull", 0)
            distinct = row_dict.get(f"{name}__distinct", 0)

            if nonnull == 0:
                detected[name] = {
                    "detected_type": "empty",
                    "distinct_count": 0,
                    "nonnull_count": 0,
                }
                continue

            int_ct = row_dict.get(f"{name}__integer", 0)
            num_ct = row_dict.get(f"{name}__numeric", 0)
            date_ct = row_dict.get(f"{name}__date", 0)
            json_ct = row_dict.get(f"{name}__json", 0)

            if int_ct / nonnull >= 0.9:
                dtype = "integer"
            elif num_ct / nonnull >= 0.9:
                dtype = "numeric"
            elif date_ct / nonnull >= 0.9:
                dtype = "date"
            elif json_ct / nonnull >= 0.5:
                dtype = "json"
            else:
                dtype = "categorical"

            detected[name] = {
                "detected_type": dtype,
                "distinct_count": distinct,
                "nonnull_count": nonnull,
            }

        return detected

    def _compute_column_stats(
        self, conn, table: str, column: str, detected_type: str, row_count: int
    ) -> dict | None:
        """Вычисляет статистики для колонки в зависимости от типа."""
        safe = column.replace('"', '""')

        try:
            if detected_type in ("integer", "numeric"):
                if row_count > 50000:
                    res = conn.execute(text(
                        f'SELECT MIN("{safe}"::numeric), MAX("{safe}"::numeric), '
                        f'ROUND(AVG("{safe}"::numeric), 2) '
                        f'FROM "{table}" '
                        f"WHERE \"{safe}\" IS NOT NULL AND \"{safe}\"::text ~ '^-?[0-9]+(\\.[0-9]+)?$'"
                    ))
                    r = res.fetchone()
                    if r:
                        return {
                            "type": detected_type,
                            "min": r[0], "max": r[1], "mean": r[2],
                        }
                else:
                    res = conn.execute(text(
                        f'SELECT MIN("{safe}"::numeric), MAX("{safe}"::numeric), '
                        f'ROUND(AVG("{safe}"::numeric), 2), '
                        f'PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "{safe}"::numeric) '
                        f'FROM "{table}" '
                        f"WHERE \"{safe}\" IS NOT NULL AND \"{safe}\"::text ~ '^-?[0-9]+(\\.[0-9]+)?$'"
                    ))
                    r = res.fetchone()
                    if r:
                        return {
                            "type": detected_type,
                            "min": r[0], "max": r[1], "mean": r[2], "median": r[3],
                        }

            elif detected_type == "date":
                res = conn.execute(text(
                    f'SELECT MIN("{safe}"::date), MAX("{safe}"::date) '
                    f'FROM "{table}" '
                    f"WHERE \"{safe}\" IS NOT NULL AND \"{safe}\"::text ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}'"
                ))
                r = res.fetchone()
                if r:
                    return {"type": "date", "min": str(r[0]), "max": str(r[1])}

            elif detected_type == "categorical":
                res = conn.execute(text(
                    f'SELECT "{safe}", COUNT(*) AS freq '
                    f'FROM "{table}" WHERE "{safe}" IS NOT NULL '
                    f'GROUP BY "{safe}" ORDER BY freq DESC LIMIT 5'
                ))
                rows = res.fetchall()
                if rows:
                    return {
                        "type": "categorical",
                        "top": [(str(r[0]), int(r[1])) for r in rows],
                    }

            elif detected_type == "json":
                return {"type": "json"}

        except Exception:
            logger.exception(
                f"Stats computation failed for {table}.{column}"
            )
            if FEAT_8:
                try:
                    conn.rollback()
                except Exception:
                    pass

        return None

    def _explore_column_statistics(
        self, use_regex: bool = True, use_pg_stat: bool = True
    ) -> dict:
        """
        Оркестратор: собирает row counts, типы и статистики для всех таблиц.

        Args:
            use_regex: FEAT_3 — use regex sampling to detect real column types.
                       False → treat all TEXT columns as categorical.
            use_pg_stat: FEAT_4 — use pg_stat_user_tables for row count.
                         False → skip row count (set to 0).
        """
        inspector = inspect(self.engine)
        tables = inspector.get_table_names(schema="public")
        result = {}

        with self.engine.connect() as conn:
            for table in tables:
                try:
                    row_count = self._get_table_row_count(conn, table) if use_pg_stat else 0
                    columns = inspector.get_columns(table, schema="public")
                    if use_regex:
                        col_types = self._detect_column_types_for_table(conn, table, columns)
                    else:
                        col_types = {
                            col["name"]: {
                                "detected_type": "categorical",
                                "distinct_count": 0,
                                "nonnull_count": 1,
                            }
                            for col in columns
                        }

                    col_stats = {}
                    for col in columns:
                        name = col["name"]
                        info = col_types.get(name)
                        if not info or info["detected_type"] == "empty":
                            continue

                        stats = self._compute_column_stats(
                            conn, table, name,
                            info["detected_type"], row_count,
                        )
                        col_stats[name] = {
                            "detected_type": info["detected_type"],
                            "distinct_count": info["distinct_count"],
                            "stats": stats,
                        }

                    result[table] = {
                        "row_count": row_count,
                        "columns": col_stats,
                    }
                    logger.info(
                        f"Stats collected for table '{table}': "
                        f"{row_count} rows, {len(col_stats)} columns"
                    )
                except Exception:
                    logger.exception(
                        f"Column statistics exploration failed for table {table}"
                    )
                    if FEAT_8:
                        try:
                            conn.rollback()
                        except Exception:
                            pass

        logger.info(f"Column statistics: {len(result)}/{len(tables)} tables processed")
        return result

    def _format_column_statistics(self, stats: dict) -> str:
        """Форматирует статистики компактно для включения в промпт."""
        parts = []

        for table, tinfo in stats.items():
            row_count = tinfo["row_count"]
            parts.append(f"TABLE {table} ({row_count:,} rows)")

            for col, cinfo in tinfo["columns"].items():
                dtype = cinfo["detected_type"]
                distinct = cinfo["distinct_count"]
                s = cinfo.get("stats")

                # ID-колонка: distinct == row_count
                if distinct == row_count and row_count > 1:
                    parts.append(f"  - {col}: {dtype}, unique (id)")
                    continue

                if not s:
                    parts.append(f"  - {col}: {dtype}")
                    continue

                if s["type"] in ("integer", "numeric"):
                    line = f"  - {col}: {s['type']}, range [{s['min']}, {s['max']}], mean={s['mean']}"
                    if "median" in s:
                        line += f", median={s['median']}"
                    parts.append(line)

                elif s["type"] == "date":
                    parts.append(
                        f"  - {col}: date, range [{s['min']}, {s['max']}]"
                    )

                elif s["type"] == "categorical":
                    top = s.get("top", [])
                    if distinct <= 10:
                        vals = ", ".join(f'"{v}"' for v, _ in top)
                        parts.append(
                            f"  - {col}: categorical, {distinct} unique, values: {vals}"
                        )
                    else:
                        top_str = ", ".join(
                            f'"{v}"({cnt})' for v, cnt in top
                        )
                        parts.append(
                            f"  - {col}: categorical, {distinct} unique, top: {top_str}"
                        )

                elif s["type"] == "json":
                    parts.append(f"  - {col}: json/list")

            parts.append("")

        return "\n".join(parts).strip() if parts else "-- No statistics available"

    def _format_column_statistics_verbose(self, stats: dict) -> str:
        """Форматирует статистики без компактных оптимизаций (FEAT_7=false)."""
        parts = []
        for table, tinfo in stats.items():
            row_count = tinfo["row_count"]
            parts.append(f"TABLE {table} ({row_count:,} rows)")
            for col, cinfo in tinfo["columns"].items():
                dtype = cinfo["detected_type"]
                distinct = cinfo["distinct_count"]
                s = cinfo.get("stats") or {}
                line = f"  - {col}: {dtype}, {distinct} distinct values"
                details = [f"{k}={v}" for k, v in s.items() if k != "type"]
                if details:
                    line += " | " + ", ".join(details)
                parts.append(line)
            parts.append("")
        return "\n".join(parts).strip() if parts else "-- No statistics available"

    async def _llm_call_with_retry(
        self, messages, max_retries: int = 6, base_delay: float = 5.0, max_delay: float = 120.0
    ):
        """Обёртка LLM вызовов с retry и exponential backoff при 429. FEAT_18 gates backoff."""
        import asyncio as _asyncio

        effective_retries = max_retries if FEAT_18 else 2
        for attempt in range(effective_retries):
            if attempt > 0:
                await _asyncio.sleep(1.0)  # min 1s between calls to stay under 1 RPS
            try:
                return await self.llm_client.create(messages)
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in str(e) or "rate" in err_str
                if is_rate_limit:
                    delay = min(base_delay * (2 ** attempt), max_delay) if FEAT_18 else 30.0
                    logger.warning(
                        f"Rate limited, retrying in {delay:.0f}s "
                        f"(attempt {attempt + 1}/{effective_retries})"
                    )
                    await _asyncio.sleep(delay)
                else:
                    raise
        # Last attempt — let exception propagate
        await _asyncio.sleep(1.0)
        return await self.llm_client.create(messages)

    def _detect_structural_ambiguity(self, user_query: str) -> bool:
        """
        Python-детектор двух синтаксических паттернов неоднозначности (без LLM).
        AND-scope: "[Тип A] and [Тип B] with/who/that/where [условие]"
        Every/each: "systems every specialist works with" (не "for each X")
        TP=6/9, FP=0 на Ambrosia. FP=0 на BIRD.
        """
        q = user_query.lower()
        # AND-scope: letter-only words (exclude year/number ranges)
        if re.search(
            r'\b[a-z]+(?:\s+[a-z]+)?\s+and\s+[a-z]+(?:\s+[a-z]+)?\s+'
            r'(?:with|who|which|that|where|serving|associated\s+with)(?!\s+either)',
            q,
        ):
            return True
        # every/each as universal quantifier — strip "for each/every" first
        q_no_for_each = re.sub(r'\bfor\s+(?:every|each)\b', '', q)
        if re.search(r'\b(?:every|each)\b', q_no_for_each):
            return True
        return False

    async def _check_ambiguity(self, user_query: str) -> dict[str, Any]:
        """
        Проверяет пользовательский запрос на неоднозначность с помощью LLM.
        """
        template = AMBIGUITY_PROMPT_TEMPLATE if FEAT_13 else AMBIGUITY_PROMPT_SIMPLE
        ambiguity_prompt = template.format(
            db_schema=self.db_schema,
            db_relationships=self.relationships_str,
            column_statistics=self.column_stats_str,
            user_query=user_query,
        )

        messages = [
            SystemMessage(content=ambiguity_prompt),
            UserMessage(
                source="user",
                content="Проверь запрос на однозначность.",
            ),
        ]

        """
        content=(
            "Определи, можно ли вообще ответить на этот вопрос, "
            "используя ТОЛЬКО данную схему БД. "
            "Если ответить нельзя — напиши NOT_ANSWERABLE. "
            "Если ответ возможен, но запрос неоднозначен — опиши, что нужно уточнить. "
            "Если запрос полностью однозначен и ответим — напиши OK."
        ),
        """

        try:
            result = await self._llm_call_with_retry(messages)
            response_text = result.content.strip()

            if response_text.strip().split('\n')[0].strip().strip('*').rstrip('.!?,').lower() == "ok":
                logger.info(f"Ambiguity check: OK")
                return {"status": "success", "ambiguous": False}
            else:
                logger.info(f"Ambiguity check: ambiguous — {response_text[:100]}")
                return {
                    "status": "success",
                    "ambiguous": True,
                    "clarification_needed": response_text,
                }

        except Exception as e:
            logger.exception(f"Failed to check ambiguity for query: {user_query}")
            return {"status": "error"}

    @staticmethod
    def _taxonomy_stop_filter(query: str) -> bool:
        """Returns True if query matches a structural non-ambiguous pattern → skip taxonomy LLM call."""
        q = query.lower()
        return bool(
            # Dual-list patterns: "X and also list/provide/show Y"
            re.search(r"\band also\s+(list|provide|show|give|display)\b", q)
            # Parallel sub-questions: "and what/which/who X verb"
            or re.search(r"\band\s+(what|which|who)\s+\w", q)
            # "both X and Y [clause]"
            or re.search(r"\b(give me both|show both|list both|both\s+\w+\s+and\s+\w+\s+(who|which|that|where|with))\b", q)
            # Explicit disjunction: "where either", "either of which"
            or re.search(r"\b(where either|either of which)\b", q)
            # "as well as"
            or re.search(r"\bas well as\b", q)
            # Explicit iteration at sentence start: "For each/every X, verb Y" — unambiguous grouping
            # NOTE: "Y for each/every X" (each at end) is Pattern 2 ambiguous — NOT blocked
            or re.search(r"^\s*for (each|every)\s+\w", q)
            or re.search(r"^\s*for all\s+\w", q)
            # Intersection/aggregation: "in common", "shared between/among", "common to/for all/every"
            or re.search(r"\bin common\b", q)
            or re.search(r"\bshared (between|among)\b", q)
            or re.search(r"\bcommon (to|for) (all|every)\b", q)
            or re.search(r"\bsame (for|by|among) all\b", q)
        )

    async def _detect_ambiguity_by_taxonomy(self, user_query: str) -> dict[str, Any]:
        """Stage 1: классифицирует запрос по таксономии AmbiSQL (A1-A6). FEAT_32."""
        if self._taxonomy_stop_filter(user_query):
            logger.info("Taxonomy stop-filter matched: is_ambiguous=False (no LLM call)")
            return {"is_ambiguous": False, "ambiguity_type": None, "reason": "stop-filter"}

        prompt = TAXONOMY_DETECTION_PROMPT.format(
            db_schema=self.db_schema,
            user_query=user_query,
        )
        messages = [
            SystemMessage(content=prompt),
            UserMessage(source="user", content="Classify the query."),
        ]
        try:
            result = await self._llm_call_with_retry(messages)
            text = result.content.strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if 0 <= start < end:
                text = text[start:end]
            try:
                data = json_module.loads(text)
            except json_module.JSONDecodeError:
                # Replace all control chars (incl. newlines) with space —
                # safe because space is valid JSON whitespace between tokens
                # and harmless inside string values.
                sanitized = re.sub(r"[\x00-\x1f]", " ", text)
                data = json_module.loads(sanitized)
            logger.info(f"Taxonomy detection: is_ambiguous={data.get('is_ambiguous')} type={data.get('ambiguity_type')}")
            return data
        except Exception:
            logger.exception(f"Taxonomy detection failed for: {user_query}")
            return {"is_ambiguous": False, "ambiguity_type": None, "reason": ""}

    async def _generate_clarification_question(
        self, user_query: str, ambiguity_type: str, reason: str
    ) -> dict[str, Any]:
        """Stage 2a: генерирует multiple-choice вопрос для уточнения. FEAT_33."""
        prompt = CLARIFICATION_PROMPT.format(
            db_schema=self.db_schema,
            user_query=user_query,
            ambiguity_type=ambiguity_type,
            reason=reason,
        )
        messages = [
            SystemMessage(content=prompt),
            UserMessage(source="user", content="Generate clarification question."),
        ]
        try:
            result = await self._llm_call_with_retry(messages)
            text = result.content.strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
            data = json_module.loads(text)
            logger.info(f"Clarification question: {data.get('question', '')[:80]}")
            return data
        except Exception:
            logger.exception(f"Clarification generation failed for: {user_query}")
            return {"question": "", "options": []}

    async def _auto_resolve_ambiguity(
        self, user_query: str, question: str, options: list
    ) -> str:
        """Stage 2b: LLM выбирает наиболее вероятную опцию и переписывает запрос. FEAT_34."""
        prompt = AUTO_RESOLVE_PROMPT.format(
            user_query=user_query,
            clarification_question=question,
            options="\n".join(f"- {o}" for o in options),
            db_schema=self.db_schema,
        )
        messages = [
            SystemMessage(content=prompt),
            UserMessage(source="user", content="Choose and rewrite."),
        ]
        try:
            result = await self._llm_call_with_retry(messages)
            text = result.content.strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
            data = json_module.loads(text)
            rewritten = data.get("rewritten_query", user_query)
            logger.info(f"Auto-resolve: chose '{data.get('chosen_option', '')}' → '{rewritten[:80]}'")
            return rewritten
        except Exception:
            logger.exception(f"Auto-resolve failed for: {user_query}")
            return user_query

    def _strip_sql_comments(self, sql: str) -> str:
        sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
        sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql.strip(), flags=re.IGNORECASE)

        return sql.strip()

    def _is_sql_complete(self, sql: str) -> bool:
        upper = sql.upper().strip()

        required = ["SELECT", "FROM"]
        if not all(k in upper for k in required):
            return False

        if upper.endswith(
            (
                "WHERE",
                "JOIN",
                "FROM",
                "ON",
                "AND",
                "OR",
            )
        ):
            return False

        return True

    def _validate_sql(self, sql: str) -> bool:
        """Валидация SQL запроса с помощью SQLGlot. FEAT_16 gates this check."""
        if not FEAT_16:
            return True
        try:
            if not self._is_sql_complete(sql):
                raise ValueError("Incomplete SQL generated")

            parsed = parse_one(sql, dialect=Dialects.POSTGRES)

            # Проверка на запрещенные операции
            for node in parsed.walk():
                if isinstance(node, (exp.Drop | exp.Delete | exp.Update)):
                    raise ValueError(f"Запрещенная операция: {node.sql()}")

            return True
        except (ParseError, ValueError):
            logger.exception("SQL validation error.")
            return False

    def sanitize_sql(self, sql: str) -> str:
        """
        Minimal SQL sanitizer for Postgres execution.
        Fixes common LLM-generated SQL issues:
        1. Strips markdown/code fences
        2. Removes misplaced DISTINCT after comma in SELECT list
        3. Normalizes CAST and NULLIF usage
        """
        sql = sql.strip()

        # Remove markdown/code fences
        sql = re.sub(r"^```[a-z]*|```$", "", sql, flags=re.IGNORECASE).strip()

        # Remove DISTINCT when misplaced after a comma in SELECT list
        # e.g. "SELECT SUM(x), DISTINCT col" -> "SELECT SUM(x), col"
        sql = re.sub(r",\s*DISTINCT\s+", ", ", sql, flags=re.IGNORECASE)

        # Replace CAST(... AS REAL) if inside math operations
        sql = re.sub(
            r"CAST\((.*?)\s+AS\s+REAL\)", r"\1::REAL", sql, flags=re.IGNORECASE
        )
        # Remove invalid nested NULLIF patterns like NULLIF(NULLIF,0)(x)
        sql = re.sub(
            r"NULLIF\(NULLIF,0\)\((.*?)\)", r"NULLIF(\1,0)", sql, flags=re.IGNORECASE
        )

        # remove multiple newlines
        sql = re.sub(r"\n\s*\n", "\n", sql)

        # FEAT_36: inject LIMIT 101 (N+1) when SELECT has no LIMIT/FETCH clause
        if FEAT_15 and FEAT_36:
            if re.search(r'\bSELECT\b', sql, re.IGNORECASE) and \
               not re.search(r'\b(LIMIT|FETCH)\b', sql, re.IGNORECASE):
                sql = sql.rstrip('; \t\n') + '\nLIMIT 101'

        return sql

    def _classify_query_complexity(self, question: str) -> str:
        """FEAT_29: keyword heuristic — no extra LLM call, zero latency."""
        q = question.lower()
        complex_signals = [
            "join", "group by", "having", "union", "except", "intersect",
            "with ", "rank(", "partition by", "case when", "subquery",
            "подзапрос", "рекурс", "окно", "ранг",
        ]
        moderate_signals = [
            "count", "sum", "avg", "average", "max", "min",
            "среднее", "сумм", "количество", "максимальн", "минимальн",
            "group", "order by", "distinct",
        ]
        complex_score = sum(1 for s in complex_signals if s in q)
        if complex_score >= 2:
            return "complex"
        if complex_score == 1 or any(s in q for s in moderate_signals):
            return "moderate"
        return "simple"

    async def generate_sql(self, user_query: str, prev_sql: str | None = None,
                           error_feedback: str | None = None,
                           complexity: str | None = None,
                           decomposition: str | None = None) -> dict[str, Any]:
        """
        Генерирует SQL запрос из естественно-языкового запроса

        Args:
            user_query (str): Запрос на естественном языке

        Returns:
            Dict[str, Any]: Результат в формате Model Context Protocol
        """
        # FEAT_27: self-correction prompt when error context is available
        # FEAT_29: complexity-based prompt selection (fallback to standard)
        if FEAT_27 and prev_sql and error_feedback:
            sql_prompt = SQL_RETRY_PROMPT_TEMPLATE.format(
                user_query=user_query,
                prev_sql=prev_sql,
                error_feedback=error_feedback,
                sql_dialect="PostgreSQL",
            )
        elif FEAT_29 and complexity == "complex":
            sql_prompt = SQL_PROMPT_COMPLEX_TEMPLATE.format(
                user_query=user_query, sql_dialect="PostgreSQL"
            )
        elif FEAT_29 and complexity == "simple":
            sql_prompt = SQL_PROMPT_SIMPLE_TEMPLATE.format(
                user_query=user_query, sql_dialect="PostgreSQL"
            )
        else:
            # FEAT_14: use strict SQL rules prompt; fallback to basic prompt without rules
            template = SQL_PROMPT_TEMPLATE if FEAT_14 else SQL_PROMPT_BASIC
            sql_prompt = template.format(
                user_query=user_query, sql_dialect="PostgreSQL"
            )
        try:
            messages = [SystemMessage(content=self.system_prompt)]
            # FEAT_25: prepend pruned schema as extra context before SQL prompt
            if FEAT_25:
                pruned = await self._prune_schema_for_question(user_query)
                if pruned and pruned != self.db_schema:
                    messages.append(UserMessage(
                        source="user",
                        content=f"Relevant schema for this question:\n{pruned}",
                    ))
            # FEAT_30: prepend CoT decomposition before SQL prompt
            if decomposition:
                messages.append(UserMessage(
                    source="user",
                    content=f"Декомпозиция запроса по шагам:\n{decomposition}",
                ))
            messages.append(UserMessage(source="user", content=dedent(sql_prompt)))

            result = await self._llm_call_with_retry(messages)
            raw_sql = result.content.strip()
            raw_sql_attempted = raw_sql  # preserve for FEAT_27 retry context

            # FEAT_15: sanitize LLM output (markdown, DISTINCT, CAST fixes)
            sanitized_sql = self.sanitize_sql(raw_sql) if FEAT_15 else raw_sql

            # Unconditional semantic fixes (always on, regardless of FEAT_15):
            # 1. TO_DATE on DATE columns → strip to bare column ref (PG errors on DATE arg)
            sanitized_sql = re.sub(
                r"\bTO_DATE\(\s*((?:\w+\.)?(?:\"[^\"]+\"|\b[a-zA-Z_]\w*\b))\s*,\s*'[^']+'\)",
                r"\1", sanitized_sql, flags=re.IGNORECASE
            )
            # 2. NULLS ordering: PG default (DESC→NULLS FIRST) is opposite SQLite (DESC→NULLS LAST)
            sanitized_sql = re.sub(r'\bDESC\b(?!\s+NULLS)', 'DESC NULLS LAST', sanitized_sql, flags=re.IGNORECASE)
            sanitized_sql = re.sub(r'\bASC\b(?!\s+NULLS)', 'ASC NULLS FIRST', sanitized_sql, flags=re.IGNORECASE)

            sanitized_sql = self._strip_sql_comments(sanitized_sql)

            if not self._validate_sql(sanitized_sql):
                raise ValueError("Сгенерированный SQL не прошел валидацию")

            logger.info(f"SQL generated successfully: {sanitized_sql[:120]}...")
            return {
                "status": "success",
                "user_query": user_query,
                "sql_query": sanitized_sql,
                "metadata": {
                    "tables_accessed": self._get_accessed_tables(sanitized_sql),
                    "validation_passed": True,
                    "sanitization_passed": True,
                },
            }
        except Exception as e:
            logger.exception(f"Failed to generate SQL from {user_query=}")
            return {
                "status": "error",
                "error": str(e),
                "sql_attempted": locals().get("raw_sql_attempted"),  # for FEAT_27 retry context
                "user_query": user_query,
                "metadata": {
                    "validation_passed": False,
                    "sanitization_passed": False,
                },
            }

    def _get_accessed_tables(self, sql: str) -> list[str]:
        try:
            parsed = parse_one(sql, dialect=Dialects.POSTGRES)
            return sorted({table.name.lower() for table in parsed.find_all(exp.Table)})
        except Exception:
            return []

    # ── FEAT_26: learnt_hints ─────────────────────────────────────────────────

    def _hints_path(self) -> pathlib.Path:
        return pathlib.Path("hints") / f"{self._db_id}.json"

    def _load_learnt_hints(self) -> list[dict]:
        path = self._hints_path()
        if not path.exists():
            return []
        try:
            return json_module.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_learnt_hint(self, question: str, sql: str) -> None:
        path = self._hints_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        hints = self._load_learnt_hints()
        # Avoid duplicates by question text
        if not any(h["question"] == question for h in hints):
            hints.append({"question": question, "sql": sql})
            path.write_text(json_module.dumps(hints, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── FEAT_21: MinHash column similarity ───────────────────────────────────

    def _sample_column_values(self, table: str, column: str, limit: int = 50) -> list[str]:
        """Sample distinct non-null values from a column."""
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(
                    text(f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT {limit}')
                )
                return [str(r[0]) for r in rows]
        except Exception:
            return []

    def _minhash_similarity(self, vals1: list[str], vals2: list[str], n: int = 64) -> float:
        """Estimate Jaccard similarity via MinHash (no external libs)."""
        if not vals1 or not vals2:
            return 0.0
        s1, s2 = set(vals1), set(vals2)
        matches = sum(
            min(hash((seed, v)) & 0x7FFFFFFF for v in s1) ==
            min(hash((seed, v)) & 0x7FFFFFFF for v in s2)
            for seed in range(n)
        )
        return matches / n

    def _find_similar_columns(self) -> str:
        """FEAT_21: find cross-table column pairs with MinHash similarity > 0.3."""
        inspector = inspect(self.engine)
        tables = inspector.get_table_names(schema="public")

        # Collect TEXT columns and their sampled values
        col_values: dict[tuple[str, str], list[str]] = {}
        for table in tables:
            for col in inspector.get_columns(table, schema="public"):
                if "TEXT" in str(col["type"]).upper() or "VARCHAR" in str(col["type"]).upper():
                    key = (table, col["name"])
                    col_values[key] = self._sample_column_values(table, col["name"])

        # Compare cross-table pairs
        keys = list(col_values.keys())
        pairs = []
        for i, (t1, c1) in enumerate(keys):
            for t2, c2 in keys[i + 1:]:
                if t1 == t2:
                    continue
                sim = self._minhash_similarity(col_values[(t1, c1)], col_values[(t2, c2)])
                if sim >= 0.3:
                    pairs.append((sim, t1, c1, t2, c2))

        if not pairs:
            return ""
        pairs.sort(reverse=True)
        lines = [f"{t1}.{c1} ≈ {t2}.{c2} (сходство: {sim:.2f})" for sim, t1, c1, t2, c2 in pairs[:10]]
        return "\n".join(lines)

    # ── FEAT_23: NL table descriptions ───────────────────────────────────────

    def _nl_descriptions_path(self) -> pathlib.Path:
        return pathlib.Path("nl_descriptions") / f"{self._db_id}.json"

    async def _generate_table_descriptions(self) -> dict[str, str]:
        """FEAT_23: generate 1-2 sentence NL descriptions per table, cached in nl_descriptions/."""
        cache_path = self._nl_descriptions_path()
        if cache_path.exists():
            try:
                return json_module.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        inspector = inspect(self.engine)
        tables = inspector.get_table_names(schema="public")

        descriptions: dict[str, str] = {}
        for table in tables:
            cols = inspector.get_columns(table, schema="public")
            col_list = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
            prompt = TABLE_DESCRIPTION_PROMPT.format(table_name=table, columns=col_list)
            try:
                result = await self._llm_call_with_retry(
                    [UserMessage(source="user", content=prompt)]
                )
                descriptions[table] = result.content.strip()
            except Exception as e:
                logger.warning(f"FEAT_23: failed to describe table {table}: {e}")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json_module.dumps(descriptions, ensure_ascii=False, indent=2), encoding="utf-8")
        return descriptions

    # ── FEAT_30: CoT sub-question decomposition ──────────────────────────────

    async def _decompose_question(self, user_query: str) -> str:
        """FEAT_30: one LLM call to break complex question into sub-steps."""
        prompt = DECOMPOSITION_PROMPT.format(user_query=user_query, db_schema=self.db_schema)
        try:
            result = await self._llm_call_with_retry(
                [UserMessage(source="user", content=prompt)]
            )
            return result.content.strip()
        except Exception:
            return ""

    # ── FEAT_25: per-question schema pruning ─────────────────────────────────

    async def _prune_schema_for_question(self, user_query: str) -> str:
        """FEAT_25: one LLM call to select only relevant tables/columns."""
        if not self.db_schema:
            return self.db_schema
        prompt = SCHEMA_PRUNING_PROMPT.format(
            user_query=user_query,
            db_schema=self.db_schema,
        )
        try:
            messages = [UserMessage(source="user", content=prompt)]
            result = await self._llm_call_with_retry(messages)
            pruned = result.content.strip()
            return pruned if pruned else self.db_schema
        except Exception:
            return self.db_schema

    # ── FEAT_28: self-consistency (3 samples + majority vote) ────────────────

    async def _generate_sql_with_voting(
        self, user_query: str, n: int = 3, complexity: str | None = None
    ) -> str | None:
        """FEAT_28: generate n SQL candidates and pick the majority-vote winner."""
        candidates: list[str] = []
        for _ in range(n):
            result = await self.generate_sql(user_query, complexity=complexity)
            if result["status"] == "success":
                candidates.append(result["sql_query"])
        if not candidates:
            return None
        normalized = [re.sub(r"\s+", " ", s.strip().lower()) for s in candidates]
        winner_norm = Counter(normalized).most_common(1)[0][0]
        for orig, norm in zip(candidates, normalized):
            if norm == winner_norm:
                return orig
        return candidates[0]

    # ── SQL execution ─────────────────────────────────────────────────────────

    def execute_safe(self, sql: str) -> dict[str, Any]:
        """
        Safe execution of SQL query (SELECT-only) using SQLAlchemy.
        """
        try:
            sanitized_sql = sql
            if not self._validate_sql(sanitized_sql):
                raise ValueError("Запрос не прошел валидацию")

            with self.engine.connect() as conn:
                result: Result = conn.execute(text(sanitized_sql))
                rows = result.fetchall()

            if not rows:
                raise ValueError("Запрос вернулся пустым")

            columns = result.keys()
            results = [dict(zip(columns, row)) for row in rows]

            return {
                "status": "success",
                "results": results,
                "columns": list(columns),
                "row_count": len(results),
                "sql_executed": sanitized_sql,
            }

        except Exception as e:
            logger.exception(f"Failed to execute query {sql}.")
            return {
                "status": "error",
                "error": str(e),
                "sql_attempted": sql,
            }

    async def _verify_sql_against_query(
        self, user_query: str, sql_query: str
    ) -> dict[str, Any]:
        """
        Проверяет, соответствует ли сгенерированный SQL-запрос оригинальному запросу пользователя.
        """
        verification_prompt = VERIFICATION_PROMPT_TEMPLATE.format(
            user_query=user_query,
            sql_query=sql_query,
        )

        messages = [SystemMessage(content=verification_prompt)]

        try:
            result = await self._llm_call_with_retry(messages)
            response_text = result.content.strip()

            if response_text.lower() == "ok":
                return {"status": "success", "is_correct": True}
            else:
                return {
                    "status": "success",
                    "is_correct": False,
                    "reason": response_text,
                }
        except Exception as e:
            logger.exception(
                f"Failed to verify generated query matches user query: {sql_query=} {user_query=}"
            )
            return {"status": "error", "error": str(e)}

    @print_result()
    async def query(
        self,
        user_query: str,
    ) -> dict[str, Any]:
        """
        Полный цикл: генерация SQL + выполнение
        Args:
            user_query (str): Запрос на естественном языке
            check_sql_query (bool): Флаг, требуется ли проверять SQL-запрос на корректность
        Returns:
            Dict[str, Any]: Объединенные результаты генерации и выполнения
        """
        logger.info(f"query() started: {user_query[:100]}")

        # Ambiguity checking — FEAT_32: AmbiSQL taxonomy pipeline; else: classic _check_ambiguity
        logger.info("Проверяю запрос на неоднозначность...")
        was_flagged_ambiguous = False

        if FEAT_32:
            try:
                taxonomy = await self._detect_ambiguity_by_taxonomy(user_query)
                if taxonomy.get("is_ambiguous"):
                    ambiguity_type = taxonomy.get("ambiguity_type", "unknown")
                    reason = taxonomy.get("reason", "")
                    logger.info(f"AmbiSQL: тип={ambiguity_type} — {reason[:80]}")
                    if FEAT_33:
                        clarification = await self._generate_clarification_question(
                            user_query, ambiguity_type, reason
                        )
                        if FEAT_34:
                            user_query = await self._auto_resolve_ambiguity(
                                user_query,
                                clarification.get("question", ""),
                                clarification.get("options", []),
                            )
                            logger.info(f"AmbiSQL: rewritten → {user_query[:80]}")
                            # rewritten query is unambiguous — proceed to SQL generation
                        else:
                            return {"status": "ambiguous"}
                    else:
                        return {"status": "ambiguous"}
            except Exception as _ambisql_err:
                logger.warning(f"AmbiSQL pipeline failed ({_ambisql_err}), falling back to classic check")
                ambiguity_check = await self._check_ambiguity(user_query)
                was_flagged_ambiguous = (
                    ambiguity_check["status"] == "success" and ambiguity_check["ambiguous"]
                )
                if was_flagged_ambiguous:
                    logger.info(f"Запрос неоднозначен: {ambiguity_check['clarification_needed']}")
                    if not FEAT_12:
                        return {"status": "ambiguous"}
                    logger.info("Optimistic fallback включён после сбоя AmbiSQL")
                if ambiguity_check["status"] == "error":
                    logger.warning("Ambiguity check failed, proceeding with SQL generation")
        else:
            ambiguity_check = await self._check_ambiguity(user_query)
            was_flagged_ambiguous = (
                ambiguity_check["status"] == "success" and ambiguity_check["ambiguous"]
            )
            if was_flagged_ambiguous:
                logger.info(f"Запрос неоднозначен: {ambiguity_check['clarification_needed']}")
                if not FEAT_12:
                    return {"status": "ambiguous"}
                logger.info("Optimistic fallback включён — пробуем сгенерировать SQL несмотря на флаг ambiguous")
            if ambiguity_check["status"] == "error":
                logger.warning("Ambiguity check failed, proceeding with SQL generation")

        # Main query generation
        retries = 0
        success = False
        raw_sql = ""
        final_result = {"status": "ambiguous"} if was_flagged_ambiguous else {"status": "error"}

        # FEAT_27: track previous attempt for error-feedback self-correction
        last_error: str | None = None
        last_sql: str | None = None

        # FEAT_29: classify once before the retry loop (zero extra LLM calls)
        complexity: str | None = self._classify_query_complexity(user_query) if FEAT_29 else None
        if FEAT_29:
            logger.info(f"Query complexity: {complexity}")

        # FEAT_30: CoT decomposition for complex/moderate queries (one extra LLM call)
        decomposition: str | None = None
        if FEAT_30 and complexity in ("complex", "moderate", None):
            decomposition = await self._decompose_question(user_query)
            if decomposition:
                logger.info(f"FEAT_30 decomposition: {decomposition[:100]}")

        # FEAT_28: self-consistency replaces the retry loop
        if FEAT_28:
            winner = await self._generate_sql_with_voting(user_query, n=3, complexity=complexity)
            if winner:
                if FEAT_26:
                    self._save_learnt_hint(user_query, winner)
                return {"status": "success", "query": winner}
            return {"status": "error"}

        max_retries_effective = MAX_RETRIES if FEAT_17 else 1
        while not success and retries < max_retries_effective:
            logger.info(
                f"Попытка {retries + 1}/{max_retries_effective}. Генерирую валидный SQL-запрос... "
            )
            generation_result = await self.generate_sql(
                user_query,
                prev_sql=last_sql if FEAT_27 else None,
                error_feedback=last_error if FEAT_27 else None,
                complexity=complexity,
                decomposition=decomposition,
            )

            if generation_result["status"] != "success":
                last_error = generation_result.get("error", "Ошибка генерации SQL")
                last_sql = generation_result.get("sql_attempted")
                retries += 1
                continue

            raw_sql = generation_result["sql_query"]
            # Извлекает SQL-запрос из markdown блока, если ответ полностью обернут в ```
            if raw_sql[:3] == "```":
                raw_sql = re.sub(
                    r"^```(?:sql)?\s*|\s*```$",
                    "",
                    raw_sql.strip(),
                    flags=re.IGNORECASE,
                )

            # FEAT_20: LLM verification — check SQL matches the user's intent
            if FEAT_20:
                verify_result = await self._verify_sql_against_query(user_query, raw_sql)
                if not verify_result.get("is_correct"):
                    reason = verify_result.get("reason", "")
                    logger.info(f"LLM verify: не OK — {reason[:100]}")
                    last_sql = raw_sql
                    last_error = f"SQL не соответствует запросу: {reason}"
                    retries += 1
                    continue

            final_result = {"status": "success", "query": raw_sql}
            success = True
            # FEAT_26: persist successful SQL pattern for future runs
            if FEAT_26:
                self._save_learnt_hint(user_query, raw_sql)

        if was_flagged_ambiguous and success:
            logger.info("Optimistic fallback: SQL успешно сгенерирован для ранее помеченного запроса")

        return final_result
