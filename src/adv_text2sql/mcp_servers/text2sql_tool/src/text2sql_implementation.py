import json as json_module
import logging
import os
import re
import time
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
    SQL_PROMPT_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE,
    VERIFICATION_PROMPT_TEMPLATE,
)
from .utils import print_result

# Load environment variables for the main block
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 7))
OPTIMISTIC_AMBIGUITY_FALLBACK = os.getenv("OPTIMISTIC_AMBIGUITY_FALLBACK", "true").lower() == "true"

# Ablation feature flags (default true = full feature set when not running ablation)
FEAT_1 = os.getenv("FEAT_1", "true").lower() == "true"  # FK/PK relationships in prompt
FEAT_3 = os.getenv("FEAT_3", "true").lower() == "true"  # regex type detection + column stats
FEAT_5 = os.getenv("FEAT_5", "true").lower() == "true"  # light schema (names + types)
FEAT_7 = os.getenv("FEAT_7", "true").lower() == "true"  # compact stats formatting


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

    # File handler — JSON DEBUG
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
        )

        self.llm_client = llm_client

        logger.info("Initialized Text2SQLGenerator")

    def build(self):
        start = time.time()
        logger.info(f"Build started for {self.db_uri}")
        logger.info(f"Feature flags: FEAT_1={FEAT_1} FEAT_3={FEAT_3} FEAT_5={FEAT_5} FEAT_7={FEAT_7}")

        self.db_schema = self._get_db_schema_light() if FEAT_5 else ""

        if FEAT_1:
            self.db_relationships = self._explore_db_relationships()
            self.relationships_str = self._format_relationships()
        else:
            self.db_relationships = {"foreign_keys": [], "primary_keys": {}}
            self.relationships_str = ""

        if FEAT_3:
            self.column_stats = self._explore_column_statistics()
            self.column_stats_str = (
                self._format_column_statistics(self.column_stats) if FEAT_7
                else self._format_column_statistics_verbose(self.column_stats)
            )
        else:
            self.column_stats = {}
            self.column_stats_str = ""

        self.system_prompt = self._create_system_prompt()

        elapsed = time.time() - start
        logger.info(
            f"Build completed in {elapsed:.1f}s — "
            f"stats for {len(self.column_stats)} tables, "
            f"system prompt {len(self.system_prompt)} chars"
        )

    def _update_db_schema(self, db_uri):
        self.db_uri = db_uri
        self.engine.dispose()
        self.engine = create_engine(db_uri, pool_pre_ping=True)
        self.build()

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
            try:
                conn.rollback()
            except Exception:
                pass

        try:
            res = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"'))
            return int(res.scalar())
        except Exception:
            logger.exception(f"COUNT(*) failed for table {table}")
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
            try:
                conn.rollback()
            except Exception:
                pass

        return None

    def _explore_column_statistics(self) -> dict:
        """
        Оркестратор: собирает row counts, типы и статистики для всех таблиц.
        """
        inspector = inspect(self.engine)
        tables = inspector.get_table_names(schema="public")
        result = {}

        with self.engine.connect() as conn:
            for table in tables:
                try:
                    row_count = self._get_table_row_count(conn, table)
                    columns = inspector.get_columns(table, schema="public")
                    col_types = self._detect_column_types_for_table(
                        conn, table, columns
                    )

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
        self, messages, max_retries: int = 3, base_delay: float = 2.0
    ):
        """Обёртка LLM вызовов с retry и exponential backoff при 429."""
        import asyncio as _asyncio

        for attempt in range(max_retries):
            try:
                return await self.llm_client.create(messages)
            except Exception as e:
                err_str = str(e).lower()
                if "429" in str(e) or "rate" in err_str:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Rate limited, retrying in {delay}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await _asyncio.sleep(delay)
                else:
                    raise
        # Last attempt — let exception propagate
        return await self.llm_client.create(messages)

    async def _check_ambiguity(self, user_query: str) -> dict[str, Any]:
        """
        Проверяет пользовательский запрос на неоднозначность с помощью LLM.
        """
        ambiguity_prompt = AMBIGUITY_PROMPT_TEMPLATE.format(
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

            if response_text.lower() == "ok":
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
        """Валидация SQL запроса с помощью SQLGlot"""
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

        return sql

    async def generate_sql(self, user_query: str) -> dict[str, Any]:
        """
        Генерирует SQL запрос из естественно-языкового запроса

        Args:
            user_query (str): Запрос на естественном языке

        Returns:
            Dict[str, Any]: Результат в формате Model Context Protocol
        """
        sql_prompt = SQL_PROMPT_TEMPLATE.format(
            user_query=user_query, sql_dialect="PostgreSQL"
        )
        try:
            # Создаем цепочку обработки запроса
            messages = [
                SystemMessage(content=self.system_prompt),
                UserMessage(source="user", content=dedent(sql_prompt)),
            ]

            # Прямой вызов LLM
            result = await self._llm_call_with_retry(messages)
            raw_sql = result.content.strip()

            # Очистка и валидация
            # sanitized_sql = self._sanitize_sql(raw_sql) # TODO сделать нормально
            sanitized_sql = self.sanitize_sql(raw_sql)

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

        # Ambiguity checking
        logger.info("Проверяю запрос на неоднозначность...")
        ambiguity_check = await self._check_ambiguity(user_query)

        was_flagged_ambiguous = (
            ambiguity_check["status"] == "success" and ambiguity_check["ambiguous"]
        )

        if was_flagged_ambiguous:
            logger.info(f"Запрос неоднозначен: {ambiguity_check['clarification_needed']}")
            if not OPTIMISTIC_AMBIGUITY_FALLBACK:
                return {"status": "ambiguous"}
            logger.info("Optimistic fallback включён — пробуем сгенерировать SQL несмотря на флаг ambiguous")

        if ambiguity_check["status"] == "error":
            logger.warning("Ambiguity check failed, proceeding with SQL generation")
            # НЕ возвращаем error — продолжаем генерацию SQL

        # Main query generation
        retries = 0
        success = False
        raw_sql = ""
        final_result = {"status": "ambiguous"} if was_flagged_ambiguous else {"status": "error"}

        while not success and retries < MAX_RETRIES:
            logger.info(
                f"Попытка {retries + 1}/{MAX_RETRIES}. Генерирую валидный SQL-запрос... "
            )
            generation_result = await self.generate_sql(user_query)

            if generation_result["status"] != "success":
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

            final_result = {"status": "success", "query": raw_sql}
            success = True

        if was_flagged_ambiguous and success:
            logger.info("Optimistic fallback: SQL успешно сгенерирован для ранее помеченного запроса")

        return final_result
