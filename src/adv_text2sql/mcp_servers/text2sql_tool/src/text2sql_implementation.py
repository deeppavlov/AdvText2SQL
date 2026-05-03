import asyncio
import logging
import os
import re
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


logger = logging.getLogger("text2sql_tool")

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
        self.db_schema = self._get_db_schema_light()
        # self.db_schema = self._get_db_schema_heavy()
        self.system_prompt = self._create_system_prompt()

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

    # def _get_db_schema_light(self) -> str:
    #     """
    #     Lightweight schema extraction for LLM prompts.

    #     Includes:
    #     - table names
    #     - column names
    #     - column types
    #     """
    #     inspector = inspect(self.engine)
    #     schema_parts = []

    #     # Default schema for PostgreSQL
    #     # print(self.db_uri)
    #     tables = inspector.get_table_names(schema="public")
    #     if not tables:
    #         print("this db failed: ", self.db_uri)
    #         raise RuntimeError("No tables found in public schema")

    #     for table in tables:
    #         schema_parts.append(f"TABLE {table}")

    #         columns = inspector.get_columns(table, schema="public")
    #         for col in columns:
    #             col_type = str(col["type"])
    #             schema_parts.append(f"  - {col['name']} ({col_type})")

    #         schema_parts.append("")

    #     return "\n".join(schema_parts).strip()

    def _get_db_schema_light(self) -> str:
        """
        Robust schema extraction. Handles empty DBs, custom schemas, and search paths.
        """
        inspector = inspect(self.engine)
        schema_parts = []
        system_schemas = {"information_schema", "pg_catalog", "pg_toast"}
        
        # 1. Try default search path first (fastest & covers 95% cases)
        default_tables = inspector.get_table_names()
        if default_tables:
            for table in default_tables:
                schema_parts.append(f"TABLE {table}")
                columns = inspector.get_columns(table)
                for col in columns:
                    schema_parts.append(f"  - {col['name']} ({str(col['type'])})")
                schema_parts.append("")
            return "\n".join(schema_parts).strip()

        # 2. Fallback: scan all non-system schemas explicitly
        for schema in inspector.get_schema_names():
            if schema in system_schemas:
                continue
            tables = inspector.get_table_names(schema=schema)
            if tables:
                for table in tables:
                    schema_parts.append(f"TABLE {schema}.{table}")
                    columns = inspector.get_columns(table, schema=schema)
                    for col in columns:
                        schema_parts.append(f"  - {col['name']} ({str(col['type'])})")
                    schema_parts.append("")
                    
        # 3. Graceful degradation for empty/unreachable DBs
        if not schema_parts:
            logger.warning(f"No tables found in {self.db_uri}. Returning placeholder.")
            return "DATABASE EMPTY OR INACCESSIBLE"
            
        return "\n".join(schema_parts).strip()
    
    def _get_calibrated_ambiguity_prompt(self, user_query: str) -> str:
        """
        Calibrated prompt with REAL examples from ambrosia_train.json.
        Focus: detect ONLY ambiguities that lead to DIFFERENT SQL queries.
        """
        return f"""
    You are an ambiguity detection expert for Text-to-SQL. Use the AmbiSQL taxonomy.

    ### Database Schema:
    {self.db_schema}

    ### Ambiguity Taxonomy (mark ONLY if leads to DIFFERENT SQL):
    1. ambiguous_scope: unclear quantifier scope (e.g., "each can hold 200" → per room or combined?)
    2. unclear_schema: ambiguous column reference (e.g., "ranked 2" → which column: rank/position/positionOrder?)
    3. unclear_value: ambiguous value reference (e.g., "after Vietnam War" → which date: 1975-04-30 or 1975-12-31?)
    4. missing_sql_keywords: unclear operation (e.g., "users by date" → ORDER BY? GROUP BY? WHERE?)

    ### CRITICAL RULES:
    MARK AS AMBIGUOUS if query has phrases like:
    - "each" / "every" with capacity/quantity → ambiguous_scope
    - "after/before [historical event]" → unclear_value (needs exact date)
    - "ranked N" / "position N" without column name → unclear_schema
    - "by [attribute]" without operation context → missing_sql_keywords

    DO NOT mark as ambiguous if:
    - Explicit metric: "average", "max", "min", "sum"
    - Explicit period: month name, year (e.g., "January 2025")
    - "top-N" → implies ORDER BY + LIMIT
    - Simple filter: "WHERE condition" is obvious from context

    ### REAL EXAMPLES FROM AMBROSIA DATASET:

    AMBIGUOUS (mark as ambiguous):
    Query: "Show banquet halls and conference rooms where each can hold 200 people"
    Reason: "each" → ambiguous_scope (per room individually or combined capacity?)

    Query: "How many drivers born after the Vietnam War have been ranked 2?"
    Reason: "after Vietnam War" → unclear_value (needs exact date); "ranked 2" → unclear_schema (which column?)

    Query: "Show users by registration date"
    Reason: "by date" → missing_sql_keywords (ORDER BY? GROUP BY? WHERE?)

    UNAMBIGUOUS (do NOT mark):
    Query: "Show top-5 customers by purchase amount in January 2025"
    Reason: "top-5" → ORDER BY + LIMIT; "January 2025" → explicit period

    Query: "List all restaurants serving Italian cuisine"
    Reason: Simple filter with UNION — no ambiguity

    ### User Query:
    "{user_query}"

    ### Response Format (VALID JSON ONLY):
    {{
        "ambiguous": true/false,
        "ambiguity_types": ["type1", ...],  // empty [] if unambiguous
        "ambiguous_phrases": ["phrase1", ...],
        "reasoning": "1 sentence: why ambiguous or why NOT ambiguous"
    }}
    """

    def _is_false_positive(self, user_query: str) -> bool:
        """
        Aggressive post-filtering of FALSE POSITIVES (overly cautious LLM).
        Returns True if query is CLEARLY unambiguous despite LLM's hesitation.
        """
        query_lower = user_query.lower().strip()
        
        # STRONG signals of unambiguity (override LLM's false positive)
        strong_unambiguous = [
            r'\b(top|first|last)\s*-?\d+\b',           # "top-5", "first 10"
            r'\b(average|avg|mean)\b',                 # "average salary"
            r'\b(max|maximum|min|minimum|sum)\b',      # explicit aggregation
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',  # months
            r'\b202[0-9]\b',                           # years 2020-2029
            r'\blist\s+all\b',                         # "list all users"
            r'\bshow\s+all\b',                         # "show all records"
            r'\b(id|name|address|city|country)\b.*\blist\b',  # basic entity listing
        ]
        
        # WEAK signals (don't override LLM if it detected real ambiguity)
        weak_unambiguous = [
            r'\bby\s+(date|month|year|category)\b',    # "users by date" — could be ambiguous!
            r'\bfor\s+(january|february|202[0-9])\b',
        ]
        
        # Check STRONG signals FIRST (high confidence unambiguous)
        for pattern in strong_unambiguous:
            if re.search(pattern, query_lower):
                return True
        
        # DO NOT filter weak signals — they might be truly ambiguous!
        # (e.g., "users by date" IS ambiguous per Ambrosia taxonomy)
        
        return False

    def _create_system_prompt(self) -> str:
        """Создает системный промпт с описанием схемы БД"""
        return SYSTEM_PROMPT_TEMPLATE.format(
            db_schema=self.db_schema, sql_dialect="PostgreSQL"
        )

    # async def _check_ambiguity(self, user_query: str) -> dict[str, Any]:
    #     """
    #     Проверяет пользовательский запрос на неоднозначность с помощью LLM.
    #     """
    #     ambiguity_prompt = AMBIGUITY_PROMPT_TEMPLATE.format(
    #         db_schema=self.db_schema,
    #         user_query=user_query,
    #     )

    #     messages = [
    #         SystemMessage(content=ambiguity_prompt),
    #         UserMessage(
    #             source="user",
    #             content="Проверь запрос на однозначность.",
    #         ),
    #     ]

    #     """
    #     content=(
    #         "Определи, можно ли вообще ответить на этот вопрос, "
    #         "используя ТОЛЬКО данную схему БД. "
    #         "Если ответить нельзя — напиши NOT_ANSWERABLE. "
    #         "Если ответ возможен, но запрос неоднозначен — опиши, что нужно уточнить. "
    #         "Если запрос полностью однозначен и ответим — напиши OK."
    #     ),
    #     """

    #     try:
    #         result = await self.llm_client.create(messages)
    #         response_text = result.content.strip()

    #         if response_text.lower() == "ok":
    #             return {"status": "success", "ambiguous": False}
    #         else:
    #             return {
    #                 "status": "success",
    #                 "ambiguous": True,
    #                 "clarification_needed": response_text,
    #             }

    #     except Exception as e:
    #         logger.exception(f"Failed to check ambiguity for query: {user_query}")
    #         return {"status": "error"}

    async def _check_ambiguity(self, user_query: str) -> dict[str, Any]:
        """
        Hybrid ambiguity detection: LLM + aggressive post-filtering.
        Optimized for Ambrosia benchmark with minimal false positives on BIRD.
        """
        # Step 1: LLM detection with calibrated English prompt
        ambiguity_prompt = self._get_calibrated_ambiguity_prompt(user_query)
        
        messages = [
            SystemMessage(content=ambiguity_prompt),
            UserMessage(source="user", content="Respond with VALID JSON ONLY:"),
        ]
        
        try:
            result = await self.llm_client.create(messages)
            response_text = result.content.strip()
            response_text = re.sub(r'^```(?:json)?\s*|\s*```$', '', response_text.strip(), flags=re.IGNORECASE)
            
            import json
            parsed = json.loads(response_text)
            is_ambiguous = parsed.get("ambiguous", False)
            
            # Step 2: AGGRESSIVE post-filtering (critical for BIRD accuracy)
            if is_ambiguous and self._is_false_positive(user_query):
                logger.info(f"FILTERED_FALSE_POSITIVE | query='{user_query[:70]}'")
                return {"status": "success", "ambiguous": False, "ambiguity_details": {"method": "post_filter"}}
            
            # Log ONLY truly ambiguous queries for analysis
            if is_ambiguous:
                types = parsed.get("ambiguity_types", ["unknown"])
                logger.info(f"AMBIGUOUS_QUERY | types={types} | query='{user_query[:70]}'")
            
            return {
                "status": "success",
                "ambiguous": is_ambiguous,
                "ambiguity_details": {**parsed, "method": "llm_calibrated"}
            }
            
        except Exception as e:
            logger.warning(f"LLM ambiguity check failed (defaulting to unambiguous): {str(e)[:100]}")
            return {"status": "success", "ambiguous": False, "ambiguity_details": {"method": "fallback"}}


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
        2. Wraps DISTINCT queries with ORDER BY in a subquery
        3. Normalizes CAST and NULLIF usage
        """
        sql = sql.strip()

        # Remove markdown/code fences
        sql = re.sub(r"^```[a-z]*|```$", "", sql, flags=re.IGNORECASE).strip()

        # Detect DISTINCT + ORDER BY and wrap in a subquery
        distinct_order_by_pattern = re.compile(
            r"SELECT\s+DISTINCT\s+(.*?)\s+FROM\s+(.*?)\s+ORDER\s+BY\s+(.*?)(LIMIT\s+\d+)?;",
            flags=re.IGNORECASE | re.DOTALL,
        )
        match = distinct_order_by_pattern.search(sql)
        if match:
            select_cols, from_clause, order_by_clause, limit_clause = match.groups()
            limit_clause = limit_clause or ""
            sql = f"""
            SELECT *
            FROM (
                SELECT DISTINCT {select_cols}
                FROM {from_clause}
            ) sub
            ORDER BY {order_by_clause} {limit_clause};
            """.strip()

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
            result = await self.llm_client.create(messages)
            raw_sql = result.content.strip()

            # Очистка и валидация
            # sanitized_sql = self._sanitize_sql(raw_sql) # TODO сделать нормально
            sanitized_sql = self.sanitize_sql(raw_sql)

            sanitized_sql = self._strip_sql_comments(sanitized_sql)

            if not self._validate_sql(sanitized_sql):
                raise ValueError("Сгенерированный SQL не прошел валидацию")

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
            result = await self.llm_client.create(messages)
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

        # Ambiguity checking
        logger.info("Проверяю запрос на неоднозначность...")
        ambiguity_check = await self._check_ambiguity(user_query)

        # if ambiguity_check["status"] == "success" and ambiguity_check["ambiguous"]:
        #     logger.info(f"Запрос неоднозначен. Требуется уточнение по причине: {ambiguity_check['clarification_needed']}")
        #     return {"status": "ambiguous"}

        
        if ambiguity_check["status"] == "success" and ambiguity_check["ambiguous"]:
            # Извлекаем типы из структурированного ответа
            ambiguity_types = ambiguity_check.get("ambiguity_details", {}).get("ambiguity_types", ["unknown"])
            logger.info(f"AMBIGUOUS_QUERY | types={ambiguity_types} | query='{user_query[:60]}'")
            return {"status": "ambiguous"}

        if ambiguity_check["status"] == "error":
            logger.info("Произошла ошибка при проверка неоднозначности.")
            return {"status": "error"}


        # Main query generation
        retries = 0
        success = False
        raw_sql = ""
        final_result = {"status": "error"}

        while not success and retries < MAX_RETRIES:
            logger.info(
                f"Попытка {retries + 1}/{MAX_RETRIES}. Генерирую валидный SQL-запрос... "
            )
            generation_result = await self.generate_sql(user_query)

            if generation_result["status"] != "success":
                wait = min(60, 5 * (2 ** retries))
                logger.warning(f"SQL generation failed, retrying in {wait}s ({retries + 1}/{MAX_RETRIES})")
                await asyncio.sleep(wait)
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

        return final_result

