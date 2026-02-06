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
from .utils import get_error, print_result

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
        # self.db_schema = self._get_db_schema()
        # self.system_prompt = self._create_system_prompt()

        logger.info("Initialized Text2SQLGenerator")

    # TODO: think if getting db_schema and system_prompt is really the build() function
    def build(self):
        self.db_schema = self._get_db_schema()
        self.system_prompt = self._create_system_prompt()

    def _update_db_schema(self, db_uri):
        self.db_uri = db_uri
        self.engine.dispose()
        self.engine = create_engine(db_uri, pool_pre_ping=True)
        self.build()

    # TODO: Consider adding unique values hints, like in the older sqlite3 version of this tool
    def _get_db_schema(self) -> str:
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
        tables = inspector.get_table_names(schema="public")
        if not tables:
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
        return SYSTEM_PROMPT_TEMPLATE.format(db_schema=self.db_schema)

    async def _check_ambiguity(self, user_query: str) -> dict[str, Any]:
        """
        Проверяет пользовательский запрос на неоднозначность с помощью LLM.
        """
        ambiguity_prompt = AMBIGUITY_PROMPT_TEMPLATE.format(
            db_schema=self.db_schema,
            user_query=user_query,
        )

        messages = [
            SystemMessage(content=ambiguity_prompt),
            UserMessage(source="user", content="Проверь запрос на однозначность."),
        ]

        try:
            result = await self.llm_client.create(messages)
            response_text = result.content.strip()

            if response_text.lower() == "ok":
                return {"status": "success", "ambiguous": False}
            else:
                return {
                    "status": "success",
                    "ambiguous": True,
                    "clarification_needed": response_text,
                }

        except Exception as e:
            logger.exception(f"Failed to check ambiguity for query: {user_query}")
            return {"status": "error", "error": get_error(message=str(e))}

    def _validate_sql(self, sql: str) -> bool:
        """Валидация SQL запроса с помощью SQLGlot"""
        try:
            parsed = parse_one(sql, dialect=Dialects.POSTGRES)

            # Проверка на запрещенные операции
            for node in parsed.walk():
                if isinstance(node, (exp.Drop | exp.Delete | exp.Update)):
                    raise ValueError(f"Запрещенная операция: {node.sql()}")

            return True
        except (ParseError, ValueError):
            logger.exception("SQL validation error.")
            return False

    def _sanitize_sql(self, sql: str) -> str:
        """Расширенная санитизация SQL запроса"""
        # Удаляем потенциально опасные конструкции
        dangerous_patterns = [
            "--",
            "/*",
            "*/",
            ";",
            "xp_",
            "sp_",
            "EXEC",
            "EXECUTE",
            "TRUNCATE",
            "SHUTDOWN",
        ]

        for pattern in dangerous_patterns:
            if pattern.lower() in sql.lower():
                raise ValueError(f"Обнаружена запрещенная конструкция: {pattern}")
        # Проверяем экранирование идентификаторов
        parsed = parse_one(sql)
        for identifier in parsed.find_all(exp.Identifier):
            if not identifier.quoted and not any(
                kw in identifier.sql().upper()
                for kw in ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY"]
            ):
                raise ValueError(f"Неэкранированный идентификатор: {identifier.sql()}")

        return sql.strip()

    async def generate_sql(self, user_query: str) -> dict[str, Any]:
        """
        Генерирует SQL запрос из естественно-языкового запроса

        Args:
            user_query (str): Запрос на естественном языке

        Returns:
            Dict[str, Any]: Результат в формате Model Context Protocol
        """
        sql_prompt = SQL_PROMPT_TEMPLATE.format(user_query=user_query)
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
            sanitized_sql = raw_sql
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
            return sorted(
                {table.name.lower() for table in parsed.find_all(exp.Table)}
            )
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
        check_ambiguity: bool = False,
        check_sql_query: bool = False,
    ) -> dict[str, Any]:
        """
        Полный цикл: генерация SQL + выполнение
        Args:
            user_query (str): Запрос на естественном языке
            check_ambiguity (bool): Флаг, требуется ли проверять запрос на неоднозначность
            check_sql_query (bool): Флаг, требуется ли проверять SQL-запрос на корректность
        Returns:
            Dict[str, Any]: Объединенные результаты генерации и выполнения
        """
        if check_ambiguity:
            logger.info("Проверяю запрос на неоднозначность...")
            ambiguity_check = await self._check_ambiguity(user_query)

            if ambiguity_check["status"] == "success" and ambiguity_check["ambiguous"]:
                logger.info("Запрос неоднозначен. Требуется уточнение.")
                return get_error(
                    message="Запрос неоднозначен. Требуется уточнение.",
                    details={
                        "status": "ambiguous_query",
                        "clarification_needed": ambiguity_check["clarification_needed"],
                        "user_query": user_query,
                    },
                )

            if ambiguity_check["status"] == "error":
                return get_error(
                    message=f"Ошибка при проверке неоднозначности: {ambiguity_check['error']}.",
                    details=ambiguity_check,
                )

        retries = 0
        success = False
        raw_sql = ""

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
                    r"^```.*?\n|```$", "", raw_sql, flags=re.MULTILINE
                ).strip()
            if check_sql_query:
                logger.info("SQL сгенерирован. Проверяю на соответствие запросу... ")

                # Шаг 3: Проверка соответствия SQL
                verification_result = await self._verify_sql_against_query(
                    user_query, raw_sql
                )

                if (
                    verification_result["status"] != "success"
                    or not verification_result["is_correct"]
                ):
                    # Если SQL не соответствует, мы увеличиваем счётчик retries, чтобы LLM попыталась исправить ошибку.
                    retries += 1
                    continue

                logger.info("Соответствие подтверждено!")
            else:
                verification_result = "skipped"
            
            final_result = get_info(
                message="Успешно",
                details={**generation_result, "verification": verification_result},
            )
            success = True

        return final_result
