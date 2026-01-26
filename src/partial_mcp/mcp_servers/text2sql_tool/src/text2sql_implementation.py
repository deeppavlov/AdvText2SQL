import logging
import os
import re
import sqlite3
from textwrap import dedent
from typing import Any

from autogen_core.models import SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
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
    def __init__(self, db_path: str, llm_client: OpenAIChatCompletionClient):
        """
        Initializes the Text2SQL generator with a pre-existing async LLM client.

        Args:
            db_path (str): Path to the SQLite database file.
            llm_client (openai.AsyncOpenAI): An initialized asynchronous OpenAI client.
            model_name (str): The name of the language model to use.
        """
        self.db_path = db_path
        self.llm_client = llm_client
        self.db_schema = self._get_db_schema()
        self.system_prompt = self._create_system_prompt()
        logger.info(f"Initialized with DB: {db_path}")

    # TODO: think if getting db_schema and system_prompt is actually just the build() function
    def build(self):
        """
        self.db_schema = self._get_db_schema()
        self.system_prompt = self._create_system_prompt()
        """
        pass

    def _update_db_schema(self, db_path):
        self.db_path = db_path
        self.db_schema = self._get_db_schema()
        self.system_prompt = self._create_system_prompt()

    def _get_db_schema(self) -> str:
        """
        Получает схему базы данных, включая CREATE TABLE statements и примеры данных,
        а также уникальные значения для столбцов aggregated_data с малым количеством уникальных значений.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Получаем все таблицы, кроме системных
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tables = cursor.fetchall()

        schema_info_parts = []

        # Список для хранения описаний уникальных значений, которые будут добавлены в самом конце
        unique_values_descriptions = []

        for table_name_tuple in tables:
            table_name = table_name_tuple[0]

            # 1. Получаем CREATE TABLE statement
            # cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
            cursor.execute(
                f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';"
            )
            create_table_sql = cursor.fetchone()[0]
            schema_info_parts.append(create_table_sql)

            # 2. Получаем примеры строк в зависимости от имени таблицы
            try:
                limit_rows = 3  # Значение по умолчанию для других таблиц
                if table_name == "tables_info":
                    limit_rows = None  # Без ограничения, выбираем все строки
                elif table_name == "aggregated_data":
                    limit_rows = 5  # Первые 5 строк для aggregated_data

                if limit_rows is None:
                    # cursor.execute("SELECT * FROM ?;", (table_name,))
                    cursor.execute(f'SELECT * FROM "{table_name}";')
                else:
                    # cursor.execute("SELECT * FROM ? LIMIT ?;", (table_name, limit_rows))
                    cursor.execute(f'SELECT * FROM "{table_name}" LIMIT {limit_rows};')

                sample_rows = cursor.fetchall()

                if sample_rows:
                    # Получаем имена столбцов
                    column_names = [
                        description[0] for description in cursor.description
                    ]

                    # Форматируем примеры данных
                    sample_data_str = "/*\n"
                    sample_data_str += "\t".join(column_names) + "\n"
                    for row in sample_rows:
                        sample_data_str += "\t".join(map(str, row)) + "\n"
                    sample_data_str += "*/"
                    schema_info_parts.append(sample_data_str)

            except sqlite3.Error:
                logger.exception(f"Could not fetch sample data for table {table_name}.")

            schema_info_parts.append(
                "\n\n"
            )  # Добавляем пустые строки между таблицами для читаемости

        # --- Дополнительная логика: подсчет уникальных значений для aggregated_data ---
        # Проверяем, существует ли таблица aggregated_data
        if "aggregated_data" in [t[0] for t in tables]:
            try:
                # Получаем имена столбцов для таблицы aggregated_data
                cursor.execute("PRAGMA table_info(aggregated_data);")
                agg_data_columns_info = cursor.fetchall()
                # Исключаем столбцы 'id' и 'table_id', так как они обычно не требуют перечисления уникальных значений
                agg_data_column_names = [
                    col_info[1]
                    for col_info in agg_data_columns_info
                    if col_info[1] not in ["id", "table_id"]
                ]

                for col_name in agg_data_column_names:
                    # Выбираем уникальные значения для текущего столбца
                    cursor.execute(
                        f'SELECT DISTINCT "{col_name}" FROM "aggregated_data";'
                    )
                    unique_values = [row[0] for row in cursor.fetchall()]

                    # Если количество уникальных значений меньше 15, добавляем их описание
                    if len(unique_values) < 15:  # TODO: расхардкодить
                        # Форматируем список уникальных значений, заключая каждое в одинарные кавычки
                        formatted_values = [f"'{str(val)}'" for val in unique_values]
                        unique_values_descriptions.append(
                            f'Возможные значения атрибута "{col_name}": [{", ".join(formatted_values)}].'
                        )
            except sqlite3.Error:
                logger.exception("Could not fetch unique values for aggregated_data.")

        # Добавляем описания уникальных значений в самом конце схемы
        if unique_values_descriptions:
            schema_info_parts.append(
                "\n        ### Дополнительная информация об уникальных значениях --\n"
            )
            schema_info_parts.extend(unique_values_descriptions)
            schema_info_parts.append(
                "\n"
            )  # Добавляем пустую строку после этого раздела

        conn.close()
        return "\n".join(
            schema_info_parts
        ).strip()  # Убираем лишние пустые строки в конце

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
            parsed = parse_one(sql, dialect=Dialects.SQLITE)

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

    def _get_accessed_tables(self, sql: str) -> list:
        """Определяет какие таблицы затрагивает запрос"""
        try:
            parsed = parse_one(sql)
            return list({table.name for table in parsed.find_all(exp.Table)})
        except Exception:
            return []

    def execute_safe(self, sql: str) -> dict[str, Any]:
        """
        Безопасное выполнение SQL запроса (только SELECT)

        Args:
            sql (str): SQL запрос

        Returns:
            Dict[str, Any]: Результаты выполнения
        """
        try:
            # sanitized_sql = self._sanitize_sql(sql) # Сделать нормально
            sanitized_sql = sql
            if not self._validate_sql(sanitized_sql):
                raise ValueError("Запрос не прошел валидацию")

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(sanitized_sql)
                results = [dict(row) for row in cursor.fetchall()]
                if not len(results):
                    raise ValueError("Запрос вернулся пустым")

                return {
                    "status": "success",
                    "results": results,
                    "columns": list(results[0].keys()) if results else [],
                    "row_count": len(results),
                    "sql_executed": sanitized_sql,
                }
        except Exception as e:
            logger.exception(f"Failed to execute query {sql}.")
            return {"status": "error", "error": str(e), "sql_attempted": sql}

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

        return raw_sql
