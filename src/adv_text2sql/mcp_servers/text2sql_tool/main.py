"""
FastMCP-обёртка над Text2SQLGenerator.

Поддерживает два режима старта:

  1. **Cold start с обращением в БД** (default):
     - Требует DB_USER, DB_PASS, BENCHMARK_DB_URL, BENCHMARK_DB_NAME
     - Text2SQLGenerator.build() → ~30s на извлечение schema/stats из PG

  2. **Hot start с pre-computed profile** (для production):
     - Требует TEXT2SQL_PROFILE_PATH=data/profiles/<db_id>/profile.json
     - Text2SQLGenerator.load_from_profile() → <50ms
     - Профиль создаётся заранее через `text2sql profile --db-url ...`
     - БД нужна только для исполнения SQL в tool — credentials через
       BENCHMARK_DB_URL/BENCHMARK_DB_NAME (или TEXT2SQL_DB_URL напрямую)

Каждый вызов tool логируется через `QueryLogger` в `data/logs/`. Failed-events
дублируются в `failed.jsonl` — оттуда `text2sql heal` их подбирает для retraining.
"""
from __future__ import annotations

import logging
import os
from typing import Annotated

from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import Field
from tabulate import tabulate

from adv_text2sql.serve.query_logger import QueryEvent, QueryLogger

from .src.text2sql_implementation import Text2SQLGenerator

load_dotenv()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config from env
# ─────────────────────────────────────────────────────────────────────────────


def _build_db_uri() -> str:
    """Собрать PG URI из env vars. Поддерживает прямой TEXT2SQL_DB_URL."""
    if direct := os.environ.get("TEXT2SQL_DB_URL"):
        return direct
    db_user = os.environ["DB_USER"]
    db_pass = os.environ["DB_PASS"]
    db_url = os.environ.get("BENCHMARK_DB_URL", "localhost:5444")
    db_name = os.environ.get("BENCHMARK_DB_NAME", "card_games")
    return f"postgresql+psycopg://{db_user}:{db_pass}@{db_url}/{db_name}"


def _build_llm_client() -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
        model=os.environ["LLM_MODEL_NAME"],
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.6")),
        model_info={
            "json_output": False,
            "function_calling": True,
            "vision": False,
            "family": "unknown",
            "structured_output": False,
            "context_window": int(os.environ.get("LLM_CONTEXT_WINDOW", "32768")),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap: build OR load_from_profile
# ─────────────────────────────────────────────────────────────────────────────


server = FastMCP("text2sql_tool_server")

db_uri = _build_db_uri()
llm_client = _build_llm_client()
query_logger = QueryLogger(log_dir=os.environ.get("TEXT2SQL_LOG_DIR", "data/logs"))

logger.info("Инициализация Text2SQLGenerator...")
text2sql_agent = Text2SQLGenerator(db_uri=db_uri, llm_client=llm_client)

profile_path = os.environ.get("TEXT2SQL_PROFILE_PATH")
if profile_path:
    logger.info(f"Hot start: загружаю профиль из {profile_path}")
    text2sql_agent.load_from_profile(profile_path)
    logger.info("Hot start завершён за <50ms")
else:
    logger.info("Cold start: build() с обращением в БД (~30s)...")
    text2sql_agent.build()
    logger.info("Cold start завершён")

def _build_tool_description(agent: Text2SQLGenerator) -> str:
    """Статическое описание tool'а (без LLM-вызова на старте).

    Раньше описание генерировал отдельный LLM при импорте модуля (через
    DESC_LLM_* env) — это давало лишнюю сетевую зависимость и хрупкий boot.
    Описание схемо-осведомлённое: подставляем список таблиц из агента.
    """
    import re

    # db_schema — строка вида "TABLE cards\n  - id ...\nTABLE sets\n ...".
    # Имена таблиц достаём из неё (отдельного списка tables у агента нет).
    schema = getattr(agent, "db_schema", "") or ""
    table_names = re.findall(r"(?m)^TABLE\s+(\w+)", schema)
    tables = ", ".join(table_names) or "—"
    return (
        "## Приоритет вызова\nСредний\n\n"
        "## Описание\n"
        "Генерирует SQL-запрос из естественно-языкового вопроса к целевой БД "
        "с контролем безопасности (только SELECT) и исполняет его.\n"
        f"Доступные таблицы: {tables}.\n\n"
        "## Условия вызова\n"
        "- Запрос должен быть однозначным, термины — расшифрованы.\n"
        "- Должны быть заданы параметры/значения для поиска в БД.\n\n"
        "## Возвращает\n"
        "- str: таблица результатов в формате markdown, либо текст ошибки."
    )


# Tool-description собирается один раз на старте (статически, без LLM).
generated_description = _build_tool_description(text2sql_agent)
logger.debug(f"Системная инструкция:\n\n{text2sql_agent.system_prompt[:500]}...")


# ─────────────────────────────────────────────────────────────────────────────
# MCP tool with QueryLogger middleware
# ─────────────────────────────────────────────────────────────────────────────


@server.tool(description=generated_description)
async def text2sql(
    user_query_text: Annotated[
        str,
        Field(
            description=(
                "Текстовый запрос (напр. 'все отчёты по западному округу за первый квартал')"
            )
        ),
    ],
) -> str | None:
    """Вход: NL → выход: markdown-таблица результатов. Логирование embedded."""
    with query_logger.time_query() as t:
        event = QueryEvent(
            db_id=_db_id_from_uri(text2sql_agent.db_uri),
            question=user_query_text,
        )
        t.event = event

        try:
            result = await text2sql_agent.query(user_query_text)
        except Exception as e:
            event.status = "error_uncaught"
            event.error_message = str(e)[:200]
            return f"_Ошибка_: {e}"

        # query() возвращает плоский dict: {"status": "success", "query": sql}
        # либо {"status": "ambiguous"} / {"status": "error"}.
        status = result.get("status") if isinstance(result, dict) else None

        if status == "ambiguous":
            event.status = "ambiguous"
            event.error_message = "Запрос неоднозначен"
            return "_Неоднозначный запрос_: уточните формулировку."

        if status != "success":
            event.status = "error"
            event.error_message = "Не удалось сгенерировать SQL"
            return "_Ошибка_: не удалось сгенерировать SQL по запросу."

        sql = result.get("query", "") or ""
        event.generated_sql = sql

        # query() только генерирует SQL — исполняем отдельно (нужен доступ к БД).
        exec_res = text2sql_agent.execute_safe(sql)
        if exec_res.get("status") == "success":
            rows = exec_res.get("results", [])
            event.status = "success"
            event.row_count = exec_res.get("row_count")
            return tabulate(rows, headers="keys", tablefmt="pipe") if rows else "Нет данных"

        # SQL сгенерирован, но исполнение упало (напр. нет доступа к БД) —
        # всё равно отдаём SQL: tool полезен и без живого подключения.
        event.status = "error_execute"
        event.error_message = exec_res.get("error", "Выполнение запроса неуспешно")
        return (
            "SQL сгенерирован, но исполнение не удалось "
            f"(нет доступа к БД?).\n\n```sql\n{sql}\n```\n\n"
            f"_Ошибка_: {event.error_message}"
        )


def _db_id_from_uri(uri: str) -> str:
    from urllib.parse import urlparse
    return urlparse(uri).path.lstrip("/") or "unknown"


logger.info("Tool description: %s", text2sql.description)


if __name__ == "__main__":
    port = int(os.environ.get("MCP_PORT", "8000"))
    server.run(transport="http", host="0.0.0.0", port=port, show_banner=False)
