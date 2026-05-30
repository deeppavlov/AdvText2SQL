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

import asyncio
import logging
import os
from typing import Annotated, Any

from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import Field
from tabulate import tabulate

from adv_text2sql.serve.query_logger import QueryEvent, QueryLogger

from .src.generate_tool_description import generate_description_text2sql
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

# Tool-description генерируется один раз
generated_description = asyncio.run(
    generate_description_text2sql(
        tool_description=(
            "Генератор SQL-запросов на естественном языке с контролем безопасности."
        ),
        text2sql_agent=text2sql_agent,
    )
)
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
            result = await text2sql_agent.query(
                user_query_text, check_ambiguity=False, check_sql_query=False
            )
        except Exception as e:
            event.status = "error_uncaught"
            event.error_message = str(e)[:200]
            return f"_Ошибка_: {e}"

        # Распаковка ответа Text2SQLGenerator.query()
        root = result.get("params", result) if isinstance(result, dict) else {}
        details = root.get("data", {}).get("details", {})
        exec_info = details.get("execution", {})
        event.generated_sql = details.get("generated_sql") or root.get("query", "") or ""

        if exec_info.get("status") == "success":
            data = exec_info.get("results", [])
            event.status = "success"
            event.row_count = len(data) if isinstance(data, list) else None
            return tabulate(data, headers="keys", tablefmt="pipe") if data else "Нет данных"

        # Не-success ветка
        event.status = exec_info.get("status") or "error"
        event.error_message = (
            details.get("error") or root.get("message") or "Выполнение запроса неуспешно"
        )
        return f"_Ошибка_: {event.error_message}"


def _db_id_from_uri(uri: str) -> str:
    from urllib.parse import urlparse
    return urlparse(uri).path.lstrip("/") or "unknown"


logger.info("Tool description: %s", text2sql.description)


if __name__ == "__main__":
    port = int(os.environ.get("MCP_PORT", "8000"))
    server.run(transport="http", host="0.0.0.0", port=port, show_banner=False)
