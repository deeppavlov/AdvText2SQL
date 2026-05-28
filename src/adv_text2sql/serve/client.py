"""
Text2SQLClient — Python SDK для inference-сервиса.

Архитектура:

    user → Text2SQLClient.query("Сколько карт типа Creature?")
         → POST /v1/chat/completions с system_prompt (из profile.json) + question
         → LLM возвращает SQL
         → SQL валидируется, опционально исполняется через DBConnector
         → результат: {sql, results, error}

SDK переиспользует SYSTEM_PROMPT_TEMPLATE из mcp_servers/text2sql_tool/src/prompts.py
— тот же что в обучении. Это гарантирует train==inference инвариант на стороне
клиента.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from adv_text2sql.mcp_servers.text2sql_tool.src.prompts import (
    SQL_PROMPT_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE,
)

from ..profiler.connector import DBConnector
from ..profiler.profile import Profile
from ..synth.validator import check_parse, check_whitelist

logger = logging.getLogger("text2sql_tool.serve.client")


@dataclass
class QueryResult:
    sql: str
    results: list[dict[str, Any]] | None = None
    columns: list[str] | None = None
    error: str | None = None
    latency_ms: float | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)

    def success(self) -> bool:
        return self.error is None


class Text2SQLClient:
    """Клиент для inference через vLLM/LM Studio/любой OpenAI-compatible сервер.

    Использование:
        profile = Profile.load_json("data/profiles/card_games/profile.json")
        client = Text2SQLClient(
            profile=profile,
            api_url="http://localhost:8000/v1",
            model_name="text2sql",          # имя LoRA-адаптера в vLLM
        )
        result = client.query("Сколько карт типа Creature?")
        print(result.sql)
        print(result.results)
    """

    def __init__(
        self,
        profile: Profile,
        api_url: str = "http://localhost:8000/v1",
        model_name: str = "text2sql",
        api_key: str = "dummy",
        execute_db_url: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        self.profile = profile
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, base_url=api_url)
        self.execute_db_url = execute_db_url
        self._system_prompt = self._build_system_prompt()

    def query(self, question: str, execute: bool = True) -> QueryResult:
        """NL-вопрос → SQL → (опц.) execute → результат.

        execute=False вернёт только SQL без подключения к БД.
        """
        import time

        started = time.time()

        # 1. Генерируем SQL через LLM
        try:
            sql = self._call_llm(question)
        except Exception as e:
            return QueryResult(sql="", error=f"llm_error: {e}")

        # 2. Валидация
        ok, reason = check_parse(sql)
        if not ok:
            return QueryResult(sql=sql, error=f"parse_error: {reason}")
        ok, reason = check_whitelist(sql)
        if not ok:
            return QueryResult(sql=sql, error=f"whitelist_error: {reason}")

        # 3. Опциональное исполнение
        results = None
        columns = None
        error = None
        if execute and self.execute_db_url:
            try:
                results, columns = self._execute(sql)
            except Exception as e:
                error = f"execute_error: {e}"

        return QueryResult(
            sql=sql,
            results=results,
            columns=columns,
            error=error,
            latency_ms=(time.time() - started) * 1000,
        )

    # ── Internals ────────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Инвариант train==inference: используем тот же SYSTEM_PROMPT_TEMPLATE."""
        return SYSTEM_PROMPT_TEMPLATE.format(
            db_schema=self.profile.schema_str,
            db_relationships=self.profile.relationships_str,
            column_statistics=self.profile.column_stats_str,
            sql_dialect="PostgreSQL",
        )

    def _call_llm(self, question: str) -> str:
        user_content = SQL_PROMPT_TEMPLATE.format(
            user_query=question, sql_dialect="PostgreSQL"
        )
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        content = resp.choices[0].message.content or ""
        return _strip_sql_fences(content)

    def _execute(
        self, sql: str
    ) -> tuple[list[dict[str, Any]], list[str]]:
        from sqlalchemy import text

        with DBConnector(self.execute_db_url) as conn:
            with conn.engine.connect() as db_conn:
                result = db_conn.execute(text(sql))
                rows = result.fetchall()
                columns = list(result.keys())
                return [dict(zip(columns, row)) for row in rows], columns


def _strip_sql_fences(text: str) -> str:
    """Снять ```sql ... ``` обёртку если есть."""
    s = text.strip()
    if s.startswith("```"):
        # ```sql\n...\n```
        lines = s.splitlines()
        # убираем первую строку (```sql) и последнюю если она ```
        body = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        s = "\n".join(body)
    return s.strip()
