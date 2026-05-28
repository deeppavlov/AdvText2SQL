"""
query_logger — структурированное логирование всех запросов и неудач.

В production-сервисе (vLLM или MCP-wrapper) подключается middleware'ом:
каждый запрос пользователя → один лог-event с question, generated_sql,
execution_result, latency, status.

Failed-events (где SQL не сгенерировался или execute упал) идут в отдельный
файл `failed.jsonl` — отсюда heal-collector их читает для retraining-цикла.

Формат событий — JSONL, чтобы можно было анализировать pandas/jq/etc.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger("text2sql_tool.serve.logger")


@dataclass
class QueryEvent:
    """Один лог-event для запроса."""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    db_id: str = ""
    question: str = ""
    generated_sql: str = ""
    status: str = "unknown"             # "success" | "error_parse" | "error_execute" | ...
    error_message: str | None = None
    row_count: int | None = None
    latency_ms: float | None = None

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class QueryLogger:
    """Thread-safe append-only writer для двух jsonl-файлов: queries и failed."""

    def __init__(self, log_dir: str | Path = "data/logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.queries_path = self.log_dir / "queries.jsonl"
        self.failed_path = self.log_dir / "failed.jsonl"
        self._lock = Lock()

    def log_query(self, event: QueryEvent) -> None:
        line = event.to_jsonl() + "\n"
        with self._lock:
            with self.queries_path.open("a", encoding="utf-8") as f:
                f.write(line)
            # Дублируем failed-эвенты в отдельный файл
            if event.status != "success":
                with self.failed_path.open("a", encoding="utf-8") as f:
                    f.write(line)

    def time_query(self):
        """Context manager: измеряет latency и автозаполняет latency_ms."""
        return _Timer(self)


class _Timer:
    """Helper для замера времени запроса."""

    def __init__(self, query_logger: QueryLogger) -> None:
        self.query_logger = query_logger
        self.started: float | None = None
        self.event: QueryEvent | None = None

    def __enter__(self) -> "_Timer":
        self.started = time.time()
        self.event = QueryEvent()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.event and self.started:
            self.event.latency_ms = (time.time() - self.started) * 1000
            if exc is not None and self.event.status == "unknown":
                self.event.status = "error_uncaught"
                self.event.error_message = str(exc)[:200]
            self.query_logger.log_query(self.event)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: глобальный singleton для простых сценариев
# ─────────────────────────────────────────────────────────────────────────────


_default_logger: QueryLogger | None = None


def get_default_logger() -> QueryLogger:
    global _default_logger
    if _default_logger is None:
        _default_logger = QueryLogger()
    return _default_logger
