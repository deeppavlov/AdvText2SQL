"""
sample_collector — извлечение sample rows + low-cardinality values из БД.

Эти данные нужны в Week 2 для **value-grounding** при генерации синтетики:
без них template-генератор и LLM создают абстрактные вопросы про
гипотетические значения колонок, а с ними — вопросы про РЕАЛЬНО существующие
значения, на которых SQL гарантированно вернёт ≥1 строку.

Источник: часть логики `Text2SQLGenerator._get_db_schema_heavy()`, переписанная
как standalone-функции без зависимости от self.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection, Engine

from .connector import DBConnector

logger = logging.getLogger("text2sql_tool.profiler.sample")


def collect_sample_rows(
    conn: Connection,
    table: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """3-5 строк из таблицы для grounding LLM-генератора.

    Возвращает [{col1: val, col2: val, ...}, ...]. JSON-safe — все значения
    приведены к строкам (json-сериализация может ломаться на datetime/Decimal).
    """
    try:
        result = conn.execute(
            text(f'SELECT * FROM "{table}" LIMIT :limit'),
            {"limit": limit},
        )
        rows = result.fetchall()
        col_names = list(result.keys())
        return [
            {col: _json_safe(val) for col, val in zip(col_names, row)}
            for row in rows
        ]
    except Exception:
        logger.exception(f"Could not fetch sample rows for table {table}")
        _safe_rollback(conn)
        return []


def collect_low_cardinality_values(
    conn: Connection,
    table: str,
    columns: list[dict],
    threshold: int = 10,
) -> dict[str, list[Any]]:
    """Для каждой колонки с distinct count ≤ threshold — список значений.

    Это enum-like колонки (статусы, категории, типы) — на них основаны фильтры
    в template-генераторе. Возвращает {col_name: [val1, val2, ...]}.
    """
    out: dict[str, list[Any]] = {}

    for col in columns:
        col_name = col["name"]
        try:
            res = conn.execute(
                text(
                    f'SELECT DISTINCT "{col_name}" '
                    f'FROM "{table}" WHERE "{col_name}" IS NOT NULL'
                )
            )
            values = [r[0] for r in res.fetchall()]
            if 0 < len(values) <= threshold:
                out[col_name] = [_json_safe(v) for v in values]
        except Exception:
            # Молча пропускаем сложные типы (JSON, ARRAY): для них regex-DISTINCT падает
            _safe_rollback(conn)
            continue

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Class wrapper
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SampleCollectionResult:
    sample_rows: dict[str, list[dict[str, Any]]]      # {table: [{col: val}]}
    low_cardinality_values: dict[str, list[Any]]      # {"table.col": [v1, v2, ...]}


class SampleCollector:
    def __init__(
        self,
        connector: DBConnector,
        sample_rows_per_table: int = 3,
        low_cardinality_threshold: int = 10,
    ) -> None:
        self.connector = connector
        self.sample_rows_per_table = sample_rows_per_table
        self.low_cardinality_threshold = low_cardinality_threshold

    def collect(self) -> SampleCollectionResult:
        engine: Engine = self.connector.engine
        inspector = inspect(engine)
        schema = self.connector.default_schema
        tables = inspector.get_table_names(schema=schema)

        sample_rows: dict[str, list[dict[str, Any]]] = {}
        low_card: dict[str, list[Any]] = {}

        with engine.connect() as conn:
            for table in tables:
                sample_rows[table] = collect_sample_rows(
                    conn, table, limit=self.sample_rows_per_table
                )

                columns = inspector.get_columns(table, schema=schema)
                table_low_card = collect_low_cardinality_values(
                    conn, table, columns, threshold=self.low_cardinality_threshold
                )
                for col_name, values in table_low_card.items():
                    low_card[f"{table}.{col_name}"] = values

        logger.info(
            f"Sampling done: {len(sample_rows)} tables, "
            f"{len(low_card)} low-cardinality columns"
        )

        return SampleCollectionResult(
            sample_rows=sample_rows,
            low_cardinality_values=low_card,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _json_safe(val: Any) -> Any:
    """Конвертирует не-JSON-сериализуемые значения в строки."""
    if val is None:
        return None
    if isinstance(val, (str, int, float, bool)):
        return val
    return str(val)


def _safe_rollback(conn: Connection) -> None:
    try:
        conn.rollback()
    except Exception:
        pass
