"""
stats_collector — pure-functions версия column-statistics из Text2SQLGenerator.

Источник: `Text2SQLGenerator._get_table_row_count`, `_detect_column_types_for_table`,
`_compute_column_stats`, `_explore_column_statistics`, `_format_column_statistics`.

Здесь они переписаны как функции без `self.engine`, принимающие `Engine` явно.

ИНВАРИАНТ ФОРМАТА: вывод `format_column_statistics()` должен byte-identical
совпадать с `Text2SQLGenerator._format_column_statistics()` — иначе train==
inference дрифт.

PostgreSQL-only: regex-based type detection использует PG-specific операторы
(`::text ~ '^-?[0-9]+$'`). Для других диалектов нужно адаптировать запросы.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection, Engine

from .connector import DBConnector

logger = logging.getLogger("text2sql_tool.profiler.stats")


# ─────────────────────────────────────────────────────────────────────────────
# Row count
# ─────────────────────────────────────────────────────────────────────────────


def get_table_row_count(conn: Connection, table: str) -> int:
    """Сначала пытается из `pg_stat_user_tables` (быстро), fallback на COUNT(*).

    Pg-only: использует системную статистику PG, которая в ~1000 раз быстрее
    COUNT(*) для больших таблиц, но может быть устаревшей.
    """
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
        _safe_rollback(conn)

    try:
        res = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"'))
        return int(res.scalar() or 0)
    except Exception:
        logger.exception(f"COUNT(*) failed for table {table}")
        _safe_rollback(conn)
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Type detection
# ─────────────────────────────────────────────────────────────────────────────


def detect_column_types_for_table(
    conn: Connection,
    table: str,
    columns: list[dict],
    sample_size: int = 500,
) -> dict[str, dict]:
    """Определяет реальный тип данных колонки через regex-фильтры на sample.

    Возвращает {col_name: {detected_type, distinct_count, nonnull_count}}.

    `detected_type` ∈ {"integer", "numeric", "date", "json", "categorical", "empty"}.

    Эвристика: если ≥90% non-null значений матчатся под integer-regex → integer,
    иначе пробуем numeric → date → json, оставшееся = categorical.
    """
    if not columns:
        return {}

    select_parts = []
    for col in columns:
        name = col["name"]
        safe = name.replace('"', '""')
        select_parts.extend(
            [
                f'COUNT(*) FILTER (WHERE "{safe}" IS NOT NULL) AS "{safe}__nonnull"',
                f"COUNT(*) FILTER (WHERE \"{safe}\"::text ~ '^-?[0-9]+$') AS \"{safe}__integer\"",
                f"COUNT(*) FILTER (WHERE \"{safe}\"::text ~ '^-?[0-9]+(\\.[0-9]+)?$') AS \"{safe}__numeric\"",
                f"COUNT(*) FILTER (WHERE \"{safe}\"::text ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}') AS \"{safe}__date\"",
                f"COUNT(*) FILTER (WHERE \"{safe}\"::text ~ '^[\\[{{]') AS \"{safe}__json\"",
                f'COUNT(DISTINCT "{safe}") AS "{safe}__distinct"',
            ]
        )

    query = (
        f"SELECT {', '.join(select_parts)} "
        f'FROM (SELECT * FROM "{table}" ORDER BY RANDOM() LIMIT {sample_size}) s'
    )

    try:
        result = conn.execute(text(query))
        row = result.fetchone()
    except Exception:
        logger.exception(f"Column type detection failed for table {table}")
        _safe_rollback(conn)
        return {}

    if not row:
        return {}

    row_dict = dict(row._mapping)
    detected: dict[str, dict] = {}

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


# ─────────────────────────────────────────────────────────────────────────────
# Per-column statistics
# ─────────────────────────────────────────────────────────────────────────────


def compute_column_stats(
    conn: Connection,
    table: str,
    column: str,
    detected_type: str,
    row_count: int,
) -> dict | None:
    """Считает min/max/mean/median для numeric, top-5 для categorical, etc."""
    safe = column.replace('"', '""')

    try:
        if detected_type in ("integer", "numeric"):
            if row_count > 50000:
                res = conn.execute(
                    text(
                        f'SELECT MIN("{safe}"::numeric), MAX("{safe}"::numeric), '
                        f'ROUND(AVG("{safe}"::numeric), 2) '
                        f'FROM "{table}" '
                        f'WHERE "{safe}" IS NOT NULL '
                        f"AND \"{safe}\"::text ~ '^-?[0-9]+(\\.[0-9]+)?$'"
                    )
                )
                r = res.fetchone()
                if r:
                    return {
                        "type": detected_type,
                        "min": r[0], "max": r[1], "mean": r[2],
                    }
            else:
                res = conn.execute(
                    text(
                        f'SELECT MIN("{safe}"::numeric), MAX("{safe}"::numeric), '
                        f'ROUND(AVG("{safe}"::numeric), 2), '
                        f'PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "{safe}"::numeric) '
                        f'FROM "{table}" '
                        f'WHERE "{safe}" IS NOT NULL '
                        f"AND \"{safe}\"::text ~ '^-?[0-9]+(\\.[0-9]+)?$'"
                    )
                )
                r = res.fetchone()
                if r:
                    return {
                        "type": detected_type,
                        "min": r[0], "max": r[1], "mean": r[2], "median": r[3],
                    }

        elif detected_type == "date":
            res = conn.execute(
                text(
                    f'SELECT MIN("{safe}"::date), MAX("{safe}"::date) '
                    f'FROM "{table}" '
                    f'WHERE "{safe}" IS NOT NULL '
                    f"AND \"{safe}\"::text ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}'"
                )
            )
            r = res.fetchone()
            if r:
                return {"type": "date", "min": str(r[0]), "max": str(r[1])}

        elif detected_type == "categorical":
            res = conn.execute(
                text(
                    f'SELECT "{safe}", COUNT(*) AS freq '
                    f'FROM "{table}" WHERE "{safe}" IS NOT NULL '
                    f'GROUP BY "{safe}" ORDER BY freq DESC LIMIT 5'
                )
            )
            rows = res.fetchall()
            if rows:
                return {
                    "type": "categorical",
                    "top": [(str(r[0]), int(r[1])) for r in rows],
                }

        elif detected_type == "json":
            return {"type": "json"}

    except Exception:
        logger.exception(f"Stats computation failed for {table}.{column}")
        _safe_rollback(conn)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────


def explore_column_statistics(
    engine: Engine, schema: str = "public", sample_size: int = 500
) -> dict[str, Any]:
    """Оркестратор: собирает row counts, типы и статистики для всех таблиц.

    Возвращает {table_name: {row_count, columns: {col: {detected_type, distinct_count, stats}}}}.
    Идентичный формат с `Text2SQLGenerator._explore_column_statistics`.
    """
    inspector = inspect(engine)
    tables = inspector.get_table_names(schema=schema)
    result: dict[str, Any] = {}

    with engine.connect() as conn:
        for table in tables:
            try:
                row_count = get_table_row_count(conn, table)
                columns = inspector.get_columns(table, schema=schema)
                col_types = detect_column_types_for_table(
                    conn, table, columns, sample_size=sample_size
                )

                col_stats: dict[str, Any] = {}
                for col in columns:
                    name = col["name"]
                    info = col_types.get(name)
                    if not info or info["detected_type"] == "empty":
                        continue
                    stats = compute_column_stats(
                        conn, table, name,
                        info["detected_type"], row_count,
                    )
                    col_stats[name] = {
                        "detected_type": info["detected_type"],
                        "distinct_count": info["distinct_count"],
                        "stats": stats,
                    }

                result[table] = {"row_count": row_count, "columns": col_stats}
                logger.info(
                    f"Stats collected for table '{table}': "
                    f"{row_count} rows, {len(col_stats)} columns"
                )
            except Exception:
                logger.exception(f"Column statistics failed for table {table}")
                _safe_rollback(conn)

    logger.info(f"Column statistics: {len(result)}/{len(tables)} tables processed")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Formatter (byte-identical инвариант с Text2SQLGenerator._format_column_statistics)
# ─────────────────────────────────────────────────────────────────────────────


def format_column_statistics(stats: dict) -> str:
    """Форматирует статистики компактно для system prompt.

    КРИТИЧЕСКИЙ ИНВАРИАНТ: byte-identical с `Text2SQLGenerator._format_column_statistics`.
    """
    parts: list[str] = []

    for table, tinfo in stats.items():
        row_count = tinfo["row_count"]
        parts.append(f"TABLE {table} ({row_count:,} rows)")

        for col, cinfo in tinfo["columns"].items():
            dtype = cinfo["detected_type"]
            distinct = cinfo["distinct_count"]
            s = cinfo.get("stats")

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
                parts.append(f"  - {col}: date, range [{s['min']}, {s['max']}]")
            elif s["type"] == "categorical":
                top = s.get("top", [])
                if distinct <= 10:
                    vals = ", ".join(f'"{v}"' for v, _ in top)
                    parts.append(
                        f"  - {col}: categorical, {distinct} unique, values: {vals}"
                    )
                else:
                    top_str = ", ".join(f'"{v}"({cnt})' for v, cnt in top)
                    parts.append(
                        f"  - {col}: categorical, {distinct} unique, top: {top_str}"
                    )
            elif s["type"] == "json":
                parts.append(f"  - {col}: json/list")

        parts.append("")

    return "\n".join(parts).strip() if parts else "-- No statistics available"


# ─────────────────────────────────────────────────────────────────────────────
# Class wrapper
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StatsCollectionResult:
    column_stats: dict[str, Any]   # структурированно
    column_stats_str: str          # для system_prompt; byte-identical с генератором


class StatsCollector:
    """High-level API над explore_column_statistics + format_column_statistics."""

    def __init__(self, connector: DBConnector, sample_size: int = 500) -> None:
        self.connector = connector
        self.sample_size = sample_size

    def collect(self) -> StatsCollectionResult:
        stats = explore_column_statistics(
            self.connector.engine,
            schema=self.connector.default_schema,
            sample_size=self.sample_size,
        )
        return StatsCollectionResult(
            column_stats=stats,
            column_stats_str=format_column_statistics(stats),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _safe_rollback(conn: Connection) -> None:
    """Откатить транзакцию, ничего не делать если откат тоже падает."""
    try:
        conn.rollback()
    except Exception:
        pass
