"""
profiler/cli.py — реализация `text2sql profile`.

Прогоняет 4 стадии последовательно:

    DBConnector → SchemaExtractor → StatsCollector → SampleCollector → Profile

Артефакт: `<out_dir>/<db_id>/profile.json`.

Безопасно к запуску повторно (idempotent) — перезаписывает файл целиком.
"""
from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console

from .connector import DBConnector
from .profile import Profile
from .sample_collector import SampleCollector
from .schema_extractor import SchemaExtractor
from .stats_collector import StatsCollector

console = Console()


def run_profile(
    db_url: str,
    out_dir: str = "data/profiles",
    sample_size: int = 500,
    skip_stats: bool = False,
) -> Path:
    """Полный профилинг БД → запись profile.json. Возвращает путь к файлу.

    Args:
        db_url: PostgreSQL connection string.
        out_dir: куда писать; будет создан подкаталог `<db_id>/`.
        sample_size: сколько строк семплировать для type detection (Week 1 day 3).
        skip_stats: пропустить column statistics (для быстрого smoke-теста).
    """
    started = time.time()

    with DBConnector(db_url) as connector:
        console.print(
            f"[bold cyan]Profiling[/bold cyan] {connector.redacted_uri()}\n"
            f"  db_id  = {connector.db_id}\n"
            f"  dialect = {connector.dialect}"
        )

        # ── Stage 1: Schema ─────────────────────────────────────────────────
        console.print("\n[bold]Stage 1/3:[/bold] schema extraction…")
        s_start = time.time()
        schema = SchemaExtractor(connector).extract()
        console.print(
            f"  → {len(schema.tables)} tables, "
            f"{sum(len(c) for c in schema.column_info.values())} columns, "
            f"{len(schema.relationships['foreign_keys'])} FKs "
            f"({time.time() - s_start:.1f}s)"
        )

        # ── Stage 2: Statistics ─────────────────────────────────────────────
        if skip_stats:
            console.print("\n[bold]Stage 2/3:[/bold] [yellow]skipped[/yellow] (--skip-stats)")
            stats_result = None
        else:
            console.print("\n[bold]Stage 2/3:[/bold] column statistics…")
            s_start = time.time()
            stats_result = StatsCollector(connector, sample_size=sample_size).collect()
            console.print(
                f"  → stats for {len(stats_result.column_stats)} tables "
                f"({time.time() - s_start:.1f}s)"
            )

        # ── Stage 3: Samples ────────────────────────────────────────────────
        console.print("\n[bold]Stage 3/3:[/bold] sample rows + low-cardinality values…")
        s_start = time.time()
        samples = SampleCollector(connector).collect()
        console.print(
            f"  → samples for {len(samples.sample_rows)} tables, "
            f"{len(samples.low_cardinality_values)} low-cardinality columns "
            f"({time.time() - s_start:.1f}s)"
        )

        # ── Assemble Profile ────────────────────────────────────────────────
        profile = Profile(
            db_id=connector.db_id,
            db_uri_redacted=connector.redacted_uri(),
            dialect=connector.dialect,
            tables=schema.tables,
            column_info=schema.column_info,
            relationships=dict(schema.relationships),
            schema_str=schema.schema_str,
            relationships_str=schema.relationships_str,
            column_stats_str=stats_result.column_stats_str if stats_result else "",
            column_stats=stats_result.column_stats if stats_result else {},
            sample_rows=samples.sample_rows,
            low_cardinality_values=samples.low_cardinality_values,
        )
        profile.compute_complexity_score()

        # ── Write ───────────────────────────────────────────────────────────
        out_path = profile.write_json(out_dir)

    elapsed = time.time() - started
    console.print(
        f"\n[bold green]✓ Profile written[/bold green] to {out_path}\n"
        f"  complexity_score = {profile.complexity_score} "
        f"→ adaptive target_count = {profile.target_synthetic_count()}\n"
        f"  total time = {elapsed:.1f}s"
    )

    return out_path
