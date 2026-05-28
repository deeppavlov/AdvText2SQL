"""Unit-тесты для serve/ и heal/ модулей (без сети, без БД)."""
from __future__ import annotations

import json
from pathlib import Path

from adv_text2sql.heal.collector import HealCollector
from adv_text2sql.serve.query_logger import QueryEvent, QueryLogger


# ─────────────────────────────────────────────────────────────────────────────
# QueryLogger
# ─────────────────────────────────────────────────────────────────────────────


def test_query_logger_appends_to_queries(tmp_path: Path) -> None:
    logger = QueryLogger(log_dir=tmp_path)
    event = QueryEvent(
        db_id="db1", question="hi", generated_sql="SELECT 1",
        status="success", latency_ms=10,
    )
    logger.log_query(event)

    queries = tmp_path / "queries.jsonl"
    assert queries.exists()
    line = json.loads(queries.read_text().splitlines()[0])
    assert line["db_id"] == "db1"
    assert line["status"] == "success"


def test_query_logger_duplicates_failed_to_failed_file(tmp_path: Path) -> None:
    """Failed-эвенты идут И в queries.jsonl, И в failed.jsonl."""
    logger = QueryLogger(log_dir=tmp_path)
    logger.log_query(QueryEvent(
        db_id="db", question="?", generated_sql="bad sql",
        status="error_execute", error_message="syntax error",
    ))

    assert (tmp_path / "queries.jsonl").exists()
    assert (tmp_path / "failed.jsonl").exists()
    failed_line = json.loads((tmp_path / "failed.jsonl").read_text().splitlines()[0])
    assert failed_line["status"] == "error_execute"


def test_query_logger_success_does_not_pollute_failed(tmp_path: Path) -> None:
    logger = QueryLogger(log_dir=tmp_path)
    logger.log_query(QueryEvent(status="success"))
    # failed.jsonl должен либо не существовать, либо быть пустым
    failed = tmp_path / "failed.jsonl"
    if failed.exists():
        assert failed.read_text() == ""


# ─────────────────────────────────────────────────────────────────────────────
# HealCollector
# ─────────────────────────────────────────────────────────────────────────────


def _write_failed_log(tmp_path: Path, records: list[dict]) -> None:
    """Helper: эмулируем failed.jsonl как пишет QueryLogger."""
    (tmp_path / "failed.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )


def test_heal_collector_dedups_exact(tmp_path: Path) -> None:
    """Одинаковые (db_id, question) — один кандидат."""
    _write_failed_log(tmp_path, [
        {"db_id": "x", "question": "How many?", "generated_sql": "...", "error_message": "e"},
        {"db_id": "x", "question": "How many?", "generated_sql": "...", "error_message": "e"},
        {"db_id": "x", "question": "Different?", "generated_sql": "...", "error_message": "e"},
    ])
    collector = HealCollector(log_dir=tmp_path, min_samples=1)
    result = collector.collect()
    assert result.total_failed == 3
    assert result.after_dedup_exact == 2


def test_heal_collector_filters_by_db_id(tmp_path: Path) -> None:
    _write_failed_log(tmp_path, [
        {"db_id": "db_a", "question": "q1", "generated_sql": ""},
        {"db_id": "db_b", "question": "q2", "generated_sql": ""},
        {"db_id": "db_a", "question": "q3", "generated_sql": ""},
    ])
    collector = HealCollector(log_dir=tmp_path, min_samples=1)
    result_a = collector.collect(db_id="db_a")
    assert len(result_a.candidates) == 2
    assert all(c.db_id == "db_a" for c in result_a.candidates)


def test_heal_collector_no_log_returns_empty(tmp_path: Path) -> None:
    collector = HealCollector(log_dir=tmp_path)
    result = collector.collect()
    assert result.total_failed == 0
    assert len(result.candidates) == 0


def test_heal_collector_respects_max_samples(tmp_path: Path) -> None:
    """Если failed > max_samples, отдаём последние."""
    _write_failed_log(tmp_path, [
        {"db_id": "x", "question": f"q{i}", "generated_sql": ""}
        for i in range(20)
    ])
    collector = HealCollector(log_dir=tmp_path, min_samples=1, max_samples=10)
    result = collector.collect()
    assert len(result.candidates) == 10
    # Берём последние 10 → должны быть q10..q19
    questions = [c.question for c in result.candidates]
    assert questions[0] == "q10"
    assert questions[-1] == "q19"
