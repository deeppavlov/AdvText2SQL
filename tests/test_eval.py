"""Unit-тесты для eval/ модулей (без сети, без БД)."""
from __future__ import annotations

import json

import pytest

from adv_text2sql.eval.reporter import render_markdown
from adv_text2sql.eval.runner import EvalItem, EvalReport, load_bird_subset


@pytest.fixture()
def bird_json(tmp_path):
    path = tmp_path / "bird.json"
    records = [
        {"question_id": 1, "db_id": "card_games", "question": "Q1", "SQL": "SELECT 1", "difficulty": "simple", "evidence": ""},
        {"question_id": 2, "db_id": "card_games", "question": "Q2", "SQL": "SELECT 2", "difficulty": "moderate", "evidence": ""},
        {"question_id": 3, "db_id": "other_db", "question": "Q3", "SQL": "SELECT 3", "difficulty": "simple", "evidence": ""},
    ]
    path.write_text(json.dumps(records))
    return path


def test_load_bird_subset_filters_by_db_id(bird_json):
    items = load_bird_subset(bird_json, db_id="card_games")
    assert len(items) == 2
    assert all(i.db_id == "card_games" for i in items)


def test_load_bird_subset_respects_limit(bird_json):
    items = load_bird_subset(bird_json, limit=1)
    assert len(items) == 1


def test_load_bird_subset_returns_all_when_no_filter(bird_json):
    items = load_bird_subset(bird_json)
    assert len(items) == 3


def test_eval_report_summary_aggregates_correctly():
    items = [
        EvalItem("1", "x", "Q", "g", "p", "simple", score=True, latency_ms=10),
        EvalItem("2", "x", "Q", "g", "p", "simple", score=False, latency_ms=20, error="bad"),
        EvalItem("3", "x", "Q", "g", "p", "moderate", score=True, latency_ms=15),
    ]
    report = EvalReport(
        config={"db_id": "x"},
        items=items,
        overall_accuracy=2 / 3,
        accuracy_by_difficulty={"simple": 0.5, "moderate": 1.0},
        error_count=1,
        total_latency_s=0.045,
        started_at="2025-01-01",
    )
    s = report.summary()
    assert s["overall_accuracy_pct"] == 66.67
    assert s["by_difficulty"]["simple"] == 50.0
    assert s["by_difficulty"]["moderate"] == 100.0
    assert s["error_count"] == 1


def test_render_markdown_includes_key_sections():
    items = [
        EvalItem("1", "card_games", "What?", "SELECT 1", "SELECT 2", "simple", score=False, error="mismatch"),
    ]
    report = EvalReport(
        config={"db_id": "card_games", "model_name": "test", "api_url": "http://x"},
        items=items,
        overall_accuracy=0.0,
        accuracy_by_difficulty={"simple": 0.0},
        error_count=1,
        total_latency_s=1.0,
        started_at="2025-01-01",
    )
    md = render_markdown(report)
    assert "# Eval report" in md
    assert "## Overall" in md
    assert "## By difficulty" in md
    assert "Sample failures" in md
    assert "SELECT 1" in md   # gold sql
    assert "SELECT 2" in md   # predicted sql


def test_render_markdown_handles_no_failures():
    items = [
        EvalItem("1", "x", "Q", "g", "p", "simple", score=True),
    ]
    report = EvalReport(
        config={"db_id": "x"},
        items=items,
        overall_accuracy=1.0,
        accuracy_by_difficulty={"simple": 1.0},
        error_count=0,
        total_latency_s=1.0,
        started_at="now",
    )
    md = render_markdown(report)
    assert "No failures" in md
