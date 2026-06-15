"""
Тесты для scripts/_registry.py — резолв adaptive vs fixed count_per_lang.

scripts/ не пакет, поэтому добавляем его в sys.path как это делают сами скрипты.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from _registry import Experiment  # noqa: E402

from adv_text2sql.profiler.profile import Profile  # noqa: E402


def _profile() -> Profile:
    p = Profile(
        db_id="card_games",
        db_uri_redacted="postgresql://...",
        tables=[f"t{i}" for i in range(6)],
        column_info={f"t{i}": [{"name": f"c{j}"} for j in range(19)] for i in range(6)},
        relationships={"foreign_keys": [{"from_table": "t1", "to_table": "t0"}], "primary_keys": {}},
        schema_str="",
        relationships_str="",
        column_stats_str="",
    )
    p.compute_complexity_score()
    return p


def test_fixed_count_is_literal() -> None:
    exp = Experiment("tc_fixed", "gpt-4.1", ["ru", "en"], 500)
    assert exp.is_adaptive is False
    assert exp.resolved_count_per_lang(_profile()) == 500


def test_auto_count_splits_budget_across_languages() -> None:
    profile = _profile()
    exp = Experiment("tc_auto", "gpt-4.1", ["ru", "en"], "auto")
    assert exp.is_adaptive is True
    expected = max(1, profile.target_synthetic_count() // 2)
    assert exp.resolved_count_per_lang(profile) == expected


def test_auto_single_language_uses_full_budget() -> None:
    profile = _profile()
    exp = Experiment("tc_auto_ru", "gpt-4.1", ["ru"], "auto")
    assert exp.resolved_count_per_lang(profile) == profile.target_synthetic_count()
