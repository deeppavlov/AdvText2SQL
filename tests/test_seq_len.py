"""
Тесты для training.seq_len — динамический подбор max_seq_length под профиль.

Без реальной БД: профили конструируются вручную, размер промпта задаётся длиной
column_stats_str (доминирующая часть системного промпта).
"""
from __future__ import annotations

from adv_text2sql.profiler.profile import Profile
from adv_text2sql.training.seq_len import (
    HARD_CAP,
    SEQ_LEN_BUCKETS,
    estimate_prompt_tokens,
    recommend_max_seq_len,
)


def _profile(column_stats_str: str) -> Profile:
    return Profile(
        db_id="t",
        db_uri_redacted="postgresql://...",
        tables=["t"],
        column_info={"t": [{"name": "id", "type": "INT"}]},
        relationships={"foreign_keys": [], "primary_keys": {"t": ["id"]}},
        schema_str="TABLE t\n  - id (INT)",
        relationships_str="",
        column_stats_str=column_stats_str,
    )


def test_small_profile_picks_smallest_bucket() -> None:
    """Пустая статистика → промпт ~только шаблон → наименьший bucket, без overflow."""
    rec = recommend_max_seq_len(_profile(""))
    assert rec.max_seq_len == SEQ_LEN_BUCKETS[0]
    assert rec.overflow is False
    assert rec.estimated_tokens > 0


def test_recommendation_rounds_up_to_bucket() -> None:
    """Средняя схема → bucket, который >= оценки, и это не самый большой."""
    rec = recommend_max_seq_len(_profile("x" * 20_000))
    assert rec.max_seq_len >= rec.estimated_tokens
    assert rec.max_seq_len in SEQ_LEN_BUCKETS
    assert rec.overflow is False
    # наименьший достаточный: предыдущий bucket был бы мал
    smaller = [b for b in SEQ_LEN_BUCKETS if b < rec.max_seq_len]
    if smaller:
        assert max(smaller) < rec.estimated_tokens


def test_huge_schema_overflows_and_caps() -> None:
    """Огромная статистика (500+ колонок) → превышение HARD_CAP, флаг overflow."""
    rec = recommend_max_seq_len(_profile("x" * 60_000))
    assert rec.estimated_tokens > HARD_CAP
    assert rec.max_seq_len == HARD_CAP
    assert rec.overflow is True
    assert "усеч" in rec.message.lower()


def test_estimate_is_monotonic() -> None:
    """Больше статистики → больше оценка токенов."""
    assert estimate_prompt_tokens(_profile("x" * 1_000)) < estimate_prompt_tokens(
        _profile("x" * 50_000)
    )
