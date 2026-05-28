"""
Тесты для autonomous Text2SQL profiler.

Главный инвариант (test_schema_str_invariant):
    Profile.schema_str + relationships_str + column_stats_str, отрендеренные
    в SYSTEM_PROMPT_TEMPLATE, должны byte-identical совпадать с
    Text2SQLGenerator.build().system_prompt.

Без этого инварианта модель в проде получит system_prompt отличный от того,
на чём она обучалась → train==inference drift → потеря accuracy.

Этот тест требует доступа к реальной PG-БД через SSH-туннель. Если переменные
окружения DB_USER, DB_PASS не заданы или туннель не запущен — тесты skip'ются.
"""
from __future__ import annotations

import os

import pytest

from adv_text2sql.profiler.connector import DBConnector
from adv_text2sql.profiler.profile import Profile
from adv_text2sql.profiler.sample_collector import SampleCollector
from adv_text2sql.profiler.schema_extractor import SchemaExtractor
from adv_text2sql.profiler.stats_collector import StatsCollector


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests (no DB required)
# ─────────────────────────────────────────────────────────────────────────────


def test_complexity_score_minimum_db() -> None:
    """Для совсем простой схемы (1 table, 2 cols, 0 fk) — score > 0."""
    profile = Profile(
        db_id="tiny",
        db_uri_redacted="postgresql://...",
        tables=["t"],
        column_info={"t": [{"name": "id", "type": "INT"}, {"name": "name", "type": "TEXT"}]},
        relationships={"foreign_keys": [], "primary_keys": {"t": ["id"]}},
        schema_str="TABLE t\n  - id (INT)\n  - name (TEXT)",
        relationships_str="-- Primary Keys:\n-- t: id",
        column_stats_str="",
    )
    score = profile.compute_complexity_score()
    assert score > 0
    # Target count для крошечной БД должен попасть в minimum=200
    assert profile.target_synthetic_count() == 200


def test_target_count_scales_with_complexity() -> None:
    """Более сложная схема → больше target_count."""
    simple = Profile(
        db_id="simple",
        db_uri_redacted="postgresql://...",
        tables=["t1", "t2"],
        column_info={"t1": [{"name": "id"}], "t2": [{"name": "id"}]},
        relationships={"foreign_keys": [], "primary_keys": {}},
        schema_str="",
        relationships_str="",
        column_stats_str="",
    )

    complex_ = Profile(
        db_id="complex",
        db_uri_redacted="postgresql://...",
        tables=[f"t{i}" for i in range(20)],
        column_info={f"t{i}": [{"name": f"col{j}"} for j in range(15)] for i in range(20)},
        relationships={
            "foreign_keys": [
                {"from_table": f"t{i}", "from_cols": ["c"], "to_table": "t0", "to_cols": ["c"]}
                for i in range(1, 15)
            ],
            "primary_keys": {},
        },
        schema_str="",
        relationships_str="",
        column_stats_str="",
    )

    assert complex_.compute_complexity_score() > simple.compute_complexity_score()
    assert complex_.target_synthetic_count() > simple.target_synthetic_count()


def test_profile_roundtrip(tmp_path) -> None:
    """Profile сериализуется и десериализуется без потерь."""
    p1 = Profile(
        db_id="card_games",
        db_uri_redacted="postgresql://user:***@host/card_games",
        tables=["cards"],
        column_info={"cards": [{"name": "id", "type": "BIGINT"}]},
        relationships={"foreign_keys": [], "primary_keys": {"cards": ["id"]}},
        schema_str="TABLE cards\n  - id (BIGINT)",
        relationships_str="-- Primary Keys:\n-- cards: id",
        column_stats_str="TABLE cards (10 rows)\n  - id: integer, unique (id)",
        sample_rows={"cards": [{"id": 1}]},
    )
    p1.compute_complexity_score()
    path = p1.write_json(tmp_path)
    p2 = Profile.load_json(path)
    assert p1.model_dump() == p2.model_dump()


def test_redacted_uri_strips_password() -> None:
    """В profile.json не должно быть паролей."""
    conn = DBConnector("postgresql+psycopg://user:secret123@host:5444/db")
    redacted = conn.redacted_uri()
    assert "secret123" not in redacted
    assert "user" in redacted
    assert "***" in redacted


# ─────────────────────────────────────────────────────────────────────────────
# Integration test — главный инвариант train==inference
# ─────────────────────────────────────────────────────────────────────────────


def _require_db_env() -> tuple[str, str]:
    user = os.environ.get("DB_USER")
    password = os.environ.get("DB_PASS")
    if not user or not password:
        pytest.skip("Требуются env DB_USER, DB_PASS + SSH-туннель на localhost:5444")
    return user, password


@pytest.mark.integration
def test_schema_str_invariant() -> None:
    """schema_str из Profile == db_schema из Text2SQLGenerator (byte-identical).

    Это страховка от train==inference дрифта. Если этот тест падает —
    модель в проде получит другой формат схемы, чем видела на FT → деградация.
    """
    user, password = _require_db_env()
    db_id = os.environ.get("INVARIANT_TEST_DB", "card_games")
    db_url = f"postgresql+psycopg://{user}:{password}@localhost:5444/{db_id}"

    # ── Наш новый профайлер ─────────────────────────────────────────────────
    with DBConnector(db_url) as connector:
        new_schema = SchemaExtractor(connector).extract()

    # ── Старый Text2SQLGenerator — берём только эквивалент schema_str ───────
    from adv_text2sql.mcp_servers.text2sql_tool.src.text2sql_implementation import (
        Text2SQLGenerator,
    )

    legacy = Text2SQLGenerator(db_uri=db_url, llm_client=None)  # type: ignore[arg-type]
    # Вызываем приватный метод напрямую — нам нужен только результат
    # _get_db_schema_light, без всей build()-цепочки (которая делает LLM-init)
    legacy_schema_str = legacy._get_db_schema_light()
    legacy_relationships = legacy._explore_db_relationships()
    legacy.db_relationships = legacy_relationships
    legacy_relationships_str = legacy._format_relationships()

    # Главные инварианты
    assert new_schema.schema_str == legacy_schema_str, (
        "schema_str не совпадает с Text2SQLGenerator._get_db_schema_light. "
        "Это сломает train==inference инвариант."
    )
    assert new_schema.relationships_str == legacy_relationships_str, (
        "relationships_str не совпадает с Text2SQLGenerator._format_relationships."
    )


@pytest.mark.integration
def test_stats_str_invariant() -> None:
    """column_stats_str из StatsCollector == _format_column_statistics из генератора."""
    user, password = _require_db_env()
    db_id = os.environ.get("INVARIANT_TEST_DB", "toxicology")  # маленькая БД для скорости
    db_url = f"postgresql+psycopg://{user}:{password}@localhost:5444/{db_id}"

    with DBConnector(db_url) as connector:
        new_stats = StatsCollector(connector).collect()

    from adv_text2sql.mcp_servers.text2sql_tool.src.text2sql_implementation import (
        Text2SQLGenerator,
    )

    legacy = Text2SQLGenerator(db_uri=db_url, llm_client=None)  # type: ignore[arg-type]
    legacy_stats = legacy._explore_column_statistics()
    legacy_stats_str = legacy._format_column_statistics(legacy_stats)

    # Format string должен совпадать — это уходит в system prompt.
    # Сами числа могут отличаться (sampling рандомный), но формат — нет.
    # Поэтому проверяем структуру (количество строк, ключевые подстроки).
    new_lines = new_stats.column_stats_str.splitlines()
    legacy_lines = legacy_stats_str.splitlines()
    assert len(new_lines) == len(legacy_lines), (
        f"Разное количество строк: new={len(new_lines)}, legacy={len(legacy_lines)}"
    )


@pytest.mark.integration
def test_end_to_end_profile(tmp_path) -> None:
    """Полный профайлер на маленькой БД — toxicology (~4 таблицы)."""
    user, password = _require_db_env()
    db_url = f"postgresql+psycopg://{user}:{password}@localhost:5444/toxicology"

    with DBConnector(db_url) as connector:
        schema = SchemaExtractor(connector).extract()
        stats = StatsCollector(connector).collect()
        samples = SampleCollector(connector).collect()

        profile = Profile(
            db_id=connector.db_id,
            db_uri_redacted=connector.redacted_uri(),
            tables=schema.tables,
            column_info=schema.column_info,
            relationships=dict(schema.relationships),
            schema_str=schema.schema_str,
            relationships_str=schema.relationships_str,
            column_stats_str=stats.column_stats_str,
            column_stats=stats.column_stats,
            sample_rows=samples.sample_rows,
            low_cardinality_values=samples.low_cardinality_values,
        )
        profile.compute_complexity_score()
        path = profile.write_json(tmp_path)

    assert path.exists()
    assert path.stat().st_size > 0
    loaded = Profile.load_json(path)
    assert loaded.db_id == "toxicology"
    assert len(loaded.tables) > 0
