"""Unit tests for synthetic generation pipeline (без БД)."""
from __future__ import annotations

import json

import pytest

from adv_text2sql.profiler.profile import Profile
from adv_text2sql.synth.template_generator import TemplateSyntheticGenerator
from adv_text2sql.synth.validator import check_parse, check_whitelist


@pytest.fixture()
def toy_profile() -> Profile:
    """Маленький Profile для тестов без БД."""
    return Profile(
        db_id="toy",
        db_uri_redacted="postgresql://...",
        tables=["users", "orders"],
        column_info={
            "users": [
                {"name": "id", "type": "BIGINT"},
                {"name": "name", "type": "TEXT"},
            ],
            "orders": [
                {"name": "id", "type": "BIGINT"},
                {"name": "user_id", "type": "BIGINT"},
                {"name": "amount", "type": "NUMERIC"},
                {"name": "status", "type": "TEXT"},
            ],
        },
        relationships={
            "foreign_keys": [
                {
                    "from_table": "orders",
                    "from_cols": ["user_id"],
                    "to_table": "users",
                    "to_cols": ["id"],
                },
            ],
            "primary_keys": {"users": ["id"], "orders": ["id"]},
        },
        schema_str="TABLE users\n  - id (BIGINT)\n  - name (TEXT)\n\nTABLE orders\n  - id (BIGINT)\n  - user_id (BIGINT)\n  - amount (NUMERIC)\n  - status (TEXT)",
        relationships_str="-- Primary Keys:\n-- users: id\n-- orders: id\n\n-- Foreign Keys:\n-- orders.user_id -> users.id",
        column_stats_str="",
        column_stats={
            "users": {
                "row_count": 10,
                "columns": {
                    "id": {"detected_type": "integer", "distinct_count": 10, "stats": {"type": "integer", "min": 1, "max": 10, "mean": 5.5}},
                    "name": {"detected_type": "categorical", "distinct_count": 10, "stats": {"type": "categorical", "top": [("Alice", 1)]}},
                },
            },
            "orders": {
                "row_count": 100,
                "columns": {
                    "amount": {"detected_type": "numeric", "distinct_count": 50, "stats": {"type": "numeric", "min": 10, "max": 1000, "mean": 500}},
                    "status": {"detected_type": "categorical", "distinct_count": 3, "stats": {"type": "categorical", "top": [("paid", 50), ("pending", 30)]}},
                },
            },
        },
        sample_rows={
            "users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            "orders": [{"id": 1, "user_id": 1, "amount": 100, "status": "paid"}],
        },
        low_cardinality_values={
            "orders.status": ["paid", "pending", "cancelled"],
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Template generator
# ─────────────────────────────────────────────────────────────────────────────


def test_template_generator_produces_examples(toy_profile: Profile) -> None:
    """Шаблонный генератор выдаёт примеры на toy-профиле."""
    gen = TemplateSyntheticGenerator(toy_profile, seed=42)
    examples = gen.generate(20)
    assert len(examples) > 0, "должны быть сгенерированы хоть какие-то примеры"
    # Проверка что каждое поле заполнено
    for ex in examples:
        assert ex.question
        assert ex.sql
        assert ex.db_id == "toy"
        assert ex.template_id


def test_template_generator_sql_is_parseable(toy_profile: Profile) -> None:
    """Все сгенерированные SQL должны парситься sqlglot — это контракт шаблонов."""
    gen = TemplateSyntheticGenerator(toy_profile, seed=42)
    examples = gen.generate(30)
    for ex in examples:
        ok, reason = check_parse(ex.sql)
        assert ok, f"SQL {ex.sql!r} не парсится: {reason} (template={ex.template_id})"


def test_template_generator_sql_is_select_only(toy_profile: Profile) -> None:
    """Шаблоны не должны генерировать DROP/DELETE/UPDATE."""
    gen = TemplateSyntheticGenerator(toy_profile, seed=42)
    examples = gen.generate(30)
    for ex in examples:
        ok, reason = check_whitelist(ex.sql)
        assert ok, f"SQL {ex.sql!r} не SELECT-only: {reason}"


def test_template_generator_jsonl_format(toy_profile: Profile, tmp_path) -> None:
    """to_jsonl_record даёт совместимый с пайплайном формат."""
    gen = TemplateSyntheticGenerator(toy_profile, seed=42)
    examples = gen.generate(5)
    record = examples[0].to_jsonl_record(0)

    assert "question_id" in record
    assert "db_id" in record
    assert "question" in record
    assert "SQL" in record
    assert record["task"] == "sql"
    # Должно сериализоваться без ошибок
    line = json.dumps(record, ensure_ascii=False)
    assert json.loads(line) == record


def test_template_generator_with_empty_profile() -> None:
    """Деградирует gracefully на профиле без таблиц."""
    empty = Profile(
        db_id="empty",
        db_uri_redacted="postgresql://...",
        tables=[],
        column_info={},
        relationships={"foreign_keys": [], "primary_keys": {}},
        schema_str="",
        relationships_str="",
        column_stats_str="",
    )
    gen = TemplateSyntheticGenerator(empty)
    examples = gen.generate(10)
    # Может вернуть 0 — это валидно (с warning в логах)
    assert isinstance(examples, list)


# ─────────────────────────────────────────────────────────────────────────────
# Validator unit tests
# ─────────────────────────────────────────────────────────────────────────────


def test_validator_parse_basic() -> None:
    """check_parse: валидный SELECT проходит."""
    ok, _ = check_parse("SELECT * FROM users WHERE id = 1")
    assert ok


def test_validator_parse_invalid() -> None:
    """check_parse: гарбидж не проходит."""
    ok, reason = check_parse("THIS IS NOT SQL random words")
    assert not ok
    assert "parse" in reason


def test_validator_whitelist_blocks_drop() -> None:
    ok, reason = check_whitelist("DROP TABLE users")
    assert not ok
    assert "forbidden_op" in reason or "not_select" in reason


def test_validator_whitelist_blocks_delete() -> None:
    ok, reason = check_whitelist("DELETE FROM users WHERE id = 1")
    assert not ok


def test_validator_whitelist_blocks_update() -> None:
    ok, reason = check_whitelist("UPDATE users SET name='X' WHERE id = 1")
    assert not ok


def test_validator_whitelist_allows_select() -> None:
    ok, _ = check_whitelist("SELECT * FROM users")
    assert ok


def test_validator_whitelist_allows_cte() -> None:
    ok, _ = check_whitelist(
        "WITH t AS (SELECT 1 AS x) SELECT * FROM t"
    )
    assert ok
