"""Unit-тесты для training/ модулей (без БД, без Colab)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from adv_text2sql.profiler.profile import Profile
from adv_text2sql.training.dataset_builder import DatasetBuilder
from adv_text2sql.training.notebook_generator import generate_finetune_notebook


@pytest.fixture()
def toy_profile() -> Profile:
    return Profile(
        db_id="toy",
        db_uri_redacted="postgresql://...",
        tables=["users"],
        column_info={"users": [{"name": "id", "type": "BIGINT"}]},
        relationships={"foreign_keys": [], "primary_keys": {"users": ["id"]}},
        schema_str="TABLE users\n  - id (BIGINT)",
        relationships_str="-- Primary Keys:\n-- users: id",
        column_stats_str="TABLE users (10 rows)\n  - id: integer, unique (id)",
    )


@pytest.fixture()
def synthetic_jsonl(tmp_path: Path) -> Path:
    """30 примеров: 15 simple, 10 moderate, 5 challenging."""
    path = tmp_path / "synth.jsonl"
    records = []
    for i in range(15):
        records.append({
            "question_id": f"q{i}", "db_id": "toy",
            "question": f"Q simple {i}", "SQL": f"SELECT {i}",
            "task": "sql", "difficulty": "simple",
        })
    for i in range(10):
        records.append({
            "question_id": f"q{15+i}", "db_id": "toy",
            "question": f"Q mod {i}", "SQL": f"SELECT * FROM users WHERE id={i}",
            "task": "sql", "difficulty": "moderate",
        })
    for i in range(5):
        records.append({
            "question_id": f"q{25+i}", "db_id": "toy",
            "question": f"Q hard {i}", "SQL": f"SELECT COUNT(*) FROM users GROUP BY id HAVING id={i}",
            "task": "sql", "difficulty": "challenging",
        })
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# DatasetBuilder
# ─────────────────────────────────────────────────────────────────────────────


def test_dataset_builder_creates_train_val(toy_profile, synthetic_jsonl, tmp_path):
    builder = DatasetBuilder(toy_profile, val_ratio=0.1, seed=42)
    stats = builder.build(synthetic_jsonl, tmp_path)

    assert stats.train_path.exists()
    assert stats.val_path.exists()
    assert stats.train_count + stats.val_count == 30


def test_dataset_builder_stratified_split(toy_profile, synthetic_jsonl, tmp_path):
    """Stratified split сохраняет распределение difficulty между train/val."""
    builder = DatasetBuilder(toy_profile, val_ratio=0.2, seed=42)
    stats = builder.build(synthetic_jsonl, tmp_path)

    # При val_ratio=0.2 и 15 simple — должно быть ~3 в val, ~12 в train
    train_records = [json.loads(line) for line in stats.train_path.open()]
    val_records = [json.loads(line) for line in stats.val_path.open()]

    # В каждом сплите должна быть представлена каждая difficulty (хотя бы simple)
    train_diffs = {r["_meta"]["difficulty"] for r in train_records}
    val_diffs = {r["_meta"]["difficulty"] for r in val_records}
    assert "simple" in train_diffs
    assert "simple" in val_diffs


def test_dataset_builder_chat_format(toy_profile, synthetic_jsonl, tmp_path):
    """Каждая запись — 3-message chat (system + user + assistant)."""
    builder = DatasetBuilder(toy_profile, val_ratio=0.1, seed=42)
    stats = builder.build(synthetic_jsonl, tmp_path)

    with stats.train_path.open() as f:
        first = json.loads(f.readline())
    assert "messages" in first
    assert len(first["messages"]) == 3
    assert first["messages"][0]["role"] == "system"
    assert first["messages"][1]["role"] == "user"
    assert first["messages"][2]["role"] == "assistant"
    # System prompt должен содержать схему из Profile
    assert "TABLE users" in first["messages"][0]["content"]


def test_dataset_builder_invariant_with_text2sql_generator(toy_profile, synthetic_jsonl, tmp_path):
    """Главный инвариант: system_prompt в train.jsonl == system_prompt из
    Text2SQLGenerator.load_from_profile() для того же Profile."""
    builder = DatasetBuilder(toy_profile, val_ratio=0.1, seed=42)
    stats = builder.build(synthetic_jsonl, tmp_path)

    with stats.train_path.open() as f:
        train_system = json.loads(f.readline())["messages"][0]["content"]

    # Используем load_from_profile, чтобы получить тот же system_prompt
    # без обращения к БД (мок engine для конструктора)
    profile_path = toy_profile.write_json(tmp_path / "profiles")

    from unittest.mock import MagicMock
    from adv_text2sql.mcp_servers.text2sql_tool.src.text2sql_implementation import Text2SQLGenerator
    # engine — не используется в load_from_profile, моки достаточно
    gen = Text2SQLGenerator.__new__(Text2SQLGenerator)
    gen.engine = MagicMock()
    gen.db_uri = "postgresql://mock"
    gen.llm_client = None  # type: ignore
    gen.load_from_profile(str(profile_path))

    # Главный инвариант — byte-to-byte идентичность
    assert train_system == gen.system_prompt, (
        "system_prompt в train.jsonl и Text2SQLGenerator.load_from_profile "
        "должны совпадать byte-to-byte — иначе train==inference drift"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Notebook generator
# ─────────────────────────────────────────────────────────────────────────────


def test_notebook_generator_writes_valid_ipynb(tmp_path):
    path = generate_finetune_notebook(db_id="card_games", output_dir=tmp_path)
    assert path.exists()
    nb = json.loads(path.read_text())
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) > 5
    # Базовая модель и db_id должны быть в коде ноутбука
    full_source = "".join(
        c["source"] for c in nb["cells"] if c["cell_type"] == "code"
    )
    assert "card_games" in full_source
    assert "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit" in full_source


def test_notebook_generator_respects_hyperparams(tmp_path):
    path = generate_finetune_notebook(
        db_id="x", r=8, lora_alpha=16, lr=1e-4, output_dir=tmp_path
    )
    nb = json.loads(path.read_text())
    full = "".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    assert "r=8" in full
    assert "lora_alpha=16" in full
    assert "0.0001" in full or "1e-04" in full or "1e-4" in full
