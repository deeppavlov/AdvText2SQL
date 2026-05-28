"""
training/dataset_builder.py — финальный chat-format JSONL для FT.

На входе: профайл БД (`profile.json`) + отфильтрованная синтетика (`*_validated.jsonl`).
На выходе: `data/finetune/<db_id>/{train,val}.jsonl` в формате, который ест
SFTTrainer через `tokenizer.apply_chat_template` (см. lora_finetune_2.ipynb).

КРИТИЧЕСКИЙ ИНВАРИАНТ train==inference:
    system_prompt в каждом обучающем примере = SYSTEM_PROMPT_TEMPLATE.format(
        db_schema=profile.schema_str,
        db_relationships=profile.relationships_str,
        column_statistics=profile.column_stats_str,
        sql_dialect="PostgreSQL",
    )

Тот же `system_prompt` собирает `Text2SQLGenerator._create_system_prompt` на
инференсе. Если форматирование расходится — модель в проде увидит чужой
формат → деградация. Поэтому импортируем SYSTEM_PROMPT_TEMPLATE напрямую,
а не дублируем.

Используем тот же `SQL_PROMPT_TEMPLATE`, который применяется в
`Text2SQLGenerator.generate_sql()` для оборачивания user-вопроса.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Импортируем ТЕ ЖЕ промпт-шаблоны, что использует Text2SQLGenerator на инференсе.
# Это и есть инвариант train==inference.
from adv_text2sql.mcp_servers.text2sql_tool.src.prompts import (
    SQL_PROMPT_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE,
)

from ..profiler.profile import Profile

logger = logging.getLogger("text2sql_tool.training.dataset_builder")


@dataclass
class DatasetStats:
    train_count: int
    val_count: int
    train_path: Path
    val_path: Path
    by_difficulty: dict[str, int]


class DatasetBuilder:
    """Сборка train/val JSONL для FT из Profile + synthetic examples."""

    def __init__(
        self,
        profile: Profile,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.profile = profile
        self.val_ratio = val_ratio
        self.rng = random.Random(seed)

    def build(
        self,
        synthetic_jsonl: str | Path,
        out_dir: str | Path,
    ) -> DatasetStats:
        synthetic_jsonl = Path(synthetic_jsonl)
        out_dir = Path(out_dir) / self.profile.db_id
        out_dir.mkdir(parents=True, exist_ok=True)
        train_path = out_dir / "train.jsonl"
        val_path = out_dir / "val.jsonl"

        # ── Загружаем все примеры в память (типично <10k для одной БД) ──────
        records = self._load_synthetic(synthetic_jsonl)
        self.rng.shuffle(records)

        # ── Stratified split по difficulty если она есть, иначе случайно ────
        train_records, val_records = self._split(records)

        # ── Рендерим в chat-format ─────────────────────────────────────────
        system_prompt = self._build_system_prompt()

        by_difficulty: dict[str, int] = {}
        with train_path.open("w", encoding="utf-8") as ft:
            for r in train_records:
                ft.write(self._render(r, system_prompt) + "\n")
                by_difficulty[r.get("difficulty", "unknown")] = (
                    by_difficulty.get(r.get("difficulty", "unknown"), 0) + 1
                )

        with val_path.open("w", encoding="utf-8") as fv:
            for r in val_records:
                fv.write(self._render(r, system_prompt) + "\n")

        stats = DatasetStats(
            train_count=len(train_records),
            val_count=len(val_records),
            train_path=train_path,
            val_path=val_path,
            by_difficulty=by_difficulty,
        )
        logger.info(
            f"Dataset built: train={stats.train_count}, val={stats.val_count}, "
            f"by_difficulty={by_difficulty}"
        )
        return stats

    # ── Internals ────────────────────────────────────────────────────────────

    def _load_synthetic(self, path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line in {path}")
        return records

    def _split(
        self, records: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Stratified split по difficulty для сохранения распределения."""
        by_diff: dict[str, list[dict[str, Any]]] = {}
        for r in records:
            by_diff.setdefault(r.get("difficulty", "unknown"), []).append(r)

        train: list[dict[str, Any]] = []
        val: list[dict[str, Any]] = []
        for _, items in by_diff.items():
            n_val = max(1, int(len(items) * self.val_ratio)) if len(items) >= 10 else 0
            val.extend(items[:n_val])
            train.extend(items[n_val:])

        self.rng.shuffle(train)
        self.rng.shuffle(val)
        return train, val

    def _build_system_prompt(self) -> str:
        """Рендер SYSTEM_PROMPT_TEMPLATE на основе Profile. Тот же что на инференсе."""
        return SYSTEM_PROMPT_TEMPLATE.format(
            db_schema=self.profile.schema_str,
            db_relationships=self.profile.relationships_str,
            column_statistics=self.profile.column_stats_str,
            sql_dialect="PostgreSQL",
        )

    def _render(self, record: dict[str, Any], system_prompt: str) -> str:
        """Рендер одной записи в chat-format JSONL.

        Совместимо с notebooks/lora_finetune_2.ipynb (apply_chat_template).
        """
        question = record.get("question") or record.get("user_query") or ""
        sql = record.get("SQL") or record.get("sql") or ""

        user_content = SQL_PROMPT_TEMPLATE.format(
            user_query=question, sql_dialect="PostgreSQL"
        )

        return json.dumps(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": sql},
                ],
                "_meta": {
                    "db_id": self.profile.db_id,
                    "question_id": record.get("question_id"),
                    "task": "sql",
                    "source": record.get("_meta", {}).get("source", "unknown"),
                    "difficulty": record.get("difficulty", "unknown"),
                },
            },
            ensure_ascii=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────


def run_build_dataset(
    profile_path: str,
    synthetic_path: str,
    out_dir: str = "data/finetune",
    val_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetStats:
    """CLI entry point — `text2sql build-dataset`."""
    from rich.console import Console

    console = Console()
    profile = Profile.load_json(profile_path)
    console.print(
        f"[bold cyan]Building dataset[/bold cyan] for {profile.db_id}\n"
        f"  synthetic = {synthetic_path}\n"
        f"  out_dir   = {out_dir}/{profile.db_id}/"
    )

    builder = DatasetBuilder(profile, val_ratio=val_ratio, seed=seed)
    stats = builder.build(synthetic_path, out_dir)

    console.print(
        f"[bold green]✓ Dataset built[/bold green]\n"
        f"  train = {stats.train_count} ({stats.train_path})\n"
        f"  val   = {stats.val_count} ({stats.val_path})\n"
        f"  by difficulty: {stats.by_difficulty}"
    )
    return stats
