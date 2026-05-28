"""
Profile — pydantic-модель результата стадии PROFILE.

Сериализуется в `data/profiles/<db_id>/profile.json` и потребляется всеми
последующими стадиями (generate, build-dataset, train, serve). Это контракт
между стадиями — менять поля можно только с обновлением downstream.

КРИТИЧЕСКИЙ ИНВАРИАНТ: поле `schema_str` должно byte-to-byte совпадать с тем,
что `Text2SQLGenerator.build()` кладёт в `self.db_schema`. Проверяется тестом
`tests/test_profiler.py::test_schema_str_invariant`.

`complexity_score` используется в Week 2 для адаптивного выбора `target_count`
синтетических примеров — большая схема → больше примеров нужно.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Profile(BaseModel):
    """Полный профиль PostgreSQL БД для autonomous Text2SQL pipeline."""

    # ── Identity ────────────────────────────────────────────────────────────
    db_id: str
    db_uri_redacted: str
    dialect: str = "postgresql"

    # ── Schema (структурированно) ───────────────────────────────────────────
    tables: list[str]
    column_info: dict[str, list[dict[str, Any]]]
    relationships: dict[str, Any]

    # ── Schema (готовые строки для system prompt) ───────────────────────────
    # ВАЖНО: эти поля попадают напрямую в SYSTEM_PROMPT_TEMPLATE — должны
    # byte-identical с тем, что выдаёт Text2SQLGenerator.build().
    schema_str: str
    relationships_str: str
    column_stats_str: str

    # ── Statistics (для downstream использования) ───────────────────────────
    column_stats: dict[str, Any] = Field(default_factory=dict)

    # ── Samples (для value-grounding генератора в Week 2) ───────────────────
    sample_rows: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    low_cardinality_values: dict[str, list[Any]] = Field(default_factory=dict)

    # ── Metadata ────────────────────────────────────────────────────────────
    complexity_score: float = 0.0
    profiled_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    profiler_version: str = "0.1.0"

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def write_json(self, out_dir: str | Path) -> Path:
        """Записывает в `<out_dir>/<db_id>/profile.json`. Создаёт директорию."""
        target_dir = Path(out_dir) / self.db_id
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / "profile.json"
        # mode='json' даёт нам сериализацию datetime и др. сложных типов
        path.write_text(
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load_json(cls, path: str | Path) -> "Profile":
        text = Path(path).read_text(encoding="utf-8")
        return cls.model_validate_json(text)

    # ─────────────────────────────────────────────────────────────────────────
    # Derived metrics
    # ─────────────────────────────────────────────────────────────────────────

    def compute_complexity_score(self) -> float:
        """Эвристика сложности схемы → определяет target_count синтетики.

            score = log(#tables) * sqrt(#columns) * (1 + #fk / #tables)

        Эмпирические значения:
            Sakila    (~16 tab, ~90 col, ~22 fk) → ~12
            card_games (~6 tab, ~113 col, ~2 fk)  → ~9
            financial (~8 tab, ~58 col, ~8 fk)    → ~10
        """
        n_tables = max(1, len(self.tables))
        n_columns = sum(len(cols) for cols in self.column_info.values())
        n_fk = len(self.relationships.get("foreign_keys", []))

        score = (
            math.log(n_tables + 1)
            * math.sqrt(max(1, n_columns))
            * (1 + n_fk / n_tables)
        )
        self.complexity_score = round(score, 2)
        return self.complexity_score

    def target_synthetic_count(self, multiplier: int = 50, minimum: int = 200) -> int:
        """Адаптивный target_count для синтетики на основе сложности схемы.

        Args:
            multiplier: сколько примеров на единицу сложности (default 50).
            minimum: пол — даже для крошечных схем не меньше этого числа.
        """
        if self.complexity_score == 0:
            self.compute_complexity_score()
        return max(minimum, int(self.complexity_score * multiplier))
