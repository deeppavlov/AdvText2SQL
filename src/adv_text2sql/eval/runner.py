"""
eval/runner — прогон fine-tuned модели на BIRD/Ambrosia subset.

Главная стадия для защиты: даёт ответ на вопрос «насколько модель стала лучше
после FT под нашу БД». Прогоняет N gold-вопросов через `Text2SQLClient` →
исполняет gold + predicted SQL на реальной PG → сравнивает result sets →
накапливает accuracy.

Архитектура: переиспользует логику сравнения из `benchmarks/evaluate_bird.py`,
но без зависимости от benchmark-framework (TokenTrackingClient/AutoGen).

Параметры eval вынесены в отдельный pydantic — чтобы каждый ран был
полностью описан в `EvalReport.config` для воспроизводимости.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from ..profiler.connector import DBConnector
from ..profiler.profile import Profile
from ..serve.client import Text2SQLClient

logger = logging.getLogger("text2sql_tool.eval.runner")


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GoldItem:
    question_id: str
    db_id: str
    question: str
    gold_sql: str
    difficulty: str
    evidence: str = ""


@dataclass
class EvalItem:
    """Результат одного eval-вопроса."""
    question_id: str
    db_id: str
    question: str
    gold_sql: str
    predicted_sql: str
    difficulty: str
    score: bool                       # True если result sets совпадают
    error: str | None = None
    latency_ms: float | None = None


@dataclass
class EvalReport:
    config: dict[str, Any]
    items: list[EvalItem]
    overall_accuracy: float
    accuracy_by_difficulty: dict[str, float]
    error_count: int
    total_latency_s: float
    started_at: str

    def summary(self) -> dict[str, Any]:
        return {
            "overall_accuracy_pct": round(self.overall_accuracy * 100, 2),
            "by_difficulty": {k: round(v * 100, 2) for k, v in self.accuracy_by_difficulty.items()},
            "error_count": self.error_count,
            "total_items": len(self.items),
            "total_latency_s": round(self.total_latency_s, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────


def load_bird_subset(
    gold_json_path: str | Path,
    db_id: str | None = None,
    limit: int | None = None,
) -> list[GoldItem]:
    """Читает BIRD-style JSON ({question_id, db_id, question, SQL, difficulty, evidence}).

    Если db_id задан — фильтрует только этот subset. limit — отрезает первые N.
    """
    with open(gold_json_path, encoding="utf-8") as f:
        records = json.load(f)

    out: list[GoldItem] = []
    for r in records:
        if db_id and r.get("db_id") != db_id:
            continue
        out.append(
            GoldItem(
                question_id=str(r["question_id"]),
                db_id=r["db_id"],
                question=r["question"],
                gold_sql=r["SQL"],
                difficulty=r.get("difficulty", "unknown"),
                evidence=r.get("evidence", ""),
            )
        )
        if limit and len(out) >= limit:
            break
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────


def compare_result_sets(
    db_url: str, gold_sql: str, predicted_sql: str
) -> tuple[bool, str | None]:
    """Исполнить оба SQL → set-equality. Возвращает (passed, error_msg)."""
    if predicted_sql == "" or predicted_sql == "error":
        return False, "predicted_sql is empty/error"

    # Ambiguous-кейсы (для совместимости с Ambrosia)
    if gold_sql == "ambiguous" or predicted_sql == "ambiguous":
        return gold_sql == predicted_sql, None

    try:
        with DBConnector(db_url) as conn:
            with conn.engine.connect() as db_conn:
                gold_res = db_conn.execute(text(gold_sql)).fetchall()
                pred_res = db_conn.execute(text(predicted_sql)).fetchall()
        # Set comparison игнорирует порядок. Tuple для hashable.
        gold_set = {tuple(r) for r in gold_res}
        pred_set = {tuple(r) for r in pred_res}
        return gold_set == pred_set, None
    except SQLAlchemyError as e:
        return False, str(e)[:200]
    except Exception as e:
        return False, f"unknown: {str(e)[:200]}"


def run_eval(
    profile: Profile,
    api_url: str,
    model_name: str,
    db_url: str,
    gold_items: list[GoldItem],
    api_key: str = "dummy",
    use_evidence: bool = False,
) -> EvalReport:
    """Прогоняет gold_items через Text2SQLClient и аккумулирует accuracy."""
    client = Text2SQLClient(
        profile=profile,
        api_url=api_url,
        model_name=model_name,
        api_key=api_key,
        execute_db_url=None,  # сами исполним — нам нужен contextful compare
    )

    started = time.time()
    items: list[EvalItem] = []

    for i, gold in enumerate(gold_items):
        question = gold.question
        if use_evidence and gold.evidence:
            question = f"question: {gold.question}, evidence (may be empty): {gold.evidence}"

        logger.info(f"[{i+1}/{len(gold_items)}] q_id={gold.question_id} db={gold.db_id}")
        q_started = time.time()

        try:
            result = client.query(question, execute=False)
            predicted_sql = result.sql or ("error" if result.error else "")
        except Exception as e:
            predicted_sql = "error"
            logger.exception(f"Query failed for q_id={gold.question_id}: {e}")

        q_latency = (time.time() - q_started) * 1000

        # Compare result sets
        score, err = compare_result_sets(db_url, gold.gold_sql, predicted_sql)

        items.append(
            EvalItem(
                question_id=gold.question_id,
                db_id=gold.db_id,
                question=gold.question,
                gold_sql=gold.gold_sql,
                predicted_sql=predicted_sql,
                difficulty=gold.difficulty,
                score=score,
                error=err,
                latency_ms=q_latency,
            )
        )

    return _build_report(
        items=items,
        config={
            "api_url": api_url,
            "model_name": model_name,
            "db_id": profile.db_id,
            "use_evidence": use_evidence,
            "n_items": len(gold_items),
        },
        started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        total_latency_s=time.time() - started,
    )


def _build_report(
    items: list[EvalItem],
    config: dict[str, Any],
    started_at: str,
    total_latency_s: float,
) -> EvalReport:
    overall = sum(i.score for i in items) / len(items) if items else 0
    by_diff: dict[str, float] = {}
    for diff in {i.difficulty for i in items}:
        subset = [i for i in items if i.difficulty == diff]
        by_diff[diff] = sum(i.score for i in subset) / len(subset) if subset else 0
    return EvalReport(
        config=config,
        items=items,
        overall_accuracy=overall,
        accuracy_by_difficulty=by_diff,
        error_count=sum(1 for i in items if i.error),
        total_latency_s=total_latency_s,
        started_at=started_at,
    )
