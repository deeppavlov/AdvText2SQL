"""
heal/collector — собирает уникальные failed-запросы из production логов.

`QueryLogger` в `serve/query_logger.py` пишет каждый неудачный запрос в
`data/logs/failed.jsonl`. Этот модуль читает их, дедуплицирует и собирает
батч >= `min_samples`, который потом обрабатывает `HealSQLGenerator`.

Дедупликация на двух уровнях:
  1. Exact-match по `question` (тривиальные дубликаты)
  2. Эмбеддинг-similarity (cosine > 0.92) — для перефразированных версий
     одного и того же запроса. Опционально (требует sentence-transformers).

Без embedding-dedup может быть OK для маленьких volume'ов (≤1000 failed).
"""
from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("text2sql_tool.heal.collector")


@dataclass
class FailedQuery:
    question: str
    generated_sql: str
    error_message: str
    db_id: str
    timestamp: str


@dataclass
class CollectionResult:
    candidates: list[FailedQuery]
    total_failed: int
    after_dedup_exact: int
    after_dedup_semantic: int | None = None     # None если semantic-dedup выключен

    @property
    def has_enough(self) -> bool:
        return len(self.candidates) > 0


class HealCollector:
    """Читает failed.jsonl и формирует batch для retraining."""

    def __init__(
        self,
        log_dir: str | Path = "data/logs",
        min_samples: int = 50,
        max_samples: int = 500,
        use_semantic_dedup: bool = False,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.use_semantic_dedup = use_semantic_dedup

    def collect(self, db_id: str | None = None) -> CollectionResult:
        failed_path = self.log_dir / "failed.jsonl"
        if not failed_path.exists():
            logger.warning(f"No failed.jsonl found at {failed_path}")
            return CollectionResult(candidates=[], total_failed=0, after_dedup_exact=0)

        records = self._read_jsonl(failed_path, db_id=db_id)
        logger.info(f"Read {len(records)} failed records from {failed_path}")

        # Exact dedup по (db_id, question)
        unique: OrderedDict[tuple[str, str], FailedQuery] = OrderedDict()
        for r in records:
            key = (r.db_id, r.question)
            if key not in unique:
                unique[key] = r

        candidates = list(unique.values())

        semantic_count: int | None = None
        if self.use_semantic_dedup and candidates:
            candidates = self._semantic_dedup(candidates)
            semantic_count = len(candidates)

        # Берём последние max_samples (свежие важнее)
        candidates = candidates[-self.max_samples :]

        return CollectionResult(
            candidates=candidates,
            total_failed=len(records),
            after_dedup_exact=len(unique),
            after_dedup_semantic=semantic_count,
        )

    # ── Internals ────────────────────────────────────────────────────────────

    def _read_jsonl(self, path: Path, db_id: str | None) -> list[FailedQuery]:
        out: list[FailedQuery] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if db_id and r.get("db_id") != db_id:
                    continue
                out.append(
                    FailedQuery(
                        question=r.get("question", ""),
                        generated_sql=r.get("generated_sql", ""),
                        error_message=r.get("error_message") or "",
                        db_id=r.get("db_id", ""),
                        timestamp=r.get("timestamp", ""),
                    )
                )
        return out

    def _semantic_dedup(self, items: list[FailedQuery]) -> list[FailedQuery]:
        """Cosine-dedup через sentence-transformers (опц., тяжёлая зависимость)."""
        try:
            from sentence_transformers import SentenceTransformer, util  # type: ignore
        except ImportError:
            logger.warning("sentence-transformers не установлен — пропускаем semantic-dedup")
            return items

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        questions = [q.question for q in items]
        embeddings = model.encode(questions, convert_to_tensor=True, show_progress_bar=False)

        keep_mask = [True] * len(items)
        threshold = 0.92
        for i in range(len(items)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(items)):
                if not keep_mask[j]:
                    continue
                sim = util.cos_sim(embeddings[i], embeddings[j]).item()
                if sim > threshold:
                    keep_mask[j] = False
        return [items[i] for i in range(len(items)) if keep_mask[i]]
