"""
validator — execution-based фильтр для синтетического Q-SQL датасета.

Три уровня проверки (в порядке возрастания стоимости):

  1. **Parse** — sqlglot успешно парсит SQL как PostgreSQL.
  2. **Whitelist** — только SELECT, нет DROP/DELETE/UPDATE/CREATE.
  3. **Execute** — запрос реально выполняется на целевой БД и возвращает
     ≥`min_rows` строк (по умолчанию 1).

Защита от плохих примеров — этот фильтр самый надёжный, потому что использует
реальную семантику БД, а не эвристики. Запрос проходящий execute гарантированно
синтаксически валиден и возвращает данные.

Поверх execution-фильтра опционально работает LLMJudge (в llm_judge.py),
проверяющий семантику пары (вопрос ↔ SQL).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from ..profiler.connector import DBConnector

logger = logging.getLogger("text2sql_tool.synth.validator")


FORBIDDEN_OPS = (exp.Drop, exp.Delete, exp.Update, exp.Insert, exp.Create, exp.Alter)


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    passed: int
    rejected: int
    by_reason: dict[str, int]            # счётчик причин отбраковки
    passed_path: Path
    rejected_path: Path

    @property
    def pass_rate(self) -> float:
        total = self.passed + self.rejected
        return self.passed / total if total else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Core checks
# ─────────────────────────────────────────────────────────────────────────────


def check_parse(sql: str) -> tuple[bool, str]:
    """Парсится ли SQL как PostgreSQL?"""
    try:
        parsed = sqlglot.parse_one(sql, dialect="postgres")
        if parsed is None:
            return False, "empty_parse"
        return True, "ok"
    except ParseError as e:
        return False, f"parse_error: {str(e)[:100]}"
    except Exception as e:
        # TokenError и другие исключения sqlglot — тоже невалидный SQL
        return False, f"parse_error: {type(e).__name__}: {str(e)[:80]}"


def check_whitelist(sql: str) -> tuple[bool, str]:
    """Только SELECT — отсекаем DDL и DML которые могут модифицировать БД."""
    try:
        parsed = sqlglot.parse_one(sql, dialect="postgres")
    except ParseError:
        return False, "parse_error"

    for forbidden in FORBIDDEN_OPS:
        if parsed.find(forbidden):
            return False, f"forbidden_op: {forbidden.__name__}"

    if not isinstance(parsed, (exp.Select, exp.Union, exp.With)):
        return False, f"not_select: {type(parsed).__name__}"
    return True, "ok"


def check_execute(connector: DBConnector, sql: str, min_rows: int = 1) -> tuple[bool, str]:
    """Запрос реально работает на БД и возвращает >=min_rows строк."""
    try:
        with connector.engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.fetchmany(max(min_rows, 5))
            if len(rows) < min_rows:
                return False, f"too_few_rows: {len(rows)} < {min_rows}"
            return True, "ok"
    except SQLAlchemyError as e:
        msg = str(e)[:150].replace("\n", " ")
        return False, f"sql_error: {msg}"
    except Exception as e:
        return False, f"unknown_error: {str(e)[:100]}"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class Validator:
    """Прогон через 3 фильтра + запись в passed/rejected jsonl."""

    def __init__(
        self,
        connector: DBConnector,
        min_rows: int = 1,
        skip_execute: bool = False,
    ) -> None:
        self.connector = connector
        self.min_rows = min_rows
        self.skip_execute = skip_execute

    def validate_jsonl(
        self,
        in_path: str | Path,
        out_dir: str | Path,
    ) -> ValidationResult:
        in_path = Path(in_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        passed_path = out_dir / f"{in_path.stem}_validated.jsonl"
        rejected_path = out_dir / f"{in_path.stem}_rejected.jsonl"

        by_reason: dict[str, int] = {}
        n_passed = n_rejected = 0

        with (
            in_path.open(encoding="utf-8") as fin,
            passed_path.open("w", encoding="utf-8") as fpass,
            rejected_path.open("w", encoding="utf-8") as frej,
        ):
            for line_num, line in enumerate(fin, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    _bump(by_reason, "json_decode_error")
                    n_rejected += 1
                    continue

                sql = record.get("SQL") or record.get("sql", "")
                if not sql:
                    _bump(by_reason, "empty_sql")
                    n_rejected += 1
                    frej.write(_with_reason(record, "empty_sql"))
                    continue

                # Stage 1 — parse
                ok, reason = check_parse(sql)
                if not ok:
                    _bump(by_reason, reason.split(":")[0])
                    n_rejected += 1
                    frej.write(_with_reason(record, reason))
                    continue

                # Stage 2 — whitelist
                ok, reason = check_whitelist(sql)
                if not ok:
                    _bump(by_reason, reason.split(":")[0])
                    n_rejected += 1
                    frej.write(_with_reason(record, reason))
                    continue

                # Stage 3 — execute (опционально, требует БД)
                if not self.skip_execute:
                    ok, reason = check_execute(self.connector, sql, self.min_rows)
                    if not ok:
                        _bump(by_reason, reason.split(":")[0])
                        n_rejected += 1
                        frej.write(_with_reason(record, reason))
                        continue

                # passed
                _bump(by_reason, "ok")
                n_passed += 1
                fpass.write(json.dumps(record, ensure_ascii=False) + "\n")

        result = ValidationResult(
            passed=n_passed,
            rejected=n_rejected,
            by_reason=by_reason,
            passed_path=passed_path,
            rejected_path=rejected_path,
        )
        logger.info(
            f"Validation done: {n_passed} passed / {n_rejected} rejected "
            f"({result.pass_rate * 100:.1f}% pass rate), reasons={by_reason}"
        )
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _bump(d: dict[str, int], key: str) -> None:
    d[key] = d.get(key, 0) + 1


def _with_reason(record: dict[str, Any], reason: str) -> str:
    record = {**record, "_validation_reason": reason}
    return json.dumps(record, ensure_ascii=False) + "\n"
