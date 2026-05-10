"""
Execution-фильтр для синтетических примеров.

Зачем: даже Claude иногда галлюцинирует имя колонки, путает таблицы или
пишет PostgreSQL-некомпатибельный SQL. Запуск каждого SQL на реальной
БД — самая дешёвая страховка качества.

Учебный нюанс — почему именно execution-check, а не статический парсинг:
  sqlglot валидирует синтаксис, но НЕ проверяет, существуют ли колонки.
  Запрос `SELECT nonexistent_col FROM cards` пройдёт sqlglot, но упадёт
  на PG. И наоборот, корректный SQL с реальной БД не пройдёт через
  «просто валидатор» если в нём, например, есть PG-specific функция
  типа DATE_TRUNC. Поэтому только реальное выполнение даёт правду.

Правила фильтрации:
  - SQL ВЫПОЛНИЛСЯ без ошибок                     → keep
  - SQL вернул ровно 0 строк                      → reject (вероятно WHERE c
                                                    несуществующим литералом)
  - SQL упал с ProgrammingError / DatabaseError   → reject
  - timeout                                        → reject (5 сек на запрос)

Запуск:
  uv run --env-file .env scripts/filter_synthetic_by_execution.py INPUT.jsonl OUTPUT.jsonl
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("filter_synthetic")

QUERY_TIMEOUT_MS = 5000
ENGINE_CACHE: dict[str, Engine] = {}


def get_engine(db_id: str) -> Engine:
    """Кэш engine'ов на db_id. Postgres connection pool одной БД переиспользуется
    между всеми запросами — это снижает overhead на коннект."""
    if db_id in ENGINE_CACHE:
        return ENGINE_CACHE[db_id]
    user = os.environ["DB_USER"]
    pwd = os.environ["DB_PASS"]
    host = os.environ.get("DB_URL", "localhost:5444")
    uri = f"postgresql+psycopg://{user}:{pwd}@{host}/{db_id}"
    eng = create_engine(uri, pool_pre_ping=True)
    ENGINE_CACHE[db_id] = eng
    return eng


def try_execute(db_id: str, sql: str) -> tuple[bool, str, int]:
    """Запускает SELECT и возвращает (ok, error_msg, row_count).
    На любой ошибке откатывает транзакцию (см. практику #8)."""
    if not sql or sql.strip().lower() == "ambiguous":
        # ambiguity-примеры пропускаем без выполнения
        return True, "skipped (ambiguous)", 0
    engine = get_engine(db_id)
    try:
        with engine.connect() as conn:
            conn.execute(text(f"SET statement_timeout = {QUERY_TIMEOUT_MS}"))
            result = conn.execute(text(sql))
            rows = result.fetchall()
            return True, "", len(rows)
    except SQLAlchemyError as e:
        # Очищаем aborted-транзакцию (всё ещё актуально на pool-коннекте)
        try:
            engine.dispose()
            ENGINE_CACHE.pop(db_id, None)
        except Exception:
            pass
        return False, f"{type(e).__name__}: {str(e)[:200]}", 0
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}", 0


def main(in_path: Path, out_path: Path) -> None:
    rejected_path = out_path.with_suffix(".rejected.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = n_kept = n_rejected = 0
    with in_path.open("r") as fin, \
         out_path.open("w", encoding="utf-8") as fout, \
         rejected_path.open("w", encoding="utf-8") as frej:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            row = json.loads(line)
            sql = row.get("SQL", "")
            db_id = row["db_id"]

            # ambiguity-примеры идут без проверки SQL (там нет SQL)
            if row.get("task") == "ambiguity" or str(sql).strip().lower() == "ambiguous":
                row["_exec"] = {"ok": True, "row_count": 0, "note": "ambiguity, no exec"}
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_kept += 1
                continue

            ok, err, rc = try_execute(db_id, sql)
            row["_exec"] = {"ok": ok, "error": err, "row_count": rc}
            # Critère: ok=True И вернул хотя бы 1 строку (защита от галлюцинаций
            # вида WHERE col = 'literal-which-does-not-exist')
            if ok and rc > 0:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_kept += 1
            else:
                frej.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rejected += 1
                if n_rejected <= 5:
                    logger.info(f"  reject [{db_id}] rc={rc} err={err[:120]}")

    logger.info(f"total={n_total}  kept={n_kept}  rejected={n_rejected}")
    logger.info(f"  → kept     : {out_path}")
    logger.info(f"  → rejected : {rejected_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: filter_synthetic_by_execution.py INPUT.jsonl OUTPUT.jsonl", file=sys.stderr)
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
