"""
Скачать официальный BIRD train.json и сохранить в формате, совместимом с
существующим data/train_queries.json.

Источник: huggingface.co/datasets/xu3kev/BIRD-SQL-data-train  (9 428 строк, ~2 MB)
Поля HF-датасета:           db_id, question, evidence, SQL, schema
Поля train_queries.json:     question_id, db_id, question, evidence, SQL, difficulty

Пайплайн:
  1) HF Hub → parquet → in-memory list
  2) Drop поле schema (нам не нужно — мы тянем схему из живой PG напрямую)
  3) Добавить question_id (индекс) и difficulty="" (опц)
  4) Фильтр по db_id ∩ PG.databases ─ оставляем только то, для чего у нас
     реально есть БД на сервере. Если db_id отсутствует в PG, пример всё
     равно будет отфильтрован build_finetune_dataset.py на стадии fetch_schema,
     но фильтрация сейчас экономит ~5 секунд на каждой битой БД.
  5) Hold-out НЕ фильтруем здесь — это делает build_finetune_dataset.py.

Запуск:
  uv run --env-file .env scripts/download_bird_train.py

Учебный нюанс:
  HuggingFace datasets кэширует Parquet в ~/.cache/huggingface/. После
  первого запуска повторный займёт <1 сек. Хочешь почистить — `rm -rf
  ~/.cache/huggingface/datasets/xu3kev___bird-sql-data-train`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import datasets as hfds
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_bird_train")

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "bird_train_full.json"
HF_REPO = "xu3kev/BIRD-SQL-data-train"


def list_pg_databases() -> set[str]:
    """Имена всех БД в нашем PG (для фильтра)."""
    user = os.environ["DB_USER"]
    pwd = os.environ["DB_PASS"]
    host = os.environ.get("DB_URL", "localhost:5444")
    eng = create_engine(f"postgresql+psycopg://{user}:{pwd}@{host}/postgres")
    with eng.connect() as c:
        rows = c.execute(text(
            "SELECT datname FROM pg_database WHERE datistemplate=false"
        )).fetchall()
    return {r[0] for r in rows}


def main() -> None:
    logger.info(f"loading {HF_REPO} from HuggingFace ...")
    ds = hfds.load_dataset(HF_REPO, split="train")
    logger.info(f"  HF dataset: {len(ds)} rows, columns={ds.column_names}")

    logger.info("listing remote PG databases ...")
    pg_dbs = list_pg_databases()
    logger.info(f"  PG: {len(pg_dbs)} databases")

    hf_dbs = set(ds["db_id"])
    in_pg = hf_dbs & pg_dbs
    not_in_pg = hf_dbs - pg_dbs
    logger.info(f"  HF db_ids: {len(hf_dbs)}, of which {len(in_pg)} found in PG, "
                f"{len(not_in_pg)} missing")
    if not_in_pg:
        logger.info(f"  examples of missing db_ids (first 10): {sorted(not_in_pg)[:10]}")

    out: list[dict] = []
    for i, row in enumerate(ds):
        if row["db_id"] not in pg_dbs:
            continue
        out.append({
            "question_id": i,                         # индекс из исходного датасета
            "db_id": row["db_id"],
            "question": row["question"],
            "evidence": row.get("evidence", "") or "",
            "SQL": row["SQL"],
            "difficulty": "",                          # опциональное поле
        })
    logger.info(f"  kept {len(out)} / {len(ds)} examples (filtered to PG-available DBs)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f"  → wrote {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")

    # Распределение по db_id (топ-10) — sanity check
    by_db: dict[str, int] = {}
    for r in out:
        by_db[r["db_id"]] = by_db.get(r["db_id"], 0) + 1
    logger.info(f"  unique db_ids in output: {len(by_db)}")
    logger.info("  top-10 db_ids by question count:")
    for db_id, n in sorted(by_db.items(), key=lambda kv: -kv[1])[:10]:
        logger.info(f"    {n:5d}  {db_id}")


if __name__ == "__main__":
    main()
