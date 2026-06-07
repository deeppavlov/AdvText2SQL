"""
Сборка LoRA-датасета для Qwen3.5-4B Text2SQL.

Источники:
  1) data/train_queries.json     — BIRD-train, 357 q на 8 БД (7 в hold-out → 1 чистая)
  2) data/bird_train_full.json   — официальный BIRD train (опционально, ~9k q на 95 БД)
  3) data/ambrosia_train.json    — Ambrosia-train, 480 q на 104 БД (0 пересечений с hold-out)
  4) data/synthetic/*.jsonl      — мои сгенерированные пары (см. README в папке)

Hold-out (всё, что НИКОГДА не попадает в trainset):
  bird:     все db_id из data/bird_small.json
  ambrosia: все db_id из data/ambrosia_small.json

Выход:
  data/finetune/train.jsonl                — финальный train в chat-формате
  data/finetune/val.jsonl                  — 10% валидация (in-domain, тот же db_id)
  data/finetune/_schema_cache.json         — кэш схем (растёт инкрементно)
  data/finetune/_stats.json                — отчёт о размере, источниках, db_id

Учебный нюанс:
  Формат сообщений в JSONL ОБЯЗАН совпадать с тем, что отправляется в
  Text2SQLGenerator.query() — иначе модель учится одному pattern, а в
  проде увидит другой. Поэтому system / user построены ровно теми же
  *_PROMPT_TEMPLATE.format(...), что используются на инференсе.

Запуск:
  uv run --env-file .env scripts/build_finetune_dataset.py
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from glob import glob
from pathlib import Path
from textwrap import dedent
from typing import Any

# Делаем `src/` импортируемым — чтобы переиспользовать промпты и Text2SQLGenerator
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from adv_text2sql.mcp_servers.text2sql_tool.src.prompts import (  # noqa: E402
    AMBIGUITY_PROMPT_TEMPLATE,
    SQL_PROMPT_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE,
)
from adv_text2sql.mcp_servers.text2sql_tool.src.text2sql_implementation import (  # noqa: E402
    Text2SQLGenerator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_finetune_dataset")

# ─────────────────────────────────────────────────────────────────────────────
# Конфиг
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = ROOT / "data"
FT_DIR = DATA_DIR / "finetune"
SYN_DIR = DATA_DIR / "synthetic"

BIRD_TRAIN_FILES = [
    # PG-транспилированные и провалидированные версии (см. scripts/transpile_train_sql.py).
    # Раньше тут лежали SQLite-исходники → модель училась SQLite-диалекту → на инференсе
    # PG падал. Теперь поле `SQL` уже в PG, поле `SQL_original` хранит SQLite-исходник для аудита.
    DATA_DIR / "train_queries_pg.json",       # 357 → ~340 после фильтра
    DATA_DIR / "bird_train_full_pg.json",     # ~9k → ~1-2k после фильтра по доступным БД
]
AMBROSIA_TRAIN_FILES = [DATA_DIR / "ambrosia_train_pg.json"]

BIRD_HOLDOUT_FILE = DATA_DIR / "bird_small.json"
AMBROSIA_HOLDOUT_FILE = DATA_DIR / "ambrosia_small.json"

SCHEMA_CACHE_FILE = FT_DIR / "_schema_cache.json"
STATS_FILE = FT_DIR / "_stats.json"
TRAIN_FILE = FT_DIR / "train.jsonl"
VAL_FILE = FT_DIR / "val.jsonl"

VAL_FRACTION = 0.10
RANDOM_SEED = 42
SQL_DIALECT = "PostgreSQL"
USER_AMBIGUITY_PROMPT = "Проверь запрос на однозначность."

# ─────────────────────────────────────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────────────────────────────────────


def load_holdout_db_ids(path: Path) -> set[str]:
    """Читает hold-out JSON и возвращает множество db_id."""
    if not path.exists():
        logger.warning(f"hold-out file missing: {path}")
        return set()
    with path.open("r") as f:
        data = json.load(f)
    return {item["db_id"] for item in data}


def load_json_examples(path: Path) -> list[dict]:
    """JSON-массив с полями question_id, db_id, question, SQL, ..."""
    if not path.exists():
        logger.info(f"skip (missing): {path}")
        return []
    with path.open("r") as f:
        return json.load(f)


def load_jsonl_examples(path: Path) -> list[dict]:
    """JSONL по одному примеру на строку."""
    out: list[dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_synthetic_files() -> list[dict]:
    """Читает все .jsonl в data/synthetic/, КРОМЕ raw_*.jsonl (неотфильтрованные).
    Каждая строка должна иметь те же поля, что и BIRD/Ambrosia (question_id,
    db_id, question, SQL, ...). Дополнительное поле `task`: 'sql' или 'ambiguity'.
    """
    files = [
        f for f in sorted(glob(str(SYN_DIR / "*.jsonl")))
        if not Path(f).name.startswith("raw_")
    ]
    out: list[dict] = []
    for f in files:
        rows = load_jsonl_examples(Path(f))
        logger.info(f"synthetic: +{len(rows)} from {Path(f).name}")
        out.extend(rows)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Schema cache (инкрементный)
# ─────────────────────────────────────────────────────────────────────────────


def load_schema_cache() -> dict[str, dict]:
    if SCHEMA_CACHE_FILE.exists():
        with SCHEMA_CACHE_FILE.open("r") as f:
            return json.load(f)
    return {}


def save_schema_cache(cache: dict[str, dict]) -> None:
    SCHEMA_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SCHEMA_CACHE_FILE.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def build_db_uri(db_id: str) -> str:
    """Та же схема, что в benchmarks/base.py."""
    user = os.environ["DB_USER"]
    pwd = os.environ["DB_PASS"]
    host = os.environ.get("DB_URL", "localhost:5444")
    return f"postgresql+psycopg://{user}:{pwd}@{host}/{db_id}"


def fetch_schema_for_db(db_id: str) -> dict[str, str]:
    """Строит Text2SQLGenerator на db_id, запускает build(), забирает 3 строки.

    Учебный нюанс: build() инициирует эксплорацию БД (light schema +
    relationships + column stats) — это сетевой round-trip к удалённому PG.
    Ожидаемое время: 3-10 секунд на db_id. Поэтому кэшируем.
    """
    db_uri = build_db_uri(db_id)
    # llm_client здесь не нужен (build() не делает LLM-вызовов),
    # но Text2SQLGenerator требует объект — дадим None и положимся на то,
    # что фаза извлечения схемы его не дёргает.
    tool = Text2SQLGenerator(db_uri=db_uri, llm_client=None)  # type: ignore[arg-type]
    tool.build()
    return {
        "schema": tool.db_schema,
        "relationships": getattr(tool, "relationships_str", ""),
        "column_stats": getattr(tool, "column_stats_str", ""),
    }


def ensure_schemas(needed_db_ids: set[str], cache: dict[str, dict]) -> dict[str, dict]:
    """Тянет недостающие схемы, обновляет cache, сохраняет на диск инкрементно.

    Запись считается «не в кэше» если её нет ИЛИ если в ней пустая schema
    (например, прошлый запуск упал на сетевой ошибке и записал заглушку).
    Это позволяет просто перезапустить скрипт после восстановления SSH —
    без ручной очистки кэша.

    Дополнительно: если первая попытка валится с OperationalError (нет
    туннеля), bail out немедленно — нет смысла ждать 105 одинаковых ошибок.
    """
    missing = sorted(d for d in needed_db_ids if not cache.get(d, {}).get("schema"))
    if not missing:
        return cache
    logger.info(f"fetching {len(missing)} schemas from PG ...")
    consecutive_conn_errors = 0
    for i, db_id in enumerate(missing, 1):
        try:
            cache[db_id] = fetch_schema_for_db(db_id)
            logger.info(f"  [{i}/{len(missing)}] {db_id} ✓")
            consecutive_conn_errors = 0
        except Exception as e:
            err_str = str(e)
            logger.warning(f"  [{i}/{len(missing)}] {db_id} ✗ {err_str[:140]}")
            cache[db_id] = {
                "schema": "", "relationships": "", "column_stats": "", "error": err_str[:500],
            }
            if "Connection refused" in err_str or "could not connect" in err_str:
                consecutive_conn_errors += 1
                if consecutive_conn_errors >= 3:
                    logger.error("3 connection failures in a row → SSH tunnel down. "
                                 "Bail out without scanning the rest. "
                                 "Restart `ssh -N -L 5444:...` then rerun this script.")
                    save_schema_cache(cache)
                    sys.exit(2)
        save_schema_cache(cache)
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# Форматирование примера в chat-сообщения
# ─────────────────────────────────────────────────────────────────────────────


def format_sql_example(example: dict, schema: dict[str, str]) -> dict:
    """BIRD-стиль: question + gold SQL → train-сообщения для SQL-генерации."""
    system_content = SYSTEM_PROMPT_TEMPLATE.format(
        db_schema=schema.get("schema", ""),
        db_relationships=schema.get("relationships", ""),
        column_statistics=schema.get("column_stats", ""),
        sql_dialect=SQL_DIALECT,
    )
    user_query = example["question"]
    if example.get("evidence"):
        # Тот же приём, что в benchmarks/bird.py: evidence приклеивается к запросу
        user_query = f"{user_query}\nevidence: {example['evidence']}"
    user_content = dedent(
        SQL_PROMPT_TEMPLATE.format(user_query=user_query, sql_dialect=SQL_DIALECT)
    )
    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            # SQL уже PG — транспиляция и валидация делается в
            # scripts/transpile_train_sql.py до этого этапа. Здесь только формат.
            {"role": "assistant", "content": example["SQL"].strip()},
        ],
        "_meta": {
            "db_id": example["db_id"],
            "question_id": example.get("question_id"),
            "task": "sql",
            "source": example.get("_source", "unknown"),
        },
    }


def format_ambiguity_example(example: dict, schema: dict[str, str]) -> dict:
    """Ambrosia-стиль: question → 'OK' (если есть SQL) / 'Неоднозначность: ...' (если SQL == 'ambiguous').

    Учебный нюанс: ambiguous-разметка в Ambrosia — это просто строка 'ambiguous'
    в поле SQL. Отрицательный класс — все остальные (где есть конкретный SELECT).
    Чтобы научить модель формату ответа, на ambiguous-классе используем явное
    'Неоднозначность: ...' с расшифровкой через ambig_queries — если они есть в датасете.
    """
    system_content = AMBIGUITY_PROMPT_TEMPLATE.format(
        db_schema=schema.get("schema", ""),
        db_relationships=schema.get("relationships", ""),
        column_statistics=schema.get("column_stats", ""),
        user_query=example["question"],
    )
    is_ambiguous = (str(example.get("SQL", "")).strip().lower() == "ambiguous")
    if is_ambiguous:
        ambig_q = example.get("ambig_question") or example.get("ambig_queries")
        if isinstance(ambig_q, list):
            hint = "; ".join(str(x) for x in ambig_q[:3])
        else:
            hint = str(ambig_q) if ambig_q else "запрос имеет несколько интерпретаций"
        assistant = f"Неоднозначность: {hint}"
    else:
        assistant = "OK"
    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": USER_AMBIGUITY_PROMPT},
            {"role": "assistant", "content": assistant},
        ],
        "_meta": {
            "db_id": example["db_id"],
            "question_id": example.get("question_id"),
            "task": "ambiguity",
            "source": example.get("_source", "unknown"),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Главный пайплайн
# ─────────────────────────────────────────────────────────────────────────────


def collect_raw_examples() -> tuple[list[dict], dict[str, int]]:
    """Сводит все источники в плоский список. Каждому добавляет _source и task."""
    examples: list[dict] = []
    counts: dict[str, int] = {}

    # 1) BIRD train
    bird_holdout = load_holdout_db_ids(BIRD_HOLDOUT_FILE)
    logger.info(f"BIRD hold-out db_ids: {sorted(bird_holdout)}")
    for path in BIRD_TRAIN_FILES:
        rows = load_json_examples(path)
        before = len(rows)
        rows = [r for r in rows if r["db_id"] not in bird_holdout]
        for r in rows:
            r["_source"] = path.name
            r["_task"] = "sql"
        examples.extend(rows)
        counts[path.name] = len(rows)
        logger.info(f"BIRD {path.name}: {before} → {len(rows)} after hold-out filter")

    # 2) Ambrosia train (mixed: SQL и ambiguity)
    amb_holdout = load_holdout_db_ids(AMBROSIA_HOLDOUT_FILE)
    logger.info(f"Ambrosia hold-out db_ids: {sorted(amb_holdout)}")
    for path in AMBROSIA_TRAIN_FILES:
        rows = load_json_examples(path)
        before = len(rows)
        rows = [r for r in rows if r["db_id"] not in amb_holdout]
        # Каждая запись Ambrosia может породить ДВА примера:
        #   - ambiguity classification (всегда)
        #   - SQL generation (только если SQL — конкретный SELECT)
        # Здесь только маркируем; expand сделаем в format-фазе.
        for r in rows:
            r["_source"] = path.name
            r["_task"] = "ambiguity"
        examples.extend(rows)
        counts[path.name] = len(rows)
        logger.info(f"Ambrosia {path.name}: {before} → {len(rows)} after hold-out filter")

    # 3) Synthetic — могут быть оба типа
    syn_rows = load_synthetic_files()
    # Синтетика может содержать вопросы с db_id из hold-out — отфильтруем
    syn_rows = [
        r for r in syn_rows
        if r["db_id"] not in bird_holdout and r["db_id"] not in amb_holdout
    ]
    for r in syn_rows:
        r.setdefault("_source", "synthetic")
        r.setdefault("_task", "sql")  # default, можно перебить полем `task` в самой строке
        if "task" in r:
            r["_task"] = r["task"]
    examples.extend(syn_rows)
    counts["synthetic"] = len(syn_rows)
    logger.info(f"synthetic: {len(syn_rows)} examples")

    return examples, counts


def expand_to_messages(
    raw_examples: list[dict], schemas: dict[str, dict]
) -> list[dict]:
    """Каждый сырой пример → 1-2 chat-сообщения в зависимости от task."""
    out: list[dict] = []
    skipped_no_schema = 0
    for ex in raw_examples:
        db_id = ex["db_id"]
        schema = schemas.get(db_id)
        if not schema or not schema.get("schema"):
            skipped_no_schema += 1
            continue

        task = ex.get("_task", "sql")
        if task == "sql":
            if not ex.get("SQL") or str(ex["SQL"]).strip().lower() == "ambiguous":
                continue
            out.append(format_sql_example(ex, schema))
        elif task == "ambiguity":
            # ambiguity-классификация всегда
            out.append(format_ambiguity_example(ex, schema))
            # плюс SQL-пример, если он валидный
            if ex.get("SQL") and str(ex["SQL"]).strip().lower() != "ambiguous":
                out.append(format_sql_example(ex, schema))
        else:
            logger.warning(f"unknown task={task}, skipping")
    if skipped_no_schema:
        logger.warning(f"skipped {skipped_no_schema} examples — no schema in cache")
    return out


def split_train_val(formatted: list[dict], rng: random.Random) -> tuple[list[dict], list[dict]]:
    """In-domain split: 10% от каждого db_id уходит в val.
    Это даёт честный signal на overfit, но НЕ заменяет hold-out (тот строгий по db_id)."""
    by_db: dict[str, list[dict]] = {}
    for ex in formatted:
        by_db.setdefault(ex["_meta"]["db_id"], []).append(ex)

    train, val = [], []
    for db_id, items in by_db.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * VAL_FRACTION)) if len(items) >= 10 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            # _meta нужно для отладки, но trainer его игнорирует — оставляем
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(f"wrote {len(rows)} rows → {path}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--skip-pg", action="store_true",
        help="Не ходить в PG за новыми схемами — использовать только кэш. "
             "Удобно когда туннель лежит: записи с отсутствующей схемой будут "
             "пропущены (логируется skipped_no_schema)."
    )
    args = ap.parse_args()

    rng = random.Random(RANDOM_SEED)
    FT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase A — собираем сырые примеры
    raw, raw_counts = collect_raw_examples()
    logger.info(f"raw examples total: {len(raw)}")

    # Phase B — определяем нужные db_id и тянем схемы (с кэшем)
    needed_db_ids = {ex["db_id"] for ex in raw}
    logger.info(f"need schemas for {len(needed_db_ids)} unique db_ids")
    cache = load_schema_cache()
    if args.skip_pg:
        # Не дёргаем PG — works со всем, что уже в _schema_cache.json.
        missing = [d for d in needed_db_ids if not cache.get(d, {}).get("schema")]
        logger.info(f"--skip-pg: using cache only. {len(missing)} db_ids without schema → их примеры будут пропущены.")
        schemas = cache
    else:
        schemas = ensure_schemas(needed_db_ids, cache)

    # Phase C — формируем chat-сообщения
    formatted = expand_to_messages(raw, schemas)
    logger.info(f"formatted: {len(formatted)} chat examples")

    # Phase D — train/val split + запись
    train, val = split_train_val(formatted, rng)
    rng.shuffle(train)
    write_jsonl(train, TRAIN_FILE)
    write_jsonl(val, VAL_FILE)

    # Phase E — отчёт
    by_task = {}
    by_source = {}
    for ex in formatted:
        m = ex["_meta"]
        by_task[m["task"]] = by_task.get(m["task"], 0) + 1
        by_source[m["source"]] = by_source.get(m["source"], 0) + 1
    stats = {
        "raw_total": len(raw),
        "raw_per_source": raw_counts,
        "formatted_total": len(formatted),
        "by_task": by_task,
        "by_source": by_source,
        "train_count": len(train),
        "val_count": len(val),
        "unique_db_ids": len(needed_db_ids),
        "schemas_resolved": sum(1 for s in schemas.values() if s.get("schema")),
        "schemas_missing": sum(1 for s in schemas.values() if not s.get("schema")),
    }
    with STATS_FILE.open("w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"stats:\n{json.dumps(stats, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
