"""
Транспиляция BIRD-train SQLite → PostgreSQL (опционально с валидацией на PG).

Источник: BIRD gold SQL написан под SQLite (backticks, STRFTIME, INTEGER/INTEGER=float),
а наш пайплайн работает с PostgreSQL. Если обучать модель на SQLite-диалекте,
она запоминает SQLite-паттерны и на инференсе генерирует SQL, который PG не понимает.

Что делает этот скрипт (default):
  1. Читает каждый файл из SOURCE_FILES.
  2. Для каждой записи: sqlglot.transpile(sqlite → postgres) + NULLIF-фикс деления.
  3. Пишет `*_pg.json` с полем `SQL` = транспилированный, `SQL_original` = исходник.

С флагом `--validate` дополнительно:
  4. Запускает транспилированный SQL на PG (читает доступные БД через pg_database).
  5. Записи с execution-ошибкой уходят в `*_pg_errors.json`.
  6. Требует поднятого SSH-туннеля или локального PG.

Без валидации (default) — sqlglot для BIRD-паттернов работает в ~99% случаев,
оставшийся ~1% либо безвреден (модель его пропустит), либо обнаружится при eval.

Запуск:
  uv run --env-file .env scripts/transpile_train_sql.py              # только транспиляция
  uv run --env-file .env scripts/transpile_train_sql.py --validate   # + PG-валидация
  uv run --env-file .env scripts/transpile_train_sql.py --limit 50   # на 50 записях
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import sqlglot
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("transpile_train_sql")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Какие файлы транспилируем. Ambrosia уже почти-PG, но содержит double-quoted
# string literals («Sundance»), которые PG считает идентификаторами →
# pre-norm обязателен. sqlglot для остальных конструкций будет no-op.
SOURCE_FILES = [
    DATA_DIR / "train_queries.json",
    DATA_DIR / "bird_train_full.json",
    DATA_DIR / "ambrosia_train.json",
]


# ─────────────────────────────────────────────────────────────────────────────
# Транспиляция
# ─────────────────────────────────────────────────────────────────────────────


def _apply_nullif_outside_strings(sql: str) -> str:
    """Применяет NULLIF-регексп только к участкам ВНЕ '...' и "..." регионов.

    Учебный нюанс — три класса ловушек, которые здесь обходятся:

    1) Имена колонок типа `"Charter School (Y/N)"` ломались бы наивным
       `(\\b\\w+\\b)\\s*/\\s*(\\b\\w+\\b)` (как в evaluate_bird.py) — он
       поймал бы `Y/N` ВНУТРИ кавычек и порвал бы идентификатор.
       Фикс: маскируем кавычки placeholder'ами до regex, восстанавливаем после.

    2) RHS-выражения вида `... / NULLIF(...)` ловились бы regex'ом
       как `word / NULLIF`, что давало бы `NULLIF(NULLIF, 0)(COUNT(...))`
       (синтаксический мусор). Lookahead `(?!\\s*\\()` гарантирует, что
       RHS — это БАРЕ ИДЕНТИФИКАТОР, а не вызов функции.

    3) Числовые литералы (`x / 100`) тоже матчатся, но это безвредно:
       `NULLIF(100, 0)` всегда вернёт 100. Лишний overhead, не лишний баг.
    """
    PLACEHOLDER = "\x00MASK{}\x00"
    stash: list[str] = []

    def _mask(m):
        stash.append(m.group(0))
        return PLACEHOLDER.format(len(stash) - 1)

    masked = re.sub(r"'(?:[^']|'')*'", _mask, sql)
    masked = re.sub(r'"(?:[^"]|"")*"', _mask, masked)

    # NULLIF-обёртка для деления голых идентификаторов; lookahead исключает функции.
    masked = re.sub(
        r"(\b\w+\b)\s*/\s*(\b\w+\b)(?!\s*\()",
        r"\1 / NULLIF(\2,0)",
        masked,
    )

    for i, frag in enumerate(stash):
        masked = masked.replace(PLACEHOLDER.format(i), frag)
    return masked


def _normalize_sqlite_double_quoted_strings(sql: str) -> str:
    """SQLite разрешает '"foo"' как строковый литерал когда foo не идентификатор.
    PostgreSQL же '"foo"' всегда трактует как quoted identifier (column name).
    Конвертируем `op "..."` в `op '...'` ДО sqlglot — иначе модель училась бы
    писать `WHERE name = "Sundance"`, что в PG = UndefinedColumn.

    Эвристика: `"..."` СРАЗУ после оператора сравнения / IN-списка / VALUES /
    запятой внутри IN — это всегда строковый литерал, не идентификатор.
    Покрывает Ambrosia (там double-quoted strings везде) и небольшую часть BIRD.
    """
    # Pattern: capture (op + whitespace) + "..." → keep op, replace quotes.
    # Не трогаем экранированные внутри кавычек (упрощённо: считаем что "" не встретится).
    def _swap(m):
        op, content = m.group(1), m.group(2)
        # Если внутри есть одинарная кавычка — экранируем '' (PG-стандарт).
        content_escaped = content.replace("'", "''")
        return f"{op}'{content_escaped}'"

    pattern = re.compile(
        r"(=|<>|!=|<|>|<=|>=|\bLIKE\b|\bILIKE\b|\bIN\s*\(|,)\s*\"([^\"]*)\"",
        flags=re.IGNORECASE,
    )
    prev = None
    out = sql
    # Повторяем пока что-то меняется (для `IN ("a", "b", "c")` нужно 3 прохода).
    while prev != out:
        prev = out
        out = pattern.sub(_swap, out)
    return out


def transpile_to_postgres(sql: str) -> str:
    """SQLite → PostgreSQL pipeline:

      1) Снимаем markdown-обёртку и SQL-комменты (если есть).
      2) Pre-norm: конвертим '"..."' string-literals в "'...'", чтобы PG не
         подумал что это quoted identifier.
      3) sqlglot.transpile(sqlite→postgres) — основная работа: backticks,
         STRFTIME, IIF, CAST AS REAL, LIMIT/ORDER NULLS handling.
      4) Post-fix: NULLIF на RHS деления для защиты от div-by-zero,
         но только на голых идентификаторах и вне кавычек.
    """
    sql = re.sub(r"```sql|```", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL).strip()
    sql = _normalize_sqlite_double_quoted_strings(sql)
    try:
        out = sqlglot.transpile(sql, read="sqlite", write="postgres")[0]
    except Exception as e:
        raise ValueError(f"sqlglot parse error: {e!r}") from e
    return _apply_nullif_outside_strings(out)


# ─────────────────────────────────────────────────────────────────────────────
# Подключение к PG (та же схема URI, что и в benchmarks/base.py)
# ─────────────────────────────────────────────────────────────────────────────


def build_db_uri(db_id: str) -> str:
    user = os.environ["DB_USER"]
    pwd = os.environ["DB_PASS"]
    host = os.environ.get("DB_URL", "localhost:5444")
    return f"postgresql+psycopg://{user}:{pwd}@{host}/{db_id}"


# Кэш engines по db_id — чтобы не пересоздавать пул на каждый запрос
_engines: dict[str, Any] = {}


def get_engine(db_id: str):
    if db_id not in _engines:
        _engines[db_id] = create_engine(build_db_uri(db_id), pool_pre_ping=True)
    return _engines[db_id]


def validate_sql_on_pg(sql: str, db_id: str) -> tuple[bool, str]:
    """Запускает SELECT-запрос на PG. Возвращает (success, error_str).

    Учебный нюанс: используем fetchmany(1) вместо fetchall — нам нужен только
    факт «запрос корректно отработал», полный result set тратит память и время.
    """
    try:
        engine = get_engine(db_id)
        with engine.connect() as conn:
            res = conn.execute(text(sql))
            # touch one row to force the query to actually run (PG может ленясь
            # планировать, но execute обычно уже выполнил всё нужное)
            res.fetchmany(1)
        return True, ""
    except SQLAlchemyError as e:
        return False, str(e)[:300]
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"[:300]


def list_available_databases() -> set[str]:
    """Запрашивает у PG список существующих баз. Используется как pre-flight
    фильтр: BIRD train содержит ~69 db_id, у нас доступно ~8-9 — без этого
    скрипт бы запускал ~6000 заведомо обречённых запросов.

    Подключаемся к системной БД `postgres` (она есть всегда у readonly_user),
    выгребаем `pg_database.datname`.
    """
    user = os.environ["DB_USER"]
    pwd = os.environ["DB_PASS"]
    host = os.environ.get("DB_URL", "localhost:5444")
    sys_uri = f"postgresql+psycopg://{user}:{pwd}@{host}/postgres"
    engine = create_engine(sys_uri)
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT datname FROM pg_database WHERE datistemplate=false"
        )).fetchall()
    return {r[0] for r in rows}


def preflight() -> set[str]:
    """Бьёт по PG, проверяет туннель и возвращает список доступных db_id.
    При connection-refused сразу падает с понятным сообщением (вместо того
    чтобы дальше прокручивать сотни запросов с тем же error)."""
    try:
        dbs = list_available_databases()
    except Exception as e:
        msg = str(e)
        if "Connection refused" in msg or "could not connect" in msg:
            logger.error(
                "PG unreachable on localhost:5444. SSH tunnel down?\n"
                "  ssh -N -L 5444:10.11.1.6:5444 user_name@lnsigo.mipt.ru -p2278"
            )
        else:
            logger.error(f"PG preflight failed: {msg[:300]}")
        sys.exit(2)
    logger.info(f"PG available, {len(dbs)} databases: {sorted(dbs)}")
    return dbs


# ─────────────────────────────────────────────────────────────────────────────
# Основной пайплайн
# ─────────────────────────────────────────────────────────────────────────────


def process_file(
    src: Path, limit: int | None, available_dbs: set[str] | None
) -> tuple[list[dict], list[dict], dict]:
    """Returns (good, errors, stats).

    available_dbs:
      - None → режим «только транспиляция», без обращений к PG.
      - set[str] → режим «транспиляция + валидация»; записи с db_id вне
        множества помечаются `_error_phase=missing_db` и не дёргают сеть впустую.
    """
    validate = available_dbs is not None
    if not src.exists():
        logger.warning(f"missing: {src}")
        return [], [], {"file": src.name, "missing": True}

    with src.open("r") as f:
        rows = json.load(f)
    if limit:
        rows = rows[:limit]
    logger.info(f"{src.name}: {len(rows)} rows")

    good: list[dict] = []
    errors: list[dict] = []
    stats = {
        "file": src.name,
        "total": len(rows),
        "transpile_failed": 0,
        "exec_failed": 0,
        "missing_db": 0,
        "ok": 0,
        "by_db_id_ok": {},
        "by_db_id_total": {},
    }

    # Группируем по db_id чтобы переиспользовать engine + бить fast-fail-ом
    # по `consecutive_conn_errors` строго в пределах одной БД.
    rows_by_db: dict[str, list[tuple[int, dict]]] = {}
    for i, r in enumerate(rows):
        rows_by_db.setdefault(r["db_id"], []).append((i, r))

    consecutive_conn_errors = 0
    CONN_ERROR_BAIL = 5  # если 5 раз подряд "Connection refused" — туннель умер

    for db_id, db_rows in rows_by_db.items():
        stats["by_db_id_total"][db_id] = len(db_rows)

        # Pre-filter: если валидируем и БД отсутствует на сервере — все её записи
        # помечаем как missing_db и идём дальше, не тратя сеть.
        if validate and db_id not in available_dbs:
            stats["missing_db"] += len(db_rows)
            for i, row in db_rows:
                errors.append({**row, "_error_phase": "missing_db",
                               "_error": f"db_id '{db_id}' not present on PG server"})
            stats["by_db_id_ok"][db_id] = 0
            continue

        ok_for_db = 0
        for i, row in db_rows:
            original_sql = row.get("SQL", "").strip()

            # «ambiguous»-метки нужны Ambrosia-ambiguity-задаче и не требуют
            # транспиляции — пропускаем как есть в good, чтобы build_finetune
            # увидел весь корпус.
            if not original_sql or original_sql.lower() == "ambiguous":
                good_row = dict(row)
                good_row["SQL_original"] = original_sql
                # SQL оставляем как есть ('ambiguous' или '')
                good.append(good_row)
                stats["ok"] += 1
                ok_for_db += 1
                continue

            # 1) Транспиляция (всегда)
            try:
                pg_sql = transpile_to_postgres(original_sql)
            except ValueError as e:
                stats["transpile_failed"] += 1
                errors.append({**row, "_error_phase": "transpile", "_error": str(e)})
                continue

            # 2) Опциональная PG-валидация
            if validate:
                ok, err = validate_sql_on_pg(pg_sql, db_id)
                if not ok:
                    stats["exec_failed"] += 1
                    errors.append({
                        **row,
                        "_error_phase": "execute",
                        "_error": err,
                        "_transpiled_sql": pg_sql,
                    })
                    if "Connection refused" in err or "could not connect" in err:
                        consecutive_conn_errors += 1
                        if consecutive_conn_errors >= CONN_ERROR_BAIL:
                            logger.error(
                                f"{CONN_ERROR_BAIL} consecutive Connection-refused → "
                                "SSH tunnel down. Bail out so partial results aren't lost."
                            )
                            return good, errors, stats
                    continue
                # успех → сбрасываем счётчик
                consecutive_conn_errors = 0

            # 3) Успех — пишем транспилированную, сохраняем оригинал для аудита
            good_row = dict(row)
            good_row["SQL_original"] = original_sql
            good_row["SQL"] = pg_sql
            good.append(good_row)
            stats["ok"] += 1
            ok_for_db += 1

            if stats["ok"] % 100 == 0:
                logger.info(
                    f"  progress: ok={stats['ok']}  "
                    f"transpile_fail={stats['transpile_failed']}  "
                    f"exec_fail={stats['exec_failed']}  "
                    f"missing_db={stats['missing_db']}"
                )

        stats["by_db_id_ok"][db_id] = ok_for_db

    return good, errors, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Обрабатывать только первые N записей из каждого файла (для теста)")
    ap.add_argument("--validate", action="store_true",
                    help="Дополнительно валидировать на PG (нужен туннель/локальная БД). "
                         "По умолчанию — только sqlglot transpile.")
    args = ap.parse_args()

    # Pre-flight только если просили валидацию.
    available_dbs = preflight() if args.validate else None
    if not args.validate:
        logger.info("validate=False → транспиляция без PG-проверки (no tunnel needed)")

    all_stats = []
    for src in SOURCE_FILES:
        good, errors, stats = process_file(src, args.limit, available_dbs)

        # Имя выходного файла: foo.json → foo_pg.json
        if not good and not errors:
            continue
        out_good = src.with_name(src.stem + "_pg.json")
        out_err = src.with_name(src.stem + "_pg_errors.json")

        with out_good.open("w", encoding="utf-8") as f:
            json.dump(good, f, ensure_ascii=False, indent=2)
        with out_err.open("w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)

        logger.info(
            f"WROTE {out_good.name}: {len(good)} rows  "
            f"| {out_err.name}: {len(errors)} errors"
        )
        all_stats.append(stats)

    # Финальная сводка
    print("\n=================== TRANSPILE REPORT ===================\n")
    for s in all_stats:
        if s.get("missing"):
            print(f"{s['file']}: MISSING — skipped")
            continue
        total = s["total"]
        ok = s["ok"]
        tf = s["transpile_failed"]
        ef = s["exec_failed"]
        md = s.get("missing_db", 0)
        rate = 100.0 * ok / total if total else 0.0
        eligible = total - md
        rate_eligible = 100.0 * ok / eligible if eligible else 0.0
        print(f"{s['file']}:")
        print(f"  total          : {total}")
        print(f"  missing_db     : {md}   (db_id отсутствует на PG-сервере)")
        print(f"  transpile_fail : {tf}")
        print(f"  exec_fail      : {ef}")
        print(f"  ok             : {ok}  ({rate:.1f}% от total | {rate_eligible:.1f}% от eligible)")
        # Топ-5 db_id с наихудшим успехом
        ratios = []
        for db, t in s["by_db_id_total"].items():
            k = s["by_db_id_ok"].get(db, 0)
            ratios.append((db, k, t, 100.0 * k / t if t else 0.0))
        ratios.sort(key=lambda x: x[3])
        print(f"  worst 5 db_ids by ok-rate:")
        for db, k, t, r in ratios[:5]:
            print(f"    {db}: {k}/{t} ({r:.1f}%)")
        print()
    print("========================================================\n")


if __name__ == "__main__":
    main()
