"""Connection-only revision of the guarded runtime; no change to SQL/LLM behavior."""

import argparse
import asyncio
import os
from pathlib import Path
import time

from atomic_artifacts import write_json
import guarded_hybrid_runtime as guarded
import static_hybrid_runtime as legacy
import static_hybrid_search as search

ATTEMPTS = 5
RETRY_DELAY = 5


def connection_reason(error):
    """Return only a fixed diagnostic code; exception strings can contain credentials."""
    original = getattr(error, "orig", error)
    sqlstate = getattr(original, "sqlstate", None)
    message = str(original).lower()
    if sqlstate in ("28P01", "28000") or "password authentication failed" in message:
        return "authentication_failed", False
    if sqlstate == "3D000" or ("database" in message and "does not exist" in message):
        return "database_missing", False
    if sqlstate == "42501" or "permission denied" in message:
        return "permission_denied", False
    if "timeout" in message or "timed out" in message or sqlstate == "57014":
        return "connection_or_statement_timeout", True
    if "connection refused" in message or "actively refused" in message:
        return "tunnel_port_unavailable", True
    if "server closed" in message or "connection is closed" in message:
        return "connection_closed", True
    return "database_connection_error", True


def validate_live_schemas(
    items,
    schemas,
    *,
    progress=None,
    attempts=ATTEMPTS,
    retry_delay=RETRY_DELAY,
    sleep=time.sleep,
):
    from sqlalchemy import inspect
    from sqlalchemy.engine import URL
    from sqlalchemy.exc import InterfaceError, OperationalError, TimeoutError
    from baseline_connection import database_engine

    progress = progress or (lambda event: None)
    databases = sorted({item["db_id"] for item in items})
    for index, db in enumerate(databases, 1):
        for attempt in range(1, attempts + 1):
            event = {
                "phase": "schema_preflight",
                "current_database": db,
                "databases_ready": index - 1,
                "databases_total": len(databases),
                "preflight_attempt": attempt,
                "preflight_attempts": attempts,
            }
            progress(event)
            print(
                f"[SCHEMA {index}/{len(databases)}] {db}, attempt {attempt}/{attempts}",
                flush=True,
            )
            engine = database_engine(
                URL.create(
                    "postgresql+psycopg",
                    username=os.environ["DB_USER"],
                    password=os.environ["DB_PASS"],
                    host="127.0.0.1",
                    port=5444,
                    database=db,
                )
            )
            retry = False
            try:
                # A single connection per database, instead of a new NullPool
                # connection/channel for each table and inspector operation.
                with engine.connect() as connection:
                    connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                    inspector = inspect(connection)
                    rows = sorted(
                        (table, col["name"], str(col["type"]))
                        for table in inspector.get_table_names(schema="public")
                        for col in inspector.get_columns(table, schema="public")
                    )
                if rows != sorted(tuple(row) for row in schemas[db]["rows"]):
                    progress({**event, "event": "schema_mismatch"})
                    raise ValueError(
                        f"Live schema changed for {db}; re-audit before paid generation"
                    )
            except (OperationalError, InterfaceError, TimeoutError, OSError) as error:
                reason, retryable = connection_reason(error)
                retry = retryable and attempt < attempts
                progress(
                    {
                        **event,
                        "event": "connection_error",
                        "reason": reason,
                        "error_type": type(error).__name__,
                        "will_retry": retry,
                    }
                )
                if not retry:
                    raise RuntimeError(
                        f"Schema preflight failed for {db}: {reason} after {attempt} attempt(s); "
                        "no paid generation started. Inspect the SSH tunnel and database access."
                    ) from None
                print(
                    f"[SCHEMA RETRY] {db}: {reason}; retry in {retry_delay}s (no LLM calls)",
                    flush=True,
                )
            except ValueError:
                raise
            except Exception as error:
                progress(
                    {
                        **event,
                        "event": "catalog_error",
                        "error_type": type(error).__name__,
                    }
                )
                raise RuntimeError(
                    f"Catalog inspection failed for {db} ({type(error).__name__}); no paid generation started"
                ) from None
            finally:
                engine.dispose()
            if retry:
                sleep(retry_delay)
            else:
                progress(
                    {**event, "event": "schema_verified", "databases_ready": index}
                )
                break


async def run(args):
    output = Path(args.search).resolve()
    if not output.is_relative_to(
        search.ROOT / "research_results"
    ) or args.scope not in ("train", "dev"):
        raise ValueError("Use a train/dev campaign folder inside research_results")
    manifest = search.read(output / "experiment.json")
    guarded.validate(manifest)
    if (
        manifest["protocol"].get("connection_preflight", {}).get("version")
        != "resilient-catalog-v1"
    ):
        raise ValueError(
            "Use the separately prepared resilient campaign; never rewrite an old manifest"
        )
    case = next(c for c in manifest["cases"] if c["name"] == args.case)
    folder = search.run_folder(output, args.scope, args.case, args.benchmark)
    folder.mkdir(parents=True, exist_ok=True)
    metadata_path = folder / "metadata.json"
    existing = search.read(metadata_path) if metadata_path.exists() else None
    identity = search.identity(manifest, args.scope, case, args.benchmark)
    if existing and existing["configuration"] != identity:
        raise ValueError("Existing checkpoint identity differs from resilient campaign")
    if existing and existing["status"] == "completed":
        search.result(output, manifest, args.scope, case, args.benchmark)
        return
    expected_ids = {
        str(i["question_id"]) for i in manifest["cohorts"][args.scope][args.benchmark]
    }
    saved = legacy.checkpoint(folder, expected_ids)
    if existing:
        legacy.append(folder / "attempt_history.jsonl", existing)
    metadata = {
        "configuration": identity,
        "status": "preflight",
        "phase": "schema_preflight",
        "questions": len(expected_ids),
        "saved": len(saved),
        "started_at_utc": search.now(),
    }
    write_json(metadata_path, metadata)

    def progress(event):
        metadata.update(event, last_preflight_activity_utc=search.now())
        write_json(metadata_path, metadata)
        legacy.append(
            folder / "schema_preflight.jsonl", {"at_utc": search.now(), **event}
        )

    original = guarded.validate_live_schemas
    guarded.validate_live_schemas = lambda items, schemas: validate_live_schemas(
        items, schemas, progress=progress
    )
    try:
        await guarded.run(args)
    except BaseException as error:
        # Preserve preflight failure without replacing an exception by an I/O
        # error for a directory that the child never got a chance to create.
        current = search.read(metadata_path) if metadata_path.exists() else metadata
        if current.get("phase") == "schema_preflight":
            current.update(
                status="preflight_failed",
                error_type=type(error).__name__,
                finished_at_utc=search.now(),
            )
            write_json(metadata_path, current)
        raise
    finally:
        guarded.validate_live_schemas = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", required=True)
    parser.add_argument("--env-file", default=".env.deepseek")
    parser.add_argument("--scope", choices=("train", "dev"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--benchmark", choices=search.BENCHMARKS, required=True)
    asyncio.run(run(parser.parse_args()))
