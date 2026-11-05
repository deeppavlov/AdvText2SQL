"""Versioned adapter: fail invalid outputs per question and verify live schemas before paid calls."""

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from sql_output_contract import guarded_response
from guarded_hybrid_search import validate
import static_hybrid_runtime as legacy
import static_hybrid_search as search

ORIGINAL_PREDICT = legacy.predict


async def guarded_predict(
    items, tools, predictions, folder, benchmark, context, progress, **kwargs
):
    class GuardedTool:
        def __init__(self, tool):
            self.tool = tool

        async def query(self, question):
            original = await self.tool.query(question)
            response, issue = guarded_response(original)
            if issue:
                current = context.get()
                legacy.append(
                    folder / "invalid_outputs.jsonl",
                    {
                        "question_id": current["question_id"],
                        "db_id": current["db_id"],
                        "reason": issue,
                        "original_response": original,
                        "recorded_as": "error",
                        "extra_llm_calls": 0,
                    },
                )
            return response

    # The frozen predictor handles checkpoint/usage exactly as before. Only model
    # output conversion differs; no extra retries, SQL repair, gold or EXPLAIN.
    return await ORIGINAL_PREDICT(
        items,
        {db: GuardedTool(t) for db, t in tools.items()},
        predictions,
        folder,
        benchmark,
        context,
        progress,
        **kwargs,
    )


def validate_live_schemas(items, schemas):
    from sqlalchemy import inspect
    from sqlalchemy.engine import URL
    from baseline_connection import database_engine

    for db in sorted({i["db_id"] for i in items}):
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
        try:
            inspector = inspect(engine)
            rows = sorted(
                (table, col["name"], str(col["type"]))
                for table in inspector.get_table_names(schema="public")
                for col in inspector.get_columns(table, schema="public")
            )
            expected = sorted(tuple(row) for row in schemas[db]["rows"])
            if rows != expected:
                raise ValueError(
                    f"Live schema changed for {db}; re-audit before paid generation"
                )
        except ValueError:
            raise
        except Exception as error:
            raise RuntimeError(
                f"Schema preflight failed for {db} ({type(error).__name__}); no paid generation started"
            ) from None
        finally:
            engine.dispose()


async def run(args):
    output = Path(args.search).resolve()
    manifest = search.read(output / "experiment.json")
    validate(manifest)
    if manifest["protocol"].get("version") != "guarded-hybrid-v3":
        raise ValueError(
            "Guarded runtime requires a v3 protocol, never a historical v2 output folder"
        )
    items = manifest["cohorts"][args.scope][args.benchmark]
    issues = manifest["schema_audit"]["blocked_questions"][args.benchmark]
    if any(str(i["question_id"]) in issues for i in items):
        raise ValueError(
            "Selected questions do not match saved schemas; restore original DBs or explicitly prepare an excluded cohort"
        )
    folder = search.run_folder(output, args.scope, args.case, args.benchmark)
    if (folder / "metadata.json").exists() and search.read(folder / "metadata.json")[
        "status"
    ] == "completed":
        return  # The parent already validates completed reports/checkpoints.
    load_dotenv(search.ROOT / args.env_file, override=True)
    # Catalog introspection only. No reference SQL is executed or passed to LLM.
    validate_live_schemas(items, manifest["schema_audit"]["schemas"][args.benchmark])
    previous = legacy.predict
    legacy.predict = guarded_predict
    try:
        await legacy.run(args)
    finally:
        legacy.predict = previous


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", required=True)
    parser.add_argument("--env-file", default=".env.deepseek")
    parser.add_argument("--scope", choices=search.SCOPES, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--benchmark", choices=search.BENCHMARKS, required=True)
    asyncio.run(run(parser.parse_args()))
