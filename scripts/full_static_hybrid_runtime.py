"""Full-cohort evaluation using frozen inference, schema and transport guards."""

import argparse
import asyncio

from atomic_artifacts import write_json
import guarded_hybrid_runtime as guarded
import resilient_hybrid_runtime as resilient
import static_hybrid_runtime as legacy
import static_hybrid_search as search
from transport_safe_runtime import TransportRecordingClient


async def run(args):
    from full_static_hybrid import validate, checked_output

    output = checked_output(args.search)
    manifest = search.read(output / "experiment.json")
    validate(manifest)
    if args.scope != "full":
        raise ValueError("This runtime is for the separately frozen full cohort")
    case = next(c for c in manifest["cases"] if c["name"] == args.case)
    folder = search.run_folder(output, "full", case["name"], args.benchmark)
    folder.mkdir(parents=True, exist_ok=True)
    identity = search.identity(manifest, "full", case, args.benchmark)
    path = folder / "metadata.json"
    existing = search.read(path) if path.exists() else None
    if existing and existing["configuration"] != identity:
        raise ValueError("Checkpoint differs from the frozen full protocol")
    if existing and existing["status"] == "completed":
        search.result(output, manifest, "full", case, args.benchmark)
        return
    items = manifest["cohorts"]["full"][args.benchmark]
    saved = legacy.checkpoint(folder, {str(i["question_id"]) for i in items})
    if existing:
        legacy.append(folder / "attempt_history.jsonl", existing)
    metadata = {
        "configuration": identity,
        "status": "preflight",
        "phase": "schema_preflight",
        "questions": len(items),
        "saved": len(saved),
        "started_at_utc": search.now(),
    }
    write_json(path, metadata)

    def progress(event):
        metadata.update(event, last_preflight_activity_utc=search.now())
        write_json(path, metadata)
        legacy.append(
            folder / "schema_preflight.jsonl", {"at_utc": search.now(), **event}
        )

    original_preflight, original_recorder = (
        guarded.validate_live_schemas,
        legacy.RecordingClient,
    )
    guarded.validate_live_schemas = (
        lambda rows, schemas: resilient.validate_live_schemas(
            rows, schemas, progress=progress
        )
    )
    legacy.RecordingClient = TransportRecordingClient
    try:
        await guarded.run(args)
    except BaseException as error:
        current = search.read(path)
        if current.get("phase") == "schema_preflight":
            current.update(
                status="preflight_failed",
                error_type=type(error).__name__,
                finished_at_utc=search.now(),
            )
            write_json(path, current)
        raise
    finally:
        guarded.validate_live_schemas, legacy.RecordingClient = (
            original_preflight,
            original_recorder,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", required=True)
    parser.add_argument("--env-file", default=".env.deepseek")
    parser.add_argument("--scope", choices=("full",), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--benchmark", choices=search.BENCHMARKS, required=True)
    asyncio.run(run(parser.parse_args()))
