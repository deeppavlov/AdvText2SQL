"""Manual schema inventory or fresh paired train repeats; never changes v2 labels."""

import argparse
from collections import Counter, defaultdict
import json
import os

import collect_classifier_v2 as collect
import prepare_classifier_data_stage as stage
from durable_prompt_store import write_json

data = stage.data


def export_repeats(root, manifest):
    examples = {r["id"]: r for r in (
        json.loads(s) for s in (data.ROOT / data.DATA_OUTPUT / "examples.jsonl").read_text(encoding="utf-8").splitlines())}
    points = {}
    for task, ids in manifest["selection"].items():
        for identity, reasons in ids.items():
            for config in data.TASK_CONFIGURATIONS[task]:
                first = examples[identity]["observations"][config]
                points[identity, config] = {"id": identity, "benchmark": task, "configuration": config,
                    "selection_reasons": reasons, "observations": [{"repeat": 1, "score": first["score"],
                    "source": first["source"], "key": first["key"]}]}
    completed = []
    for job in manifest["jobs"]:
        if collect.completed(root, manifest, job):
            completed.append(job["name"])
        path = root / "jobs" / job["name"] / "evaluations.json"
        if not path.exists():
            continue
        values = data.read(path)
        allowed = {collect.expected_key(manifest, job, item): item for item in job["items"]}
        if not set(values) <= set(allowed):
            raise ValueError("Unexpected repeat checkpoint IDs")
        for key, value in values.items():
            item = allowed[key]
            score = data.validate_observation(value, item, job["benchmark"])
            if score is None:
                raise ValueError("Unknown infrastructure failure is not a model label")
            identity = f"{job['benchmark']}:{item['question_id']}"
            points[identity, job["configuration"]]["observations"].append({"repeat": job["repeat"], "score": score,
                "source": path.relative_to(data.ROOT).as_posix(), "key": key})
    records = []
    for (identity, config), point in points.items():
        observations = point["observations"]
        if len({v["repeat"] for v in observations}) != len(observations):
            raise ValueError("Repeated pass number in a configuration point")
        point["mean_quality_observed_passes"] = sum(v["score"] for v in observations) / len(observations)
        point["outcome_disagreement"] = len({v["score"] for v in observations}) > 1
        anchor = "compact" if point["benchmark"] == "bird" else "rules_taxonomy"
        anchor_scores = {v["repeat"]: v["score"] for v in points[identity, anchor]["observations"]}
        paired = [v["score"] - anchor_scores[v["repeat"]] for v in observations if v["repeat"] in anchor_scores]
        point["paired_passes_with_anchor"] = len(paired)
        point["mean_paired_gain_observed_passes"] = sum(paired) / len(paired)
        records.append(point)
    report = {"version": stage.VERSION, "manifest_sha256": manifest["sha256"],
              "scope": "selected train uncertainty audit, not population EX or independent test; no best-of-repeat selection",
              "completed_jobs": completed, "remaining_jobs": len(manifest["jobs"]) - len(completed),
              "new_saved_evaluations": sum(len(p["observations"]) - 1 for p in records),
              "three_pass_points": sum(len(p["observations"]) == 3 for p in records),
              "points_with_outcome_disagreement": sum(p["outcome_disagreement"] for p in records),
              "dataset_rewritten": False, "records": records}
    write_json(root / "repeat_report.json", report)
    return report


def inventory_snapshot(pilot, inspect_database):
    """Catalog introspection only; SQL references are parsed, never executed."""
    from audit_question_schemas import audit_items

    snapshots, availability, audits = {}, {}, []
    by_db = defaultdict(list)
    for row in pilot:
        by_db[row["db_id"]].append(row["item"])
    for db, items in sorted(by_db.items()):
        rows = inspect_database(db)
        if rows is None:
            availability[db] = "database_missing"
        elif not rows:
            availability[db] = "empty_schema"
        else:
            snapshots[db] = {"rows": rows, "schema_sha256": data.fingerprint(rows)}
            availability[db] = "schema_present"
            audits.extend(audit_items(items, snapshots))
    return {"schemas": snapshots, "availability": availability, "questions": audits,
            "approved_reference_names": sum(v["status"] == "reference_consistent" for v in audits),
            "reference_sql_executed": False, "llm_calls": 0,
            "training_ready": False, "note": "Structural name consistency alone does not establish reference execution correctness."}


def snapshot(root):
    audit = data.read(root / "source_audit.json")
    pilot = data.read(root / "ambrosia_pilot.json")
    for path, sha in audit["source_sha256"].items():
        if data.digest(data.local_path(data.ROOT, path)) != sha:
            raise ValueError("Prepared source changed")
    if data.fingerprint(pilot) != audit["pilot_fingerprint"]:
        raise ValueError("Pilot inputs changed")
    if (root / "schema_inventory.json").exists():
        raise ValueError("Schema inventory already exists; preserve it and prepare a new directory for another snapshot")
    from dotenv import load_dotenv
    from sqlalchemy import inspect, text
    from sqlalchemy.engine import URL
    from baseline_connection import database_engine

    load_dotenv(data.ROOT / ".env.deepseek", override=True)

    # Query existence through a known project DB: startup failures may omit SQLSTATE.
    # Only availability of selected pilot DBs is exported, never the full server inventory.
    known = data.read(root / "repeats/experiment.json")["runtime_base"]["schema_audit"]["schemas"]["bird"]
    known_db = "student_club" if "student_club" in known else sorted(known)[0]
    engine = database_engine(URL.create("postgresql+psycopg", username=os.environ["DB_USER"],
        password=os.environ["DB_PASS"], host="127.0.0.1", port=5444, database=known_db))
    try:
        with engine.connect() as connection, connection.begin():
            connection.execute(text("SET TRANSACTION READ ONLY"))
            available = set(connection.execute(text("SELECT datname FROM pg_database WHERE datallowconn")).scalars())
    except Exception as error:
        raise RuntimeError(f"Schema inventory connection failed ({type(error).__name__}); no LLM calls") from None
    finally:
        engine.dispose()

    def inspect_database(db):
        if db not in available:
            return None
        engine = database_engine(URL.create("postgresql+psycopg", username=os.environ["DB_USER"],
            password=os.environ["DB_PASS"], host="127.0.0.1", port=5444, database=db))
        try:
            with engine.connect() as connection, connection.begin():
                connection.execute(text("SET TRANSACTION READ ONLY"))
                inspector = inspect(connection)
                return sorted((t, c["name"], str(c["type"])) for t in inspector.get_table_names(schema="public")
                              for c in inspector.get_columns(t, schema="public"))
        except Exception as error:
            if getattr(getattr(error, "orig", None), "sqlstate", None) == "3D000":
                return None
            raise RuntimeError(f"Schema inventory stopped for {db} ({type(error).__name__}); no LLM calls") from None
        finally:
            engine.dispose()

    result = inventory_snapshot(pilot, inspect_database)
    result["source_audit_sha256"] = data.digest(root / "source_audit.json")
    result["pilot_fingerprint"] = audit["pilot_fingerprint"]
    write_json(root / "schema_inventory.json", result)
    print("Schema inventory:", dict(Counter(result["availability"].values())),
          "structurally consistent questions:", result["approved_reference_names"], "; LLM calls: 0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=stage.DEFAULT_OUTPUT)
    actions = parser.add_mutually_exclusive_group(required=True)
    for flag in ("run", "dry-run", "export", "snapshot"):
        actions.add_argument("--" + flag, action="store_true")
    parser.add_argument("--benchmark", choices=("bird", "ambrosia", "both"), default="both")
    parser.add_argument("--configuration", choices=tuple(data.CONFIGURATIONS))
    parser.add_argument("--repeat", type=int, choices=(2, 3))
    parser.add_argument("--max-jobs", type=int, default=1)
    args = parser.parse_args()
    stage_root = data.local_path(data.ROOT, args.output)
    if not stage_root.is_relative_to(data.ROOT / "research_results") or stage_root == data.ROOT / "research_results":
        parser.error("Use a dedicated stage subfolder")
    if args.snapshot:
        snapshot(stage_root)
        return
    root = stage_root / "repeats"
    manifest = data.read(root / "experiment.json")
    stage.validate(manifest)
    from run_feature_ablation import suite_lock
    if args.export:
        with suite_lock(root):
            report = export_repeats(root, manifest)
        print("New saved evaluations:", report["new_saved_evaluations"], "remaining jobs:", report["remaining_jobs"])
        return
    if args.max_jobs < 1:
        parser.error("max-jobs must be positive")
    jobs = [j for j in manifest["jobs"] if (args.benchmark == "both" or j["benchmark"] == args.benchmark)
            and (not args.configuration or j["configuration"] == args.configuration)
            and (not args.repeat or j["repeat"] == args.repeat) and not collect.completed(root, manifest, j)]
    jobs = jobs[:args.max_jobs]
    for job in jobs:
        print(job["name"], "questions:", len(job["items"]), "features:", job["features"])
    if args.dry_run or not jobs:
        print("OFFLINE; selected jobs:", len(jobs), "; no API/SSH/DB calls")
        return
    collect.check_packages(manifest)
    with suite_lock(root):
        for job in jobs:
            collect.run_job(root, manifest, job)
            report = export_repeats(root, manifest)
            print("Saved:", report["new_saved_evaluations"], "remaining jobs:", report["remaining_jobs"])


if __name__ == "__main__":
    main()
