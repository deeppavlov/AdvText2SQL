"""Manual audit regression: rescore frozen answers, then measure two fixed static profiles."""

import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
import importlib.metadata
import os
import sys
import time
from unittest.mock import patch

from dotenv import dotenv_values, load_dotenv

import bird_compact_transfer as shared
from run_feature_ablation import suite_lock

source, runtime, base = shared.source, shared.runtime, shared.base
read, digest, store = shared.read, shared.digest, shared.store
fingerprint = source.search.fingerprint
ROOT = source.ROOT
VERSION = "static-pipeline-audit-v2"
DEFAULT_OUTPUT = "research_results/static_pipeline_audit_20261116_r2"
PREVIOUS_CODE = ".cache/static_pipeline_audit_before_recovery_20261116"
RECOVERY_CHANGES = {
    "scripts/static_pipeline_regression.py", "scripts/prompt_pilot_runtime.py",
    "benchmarks/evaluate_bird.py", "src/adv_text2sql/sql_safety.py",
}
SOURCES = {
    "bird": "research_results/bird_prompt_full_20261107",
    "ambrosia": "research_results/ambrosia_prompt_full_20261107",
}
FEATURES = {"bird": [2, 7, 14, 32, 35], "ambrosia": [14, 32]}
COUNTS = {"bird": 241, "ambrosia": 211}
LIMITS = {
    "bird": dict(examples=241, task_calls=1200, teacher_calls=0, tokens=4_500_000, active_seconds=10800),
    "ambrosia": dict(examples=211, task_calls=800, teacher_calls=0, tokens=2_000_000, active_seconds=7200),
}
PROMPTS = "src/adv_text2sql/mcp_servers/text2sql_tool/src/prompts.py"
PACKAGES = ("sqlglot", "sqlalchemy", "autogen-ext", "autogen-core", "openai", "pydantic")


def output_path(value):
    path = (ROOT / value).resolve()
    if path.parent != (ROOT / "research_results").resolve() or not path.name.startswith("static_pipeline_audit_"):
        raise ValueError("Use a dedicated research_results/static_pipeline_audit_<name> folder")
    return path


def files():
    names = (
        "static_pipeline_regression.py", "prompt_comparison.py", "prompt_comparison_runtime.py",
        "prompt_pilot.py", "prompt_pilot_runtime.py", "prompt_pilot_budget.py",
        "bird_compact_transfer.py", "durable_prompt_store.py", "sql_output_contract.py",
        "transport_safe_runtime.py", "resilient_hybrid_runtime.py", "guarded_hybrid_runtime.py",
    )
    paths = [*source.search.code_paths(), *(ROOT / "scripts" / p for p in names),
             ROOT / "local/run_static_pipeline_regression.sh"]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def source_snapshot(benchmark):
    folder = ROOT / SOURCES[benchmark]
    old = read(folder / "experiment.json")
    if old["sha256"] != fingerprint({k: v for k, v in old.items() if k != "sha256"}):
        raise ValueError("Historical manifest fingerprint changed")
    if old["files_sha256"][PROMPTS] != digest(ROOT / PROMPTS):
        raise ValueError("Original prompt templates changed; do not silently transfer the old result")
    items = old["full"][benchmark]
    if len(items) != COUNTS[benchmark] or len({source.item_id(i) for i in items}) != len(items):
        raise ValueError("Historical full question coverage changed")
    if old["feature_mask"] != FEATURES[benchmark]:
        raise ValueError("Historical feature mask changed")
    job = old["jobs"][0]
    arm = "selected_sql" if benchmark == "bird" else "original"
    value = job["arms"][arm]
    if value["demos"]:
        raise ValueError("The agreed fixed profiles have no demonstrations")
    meta_path = folder / "jobs" / job["name"] / "confirm/metadata.json"
    meta = read(meta_path)
    if meta.get("status") != "completed" or meta.get("manifest_sha256") != old["sha256"]:
        raise ValueError("Historical full checkpoint is incomplete or foreign")
    records_path = meta_path.parent.parent / "evaluations.json"
    records = read(records_path)
    expected = {key(candidate, item, benchmark) for candidate in job["arms"].values() for item in items}
    if set(records) != expected:
        raise ValueError("Historical full checkpoint keys differ")
    historical = {source.item_id(i): records[key(value, i, benchmark)] for i in items}
    for item in items:
        validate_result(historical[source.item_id(item)], item)
    slot = "sql" if benchmark == "bird" else "ambiguity"
    templates = source.all_templates()
    if benchmark == "bird":
        template, demos = source.render(value, old, slot, old["cohorts"][slot]["fit"], items[0])
        if demos:
            raise ValueError("Unexpected demonstration")
        templates["SQL_PROMPT_TEMPLATE"] = template
    else:
        if old["fixed_prompts"] != {name: templates[name] for name in old["fixed_prompts"]}:
            raise ValueError("Ambrosia fixed prompts changed")
        templates.update(old["fixed_prompts"])
    paths = [folder / "experiment.json", records_path, meta_path]
    return {
        "source_files_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
        "items": items, "features": FEATURES[benchmark], "candidate": value,
        "historical": historical, "frozen_prompts": templates,
        "cohorts": old["cohorts"], "seed_instructions": old["seed_instructions"],
        "schema_audit": old["schema_audit"], "model": old["model"],
    }


def key(value, item, benchmark):
    return f"confirm:{source.cid(value)}:hybrid:False:{benchmark}:{source.item_id(item)}"


def validate_result(row, item):
    if (row.get("question_id") != source.item_id(item) or row.get("db_id") != item["db_id"]
            or not isinstance(row.get("prediction"), str) or not row["prediction"].strip()
            or type(row.get("score")) not in (int, float) or row["score"] not in (0, 1)):
        raise ValueError("Invalid or foreign scored answer")


def make_manifest(env_file):
    from adv_text2sql.bird_results import EX_POLICY, EXECUTION_POLICY, DIAGNOSTIC_POLICY
    env_path = (ROOT / env_file).resolve()
    env_name = env_path.relative_to(ROOT).as_posix()
    snapshots = {b: source_snapshot(b) for b in FEATURES}
    model = source.pilot.public_model(dotenv_values(env_path))
    if any(model != spec["model"] for spec in snapshots.values()):
        raise ValueError("Selected env model/provider/settings differ from the historical static controls")
    value = {
        "version": VERSION, "created_at_utc": source.search.now(), "env_file": env_name,
        "benchmarks": snapshots, "limits": LIMITS, "files_sha256": files(),
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "policy": {
            "bird_evaluation": {"ex": EX_POLICY, "postgresql_execution": EXECUTION_POLICY,
                                "numeric_diagnostic": DIAGNOSTIC_POLICY, "diagnostic_affects_score": False},
            "scope": "retrospective full regression; no optimization, classifier or deployment",
            "comparison": "A historical score; B historical prediction with current evaluator; C fresh prediction with current evaluator",
            "reference_errors": "block paid generation; no question exclusion or denominator changes",
            "runtime": "same task-only Session as full controls; temperature=0, max_tokens=4096, client retries=0, 3s pacing",
            "model_metadata": "historical public_model max_tokens=2048 is nominal; inherited task client actually uses 4096",
        },
    }
    return {**value, "sha256": fingerprint(value)}


def validate(manifest, *, model=False):
    if (manifest.get("version") != VERSION or manifest.get("limits") != LIMITS
            or manifest.get("sha256") != fingerprint({k: v for k, v in manifest.items() if k != "sha256"})):
        raise ValueError("Audit protocol changed; use a new output folder")
    if manifest["files_sha256"] != files():
        raise ValueError("Runtime source files changed; prepare a new audit folder")
    if manifest["packages"] != {name: importlib.metadata.version(name) for name in PACKAGES}:
        raise ValueError("Runtime dependency versions changed")
    for benchmark, spec in manifest["benchmarks"].items():
        for name, sha in spec["source_files_sha256"].items():
            if digest(ROOT / name) != sha:
                raise ValueError("Historical source artifact changed")
        if model and source.pilot.public_model(dotenv_values(ROOT / manifest["env_file"])) != spec["model"]:
            raise ValueError(f"Model/provider differs for {benchmark}")


def declaration(path, name, parent=None):
    nodes = ast.parse(path.read_text(encoding="utf8")).body
    if parent:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == parent).body
    return next(n for n in nodes if getattr(n, "name", None) == name)


def verify_recovery_code(previous, manifest, archived):
    """Attest a narrow correction; never call arbitrary old code or relax hashes."""
    before, after = previous["files_sha256"], manifest["files_sha256"]
    if set(before) != set(after) or previous["packages"] != manifest["packages"]:
        raise ValueError("Source inventory or dependencies changed; do not import old scores")
    changed = {name for name in before if before[name] != after[name]}
    if not changed <= RECOVERY_CHANGES:
        raise ValueError("Recovery includes unrelated runtime changes")
    for name, sha in before.items():
        if digest(archived / name) != sha:
            raise ValueError("Previous code archive does not match the frozen source SHA")
    path = "benchmarks/evaluate_bird.py"
    old = declaration(archived / path, "sqlite_to_postgres")
    new = declaration(ROOT / path, "sqlite_to_postgres")
    expected_guard = ast.parse("validate_read_query(query, dialect='sqlite')").body[0]
    if ast.dump(new.body[1]) != ast.dump(expected_guard):
        raise ValueError("Expected only a SQLite source safety check before conversion")
    del new.body[1]
    if ast.dump(old) != ast.dump(new):
        raise ValueError("SQL translation changed; saved scores require DB reevaluation")
    for path, name, parent in (
        ("scripts/prompt_pilot_runtime.py", "score", "Session"),
        ("scripts/prompt_pilot_runtime.py", "prepare_tools", "Session"),
        ("src/adv_text2sql/sql_safety.py", "read_only_connection", None),
        ("scripts/static_pipeline_regression.py", "AuditSession", None),
    ):
        if ast.dump(declaration(archived / path, name, parent)) != ast.dump(declaration(ROOT / path, name, parent)):
            raise ValueError("Execution/scoring/generation changed; do not transfer saved scores")
    return {name: {"before": before[name], "after": after[name]} for name in sorted(changed)}


def recover_reevaluation(parent, root, manifest, archived):
    """Copy verified scores to a new protocol; preserve all previous artifacts."""
    from benchmarks.evaluate_bird import sqlite_to_postgres
    from adv_text2sql.sql_safety import validate_read_query

    if parent == root:
        raise ValueError("Recovery must use a separate output folder")
    previous = read(parent / "experiment.json")
    if (previous.get("version") != "static-pipeline-audit-v1"
            or previous["sha256"] != fingerprint({k: v for k, v in previous.items() if k != "sha256"})):
        raise ValueError("Unverified previous protocol")
    if previous["benchmarks"] != manifest["benchmarks"] or previous["limits"] != manifest["limits"]:
        raise ValueError("Questions, model, prompts, schemas, predictions or limits changed")
    changes = verify_recovery_code(previous, manifest, archived)
    if list(parent.rglob("budget.json")) or list(parent.glob("*/jobs/*/evaluations.json")):
        raise ValueError("This transfer is restricted to LLM-free reevaluation; paid ledgers cannot be rebound")
    if (root / "checkpoint_transfer.json").exists():
        raise ValueError("Recovery already recorded; continue reevaluation instead of importing again")
    plan = {}
    for benchmark, spec in manifest["benchmarks"].items():
        if rescore_path(root, benchmark).exists():
            raise ValueError("Destination already contains reevaluation answers; never overwrite them")
        old_records = rescore_records(parent, previous, benchmark)
        carried, retry_ids = {}, []
        lookup = {source.item_id(i): i for i in spec["items"]}
        for qid, row in old_records.items():
            if row["status"] != "scored":
                retry_ids.append(qid)
                continue
            if benchmark == "bird":
                # Conversion body, dependencies and PostgreSQL execution/scorer
                # were proven unchanged above; newly check the input grammar.
                validate_read_query(sqlite_to_postgres(lookup[qid]["SQL"]))
            carried[qid] = row
        plan[benchmark] = {"carried": carried, "retry_ids": retry_ids}
    archive = root / "recovery_source"
    archive.mkdir(exist_ok=True)
    source_hashes = {}
    for path in [parent / "experiment.json", *parent.glob("*/reevaluation*.json")]:
        relative = path.relative_to(parent)
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        store.write_bytes(target, path.read_bytes())
        source_hashes[relative.as_posix()] = digest(path)
    journal = {"manifest_sha256": manifest["sha256"], "previous_manifest_sha256": previous["sha256"],
               "previous_path": parent.relative_to(ROOT).as_posix(), "previous_files_sha256": source_hashes,
               "code_changes": changes, "paid_answers_transferred": 0,
               "benchmarks": {b: {"carried_ids": list(p["carried"]), "retry_ids": p["retry_ids"]} for b, p in plan.items()}}
    store.write_json(root / "checkpoint_transfer.json", journal)
    for benchmark, value in plan.items():
        folder = root / benchmark
        folder.mkdir(exist_ok=True)
        store.write_json(rescore_path(root, benchmark), {
            "manifest_sha256": manifest["sha256"], "records": value["carried"],
            "records_sha256": fingerprint(value["carried"]), "recovery": journal["previous_manifest_sha256"]})
        print(f"Recovered {benchmark}: {len(value['carried'])} scored answers; "
              f"{len(value['retry_ids'])} old reference errors will be checked again; no API/DB calls")


def rescore_path(root, benchmark):
    return root / benchmark / "reevaluation.json"


def rescore_records(root, manifest, benchmark):
    path = rescore_path(root, benchmark)
    if not path.exists():
        return {}
    value = read(path)
    if (value.get("manifest_sha256") != manifest["sha256"]
            or value.get("records_sha256") != fingerprint(value["records"])):
        raise ValueError("Reevaluation checkpoint identity/hash changed")
    spec = manifest["benchmarks"][benchmark]
    items = {source.item_id(i): i for i in spec["items"]}
    if not set(value["records"]) <= set(items):
        raise ValueError("Foreign reevaluation IDs")
    for qid, row in value["records"].items():
        if row["prediction"] != spec["historical"][qid]["prediction"]:
            raise ValueError("Reevaluation changed a frozen prediction")
        if row.get("status") == "scored":
            validate_result(row, items[qid])
        elif row.get("status") != "reference_error" or row.get("score") is not None:
            raise ValueError("Invalid reevaluation outcome")
    return value["records"]


def transient_reference_failure(details):
    code = details["sqlstate"]
    return bool((isinstance(code, str) and code.startswith("08"))
            or (code is None and {details["error_type"], details.get("exception_type")}
                & {"OperationalError", "InterfaceError", "TimeoutError"}))


def reference_rows(engine, sql, *, on_failure, sleep=None, attempts=3, on_types=None):
    """Retry only confirmed connection failures, with fresh read-only transactions."""
    from sqlalchemy import text
    from adv_text2sql.sql_safety import read_only_connection
    from adv_text2sql.evaluation import database_failure
    from adv_text2sql.bird_results import bird_column_types, prepare_bird_evaluation

    sleep = sleep or time.sleep
    for attempt in range(1, attempts + 1):
        try:
            with read_only_connection(engine, sql) as connection:
                prepare_bird_evaluation(connection)
                result = connection.execute(text(sql))
                if on_types is not None:
                    on_types(bird_column_types(result))
                return base.normalize(result.fetchall())
        except Exception as error:
            details = {**database_failure(error), "exception_type": type(error).__name__}
            retry = attempt < attempts and transient_reference_failure(details)
            on_failure({**details, "attempt": attempt, "will_retry": retry})
            if not retry:
                raise
            engine.dispose()
            sleep(5 * attempt)


def reevaluate(root, manifest, benchmark):
    """No LLM client constructed; execute saved SQL with the current real scorer."""
    from sqlalchemy.engine import URL
    from baseline_connection import database_engine
    from resilient_hybrid_runtime import validate_live_schemas
    from benchmarks.evaluate_bird import sqlite_to_postgres
    from adv_text2sql.evaluation import database_failure

    spec = manifest["benchmarks"][benchmark]
    folder = root / benchmark
    folder.mkdir(exist_ok=True)
    records = rescore_records(root, manifest, benchmark)
    scorer = object.__new__(base.Session)
    scorer.engines, scorer.golds, scorer.gold_type_oids = {}, {}, {}
    progress = {"manifest_sha256": manifest["sha256"], "phase": "reevaluating", "saved": len(records)}

    def save():
        store.write_json(rescore_path(root, benchmark), {
            "manifest_sha256": manifest["sha256"], "records": records, "records_sha256": fingerprint(records)})
        store.write_json(folder / "reevaluation_progress.json", progress)

    save()
    try:
        if benchmark == "bird":
            validate_live_schemas(spec["items"], spec["schema_audit"]["schemas"][benchmark])
        for item in spec["items"]:
            qid, db = source.item_id(item), item["db_id"]
            if qid in records:
                continue
            prediction = spec["historical"][qid]["prediction"]
            row = dict(question_id=qid, db_id=db, prediction=prediction)
            progress.update(current_question=qid, current_database=db)
            if benchmark == "bird":
                if db not in scorer.engines:
                    scorer.engines[db] = database_engine(URL.create(
                        "postgresql+psycopg", username=os.environ["DB_USER"], password=os.environ["DB_PASS"],
                        host="127.0.0.1", port=5444, database=db))
                def failure(details):
                    progress.update(failed_question=qid, failed_database=db, failure_stage="reference", **details)
                    save()
                    store.append(folder / "reevaluation_events.jsonl", {"at_utc": source.search.now(),
                                 "question_id": qid, "db_id": db, **details})
                    print(f"[REFERENCE] question={qid}, type={details['error_type']}, SQLSTATE={details['sqlstate']}, "
                          f"attempt={details['attempt']}/3, retry={details['will_retry']}", flush=True)
                try:
                    sql = sqlite_to_postgres(item["SQL"])
                    scorer.golds[qid] = reference_rows(
                        scorer.engines[db], sql, on_failure=failure,
                        on_types=lambda types: scorer.gold_type_oids.update({qid: types}))
                except Exception as error:
                    details = {**database_failure(error), "exception_type": type(error).__name__}
                    kind = details["failure_kind"]
                    progress.update(failed_question=qid, failed_database=db, failure_stage="reference", **details)
                    save()
                    if kind != "query" and not isinstance(error, ValueError):
                        raise RuntimeError(f"Reference failed at question {qid}, database={db} "
                                           f"({details['error_type']}, SQLSTATE={details['sqlstate']}); "
                                           "no model label saved") from None
                    records[qid] = {**row, "status": "reference_error", "score": None,
                                    "error": type(error).__name__, **details}
                    progress.update(saved=len(records), last_question=qid)
                    save()
                    print(f"[REEVALUATE] {benchmark} {qid}: reference_error (paid generation blocked)", flush=True)
                    continue
            score, issue = scorer.score(item, prediction, benchmark)
            records[qid] = {**row, "status": "scored", "score": score, "error": issue}
            if benchmark == "bird" and scorer.last_bird_comparison is not None:
                records[qid]["bird_comparison"] = scorer.last_bird_comparison
            progress.update(saved=len(records), last_question=qid)
            save()
            print(f"[REEVALUATE] {benchmark} {len(records)}/{len(spec['items'])}, question={qid}, score={score:g}", flush=True)
        progress["phase"] = "completed" if all(r["status"] == "scored" for r in records.values()) else "blocked_references"
    except BaseException as error:
        progress.update(phase="stopped", stop_error_type=type(error).__name__)
        raise
    finally:
        save()
        for engine in scorer.engines.values():
            engine.dispose()


def view(manifest, benchmark):
    spec = manifest["benchmarks"][benchmark]
    return {**deepcopy(spec), "sha256": manifest["sha256"],
            "env_file": manifest["env_file"], "teacher_env_file": manifest["env_file"],
            "teacher": spec["model"], "limits": LIMITS[benchmark], "global_limits": LIMITS[benchmark],
            "dev": {b: spec["items"] if b == benchmark else [] for b in FEATURES}}


def job_for(benchmark):
    return {"name": "static_control", "slot": "sql" if benchmark == "bird" else "ambiguity"}


class AuditSession(runtime.Session):
    async def prepare_tools(self, profile):
        with patch.dict(base.pilot.PROFILES, {"hybrid": self.manifest["features"]}):
            await super().prepare_tools(profile)

    @contextmanager
    def apply(self, value, item, profile, original):
        if profile != "hybrid" or original or value != self.manifest["candidate"]:
            raise ValueError("Only the frozen static candidate is authorized")
        with base.settings(self.impl, self.manifest["features"]), \
                patch.multiple(self.impl, **self.manifest["frozen_prompts"]):
            yield []

    def reflect(self, prompt):
        raise RuntimeError("Teacher and optimization are forbidden in audit regression")


def fresh_records(root, manifest, benchmark):
    path = root / benchmark / "jobs/static_control/evaluations.json"
    if not path.exists():
        return {}
    metadata = read(path.parent / "confirm/metadata.json")
    if metadata.get("manifest_sha256") != manifest["sha256"]:
        raise ValueError("Foreign fresh checkpoint")
    spec = manifest["benchmarks"][benchmark]
    values = read(path)
    keys = {key(spec["candidate"], i, benchmark): i for i in spec["items"]}
    if not set(values) <= set(keys):
        raise ValueError("Foreign fresh checkpoint question/candidate")
    for name, row in values.items():
        validate_result(row, keys[name])
    return {source.item_id(keys[k]): row for k, row in values.items()}


def require_reevaluation(root, manifest, benchmark):
    records = rescore_records(root, manifest, benchmark)
    if len(records) != len(manifest["benchmarks"][benchmark]["items"]) or any(
            r["status"] != "scored" for r in records.values()):
        raise ValueError(f"Finish {benchmark} reevaluation with all references valid before paid generation")


def run(root, manifest, benchmark):
    require_reevaluation(root, manifest, benchmark)
    records = fresh_records(root, manifest, benchmark)
    spec = manifest["benchmarks"][benchmark]
    if len(records) == len(spec["items"]):
        print(f"{benchmark}: all {len(records)} fresh answers already saved; no API calls")
        return
    folder = root / benchmark
    folder.mkdir(exist_ok=True)
    session = AuditSession(folder, view(manifest, benchmark), job_for(benchmark), "confirm")
    try:
        session.progress({"phase": "starting", "feature_mask": spec["features"]})
        session.preflight()
        for item in spec["items"]:
            session.evaluate(item, spec["candidate"], benchmark=benchmark)
        session.metadata.update(status="completed", finished_at_utc=source.search.now())
    except BaseException as error:
        session.metadata.update(status="stopped", error_type=type(error).__name__)
        raise
    finally:
        store.write_json(session.folder / "metadata.json", session.metadata)
        session.close()


def summary(items, records, benchmark):
    complete = len(records) == len(items) and all(r.get("status", "scored") == "scored" for r in records.values())
    result = {"saved": len(records), "total": len(items), "complete": complete,
              "reference_errors": [q for q, r in records.items() if r.get("status") == "reference_error"],
              "correct": sum(r["score"] for r in records.values() if r.get("score") is not None) if complete else None,
              "accuracy": sum(r["score"] for r in records.values()) / len(items) if complete else None}
    if complete and benchmark == "ambrosia":
        from adv_text2sql.evaluation import ambrosia_report
        result["ambiguity_metrics"] = ambrosia_report(
            items, {q: r["prediction"] for q, r in records.items()})["ambiguity_metrics"]
    return result


def report(root, manifest):
    result = {"manifest_sha256": manifest["sha256"], "benchmarks": {}, "deployed": False}
    lines = ["# Регрессионный контроль статического pipeline", "",
             "A — исторический результат; B — тот же SQL с новым evaluator; C — новая генерация с новым evaluator.", "",
             "| Бенчмарк | Условие | Ответы | Правильно | EX / Accuracy | F1 | Recall |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for benchmark, spec in manifest["benchmarks"].items():
        rescored = rescore_records(root, manifest, benchmark)
        fresh = fresh_records(root, manifest, benchmark)
        rows = {"A historical": summary(spec["items"], spec["historical"], benchmark),
                "B rescored": summary(spec["items"], rescored, benchmark),
                "C fresh": summary(spec["items"], fresh, benchmark)}
        paired = {}
        for name, control, treatment in (("evaluator_B_vs_A", spec["historical"], rescored),
                                         ("generation_C_vs_B", rescored, fresh)):
            ids = [q for q in treatment if q in control and treatment[q].get("score") is not None
                   and control[q].get("score") is not None]
            paired[name] = {"compared": len(ids),
                            "rescued_ids": [q for q in ids if treatment[q]["score"] > control[q]["score"]],
                            "harmed_ids": [q for q in ids if treatment[q]["score"] < control[q]["score"]]}
        ledger = root / benchmark / "budget.json"
        result["benchmarks"][benchmark] = {"features": spec["features"], "rows": rows, "paired": paired,
                                            "budget": read(ledger) if ledger.exists() else None}
        for name, row in rows.items():
            percent = lambda value: "—" if value is None else f"{100 * value:.2f}%"  # noqa: E731
            metrics = row.get("ambiguity_metrics", {})
            lines.append(f"| {benchmark} | {name} | {row['saved']}/{row['total']} | {row['correct'] if row['complete'] else '—'} | {percent(row['accuracy'])} | {percent(metrics.get('f1'))} | {percent(metrics.get('recall'))} |")
    lines += ["", "Incomplete coverage/reference errors block a full score and paid generation. No automatic deployment.",
              "One retrospective pass does not establish generalization or remove LLM run-to-run variability."]
    store.write_json(root / "comparison.json", result)
    store.write_bytes(root / "comparison.md", ("\n".join(lines) + "\n").encode("utf8"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", default=".env.deepseek")
    parser.add_argument("--benchmark", choices=(*FEATURES, "both"), default="both")
    parser.add_argument("--action", choices=("reevaluate", "run"))
    parser.add_argument("--recover-from", help="Offline transfer from an attested v1 reevaluation; never paid answers")
    parser.add_argument("--previous-code", default=PREVIOUS_CODE, help="SHA-verified archive of the previous runtime")
    for flag in ("prepare", "dry-run", "status", "report"):
        parser.add_argument("--" + flag, action="store_true")
    args = parser.parse_args()
    if sum((args.prepare, args.status, args.report, bool(args.recover_from))) > 1 or args.action and (
            args.prepare or args.status or args.report or args.recover_from):
        parser.error("Choose one action")
    root = output_path(args.output)
    path = root / "experiment.json"
    if path.exists():
        manifest = read(path)
    elif args.prepare or args.dry_run or args.recover_from:
        manifest = make_manifest(args.env_file)
    else:
        parser.error("Prepare the new audit protocol with --prepare first")
    validate(manifest, model=args.prepare or args.dry_run or bool(args.action) or bool(args.recover_from))
    selected = list(FEATURES) if args.benchmark == "both" else [args.benchmark]
    if args.action == "run":
        for benchmark in selected:
            require_reevaluation(root, manifest, benchmark)
    if args.dry_run:
        for benchmark in selected:
            print(f"OFFLINE {benchmark}: {COUNTS[benchmark]} questions; features={FEATURES[benchmark]}; teacher=0")
            print(f"Limits: {LIMITS[benchmark]}")
        print("Reevaluation: no LLM calls. Fresh run requires a complete, unblocked reevaluation; no old checkpoint reuse.")
        return
    if args.prepare or args.recover_from:
        if not path.exists():
            root.mkdir(parents=True, exist_ok=True)
            store.write_json(path, manifest)
        if args.recover_from:
            recover_reevaluation(output_path(args.recover_from), root, manifest, (ROOT / args.previous_code).resolve())
        report(root, manifest)
        print(f"Prepared offline: {root.relative_to(ROOT)}")
        return
    if args.report or args.status or not args.action:
        report(root, manifest)
        print((root / "comparison.md").read_text(encoding="utf8"))
        return
    load_dotenv(ROOT / manifest["env_file"], override=True)
    with suite_lock(root), shared.durable_runtime():
        for benchmark in selected:
            try:
                folder = root / benchmark
                folder.mkdir(exist_ok=True)
                secrets = [os.environ.get("LLM_API_KEY", ""), os.environ.get("DB_PASS", "")]
                with (folder / "audit.log").open("a", encoding="utf8") as log, patch.multiple(
                        sys, stdout=shared.Tee(sys.stdout, log, secrets), stderr=shared.Tee(sys.stderr, log, secrets)):
                    if args.action == "reevaluate":
                        reevaluate(root, manifest, benchmark)
                    else:
                        run(root, manifest, benchmark)
            finally:
                report(root, manifest)
        if args.action == "reevaluate":
            for benchmark in selected:
                require_reevaluation(root, manifest, benchmark)


if __name__ == "__main__":
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    main()
