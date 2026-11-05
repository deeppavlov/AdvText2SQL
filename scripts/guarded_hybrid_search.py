"""Explicit v3 protocol for corrected output handling and audited question/schema cohorts."""

import argparse
import hashlib
import subprocess
import sys

from dotenv import dotenv_values

from atomic_artifacts import write_json
from llm_endpoints import validate_connection
from sql_output_contract import VERSION, sql_output_issue
import static_hybrid_search as base
from run_feature_ablation import monitor, now, suite_lock

ROOT = base.ROOT
DEFAULT_OUTPUT = (
    "research_results/deepseek_combinations_large_20261026/search_v3_20261029"
)
DEFAULT_AUDIT = "research_results/deepseek_combinations_large_20261026/quality_audit_20261029/audit.json"


def make_manifest(config, audit_path, policy):
    report = base.read(audit_path)
    for name, expected in report["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Audit source changed: {name}; rebuild the audit first")
    manifest = base.make_manifest(config)
    blocked = {
        b: {
            q["question_id"]: q for q in values if q["status"] != "reference_consistent"
        }
        for b, values in report["questions"].items()
    }
    exclusions = {}
    for scope, benchmarks in manifest["cohorts"].items():
        exclusions[scope] = {}
        for b, items in benchmarks.items():
            exclusions[scope][b] = [
                str(i["question_id"])
                for i in items
                if str(i["question_id"]) in blocked[b]
            ]
            if policy == "exclude_mismatches":
                benchmarks[b] = [
                    i for i in items if str(i["question_id"]) not in blocked[b]
                ]
            if not benchmarks[b]:
                raise ValueError("Audited cohort is empty")
    # Apply the same output contract to descriptive historical controls. They are
    # not treated as equivalent fresh v3 runs, and the original bundle stays frozen.
    source_manifest = base.read(base.DATASET / "manifest.json")
    corrections = []
    indexed = {
        (r["benchmark"], r["question_id"]): r for r in manifest["references"]["rows"]
    }
    for case, spec in source_manifest["source_runs"].items():
        for b, info in spec["benchmarks"].items():
            predictions = base.read(ROOT / info["path"] / "query_results.json")
            for qid, text in predictions.items():
                if text in ("error", "ambiguous") or not sql_output_issue(text):
                    continue
                indexed[(b, qid)]["observations"][case].update(
                    correct=False,
                    observed=True,
                    generation_error=True,
                    predicted_ambiguous=False,
                )
                corrections.append(
                    {
                        "case": case,
                        "benchmark": b,
                        "question_id": qid,
                        "reason": sql_output_issue(text),
                    }
                )
    manifest["cases"].insert(
        0,
        {
            "name": "baseline_control",
            "stage": "control",
            "features": [],
            "controls": ["baseline"],
        },
    )
    extra = [
        ROOT / "scripts" / p
        for p in (
            "guarded_hybrid_search.py",
            "guarded_hybrid_runtime.py",
            "sql_output_contract.py",
            "audit_question_schemas.py",
        )
    ]
    extra += [
        ROOT / "local/run_guarded_hybrid_search.ps1",
        ROOT / "local/run_guarded_hybrid_search.sh",
        audit_path,
    ]
    manifest["protocol"].update(
        version="guarded-hybrid-v3",
        output_contract=VERSION,
        schema_policy=policy,
        runtime="scripts/guarded_hybrid_runtime.py",
        historical_controls="descriptive re-scoring under output contract, not fresh v3 runs",
    )
    manifest["protocol"]["files_sha256"].update(
        {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in extra
        }
    )
    manifest["schema_audit"] = {
        "schemas": report["schemas"],
        "blocked_questions": blocked,
        "exclusions": exclusions,
        "policy": policy,
        "output_contract_corrections_to_historical_controls": corrections,
    }
    manifest["protocol"]["schema_audit_sha256"] = base.fingerprint(
        manifest["schema_audit"]
    )
    for key, digest in (
        ("protocol", "protocol_sha256"),
        ("cohorts", "cohorts_sha256"),
        ("references", "references_sha256"),
        ("cases", "cases_sha256"),
    ):
        manifest[digest] = base.fingerprint(manifest[key])
    return manifest


def validate(manifest, *, code=True):
    base.validate_manifest(manifest, code=code)
    if manifest["protocol"].get("version") != "guarded-hybrid-v3" or manifest[
        "protocol"
    ]["schema_audit_sha256"] != base.fingerprint(manifest["schema_audit"]):
        raise ValueError("Guarded protocol/schema audit was modified")


def gate(manifest, scope, benchmarks):
    for b in benchmarks:
        bad = [
            i
            for i in manifest["cohorts"][scope][b]
            if str(i["question_id"]) in manifest["schema_audit"]["blocked_questions"][b]
        ]
        if bad:
            raise ValueError(
                f"Schema gate: {b} has {len(bad)} inconsistent questions. Restore original DBs and re-audit, "
                "or explicitly choose --schema-policy exclude_mismatches in a new output folder"
            )


def select_cases(manifest, names, stage):
    checked = base.selected_cases(manifest["cases"], stage or "core", names)
    if names:
        registry = {c["name"]: c for c in checked}
        return [registry[name] for name in names]
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", default=".env.deepseek")
    parser.add_argument("--audit", default=DEFAULT_AUDIT)
    parser.add_argument(
        "--schema-policy", choices=("require_all", "exclude_mismatches")
    )
    parser.add_argument("--scope", choices=base.SCOPES, default="train")
    parser.add_argument(
        "--benchmark", choices=(*base.BENCHMARKS, "both"), default="both"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cases", nargs="+")
    group.add_argument("--stage", choices=(*base.STAGES, "all"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--stall-seconds", type=int, default=900)
    parser.add_argument("--error-limit", type=int, default=8)
    args = parser.parse_args()
    output, audit_path = (ROOT / args.output).resolve(), (ROOT / args.audit).resolve()
    if (
        not output.is_relative_to(ROOT / "research_results")
        or output == ROOT / "research_results"
        or output == (ROOT / DEFAULT_OUTPUT).parent
        or not audit_path.is_relative_to(ROOT / "research_results")
        or args.stall_seconds < 60
        or args.error_limit < 1
    ):
        parser.error(
            "Use dedicated v3 output/audit inside research_results; stall>=60 and error-limit>=1"
        )
    path = output / "experiment.json"
    try:
        if path.exists():
            manifest = base.read(path)
            validate(manifest, code=not args.status)
            if (
                args.schema_policy
                and args.schema_policy != manifest["schema_audit"]["policy"]
            ):
                raise ValueError("Schema policy differs; choose a new output folder")
        elif args.status:
            raise ValueError("Prepare v3 first")
        else:
            manifest = make_manifest(
                dotenv_values(ROOT / args.env_file),
                audit_path,
                args.schema_policy or "require_all",
            )
        if not args.status:
            config = dotenv_values(ROOT / args.env_file)
            validate_connection(config)
            model = manifest["protocol"]["model"]
            if (
                config["LLM_MODEL_NAME"] != model["model"]
                or config.get("LLM_MODEL_VERSION") != model["model_version"]
                or config["LLM_BASE_URL"].rstrip("/") != model["base_url"]
            ):
                raise ValueError("Env model identity differs from frozen v3 protocol")
        names = args.cases or (
            None if args.stage else ["baseline_control", "hybrid_control_repeat"]
        )
        cases = select_cases(manifest, names, args.stage)
        benchmarks = base.BENCHMARKS if args.benchmark == "both" else (args.benchmark,)
        if not args.status and not args.prepare:
            gate(manifest, args.scope, benchmarks)
            base.check_controls(output, manifest, args.scope, cases, benchmarks)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    if args.status:
        rows = base.summary(output, manifest, write=False)
        print(
            [
                (r["case"], r["benchmark"], r["status"], r.get("saved"))
                for r in rows
                if r["status"] != "historical"
            ]
        )
        return
    print(
        f"V3 output contract={VERSION}; schema-policy={manifest['schema_audit']['policy']}; scope={args.scope}"
    )
    print({b: len(manifest["cohorts"][args.scope][b]) for b in benchmarks})
    print([(c["name"], c["features"]) for c in cases])
    if args.dry_run:
        print(
            "DRY RUN: no SSH/API/DB calls. Live catalog must pass before paid generation."
        )
        return
    output.mkdir(parents=True, exist_ok=True)
    with suite_lock(output):
        if path.exists() and base.read(path) != manifest:
            raise ValueError("Manifest changed while acquiring lock")
        if not path.exists():
            if any(p.name != "suite.lock" for p in output.iterdir()):
                raise ValueError("Output has artifacts without a v3 manifest")
            write_json(path, manifest)
        base.summary(output, manifest)
        if args.prepare:
            print(
                "PREPARED offline. Paid runs not started; schema gate checked on launch."
            )
            return
        write_json(
            output / "suite_status.json",
            {"status": "running", "scope": args.scope, "at_utc": now()},
        )
        try:
            for case in cases:
                for b in benchmarks:
                    if (
                        base.result(output, manifest, args.scope, case, b)["status"]
                        == "completed"
                    ):
                        continue
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            str(ROOT / "scripts/guarded_hybrid_runtime.py"),
                            "--search",
                            str(output),
                            "--env-file",
                            args.env_file,
                            "--scope",
                            args.scope,
                            "--case",
                            case["name"],
                            "--benchmark",
                            b,
                        ],
                        cwd=ROOT,
                    )
                    monitor(
                        child,
                        base.run_folder(output, args.scope, case["name"], b),
                        b,
                        args,
                    )
                    if (
                        base.result(output, manifest, args.scope, case, b)["status"]
                        != "completed"
                    ):
                        raise RuntimeError("Incomplete child run")
                    base.summary(output, manifest)
            write_json(
                output / "suite_status.json",
                {"status": "selection_completed", "scope": args.scope, "at_utc": now()},
            )
        except BaseException as error:
            write_json(
                output / "suite_status.json",
                {
                    "status": "stopped",
                    "error_type": type(error).__name__,
                    "at_utc": now(),
                },
            )
            raise
        finally:
            base.summary(output, manifest)


if __name__ == "__main__":
    main()
