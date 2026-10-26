"""Three staged feature combinations, with immutable controls and manual launch."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys

from dotenv import dotenv_values

from ablation_maintenance import approved_protocols
from feature_presets import feature_flags
from llm_endpoints import validate_connection
from research_metrics import ambrosia_report
from run_feature_ablation import (
    ROOT, fingerprint, monitor, protocol as source_protocol, read_json, save_json, suite_lock,
)

DEFAULT_SOURCE = "research_results/deepseek_direct_ablation_large_20261024"
DEFAULT_OUTPUT = "research_results/deepseek_combinations_large_20261026"
BENCHMARKS = ("bird", "ambrosia")
REFERENCES = {"baseline": [], "feat_2": [2], "feat_35": [35], "feat_14": [14], "feat_32": [32]}
# Reviewed changes: explicit multi-flag dispatch and scoring only. Generator,
# prompts, connection handling, transport and datasets must still match exactly.
REVIEWED_FILES = {
    "scripts/feature_presets.py", "scripts/run_openrouter_baseline.py",
    "scripts/research_metrics.py", "scripts/run_feature_combinations.py",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_manifest(manifest):
    if fingerprint(manifest["protocol"]) != manifest["protocol_sha256"]:
        raise ValueError("Experiment protocol hash is invalid")
    if fingerprint(manifest["references"]) != manifest["protocol"]["references_sha256"]:
        raise ValueError("Frozen reference snapshot was modified")


def directory(output, case, benchmark):
    return output / case / ("ambrosia_227" if benchmark == "ambrosia" else "bird")


def dataset(benchmark):
    return ROOT / "data" / ("bird_large.json" if benchmark == "bird" else "ambrosia_large_available_227.json")


def usage(folder):
    totals = {"prompt_tokens": 0, "completion_tokens": 0}
    path = folder / "raw_responses.jsonl"
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                value = json.loads(line)
                for key in totals:
                    totals[key] += value.get(key, 0)
    return sum(totals.values())


def result(folder, benchmark, enabled, protocol, accepted):
    metadata = read_json(folder / "metadata.json", {})
    predictions = read_json(folder / "query_results.json", {})
    row = {"status": metadata.get("status", "pending"), "saved": len(predictions),
           "accuracy": None, "f1": None, "errors": sum(v == "error" for v in predictions.values()),
           "tokens": None}
    if metadata.get("status") != "completed":
        return row
    config = metadata["configuration"]
    expected = {
        "model": protocol["model"], "base_url": protocol["base_url"],
        "api_backend": protocol["api_backend"], "model_version": protocol.get("model_version"),
        "temperature": 0, "max_output_tokens": 4096, "reasoning_effort": "none",
        "provider": protocol["provider"], "request_options": protocol["request_options"],
        "max_retries": 5, "use_evidence": benchmark == "bird",
        "dataset_sha256": hashlib.sha256(dataset(benchmark).read_bytes()).hexdigest(),
        "feature_flags": {f"FEAT_{n}": n in enabled for n in range(1, 37)},
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Completed run has incompatible settings: {folder.name}/{folder.parent.name}")
    if config.get("protocol_sha256") not in accepted:
        raise ValueError("Completed run has an unapproved protocol hash")
    items = json.loads(dataset(benchmark).read_text(encoding="utf-8"))
    ids = {str(item["question_id"]) for item in items}
    if len(ids) != len(items) or set(predictions) != ids:
        raise ValueError("Completed run has missing, duplicate or unexpected question IDs")
    report = read_json(folder / "report.json")
    if not report:
        raise ValueError("Completed run is missing report.json")
    if benchmark == "ambrosia":
        report = ambrosia_report(items, predictions)
        score = report["accuracy"]
    else:
        scores = report.get("results", [])
        if len(scores) != len(ids) or {str(r["question_id"]) for r in scores} != ids:
            raise ValueError("BIRD score coverage differs from the dataset")
        score = 100 * sum(bool(r["score"]) for r in scores) / len(ids)
        if abs(score - report["overall_accuracy"]) > 1e-8:
            raise ValueError("BIRD overall score disagrees with per-question scores")
    row.update(accuracy=score, f1=report.get("ambiguity_metrics", {}).get("f1"), tokens=usage(folder))
    return row


def reference_snapshot(source, current):
    manifest = read_json(source / "experiment.json")
    if not manifest or read_json(source / "suite_status.json", {}).get("status") != "completed":
        raise ValueError("Reference ablation suite must be completed")
    original = manifest.get("execution_revisions", [{}])[-1].get("protocol", manifest["protocol"])
    accepted = approved_protocols(manifest, original)
    before = {k: v for k, v in original.items() if k != "files_sha256"}
    after = {k: v for k, v in current.items() if k != "files_sha256"}
    if before != after:
        raise ValueError("Model, provider or experimental settings differ from the reference suite")
    first, second = original["files_sha256"], current["files_sha256"]
    changes = sorted(p for p in first.keys() | second.keys() if first.get(p) != second.get(p))
    if not set(changes) <= REVIEWED_FILES:
        raise ValueError(f"Unreviewed source/data changes prevent reference reuse: {changes}")
    rows = {}
    files = {}
    for name, enabled in REFERENCES.items():
        rows[name] = {"enabled": enabled, "benchmarks": {}}
        for benchmark in BENCHMARKS:
            folder = directory(source, name, benchmark)
            row = result(folder, benchmark, enabled, current, accepted)
            if row["status"] != "completed":
                raise ValueError(f"Reference {name}/{benchmark} is not completed")
            rows[name]["benchmarks"][benchmark] = row
            for filename in ("metadata.json", "report.json", "query_results.json", "raw_responses.jsonl"):
                path = folder / filename
                files[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"source": source.relative_to(ROOT).as_posix(), "reviewed_source_changes": changes,
            "reference_protocol_sha256": sorted(accepted), "files_sha256": files, "results": rows}


def case_results(output, manifest):
    rows = dict(manifest["references"]["results"])
    digest = fingerprint(manifest["protocol"])
    for case in manifest["cases"]:
        rows[case["name"]] = {
            "enabled": case["enabled"],
            "benchmarks": {b: result(directory(output, case["name"], b), b, case["enabled"],
                                     manifest["protocol"], {digest}) for b in BENCHMARKS},
        }
    return rows


def best_sql(rows, candidates):
    for name in candidates:
        if any(rows[name]["benchmarks"][b]["status"] != "completed" for b in BENCHMARKS):
            raise ValueError("Both benchmarks must be completed before selecting the next stage")
    return min(candidates, key=lambda n: (
        -rows[n]["benchmarks"]["bird"]["accuracy"],
        rows[n]["benchmarks"]["bird"]["tokens"] or float("inf"), len(rows[n]["enabled"]), n,
    ))


def next_case(stage, rows):
    if stage == 1:
        return {"stage": 1, "name": "schema_2_35", "enabled": [2, 35], "parents": ["feat_2", "feat_35"]}
    base = best_sql(rows, ["feat_2", "feat_35", "schema_2_35"])
    if stage == 2:
        return {"stage": 2, "name": "sql_rules", "enabled": sorted([*rows[base]["enabled"], 14]),
                "parents": [base], "selection": "BIRD accuracy; ties: fewer recorded tokens, then fewer flags"}
    core = best_sql(rows, [base, "sql_rules"])
    return {"stage": 3, "name": "sql_ambiguity", "enabled": sorted([*rows[core]["enabled"], 32]),
            "parents": [core, "feat_32"], "selection": "BIRD accuracy; ties: fewer recorded tokens, then fewer flags"}


def summary(output, manifest, write=True):
    rows = case_results(output, manifest)
    records = []
    cases = [{"name": name, "stage": "reference", "enabled": value["enabled"], "parents": []}
             for name, value in manifest["references"]["results"].items()] + manifest["cases"]
    lines = ["# DeepSeek: staged combinations", "", "Ambrosia: error is always incorrect (strict-v1).", "",
             "| Case | Flags | Benchmark | Status | Saved | Accuracy | Delta baseline, pp | Delta parent, pp | F1 | Tokens | Errors |",
             "| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |"]
    for case in cases:
        for benchmark in BENCHMARKS:
            row = rows[case["name"]]["benchmarks"][benchmark]
            score = row["accuracy"]
            baseline = rows["baseline"]["benchmarks"][benchmark]["accuracy"]
            deltas = {parent: score - rows[parent]["benchmarks"][benchmark]["accuracy"]
                      for parent in case["parents"]} if score is not None else {}
            record = {"case": case["name"], "stage": case["stage"], "features": case["enabled"],
                      "benchmark": benchmark, **row,
                      "delta_baseline_pp": score - baseline if score is not None else None,
                      "delta_parent_pp": deltas}
            records.append(record)
            def fmt(value):
                return f"{value:.2f}" if value is not None else "—"
            delta = fmt(record["delta_baseline_pp"])
            parents = "; ".join(f"{name}: {value:+.2f}" for name, value in deltas.items()) or "—"
            f1 = fmt(100 * row["f1"] if row["f1"] is not None else None)
            flags = ",".join(map(str, case["enabled"])) or "all false"
            lines.append(f"| {case['name']} | {flags} | {benchmark} | {row['status']} | {row['saved']} | "
                         f"{fmt(score)} | {delta} | {parents} | {f1} | {row['tokens'] or '—'} | {row['errors']} |")
    lines += ["", "Reference rows reuse saved answers; no paid baseline reruns.",
              "Stages 2/3 use exploratory selection on BIRD, not an independent test.",
              "References and source hashes are frozen in experiment.json. Original reports are unchanged."]
    if write:
        save_json(output / "summary.json", records)
        (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env.deepseek")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--through-stage", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--provider")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare", action="store_true", help="Freeze plan and controls offline; do not run benchmarks")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--stall-seconds", type=int, default=900)
    parser.add_argument("--error-limit", type=int, default=8)
    args = parser.parse_args()
    source, output = (ROOT / args.source).resolve(), (ROOT / args.output).resolve()
    if any(not p.is_relative_to(ROOT / "research_results") for p in (source, output)):
        parser.error("Source and output must be inside research_results")
    if source == output or source in output.parents or output in source.parents:
        parser.error("Source and output must be separate directories")
    if args.stall_seconds < 60 or args.error_limit < 1:
        parser.error("stall-seconds must be >=60 and error-limit >=1")
    if args.status:
        manifest = read_json(output / "experiment.json")
        if not manifest:
            parser.error("Prepare or launch the experiment first")
        validate_manifest(manifest)
        for row in summary(output, manifest, write=False):
            print(f"{row['case']}/{row['benchmark']}: {row['status']}, {row['saved']}, {row['accuracy']}")
        return
    config = dotenv_values(ROOT / args.env_file)
    try:
        validate_connection(config, args.provider)
        current = source_protocol(config, args.provider)
        references = reference_snapshot(source, current)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    current.update(version="feature-combinations-v1", ablation_design="three_staged_combinations",
                   evaluation_policy="ambrosia-strict-v1", references_sha256=fingerprint(references),
                   selection_rule="BIRD accuracy descending; tokens ascending; flag count ascending; name ascending")
    digest = fingerprint(current)
    manifest = read_json(output / "experiment.json")
    if manifest:
        validate_manifest(manifest)
    if manifest and (manifest["protocol"] != current or manifest["protocol_sha256"] != digest):
        parser.error("Protocol, controls or source changed; use a new output folder")
    print(f"Model: {current['model']}; output: {output}", flush=True)
    print("1: FEAT_2+35, both benchmarks; compare with baseline/2/35.", flush=True)
    print("2: best SQL base + FEAT_14; 3: best SQL configuration + FEAT_32.", flush=True)
    print("Selection: BIRD accuracy; ties use fewer tokens. All Ambrosia errors count as incorrect.", flush=True)
    if args.dry_run:
        print("DRY RUN: controls validated; no writes, SSH, database or API calls.")
        return
    output.mkdir(parents=True, exist_ok=True)
    with suite_lock(output):
        manifest = read_json(output / "experiment.json")
        if manifest is None:
            if any(p.name != "suite.lock" for p in output.iterdir()):
                raise ValueError("Output has artifacts without an experiment.json")
            manifest = {"created_at_utc": now(), "protocol": current, "protocol_sha256": digest,
                        "references": references, "cases": [next_case(1, {})]}
            save_json(output / "experiment.json", manifest)
        if manifest["protocol"] != current or manifest["protocol_sha256"] != digest:
            raise ValueError("Protocol changed while waiting for the suite lock")
        validate_manifest(manifest)
        if args.prepare:
            summary(output, manifest)
            print("PREPARED: immutable controls saved; no SSH, database or API calls.")
            return
        save_json(output / "suite_status.json", {"status": "running", "at_utc": now()})
        try:
            for stage in range(1, args.through_stage + 1):
                case = next_case(stage, case_results(output, manifest))
                feature_flags("combination", case["enabled"])
                previous = next((c for c in manifest["cases"] if c["stage"] == stage), None)
                if previous and previous != case:
                    raise ValueError("Frozen stage selection no longer matches its controls")
                if not previous:
                    manifest["cases"].append(case)
                    save_json(output / "experiment.json", manifest)
                print(f"[STAGE {stage}] {case['name']}: {case['enabled']}, parents={case['parents']}", flush=True)
                for benchmark in BENCHMARKS:
                    folder = directory(output, case["name"], benchmark)
                    row = result(folder, benchmark, case["enabled"], current, {digest})
                    if row["status"] == "completed":
                        print(f"[SKIP] {case['name']}/{benchmark}: completed", flush=True)
                        continue
                    command = [sys.executable, "-u", str(ROOT / "scripts/run_openrouter_baseline.py"),
                               "--configuration", "combination", "--enabled-features", *map(str, case["enabled"]),
                               "--benchmark", benchmark, "--model", config["LLM_MODEL_NAME"],
                               "--env-file", args.env_file, "--output", str(output / case["name"]),
                               "--protocol-sha256", digest, "--evaluation-policy", "strict"]
                    if args.provider:
                        command += ["--provider", args.provider]
                    child = subprocess.Popen(command, cwd=ROOT)
                    monitor(child, folder, benchmark, args)
                    if result(folder, benchmark, case["enabled"], current, {digest})["status"] != "completed":
                        raise RuntimeError("Child exited without a completed benchmark")
                    summary(output, manifest)
            save_json(output / "suite_status.json", {
                "status": "completed" if args.through_stage == 3 else "stage_completed",
                "through_stage": args.through_stage, "at_utc": now(),
            })
        except BaseException as error:
            save_json(output / "suite_status.json", {"status": "stopped", "at_utc": now(),
                                                      "error_type": type(error).__name__})
            raise
        finally:
            summary(output, manifest)


if __name__ == "__main__":
    main()
