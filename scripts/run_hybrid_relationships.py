"""Add PK/FK hints to the frozen 14+32+35 hybrid on both benchmarks."""

import argparse
import hashlib
import subprocess
import sys

from dotenv import dotenv_values

from llm_endpoints import validate_connection
from run_feature_ablation import ROOT, fingerprint, monitor, protocol, read_json, save_json, suite_lock
from run_feature_combinations import (
    BENCHMARKS, case_results, directory, now, result, summary, validate_manifest,
)

SOURCE = "research_results/deepseek_combinations_large_20261026"
OUTPUT = "research_results/deepseek_relationships_large_20261026"
CASE = {"stage": "relationships", "name": "hybrid_with_relationships", "enabled": [1, 14, 32, 35],
        "parents": ["sql_ambiguity"]}


def check_source_protocol(original, current):
    adapted = {"version", "ablation_design", "evaluation_policy", "references_sha256", "selection_rule", "files_sha256"}
    if original.get("evaluation_policy") != "ambrosia-strict-v1":
        raise ValueError("Source experiment must use strict Ambrosia scoring")
    if {k: v for k, v in original.items() if k not in adapted} != {
        k: v for k, v in current.items() if k not in adapted
    }:
        raise ValueError("Model, provider or generation settings differ from the source hybrid")
    before, after = original["files_sha256"], current["files_sha256"]
    changes = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
    if not set(changes) <= {"scripts/run_hybrid_relationships.py"}:
        raise ValueError(f"Source code or dataset changed beyond the new launcher: {changes}")
    return changes


def snapshot(source, current):
    manifest = read_json(source / "experiment.json")
    if not manifest or read_json(source / "suite_status.json", {}).get("status") != "completed":
        raise ValueError("The three-stage source experiment must be completed")
    validate_manifest(manifest)
    changes = check_source_protocol(manifest["protocol"], current)
    parent = next((c for c in manifest["cases"] if c["name"] == "sql_ambiguity"), None)
    if not parent or parent["enabled"] != [14, 32, 35]:
        raise ValueError("The source hybrid must have exactly FEAT_14+32+35")
    files = dict(manifest["references"]["files_sha256"])
    for name, expected in files.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen control file changed: {name}")
    rows = case_results(source, manifest)
    controls = {name: rows[name] for name in ("baseline", "feat_35", "sql_ambiguity")}
    if any(row["status"] != "completed" for c in controls.values() for row in c["benchmarks"].values()):
        raise ValueError("Both benchmarks of all controls must be completed")
    paths = [source / "experiment.json", source / "suite_status.json"]
    for benchmark in BENCHMARKS:
        folder = directory(source, "sql_ambiguity", benchmark)
        paths += [folder / name for name in ("metadata.json", "report.json", "query_results.json", "raw_responses.jsonl")]
    for path in paths:
        files[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"source": source.relative_to(ROOT).as_posix(), "reviewed_source_changes": changes,
            "files_sha256": files, "results": controls}


def write_summary(output, manifest, write=True):
    rows = summary(output, manifest, write=False)
    lines = ["# DeepSeek: PK/FK hints added to the static hybrid", "",
             "Ambrosia: strict-v1; generation error is always incorrect.", "",
             "| Case | Flags | Benchmark | Status | Saved | Accuracy | Delta baseline, pp | Delta 14+32+35, pp | F1 | Tokens | Errors |",
             "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        def fmt(value):
            return f"{value:.2f}" if value is not None else "—"

        f1 = fmt(100 * row["f1"] if row["f1"] is not None else None)
        parent_delta = row["delta_parent_pp"].get("sql_ambiguity")
        flags = ",".join(map(str, row["features"])) or "all false"
        lines.append(f"| {row['case']} | {flags} | {row['benchmark']} | {row['status']} | {row['saved']} | "
                     f"{fmt(row['accuracy'])} | {fmt(row['delta_baseline_pp'])} | {fmt(parent_delta)} | "
                     f"{f1} | {row['tokens'] or '—'} | {row['errors']} |")
    lines += ["", "Only FEAT_1 is added; all other feature settings are fixed.",
              "Saved controls are reused without LLM calls. Original artifacts are unchanged.",
              "This is exploratory model selection on previously used questions, not an independent final test."]
    if write:
        save_json(output / "summary.json", rows)
        (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env.deepseek")
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--provider")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare", action="store_true")
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
            parser.error("Prepare or launch this experiment first")
        validate_manifest(manifest)
        for row in write_summary(output, manifest, write=False):
            print(f"{row['case']}/{row['benchmark']}: {row['status']}, {row['saved']}, {row['accuracy']}")
        return
    config = dotenv_values(ROOT / args.env_file)
    try:
        validate_connection(config, args.provider)
        current = protocol(config, args.provider)
        references = snapshot(source, current)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    current.update(version="hybrid-relationships-v1", ablation_design="add_one_to_fixed_hybrid",
                   evaluation_policy="ambrosia-strict-v1", references_sha256=fingerprint(references),
                   added_feature=1, control_features=[14, 32, 35], candidate_features=CASE["enabled"])
    digest = fingerprint(current)
    previous = read_json(output / "experiment.json")
    if previous:
        validate_manifest(previous)
        if previous["protocol"] != current or previous["cases"] != [CASE]:
            parser.error("Protocol, controls or source changed; use a new output folder")
    print(f"Model: {current['model']}; FEAT_1+14+32+35; BIRD 241 / Ambrosia 227.", flush=True)
    print(f"Output: {output}; controls: baseline, FEAT_35, FEAT_14+32+35.", flush=True)
    if args.dry_run:
        print("DRY RUN: controls validated; no writes, SSH, database or API calls.")
        return
    output.mkdir(parents=True, exist_ok=True)
    with suite_lock(output):
        manifest = read_json(output / "experiment.json")
        if manifest is None:
            if any(p.name != "suite.lock" for p in output.iterdir()):
                raise ValueError("Output contains artifacts without an experiment.json")
            manifest = {"created_at_utc": now(), "protocol": current, "protocol_sha256": digest,
                        "references": references, "cases": [CASE]}
            save_json(output / "experiment.json", manifest)
        validate_manifest(manifest)
        if manifest["protocol"] != current or manifest["cases"] != [CASE]:
            raise ValueError("Frozen experiment changed while waiting for the lock")
        if args.prepare:
            write_summary(output, manifest)
            print("PREPARED: controls frozen; no SSH, database or API calls.")
            return
        save_json(output / "suite_status.json", {"status": "running", "at_utc": now()})
        try:
            for benchmark in BENCHMARKS:
                folder = directory(output, CASE["name"], benchmark)
                if result(folder, benchmark, CASE["enabled"], current, {digest})["status"] == "completed":
                    print(f"[SKIP] {benchmark}: completed", flush=True)
                    continue
                command = [sys.executable, "-u", str(ROOT / "scripts/run_openrouter_baseline.py"),
                           "--configuration", "combination", "--enabled-features", *map(str, CASE["enabled"]),
                           "--benchmark", benchmark, "--model", config["LLM_MODEL_NAME"],
                           "--env-file", args.env_file, "--output", str(output / CASE["name"]),
                           "--protocol-sha256", digest, "--evaluation-policy", "strict"]
                if args.provider:
                    command += ["--provider", args.provider]
                print(f"[START] {benchmark}: {CASE['enabled']}", flush=True)
                child = subprocess.Popen(command, cwd=ROOT)
                monitor(child, folder, benchmark, args)
                if result(folder, benchmark, CASE["enabled"], current, {digest})["status"] != "completed":
                    raise RuntimeError("Child exited without a completed benchmark")
                write_summary(output, manifest)
            save_json(output / "suite_status.json", {"status": "completed", "at_utc": now()})
        except BaseException as error:
            save_json(output / "suite_status.json", {"status": "stopped", "at_utc": now(),
                                                      "error_type": type(error).__name__})
            raise
        finally:
            write_summary(output, manifest)


if __name__ == "__main__":
    main()
