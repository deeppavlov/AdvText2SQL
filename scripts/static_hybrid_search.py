"""Staged manual search; immutable historical references and grouped train/dev/full cohorts."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from dotenv import dotenv_values

from atomic_artifacts import write_json
from audit_static_hybrid import DATASET, FEATURES, KNOWN_COMBINATIONS, proposed_probes
import build_multilabel_dataset as source
from feature_presets import feature_flags
from llm_endpoints import validate_connection
from run_feature_ablation import fingerprint, monitor, now, suite_lock

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = "research_results/deepseek_combinations_large_20261026/search_v2_20261029"
BENCHMARKS = ("bird", "ambrosia")
SCOPES = ("train", "dev", "full")
STAGES = {"core": "core_factorial", "additions": "one_mechanism", "statistics": "statistics_bundle",
          "repair": "repair_bundle", "interactions": "workflow_interaction"}
DATA_FILES = {"bird": "data/bird_large.json", "ambrosia": "data/ambrosia_large_available_227.json"}


def read(path):
    # A corrupt existing file must never silently become an empty checkpoint or plan.
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def case_registry():
    cases = proposed_probes()
    cases.append({"name": "hybrid_control_repeat", "stage": "repeat",
                  "features": [14, 32, 35], "controls": ["sql_ambiguity"], "status": "proposed_not_run"})
    return cases


def selected_cases(cases, stage="core", names=None):
    if names:
        wanted = set(names)
        if len(wanted) != len(names) or not wanted <= {c["name"] for c in cases}:
            raise ValueError("Unknown or duplicate case name")
        return [c for c in cases if c["name"] in wanted]
    return [c for c in cases if c["stage"] != "repeat" and
            (stage == "all" or c["stage"] == STAGES[stage])]


def validate_features(values):
    feature_flags("combination", values)
    f = set(values)
    if (any(n in f and 2 not in f for n in (3, 4, 7)) or (27 in f and 17 not in f)):
        raise ValueError("Missing prerequisite in explicit feature mask")
    if 28 in f and f.intersection((17, 20, 27, 30)):
        raise ValueError("Voting bypasses retry/judge/feedback and loses decomposition")
    if not f <= set(FEATURES) - {5, 6, 8, 9, 10, 11, 18, 19, 26, 33, 34, 36}:
        raise ValueError("Mask contains infrastructure, unfinished, stateful or other-task flags")
    # 14+29 is intentionally allowed: routing retains the strict moderate branch.


def code_paths():
    scripts = ("static_hybrid_search.py", "static_hybrid_runtime.py", "audit_static_hybrid.py",
               "run_feature_ablation.py", "run_openrouter_baseline.py", "baseline_connection.py",
               "atomic_artifacts.py", "feature_presets.py", "research_metrics.py", "llm_endpoints.py",
               "build_multilabel_dataset.py", "build_configuration_dataset.py", "multilabel_targets.py",
               "configuration_splits.py", "research_paths.py", "ablation_maintenance.py")
    return [*(ROOT / "scripts" / p for p in scripts),
            *sorted((ROOT / "benchmarks").glob("*.py")),
            *sorted((ROOT / "src/adv_text2sql/mcp_servers/text2sql_tool/src").glob("*.py"))]


def current_protocol(model):
    paths = [*code_paths(), ROOT / "local/run_static_hybrid_search.sh",
             ROOT / "local/run_static_hybrid_search.ps1", ROOT / "local/start_tunnel.sh",
             DATASET / "manifest.json", DATASET / "records.jsonl", *(ROOT / p for p in DATA_FILES.values())]
    return {"version": "static-hybrid-search-v2", "model": model,
            "evaluation_policy": "ambrosia-strict-v1", "scope_policy": "grouped train search; dev confirmation; full exploratory",
            "selection": "equal benchmark mean accuracy; retain separate BIRD/Ambrosia leaders",
            "files_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def validate_manifest(manifest, *, code=True):
    if fingerprint(manifest["protocol"]) != manifest["protocol_sha256"]:
        raise ValueError("Search protocol hash mismatch")
    if fingerprint(manifest["references"]) != manifest["references_sha256"]:
        raise ValueError("Historical references changed")
    if fingerprint(manifest["cohorts"]) != manifest["cohorts_sha256"]:
        raise ValueError("Question cohorts changed")
    if fingerprint(manifest["cases"]) != manifest["cases_sha256"]:
        raise ValueError("Frozen case plan changed")
    if code:
        for name, expected in manifest["protocol"]["files_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Source/data changed: {name}; use a new search output folder")


def make_manifest(config):
    source_manifest = source.verify(DATASET)
    model = source_manifest["model"]
    backend = validate_connection(config)
    if (backend != model["api_backend"] or config["LLM_MODEL_NAME"] != model["model"] or
            config["LLM_BASE_URL"].rstrip("/") != model["base_url"] or
            config.get("LLM_MODEL_VERSION") != model["model_version"]):
        raise ValueError("Search must preserve the recorded DeepSeek model, endpoint and declared version")
    # New wrappers are reviewed here; the SQL generator/prompts/evaluators must match the historical series.
    historic = read(ROOT / "research_results/deepseek_combinations_large_20261026/experiment.json")["protocol"]
    for name, digest in historic["files_sha256"].items():
        if name.startswith(("src/", "benchmarks/")) and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Generator/evaluator differs from historical controls: {name}")
    rows = source.lines(DATASET / "records.jsonl")
    datasets = {b: read(ROOT / p) for b, p in DATA_FILES.items()}
    cohorts = {}
    for scope in SCOPES:
        cohorts[scope] = {}
        for b, items in datasets.items():
            ids = {r["question_id"] for r in rows if r["benchmark"] == b and (scope == "full" or r["split"] == scope)}
            chosen = [i for i in items if str(i["question_id"]) in ids]
            if {str(i["question_id"]) for i in chosen} != ids or not ids:
                raise ValueError("Frozen grouped split differs from source questions")
            cohorts[scope][b] = chosen
    references = {"features": {"baseline": [], **{f"feat_{n}": [n] for n in FEATURES},
                               **{k: list(v) for k, v in KNOWN_COMBINATIONS.items()}},
                  "rows": [{**{k: r[k] for k in ("id", "benchmark", "question_id", "split")},
                            "observations": {name: {k: value[k] for k in
                                                    ("correct", "observed", "generation_error", "predicted_ambiguous")}
                                             for name, value in r["observations"].items()}}
                           for r in rows]}
    cases = case_registry()
    for case in cases:
        validate_features(case["features"])
    protocol = current_protocol(model)
    return {"created_at_utc": now(), "protocol": protocol, "protocol_sha256": fingerprint(protocol),
            "references": references, "references_sha256": fingerprint(references),
            "cohorts": cohorts, "cohorts_sha256": fingerprint(cohorts),
            "cases": cases, "cases_sha256": fingerprint(cases)}


def run_folder(output, scope, case, benchmark):
    return output / scope / case / ("bird" if benchmark == "bird" else "ambrosia_227")


def identity(manifest, scope, case, benchmark):
    return {"protocol_sha256": manifest["protocol_sha256"], "model": manifest["protocol"]["model"],
            "scope": scope, "case": case["name"], "benchmark": benchmark,
            "dataset_sha256": fingerprint(manifest["cohorts"][scope][benchmark]),
            "feature_flags": {f"FEAT_{n}": n in case["features"] for n in range(1, 37)},
            "evaluation_policy": "ambrosia-strict-v1"}


def scored_predictions(items, predictions, report, benchmark):
    ids = {str(i["question_id"]) for i in items}
    if set(predictions) != ids or any(not isinstance(v, str) or not v.strip() for v in predictions.values()):
        raise ValueError("Incomplete/invalid predictions in a completed run")
    observations = source.recorded_observations(items, predictions, report, benchmark)
    if any(not v["observed"] for v in observations.values()):
        raise ValueError("Unclassified/reference evaluation failure; inspect and re-evaluate before comparison")
    correct = {q: v["correct"] for q, v in observations.items()}
    score = 100 * sum(correct.values()) / len(correct)
    if benchmark == "ambrosia":
        from research_metrics import ambrosia_report
        strict = ambrosia_report(items, predictions)
        if report.get("evaluation_policy") != "ambrosia-strict-v1" or abs(report["accuracy"] - score) > 1e-8:
            raise ValueError("Ambrosia report differs from strict outcomes")
        metrics = strict["ambiguity_metrics"]
    else:
        results = report.get("results", [])
        if (len(results) != len(ids) or {str(r["question_id"]) for r in results} != ids or
                abs(report["overall_accuracy"] - score) > 1e-8):
            raise ValueError("BIRD report coverage/accuracy differs from predictions")
        metrics = {}
    return {"accuracy": score, "correct": sum(correct.values()), "outcomes": correct,
            "errors": sum(p == "error" for p in predictions.values()), "ambiguity_metrics": metrics,
            "execution_errors": sum(bool(v["execution_error"]) for v in observations.values())}


def result(output, manifest, scope, case, benchmark):
    folder = run_folder(output, scope, case["name"], benchmark)
    items = manifest["cohorts"][scope][benchmark]
    empty = {"status": "pending", "saved": 0, "questions": len(items), "accuracy": None}
    if not (folder / "metadata.json").exists():
        if (folder / "query_results.json").exists():
            raise ValueError("Orphan checkpoint without metadata")
        return empty
    meta = read(folder / "metadata.json")
    if meta["configuration"] != identity(manifest, scope, case, benchmark):
        raise ValueError("Existing run identity differs from frozen search plan")
    predictions = read(folder / "query_results.json") if (folder / "query_results.json").exists() else {}
    if not set(predictions) <= {str(i["question_id"]) for i in items}:
        raise ValueError("Checkpoint has questions outside its frozen cohort")
    empty.update(status=meta["status"], saved=len(predictions))
    if meta["status"] != "completed":
        return empty
    scored = scored_predictions(items, predictions, read(folder / "report.json"), benchmark)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "recorded_calls": 0}
    path = folder / "raw_responses.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            for key in ("prompt_tokens", "completion_tokens"):
                usage[key] += row[key]
            usage["recorded_calls"] += 1
    history = folder / "attempt_history.jsonl"
    attempts = ([json.loads(s) for s in history.read_text(encoding="utf-8").splitlines() if s.strip()]
                if history.exists() else []) + [meta]
    balances = {}
    for attempt in attempts:
        for currency, value in attempt.get("balance_delta_this_attempt", {}).items():
            balances[currency] = balances.get(currency, 0) + value
    return {**empty, **scored, **usage, "balance_deltas": balances}


def check_controls(output, manifest, scope, cases, benchmarks):
    """On train, complete the matched parent controls before paying for dependent probes."""
    if scope != "train":
        return  # A dev/full finalist may be confirmed directly against baseline and H.
    available = set(manifest["references"]["features"])
    registry = {c["name"]: c for c in manifest["cases"]}
    for case in cases:
        for parent in case["controls"]:
            if parent in available:
                continue
            if any(result(output, manifest, scope, registry[parent], b)["status"] != "completed" for b in benchmarks):
                raise ValueError(f"Run train parent '{parent}' before '{case['name']}', or include it earlier in --cases/--stage all")
        available.add(case["name"])


def reference_results(manifest, scope, benchmark):
    wanted = {str(i["question_id"]) for i in manifest["cohorts"][scope][benchmark]}
    rows = [r for r in manifest["references"]["rows"] if r["benchmark"] == benchmark and r["question_id"] in wanted]
    results = {}
    for case, flags in manifest["references"]["features"].items():
        observations = {r["question_id"]: r["observations"][case] for r in rows}
        # Unobserved historical blank output is explicitly failed in fixed-size descriptive accuracy.
        outcomes = {q: v["correct"] is True for q, v in observations.items()}
        item = {"status": "historical", "features": flags, "questions": len(wanted),
                "correct": sum(outcomes.values()), "accuracy": 100 * sum(outcomes.values()) / len(wanted),
                "outcomes": outcomes, "missing_observations": sum(not v["observed"] for v in observations.values())}
        if benchmark == "ambrosia":
            from research_metrics import ambrosia_report
            predictions = {q: "error" if not v["observed"] or v["generation_error"] else
                           "ambiguous" if v["predicted_ambiguous"] else "SELECT placeholder"
                           for q, v in observations.items()}
            item["ambiguity_metrics"] = ambrosia_report(manifest["cohorts"][scope][benchmark], predictions)["ambiguity_metrics"]
        results[case] = item
    return results


def summary(output, manifest, *, write=True):
    rows = []
    lines = ["# Static hybrid search v2", "", "Search: train; confirmation: dev; full is exploratory, not an independent test.",
             "Historical controls are replayed on identical question IDs, without new API calls.", "",
             "| Scope | Case | Features | Benchmark | Status | Saved / total | Accuracy | Delta baseline pp | Delta H pp | F1 | Recall | FP | Errors | Tokens | Calls | Balance delta |",
             "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for scope in SCOPES:
        for benchmark in BENCHMARKS:
            refs = reference_results(manifest, scope, benchmark)
            combined = {**refs, **{c["name"]: {**result(output, manifest, scope, c, benchmark), "features": c["features"]}
                                   for c in manifest["cases"]}}
            # Export only completed or started new runs, plus their historical controls.
            names = ["baseline", "feat_14", "feat_32", "feat_35", "sql_rules", "sql_ambiguity"]
            names += [c["name"] for c in manifest["cases"] if combined[c["name"]]["status"] != "pending"]
            for name in names:
                value = combined[name]
                score = value.get("accuracy")
                deltas, pairs = {}, {}
                controls = ["baseline", "sql_ambiguity"] + next((c["controls"] for c in manifest["cases"] if c["name"] == name), [])
                if score is not None:
                    for parent in dict.fromkeys(controls):
                        p = combined[parent]
                        if p.get("accuracy") is None:
                            continue
                        deltas[parent] = score - p["accuracy"]
                        pairs[parent] = {"rescued": sum(value["outcomes"][q] and not p["outcomes"][q] for q in value["outcomes"]),
                                         "harmed": sum(not value["outcomes"][q] and p["outcomes"][q] for q in value["outcomes"])}
                record = {"scope": scope, "case": name, "benchmark": benchmark,
                          **{k: v for k, v in value.items() if k != "outcomes"}, "delta_pp": deltas, "paired": pairs}
                rows.append(record)
                def fmt(v):
                    return "—" if v is None else f"{v:.2f}"
                metrics = value.get("ambiguity_metrics", {})
                tokens = sum(value.get(k, 0) for k in ("prompt_tokens", "completion_tokens"))
                expense = ', '.join(f'{amount:.4f} {currency}' for currency, amount in value.get('balance_deltas', {}).items()) or '—'
                lines.append(f"| {scope} | {name} | {','.join(map(str,value['features'])) or 'all false'} | {benchmark} | {value['status']} | "
                             f"{value.get('saved', value['questions'])}/{value['questions']} | {fmt(score)} | {fmt(deltas.get('baseline'))} | "
                             f"{fmt(deltas.get('sql_ambiguity'))} | {fmt(100*metrics['f1'] if metrics.get('f1') is not None else None)} | "
                             f"{fmt(100*metrics['recall'] if metrics.get('recall') is not None else None)} | {metrics.get('fp','—')} | "
                             f"{value.get('errors','—')} | {tokens or '—'} | {value.get('recorded_calls','—')} | {expense} |")
    rankings = {}
    for scope in SCOPES:
        candidates = []
        for name in {r["case"] for r in rows if r["scope"] == scope}:
            matches = [r for r in rows if r["scope"] == scope and r["case"] == name and r["accuracy"] is not None]
            if len(matches) == 2:
                candidates.append({"case": name, "features": matches[0]["features"],
                                   "macro_accuracy": sum(r["accuracy"] for r in matches)/2,
                                   **{r["benchmark"]: r["accuracy"] for r in matches}})
        rankings[scope] = sorted(candidates, key=lambda r: (-r["macro_accuracy"], len(r["features"]), r["case"]))
    lines += ["", "## Joint score (both benchmarks required)", "",
              "| Scope | Case | Flags | BIRD | Ambrosia | Mean accuracy |", "| --- | --- | --- | ---: | ---: | ---: |"]
    for scope, candidates in rankings.items():
        for c in candidates:
            lines.append(f"| {scope} | {c['case']} | {','.join(map(str,c['features'])) or 'all false'} | "
                         f"{c['bird']:.2f} | {c['ambrosia']:.2f} | {c['macro_accuracy']:.2f} |")
    leaders, best_by_size = {}, {}
    for scope, candidates in rankings.items():
        leaders[scope] = {metric: min(candidates, key=lambda c: (-c[metric], len(c["features"]), c["case"]))
                          for metric in ("bird", "ambrosia", "macro_accuracy")}
        best_by_size[scope] = {str(size): next(c for c in candidates if len(c["features"]) == size)
                               for size in sorted({len(c["features"]) for c in candidates})}
    lines += ["", "## Best measured mean score by number of enabled algorithm flags", "",
              "| Scope | X flags | Case | Mean accuracy |", "| --- | ---: | --- | ---: |"]
    for scope, values in best_by_size.items():
        for size, c in values.items():
            lines.append(f"| {scope} | {size} | {c['case']} | {c['macro_accuracy']:.2f} |")
    lines += ["", "Tokens include all recorded probes, build calls and attempts, not failed/unrecorded HTTP retries.",
              "Per-parent deltas and rescued/harmed counts are in summary.json. Historical per-question costs are unknown.",
              "A stage result is not a certificate of global optimality. No automatic next-stage paid runs.", ""]
    if write:
        write_json(output / "summary.json", {"rows": rows, "rankings": rankings,
                                              "leaders": leaders, "best_by_flag_count": best_by_size})
        (output / "summary.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env.deepseek")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--scope", choices=SCOPES, default="train")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stage", choices=(*STAGES, "all"), default="core")
    group.add_argument("--cases", nargs="+")
    parser.add_argument("--benchmark", choices=(*BENCHMARKS, "both"), default="both")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--stall-seconds", type=int, default=900)
    parser.add_argument("--error-limit", type=int, default=8)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if (not output.is_relative_to(ROOT / "research_results") or output == ROOT / "research_results" or
            output == (ROOT / DEFAULT_OUTPUT).parent or args.stall_seconds < 60 or args.error_limit < 1):
        parser.error("Use a dedicated search folder inside research_results; stall>=60 and error-limit>=1")
    path = output / "experiment.json"
    try:
        if path.exists():
            manifest = read(path)
            validate_manifest(manifest, code=not args.status)
            if not args.status:
                config = dotenv_values(ROOT / args.env_file)
                validate_connection(config)
                model = manifest["protocol"]["model"]
                if any(config.get(k) != model[v] for k, v in
                       (("LLM_MODEL_NAME", "model"), ("LLM_MODEL_VERSION", "model_version"))) or config["LLM_BASE_URL"].rstrip("/") != model["base_url"]:
                    raise ValueError("Env model/version/endpoint differs from frozen protocol")
        elif args.status:
            raise ValueError("Prepare the search first")
        else:
            manifest = make_manifest(dotenv_values(ROOT / args.env_file))
        cases = selected_cases(manifest["cases"], args.stage, args.cases)
        benchmarks = BENCHMARKS if args.benchmark == "both" else (args.benchmark,)
        if not args.prepare and not args.status:
            check_controls(output, manifest, args.scope, cases, benchmarks)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    if args.status:
        started = [r for r in summary(output, manifest, write=False) if r["status"] != "historical"]
        for r in started:
            if r["status"] != "historical":
                print(f"{r['scope']}/{r['case']}/{r['benchmark']}: {r['status']} {r['saved']}/{r['questions']} accuracy={r['accuracy']}")
        if not started:
            print("Prepared; no benchmark runs started. Default stage: two core masks on train.")
        return
    print(f"Model: {manifest['protocol']['model']['model']}; scope={args.scope}; output={output}", flush=True)
    for b in benchmarks:
        print(f"{b}: {len(manifest['cohorts'][args.scope][b])} questions", flush=True)
    for case in cases:
        print(f"{case['name']}: {case['features']} controls={case['controls']}", flush=True)
    if args.dry_run:
        print("DRY RUN: no writes, SSH, API or database calls.")
        return
    output.mkdir(parents=True, exist_ok=True)
    with suite_lock(output):
        if path.exists():
            if read(path) != manifest:
                raise ValueError("Search manifest changed while acquiring lock")
        else:
            if any(p.name != "suite.lock" for p in output.iterdir()):
                raise ValueError("Output is not empty and has no search manifest")
            write_json(path, manifest)
        summary(output, manifest)
        if args.prepare:
            print("PREPARED: frozen controls, cohorts and masks. No paid runs.")
            return
        write_json(output / "suite_status.json", {"status": "running", "scope": args.scope,
                                                  "cases": [c["name"] for c in cases], "at_utc": now()})
        try:
            for case in cases:
                for b in benchmarks:
                    folder = run_folder(output, args.scope, case["name"], b)
                    if result(output, manifest, args.scope, case, b)["status"] == "completed":
                        print(f"[SKIP] {args.scope}/{case['name']}/{b}: completed", flush=True)
                        continue
                    command = [sys.executable, "-u", str(ROOT / "scripts/static_hybrid_runtime.py"),
                               "--search", str(output), "--env-file", args.env_file,
                               "--scope", args.scope, "--case", case["name"], "--benchmark", b]
                    child = subprocess.Popen(command, cwd=ROOT)
                    monitor(child, folder, b, args)
                    if result(output, manifest, args.scope, case, b)["status"] != "completed":
                        raise RuntimeError("Child exited without complete validated results")
                    summary(output, manifest)
            write_json(output / "suite_status.json", {"status": "selection_completed", "scope": args.scope,
                                                      "cases": [c["name"] for c in cases], "at_utc": now()})
        except BaseException as error:
            write_json(output / "suite_status.json", {"status": "stopped", "scope": args.scope,
                                                      "error_type": type(error).__name__, "at_utc": now()})
            raise
        finally:
            summary(output, manifest)


if __name__ == "__main__":
    main()
