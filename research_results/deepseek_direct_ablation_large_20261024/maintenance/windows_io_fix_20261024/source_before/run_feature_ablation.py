"""Sequential, resumable feature ablations; no LLM calls on --dry-run or --status."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dotenv import dotenv_values

from feature_presets import ABLATION_FEATURES, ablation_cases
from llm_endpoints import api_backend, client_options, validate_connection

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = "research_results/deepseek_flash_ablation_large_20261024"
DIRECT_OUTPUT = "research_results/deepseek_direct_ablation_large_20261024"


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def protocol(config, provider):
    backend = api_backend(config["LLM_BASE_URL"])
    paths = [ROOT / "data/bird_large.json", ROOT / "data/ambrosia_large.json",
             ROOT / "data/ambrosia_large_available_227.json"]
    for path, expected in ((paths[0], 241), (paths[2], 227)):
        items = json.loads(path.read_text(encoding="utf-8"))
        if len(items) != expected or len({str(item["question_id"]) for item in items}) != expected:
            raise ValueError(f"Unexpected question coverage in {path.name}; review the datasets")
    for folder in ("scripts", "benchmarks", "src/adv_text2sql/mcp_servers/text2sql_tool/src"):
        paths.extend(sorted((ROOT / folder).glob("*.py")))
    return {
        "version": "feature-ablation-v1", "model": config["LLM_MODEL_NAME"],
        "base_url": config["LLM_BASE_URL"].rstrip("/"), "api_backend": backend,
        "model_version": config.get("LLM_MODEL_VERSION"),
        "provider": "direct" if backend == "deepseek" else provider or "auto",
        "request_options": client_options(backend, True, provider),
        "temperature": 0, "max_output_tokens": 4096, "reasoning": {"enabled": False},
        "max_retries": 5, "base_interval_seconds": 3, "min_interval_seconds": 3,
        "ambiguity_prompt": "current", "bird_questions": 241, "ambrosia_questions": 227,
        "files_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in paths},
    }


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@contextmanager
def suite_lock(output):
    """OS lock is released after a crash too; a stale lock file does not block resume."""
    with (output / "suite.lock").open("a+b") as stream:
        stream.seek(0)
        if stream.read(1) == b"":
            stream.write(b" ")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("This output folder already has an active ablation process") from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run_directory(output, case, benchmark):
    return output / case["name"] / ("ambrosia_227" if benchmark == "ambrosia" else "bird")


def health_reason(directory, benchmark, started, stall_seconds, error_limit):
    paths = [directory / name for name in ("metadata.json", "query_results.json", f"{benchmark}.log")]
    last_progress = max([started] + [p.stat().st_mtime for p in paths if p.exists()])
    if time.time() - last_progress > stall_seconds:
        return f"No log, metadata or checkpoint progress for {stall_seconds}s"
    predictions = read_json(directory / "query_results.json", {})
    consecutive = 0
    for result in reversed(list(predictions.values())):
        if result != "error":
            break
        consecutive += 1
    if consecutive >= error_limit:
        return f"{consecutive} consecutive generation errors; inspect logs before resuming"
    return None


def stop_child(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def monitor(child, directory, benchmark, args):
    started = time.time()
    last_notice = 0
    reason = None
    try:
        while child.poll() is None:
            reason = health_reason(directory, benchmark, started, args.stall_seconds, args.error_limit)
            if reason:
                raise RuntimeError(reason)
            if time.time() - last_notice >= 60:
                metadata = read_json(directory / "metadata.json", {})
                checkpoint = read_json(directory / "query_results.json", {})
                print(f"[WATCH] {directory.parent.name}/{benchmark}: "
                      f"phase={metadata.get('phase', 'starting')}, saved={len(checkpoint)}", flush=True)
                last_notice = time.time()
            time.sleep(2)
        if child.returncode:
            raise RuntimeError(f"Benchmark exited with code {child.returncode}; the series is stopped")
    except BaseException as error:
        reason = reason or type(error).__name__
        stop_child(child)
        metadata = read_json(directory / "metadata.json")
        if metadata and metadata.get("status") == "running":
            metadata.update(status="stopped_by_monitor", monitor_stop_reason=reason)
            save_json(directory / "metadata.json", metadata)
        save_json(directory.parent.parent / "monitor_stop.json",
                  {"at_utc": now(), "directory": str(directory), "reason": reason,
                   "child_pid": child.pid})
        raise


def accuracy(report):
    return report.get("overall_accuracy", report.get("accuracy")) if report else None


def summary(output, manifest):
    rows = []
    for case in manifest["cases"]:
        for benchmark in manifest["benchmarks"]:
            directory = run_directory(output, case, benchmark)
            metadata = read_json(directory / "metadata.json", {})
            report = read_json(directory / "report.json", {}) if metadata.get("status") == "completed" else {}
            checkpoint = read_json(directory / "query_results.json", {})
            control = read_json(run_directory(output, {"name": case["control"]}, benchmark) / "report.json", {}) \
                if case["control"] else {}
            score, parent_score = accuracy(report), accuracy(control)
            history = []
            history_path = directory / "attempt_history.jsonl"
            if history_path.exists():
                history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line]
            attempts = [*history, metadata]
            costs = [item["cost_usd_this_attempt"] for item in attempts
                     if item.get("cost_usd_this_attempt") is not None]
            cost = sum(costs) if costs else None
            native_deltas = {}
            for attempt in attempts:
                for currency, value in attempt.get("balance_delta_this_attempt", {}).items():
                    native_deltas[currency] = native_deltas.get(currency, 0) + value
            prompt_tokens = completion_tokens = calls = 0
            raw = directory / "raw_responses.jsonl"
            if raw.exists():
                with raw.open(encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            item = json.loads(line)
                        except ValueError:
                            continue
                        prompt_tokens += item["prompt_tokens"]
                        completion_tokens += item["completion_tokens"]
                        calls += 1
            rows.append({"case": case["name"], "features": case["enabled"], "control": case["control"],
                         "benchmark": benchmark, "status": metadata.get("status", "pending"),
                         "saved": len(checkpoint), "accuracy": score,
                         "delta_pp": score - parent_score if score is not None and parent_score is not None else None,
                         "f1": report.get("ambiguity_metrics", {}).get("f1"),
                         "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                         "successful_calls": calls, "key_usage_delta_usd": cost,
                         "account_balance_deltas": native_deltas})
    lines = ["# Feature ablation results", "", f"Model: `{manifest['protocol']['model']}`.", "",
             "| Configuration | Features | Control | Benchmark | Status | Saved | Accuracy | Delta, pp | F1 | Tokens | Account delta |",
             "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        score = f"{row['accuracy']:.2f}%" if row["accuracy"] is not None else "—"
        delta = f"{row['delta_pp']:+.2f}" if row["delta_pp"] is not None else "—"
        f1 = f"{100 * row['f1']:.2f}%" if row["f1"] is not None else "—"
        tokens = row["prompt_tokens"] + row["completion_tokens"]
        expense = ", ".join(f"{value:.4f} {currency}" for currency, value in row["account_balance_deltas"].items())
        if not expense:
            expense = f"{row['key_usage_delta_usd']:.4f} USD" if row["key_usage_delta_usd"] is not None else "—"
        lines.append(f"| {row['case']} | {','.join(map(str, row['features'])) or 'none'} | "
                     f"{row['control'] or '—'} | {row['benchmark']} | {row['status']} | {row['saved']} | "
                     f"{score} | {delta} | {f1} | {tokens} | {expense} |")
    lines.extend(["", "Delta is measured against the listed parent control, not always baseline.",
                  "Tokens include recorded probes/build calls and all attempts; key usage snapshots are approximate.",
                  "Ambrosia measures ambiguity labels; BIRD uses the existing local PostgreSQL evaluator."])
    save_json(output / "summary.json", rows)
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env.openrouter")
    parser.add_argument("--output", help="Default folder is separate for direct DeepSeek and OpenRouter")
    parser.add_argument("--features", nargs="+", type=int, default=list(ABLATION_FEATURES))
    parser.add_argument("--benchmark", choices=["bird", "ambrosia", "both"], default="both")
    parser.add_argument("--provider", help="OpenRouter provider slug; disables fallback to other providers")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--stall-seconds", type=int, default=900)
    parser.add_argument("--error-limit", type=int, default=8)
    args = parser.parse_args()
    config = dotenv_values(ROOT / args.env_file)
    direct = (config.get("LLM_BASE_URL", "").rstrip("/") in
              ("https://api.deepseek.com", "https://api.deepseek.com/v1") or Path(args.env_file).name == ".env.deepseek")
    output = (ROOT / (args.output or (DIRECT_OUTPUT if direct else DEFAULT_OUTPUT))).resolve()
    if not output.is_relative_to(ROOT / "research_results"):
        parser.error("--output must be inside research_results")
    if args.status:
        manifest = read_json(output / "experiment.json")
        if not manifest:
            parser.error("No experiment.json found; create the experiment with a manual launch first")
        rows = summary(output, manifest)
        for row in rows:
            print(f"{row['case']}/{row['benchmark']}: {row['status']}, saved={row['saved']}, accuracy={row['accuracy']}")
        return
    if args.stall_seconds < 60 or args.error_limit < 1:
        parser.error("stall-seconds must be >=60 and error-limit >=1")
    for key in ("LLM_MODEL_NAME", "LLM_API_KEY", "LLM_BASE_URL"):
        if not config.get(key):
            parser.error(f"Set {key} in {args.env_file}")
    try:
        backend = validate_connection(config, args.provider)
        cases = ablation_cases(args.features)
    except ValueError as error:
        parser.error(str(error))
    benchmark_names = ["bird", "ambrosia"] if args.benchmark == "both" else [args.benchmark]
    identity = protocol(config, args.provider)
    digest = fingerprint(identity)
    print(f"Model from {args.env_file}: {config['LLM_MODEL_NAME']}", flush=True)
    print(f"API: {backend}; provider: {identity['provider']}; "
          f"{len(cases)} configurations x {len(benchmark_names)} benchmarks", flush=True)
    for case in cases:
        print(f"  {case['name']}: features={case['enabled']}, control={case['control']}", flush=True)
    previous = read_json(output / "experiment.json")
    if previous and previous["protocol_sha256"] != digest:
        parser.error("Model, provider, source code or dataset changed; choose a new --output directory")
    if args.dry_run:
        print(f"DRY RUN: no SSH, database or API calls. Output: {output}")
        return
    output.mkdir(parents=True, exist_ok=True)
    with suite_lock(output):
        previous = read_json(output / "experiment.json")
        if previous and previous["protocol_sha256"] != digest:
            raise RuntimeError("Protocol changed while acquiring the suite lock; choose another output")
        if not previous and any(p.name != "suite.lock" for p in output.iterdir()):
            raise RuntimeError("Output has artifacts but no experiment.json; choose a new directory")
        manifest = previous or {"created_at_utc": now(), "protocol": identity,
                                "protocol_sha256": digest, "cases": [], "benchmarks": []}
        all_cases = {case["name"]: case for case in manifest["cases"]}
        all_cases.update({case["name"]: case for case in cases})
        manifest["cases"] = list(all_cases.values())
        manifest["benchmarks"] = sorted(set(manifest["benchmarks"] + benchmark_names))
        save_json(output / "experiment.json", manifest)
        save_json(output / "suite_status.json", {"status": "running", "at_utc": now(), "pid": os.getpid()})
        try:
            for case in cases:
                for benchmark in benchmark_names:
                    directory = run_directory(output, case, benchmark)
                    metadata = read_json(directory / "metadata.json", {})
                    if metadata.get("status") == "completed":
                        if metadata["configuration"].get("protocol_sha256") != digest:
                            raise RuntimeError("Completed run has another protocol identity")
                        print(f"[SKIP] {case['name']}/{benchmark}: completed", flush=True)
                        continue
                    command = [sys.executable, "-u", str(ROOT / "scripts/run_openrouter_baseline.py"),
                               "--configuration", "ablation" if case["enabled"] else "baseline",
                               "--benchmark", benchmark, "--model", config["LLM_MODEL_NAME"],
                               "--output", str(output / case["name"]), "--env-file", args.env_file,
                               "--protocol-sha256", digest]
                    if case["enabled"]:
                        command.extend(["--enabled-features", *map(str, case["enabled"])])
                    if args.provider:
                        command.extend(["--provider", args.provider])
                    print(f"[START] {case['name']}/{benchmark}", flush=True)
                    child = subprocess.Popen(command, cwd=ROOT)
                    monitor(child, directory, benchmark, args)
                    metadata = read_json(directory / "metadata.json", {})
                    if metadata.get("status") != "completed":
                        raise RuntimeError("Child exited without a completed benchmark")
                    summary(output, manifest)
            save_json(output / "suite_status.json", {"status": "completed", "at_utc": now()})
        except BaseException as error:
            save_json(output / "suite_status.json", {"status": "stopped", "at_utc": now(),
                                                     "error_type": type(error).__name__})
            raise
        finally:
            summary(output, manifest)


if __name__ == "__main__":
    main()
