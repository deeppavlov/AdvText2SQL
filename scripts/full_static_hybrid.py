"""Manual, descriptive full-set evaluation of four user-frozen static profiles."""

import argparse
import copy
import subprocess
import sys

from dotenv import dotenv_values

from atomic_artifacts import write_json
import guarded_hybrid_search as guarded
from llm_endpoints import validate_connection
from run_feature_ablation import monitor, suite_lock
import static_hybrid_campaign as campaign
import static_hybrid_search as search
import transport_safe_campaign as source

DEFAULT_OUTPUT = "research_results/deepseek_full_static_20261103"
PROFILES = {
    "baseline_control": [],
    "hybrid_control_repeat": [14, 32, 35],
    "core_without_tsv": [14, 32],
    "statistics_compact": [2, 7, 14, 32, 35],
}
FILES = (
    "scripts/full_static_hybrid.py",
    "scripts/full_static_hybrid_runtime.py",
    "local/run_full_static_hybrid.ps1",
    "local/run_full_static_hybrid.sh",
)
VERSION = "descriptive-full-static-v1"


def checked_output(path):
    path = (campaign.ROOT / path).resolve()
    research = (campaign.ROOT / "research_results").resolve()
    if path == research or not path.is_relative_to(research):
        raise ValueError("Use a dedicated output folder inside research_results")
    return path


def make_manifest():
    parent = campaign.ROOT / source.DEFAULT_OUTPUT
    old = search.read(parent / "experiment.json")
    source.validate(old)
    conclusion = search.read(parent / "static_hybrid_conclusion.json")
    if (
        conclusion["sha256"]
        != search.fingerprint({k: v for k, v in conclusion.items() if k != "sha256"})
        or conclusion["protocol_sha256"] != old["protocol_sha256"]
        or conclusion["status"] != "bounded_search_and_confirmation_completed"
        or any(
            source.digest(parent / name) != sha
            for name, sha in conclusion["source_sha256"].items()
        )
    ):
        raise ValueError(
            "Confirmation evidence changed; inspect before full evaluation"
        )
    manifest = copy.deepcopy(old)
    manifest["created_at_utc"] = search.now()
    manifest["cases"] = [
        {"name": name, "features": flags, "stage": "full_descriptive"}
        for name, flags in PROFILES.items()
    ]
    manifest["cases_sha256"] = search.fingerprint(manifest["cases"])
    manifest["cohorts"] = {"full": copy.deepcopy(old["cohorts"]["full"])}
    manifest["cohorts_sha256"] = search.fingerprint(manifest["cohorts"])
    manifest.pop("recovery_parent", None)
    protocol = manifest["protocol"]
    protocol.pop("bounded_campaign", None)
    protocol.update(
        runtime="scripts/full_static_hybrid_runtime.py",
        scope_policy="full descriptive evaluation after train/dev; overlapping viewed data, not independent test",
        selection="no feature selection on full; four profiles explicitly fixed by user",
        full_evaluation={
            "version": VERSION,
            "profiles": PROFILES,
            "rounds": 1,
            "expected_counts": {"bird": 241, "ambrosia": 211},
        },
    )
    for name in FILES:
        protocol["files_sha256"][name] = source.digest(campaign.ROOT / name)
    manifest["confirmation_source"] = {
        "path": source.DEFAULT_OUTPUT,
        "files_sha256": {
            name: source.digest(parent / name)
            for name in (
                "experiment.json",
                "static_hybrid_conclusion.json",
                "campaign_report.json",
            )
        },
    }
    # Bind provenance to the frozen protocol too.
    protocol["confirmation_source_sha256"] = search.fingerprint(
        manifest["confirmation_source"]
    )
    manifest["protocol_sha256"] = search.fingerprint(protocol)
    validate(manifest)
    return manifest


def validate(manifest, *, code=True):
    guarded.validate(manifest, code=code)
    policy = manifest["protocol"].get("full_evaluation", {})
    if policy.get("version") != VERSION or policy.get("profiles") != PROFILES:
        raise ValueError("Full evaluation profile policy changed")
    if {c["name"]: c["features"] for c in manifest["cases"]} != PROFILES or len(
        manifest["cases"]
    ) != 4:
        raise ValueError("Expected exactly the four agreed profiles")
    if set(manifest["cohorts"]) != {"full"}:
        raise ValueError("Full manifest must not reuse train/dev scope labels")
    for b, count in policy["expected_counts"].items():
        items = manifest["cohorts"]["full"][b]
        if len(items) != count or len({str(i["question_id"]) for i in items}) != count:
            raise ValueError("Unexpected or duplicate full-cohort questions")
    guarded.gate(manifest, "full", search.BENCHMARKS)
    if code and manifest["protocol"]["environment"] != campaign.environment():
        raise ValueError("Python/package versions changed")
    parent = manifest["confirmation_source"]
    if search.fingerprint(parent) != manifest["protocol"]["confirmation_source_sha256"]:
        raise ValueError("Confirmation provenance changed")
    parent_path = checked_output(parent["path"])
    if any(
        source.digest(parent_path / name) != sha
        for name, sha in parent["files_sha256"].items()
    ):
        raise ValueError("Frozen confirmation source changed")


def report(output, manifest, *, write=False):
    rows = []
    for case in manifest["cases"]:
        values = {
            b: search.result(output, manifest, "full", case, b)
            for b in search.BENCHMARKS
        }
        complete = all(v["status"] == "completed" for v in values.values())
        rows.append(
            {
                "configuration": case["name"],
                "features": case["features"],
                "completed": complete,
                "benchmarks": values,
                "macro_accuracy": sum(v["accuracy"] for v in values.values()) / 2
                if complete
                else None,
                "tokens": sum(
                    v.get("prompt_tokens", 0) + v.get("completion_tokens", 0)
                    for v in values.values()
                ),
            }
        )
    control = next(r for r in rows if r["configuration"] == "hybrid_control_repeat")
    for row in rows:
        if row["completed"] and control["completed"]:
            row["comparison_H"] = {
                b: {
                    "rescued": sum(
                        v and not control["benchmarks"][b]["outcomes"][q]
                        for q, v in row["benchmarks"][b]["outcomes"].items()
                    ),
                    "harmed": sum(
                        not v and control["benchmarks"][b]["outcomes"][q]
                        for q, v in row["benchmarks"][b]["outcomes"].items()
                    ),
                }
                for b in search.BENCHMARKS
            }
    result = {
        "protocol_sha256": manifest["protocol_sha256"],
        "status": "completed" if all(r["completed"] for r in rows) else "pending",
        "purpose": "descriptive full table; not an independent test or new feature selection",
        "rows": rows,
    }
    if write:
        lines = [
            "# Полные наборы: четыре статических профиля",
            "",
            "Один повтор после train/dev. Данные уже просматривались; это описательная таблица, не независимый test.",
            "",
            "| Профиль | Фичи | BIRD | Ambrosia | Средняя Accuracy | F1 Ambrosia | Токены |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            cells = []
            for b in search.BENCHMARKS:
                v = row["benchmarks"][b]
                cells.append(
                    f"{v['correct']}/{v['questions']} ({v['accuracy']:.2f}%)"
                    if v["status"] == "completed"
                    else f"{v['saved']}/{v['questions']}, {v['status']}"
                )
            score = (
                f"{row['macro_accuracy']:.2f}%"
                if row["macro_accuracy"] is not None
                else "—"
            )
            f1 = row["benchmarks"]["ambrosia"].get("ambiguity_metrics", {}).get("f1")
            f1_text = f"{100 * f1:.2f}%" if f1 is not None else "—"
            lines.append(
                f"| {row['configuration']} | {'+'.join(map(str, row['features'])) or 'все false'} | {cells[0]} | {cells[1]} | {score} | {f1_text} | {row['tokens']} |"
            )
        write_json(output / "full_report.json", result)
        (output / "full_report.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    return result


def run_jobs(output, manifest, args):
    write_json(
        output / "suite_status.json",
        {"status": "running", "scope": "full", "at_utc": search.now()},
    )
    try:
        for case in manifest["cases"]:
            for b in search.BENCHMARKS:
                if (
                    search.result(output, manifest, "full", case, b)["status"]
                    == "completed"
                ):
                    continue
                folder = search.run_folder(output, "full", case["name"], b)
                folder.mkdir(parents=True, exist_ok=True)
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(campaign.ROOT / "scripts/full_static_hybrid_runtime.py"),
                        "--search",
                        str(output),
                        "--env-file",
                        args.env_file,
                        "--scope",
                        "full",
                        "--case",
                        case["name"],
                        "--benchmark",
                        b,
                    ],
                    cwd=campaign.ROOT,
                )
                monitor(child, folder, b, args)
                if (
                    search.result(output, manifest, "full", case, b)["status"]
                    != "completed"
                ):
                    raise RuntimeError("Incomplete full benchmark")
                report(output, manifest, write=True)
        write_json(
            output / "suite_status.json",
            {"status": "completed", "scope": "full", "at_utc": search.now()},
        )
    except BaseException as error:
        write_json(
            output / "suite_status.json",
            {
                "status": "stopped",
                "scope": "full",
                "error_type": type(error).__name__,
                "at_utc": search.now(),
            },
        )
        raise
    finally:
        report(output, manifest, write=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", default=".env.deepseek")
    parser.add_argument("--stall-seconds", type=int, default=900)
    parser.add_argument("--error-limit", type=int, default=8)
    action = parser.add_mutually_exclusive_group()
    for flag in ("prepare", "dry-run", "status", "report"):
        action.add_argument("--" + flag, action="store_true")
    args = parser.parse_args()
    if args.stall_seconds < 60 or args.error_limit < 1:
        parser.error("stall>=60, error-limit>=1 required")
    output = checked_output(args.output)
    parent = checked_output(source.DEFAULT_OUTPUT)
    if (
        output == parent
        or output.is_relative_to(parent)
        or parent.is_relative_to(output)
    ):
        parser.error("Use a separate full output folder; never the confirmation folder")
    path = output / "experiment.json"
    if path.exists():
        manifest = search.read(path)
        validate(manifest)
    elif args.status or args.report:
        parser.error("Prepare the full series first")
    else:
        if output.exists() and any(output.iterdir()):
            parser.error("Output exists without a manifest; use a new folder")
        manifest = make_manifest()
    if args.status:
        summary = report(output, manifest)
        print(
            "State:",
            summary["status"],
            "complete profiles:",
            sum(r["completed"] for r in summary["rows"]),
            "/4",
        )
        return
    if args.prepare or args.report:
        output.mkdir(parents=True, exist_ok=True)
        with suite_lock(output):
            if not path.exists():
                write_json(path, manifest)
            report(output, manifest, write=True)
        print("Offline preparation/report; no SSH, DB or API calls:", output)
        return
    config = dotenv_values(campaign.ROOT / args.env_file)
    validate_connection(config)
    model = manifest["protocol"]["model"]
    if (
        config.get("LLM_MODEL_NAME") != model["model"]
        or config.get("LLM_MODEL_VERSION") != model["model_version"]
        or config.get("LLM_BASE_URL", "").rstrip("/") != model["base_url"].rstrip("/")
    ):
        parser.error("Env differs from frozen model")
    pending = sum(
        v["questions"] - v["saved"]
        for r in report(output, manifest)["rows"]
        for v in r["benchmarks"].values()
        if v["status"] != "completed"
    )
    print("FULL: four profiles, eight benchmark runs, remaining predictions:", pending)
    print("Profiles:", PROFILES, "; BIRD 241 / Ambrosia 211")
    if args.dry_run:
        print("DRY RUN: no files, SSH, DB or API calls.")
        return
    output.mkdir(parents=True, exist_ok=True)
    with suite_lock(output):
        if path.exists() and search.read(path) != manifest:
            raise ValueError("Manifest changed while acquiring lock")
        if not path.exists():
            if any(p.name != "suite.lock" for p in output.iterdir()):
                raise ValueError("Unprepared output is not empty")
            write_json(path, manifest)
        run_jobs(output, manifest, args)


if __name__ == "__main__":
    main()
