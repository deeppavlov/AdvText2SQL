"""Explicit transport revision and offline recovery of documented client failures."""

import argparse
import copy
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import sys

from atomic_artifacts import write_json
from run_feature_ablation import monitor, suite_lock
import resilient_hybrid_campaign as previous
import static_hybrid_campaign as campaign
import static_hybrid_search as search
from transport_safe_runtime import VERSION, CALL_TIMEOUT

DEFAULT_OUTPUT = "research_results/deepseek_combinations_large_20261026/final_search_20261030_transportfix"
SOURCE = previous.DEFAULT_OUTPUT
NEW_FILES = (
    "scripts/transport_safe_runtime.py",
    "scripts/transport_safe_campaign.py",
    "local/run_transport_safe_campaign.ps1",
    "local/run_transport_safe_campaign.sh",
)
FAILURE = re.compile(
    r"^(?:openai\.\w*Error:|httpx\.\w*Error:|TimeoutError:|ConnectionError:|"
    r"AttributeError: 'NoneType' object has no attribute '(?:usage|choices)')",
    re.M,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def affected_questions(log, predictions):
    """Use logged client failures, never gold scores or all `error` predictions."""
    last = {}
    for block in re.split(r"(?m)(?=^\d\d:\d\d:\d\d \[INFO\] query\(\) started:)", log):
        saved = re.findall(r"\[\d+/\d+\] question=(\d+) status=\w+", block)
        if not saved:
            continue  # An interrupted, uncheckpointed question needs no invalidation.
        if len(saved) != 1:
            raise ValueError("Ambiguous log/question mapping; inspect before recovery")
        qid = saved[0]
        if qid not in predictions:
            raise ValueError("Logged checkpoint question is missing from predictions")
        reasons = sorted(set(m.group(0) for m in FAILURE.finditer(block)))
        last[qid] = reasons
    return {q: reasons for q, reasons in last.items() if reasons}


def validate(manifest, *, code=True):
    previous.validate(manifest, code=code)
    if manifest["protocol"].get("transport", {}).get("version") != VERSION:
        raise ValueError("Use the separately migrated transport-safe campaign")


def finalists(output, manifest):
    """Freeze selection on its original snapshot, not on repaired control answers."""
    saved = search.read(output / "finalists.json")
    payload = {k: v for k, v in saved.items() if k != "sha256"}
    if (
        saved["sha256"] != search.fingerprint(payload)
        or saved["protocol_sha256"] != manifest["protocol_sha256"]
    ):
        raise ValueError("Frozen finalist identity changed")
    source = workspace_output(manifest["recovery_parent"]["path"])
    parent = search.read(source / "experiment.json")
    if (
        digest(source / "experiment.json")
        != manifest["recovery_parent"]["experiment_sha256"]
    ):
        raise ValueError("Original selection manifest changed")
    if campaign.screen_hashes(source, parent) != saved["source_sha256"]:
        raise ValueError("Original selection snapshot changed")
    if (
        campaign.choose_finalists(
            [
                campaign.measurement(source, parent, "train", name, 1)
                for name in campaign.SCREEN
            ]
        )
        != saved["candidates"]
    ):
        raise ValueError("Original finalist selection changed")
    return [campaign.BASELINE, campaign.CONTROL, *saved["candidates"]]


def workspace_output(path):
    path = (campaign.ROOT / path).resolve()
    root = (campaign.ROOT / "research_results").resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Use a dedicated folder inside research_results")
    return path


def migrate(source, output):
    source, output = workspace_output(source), workspace_output(output)
    if (
        source == output
        or source.is_relative_to(output)
        or output.is_relative_to(source)
    ):
        raise ValueError("Recovery source and destination must be disjoint")
    if output.exists():
        if (output / "migration_complete.json").exists():
            m = search.read(output / "experiment.json")
            validate(m)
            finalists(output, m)
            print("Already prepared; existing checkpoints left unchanged.")
            return output
        raise ValueError(
            "Destination exists without completed migration; choose a fresh folder"
        )
    with suite_lock(source):
        old = search.read(source / "experiment.json")
        previous.validate(old)
        if search.read(source / "suite_status.json").get("status") == "running":
            raise ValueError("Stop the source campaign before migration")
        selected = campaign.finalists(source, old)
        plan = []
        for p in sorted(source.glob("train/*/*/metadata.json")):
            folder = p.parent
            predictions = (
                search.read(folder / "query_results.json")
                if (folder / "query_results.json").exists()
                else {}
            )
            b = "bird" if folder.name == "bird" else "ambrosia"
            log = folder / f"{b}.log"
            affected = (
                affected_questions(log.read_text(encoding="utf-8"), predictions)
                if log.exists()
                else {}
            )
            if affected:
                plan.append(
                    {
                        "run": folder.relative_to(source).as_posix(),
                        "questions": {
                            q: {
                                "reason": reasons,
                                "original_prediction": predictions[q],
                            }
                            for q, reasons in affected.items()
                        },
                    }
                )
        new = copy.deepcopy(old)
        new["created_at_utc"] = search.now()
        new["protocol"]["runtime"] = "scripts/transport_safe_runtime.py"
        new["protocol"]["transport"] = {
            "version": VERSION,
            "call_deadline_seconds": CALL_TIMEOUT,
            "on_client_failure": "stop before checkpoint, including taxonomy/build calls",
            "automatic_question_retries": 0,
            "sdk_retries": "unchanged (2); bounded by outer call deadline",
            "model_format_errors": "unchanged; not selected for transport recovery",
            "imported_results": "prior guard; inference/model/masks/cohorts/evaluator unchanged",
        }
        for name in NEW_FILES:
            new["protocol"]["files_sha256"][name] = digest(campaign.ROOT / name)
        new["protocol_sha256"] = search.fingerprint(new["protocol"])
        new["recovery_parent"] = {
            "path": source.relative_to(campaign.ROOT).as_posix(),
            "protocol_sha256": old["protocol_sha256"],
            "experiment_sha256": digest(source / "experiment.json"),
        }
        validate(new)
        output.mkdir(parents=True, exist_ok=False)
        write_json(
            output / "migration_state.json",
            {"status": "preparing", "at_utc": search.now()},
        )
        sources = {}
        # No original file is edited, renamed or deleted. Refuse links escaping
        # the source even on Windows; publish completion only after validation.
        for p in sorted(source.iterdir()):
            if p.name not in ("train", "dev"):
                continue
            for item in sorted(p.rglob("*")):
                if item.is_symlink() or not item.resolve().is_relative_to(source):
                    raise ValueError("Unexpected link in source artifacts")
                if not item.is_file():
                    continue
                rel = item.relative_to(source)
                target = output / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                sources[rel.as_posix()] = digest(item)
                shutil.copy2(item, target)
        for p in output.glob("*/*/*/metadata.json"):
            meta = search.read(p)
            meta["configuration"]["protocol_sha256"] = new["protocol_sha256"]
            meta["imported_protocol_sha256"] = old["protocol_sha256"]
            write_json(p, meta)
            history = p.parent / "attempt_history.jsonl"
            if history.exists():
                import json

                attempts = [
                    json.loads(x)
                    for x in history.read_text(encoding="utf-8").splitlines()
                    if x.strip()
                ]
                for attempt in attempts:
                    if "configuration" in attempt:
                        attempt["configuration"]["protocol_sha256"] = new[
                            "protocol_sha256"
                        ]
                history.write_text(
                    "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in attempts),
                    encoding="utf-8",
                )
        from static_hybrid_runtime import append

        for entry in plan:
            folder = output / entry["run"]
            predictions = search.read(folder / "query_results.json")
            for q in entry["questions"]:
                del predictions[q]
            write_json(folder / "query_results.json", predictions)
            meta = search.read(folder / "metadata.json")
            append(
                folder / "attempt_history.jsonl",
                {**meta, "event": "archived_before_transport_recovery"},
            )
            meta = {
                k: v
                for k, v in meta.items()
                if k
                not in (
                    "balance_before",
                    "balance_after",
                    "balance_delta_this_attempt",
                    "finished_at_utc",
                    "started_at_utc",
                    "error_type",
                    "monitor_stop_reason",
                    "last_api_error_type",
                )
            }
            meta.update(
                status="recovery_pending",
                phase="recovery_pending",
                saved=len(predictions),
                consecutive_api_errors=0,
                recovery_ids=list(entry["questions"]),
            )
            write_json(folder / "metadata.json", meta)
            for name in ("report.json", "all_predicted_results.json"):
                p = folder / name
                if p.exists():
                    p.rename(folder / (p.stem + ".before_transport_recovery.json"))
        # Recheck every source hash before publishing a usable new manifest.
        if any(digest(source / rel) != sha for rel, sha in sources.items()):
            raise ValueError(
                "Source changed during migration; destination is incomplete"
            )
        write_json(
            output / "recovery_plan.json",
            {
                "created_at_utc": search.now(),
                "selection": "client exception log evidence only; no gold",
                "runs": plan,
                "questions": sum(len(x["questions"]) for x in plan),
                "source_sha256": sources,
                "parent": new["recovery_parent"],
            },
        )
        write_json(output / "experiment.json", new)
        frozen = copy.deepcopy(search.read(source / "finalists.json"))
        frozen.pop("sha256")
        frozen.update(
            protocol_sha256=new["protocol_sha256"],
            selection_source="unchanged original screening snapshot",
        )
        frozen["sha256"] = search.fingerprint(frozen)
        write_json(output / "finalists.json", frozen)
        if finalists(output, new) != selected:
            raise ValueError("Migration changed finalist selection")
        original_finalists = campaign.finalists
        campaign.finalists = finalists
        try:
            campaign.campaign_report(output, new)
        finally:
            campaign.finalists = original_finalists
        write_json(
            output / "suite_status.json",
            {"status": "recovery_prepared", "phase": "repeat", "round": 3},
        )
        write_json(
            output / "migration_complete.json",
            {"at_utc": search.now(), "protocol_sha256": new["protocol_sha256"]},
        )
        print(
            f"Prepared {output}; {sum(len(x['questions']) for x in plan)} questions pending recovery; no API calls."
        )
    return output


def run_jobs(output, manifest, scope, cases, args):
    write_json(
        output / "suite_status.json",
        {
            "status": "running",
            "phase": args.phase,
            "round": args.round,
            "at_utc": search.now(),
        },
    )
    try:
        for case in cases:
            for b in search.BENCHMARKS:
                if args.phase == "recovery":
                    targets = search.read(output / "recovery_plan.json")["runs"]
                    rel = (
                        search.run_folder(output, scope, case["name"], b)
                        .relative_to(output)
                        .as_posix()
                    )
                    if rel not in {entry["run"] for entry in targets}:
                        continue
                if (
                    search.result(output, manifest, scope, case, b)["status"]
                    == "completed"
                ):
                    continue
                folder = search.run_folder(output, scope, case["name"], b)
                folder.mkdir(parents=True, exist_ok=True)
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(campaign.ROOT / "scripts/transport_safe_runtime.py"),
                        "--search",
                        str(output),
                        "--env-file",
                        args.env_file,
                        "--scope",
                        scope,
                        "--case",
                        case["name"],
                        "--benchmark",
                        b,
                    ],
                    cwd=campaign.ROOT,
                )
                monitor(child, folder, b, args)
                if (
                    search.result(output, manifest, scope, case, b)["status"]
                    != "completed"
                ):
                    raise RuntimeError("Incomplete transport-safe child")
                campaign.campaign_report(output, manifest)
        write_json(
            output / "suite_status.json",
            {
                "status": "phase_completed",
                "phase": args.phase,
                "round": args.round,
                "at_utc": search.now(),
            },
        )
    except BaseException as error:
        write_json(
            output / "suite_status.json",
            {
                "status": "stopped",
                "phase": args.phase,
                "round": args.round,
                "error_type": type(error).__name__,
                "at_utc": search.now(),
            },
        )
        raise
    finally:
        campaign.campaign_report(output, manifest)


def reject_unprepared(*args, **kwargs):
    raise ValueError("Run transport_safe_campaign.py --prepare offline first")


def main():
    if "--prepare" in sys.argv[1:]:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--prepare", action="store_true")
        parser.add_argument("--source", default=SOURCE)
        parser.add_argument("--output", default=DEFAULT_OUTPUT)
        args = parser.parse_args()
        migrate(args.source, args.output)
        return
    saved = (
        campaign.DEFAULT_OUTPUT,
        campaign.validate,
        campaign.run_jobs,
        campaign.make_manifest,
        campaign.finalists,
    )
    campaign.DEFAULT_OUTPUT, campaign.validate = DEFAULT_OUTPUT, validate
    campaign.run_jobs, campaign.make_manifest = run_jobs, reject_unprepared
    campaign.finalists = finalists
    try:
        if "recovery" in sys.argv[1:]:
            parser = argparse.ArgumentParser(
                description="Explicit transport recovery; no automatic next phase"
            )
            parser.add_argument("--phase", choices=("recovery",), required=True)
            parser.add_argument("--output", default=DEFAULT_OUTPUT)
            parser.add_argument("--env-file", default=".env.deepseek")
            parser.add_argument("--dry-run", action="store_true")
            parser.add_argument("--stall-seconds", type=int, default=900)
            parser.add_argument("--error-limit", type=int, default=8)
            args = parser.parse_args()
            args.round = 3
            output = workspace_output(args.output)
            manifest = search.read(output / "experiment.json")
            validate(manifest)
            finalists(output, manifest)
            plan = search.read(output / "recovery_plan.json")
            names = {Path(entry["run"]).parts[1] for entry in plan["runs"]}
            cases = [case for case in manifest["cases"] if case["name"] in names]
            pending = []
            for entry in plan["runs"]:
                parts = Path(entry["run"]).parts
                b = "bird" if parts[-1] == "bird" else "ambrosia"
                case = next(c for c in cases if c["name"] == parts[1])
                value = search.result(output, manifest, "train", case, b)
                if value["status"] != "completed":
                    pending.append((entry["run"], value["questions"] - value["saved"]))
            print("Recovery runs and remaining predictions:", pending)
            if args.dry_run:
                print(
                    "DRY RUN: no files, SSH, DB or API calls. No automatic next phase."
                )
                return
            if args.stall_seconds < 60 or args.error_limit < 1:
                raise ValueError("stall>=60, error-limit>=1 required")
            with suite_lock(output):
                run_jobs(output, manifest, "train", cases, args)
        else:
            # Do not allow regular phases to consume unrepaired transport labels.
            # --status/--report remain useful during recovery.
            if not any(
                flag in sys.argv
                for flag in ("--status", "--report", "--help", "--freeze-finalists")
            ):
                parser = argparse.ArgumentParser(add_help=False)
                parser.add_argument("--output", default=DEFAULT_OUTPUT)
                opt, _ = parser.parse_known_args()
                output = workspace_output(opt.output)
                if (output / "recovery_plan.json").exists():
                    for entry in search.read(output / "recovery_plan.json")["runs"]:
                        if (
                            search.read(output / entry["run"] / "metadata.json")[
                                "status"
                            ]
                            != "completed"
                        ):
                            raise ValueError(
                                "Complete --phase recovery first; no paid call started"
                            )
            campaign.main()
    finally:
        (
            campaign.DEFAULT_OUTPUT,
            campaign.validate,
            campaign.run_jobs,
            campaign.make_manifest,
            campaign.finalists,
        ) = saved


if __name__ == "__main__":
    main()
