"""Offline, journaled recovery of unknown task usage after a comparison timeout.

No provider requests, no budget resets, no edits to frozen runtime or predictions.
Charge the entire outstanding reservation against the token budget; keep actual
input/output usage separate because the provider never supplied it.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from atomic_artifacts import write_json
import prompt_comparison as campaign
import static_hybrid_search as search
from run_feature_ablation import suite_lock


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def charge_unknown(data, event_id):
    """Upper-bound accounting only: actual token breakdown and calls stay intact."""
    result = deepcopy(data)
    if len(result["pending"]) != 1:
        raise ValueError("Recovery requires exactly one matching unresolved task call")
    ticket, request = next(iter(result["pending"].items()))
    cap = request.get("reserved")
    if request.get("role") != "task" or type(cap) is not int or cap <= 0:
        raise ValueError("Expected a positive task reservation")
    result["tokens"] += cap
    result["unknown_usage_tokens"] = result.get("unknown_usage_tokens", 0) + cap
    result.setdefault("unknown_usage_events", []).append(
        {
            "event_id": event_id,
            "ticket": ticket,
            "role": "task",
            "budget_tokens": cap,
            "actual_usage": None,
            "policy": "Full reservation charged as conservative budget estimate; not measured provider usage",
        }
    )
    del result["pending"][ticket]
    return result


def build_journal(root, job):
    manifest = read(root / "experiment.json")
    if manifest.get("version") != campaign.VERSION or manifest[
        "sha256"
    ] != search.fingerprint({k: v for k, v in manifest.items() if k != "sha256"}):
        raise ValueError("Unexpected or changed comparison manifest")
    if job not in {j["name"] for j in manifest["jobs"]}:
        raise ValueError("Unknown job")
    phase = root / "jobs" / job / "search"
    meta = read(phase / "metadata.json")
    event = meta.get("transport_failure", {})
    if (
        meta.get("status") != "stopped"
        or event.get("reason") != "call_timeout"
        or meta.get("manifest_sha256") != manifest["sha256"]
        or event.get("checkpoint_written") is not False
    ):
        raise ValueError("Expected a stopped task timeout without a checkpoint")
    event_id = search.fingerprint(
        {"job": job, "event": event, "manifest": manifest["sha256"]}
    )
    journal_path = root / "recoveries" / f"timeout_{event_id[:20]}.json"
    if journal_path.exists():
        journal = read(journal_path)
        if journal.get("event_id") != event_id:
            raise ValueError("Recovery identity mismatch")
        return journal_path, journal
    global_path, local_path = root / "budget.json", phase / "budget.json"
    global_, local = read(global_path), read(local_path)
    if (
        len(global_["pending"]) != 1
        or len(local["pending"]) != 1
        or list(global_["pending"].values()) != list(local["pending"].values())
    ):
        raise ValueError(
            "Global/local reservations differ or other calls remain unresolved"
        )
    patches = []
    for p, before in ((global_path, global_), (local_path, local)):
        patches.append(
            {
                "path": p.relative_to(root).as_posix(),
                "before": before,
                "before_file_sha256": sha(p),
                "after": charge_unknown(before, event_id),
            }
        )
    artifacts = {}
    for name in (
        "experiment.json",
        f"jobs/{job}/evaluations.json",
        f"jobs/{job}/candidates.json",
        f"jobs/{job}/search/teacher_cache.json",
        f"jobs/{job}/search/raw_responses.jsonl",
        f"jobs/{job}/search/textgrad_steps.jsonl",
    ):
        p = root / name
        if p.exists():
            artifacts[name] = sha(p)
    journal = {
        "version": "prompt-timeout-accounting-v1",
        "state": "prepared",
        "created_at_utc": search.now(),
        "event_id": event_id,
        "job": job,
        "transport_failure": event,
        "manifest_sha256": manifest["sha256"],
        "recovery_script_sha256": sha(Path(__file__)),
        "unknown_usage_budget_tokens": next(iter(local["pending"].values()))[
            "reserved"
        ],
        "actual_provider_usage": None,
        "paid_requests_started": False,
        "unchanged_artifacts_sha256": artifacts,
        "patches": patches,
    }
    return journal_path, journal


def apply_journal(root, path, journal):
    if journal["state"] == "applied":
        return False
    if journal["state"] != "prepared":
        raise ValueError("Unexpected recovery state")
    for name, expected in journal["unchanged_artifacts_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError("Source results changed during recovery; inspect journal")
    # Validate every file before the first write. A partial prior application
    # is accepted; changes from a concurrently started run are never overwritten.
    for patch in journal["patches"]:
        p = (root / patch["path"]).resolve()
        if not p.is_relative_to(root.resolve()):
            raise ValueError("Recovery target escapes experiment")
        if read(p) not in (patch["before"], patch["after"]):
            raise ValueError("Budget changed since recovery preparation")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        write_json(path, journal)  # Durable backup before either budget mutation.
    for patch in journal["patches"]:
        p = root / patch["path"]
        if read(p) != patch["after"]:
            write_json(p, patch["after"])
    journal = {**journal, "state": "applied", "applied_at_utc": search.now()}
    write_json(path, journal)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=campaign.DEFAULT_OUTPUT)
    parser.add_argument("--job", default="sql__textgrad_zero__r1")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply journaled offline accounting recovery",
    )
    args = parser.parse_args()
    root = campaign.pilot.output_path(args.output)
    with suite_lock(root):
        path, journal = build_journal(root, args.job)
        if args.apply:
            changed = apply_journal(root, path, journal)
            print("RECOVERED" if changed else "ALREADY RECOVERED", "No API/DB calls.")
        else:
            print("DRY RUN: no budget files changed.")
        print(
            "Job:",
            journal["job"],
            "question:",
            journal["transport_failure"]["question_id"],
        )
        print(
            "Unknown usage:",
            journal["unknown_usage_budget_tokens"],
            "tokens charged to budget as an upper bound, not measured usage.",
        )
        print(
            "Calls, actual prompt/completion usage, active time and saved predictions preserved."
        )
        print("Recovery journal:", path)


if __name__ == "__main__":
    main()
