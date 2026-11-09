"""Offline accounting of one unresolved task request after a provider error.

The provider's real charge is unknown. Debit the existing full reservation,
preserve all call counters/results, and journal before/after budgets durably.
No API retries or changes to the frozen benchmark protocol. HTTP 402 requires
explicit billing-error accounting; that operation cannot restore account funds.
"""

import argparse
from pathlib import Path

import durable_prompt_store as store
import prompt_comparison as campaign
import recover_prompt_timeout as recovery
from run_feature_ablation import suite_lock
from unittest.mock import patch

TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


def build_journal(root, job, *, account_billing_error=False):
    manifest = recovery.read(root / "experiment.json")
    if manifest.get("version") != campaign.VERSION or manifest[
        "sha256"
    ] != campaign.search.fingerprint(
        {k: v for k, v in manifest.items() if k != "sha256"}
    ):
        raise ValueError("Unexpected or changed experiment manifest")
    if job not in {j["name"] for j in manifest["jobs"]}:
        raise ValueError("Unknown job")
    phase = root / "jobs" / job / "search"
    meta = recovery.read(phase / "metadata.json")
    event = meta.get("transport_failure", {})
    billing_error = account_billing_error and event.get("status_code") == 402
    allowed = event.get("reason") == "client_exception" and (
        event.get("status_code") in TRANSIENT_STATUS_CODES or billing_error
    )
    if (
        not allowed
        or meta.get("status") != "stopped"
        or meta.get("manifest_sha256") != manifest["sha256"]
        or event.get("checkpoint_written") is not False
    ):
        raise ValueError(
            "Expected a stopped transient HTTP error or explicitly acknowledged 402, without a checkpoint"
        )
    event_id = campaign.search.fingerprint(
        {"job": job, "event": event, "manifest": manifest["sha256"]}
    )
    path = root / "recoveries" / f"transport_{event_id[:20]}.json"
    if path.exists():
        journal = recovery.read(path)
        if journal.get("event_id") != event_id:
            raise ValueError("Recovery identity mismatch")
        return path, journal
    global_path, local_path = root / "budget.json", phase / "budget.json"
    global_, local = recovery.read(global_path), recovery.read(local_path)
    if (
        len(global_["pending"]) != 1
        or len(local["pending"]) != 1
        or list(global_["pending"].values()) != list(local["pending"].values())
    ):
        raise ValueError("Global/local pending reservations differ")
    patches = []
    for p, before in ((global_path, global_), (local_path, local)):
        after = recovery.charge_unknown(before, event_id)
        after["unknown_task_calls"] = after.get("unknown_task_calls", 0) + 1
        patches.append(
            {
                "path": p.relative_to(root).as_posix(),
                "before": before,
                "before_file_sha256": recovery.sha(p),
                "after": after,
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
        if (root / name).exists():
            artifacts[name] = recovery.sha(root / name)
    journal = {
        "version": "prompt-billing-http-accounting-v1"
        if billing_error
        else "prompt-transient-http-accounting-v1",
        "billing_error_accounting_only": billing_error,
        "state": "prepared",
        "event_id": event_id,
        "job": job,
        "created_at_utc": campaign.search.now(),
        "manifest_sha256": manifest["sha256"],
        "transport_failure": event,
        "actual_provider_usage": None,
        "paid_requests_started": False,
        "recovery_script_sha256": recovery.sha(Path(__file__)),
        "helper_script_sha256": recovery.sha(Path(recovery.__file__)),
        "unknown_usage_budget_tokens": next(iter(local["pending"].values()))[
            "reserved"
        ],
        "unchanged_artifacts_sha256": artifacts,
        "patches": patches,
    }
    return path, journal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=campaign.DEFAULT_OUTPUT)
    parser.add_argument("--job", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--account-billing-error",
        action="store_true",
        help="Allow offline conservative accounting for HTTP 402; no retry or balance recovery",
    )
    args = parser.parse_args()
    root = campaign.pilot.output_path(args.output)
    with suite_lock(root):
        path, journal = build_journal(
            root, args.job, account_billing_error=args.account_billing_error
        )
        if args.apply:
            with patch.object(recovery, "write_json", store.write_json):
                changed = recovery.apply_journal(root, path, journal)
            print("RECOVERED" if changed else "ALREADY RECOVERED")
        else:
            print("DRY RUN: no budget changes")
        print(
            "Job:",
            args.job,
            "HTTP:",
            journal["transport_failure"]["status_code"],
            "question:",
            journal["transport_failure"]["question_id"],
        )
        print(
            "Unknown usage charged conservatively:",
            journal["unknown_usage_budget_tokens"],
            "tokens. Actual usage and call counters preserved. No paid requests.",
        )
        print("Journal:", path)
        if journal.get("billing_error_accounting_only"):
            print("HTTP 402: check/top up the provider balance before a manual resume.")


if __name__ == "__main__":
    main()
