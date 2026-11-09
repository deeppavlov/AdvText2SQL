"""Offline preparation for a manual prompt-comparison resume, without API calls."""

import argparse
from unittest.mock import patch

import durable_prompt_store as store
import prompt_comparison as campaign
import recover_prompt_transport as transport
from run_feature_ablation import suite_lock


def prepare(root, *, slot="all", method="all", repeat=None, phase="search"):
    """Account one stopped transient request; reject billing and unknown outcomes."""
    read = transport.recovery.read
    with suite_lock(root):
        manifest = read(root / "experiment.json")
        global_pending = read(root / "budget.json")["pending"]
        local_pending = []
        for path in sorted((root / "jobs").glob("*/*/budget.json")):
            if read(path)["pending"]:
                local_pending.append(path)
        if not global_pending and not local_pending:
            return "NO_PENDING: budgets ready; no accounting changes."
        if len(global_pending) != 1 or len(local_pending) != 1:
            raise ValueError(
                "Ambiguous pending reservations; inspect without resetting budgets"
            )
        path = local_pending[0]
        job_name, job_phase = path.parent.parent.name, path.parent.name
        job = next(j for j in manifest["jobs"] if j["name"] == job_name)
        if (
            phase != "search"
            or job_phase != phase
            or (slot != "all" and job["slot"] != slot)
            or (method != "all" and job["method"] != method)
            or (repeat is not None and not job_name.endswith(f"__r{repeat}"))
        ):
            raise ValueError(
                "Pending reservation belongs to another job/phase; accounting unchanged"
            )
        # Full manifest, stop-event and matching reservation checks are shared
        # with the audited standalone recovery. HTTP 402 is never allowed here.
        journal_path, journal = transport.build_journal(root, job_name)
        if journal["state"] == "applied":
            raise ValueError(
                "Applied event still has pending reservations; inspect manually"
            )
        with patch.object(transport.recovery, "write_json", store.write_json):
            transport.recovery.apply_journal(root, journal_path, journal)
        return (
            f"ACCOUNTED: {job_name}, HTTP {journal['transport_failure']['status_code']}, "
            f"{journal['unknown_usage_budget_tokens']} unknown tokens conservatively charged. "
            "Results and call counters preserved. No API retry. "
            f"Journal: {journal_path.name}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=campaign.DEFAULT_OUTPUT)
    parser.add_argument("--slot", choices=(*campaign.SLOTS, "all"), default="all")
    parser.add_argument("--method", choices=(*campaign.METHODS, "all"), default="all")
    parser.add_argument("--repeat", type=int, choices=(1, 2, 3))
    parser.add_argument("--phase", choices=("search", "confirm"), default="search")
    args, _ = parser.parse_known_args()
    print(
        prepare(
            campaign.pilot.output_path(args.output),
            slot=args.slot,
            method=args.method,
            repeat=args.repeat,
            phase=args.phase,
        )
    )


if __name__ == "__main__":
    main()
