"""Manually launched, bounded external resumes; frozen campaign stays unchanged."""

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import time

import durable_prompt_store as store
import prepare_prompt_resume as preparation
import prompt_comparison as campaign


def capped_runner(root, manifest, filters, max_jobs, command):
    """A restart must not expand the original --max-jobs authorization."""
    if max_jobs < 1:
        raise ValueError("--max-jobs must be positive")

    def pending():
        return [
            j["name"]
            for j in manifest["jobs"]
            if (filters["slot"] == "all" or j["slot"] == filters["slot"])
            and (filters["method"] == "all" or j["method"] == filters["method"])
            and (
                filters["repeat"] is None
                or j["name"].endswith(f"__r{filters['repeat']}")
            )
            and not (root / "jobs" / j["name"] / "selection.json").exists()
        ]

    authorized = pending()[:max_jobs] if filters["phase"] == "search" else None

    def run():
        if authorized is None:
            return subprocess.run(command, cwd=campaign.ROOT).returncode
        remaining = [
            name
            for name in authorized
            if not (root / "jobs" / name / "selection.json").exists()
        ]
        if not remaining:
            return 0
        if pending()[: len(remaining)] != remaining:
            raise ValueError(
                "Pending job order changed; refusing to expand the authorized queue"
            )
        return subprocess.run(
            [*command, "--max-jobs", str(len(remaining))], cwd=campaign.ROOT
        ).returncode

    return run


def supervise(
    root, *, filters, run_child, max_resumes=0, retry_delay=60, sleep=time.sleep
):
    """Every child remains fail-fast. Retry only a journaled transient task error."""
    manifest = preparation.transport.recovery.read(root / "experiment.json")

    def record(event, **fields):
        store.append(
            root / "resumption_events.jsonl",
            {
                "version": "bounded-external-resume-v1",
                "at_utc": campaign.search.now(),
                "manifest_sha256": manifest["sha256"],
                "event": event,
                **fields,
            },
        )

    print(preparation.prepare(root, **filters), flush=True)
    record(
        "manual_launch",
        filters=filters,
        max_transient_resumes=max_resumes,
        initial_delay_seconds=retry_delay,
        supervisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        preparation_sha256=hashlib.sha256(
            Path(preparation.__file__).read_bytes()
        ).hexdigest(),
    )
    resumes = 0
    while True:
        record("child_start", attempt=resumes + 1)
        code = run_child()
        record("child_exit", attempt=resumes + 1, exit_code=code)
        if code == 0:
            return 0
        try:
            result = preparation.prepare(root, **filters)
        except (ValueError, OSError, KeyError, StopIteration) as error:
            # Billing/auth/DB/unknown outcomes never trigger new paid requests.
            print(
                f"[RESUME] Stopped: accounting is not eligible for automatic resume ({type(error).__name__}).",
                flush=True,
            )
            record("resume_refused", error_type=type(error).__name__)
            return code
        print(result, flush=True)
        if not result.startswith("ACCOUNTED:"):
            record("resume_refused", reason="no_new_confirmed_transient_task_failure")
            return code
        if resumes >= max_resumes:
            print(
                "[RESUME] External resume limit reached; stop and inspect provider availability.",
                flush=True,
            )
            record("resume_limit_reached", completed_resumes=resumes)
            return code
        delay = retry_delay * 2**resumes
        resumes += 1
        print(
            f"[RESUME] Transient task failure accounted. Resume {resumes}/{max_resumes} in {delay}s. Saved evaluations will be reused.",
            flush=True,
        )
        record("wait_before_resume", resume=resumes, delay_seconds=delay)
        sleep(delay)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--transient-resumes", type=int, choices=range(4), default=0)
    args, forwarded = parser.parse_known_args()
    launcher = campaign.ROOT / "local/run_prompt_comparison_durable.ps1"
    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(launcher),
        *forwarded,
    ]
    offline_flags = {"--status", "--report", "--dry-run", "--prepare", "--help", "-h"}
    if offline_flags.intersection(forwarded):
        return subprocess.run(command, cwd=campaign.ROOT).returncode
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument("--output", default=campaign.DEFAULT_OUTPUT)
    options.add_argument("--phase", default="search", choices=("search", "confirm"))
    options.add_argument("--slot", default="all", choices=(*campaign.SLOTS, "all"))
    options.add_argument("--method", default="all", choices=(*campaign.METHODS, "all"))
    options.add_argument("--repeat", type=int, choices=(1, 2, 3))
    options.add_argument("--max-jobs", type=int, default=1)
    selected, _ = options.parse_known_args(forwarded)
    root = campaign.pilot.output_path(selected.output)
    filters = {
        key: getattr(selected, key) for key in ("phase", "slot", "method", "repeat")
    }
    manifest = preparation.transport.recovery.read(root / "experiment.json")
    return supervise(
        root,
        filters=filters,
        run_child=capped_runner(root, manifest, filters, selected.max_jobs, command),
        max_resumes=args.transient_resumes,
    )


if __name__ == "__main__":
    sys.exit(main())
