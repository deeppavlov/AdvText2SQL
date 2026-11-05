"""Connection/monitor fix in a new output folder, preserving the frozen bounded campaign."""

import hashlib
import subprocess
import sys

from atomic_artifacts import write_json
import static_hybrid_campaign as campaign
import static_hybrid_search as search
from run_feature_ablation import monitor, now

DEFAULT_OUTPUT = "research_results/deepseek_combinations_large_20261026/final_search_20261030_connectionfix"
ORIGINAL_MAKE_MANIFEST = campaign.make_manifest
ORIGINAL_VALIDATE = campaign.validate
SOURCE_OUTPUT = campaign.DEFAULT_OUTPUT


def validate(manifest, *, code=True):
    ORIGINAL_VALIDATE(manifest, code=code)
    if (
        manifest["protocol"].get("connection_preflight", {}).get("version")
        != "resilient-catalog-v1"
    ):
        raise ValueError(
            "This is an old campaign folder; use the new connectionfix output"
        )


def make_manifest(config, audit_path, policy):
    manifest = ORIGINAL_MAKE_MANIFEST(config, audit_path, policy)
    manifest["protocol"]["runtime"] = "scripts/resilient_hybrid_runtime.py"
    manifest["protocol"]["connection_preflight"] = {
        "version": "resilient-catalog-v1",
        "attempts": 5,
        "retry_delay_seconds": 5,
        "connection_scope": "one read-only catalog connection per database",
        "retry_scope": "only transient database preflight errors; never LLM requests",
        "schema_policy": "same strict snapshot comparison; mismatch/auth/missing DB do not retry",
        "monitor": "create run folder before child; preserve preflight progress",
        "supersedes_unstarted_campaign": SOURCE_OUTPUT,
    }
    for name in (
        "scripts/resilient_hybrid_campaign.py",
        "scripts/resilient_hybrid_runtime.py",
        "local/run_resilient_hybrid_campaign.ps1",
        "local/run_resilient_hybrid_campaign.sh",
    ):
        manifest["protocol"]["files_sha256"][name] = hashlib.sha256(
            (campaign.ROOT / name).read_bytes()
        ).hexdigest()
    manifest["protocol_sha256"] = search.fingerprint(manifest["protocol"])
    return manifest


def run_jobs(output, manifest, scope, cases, args):
    if (
        manifest["protocol"].get("connection_preflight", {}).get("version")
        != "resilient-catalog-v1"
    ):
        raise ValueError(
            "Choose the new resilient campaign folder; old protocols are preserved"
        )
    write_json(
        output / "suite_status.json",
        {
            "status": "running",
            "phase": args.phase,
            "round": args.round,
            "at_utc": now(),
        },
    )
    try:
        for case in cases:
            for b in search.BENCHMARKS:
                if (
                    search.result(output, manifest, scope, case, b)["status"]
                    == "completed"
                ):
                    continue
                folder = search.run_folder(output, scope, case["name"], b)
                # The monitor may need to write diagnostics even when imports or
                # preflight fail before the child's normal metadata initialization.
                folder.mkdir(parents=True, exist_ok=True)
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(campaign.ROOT / "scripts/resilient_hybrid_runtime.py"),
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
                    raise RuntimeError("Incomplete resilient child run")
                campaign.campaign_report(output, manifest)
        write_json(
            output / "suite_status.json",
            {
                "status": "phase_completed",
                "phase": args.phase,
                "round": args.round,
                "at_utc": now(),
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
                "at_utc": now(),
            },
        )
        raise
    finally:
        campaign.campaign_report(output, manifest)


def main():
    original_output, original_make, original_jobs, original_validate = (
        campaign.DEFAULT_OUTPUT,
        campaign.make_manifest,
        campaign.run_jobs,
        campaign.validate,
    )
    campaign.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    campaign.make_manifest = make_manifest
    campaign.run_jobs = run_jobs
    campaign.validate = validate
    try:
        campaign.main()
    finally:
        campaign.DEFAULT_OUTPUT, campaign.make_manifest, campaign.run_jobs = (
            original_output,
            original_make,
            original_jobs,
        )
        campaign.validate = original_validate


if __name__ == "__main__":
    main()
