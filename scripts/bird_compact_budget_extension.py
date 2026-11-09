"""Explicit, audited budget-only continuation of the frozen compact dev series."""

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
from unittest.mock import patch

import bird_compact_transfer as compact

VERSION = "compact-budget-extension-v1"
LOCAL_TOKENS = 4_500_000
GLOBAL_TOKENS = 11_000_000
FILES = (
    "scripts/bird_compact_budget_extension.py",
    "local/run_bird_compact_budget_extension.ps1",
    "local/run_bird_compact_budget_extension.sh",
)


def proposal(root, manifest):
    paths = [root / "budget.json", *root.glob("jobs/*/confirm/budget.json")]
    snapshots = {}
    for path in paths:
        value = compact.read(path)
        if value.get("pending"):
            raise ValueError(
                "Settle pending API reservations before preparing an extension"
            )
        snapshots[str(path.relative_to(root))] = {
            "sha256": compact.digest(path),
            "data": value,
        }
    value = dict(
        version=VERSION,
        manifest_sha256=manifest["sha256"],
        created_at_utc=compact.source.search.now(),
        reason="Repeat 2 reached original 3M token cap before all fixed candidates were evaluated",
        original_limits=manifest["limits"],
        original_global_limits=manifest["global_limits"],
        effective_limits={**manifest["limits"], "tokens": LOCAL_TOKENS},
        effective_global_limits={**manifest["global_limits"], "tokens": GLOBAL_TOKENS},
        files_sha256={
            name: compact.digest(compact.source.ROOT / name) for name in FILES
        },
        snapshots_before_extension=snapshots,
        policy="Only token caps change. Manual extension launcher authorizes use; preparation is offline. No counters, answers, candidates, scoring, original manifest or teacher limits are reset/changed.",
    )
    value["sha256"] = compact.source.search.fingerprint(value)
    return value


def validate_extension(value, manifest):
    if value.get("version") != VERSION or value.get(
        "sha256"
    ) != compact.source.search.fingerprint(
        {k: v for k, v in value.items() if k != "sha256"}
    ):
        raise ValueError("Budget extension fingerprint changed")
    if (
        value["manifest_sha256"] != manifest["sha256"]
        or value["original_limits"] != manifest["limits"]
        or value["original_global_limits"] != manifest["global_limits"]
        or value["effective_limits"] != {**manifest["limits"], "tokens": LOCAL_TOKENS}
        or value["effective_global_limits"]
        != {**manifest["global_limits"], "tokens": GLOBAL_TOKENS}
        or set(value["files_sha256"]) != set(FILES)
    ):
        raise ValueError("Unexpected budget extension scope")
    for name, expected in value["files_sha256"].items():
        if compact.digest(compact.source.ROOT / name) != expected:
            raise ValueError(f"Pinned extension changed: {name}")


@contextmanager
def effective_ledgers(root, extension):
    """All original Ledger references share this class; keep persistent data intact."""
    original = compact.budget.Ledger.__init__
    root = root.resolve()

    def initialize(self, folder, limits, **kwargs):
        folder = Path(folder).resolve()
        if folder == root:
            expected, replacement = (
                extension["original_global_limits"],
                extension["effective_global_limits"],
            )
        elif folder.parent.parent == root / "jobs" and folder.name == "confirm":
            expected, replacement = (
                extension["original_limits"],
                extension["effective_limits"],
            )
        else:
            raise ValueError("Ledger outside authorized compact series")
        if limits != expected:
            raise ValueError("Unexpected original ledger limits")
        original(self, folder, replacement, **kwargs)

    with patch.object(compact.budget.Ledger, "__init__", initialize):
        yield


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prepare-extension", action="store_true")
    parser.add_argument("--output", default=compact.DEFAULT_OUTPUT)
    args, _ = parser.parse_known_args(argv)
    root = compact.output_path(args.output)
    manifest = compact.read(root / "experiment.json")
    compact.validate(manifest)
    path = root / "budget_extension.json"
    if args.prepare_extension:
        with compact.suite_lock(root):
            if not path.exists():
                compact.store.write_json(path, proposal(root, manifest))
            extension = compact.read(path)
            validate_extension(extension, manifest)
        print("OFFLINE: budget-only extension prepared; no generation/counter changes.")
        return 0
    extension = compact.read(path)
    validate_extension(extension, manifest)
    print(
        f"[BUDGET EXTENSION] Per repeat tokens: {LOCAL_TOKENS:,}; whole series: {GLOBAL_TOKENS:,}. All previous usage retained.",
        flush=True,
    )
    offline = any(
        flag in argv for flag in ("--dry-run", "--status", "--report", "--help")
    )
    if "--prepare" in argv:
        raise ValueError(
            "Use --prepare-extension; the original experiment stays frozen"
        )
    if not offline and "--worker" not in argv:
        with compact.suite_lock(root):
            compact.store.append(
                root / "resumption_events.jsonl",
                dict(
                    event="manual_budget_extension_launch",
                    at_utc=compact.source.search.now(),
                    extension_sha256=extension["sha256"],
                    manifest_sha256=manifest["sha256"],
                    effective_limits=extension["effective_limits"],
                    effective_global_limits=extension["effective_global_limits"],
                ),
            )
    # Frozen supervisor uses __file__ for child launch: children must use this
    # same audited overlay, not silently fall back to the old caps.
    with (
        effective_ledgers(root, extension),
        patch.object(compact, "__file__", str(Path(__file__).resolve())),
        patch.object(sys, "argv", [str(Path(__file__)), *argv]),
    ):
        result = compact.main()
    if "--dry-run" in argv:
        print(
            "Above are ORIGINAL manifest caps. Effective extension:",
            extension["effective_limits"],
            extension["effective_global_limits"],
        )
    return result


if __name__ == "__main__":
    sys.exit(main())
