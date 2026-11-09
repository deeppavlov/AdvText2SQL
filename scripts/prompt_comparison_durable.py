"""Same frozen experiment with a versioned, synced IO overlay; no model changes."""

import argparse
from contextlib import ExitStack
from unittest.mock import patch

import atomic_artifacts
import durable_prompt_store as store
import prompt_comparison as campaign
import prompt_comparison_runtime as runtime
import prompt_comparison_methods as methods
import prompt_pilot_budget as budget
import prompt_pilot_runtime as pilot_runtime
import static_hybrid_runtime as static_runtime

FILES = (
    "scripts/durable_prompt_store.py",
    "scripts/prompt_comparison_durable.py",
    "local/run_prompt_comparison_durable.ps1",
    "local/run_prompt_comparison_durable.sh",
)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", default=campaign.DEFAULT_OUTPUT)
    args, _ = parser.parse_known_args()
    output = campaign.pilot.output_path(args.output)
    expected = {
        "version": "synced-io-overlay-v1",
        "manifest_file_sha256": campaign.pilot.digest(output / "experiment.json"),
        "files_sha256": {p: campaign.pilot.digest(campaign.ROOT / p) for p in FILES},
        "policy": "Same prompts/generator/evaluator/budgets; flush+fsync before replace and after journal append",
    }
    path = output / "io_overlay.json"
    if path.exists():
        if campaign.search.read(path) != expected:
            raise ValueError("Durable IO overlay code changed; do not mix versions")
    else:
        store.write_json(path, expected)
    original_progress = runtime.Session.progress

    def progress(session, update):
        original_progress(session, {**update, "io_overlay": "synced-io-overlay-v1"})

    with ExitStack() as stack:
        for module in (
            atomic_artifacts,
            campaign,
            runtime,
            methods,
            budget,
            pilot_runtime,
            static_runtime,
        ):
            stack.enter_context(patch.object(module, "write_json", store.write_json))
        for module in (runtime, methods, pilot_runtime, static_runtime):
            stack.enter_context(patch.object(module, "append", store.append))
        stack.enter_context(patch.object(runtime.Session, "progress", progress))
        campaign.main()


if __name__ == "__main__":
    main()
