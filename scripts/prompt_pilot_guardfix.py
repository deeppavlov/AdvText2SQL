"""SQL-example guard revision; keep the completed v1 pilot and its code immutable."""

from contextlib import contextmanager
import re
from unittest.mock import patch

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

import prompt_pilot as base

DEFAULT_OUTPUT = "research_results/prompt_optimization_20261105_guardfix"
ORIGINAL_OUTPUT = base.DEFAULT_OUTPUT
REVISION = "sqlglot-instruction-guard-v2"
VERSION = "gepa-sql-pilot-guardfix-v2"
ORIGINAL_CHECK = base.check_candidate
ORIGINAL_MAKE = base.make_manifest
ORIGINAL_VALIDATE = base.validate
FILES = (
    "scripts/prompt_pilot_guardfix.py",
    "local/run_prompt_pilot_guardfix.ps1",
    "local/run_prompt_pilot_guardfix.sh",
)
SOURCE_FILES = (
    "experiment.json",
    "selection.json",
    "gepa_result.json",
    "evaluations.json",
    "search/metadata.json",
    "search/budget.json",
    "search/raw_responses.jsonl",
)


def corrected_check(text, manifest=None):
    # Preserve the protected-slot/length checks without the v1 cross-line regex.
    ORIGINAL_CHECK(text)
    if manifest is None or text == manifest["seed_instruction"]:
        return text
    for line in text.splitlines():
        for match in re.finditer(r"\bSELECT\b", line, re.I):
            fragment = line[match.start() :].strip().strip("`\"'").rstrip(".;")
            try:
                statement = sqlglot.parse_one(fragment, read="postgres")
            except (ParseError, TokenError):
                continue
            if isinstance(statement, exp.Query) and statement.find(exp.From):
                raise ValueError("Zero-shot instruction must not contain SQL examples")
    items = manifest["cohorts"]["fit"] + manifest["cohorts"]["inner_val"]
    if any(re.search(r"\b" + re.escape(i["db_id"]) + r"\b", text, re.I) for i in items):
        raise ValueError("Instruction must not name training databases")
    folded = " ".join(text.lower().split())
    for item in items:
        words = item["question"].lower().split()
        if any(
            " ".join(words[j : j + 8]) in folded for j in range(max(0, len(words) - 7))
        ):
            raise ValueError("Instruction must not copy training questions")
    return text


def make_manifest(env_file, teacher_env_file):
    parent = base.ROOT / ORIGINAL_OUTPUT
    ORIGINAL_VALIDATE(base.search.read(parent / "experiment.json"))
    if base.search.read(parent / "search/metadata.json")["status"] != "completed":
        raise ValueError("Keep the first pilot complete before starting the revision")
    manifest = ORIGINAL_MAKE(env_file, teacher_env_file)
    manifest["version"] = VERSION
    manifest["instruction_guard_revision"] = REVISION
    manifest["correction_of"] = {
        "path": ORIGINAL_OUTPUT,
        "files_sha256": {name: base.digest(parent / name) for name in SOURCE_FILES},
        "reason": "v1 rejected ordinary SELECT/FROM prose as SQL examples in two of four proposals",
        "restart_policy": "fresh bounded search and fresh control, no transfer of v1 caches or costs",
    }
    manifest["sha256"] = base.search.fingerprint(
        {k: v for k, v in manifest.items() if k != "sha256"}
    )
    validate(manifest)
    return manifest


def validate(manifest):
    if (
        manifest.get("version") != VERSION
        or manifest.get("instruction_guard_revision") != REVISION
    ):
        raise ValueError(
            "Use the separate guardfix experiment, never the original v1 folder"
        )
    if manifest["sha256"] != base.search.fingerprint(
        {k: v for k, v in manifest.items() if k != "sha256"}
    ):
        raise ValueError("Revised pilot manifest changed")
    # Reuse all v1 data/code/accounting checks while keeping entrypoint versions
    # incompatible: the original CLI must never run the corrected experiment.
    normalized = {**manifest, "version": "gepa-sql-pilot-v1"}
    normalized["sha256"] = base.search.fingerprint(
        {k: v for k, v in normalized.items() if k != "sha256"}
    )
    ORIGINAL_VALIDATE(normalized)
    parent = manifest["correction_of"]
    if parent["path"] != ORIGINAL_OUTPUT or any(
        base.digest(base.ROOT / parent["path"] / name) != sha
        for name, sha in parent["files_sha256"].items()
    ):
        raise ValueError("Original pilot evidence changed")


@contextmanager
def revision():
    with patch.multiple(
        base,
        DEFAULT_OUTPUT=DEFAULT_OUTPUT,
        FILES=(*base.FILES, *FILES),
        check_candidate=corrected_check,
        make_manifest=make_manifest,
        validate=validate,
    ):
        yield


def main():
    with revision():
        base.main()


if __name__ == "__main__":
    main()
