"""Offline approval of a reviewed IO revision and backed-up technical checkpoint repair."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

from atomic_artifacts import write_json

IO_REVISION_FILES = {
    "scripts/run_openrouter_baseline.py", "scripts/run_feature_ablation.py",
    "scripts/atomic_artifacts.py", "scripts/ablation_maintenance.py",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def io_revision_changes(original, revision):
    first = {key: value for key, value in original.items() if key != "files_sha256"}
    second = {key: value for key, value in revision.items() if key != "files_sha256"}
    if first != second:
        raise ValueError("Model, endpoint or experimental settings changed; a new output folder is required")
    before, after = original["files_sha256"], revision["files_sha256"]
    changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
    if not set(changed) <= IO_REVISION_FILES:
        raise ValueError("Prompt, data or non-IO code changed; a new output folder is required")
    return changed


def single_flag_revision_changes(original, revision):
    """Reviewed plan change; generation settings, prompts and benchmark data stay fixed."""
    first = {key: value for key, value in original.items() if key not in ("files_sha256", "ablation_design")}
    second = {key: value for key, value in revision.items() if key not in ("files_sha256", "ablation_design")}
    if first != second or revision.get("ablation_design") != "one_flag_vs_all_false":
        raise ValueError("Single-flag plan migration must preserve model and generation settings")
    before, after = original["files_sha256"], revision["files_sha256"]
    changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
    if not set(changed) <= IO_REVISION_FILES | {"scripts/feature_presets.py"}:
        raise ValueError("Single-flag plan migration cannot change prompts, data or generator code")
    return changed


def approved_protocols(manifest, current):
    original = manifest["protocol"]
    if digest(original) != manifest["protocol_sha256"]:
        raise ValueError("Original manifest protocol hash is invalid")
    accepted = {manifest["protocol_sha256"]}
    for revision in manifest.get("execution_revisions", []):
        check = single_flag_revision_changes if revision.get("kind") == "single_flag_plan" else io_revision_changes
        check(original, revision["protocol"])
        if digest(revision["protocol"]) != revision["protocol_sha256"]:
            raise ValueError("Execution revision hash is invalid")
        accepted.add(revision["protocol_sha256"])
    if digest(current) not in accepted:
        raise ValueError("Model, provider, source code or dataset changed; approve the reviewed IO revision or use a new output folder")
    return accepted


def align_protocol(previous, current, approved):
    """Only the registered execution hash may differ; all other configuration stays strict."""
    previous = dict(previous)
    if (previous.get("protocol_sha256") and previous.get("protocol_sha256") in approved
            and current.get("protocol_sha256") and current.get("protocol_sha256") in approved):
        previous["protocol_sha256"] = current["protocol_sha256"]
    return previous


def prepare_repair(output, relative, question_id):
    output = Path(output).resolve()
    source = (output / relative).resolve()
    if not source.is_relative_to(output) or source == output:
        raise ValueError("Repair target must be a benchmark folder inside the experiment")
    metadata_path, checkpoint_path = source / "metadata.json", source / "query_results.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8-sig"))
    question_id = str(question_id)
    if metadata.get("technical_repair", {}).get("question_id") == question_id and question_id not in checkpoint:
        print("Technical repair is already prepared; no files changed")
        return
    if metadata.get("status") != "completed" or checkpoint.get(question_id) != "error":
        raise ValueError("Only a completed run with the specified error prediction can be prepared")
    log_path = source / ("ambrosia.log" if source.name == "ambrosia_227" else "bird.log")
    log = log_path.read_text(encoding="utf-8-sig")
    block = re.search(r"Processing q_id=" + re.escape(question_id) + r"\b.*?(?=Processing q_id=|\Z)", log, re.S)
    if not block or "PermissionError" not in block.group() or "metadata.json" not in block.group():
        raise ValueError("No metadata PermissionError is associated with this question; do not retry model failures selectively")
    backup = output / "maintenance/windows_io_fix_20261024" / ("_".join(source.relative_to(output).parts) + "_original")
    if backup.exists():
        raise ValueError("Backup already exists but the repair is not prepared; inspect it before continuing")
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, backup)
    hashes = {}
    for path in source.rglob("*"):
        if path.is_file():
            relative_path = path.relative_to(source)
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            if hashlib.sha256((backup / relative_path).read_bytes()).hexdigest() != expected:
                raise RuntimeError("Repair backup failed integrity verification")
            hashes[relative_path.as_posix()] = expected
    write_json(backup / "backup_integrity.json", hashes)
    checkpoint.pop(question_id)
    metadata.update(status="repair_pending", technical_repair={
        "question_id": question_id, "reason": "Windows metadata replacement failed after the model response",
        "backup": backup.relative_to(output).as_posix(), "retained_predictions": len(checkpoint),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    # Mark pending before changing the checkpoint, so an interruption never appears completed.
    write_json(metadata_path, metadata)
    write_json(checkpoint_path, checkpoint)
    write_json(output / "suite_status.json", {"status": "repair_pending",
               "at_utc": datetime.now(timezone.utc).isoformat(), "pending_directory": relative,
               "question_id": question_id})
    print(f"Repair prepared: {relative}, question {question_id}; {len(checkpoint)} predictions retained")


def main():
    from dotenv import dotenv_values
    from run_feature_ablation import ROOT, protocol, read_json, suite_lock, summary

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-file", default=".env.deepseek")
    parser.add_argument("--approve-io-revision", action="store_true")
    parser.add_argument("--use-single-feature-plan", action="store_true")
    parser.add_argument("--prepare-repair")
    parser.add_argument("--question-id")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "research_results"):
        parser.error("Output must be inside research_results")
    with suite_lock(output):
        manifest = read_json(output / "experiment.json")
        if not manifest:
            parser.error("Existing experiment.json is required")
        if args.approve_io_revision:
            current = protocol(dotenv_values(ROOT / args.env_file), None)
            changes = io_revision_changes(manifest["protocol"], current)
            current_digest = digest(current)
            revisions = manifest.setdefault("execution_revisions", [])
            if current_digest != manifest["protocol_sha256"] and not any(r["protocol_sha256"] == current_digest for r in revisions):
                revisions.append({"protocol_sha256": current_digest, "protocol": current,
                                  "approved_at_utc": datetime.now(timezone.utc).isoformat(),
                                  "reason": "Reviewed Windows artifact IO retries and pending-question resume; model, flags, data and prompts unchanged",
                                  "changed_files": changes})
                write_json(output / "experiment.json", manifest)
            print("Reviewed IO execution revision registered; original protocol preserved")
        if args.use_single_feature_plan:
            from feature_presets import ABLATION_FEATURES, ablation_cases
            current = protocol(dotenv_values(ROOT / args.env_file), None)
            changes = single_flag_revision_changes(manifest["protocol"], current)
            current_digest = digest(current)
            revisions = manifest.setdefault("execution_revisions", [])
            if not any(r["protocol_sha256"] == current_digest for r in revisions):
                revisions.append({"kind": "single_flag_plan", "protocol_sha256": current_digest,
                                  "protocol": current, "changed_files": changes,
                                  "approved_at_utc": datetime.now(timezone.utc).isoformat(),
                                  "reason": "User requested all-false baseline plus exactly one flag; every delta is against baseline"})
            combined = [case for case in manifest["cases"] if len(case["enabled"]) > 1]
            if combined:
                manifest.setdefault("archived_cases", []).extend(combined)
            manifest["cases"] = ablation_cases(ABLATION_FEATURES)
            manifest["active_ablation_design"] = "one_flag_vs_all_false"
            write_json(output / "experiment.json", manifest)
            summary(output, manifest)
            print("Single-feature plan registered: baseline + 33 flags; completed single-feature runs retained")
        if args.prepare_repair:
            if not args.question_id:
                parser.error("--prepare-repair requires --question-id")
            prepare_repair(output, args.prepare_repair, args.question_id)
            summary(output, manifest)


if __name__ == "__main__":
    main()
