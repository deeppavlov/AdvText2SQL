"""Offline source/split audit and a separate bounded training-repeat protocol."""

import argparse
from collections import Counter, defaultdict
import copy
import json

import prepare_classifier_v2 as data
from configuration_splits import family_keys, make_groups, normalize_family
from durable_prompt_store import write_json

ROOT = data.ROOT
DEFAULT_OUTPUT = "research_results/classifier_data_expansion_20261110"
VERSION = "classifier-training-repeats-v1"
POOLS = {"bird": "data/train_queries.json", "ambrosia": "data/ambrosia_train.json"}
NEW_FILES = ("scripts/prepare_classifier_data_stage.py", "scripts/classifier_data_stage.py",
             "local/run_classifier_data_stage.sh", "scripts/configuration_splits.py", "scripts/audit_question_schemas.py")


def ranked(value):
    return data.fingerprint({"seed": 42, "key": value})


def audit_pools(old_rows, pools):
    """Namespace source IDs; protect all old dev/test DB and family components."""
    old_content = {(r["benchmark"], r["db_id"], normalize_family(r["inputs"]["question"], False)) for r in old_rows}
    candidates, seen = [], set()
    old_dbs = {(r["benchmark"], r["db_id"]) for r in old_rows}
    for benchmark, items in pools.items():
        if not isinstance(items, list):
            raise ValueError("Candidate source must be a list")
        for item in items:
            if (not all(k in item for k in ("db_id", "question", "SQL", "question_id"))
                    or not all(isinstance(item[k], str) and item[k].strip() for k in ("db_id", "question", "SQL"))):
                raise ValueError("Malformed candidate question")
            identity = data.fingerprint({"benchmark": benchmark, "db": item["db_id"],
                                         "question": item["question"], "evidence": item.get("evidence", "")})
            key = f"candidate:{benchmark}:{identity[:20]}"
            if key in seen:
                raise ValueError("Duplicate source input; resolve reference/provenance before collection")
            seen.add(key)
            candidates.append({"id": key, "benchmark": benchmark, "db_id": item["db_id"],
                               "source_question_id": str(item["question_id"]), "item": copy.deepcopy(item),
                               "inputs": {"question": item["question"]}, "family_keys": family_keys(item, benchmark),
                               "new_database": (benchmark, item["db_id"]) not in old_dbs,
                               "duplicate_old_input": (benchmark, item["db_id"], normalize_family(item["question"], False)) in old_content})
    assigned, groups, near = make_groups(old_rows + candidates)
    old = {r["id"]: r for r in old_rows}
    owners = {group: sorted({old[k]["split"] for k in keys if k in old}) for group, keys in groups.items()}
    for row in candidates:
        row["audit_component"] = assigned[row["id"]]
        row["old_partitions_in_component"] = owners[row["audit_component"]]
        row["status"] = ("duplicate_old_input" if row["duplicate_old_input"] else
                         "blocked_holdout_component" if set(row["old_partitions_in_component"]) & {"dev", "test"}
                         else "eligible_train_candidate")
    return candidates, near


def pilot_candidates(candidates, database_limit=40):
    """Forty new Ambrosia DBs, three inputs each; outcome-independent family variety."""
    by_db = defaultdict(list)
    for r in candidates:
        if r["benchmark"] == "ambrosia" and r["new_database"] and r["status"] == "eligible_train_candidate":
            by_db[r["db_id"]].append(r)
    buckets = defaultdict(list)
    for db, rows in by_db.items():
        if len(rows) >= 3:
            buckets[db.split("_", 1)[0]].append(db)
    for category in buckets:
        buckets[category].sort(key=ranked)
    selected = []
    while len(selected) < database_limit and any(buckets.values()):
        for category in sorted(buckets):
            if buckets[category] and len(selected) < database_limit:
                selected.append(buckets[category].pop(0))
    return [r for db in selected for r in sorted(by_db[db], key=lambda x: ranked(x["id"]))[:3]]


def select_repeats(examples, unstable_ids, controls_per_task=8):
    selected = {}
    for task in data.TASK_CONFIGURATIONS:
        rows = [r for r in examples if r["benchmark"] == task and r["split"] == "train"]
        targets = {}
        for row in rows:
            reasons = []
            if len(set(row["quality_targets"])) > 1:
                reasons.append("different_configuration_outcomes")
            if row["id"] in unstable_ids:
                reasons.append("archived_score_disagreement")
            if reasons:
                targets[row["id"]] = reasons
        # Agreement controls cover different groups and both success/failure where available.
        candidates = sorted((r for r in rows if r["id"] not in targets),
                            key=lambda r: (ranked(r["group_id"]), r["quality_targets"][0], ranked(r["id"])))
        seen, controls = set(), []
        for r in candidates:
            bucket = (r["group_id"], r["quality_targets"][0])
            if bucket not in seen and len(controls) < controls_per_task:
                seen.add(bucket)
                controls.append(r["id"])
        for identity in controls:
            targets[identity] = ["agreement_control"]
        selected[task] = dict(sorted(targets.items()))
    return selected


def prepare(output):
    source = ROOT / data.DEFAULT_OUTPUT / "experiment.json"
    original = data.read(source)
    data.validate(original)
    _, _, _, context = data.load_context()
    old_path = ROOT / data.OLD_DATASET / "records.jsonl"
    old = [json.loads(s) for s in old_path.read_text(encoding="utf-8").splitlines()]
    examples_path = ROOT / data.DATA_OUTPUT / "examples.jsonl"
    examples = [json.loads(s) for s in examples_path.read_text(encoding="utf-8").splitlines()]
    diagnosis_path = ROOT / "research_results/classifier_v2_diagnosis_20261109/archive_repeat_audit.json"
    diagnosis = data.read(diagnosis_path)
    unstable = {p["id"] for p in diagnosis["repeated_points"] if len({v["score"] for v in p["records"]}) > 1}
    candidates, near = audit_pools(old, {b: data.read(ROOT / p) for b, p in POOLS.items()})
    pilot = pilot_candidates(candidates)
    repeats = select_repeats(examples, unstable)
    paths = [source, old_path, examples_path, diagnosis_path, ROOT / data.DATA_OUTPUT / "manifest.json",
             *(ROOT / p for p in POOLS.values()), *(ROOT / p for p in NEW_FILES)]
    hashes = {p.relative_to(ROOT).as_posix(): data.digest(p) for p in paths}
    audit = {"version": "classifier-data-source-audit-v1", "source_sha256": hashes,
             "candidate_fingerprint": data.fingerprint(candidates), "pilot_fingerprint": data.fingerprint(pilot),
             "old_split_policy": "preserve all old train/dev/test; block old holdout DB/family/near-duplicate components",
             "sources": {b: {"path": POOLS[b], "questions": len([r for r in candidates if r["benchmark"] == b]),
                              "statuses": dict(Counter(r["status"] for r in candidates if r["benchmark"] == b)),
                              "eligible_databases": len({r["db_id"] for r in candidates if r["benchmark"] == b and r["status"] == "eligible_train_candidate"}),
                              "new_databases": len({r["db_id"] for r in candidates if r["benchmark"] == b and r["new_database"] and r["status"] == "eligible_train_candidate"})}
                         for b in POOLS},
             "near_duplicate_edges": near, "pilot_questions": len(pilot),
             "pilot_databases": len({r["db_id"] for r in pilot}),
             "pilot_status": "requires_live_schema_and_reference_name_audit; not_training_labels",
             "no_new_bird_database_locally": True, "live_llm_calls": 0}
    manifest = copy.deepcopy(original)
    manifest.pop("sha256")
    manifest.pop("cached_observations", None)
    manifest.update(version=VERSION, parent_manifest_sha256=original["sha256"],
                    sources_sha256={**original["sources_sha256"], **hashes},
                    selection=repeats, jobs=[], repeats=[2, 3],
                    purpose="selected train uncertainty audit; no best-of-repeat scoring, no automatic dataset rewrite",
                    limits={"examples": 80, "task_calls": 200, "teacher_calls": 0, "tokens": 1000000, "active_seconds": 7200},
                    global_limits={"examples": 1000, "task_calls": 2500, "teacher_calls": 0, "tokens": 8000000, "active_seconds": 21600})
    token_estimate = 0
    by_id = {r["id"]: r for r in examples}
    for repeat in (2, 3):
        for task, reasons in repeats.items():
            items = [context[key]["item"] for key in reasons]
            for name in data.TASK_CONFIGURATIONS[task]:
                manifest["jobs"].append({"name": f"r{repeat}__{task}__{name}", "repeat": repeat,
                    "benchmark": task, "split": "train", "configuration": name,
                    "features": list(data.CONFIGURATIONS[name]), "items": copy.deepcopy(items)})
                for key in reasons:
                    usage = by_id[key]["observations"][name]["usage"]
                    token_estimate += usage["prompt_tokens"] + usage["completion_tokens"]
    manifest["expected_question_evaluations"] = sum(len(j["items"]) for j in manifest["jobs"])
    manifest["estimated_tokens_from_original_usage"] = token_estimate
    manifest["sha256"] = data.fingerprint(manifest)
    repeat_root = output / "repeats"
    for p, expected in ((output / "source_audit.json", audit), (repeat_root / "experiment.json", manifest)):
        if p.exists() and data.read(p) != expected:
            raise ValueError("Existing preparation differs; use a new stage directory")
    repeat_root.mkdir(parents=True, exist_ok=True)
    for name, value in (("source_audit.json", audit), ("candidate_questions.json", candidates),
                        ("ambrosia_pilot.json", pilot)):
        write_json(output / name, value)
    write_json(repeat_root / "experiment.json", manifest)
    text = ["# Classifier data stage: offline audit", "", "| Source | Questions | Eligible | New eligible DBs |", "| --- | ---: | ---: | ---: |"]
    for b, info in audit["sources"].items():
        text.append(f"| {b} | {info['questions']} | {info['statuses'].get('eligible_train_candidate', 0)} | {info['new_databases']} |")
    text += ["", f"Ambrosia pilot: {len(pilot)} questions, {audit['pilot_databases']} new databases; schema preflight required.",
             f"Repeats: {len(manifest['jobs'])} jobs, {manifest['expected_question_evaluations']} question/config evaluations, two fresh passes.",
             f"Token estimate from original observations: {token_estimate:,}; not a billing guarantee.",
             "No paid calls, schema connections, training labels or partition changes during preparation."]
    (output / "summary.md").write_text("\n".join(text)+"\n", encoding="utf-8", newline="\n")
    return audit, manifest


def validate(manifest):
    if manifest.get("version") != VERSION or manifest["sha256"] != data.fingerprint({k: v for k, v in manifest.items() if k != "sha256"}):
        raise ValueError("Repeat protocol changed")
    for path, sha in manifest["sources_sha256"].items():
        if data.digest(data.local_path(ROOT, path)) != sha:
            raise ValueError(f"Pinned source changed: {path}")
    _, _, _, originals = data.load_context()
    for job in manifest["jobs"]:
        if job["features"] != list(data.CONFIGURATIONS[job["configuration"]]) or job["repeat"] not in (2, 3):
            raise ValueError("Foreign configuration/repeat")
        for item in job["items"]:
            row = originals[f"{job['benchmark']}:{item['question_id']}"]
            if row["split"] != "train" or row["item"] != item:
                raise ValueError("Repeat includes a changed question or holdout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", required=True)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = data.local_path(ROOT, args.output)
    if not output.is_relative_to(ROOT / "research_results") or output == ROOT / "research_results":
        parser.error("Use a dedicated research_results subfolder")
    audit, manifest = prepare(output)
    print(json.dumps({"sources": audit["sources"], "pilot": [audit['pilot_questions'], audit['pilot_databases']],
                      "repeat_jobs": len(manifest['jobs']), "repeat_evaluations": manifest['expected_question_evaluations'],
                      "estimated_repeat_tokens": manifest['estimated_tokens_from_original_usage'], "paid_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
