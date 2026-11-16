"""Offline readiness audit for a new classifier; never trains or calls an API."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from atomic_artifacts import write_json
from build_configuration_dataset import schema_statistics

ROOT = Path(__file__).resolve().parents[1]
OLD_DATASET = "data/multilabel_utility/deepseek_20261029"
STAGE_SUMMARY = "research_results/prompt_stage_summary_20261107.json"
OLD_MODEL = "research_results/multilabel_classifier_deepseek_20261029/model.json"
DEFAULT_OUTPUT = "research_results/classifier_refresh_20261109"
PARTITIONS = ("train", "dev", "test")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def local_path(root, value):
    path = (root / value.replace("\\", "/")).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path must stay inside the project")
    return path


def derive_cohort(records, partitions, full):
    """Intersect existing grouped splits with curated questions; never repartition."""
    records = list(records)
    by_id = {r["id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate old dataset ID")
    assignments = {}
    for split in PARTITIONS:
        for key in partitions[split]:
            if key in assignments:
                raise ValueError("Overlapping split IDs")
            assignments[key] = split
    if set(assignments) != set(by_id):
        raise ValueError("Splits must cover the original dataset exactly")
    current = {}
    for benchmark, items in full.items():
        for item in items:
            key = f"{benchmark}:{item['question_id']}"
            if key in current:
                raise ValueError("Duplicate current question ID")
            if key not in by_id:
                raise ValueError("Current question absent from original splits")
            old = by_id[key]
            if (old["benchmark"] != benchmark or old["db_id"] != item["db_id"]
                    or old["inputs"]["question"] != item["question"]
                    or old["inputs"]["evidence"] != item.get("evidence", "")
                    or old["split"] != assignments[key]):
                raise ValueError("Question identity/input/split changed")
            current[key] = {"item": item, "old": old, "split": assignments[key]}
    # Preserve DB, family, and old connected-component separation.
    memberships = defaultdict(set)
    for row in current.values():
        old = row["old"]
        for group in (f"db:{old['benchmark']}:{old['db_id']}", old["group_id"], *old["family_keys"]):
            memberships[group].add(row["split"])
    if any(len(v) != 1 for v in memberships.values()):
        raise ValueError("Database/family/group leakage across splits")
    return current, sorted(set(by_id) - set(current))


def train_seed_coverage(evaluations, candidate_id, benchmark, current):
    """Inspect only train records of the chosen arm, never holdout scores."""
    prefix = f"confirm:{candidate_id}:hybrid:False:{benchmark}:"
    observed = set()
    for key, record in evaluations.items():
        if not key.startswith(prefix):
            continue
        question_id = key[len(prefix):]
        identity = f"{benchmark}:{question_id}"
        if identity not in current or current[identity]["split"] != "train":
            continue
        item = current[identity]["item"]
        if str(record["question_id"]) != question_id or record["db_id"] != item["db_id"]:
            raise ValueError("Evaluation/question identity mismatch")
        score = record.get("score")
        if type(score) not in (float, int) or score not in (0, 1):
            raise ValueError("Expected observed binary score")
        if record.get("error"):
            # Do not infer model-failure labels from an unaudited error string.
            continue
        observed.add(identity)
    return sorted(observed)


def audit(root=ROOT):
    root = Path(root).resolve()
    old_path = root / OLD_DATASET
    old_manifest = read(old_path / "manifest.json")
    old_model = read(root / OLD_MODEL)
    if old_manifest["target_features"] != [f"FEAT_{n}" for n in old_model["feature_numbers"]]:
        raise ValueError("Legacy model and dataset target catalogs differ")
    for name in ("records.jsonl", "splits.json"):
        if digest(old_path / name) != old_manifest["output_files_sha256"][name]:
            raise ValueError("Original classifier dataset changed")
    records = [json.loads(line) for line in (old_path / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    splits = read(old_path / "splits.json")
    summary = read(root / STAGE_SUMMARY)
    full, manifests, sources = {}, {}, {}
    for benchmark in ("bird", "ambrosia"):
        profile = summary["profiles"][benchmark]
        path = local_path(root, profile["source_manifest_path"])
        manifest = read(path)
        if (manifest["sha256"] != profile["source_manifest_sha256"]
                or manifest["sha256"] != fingerprint({k: v for k, v in manifest.items() if k != "sha256"})
                or manifest["feature_mask"] != profile["features"]):
            raise ValueError("Frozen profile/manifest changed")
        report_path = path.parent / "report.json"
        if digest(report_path) != profile["source_report_sha256"]:
            raise ValueError("Source full report changed")
        full[benchmark] = manifest["full"][benchmark]
        manifests[benchmark] = manifest
        sources[path.relative_to(root).as_posix()] = digest(path)
        sources[report_path.relative_to(root).as_posix()] = digest(report_path)
    if manifests["bird"]["model"] != manifests["ambrosia"]["model"]:
        raise ValueError("Task models differ between current profiles")
    current, excluded = derive_cohort(records, splits["partitions"], full)
    cohorts = {s: sorted(k for k, r in current.items() if r["split"] == s) for s in PARTITIONS}
    coverage = {}
    for benchmark, manifest in manifests.items():
        profile = summary["profiles"][benchmark]
        candidate = (profile["sql_candidate"] if benchmark == "bird"
                     else manifest["jobs"][0]["arms"]["original"])
        candidate_id = fingerprint(candidate)
        if benchmark == "bird" and candidate_id != profile["candidate_id"]:
            raise ValueError("BIRD candidate changed")
        if candidate not in manifest["jobs"][0]["arms"].values():
            raise ValueError("Selected candidate absent from frozen job")
        folder = local_path(root, profile["source_manifest_path"]).parent
        path = folder / "jobs" / manifest["jobs"][0]["name"] / "evaluations.json"
        ids = train_seed_coverage(read(path), candidate_id, benchmark, current)
        sources[path.relative_to(root).as_posix()] = digest(path)
        changed_schema_inputs = []
        for key in cohorts["train"]:
            row = current[key]
            if row["old"]["benchmark"] != benchmark:
                continue
            snapshot = manifest["schema_audit"]["schemas"][benchmark][row["item"]["db_id"]]
            if schema_statistics(snapshot["rows"]) != row["old"]["inputs"]["schema_stats"]:
                changed_schema_inputs.append(key)
        coverage[benchmark] = {
            "features": profile["features"], "candidate_id": candidate_id,
            "metric": "execution_accuracy" if benchmark == "bird" else "ambiguity_detection_accuracy",
            "selected_arm_train_observations": len(ids), "train_ids_with_observation": ids,
            "train_schema_inputs_to_refresh": changed_schema_inputs,
            "verified_feature_utility_pairs_in_these_sources": 0,
            "reason": "Only one feature mask per benchmark at its chosen prompt; BIRD original arm changes prompt, not feature mask",
        }
    for path in (old_path / "manifest.json", old_path / "records.jsonl", old_path / "splits.json", root / STAGE_SUMMARY, root / OLD_MODEL):
        sources[path.relative_to(root).as_posix()] = digest(path)
    old_features = old_manifest["target_features"]
    return {
        "version": "classifier-refresh-readiness-v1", "status": "audit_only_not_training_dataset",
        "paid_calls": 0, "training_started": False, "old_artifacts_overwritten": False,
        "source_files_sha256": sources, "model": manifests["bird"]["model"],
        "partitions": cohorts,
        "split_counts": {s: dict(Counter(current[k]["old"]["benchmark"] for k in cohorts[s])) for s in PARTITIONS},
        "excluded_old_ids": excluded, "profiles": coverage,
        "legacy_classifier": {"head_features": old_features, "max_features": old_model["max_features"],
                              "can_reproduce_current_bird_mask": set(coverage["bird"]["features"]) <= set(old_model["feature_numbers"])
                              and old_model["max_features"] >= len(coverage["bird"]["features"])},
        "next_requirements": [
            "Version the catalog/constraints so FEAT_7 requires FEAT_2 and the five-feature BIRD control is representable",
            "Audit other historical archives before deciding which configuration observations need regeneration",
            "Keep model/prompt/evaluator compatibility explicit; old single-flag labels are not automatically compatible",
            "Build conditional configuration utility on train; unknown outcomes stay null with loss_mask=false",
            "Use dev for selection; previous test/full are already viewed and are not an independent final test",
        ],
    }


def markdown(report):
    lines = ["# Classifier refresh: offline readiness", "", "Audit only; no training labels, API calls or training.", "",
             "| Split | BIRD | Ambrosia | Total |", "| --- | ---: | ---: | ---: |"]
    for split, counts in report["split_counts"].items():
        lines.append(f"| {split} | {counts.get('bird', 0)} | {counts.get('ambrosia', 0)} | {sum(counts.values())} |")
    lines += ["", "| Benchmark | Fixed features | Train observations of selected arm | Verified feature-utility pairs |",
              "| --- | --- | ---: | ---: |"]
    for benchmark, profile in report["profiles"].items():
        lines.append(f"| {benchmark} | {'+'.join(map(str, profile['features']))} | {profile['selected_arm_train_observations']} | {profile['verified_feature_utility_pairs_in_these_sources']} |")
    lines += ["", "The two final full sources provide one chosen feature mask per task, not feature attribution.",
              "Other historical archives still need a compatibility audit. No old labels were copied.", "",
              "## Next requirements", "", *["- " + s for s in report["next_requirements"]], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = local_path(ROOT, args.output)
    if not output.is_relative_to(ROOT / "research_results"):
        raise ValueError("Store the audit under research_results")
    if output.exists() and any(p.name not in {"readiness.json", "readiness.md"} for p in output.iterdir()):
        raise ValueError("Use a separate audit folder, not an existing experiment")
    report = audit()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "readiness.json", report)
    (output / "readiness.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
