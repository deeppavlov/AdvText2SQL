"""Recheck AUD-007/008 label sources offline; preserve every original artifact."""

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import build_configuration_dataset as configuration
import build_multilabel_dataset as multilabel
from adv_text2sql.evaluation import validate_collected_observation as validate_observation

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = "research_results/evaluation_labels_audit_20261113"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line]


class Review:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.sources = {}
        self.points = []
        self.legacy = {}

    def path(self, relative):
        path = (self.root / relative).resolve()
        path.relative_to(self.root)
        return path

    def source(self, relative):
        path = self.path(relative)
        self.sources[path.relative_to(self.root).as_posix()] = digest(path)
        return path

    def bundle(self, relative):
        folder = self.path(relative)
        manifest = read(self.source(f"{relative}/manifest.json"))
        for name, expected in manifest.get("output_files_sha256", {}).items():
            if digest(folder / name) != expected:
                raise ValueError("Existing dataset output hash changed; stop label audit")
        # Historical code changed intentionally. Check the data/raw artifacts,
        # never update old manifests to match the new code.
        for name, expected in manifest.get("source_files_sha256", {}).items():
            if Path(name).suffix in {".json", ".jsonl"}:
                if digest(self.source(name)) != expected:
                    raise ValueError("Historical source artifact hash changed")
        for name, expected in manifest.get("observation_files_sha256", {}).items():
            if digest(self.source(name)) != expected:
                raise ValueError("Recorded observation file hash changed")
        return manifest

    @lru_cache(maxsize=None)
    def observations(self, folder, benchmark):
        items = read(self.source(configuration.DATASETS[benchmark]))
        predictions = read(self.source(f"{folder}/query_results.json"))
        report = read(self.source(f"{folder}/report.json"))
        if benchmark == "ambrosia":
            gold = {str(item["question_id"]): item for item in items}
            self.legacy[folder] = sum(value == "error" and gold[key]["SQL"] != "ambiguous"
                                      for key, value in predictions.items() if key in gold)
        return multilabel.recorded_observations(items, predictions, report, benchmark)

    def point(self, dataset, row, name, before, after, source, reason=None):
        changed = (before.get("observed"), before.get("correct")) != (after["observed"], after["correct"])
        self.points.append({
            "dataset": dataset, "id": row["id"], "split": row["split"], "group_id": row["group_id"],
            "configuration": name, "source": source, "before": before,
            "after": {"observed": after["observed"], "correct": after["correct"]},
            "changed": changed, "reason": reason,
        })

    def legacy_bundles(self):
        for relative, field, runs_key in (
            ("data/configuration_utility/deepseek_20261028", "outcomes", "configurations"),
            ("data/multilabel_utility/deepseek_20261029", "observations", "source_runs"),
        ):
            manifest = self.bundle(relative)
            for row in lines(self.source(f"{relative}/records.jsonl")):
                for name, before in row[field].items():
                    folder = manifest[runs_key][name]["benchmarks"][row["benchmark"]]["path"]
                    after = self.observations(folder, row["benchmark"])[str(row["question_id"])]
                    self.point(relative, row, name, {k: before[k] for k in ("observed", "correct")},
                               after, folder, after.get("evaluation_error_kind") or after.get("observation_issue"))

    @lru_cache(maxsize=None)
    def evaluation_archive(self, relative):
        return read(self.source(relative))

    def current_bundle(self):
        relative = "data/multilabel_utility/deepseek_20261109_v2"
        self.bundle(relative)
        items = {benchmark: {str(i["question_id"]): i for i in read(self.source(path))}
                 for benchmark, path in configuration.DATASETS.items()}
        for row in lines(self.source(f"{relative}/examples.jsonl")):
            benchmark, qid = row["id"].split(":", 1)
            for name, target, mask in zip(row["configurations"], row["quality_targets"], row["loss_mask"], strict=True):
                observation = row["observations"].get(name)
                if observation is None:
                    after, source = {"observed": False, "correct": None}, None
                else:
                    source = observation["source"]
                    value = self.evaluation_archive(source)[observation["key"]]
                    score = validate_observation(value, items[benchmark][qid], benchmark)
                    after = {"observed": score is not None, "correct": bool(score) if score is not None else None}
                self.point(relative, row, name, {"observed": mask, "correct": bool(target) if mask else None},
                           after, source, "untrusted_or_infrastructure_failure" if not after["observed"] else None)

    def result(self):
        summaries = []
        for dataset in sorted({p["dataset"] for p in self.points}):
            points = [p for p in self.points if p["dataset"] == dataset]
            summaries.append({"dataset": dataset, "observations": len(points),
                              "changed": sum(p["changed"] for p in points),
                              "unknown": sum(not p["after"]["observed"] for p in points),
                              "changed_by_split": dict(Counter(p["split"] for p in points if p["changed"]))})
        repairs = [p for p in self.points if p["before"]["observed"] and not p["after"]["observed"]]
        return {"version": "audit-007-008-offline-v1", "datasets": summaries,
                "legacy_ambrosia_errors_that_would_count_as_success": sum(self.legacy.values()),
                "repair_queue": repairs, "source_files_sha256": self.sources,
                "review_code_sha256": {name: digest(self.path(name)) for name in (
                    "scripts/audit_evaluation_labels.py", "scripts/build_configuration_dataset.py",
                    "scripts/build_multilabel_dataset.py", "scripts/prepare_classifier_v2.py",
                    "src/adv_text2sql/evaluation.py")},
                "network_calls": 0, "old_artifacts_modified": False,
                "scope": "Only AUD-007/008; other audit findings and retrospective holdout limits remain"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    review = Review(ROOT)
    output = review.path(args.output)
    if output.parent != ROOT / "research_results" or not output.name.startswith("evaluation_labels_audit_"):
        raise ValueError("Use a dedicated research_results/evaluation_labels_audit_* folder")
    if output.exists():
        raise FileExistsError("Audit outputs are immutable; choose a new folder")
    review.legacy_bundles()
    review.current_bundle()
    report = review.result()
    if any(digest(review.path(name)) != expected for name, expected in review.sources.items()):
        raise ValueError("An audit input changed during review")
    # Complete the read-only review before creating the destination.
    output.mkdir()
    (output / "impact.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "reviewed_observations.jsonl").open("w", encoding="utf-8") as stream:
        for point in review.points:
            stream.write(json.dumps(point, ensure_ascii=False) + "\n")
    print(json.dumps({"output": args.output, "datasets": report["datasets"],
                      "repair_queue": len(report["repair_queue"]), "network_calls": 0}, ensure_ascii=True))


if __name__ == "__main__":
    main()
