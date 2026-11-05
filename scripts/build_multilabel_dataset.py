"""Prepare masked multilabel benefit observations offline; never train or call an LLM."""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

from atomic_artifacts import write_json
import build_configuration_dataset as parent
from feature_presets import feature_flags
from multilabel_targets import (
    CONDITIONAL_PAIRS, EXCLUSIONS, FEATURES, TARGET_FEATURES, benefit_label,
    feature_catalog, single_feature_labels, target_vectors,
)
from research_metrics import ambrosia_report

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = "data/multilabel_utility/deepseek_20261029"
FORMAT = "multilabel-benefit-v1"
OUTPUTS = ("records.jsonl", "examples.jsonl", "conditional_labels.jsonl", "splits.json",
           "cv_folds.json", "ambrosia_family_holdout.json", "summary.json", "summary.md")
BUILDER_FILES = ("scripts/build_multilabel_dataset.py", "scripts/multilabel_targets.py",
                 "scripts/build_configuration_dataset.py", "scripts/configuration_splits.py",
                 "scripts/research_metrics.py", "scripts/feature_presets.py", "scripts/atomic_artifacts.py")
EXTRA_CASES = {
    "schema_2_35": {"experiment": "combinations", "features": [2, 35]},
    "sql_rules": {"experiment": "combinations", "features": [14, 35]},
    "sql_ambiguity": {"experiment": "combinations", "features": [14, 32, 35]},
    "hybrid_with_relationships": {"experiment": "relationships", "features": [1, 14, 32, 35]},
}


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_lines(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def recorded_observations(items, predictions, report, benchmark):
    """A present but blank answer is invalid, not an Ambrosia true negative."""
    ids = {str(item["question_id"]) for item in items}
    if not set(predictions) <= ids:
        raise ValueError("Predictions contain unknown question IDs")
    if any(not isinstance(value, str) for value in predictions.values()):
        raise ValueError("Non-string predictions require a separate source audit")
    blank = {key for key, value in predictions.items() if not value.strip()}
    clean = {key: value for key, value in predictions.items() if key not in blank}
    # BIRD report SQL must still match the saved prediction, even for blank values.
    if benchmark == "bird":
        for row in report.get("results", []):
            key = str(row["question_id"])
            if key in predictions and row.get("predicted_sql") != predictions[key]:
                raise ValueError("BIRD prediction differs from the scored SQL")
    result = parent.observations(items, clean, report, benchmark)
    for key, value in result.items():
        value["invalid_prediction"] = key in blank
        value["observation_issue"] = (
            "blank_saved_prediction" if key in blank else
            "prediction_missing" if key not in predictions else
            "reference_or_unclassified_evaluation_error" if not value["observed"] else None)
    return result


def validate_run_metadata(meta, manifest, accepted, benchmark, enabled, dataset_hash):
    if meta.get("status") != "completed":
        raise ValueError("Source run is not completed")
    protocol = manifest["protocol"]
    expected = {key: protocol.get(key) for key in
                ("model", "model_version", "api_backend", "base_url", "provider", "request_options")}
    expected.update(temperature=0, max_output_tokens=4096, reasoning_effort="none", max_retries=5,
                    use_evidence=benchmark == "bird", dataset_sha256=dataset_hash,
                    feature_flags={f"FEAT_{n}": n in enabled for n in range(1, 37)})
    identity = meta["configuration"]
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("Incompatible model, settings, data, or feature flags in source metadata")
    if identity.get("protocol_sha256") not in accepted:
        raise ValueError("Unknown source execution protocol")


def read_run(root, folder, benchmark, items, enabled, manifest, accepted):
    meta = parent.read(folder / "metadata.json")
    validate_run_metadata(meta, manifest, accepted, benchmark, enabled, parent.sha(root / parent.DATASETS[benchmark]))
    predictions, report = parent.read(folder / "query_results.json"), parent.read(folder / "report.json")
    observed = recorded_observations(items, predictions, report, benchmark)
    valid = {key: value for key, value in observed.items() if value["observed"]}
    correct = sum(value["correct"] for value in valid.values())
    summary = {
        "path": folder.relative_to(root).as_posix(), "features": enabled,
        "protocol_sha256": meta["configuration"]["protocol_sha256"],
        "questions": len(items), "saved_predictions": len(predictions), "observed": len(valid),
        "correct_observed": correct, "invalid_predictions": sum(v["invalid_prediction"] for v in observed.values()),
        "accuracy_on_observed": 100 * correct / len(valid) if valid else None,
        "reported_accuracy": report.get("overall_accuracy", report.get("accuracy")),
        "generation_errors": sum(bool(v["generation_error"]) for v in observed.values()),
        "execution_errors": sum(bool(v["execution_error"]) for v in observed.values()),
        "exact_question_cost_available": False,
    }
    if benchmark == "bird" and len(valid) == len(items):
        if abs(report["overall_accuracy"] - 100 * correct / len(items)) > 1e-8:
            raise ValueError("BIRD aggregate disagrees with saved question scores")
    if benchmark == "ambrosia" and len(valid) == len(items):
        strict = ambrosia_report(items, predictions)
        summary["strict_accuracy"] = strict["accuracy"]
        summary["ambiguity_metrics"] = strict["ambiguity_metrics"]
        if report.get("evaluation_policy") == "ambrosia-strict-v1" and abs(report["accuracy"] - strict["accuracy"]) > 1e-8:
            raise ValueError("Strict Ambrosia aggregate disagrees with saved responses")
    files = {p.relative_to(root).as_posix(): parent.sha(p) for p in
             (folder / "metadata.json", folder / "query_results.json", folder / "report.json", folder / "raw_responses.jsonl")}
    return observed, summary, files


def build_rows(parent_rows, outcomes):
    result, conditional = [], []
    for source in parent_rows:
        benchmark, key = source["benchmark"], source["question_id"]
        values = {case: copy.deepcopy(observations[benchmark][key]) for case, observations in outcomes.items()}
        # Parent controls must agree; inputs/groups/splits are copied, never repartitioned.
        for case, name in (("baseline", "baseline"), ("feat_35", "economical"),
                           ("sql_ambiguity", "hybrid"), ("hybrid_with_relationships", "relationships")):
            reference = source["outcomes"][name]
            if any(reference[field] != values[case][field] for field in ("observed", "correct", "response_sha256")):
                raise ValueError("Control observation differs from the frozen Phase 0.2 dataset")
        labels = single_feature_labels(values)
        row = {field: copy.deepcopy(source[field]) for field in
               ("id", "benchmark", "task", "question_id", "db_id", "group_id", "split", "family_keys", "inputs", "analysis")}
        row.update(observations=values, feature_labels=labels, multilabel=target_vectors(labels))
        result.append(row)
        for pair in CONDITIONAL_PAIRS:
            conditional.append({
                "id": source["id"], "split": source["split"], "pair": pair["name"],
                "feature": f"FEAT_{pair['feature']}", "context_features": pair["context"],
                "treatment": pair["treatment"], "control": pair["control"],
                "label": benefit_label(values[pair["treatment"]], values[pair["control"]]),
                "merged_into_independent_targets": False,
            })
    return result, conditional


def summarise(rows, conditional):
    result = {"records": len(rows), "databases": len({(r["benchmark"], r["db_id"]) for r in rows}),
              "all_features": len(FEATURES), "target_heads": len(TARGET_FEATURES),
              "target_features": [f"FEAT_{n}" for n in TARGET_FEATURES],
              "independent_optimal_feature_masks_available": False,
              "conditional_records": len(conditional), "features": {}, "splits": {}, "conditional_pairs": {}}
    for number in FEATURES:
        name = f"FEAT_{number}"
        result["features"][name] = {}
        for benchmark in parent.DATASETS:
            labels = [r["feature_labels"][name] for r in rows if r["benchmark"] == benchmark]
            result["features"][name][benchmark] = {
                "rows": len(labels), "loss_observed": sum(v["loss_mask"] for v in labels),
                "preference_observed": sum(v["preference_mask"] for v in labels),
                "benefit_positive": sum(v["benefit_target"] == 1 for v in labels),
                "benefit_negative": sum(v["benefit_target"] == 0 for v in labels),
                "raw_outcome_classes": dict(Counter(v["outcome_class"] for v in labels)),
                "masked_reasons": dict(Counter(v["exclusion_reason"] for v in labels if not v["loss_mask"])),
            }
    for split in ("train", "dev", "test"):
        subset = [r for r in rows if r["split"] == split]
        result["splits"][split] = {
            "records": len(subset), "benchmarks": dict(Counter(r["benchmark"] for r in subset)),
            "observed_targets": sum(sum(r["multilabel"]["loss_mask"]) for r in subset),
            "positive_targets": sum(sum(v == 1 for v in r["multilabel"]["benefit_targets"]) for r in subset),
            "preference_targets": sum(sum(r["multilabel"]["preference_mask"]) for r in subset),
        }
    for pair in CONDITIONAL_PAIRS:
        subset = [r for r in conditional if r["pair"] == pair["name"]]
        result["conditional_pairs"][pair["name"]] = {
            "records": len(subset), "classes": dict(Counter(r["label"]["outcome_class"] for r in subset)),
            "preference_targets": sum(r["label"]["preference_mask"] for r in subset),
        }
    return result


def markdown_summary(summary):
    lines_ = ["# DeepSeek: weak multilabel benefit observations", "",
              f"{summary['records']} questions, {summary['databases']} DBs, {summary['target_heads']} eligible heads.", "",
              "Single-flag labels use all-false baseline. Utility = correct(feature) - correct(baseline).",
              "Benefit target=1 only for observed gain; ties=0 means no observed gain, not proven uselessness.",
              "Preference labels use only +1/-1 pairs. Invalid/missing/excluded labels stay null and masked.", "",
              "| Feature | Eligible | BIRD: + / - / ties | Ambrosia: + / - / ties | Loss masks true | Exclusion |",
              "| --- | --- | --- | --- | ---: | --- |"]
    for number in FEATURES:
        name = f"FEAT_{number}"
        data = summary["features"][name]
        cells = []
        for benchmark in parent.DATASETS:
            classes = data[benchmark]["raw_outcome_classes"]
            cells.append(f"{classes.get('helped', 0)} / {classes.get('harmed', 0)} / "
                         f"{classes.get('both_correct', 0) + classes.get('neither_correct', 0)}")
        lines_.append(f"| {name} | {number in TARGET_FEATURES} | {cells[0]} | {cells[1]} | "
                      f"{sum(v['loss_observed'] for v in data.values())} | {EXCLUSIONS.get(number, '')} |")
    lines_ += ["", "Raw + / - counts for excluded mechanisms are audit only; they are not training labels.", "",
               "| Split | Questions | Observed targets | Positive targets | Preference targets |",
               "| --- | ---: | ---: | ---: | ---: |"]
    for name, data in summary["splits"].items():
        lines_.append(f"| {name} | {data['records']} | {data['observed_targets']} | "
                      f"{data['positive_targets']} | {data['preference_targets']} |")
    lines_ += ["", "Conditional effects are exported separately; they do not overwrite independent targets.",
               "These are single-run observational labels, not causal estimates or optimal combination masks.",
               "Inputs: question/evidence/schema_stats only. Same frozen DB/family partitions as Phase 0.2.",
               "Test has already been viewed in Phase 1 and is retrospective. No independent final test.",
               "No per-question cost estimates are fabricated. BIRD and Ambrosia tasks remain separate."]
    return "\n".join(lines_) + "\n"


def examples(rows):
    return [{"id": row["id"], "inputs": row["inputs"], **row["multilabel"]} for row in rows]


def validate_rows(rows, conditional, parent_rows):
    reference = {row["id"]: row for row in parent_rows}
    if len(rows) != len(reference) or {r["id"] for r in rows} != set(reference):
        raise ValueError("Multilabel rows do not match the parent IDs")
    for row in rows:
        original = reference[row["id"]]
        if any(row[field] != original[field] for field in
               ("inputs", "split", "group_id", "db_id", "benchmark", "task", "question_id", "family_keys", "analysis")):
            raise ValueError("Inputs, task, or frozen DB/family assignment changed")
        if set(row["inputs"]) != set(parent.MODEL_INPUT_FIELDS):
            raise ValueError("Unsafe model inputs")
        if row["feature_labels"] != single_feature_labels(row["observations"]):
            raise ValueError("Incorrect feature utility or masks")
        if row["multilabel"] != target_vectors(row["feature_labels"]):
            raise ValueError("Multilabel vectors differ from the audited targets")
    by_id = {r["id"]: r for r in rows}
    expected_keys = {(r["id"], p["name"]) for r in rows for p in CONDITIONAL_PAIRS}
    if len(conditional) != len(expected_keys) or {(r["id"], r["pair"]) for r in conditional} != expected_keys:
        raise ValueError("Conditional label coverage differs")
    pairs = {p["name"]: p for p in CONDITIONAL_PAIRS}
    for row in conditional:
        pair = pairs[row["pair"]]
        source = by_id[row["id"]]
        if (row["split"] != source["split"] or row["context_features"] != pair["context"] or
                row["feature"] != f"FEAT_{pair['feature']}" or row["treatment"] != pair["treatment"] or
                row["control"] != pair["control"] or row["merged_into_independent_targets"]):
            raise ValueError("Invalid conditional label context")
        values = source["observations"]
        if row["label"] != benefit_label(values[pair["treatment"]], values[pair["control"]]):
            raise ValueError("Conditional utility disagrees with source observations")


def verify(output, root=ROOT, dataset=None):
    manifest = parent.read(output / "manifest.json")
    if manifest.get("format_version") != FORMAT:
        raise ValueError("Unknown multilabel format")
    if set(manifest["output_files_sha256"]) != set(OUTPUTS):
        raise ValueError("Incomplete multilabel bundle")
    for path, expected in manifest["output_files_sha256"].items():
        if parent.sha(output / path) != expected:
            raise ValueError(f"Output artifact hash changed: {path}")
    for path, expected in manifest["source_files_sha256"].items():
        if parent.sha(root / path) != expected:
            raise ValueError(f"Source artifact hash changed: {path}")
    dataset = dataset or root / manifest["parent_dataset"]
    parent.verify_bundle(dataset, root)
    if manifest["target_features"] != [f"FEAT_{n}" for n in TARGET_FEATURES] or manifest["feature_catalog"] != feature_catalog():
        raise ValueError("Feature policy differs from the frozen registry")
    rows, conditional = lines(output / "records.jsonl"), lines(output / "conditional_labels.jsonl")
    validate_rows(rows, conditional, lines(dataset / "records.jsonl"))
    if lines(output / "examples.jsonl") != examples(rows):
        raise ValueError("Training export differs from the audited records")
    for file in ("splits.json", "cv_folds.json", "ambrosia_family_holdout.json"):
        if (output / file).read_bytes() != (dataset / file).read_bytes():
            raise ValueError("Original grouped split/folds were changed")
    summary = summarise(rows, conditional)
    if parent.read(output / "summary.json") != summary or (output / "summary.md").read_text(encoding="utf-8") != markdown_summary(summary):
        raise ValueError("Summary differs from the audited observations")
    return manifest


def build(output, root=ROOT, dataset=None):
    dataset = dataset or root / parent.DEFAULT_OUTPUT
    parent_manifest = parent.verify_bundle(dataset, root)
    if output.exists() and any(output.iterdir()):
        verify(output, root, dataset)
        return parent.read(output / "summary.json")
    manifests, accepted, files = parent.load_experiments(root)
    files.update(parent_manifest["source_files_sha256"])
    for path in [dataset / "manifest.json", *(dataset / file for file in parent.OUTPUT_FILES)]:
        files[path.relative_to(root).as_posix()] = parent.sha(path)
    # Verify the source assumptions used by the eligibility policy, not current runner revisions.
    historical = manifests["ablation"]["protocol"]["files_sha256"]
    for path, expected in historical.items():
        if path.startswith(("src/", "benchmarks/")):
            if parent.sha(root / path) != expected:
                raise ValueError("Recorded feature implementation/evaluator changed; review eligibility policy")
            files[path] = expected
    questions = {b: parent.load_questions(root / file) for b, file in parent.DATASETS.items()}
    specs = {"baseline": {"experiment": "ablation", "features": []},
             **{f"feat_{n}": {"experiment": "ablation", "features": [n]} for n in FEATURES}, **EXTRA_CASES}
    outcomes, registry = {}, {}
    for case, spec in specs.items():
        experiment = spec["experiment"]
        outcomes[case], registry[case] = {}, {"features": spec["features"], "benchmarks": {}}
        for benchmark, items in questions.items():
            folder = root / parent.EXPERIMENTS[experiment] / case / ("bird" if benchmark == "bird" else "ambrosia_227")
            values, summary, sources = read_run(root, folder, benchmark, items, spec["features"], manifests[experiment], accepted[experiment])
            outcomes[case][benchmark] = values
            registry[case]["benchmarks"][benchmark] = summary
            files.update(sources)
    # Check that conditional pairs really differ by exactly one feature.
    for pair in CONDITIONAL_PAIRS:
        if (registry[pair["control"]]["features"] != pair["context"] or
                set(registry[pair["treatment"]]["features"]) != set(pair["context"]) | {pair["feature"]}):
            raise ValueError("Conditional comparison does not isolate its stated feature")
    records, conditional = build_rows(lines(dataset / "records.jsonl"), outcomes)
    validate_rows(records, conditional, lines(dataset / "records.jsonl"))
    stats = summarise(records, conditional)
    output.mkdir(parents=True, exist_ok=True)
    write_lines(output / "records.jsonl", records)
    write_lines(output / "examples.jsonl", examples(records))
    write_lines(output / "conditional_labels.jsonl", conditional)
    for file in ("splits.json", "cv_folds.json", "ambrosia_family_holdout.json"):
        (output / file).write_bytes((dataset / file).read_bytes())
    write_json(output / "summary.json", stats)
    (output / "summary.md").write_text(markdown_summary(stats), encoding="utf-8", newline="\n")
    manifest = {
        "format_version": FORMAT, "parent_dataset": dataset.relative_to(root).as_posix(),
        "model": parent_manifest["model"], "source_runs": registry,
        "feature_catalog": feature_catalog(), "target_features": [f"FEAT_{n}" for n in TARGET_FEATURES],
        "single_flag_background": feature_flags("baseline"), "conditional_pairs": list(CONDITIONAL_PAIRS),
        "model_input_fields": list(parent.MODEL_INPUT_FIELDS),
        "label_policy": {
            "utility": "single recorded correctness difference against all-false baseline; not causal attribution",
            "benefit_target": "1 if utility>0, otherwise 0 for observed eligible pairs; ties mean no observed gain",
            "preference_target": "1 if utility=+1, 0 if utility=-1; ties are null and preference_mask=false",
            "excluded_or_missing": "null target and false loss mask; raw observations retained for audit",
            "invalid_output": "blank prediction is unobserved/invalid, never an Ambrosia true negative",
            "error": "recorded generation error and model SQL execution error are observed failures",
            "context": "conditional labels kept separate; never merged into independent single-flag targets",
            "cost": "exact per-question costs unavailable; no invented cost labels",
        },
        "limitations": {
            "optimal_feature_combinations_available": False, "replicated_trials_available": False,
            "test_already_viewed_in_phase1": True, "independent_final_test_available": False,
            "feature_6_exclusion": "conservative global exclusion after previously audited incomplete BIRD schema preparation",
            "mechanism_activity": "eligibility checks known gating only; per-question execution not fully observed",
            "cross_model_transfer_validated": False,
        },
        "source_files_sha256": dict(sorted(files.items())),
        "builder_files_sha256": {file: parent.sha(root / file) for file in BUILDER_FILES},
        "output_files_sha256": {file: parent.sha(output / file) for file in OUTPUTS},
    }
    write_json(output / "manifest.json", manifest)
    verify(output, root, dataset)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--parent-dataset", default=parent.DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output, dataset = (ROOT / args.out).resolve(), (ROOT / args.parent_dataset).resolve()
    if not output.is_relative_to(ROOT / "data") or output == ROOT / "data" or output == dataset or output.is_relative_to(dataset):
        parser.error("Use a new dedicated folder inside data/, outside the parent dataset")
    if not dataset.is_relative_to(ROOT / "data"):
        parser.error("Parent dataset must be inside data/")
    if args.verify_only:
        verify(output, dataset=dataset)
        print(f"VERIFIED: utility, masks, provenance, original DB/family split: {output}")
    else:
        summary = build(output, dataset=dataset)
        print(f"READY: {summary['records']} records, {summary['target_heads']} eligible heads, "
              f"{summary['conditional_records']} separate conditional records: {output}")
        print("Target features:", ", ".join(summary["target_features"]))
        for split, stats in summary["splits"].items():
            print(split, stats)


if __name__ == "__main__":
    main()
