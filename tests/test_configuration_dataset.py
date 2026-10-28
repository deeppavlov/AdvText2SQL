import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_configuration_dataset as builder
from configuration_splits import (
    audit_partitions, family_keys, family_stress_folds, make_groups,
    normalize_family, partition, training_folds,
)


def outcome(correct, observed=True):
    return {"correct": correct, "observed": observed}


def grouped_records():
    rows = []
    names = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet")
    for benchmark in ("bird", "ambrosia"):
        for index, name in enumerate(names):
            for version in range(3):
                item = {"question": f"Inspect {benchmark} {name} version {version}",
                        "ambig_question": f"Inspect {benchmark} {name}"}
                rows.append({"id": f"{benchmark}:{name}:{version}", "benchmark": benchmark, "db_id": name,
                             "inputs": {"question": item["question"]}, "family_keys": family_keys(item, benchmark),
                             "analysis": {"construction_family": ("attachment", "scope", "vague")[index % 3]},
                             "labels": {"utility": index % 3 - 1}})
    return rows


class ConfigurationDatasetTests(unittest.TestCase):
    def test_correctness_is_not_utility_and_ties_are_not_quality_labels(self):
        for cheap, full, delta, expected_class in [
            (False, True, 1, "hybrid_only_correct"), (True, False, -1, "economical_only_correct"),
            (True, True, 0, "both_correct"), (False, False, 0, "neither_correct"),
        ]:
            labels = builder.selection_labels({"economical": outcome(cheap), "hybrid": outcome(full)})
            self.assertEqual(labels["utility_hybrid_vs_economical"], delta)
            self.assertEqual(labels["outcome_class"], expected_class)
            self.assertEqual(labels["quality_preference_observed"], bool(delta))
            if delta == 0:
                self.assertIsNone(labels["quality_preference"])
                self.assertEqual(labels["cost_aware_preference"], "economical")
        self.assertEqual(builder.utility(outcome(True), outcome(True)), 0)
        self.assertIsNone(builder.utility(outcome(None, False), outcome(False)))
        missing = builder.selection_labels({"economical": outcome(True), "hybrid": outcome(None, False)})
        self.assertFalse(missing["pair_observed"])
        self.assertIsNone(missing["cost_aware_preference"])

    def test_missing_predictions_are_null_and_errors_are_observed_failures(self):
        items = [{"question_id": 1, "SQL": "SELECT 1"}, {"question_id": 2, "SQL": "ambiguous"},
                 {"question_id": 3, "SQL": "SELECT 3"}]
        rows = builder.observations(items, {"1": "error", "2": "ambiguous"}, {}, "ambrosia")
        self.assertTrue(rows["1"]["observed"])
        self.assertFalse(rows["1"]["correct"])
        self.assertTrue(rows["1"]["generation_error"])
        self.assertTrue(rows["2"]["correct"])
        self.assertFalse(rows["3"]["observed"])
        self.assertIsNone(rows["3"]["correct"])
        self.assertFalse(rows["1"]["question_cost_observed"])
        self.assertIsNone(rows["1"]["prompt_tokens"])

    def test_bird_report_scores_cannot_be_strings_or_disagree_with_predictions(self):
        items = [{"question_id": 1, "SQL": "SELECT 1"}]
        report = {"results": [{"question_id": 1, "score": "False", "predicted_sql": "SELECT 2"}]}
        with self.assertRaisesRegex(ValueError, "invalid scores"):
            builder.observations(items, {"1": "SELECT 2"}, report, "bird")
        report["results"][0]["score"] = False
        with self.assertRaisesRegex(ValueError, "differs from"):
            builder.observations(items, {"1": "SELECT 3"}, report, "bird")
        report["results"][0].update(score=True, predicted_sql="error")
        with self.assertRaisesRegex(ValueError, "Generation error"):
            builder.observations(items, {"1": "error"}, report, "bird")

    def test_unknown_and_duplicate_ids_cannot_enter_the_dataset(self):
        items = [{"question_id": 1, "SQL": "SELECT 1"}]
        with self.assertRaises(ValueError):
            builder.observations(items, {"2": "SELECT 2"}, {}, "ambrosia")
        report = {"results": [{"question_id": 1, "score": False, "predicted_sql": "SELECT 2"}] * 2}
        with self.assertRaises(ValueError):
            builder.observations(items, {"1": "SELECT 2"}, report, "bird")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queries.json"
            path.write_text(json.dumps([{"question_id": 1, "question": "x"}] * 2))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                builder.load_questions(path)

    def test_reference_execution_failure_is_not_a_model_failure_label(self):
        items = [{"question_id": 1, "SQL": "SELECT broken_gold"}]
        row = {"question_id": 1, "gold_sql": "SELECT broken_gold", "predicted_sql": "SELECT 1", "score": False,
               "error": "SQL error\n[SQL: SELECT broken_gold]\n(Background on this error: example)"}
        observed = builder.observations(items, {"1": "SELECT 1"}, {"results": [row]}, "bird")["1"]
        self.assertFalse(observed["observed"])
        self.assertIsNone(observed["correct"])
        self.assertEqual(observed["evaluation_error_kind"], "reference")
        row.update(error="SQL error\n[SQL: SELECT 1]\n(Background on this error: example)")
        observed = builder.observations(items, {"1": "SELECT 1"}, {"results": [row]}, "bird")["1"]
        self.assertTrue(observed["observed"])
        self.assertFalse(observed["correct"])
        self.assertEqual(observed["evaluation_error_kind"], "prediction")

    def test_family_grouping_links_variants_across_databases_and_benchmarks(self):
        records = grouped_records()
        a, b = records[0], records[-1]
        b["family_keys"] = list(a["family_keys"])
        groups_by_id, groups, edges = make_groups(records)
        self.assertEqual(groups_by_id[a["id"]], groups_by_id[b["id"]])
        self.assertEqual(groups_by_id[b["id"]], groups_by_id[records[-2]["id"]])
        self.assertLess(len(groups), 20)
        self.assertEqual(normalize_family('Show "Alice" ranked above 100.'),
                         normalize_family('show "Bob" ranked above 250!'))

    def test_near_duplicate_questions_cannot_cross_splits(self):
        records = grouped_records()
        records[0]["inputs"]["question"] = "Please list every eligible school and its city with enrollment above threshold"
        records[-1]["inputs"]["question"] = "Please list every eligible school and its town with enrollment above threshold"
        assignments, groups, edges = make_groups(records)
        self.assertIn([records[-1]["id"], records[0]["id"]], edges)
        self.assertEqual(assignments[records[0]["id"]], assignments[records[-1]["id"]])

    def test_partitions_are_reproducible_and_do_not_use_outcomes(self):
        records = grouped_records()
        assignments, groups, _ = make_groups(records)
        original = partition(records, groups, seed=42, attempts=8)
        changed = copy.deepcopy(list(reversed(records)))
        for row in changed:
            row["labels"] = {"utility": 999}
        new_assignments, new_groups, _ = make_groups(changed)
        self.assertEqual(assignments, new_assignments)
        self.assertEqual(original, partition(changed, new_groups, seed=42, attempts=8))
        self.assertEqual(audit_partitions(records, groups, original)["database_overlap"], 0)
        tampered = copy.deepcopy(original)
        a = next(s for s in tampered if len(tampered[s]) > 1)
        b = next(s for s in tampered if s != a)
        tampered[b].append(tampered[a].pop())
        with self.assertRaises(ValueError):
            audit_partitions(records, groups, tampered)

    def test_cv_uses_only_training_and_stress_folds_hold_out_database_families(self):
        records = grouped_records()
        _, groups, _ = make_groups(records)
        splits = partition(records, groups, attempts=8)
        folds = training_folds(records, groups, splits)
        stress = family_stress_folds(records)
        builder.audit_folds(records, groups, splits, folds, stress)
        contaminated = copy.deepcopy(folds)
        contaminated[0]["validation_ids"].append(splits["test"][0])
        with self.assertRaises(ValueError):
            builder.audit_folds(records, groups, splits, contaminated, stress)

    def test_schema_features_are_extracted_without_gold_or_database_access(self):
        items = [{"db_id": "one", "question": "Show the first property", "SQL": "hidden gold"}]
        message = "Schema:\ntable\tcolumn\ttype\nhouse\tid\tINTEGER\nhouse\tcity\tTEXT\n\nEnd"
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "responses.jsonl"
            raw.write_text(json.dumps({"messages": [{"content": message}, {"content": items[0]["question"]}],
                                       "prompt_tokens": 50, "completion_tokens": 10}) + "\n")
            schemas = builder.extract_schemas(items, raw)
            self.assertEqual(schemas["one"]["statistics"]["table_count"], 1)
            self.assertEqual(schemas["one"]["statistics"]["numeric_column_count"], 1)
            ambiguous = [*items, {"db_id": "two", "question": items[0]["question"]}]
            with self.assertRaisesRegex(ValueError, "Missing or inconsistent"):
                builder.extract_schemas(ambiguous, raw)

    def test_inputs_exclude_gold_labels_difficulty_db_identity_and_outcomes(self):
        items = {"ambrosia": [{"question_id": 1, "question": "Inspect alpha", "SQL": "ambiguous",
                                "db_id": "scope_alpha", "ambig_question": "Inspect alpha", "difficulty": "hard"}]}
        outcomes = {name: {"ambrosia": {"1": outcome(name == "hybrid")}} for name in builder.CONFIGURATIONS}
        schemas = {"ambrosia": {"scope_alpha": {"sha256": "schema-hash", "statistics": {"table_count": 2}}}}
        row = builder.build_records(items, outcomes, schemas, "")[0]
        self.assertEqual(set(row["inputs"]), set(builder.MODEL_INPUT_FIELDS))
        self.assertEqual(row["inputs"]["schema_stats"], {"table_count": 2})
        self.assertTrue(row["analysis"]["gold_ambiguous"])
        self.assertEqual(row["labels"]["utility_hybrid_vs_economical"], 1)

    def test_protocol_cohort_rejects_model_generator_or_data_changes(self):
        original = {"model": "same", "files_sha256": {"src/generator.py": "one", "data/queries.json": "same"}}
        for field in ("model", "src/generator.py", "data/queries.json"):
            changed = copy.deepcopy(original)
            if field == "model":
                changed[field] = "different"
            else:
                changed["files_sha256"][field] = "different"
            with self.assertRaises(ValueError):
                builder.validate_protocols({"first": {"protocol": original}, "second": {"protocol": changed}})

    def test_quoted_prompt_examples_are_not_hidden_by_literal_normalization(self):
        question = "List all banquet halls and conference rooms with a 200 person capacity."
        items = {"ambrosia": [{"question_id": 1, "question": question, "SQL": "ambiguous",
                                "db_id": "attachment_example", "ambig_question": question}]}
        outcomes = {name: {"ambrosia": {"1": outcome(True)}} for name in builder.CONFIGURATIONS}
        schemas = {"ambrosia": {"attachment_example": {"sha256": "fixed", "statistics": {"table_count": 1}}}}
        row = builder.build_records(items, outcomes, schemas, '- "' + question + '"')[0]
        self.assertTrue(row["analysis"]["question_example_overlap"])
        self.assertTrue(row["analysis"]["family_example_overlap"])

    def test_output_artifact_tampering_fails_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for filename in builder.OUTPUT_FILES:
                (output / filename).write_text("original")
            builder.write_json(output / "manifest.json", {
                "format_version": builder.FORMAT,
                "output_files_sha256": {name: builder.sha(output / name) for name in builder.OUTPUT_FILES},
                "source_files_sha256": {},
            })
            (output / "records.jsonl").write_text("modified")
            with self.assertRaisesRegex(ValueError, "artifact hash changed"):
                builder.verify_bundle(output)


if __name__ == "__main__":
    unittest.main()
