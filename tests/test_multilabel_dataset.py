import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_multilabel_dataset as builder
from multilabel_targets import (
    CONDITIONAL_PAIRS, FEATURES, TARGET_FEATURES, benefit_label, feature_catalog,
    single_feature_labels, target_vectors,
)


def outcome(correct, observed=True):
    return {"observed": observed, "correct": correct, "response_sha256": str(correct),
            "generation_error": False, "execution_error": False}


def fixture():
    cases = ["baseline", *(f"feat_{n}" for n in FEATURES), *builder.EXTRA_CASES]
    values = {case: outcome(False) for case in cases}
    values["feat_35"] = outcome(True)
    values["sql_rules"] = outcome(False)
    values["sql_ambiguity"] = outcome(True)
    row = {"id": "bird:1", "question_id": "1", "benchmark": "bird", "task": "sql_execution",
           "db_id": "one", "group_id": "group:one", "split": "train", "family_keys": ["family:one"],
           "inputs": {"question": "List users", "evidence": "", "schema_stats": {"table_count": 1}},
           "analysis": {"difficulty": "simple", "gold_ambiguous": None},
           "outcomes": {name: copy.deepcopy(values[case]) for case, name in (
               ("baseline", "baseline"), ("feat_35", "economical"), ("sql_ambiguity", "hybrid"),
               ("hybrid_with_relationships", "relationships"))}}
    return [row], {case: {"bird": {"1": value}} for case, value in values.items()}


class MultilabelDatasetTests(unittest.TestCase):
    def test_positive_requires_gain_not_just_correctness(self):
        for treated, control, utility, target in ((True, False, 1, 1), (False, True, -1, 0),
                                                   (True, True, 0, 0), (False, False, 0, 0)):
            value = benefit_label(outcome(treated), outcome(control))
            self.assertEqual(value["utility"], utility)
            self.assertEqual(value["benefit_target"], target)
            self.assertTrue(value["loss_mask"])
            self.assertEqual(value["preference_mask"], utility != 0)
            if utility == 0:
                self.assertIsNone(value["preference_target"])

    def test_missing_and_excluded_targets_are_null_even_with_positive_raw_utility(self):
        for first, second in ((outcome(None, False), outcome(True)), (outcome(False), outcome(None, False))):
            value = benefit_label(first, second)
            self.assertIsNone(value["utility"])
            self.assertIsNone(value["benefit_target"])
            self.assertFalse(value["loss_mask"])
        excluded = benefit_label(outcome(True), outcome(False), "dependency_not_enabled")
        self.assertEqual(excluded["utility"], 1)
        self.assertIsNone(excluded["benefit_target"])
        self.assertFalse(excluded["preference_mask"])
        with self.assertRaises(ValueError):
            benefit_label(outcome("true"), outcome(True))
        with self.assertRaises(ValueError):
            benefit_label(outcome(False, False), outcome(True))

    def test_exactly_16_heads_and_33_audit_labels_with_reasons(self):
        observations = {"baseline": outcome(False), **{f"feat_{n}": outcome(True) for n in FEATURES}}
        labels = single_feature_labels(observations)
        self.assertEqual(len(labels), 33)
        self.assertEqual(len(TARGET_FEATURES), 16)
        for number in (3, 4, 7, 27, 33, 34, 36, 5, 6, 26, 8, 9, 10, 11, 17, 18, 19):
            self.assertFalse(labels[f"FEAT_{number}"]["loss_mask"])
            self.assertIsNotNone(labels[f"FEAT_{number}"]["exclusion_reason"])
        self.assertEqual(feature_catalog()["FEAT_34"]["required_features"], [32, 33])
        vectors = target_vectors(labels)
        self.assertEqual(vectors["benefit_targets"], [1] * 16)
        observations["feat_25"] = outcome(None, False)
        labels = single_feature_labels(observations)
        index = TARGET_FEATURES.index(25)
        self.assertIsNone(target_vectors(labels)["benefit_targets"][index])
        self.assertFalse(target_vectors(labels)["loss_mask"][index])

    def test_generation_failure_is_negative_when_baseline_succeeds(self):
        value = outcome(False)
        value["generation_error"] = True
        label = benefit_label(value, outcome(True))
        self.assertEqual(label["utility"], -1)
        self.assertEqual(label["benefit_target"], 0)
        self.assertTrue(label["loss_mask"])

    def test_blank_saved_output_is_masked_not_an_ambrosia_true_negative(self):
        items = [{"question_id": 1, "SQL": "SELECT 1"}, {"question_id": 2, "SQL": "ambiguous"},
                 {"question_id": 3, "SQL": "SELECT 3"}]
        rows = builder.recorded_observations(items, {"1": " ", "2": "error"}, {}, "ambrosia")
        self.assertTrue(rows["1"]["invalid_prediction"])
        self.assertFalse(rows["1"]["observed"])
        self.assertIsNone(rows["1"]["correct"])
        self.assertTrue(rows["2"]["observed"])
        self.assertFalse(rows["2"]["correct"])
        self.assertEqual(rows["3"]["observation_issue"], "prediction_missing")
        with self.assertRaises(ValueError):
            builder.recorded_observations(items, {"1": None}, {}, "ambrosia")

    def test_bird_reference_failure_is_not_a_negative_label(self):
        items = [{"question_id": 1, "SQL": "SELECT reference"}]
        report = {"results": [{"question_id": 1, "score": False, "predicted_sql": "SELECT model",
                               "gold_sql": "SELECT reference", "error": "error [SQL: SELECT reference]"}]}
        value = builder.recorded_observations(items, {"1": "SELECT model"}, report, "bird")["1"]
        self.assertFalse(value["observed"])
        self.assertIsNone(value["correct"])

    def test_conditional_utility_is_kept_separate_from_single_flag_utility(self):
        original, outcomes = fixture()
        rows, conditional = builder.build_rows(original, outcomes)
        self.assertEqual(rows[0]["feature_labels"]["FEAT_32"]["utility"], 0)
        by_pair = {r["pair"]: r for r in conditional}
        self.assertEqual(by_pair["taxonomy_given_rules_tsv"]["label"]["utility"], 1)
        self.assertEqual(by_pair["sql_rules_given_tsv"]["label"]["utility"], -1)
        self.assertEqual(len(conditional), len(CONDITIONAL_PAIRS))
        self.assertTrue(all(r["merged_into_independent_targets"] is False for r in conditional))
        builder.validate_rows(rows, conditional, original)
        self.assertEqual(rows[0]["inputs"], original[0]["inputs"])
        self.assertEqual(rows[0]["group_id"], original[0]["group_id"])
        exported = builder.examples(rows)[0]
        self.assertEqual(set(exported), {"id", "inputs", "benefit_targets", "loss_mask", "preference_targets", "preference_mask"})
        self.assertEqual(set(exported["inputs"]), {"question", "evidence", "schema_stats"})

    def test_split_input_and_conditional_contamination_are_rejected(self):
        original, outcomes = fixture()
        rows, conditional = builder.build_rows(original, outcomes)
        for field, value in (("split", "test"), ("group_id", "other"),
                             ("inputs", {**rows[0]["inputs"], "gold": "leak"})):
            changed = copy.deepcopy(rows)
            changed[0][field] = value
            with self.assertRaises(ValueError):
                builder.validate_rows(changed, conditional, original)
        changed = copy.deepcopy(conditional)
        changed[0]["merged_into_independent_targets"] = True
        with self.assertRaises(ValueError):
            builder.validate_rows(rows, changed, original)
        with self.assertRaises(ValueError):
            builder.validate_rows(rows, conditional[:-1], original)

    def test_changed_parent_controls_are_rejected(self):
        original, outcomes = fixture()
        outcomes["baseline"]["bird"]["1"]["correct"] = True
        with self.assertRaisesRegex(ValueError, "Control observation"):
            builder.build_rows(original, outcomes)

    def test_unknown_ids_and_wrong_scored_sql_are_rejected(self):
        items = [{"question_id": 1, "SQL": "SELECT 1"}]
        with self.assertRaisesRegex(ValueError, "unknown"):
            builder.recorded_observations(items, {"unexpected": "SELECT 1"}, {}, "ambrosia")
        with self.assertRaisesRegex(ValueError, "differs"):
            builder.recorded_observations(items, {"1": "SELECT 2"}, {"results": [
                {"question_id": 1, "score": True, "predicted_sql": "SELECT 3"}]}, "bird")

    def test_source_metadata_must_preserve_model_flags_and_protocol(self):
        protocol = {"model": "same", "model_version": "version", "api_backend": "direct", "base_url": "test://api",
                    "provider": "direct", "request_options": {"thinking": "disabled"}}
        identity = {**protocol, "temperature": 0, "max_output_tokens": 4096, "reasoning_effort": "none", "max_retries": 5,
                    "use_evidence": True, "dataset_sha256": "data", "protocol_sha256": "accepted",
                    "feature_flags": {f"FEAT_{n}": n == 14 for n in range(1, 37)}}
        metadata = {"status": "completed", "configuration": identity}
        args = ({"protocol": protocol}, {"accepted"}, "bird", [14], "data")
        builder.validate_run_metadata(metadata, *args)
        for field, value in (("model", "other"), ("protocol_sha256", "other"), ("feature_flags", {}),
                             ("dataset_sha256", "other")):
            changed = copy.deepcopy(metadata)
            changed["configuration"][field] = value
            with self.assertRaises(ValueError):
                builder.validate_run_metadata(changed, *args)

    def test_manifest_detects_output_tampering_before_loading_training_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for name in builder.OUTPUTS:
                (path / name).write_text("original", encoding="utf-8")
            (path / "manifest.json").write_text(json.dumps({
                "format_version": builder.FORMAT,
                "output_files_sha256": {n: builder.parent.sha(path / n) for n in builder.OUTPUTS},
            }), encoding="utf-8")
            (path / "examples.jsonl").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Output artifact hash"):
                builder.verify(path)


if __name__ == "__main__":
    unittest.main()
