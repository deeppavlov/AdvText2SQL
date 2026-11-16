"""Readiness checks use synthetic fixtures, no providers or database access."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_classifier_refresh import derive_cohort, local_path, train_seed_coverage


def fixture():
    records, full = [], {"bird": []}
    partitions = {"train": ["bird:1", "bird:4"], "dev": ["bird:2"], "test": ["bird:3"]}
    for n, split in ((1, "train"), (2, "dev"), (3, "test"), (4, "train")):
        question = {"question_id": n, "db_id": f"db{n}", "question": f"Question {n}", "evidence": ""}
        records.append({"id": f"bird:{n}", "benchmark": "bird", "db_id": f"db{n}",
                        "group_id": f"group{n}", "family_keys": [f"family{n}"], "split": split,
                        "inputs": {"question": question["question"], "evidence": "", "schema_stats": {}}})
        if n != 4:
            full["bird"].append(question)
    return records, partitions, full


class CohortTests(unittest.TestCase):
    def test_curated_subset_preserves_original_splits(self):
        current, excluded = derive_cohort(*fixture())
        self.assertEqual(excluded, ["bird:4"])
        self.assertEqual({k: v["split"] for k, v in current.items()},
                         {"bird:1": "train", "bird:2": "dev", "bird:3": "test"})

    def test_rejects_question_identity_change(self):
        records, partitions, full = fixture()
        full["bird"][0]["question"] = "Another question"
        with self.assertRaisesRegex(ValueError, "identity"):
            derive_cohort(records, partitions, full)

    def test_rejects_overlap_and_missing_ids(self):
        records, partitions, full = fixture()
        partitions["dev"].append("bird:1")
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            derive_cohort(records, partitions, full)
        records, partitions, full = fixture()
        partitions["train"].remove("bird:4")
        with self.assertRaisesRegex(ValueError, "cover"):
            derive_cohort(records, partitions, full)

    def test_rejects_db_family_and_component_leakage(self):
        records, partitions, full = fixture()
        for field in ("family_keys", "group_id", "db_id"):
            changed, items = copy.deepcopy(records), copy.deepcopy(full)
            changed[1][field] = changed[0][field]
            if field == "db_id":
                items["bird"][1][field] = changed[0][field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "leakage"):
                derive_cohort(changed, partitions, items)

    def test_rejects_duplicate_current_ids(self):
        records, partitions, full = fixture()
        full["bird"].append(full["bird"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate current"):
            derive_cohort(records, partitions, full)


class CoverageTests(unittest.TestCase):
    def test_only_train_selected_arm_is_read(self):
        current, _ = derive_cohort(*fixture())

        class Forbidden(dict):
            def __getitem__(self, key):
                raise AssertionError("Holdout/other-arm outcomes must not be inspected")

            def get(self, *args):
                raise AssertionError("Holdout/other-arm outcomes must not be inspected")

        evaluations = {"confirm:chosen:hybrid:False:bird:1": {"question_id": "1", "db_id": "db1", "score": 0, "error": None},
                       "confirm:chosen:hybrid:False:bird:2": Forbidden(),
                       "confirm:chosen:hybrid:False:bird:3": Forbidden(),
                       "confirm:other:hybrid:False:bird:1": Forbidden()}
        self.assertEqual(train_seed_coverage(evaluations, "chosen", "bird", current), ["bird:1"])

    def test_error_remains_unaudited_and_nonbinary_score_rejected(self):
        current, _ = derive_cohort(*fixture())
        key = "confirm:chosen:hybrid:False:bird:1"
        evaluations = {key: {"question_id": "1", "db_id": "db1", "score": 0, "error": "unclassified"}}
        self.assertEqual(train_seed_coverage(evaluations, "chosen", "bird", current), [])
        evaluations[key]["score"] = 0.5
        with self.assertRaisesRegex(ValueError, "binary"):
            train_seed_coverage(evaluations, "chosen", "bird", current)

    def test_output_paths_stay_inside_project(self):
        root = Path(__file__).resolve().parents[1]
        self.assertTrue(local_path(root, r"research_results\audit").is_relative_to(root))
        with self.assertRaises(ValueError):
            local_path(root, "../outside")


if __name__ == "__main__":
    unittest.main()
