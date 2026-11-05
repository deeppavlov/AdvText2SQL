import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from experiment_structured_classifier import (  # noqa: E402
    PROFILES, profile_correct, profile_scores, replay, set_quality_loss, training_cv,
)


def row(key, benchmark="bird", group=None):
    return {"id": key, "split": "train", "benchmark": benchmark, "group_id": group or key,
            "inputs": {"question": "List names", "evidence": "", "schema_stats": {
                name: 2 for name in ("table_count", "column_count", "text_column_count", "numeric_column_count",
                                     "temporal_column_count", "boolean_column_count")}},
            "analysis": {"gold_ambiguous": True if benchmark == "ambrosia" else None},
            "observations": {profile["case"]: {
                "observed": True, "correct": index > 0, "generation_error": False,
                "predicted_ambiguous": index > 0} for index, profile in enumerate(PROFILES)}}


class StructuredClassifierTests(unittest.TestCase):
    def test_joint_profile_can_win_when_parent_individual_score_is_negative(self):
        scores = profile_scores(torch.tensor([[-1.0, 3.0]]))
        self.assertEqual(scores.tolist(), [[0.0, -1.0, 2.0]])
        self.assertEqual(int(scores.argmax()), 2)

    def test_loss_accepts_multiple_correct_profiles_without_false_binary_negatives(self):
        scores = torch.zeros((1, 3), requires_grad=True)
        loss = set_quality_loss(scores, torch.tensor([[False, True, True]]), complexity_weight=0)
        self.assertAlmostEqual(float(loss.detach()), np.log(1.5), places=6)
        loss.backward()
        self.assertGreater(float(scores.grad[0, 0]), 0)
        self.assertLess(float(scores.grad[0, 1]), 0)
        self.assertLess(float(scores.grad[0, 2]), 0)

    def test_all_correct_and_all_wrong_rows_do_not_invent_quality_preferences(self):
        for target in ((True, True, True), (False, False, False)):
            scores = torch.zeros((1, 3), requires_grad=True)
            loss = set_quality_loss(scores, torch.tensor([target]), complexity_weight=0)
            self.assertEqual(float(loss.detach()), 0)
            loss.backward()
            self.assertTrue((scores.grad == 0).all())

    def test_complexity_prior_is_explicit_and_prefers_economical_on_ties(self):
        expensive = set_quality_loss(torch.tensor([[0.0, 0.0, 5.0]]), torch.ones((1, 3), dtype=torch.bool))
        cheap = set_quality_loss(torch.tensor([[5.0, 0.0, 0.0]]), torch.ones((1, 3), dtype=torch.bool))
        self.assertLess(float(cheap), float(expensive))

    def test_missing_observation_is_not_a_false_quality_label(self):
        example = row("a")
        example["observations"]["feat_35"].update(observed=False, correct=None)
        with self.assertRaises(ValueError):
            profile_correct([example])

    def test_replay_uses_exact_observed_profile_and_strict_ambiguity_metrics(self):
        rows = [row("a"), row("b", "ambrosia")]
        report = replay(rows, [1, 2])
        self.assertEqual(report["benchmarks"]["bird"]["correct"], 1)
        self.assertEqual(report["benchmarks"]["ambrosia"]["ambiguity_metrics"]["f1"], 1)
        self.assertEqual(report["macro_accuracy"], 100)

    def test_cv_rejects_dev_test_rows_and_group_leakage(self):
        rows = [row("a"), row("b", "ambrosia")]
        folds = [{"fold": 0, "train_ids": ["a"], "validation_ids": ["b"]},
                 {"fold": 1, "train_ids": ["b"], "validation_ids": ["a"]}]
        features = np.zeros((2, 2), dtype=np.float32)
        changed = copy.deepcopy(rows)
        changed[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "training rows"):
            training_cv(changed, features, folds, epochs=1)
        changed = copy.deepcopy(rows)
        changed[0]["group_id"] = changed[1]["group_id"]
        with self.assertRaisesRegex(ValueError, "groups"):
            training_cv(changed, features, folds, epochs=1)

    def test_cv_produces_complete_oof_and_stays_inside_frozen_candidate_catalog(self):
        torch.set_num_threads(1)
        rows = [row("a"), row("b", "ambrosia")]
        folds = [{"fold": 0, "train_ids": ["a"], "validation_ids": ["b"]},
                 {"fold": 1, "train_ids": ["b"], "validation_ids": ["a"]}]
        selected, reports = training_cv(rows, np.ones((2, 2), dtype=np.float32), folds, epochs=1)
        self.assertEqual(len(reports), 4)
        self.assertIn(selected, [r["candidate"] for r in reports])
        self.assertTrue(all(sum(v["questions"] for v in report["oof"]["benchmarks"].values()) == 2 for report in reports))


if __name__ == "__main__":
    unittest.main()
