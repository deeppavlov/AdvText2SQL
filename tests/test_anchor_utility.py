"""Relative labels, loss masks, schema grounding and train-only screening boundaries."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import experiment_anchor_utility as experiment  # noqa: E402
from adv_text2sql.mcp_servers.text2sql_tool.src.configuration_utility import (  # noqa: E402
    GROUND_FIELDS, UTILITY_OUTPUTS, choose_utility, grounding_features, utility_loss, utility_targets, words,
)


def example(values, task="bird", split="train"):
    return {"id": f"{task}:1", "benchmark": task, "split": split, "group_id": "g",
            "configurations": list(experiment.base.TASK_CONFIGURATIONS[task]),
            "quality_targets": values, "loss_mask": [v is not None for v in values]}


def inputs():
    return {"question": "How many schools opened after 2000/1/1?", "evidence": "Charter Funding Type",
            "schema_stats": {"table_count": 2, "column_count": 3, "text_column_count": 1,
                             "numeric_column_count": 1, "temporal_column_count": 1, "boolean_column_count": 0}}


class Tests(unittest.TestCase):
    def test_signed_configuration_utility_and_task_mask(self):
        y, mask, _ = utility_targets([example([0, 1, 0, 1, 1])])
        self.assertEqual(y[0, 0].item(), -1)
        self.assertEqual(y[0, 1].item(), 0)
        self.assertEqual(int(mask.sum()), 4)
        self.assertNotIn(('bird', 'compact'), UTILITY_OUTPUTS)
        y, _, _ = utility_targets([example([1, 0, 0, 0, 0])])
        self.assertEqual(y[0, 0].item(), 1)

    def test_unknown_anchor_cannot_create_gain_labels(self):
        _, mask, _ = utility_targets([example([1, 0, 1, 0, None])])
        self.assertFalse(mask.any())

    def test_masked_unknowns_and_other_task_have_zero_gradient(self):
        y, mask, tasks = utility_targets([example([1, None, 0, 1, 0])])
        prediction = torch.zeros_like(y, requires_grad=True)
        loss = utility_loss(prediction, y, mask, tasks)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue((prediction.grad[~mask] == 0).all())
        self.assertLess(prediction.grad[0, 0].item(), 0)

    def test_ties_are_zero_not_positive_benefit(self):
        for value in (0, 1):
            y, mask, _ = utility_targets([example([value] * 5)])
            self.assertTrue((y[mask] == 0).all())

    def test_no_switch_without_positive_margin_excess(self):
        values = {name: .1 for task, name in UTILITY_OUTPUTS if task == 'bird'}
        self.assertEqual(choose_utility(values, 'bird'), 'compact')
        values['baseline'] = .2
        self.assertEqual(choose_utility(values, 'bird'), 'baseline')
        with self.assertRaises(ValueError):
            choose_utility(values, 'ambrosia')

    def test_grounding_uses_names_types_and_evidence_without_ids(self):
        schema = [['schools', 'OpenDate', 'DATE'], ['frpm', 'Charter Funding Type', 'TEXT'],
                  ['schools', 'FundingType', 'TEXT']]
        features = dict(zip(GROUND_FIELDS, grounding_features(inputs(), schema)))
        self.assertEqual(features['count_cue'], 1)
        self.assertEqual(features['has_date'], 1)
        self.assertGreater(features['evidence_column_word_coverage'], 0)
        self.assertGreater(features['table_word_coverage'], 0)
        self.assertIn('fund', words('FundingType'))
        bad = dict(inputs(), SQL='SELECT 1')
        with self.assertRaises(ValueError):
            grounding_features(bad, schema)

    def test_extra_scaler_fits_training_only(self):
        fit = np.zeros((2, len(GROUND_FIELDS)), dtype=np.float32)
        valid = np.full_like(fit, 100)
        a, b, state = experiment.scale_grounding(fit, valid)
        self.assertTrue((a == 0).all())
        self.assertTrue((b > 0).all())
        self.assertEqual(state['mean'], [0] * len(GROUND_FIELDS))

    def test_outer_dev_cannot_enter_feature_build_fit_or_replay(self):
        dev = [example([1] * 5, split='dev')]
        for action in (
            lambda: experiment.grounded_matrix(dev, {}, {}),
            lambda: experiment.fit_utility(np.zeros((1, 2), dtype=np.float32), dev),
            lambda: experiment.summarize(dev, ['compact'], {}, {}),
        ):
            with self.assertRaises(ValueError):
                action()

    def test_gate_requires_distributed_gain_not_one_fold(self):
        anchor, candidate = {'correct': 10}, {'correct': 12}
        self.assertFalse(experiment.eligibility(anchor, candidate, [2, 0, 0, 0], 'bird')['eligible_for_next_validation'])
        self.assertTrue(experiment.eligibility(anchor, candidate, [1, 1, 0, 0], 'bird')['eligible_for_next_validation'])
        self.assertFalse(experiment.eligibility(anchor, {'correct': 11}, [1, 0, 0, 0], 'bird')['eligible_for_next_validation'])

    def test_ambiguity_accuracy_gain_cannot_hide_recall_loss(self):
        a = {'correct': 10, 'ambiguity_metrics': {'f1': .6, 'recall': .5}}
        b = {'correct': 12, 'ambiguity_metrics': {'f1': .7, 'recall': .4}}
        gate = experiment.eligibility(a, b, [1, 1, 0, 0], 'ambrosia')
        self.assertFalse(gate['eligible_for_next_validation'])
        self.assertIn('ambiguity_recall_degraded', gate['reasons'])


if __name__ == '__main__':
    unittest.main()
