"""No model downloads or API calls: verify diagnostic boundaries and attribution."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_configuration_classifier_v2 as diagnosis  # noqa: E402


def sample(task, columns, split):
    return {"benchmark": task, "split": split, "inputs": {"schema_stats": {
        "table_count": 2, "column_count": columns, "text_column_count": 3,
        "numeric_column_count": 2, "temporal_column_count": 0, "boolean_column_count": 0}}}


class Tests(unittest.TestCase):
    def test_ablation_removes_only_requested_block_without_mutation(self):
        x = np.arange(16, dtype=np.float32).reshape(2, 8)
        saved = x.copy()
        text = diagnosis.feature_view(x, "text_only", 2)
        schema = diagnosis.feature_view(x, "schema_only", 2)
        self.assertTrue(np.array_equal(x, saved))
        self.assertTrue(np.array_equal(text[:, :2], saved[:, :2]))
        self.assertTrue((text[:, 2:] == 0).all())
        self.assertTrue((schema[:, :2] == 0).all())
        self.assertTrue(np.array_equal(schema[:, 2:], saved[:, 2:]))
        self.assertTrue((diagnosis.feature_view(x, "intercept_only", 2) == 0).all())

    def test_logit_gap_reconstruction_is_exact(self):
        weight = np.array([[2, 3, 4, 5, 6, 7, 8, 9], [1]*8], dtype=np.float32)
        bias = np.array([.3, -.1], dtype=np.float32)
        x = np.arange(8, dtype=np.float32)
        result = diagnosis.gap_parts(weight, bias, x, 0, 1, 2)
        expected = (weight @ x + bias)[0] - (weight @ x + bias)[1]
        self.assertAlmostEqual(result['total_logit_gap'], float(expected), places=4)
        self.assertAlmostEqual(result['text_logit_gap'] + result['schema_logit_gap'] + result['bias_logit_gap'],
                               float(expected), places=4)

    def test_shift_uses_train_range_per_task(self):
        train = [sample('bird', 10, 'train'), sample('bird', 20, 'train'), sample('ambrosia', 5, 'train')]
        dev = [sample('bird', 30, 'dev'), sample('ambrosia', 5, 'dev')]
        report = diagnosis.schema_shift(train, dev)
        self.assertEqual(report['bird']['column_count']['train_max'], 20)
        self.assertEqual(report['bird']['column_count']['dev_outside_train_count'], 1)
        self.assertEqual(report['ambrosia']['column_count']['dev_outside_train_count'], 0)

    def test_ablation_does_not_accept_outer_dev_rows(self):
        with self.assertRaises(ValueError):
            diagnosis.ablate_train([sample('bird', 5, 'dev')], np.zeros((1, 384)), {}, {}, {})


if __name__ == '__main__':
    unittest.main()
