"""Offline checks of joint targets, leakage boundaries, constrained inference and replay."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))
from adv_text2sql.mcp_servers.text2sql_tool.src.configuration_classifier_v2 import (  # noqa: E402
    FORMAT, OUTPUTS, ConfigurationFeatureSelector, ConfigurationHead, choose, quality_loss, targets,
)
from adv_text2sql.mcp_servers.text2sql_tool.src.feature_catalog_v2 import catalog, TASK_CONFIGURATIONS  # noqa: E402
from adv_text2sql.mcp_servers.text2sql_tool.src.multilabel_classifier import SchemaScaler, digest  # noqa: E402
import train_configuration_classifier_v2 as training  # noqa: E402


def inputs():
    return {"question": "How many people?", "evidence": "", "schema_stats": {
        "table_count": 1, "column_count": 2, "text_column_count": 1,
        "numeric_column_count": 1, "temporal_column_count": 0, "boolean_column_count": 0}}


def row(task="bird", scores=None):
    names = list(TASK_CONFIGURATIONS[task])
    scores = scores or [1] * len(names)
    return {"id": task+":1", "benchmark": task, "split": "train", "group_id": "g1", "inputs": inputs(),
            "configurations": names, "quality_targets": scores, "loss_mask": [s is not None for s in scores]}


class Tests(unittest.TestCase):
    def test_unknown_and_other_task_have_zero_gradient(self):
        y, mask, task = targets([row(scores=[1, None, 0, 1, 0])])
        logits = torch.zeros_like(y, requires_grad=True)
        loss = quality_loss(logits, y, mask, task, 0.5)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue((logits.grad[~mask] == 0).all())
        self.assertLess(logits.grad[0, 0].item(), 0)
        self.assertGreater(logits.grad[0, 2].item(), 0)

    def test_equal_outcomes_do_not_create_ranking(self):
        for scores in ([1]*5, [0]*5):
            y, mask, task = targets([row(scores=scores)])
            x = torch.randn_like(y)
            self.assertEqual(quality_loss(x, y, mask, task, 0).item(), quality_loss(x, y, mask, task, 2).item())

    def test_joint_quality_is_not_an_individual_feature_label(self):
        y, mask, task = targets([row(scores=[0, 0, 0, 0, 1])])
        self.assertEqual(int(mask.sum()), 5)
        self.assertEqual(int(torch.nan_to_num(y).sum()), 1)
        # The target is compact quality, not separate positive labels for 2, 7, 14, 32, 35.
        self.assertEqual(OUTPUTS[4], ("bird", "compact"))
        self.assertLess(quality_loss(torch.tensor([[0., 0., 0., 0., 5., 0., 0., 0., 0.]]), y, mask, task, 1),
                        quality_loss(torch.zeros_like(y), y, mask, task, 1))

    def test_task_balancing_does_not_count_other_task_outputs(self):
        a, b = row(), row("ambrosia", [0]*4)
        y, mask, task = targets([a, b])
        logits = torch.zeros_like(y)
        logits[1] = 2
        together = quality_loss(logits, y, mask, task)
        separate = sum(quality_loss(logits[i:i+1], y[i:i+1], mask[i:i+1], task[i:i+1]) for i in range(2))/2
        self.assertTrue(torch.allclose(together, separate))

    def test_reject_fabricated_negative_unknown(self):
        r = row(scores=[None, 1, 1, 1, 1])
        r["loss_mask"][0] = True
        with self.assertRaises(ValueError):
            targets([r])

    def test_catalog_and_conservative_margin(self):
        p = {n: 0.5 for n in TASK_CONFIGURATIONS["bird"]}
        self.assertEqual(choose(p, "bird")["configuration"], "compact")
        p["baseline"] = 0.55
        self.assertEqual(choose(p, "bird", .1)["configuration"], "compact")
        self.assertEqual(choose(p, "bird", 0)["features"], [])
        with self.assertRaises(ValueError):
            choose(p, "ambrosia")

    def test_cv_never_splits_groups_or_accepts_dev(self):
        rows = []
        for task in TASK_CONFIGURATIONS:
            for i in range(8):
                r = row(task)
                r.update(id=f"{task}:{i}", group_id=f"{task}:group{i//2}")
                rows.append(r)
        folds = training.grouped_folds(rows)
        by_id = {r["id"]: r for r in rows}
        seen = []
        for fold in folds:
            fit, valid = (set(fold[k]) for k in ("train_ids", "validation_ids"))
            self.assertFalse(fit & valid)
            self.assertEqual(fit | valid, set(by_id))
            self.assertFalse({by_id[k]['group_id'] for k in fit} & {by_id[k]['group_id'] for k in valid})
            self.assertEqual({by_id[k]['benchmark'] for k in valid}, {'bird', 'ambrosia'})
            seen += list(valid)
        self.assertEqual(len(seen), len(set(seen)))
        rows[0]['split'] = 'dev'
        with self.assertRaises(ValueError):
            training.grouped_folds(rows)
        with self.assertRaises(ValueError):
            training.fit(np.zeros((len(rows), 2), dtype=np.float32), rows, training.CANDIDATES[0], epochs=1)

    def test_replay_uses_selected_observed_answer_and_ambiguity_policy(self):
        rs = [row(), row('ambrosia')]
        truth = {}
        originals = {}
        for r in rs:
            originals[r['id']] = {'item': {'question_id': '1', 'SQL': 'ambiguous'}}
            for name in r['configurations']:
                truth[r['id'], name] = {'score': float(name == 'baseline'),
                                       'prediction': 'ambiguous' if name == 'baseline' else 'SELECT 1',
                                       'usage': {'task_calls': 1, 'prompt_tokens': 10, 'completion_tokens': 2}}
        report = training.replay(rs, ['baseline', 'baseline'], truth, originals)
        self.assertEqual(report['macro_accuracy'], 100)
        self.assertEqual(report['benchmarks']['ambrosia']['ambiguity_metrics']['recall'], 1)
        self.assertEqual(report['saved_response_usage']['task_calls'], 2)
        with self.assertRaises(ValueError):
            training.replay(rs, ['rules', 'baseline'], truth, originals)

    def test_saved_selector_routes_tasks_and_rejects_changed_artifact(self):
        class Encoder:
            info = {'test_frozen_encoder': True}

            def encode(self, values):
                return np.zeros((len(values), 2), dtype=np.float32)

        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            head = ConfigurationHead(8)
            with torch.no_grad():
                head.linear.weight.zero_()
                head.linear.bias.zero_()
                head.linear.bias[0] = 10  # BIRD baseline should never compete with Ambrosia.
            np.savez(folder/'head.npz', weight=head.linear.weight.detach().numpy(), bias=head.linear.bias.detach().numpy())
            info = {'format_version': FORMAT, 'catalog': catalog(), 'outputs': [list(x) for x in OUTPUTS],
                    'input_size': 8, 'head_sha256': digest(folder/'head.npz'), 'encoder': Encoder.info,
                    'schema_scaler': SchemaScaler.fit([inputs()]).as_dict(), 'selected': {'margin': 0}}
            (folder/'model.json').write_text(json.dumps(info), encoding='utf-8')
            selector = ConfigurationFeatureSelector(folder, folder, encoder=Encoder())
            decisions = selector.select_many([inputs(), inputs()], ['bird', 'ambrosia'])
            self.assertEqual([v['configuration'] for v in decisions], ['baseline', 'rules_taxonomy'])
            self.assertTrue(all(not p.requires_grad for p in selector.head.parameters()))
            with (folder/'head.npz').open('ab') as f:
                f.write(b'changed')
            with self.assertRaises(ValueError):
                ConfigurationFeatureSelector(folder, folder, encoder=Encoder())


if __name__ == '__main__':
    unittest.main()
