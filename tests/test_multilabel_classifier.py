import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from adv_text2sql.mcp_servers.text2sql_tool.src.multilabel_classifier import (  # noqa: E402
    HEAD_FEATURES, SCHEMA_FIELDS, FrozenTextEncoder, LinearFeatureHead, MultilabelFeatureSelector,
    SchemaScaler, constrained_features, digest, masked_bce, positive_weights,
)
from adv_text2sql.mcp_servers.text2sql_tool.src.multilabel_generator import (  # noqa: E402
    MultilabelText2SQLGenerator, isolated_feature_class,
)
from train_multilabel_classifier import choose_threshold, decisions, fit_head, labels, metrics  # noqa: E402
from classifier_benchmark import compare_with_controls  # noqa: E402


def inputs(question="List names", count=2):
    return {"question": question, "evidence": "", "schema_stats": {name: count for name in SCHEMA_FIELDS}}


class ClassifierMathTests(unittest.TestCase):
    def test_masked_nan_target_has_zero_gradient(self):
        logits = torch.zeros((1, 2), requires_grad=True)
        target = torch.tensor([[float("nan"), 1.0]])
        loss = masked_bce(logits, target, torch.tensor([[False, True]]))
        self.assertAlmostEqual(float(loss.detach()), np.log(2), places=6)
        loss.backward()
        self.assertEqual(logits.grad[0, 0], 0)
        self.assertAlmostEqual(float(logits.grad[0, 1]), -0.5)

    def test_all_missing_targets_are_finite_zero_loss(self):
        logits = torch.randn((2, 3), requires_grad=True)
        loss = masked_bce(logits, torch.full_like(logits, float("nan")), torch.zeros_like(logits, dtype=torch.bool))
        loss.backward()
        self.assertEqual(float(loss.detach()), 0)
        self.assertTrue((logits.grad == 0).all())

    def test_invalid_observed_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            masked_bce(torch.zeros((1, 1)), torch.tensor([[2.0]]), torch.ones((1, 1), dtype=torch.bool))
        with self.assertRaises(ValueError):
            masked_bce(torch.zeros((1, 1)), torch.tensor([[float("nan")]]), torch.ones((1, 1), dtype=torch.bool))

    def test_weight_counts_ignore_masked_targets_and_cap_rare_classes(self):
        y = torch.tensor([[1.0, float("nan")], [0.0, 1.0], [0.0, 0.0]])
        mask = torch.tensor([[True, False], [True, True], [True, True]])
        weights = positive_weights(y, mask)
        self.assertAlmostEqual(float(weights[0]), np.sqrt(2), places=6)
        self.assertEqual(float(weights[1]), 1)

    def test_schema_scaling_fits_training_only_and_constant_counts_are_safe(self):
        training = [inputs(count=1), inputs(count=3)]
        scaler = SchemaScaler.fit(training)
        before = copy.deepcopy(scaler.as_dict())
        self.assertTrue(np.allclose(scaler.transform(training).mean(0), 0, atol=1e-6))
        scaler.transform([inputs(count=1000000)])
        self.assertEqual(scaler.as_dict(), before)
        constant = SchemaScaler.fit([inputs(count=0)])
        self.assertTrue(np.isfinite(constant.transform([inputs(count=0)])).all())
        with self.assertRaises(ValueError):
            scaler.transform([{**inputs(), "labels": [1]}])

    def test_constraints_remove_overrides_and_respect_feature_budget(self):
        p = {str(n): 0.0 for n in HEAD_FEATURES}
        p.update({"14": 0.9, "29": 0.95, "20": 0.8, "28": 0.85, "35": 0.7})
        result = constrained_features(p, 0.5, 3)
        self.assertEqual(result["features"], [28, 29, 35])
        self.assertFalse(result["feature_flags"]["FEAT_17"])
        p.update({"32": 0.99, "12": 0.98})
        result = constrained_features(p, 0.5, 3)
        self.assertNotIn(12, result["features"])
        self.assertEqual(len(result["features"]), 3)
        with self.assertRaises(ValueError):
            constrained_features({**p, "999": 0.9}, 0.5)

    def test_label_metrics_do_not_credit_missing_labels(self):
        target = np.zeros((1, len(HEAD_FEATURES)), dtype=np.float32)
        mask = np.zeros_like(target, dtype=bool)
        mask[0, 0] = True
        target[0, 0] = 1
        predicted = np.ones_like(mask)
        report = metrics(target, mask, predicted)
        self.assertEqual(report["tp"], 1)
        self.assertEqual(report["fp"], 0)
        self.assertEqual(report["observed_labels"], 1)
        self.assertEqual(report["micro_f1"], 1)

    def test_null_targets_survive_loading_and_are_masked(self):
        row = {"multilabel": {"benefit_targets": [None, 1], "loss_mask": [False, True]}}
        y, mask = labels([row])
        self.assertTrue(torch.isnan(y[0, 0]))
        self.assertFalse(mask[0, 0])

    def test_threshold_is_selected_using_constrained_dev_predictions(self):
        target = np.zeros((2, len(HEAD_FEATURES)))
        mask = np.ones_like(target, dtype=bool)
        probabilities = np.full_like(target, 0.05)
        target[0, 0] = 1
        probabilities[0, 0] = 0.7
        threshold, candidates = choose_threshold(target, mask, probabilities, 3)
        self.assertEqual(threshold, 0.65)
        self.assertEqual(len(candidates), 5)
        self.assertEqual(decisions(probabilities, threshold, 3).sum(), 1)

    def test_fit_is_deterministic_and_optimizes_only_head(self):
        torch.set_num_threads(1)
        x = torch.tensor([[1.0, 0], [0.0, 1], [1.0, 1], [0.0, 0]])
        y = torch.zeros((4, len(HEAD_FEATURES)))
        y[:2, 0] = 1
        mask = torch.ones_like(y, dtype=torch.bool)
        a, history = fit_head(x, y, mask, x, y, mask, epochs=5, patience=3)
        b, _ = fit_head(x, y, mask, x, y, mask, epochs=5, patience=3)
        self.assertEqual(history["best_epoch"], 5)
        for p, q in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(p, q))


class EncoderArtifactTests(unittest.TestCase):
    def test_encoder_stays_frozen_eval_and_pooling_ignores_padding(self):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, **kwargs):
                class Result:
                    pass
                result = Result()
                result.last_hidden_state = self.value * torch.tensor([[[1.0, 0], [0.0, 1], [99.0, 99]]])
                return result

        def tokenizer(texts, **kwargs):
            self.assertIn("Evidence:", texts[0])
            return {"input_ids": torch.ones((1, 3), dtype=torch.long), "attention_mask": torch.tensor([[1, 1, 0]])}

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            files = ("config.json", "model.safetensors", "tokenizer.json")
            for name in files:
                (folder / name).write_text("test")
            info = {"format_version": "frozen-encoder-v1", "files_sha256": {n: digest(folder / n) for n in files}, "max_length": 256}
            (folder / "encoder.json").write_text(json.dumps(info))
            base = Encoder()
            encoder = FrozenTextEncoder(folder, model=base, tokenizer=tokenizer)
            result = encoder.encode([inputs()])
            self.assertTrue(np.allclose(result, [[1 / np.sqrt(2), 1 / np.sqrt(2)]]))
            self.assertFalse(base.training)
            self.assertTrue(all(not p.requires_grad for p in base.parameters()))
            self.assertTrue(all(p.grad is None for p in base.parameters()))

    def test_saved_head_round_trip_and_hash_guard(self):
        class Encoder:
            info = {"model": "test-only"}

            def encode(self, values):
                return np.ones((len(values), 2), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            head = LinearFeatureHead(8)
            np.savez(folder / "head.npz", weight=head.linear.weight.detach().numpy(), bias=head.linear.bias.detach().numpy())
            info = {"format_version": "multilabel-classifier-v1", "feature_numbers": list(HEAD_FEATURES),
                    "input_size": 8, "encoder": Encoder.info, "schema_scaler": SchemaScaler.fit([inputs()]).as_dict(),
                    "threshold": 0.5, "max_features": 3, "head_sha256": digest(folder / "head.npz")}
            (folder / "model.json").write_text(json.dumps(info))
            selector = MultilabelFeatureSelector(folder, folder, encoder=Encoder())
            actual = selector.probabilities([inputs()])
            with torch.inference_mode():
                expected = head(torch.tensor([[1.0, 1, 0, 0, 0, 0, 0, 0]])).sigmoid().numpy()
            self.assertTrue(np.allclose(actual, expected))
            (folder / "head.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "head changed"):
                MultilabelFeatureSelector(folder, folder, encoder=Encoder())


class MultilabelRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_masks_use_distinct_classes_and_cached_build_contexts(self):
        built, engines, captured = [], [], []

        class Engine:
            disposed = False

            def dispose(self):
                self.disposed = True

        class Selector:
            def select_inputs(self, item):
                captured.append(item)
                return {"features": [14, 35] if "strict" in item["question"] else [35], "configuration": "multilabel"}

        def factory(features):
            class Tool:
                def __init__(self, **kwargs):
                    self.engine = Engine()
                    engines.append(self.engine)

                async def build(self):
                    built.append(features)
                    self.db_schema = "table\tcolumn\ttype\na\tid\tINTEGER"
                    self.system_prompt = "same TSV"
                    self.db_relationships, self.relationships_str, self.column_stats, self.column_stats_str, self._db_id = {}, "", {}, "", "one"

                async def query(self, question):
                    return {"status": "success", "query": "SELECT 2" if 14 in features else "SELECT 1"}

            return Tool

        tool = MultilabelText2SQLGenerator("test://one", object(), Selector(), generator_factory=factory)
        await tool.build()
        first = await tool.query("List names")
        second = await tool.query("question: strict names, evidence (may be empty): strict evidence")
        self.assertEqual(first["query"], "SELECT 1")
        self.assertEqual(second["query"], "SELECT 2")
        self.assertEqual(captured[1]["question"], "strict names")
        self.assertEqual(captured[1]["evidence"], "strict evidence")
        self.assertEqual(built, [(35,)])
        tool.close()
        self.assertTrue(all(e.disposed for e in engines))

    async def test_real_feature_classes_do_not_mutate_legacy_flags(self):
        from adv_text2sql.mcp_servers.text2sql_tool.src import text2sql_implementation as original
        before = original.FEAT_14
        cheap, strict = isolated_feature_class((35,)), isolated_feature_class((14, 35))
        self.assertFalse(cheap.generate_sql.__globals__["FEAT_14"])
        self.assertTrue(strict.generate_sql.__globals__["FEAT_14"])
        self.assertEqual(original.FEAT_14, before)

    async def test_failed_preparation_is_disposed(self):
        disposed = []

        class Fake:
            def __init__(self, **kwargs):
                self.engine = self

            def dispose(self):
                disposed.append(self)

            async def build(self):
                raise RuntimeError("build failed")

        tool = MultilabelText2SQLGenerator("test://one", object(), None, generator_factory=lambda flags: Fake)
        with self.assertRaises(RuntimeError):
            await tool._template((1,))
        self.assertEqual(len(disposed), 1)
        tool.close()


class ClassifierControlComparisonTests(unittest.TestCase):
    def test_only_exact_submitted_ids_are_compared(self):
        def observation(correct):
            return {"observed": True, "correct": correct}
        rows = [{"benchmark": "bird", "question_id": "1", "observations": {
            "baseline": observation(False), "feat_35": observation(True), "sql_ambiguity": observation(True)}}]
        items = [{"question_id": 1, "SQL": "SELECT 1"}]
        report = {"results": [{"question_id": 1, "score": True, "predicted_sql": "SELECT 1"}]}
        result = compare_with_controls(rows, "bird", items, {"1": "SELECT 1"}, report)
        self.assertEqual(result["controls"]["baseline"]["rescued"], 1)
        self.assertEqual(result["classifier"]["correct"], 1)
        with self.assertRaises(ValueError):
            compare_with_controls(rows, "bird", items, {"1": "SELECT 1", "2": "SELECT 2"}, report)


if __name__ == "__main__":
    unittest.main()
