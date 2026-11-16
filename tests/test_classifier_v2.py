"""Configuration constraints, partial supervision, provenance, and runtime isolation."""

import asyncio
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from unittest.mock import AsyncMock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from adv_text2sql.mcp_servers.text2sql_tool.src import feature_catalog_v2 as catalog
from adv_text2sql.mcp_servers.text2sql_tool.src.multilabel_classifier import HEAD_FEATURES as OLD_HEADS
import prepare_classifier_v2 as data
import collect_classifier_v2 as collector


class CatalogTests(unittest.TestCase):
    def test_v1_is_unchanged_and_v2_can_represent_both_static_controls(self):
        self.assertNotIn(7, OLD_HEADS)
        self.assertEqual(len(catalog.HEAD_FEATURES), 17)
        for features in ((2, 7, 14, 32, 35), (14, 32), ()):
            self.assertEqual(catalog.validate_features(features), tuple(features))
        for features in ((7,), (14, 29), (12, 32), (20, 28), (8,), (2, 2), (True,)):
            with self.subTest(features=features), self.assertRaises(ValueError):
                catalog.validate_features(features)

    def test_dependency_is_added_within_budget_and_never_orphaned(self):
        probabilities = {str(n): 0.01 for n in catalog.HEAD_FEATURES}
        probabilities["7"] = 0.9
        self.assertEqual(catalog.constrained_features(probabilities)["features"], [2, 7])
        self.assertEqual(catalog.constrained_features(probabilities)["dependency_added"], [2])
        self.assertEqual(catalog.constrained_features(probabilities, max_features=1)["features"], [])
        probabilities.update({str(n): 0.9 for n in (2, 14, 32, 35)})
        self.assertEqual(catalog.constrained_features(probabilities)["features"], [2, 7, 14, 32, 35])

    def test_conflicts_invalid_numbers_and_finite_catalog_are_enforced(self):
        probabilities = {str(n): 0.01 for n in catalog.HEAD_FEATURES}
        probabilities.update({"29": 0.99, "14": 0.9})
        self.assertNotIn(14, catalog.constrained_features(probabilities)["features"])
        decision = catalog.select_configuration(probabilities, "bird")
        self.assertIn(decision["configuration"], catalog.TASK_CONFIGURATIONS["bird"])
        with self.assertRaises(ValueError):
            catalog.constrained_features({**probabilities, "7": float("nan")})
        with self.assertRaises(ValueError):
            catalog.constrained_features({**probabilities, "99": 0.1})


class ObservationTests(unittest.TestCase):
    item = {"question_id": 1, "db_id": "db", "SQL": "SELECT 1"}

    def test_postgresql_model_failure_observed_transport_failure_unknown(self):
        row = {"question_id": "1", "db_id": "db", "score": 0, "prediction": "SELECT broken",
               "error": "postgresql:42883:UndefinedFunction"}
        self.assertEqual(data.validate_observation(row, self.item, "bird"), 0)
        for error in ("postgresql:08006:OperationalError", "call_timeout", "provider_unavailable"):
            self.assertIsNone(data.validate_observation({**row, "error": error}, self.item, "bird"))

    def test_ambrosia_is_decision_accuracy_and_error_never_true_negative(self):
        row = {"question_id": "1", "db_id": "db", "score": 1, "prediction": "SELECT 1", "error": None}
        self.assertEqual(data.validate_observation(row, self.item, "ambrosia"), 1)
        with self.assertRaises(ValueError):
            data.validate_observation({**row, "prediction": "error"}, self.item, "ambrosia")

    def test_wrong_id_and_nonbinary_score_rejected(self):
        row = {"question_id": "1", "db_id": "db", "score": 1, "prediction": "SELECT 1", "error": None}
        for changed in ({"db_id": "other"}, {"score": 0.5}):
            with self.assertRaises(ValueError):
                data.validate_observation(row | changed, self.item, "bird")

    def test_client_resolution_uses_source_not_incorrect_manifest_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime.py"
            path.write_text("c = OpenAIChatCompletionClient(temperature=0, max_tokens=4096, max_retries=0, timeout=90)")
            self.assertEqual(data.client_options(path)["max_tokens"], 4096)
            path.write_text("c = OpenAIChatCompletionClient(temperature=0, max_tokens=cap, max_retries=0, timeout=90)")
            with self.assertRaises((ValueError, KeyError)):
                data.client_options(path)

    def test_partial_configuration_labels_stay_null_and_joint_effect_not_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "saved.json"
            source.write_text("{}")
            rows = {"bird:1": {"id": "bird:1", "benchmark": "bird", "split": "train", "group_id": "g",
                               "inputs": {"question": "Q", "evidence": "", "schema_stats": {}}}}
            obs = {("bird:1", "rules_taxonomy"): {"score": 0, "source": "saved.json"},
                   ("bird:1", "compact"): {"score": 1, "source": "saved.json"}}
            manifest = {"sha256": "s", "catalog": catalog.catalog(), "sources_sha256": {}}
            report = data.export_dataset(root, manifest, rows, obs)
            self.assertFalse(report["ready_for_training"])
            output = root / data.DATA_OUTPUT
            example = json.loads((output / "examples.jsonl").read_text())
            self.assertEqual(example["quality_targets"], [None, 0, None, None, 1])
            self.assertEqual(example["loss_mask"], [False, True, False, False, True])
            self.assertNotIn("SQL", example["inputs"])
            pair = json.loads((output / "conditional_pairs.jsonl").read_text())
            self.assertEqual(pair["added_features"], [2, 7, 35])
            self.assertFalse(pair["individual_feature_target_available"])
            self.assertEqual(pair["utility"], 1)

    def test_holdout_and_target_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = {"id": "bird:1", "benchmark": "bird", "split": "test", "group_id": "g",
                   "inputs": {"question": "Q", "evidence": "", "schema_stats": {}}}
            manifest = {"sha256": "s", "catalog": catalog.catalog(), "sources_sha256": {}}
            with self.assertRaisesRegex(ValueError, "holdout"):
                data.export_dataset(Path(tmp), manifest, {row["id"]: row}, {})
            row["split"] = "train"
            row["inputs"]["SQL"] = "SELECT 1"
            with self.assertRaisesRegex(ValueError, "inputs"):
                data.export_dataset(Path(tmp), manifest, {row["id"]: row}, {})


class RuntimeIsolationTests(unittest.TestCase):
    def test_native_packages_are_checked_before_paid_run(self):
        manifest = {"runtime_base": {"runtime_environment": {"packages": {"sqlglot": "1"}, "openai": "2"}}}
        with patch.object(collector.importlib.metadata, "version", side_effect=lambda n: {"sqlglot": "1", "openai": "2"}[n]):
            collector.check_packages(manifest)
        with patch.object(collector.importlib.metadata, "version", return_value="changed"), self.assertRaisesRegex(ValueError, "package changed"):
            collector.check_packages(manifest)

    def fixture(self, benchmark, features):
        import prompt_comparison_runtime as runtime
        candidate = {"instruction": "fixed", "demos": []}
        prompts = {"SQL_PROMPT_TEMPLATE": "original-strict", "SQL_PROMPT_BASIC": "original-basic",
                   "TAXONOMY_DETECTION_PROMPT": "original-taxonomy"}
        manifest = {"runtime_base": {}, "sha256": "s", "model": {"max_tokens": 4096}, "env_file": ".env.deepseek",
                    "limits": {}, "global_limits": {}, "candidates": {benchmark: candidate},
                    "task_manifests": {"ambrosia": {"fixed_prompts": prompts}}}
        job = {"name": "job", "split": "train", "benchmark": benchmark, "features": features, "items": []}

        def initialize(session, root, view, worker, phase):
            session.manifest, session.metadata, session.fit = view, {}, []
            session.impl = SimpleNamespace(**{k: "before" for k in prompts})

        return runtime, manifest, job, initialize

    def test_prompt_and_flags_are_restored_and_basic_template_stays_original(self):
        import prompt_comparison as campaign
        for benchmark, features in (("bird", []), ("bird", [2, 7, 14, 32, 35]), ("ambrosia", [14, 32])):
            runtime, manifest, job, initialize = self.fixture(benchmark, features)
            with patch.object(runtime.Session, "__init__", initialize):
                session = collector.make_session(Path("unused"), manifest, job)
            with patch.object(campaign, "render", return_value=("chosen-strict", [])):
                with session.apply(manifest["candidates"][benchmark], {}, "hybrid", False):
                    self.assertEqual(session.impl.FEAT_7, 7 in features)
                    self.assertEqual(session.impl.FEAT_2, 2 in features)
                    self.assertEqual(session.impl.SQL_PROMPT_BASIC, "before" if benchmark == "bird" else "original-basic")
                self.assertEqual(session.impl.SQL_PROMPT_TEMPLATE, "before")
                self.assertFalse(hasattr(session.impl, "FEAT_7"))

    def test_build_uses_same_mask_as_query(self):
        import prompt_pilot_runtime as base
        runtime, manifest, job, initialize = self.fixture("bird", [2, 7, 14, 32, 35])
        seen = []

        async def prepare(session, profile):
            seen.append(list(base.pilot.PROFILES[profile]))

        with patch.object(runtime.Session, "__init__", initialize):
            session = collector.make_session(Path("unused"), manifest, job)
        previous = copy.deepcopy(base.pilot.PROFILES)
        with patch.object(runtime.Session, "prepare_tools", prepare):
            asyncio.run(session.prepare_tools("hybrid"))
        self.assertEqual(seen, [job["features"]])
        self.assertEqual(base.pilot.PROFILES, previous)

    def test_pending_charge_blocks_before_session_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "budget.json").write_text(json.dumps({"pending": {"ticket": {"reserved": 4096}}}))
            with patch.object(collector, "make_session") as create, self.assertRaisesRegex(RuntimeError, "Unsettled"):
                collector.run_job(root, {"sha256": "s"}, {"name": "job", "features": []})
            create.assert_not_called()

    def test_checkpoint_requires_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "jobs/job"
            folder.mkdir(parents=True)
            (folder / "evaluations.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "provenance"):
                collector.completed(root, {"sha256": "s"}, {"name": "job", "features": []})

    def test_real_session_with_fake_tool_checkpoints_and_resumes_without_network(self):
        import prompt_pilot_runtime as base
        import prompt_comparison_runtime as runtime
        import socket

        manifest = data.read(ROOT / data.DEFAULT_OUTPUT / "experiment.json")
        for benchmark, features in (("bird", []), ("bird", [2, 7, 14, 32, 35]), ("ambrosia", [14, 32])):
            seen = []
            item = {"question_id": 99999, "db_id": "fake_db", "question": "List names", "evidence": "",
                    "SQL": "SELECT secret_reference_only_for_evaluator"}
            job = {"name": "fake_job", "split": "train", "benchmark": benchmark, "features": features, "items": [item]}

            async def prepare(session, profile):
                self.assertEqual(base.pilot.PROFILES[profile], features)
                session.impl = SimpleNamespace(**manifest["task_manifests"]["ambrosia"]["fixed_prompts"])

                async def query(question):
                    seen.append(question)
                    self.assertNotIn(item["SQL"], question)
                    self.assertEqual(session.impl.FEAT_7, 7 in features)
                    return {"status": "success", "query": "SELECT 1"}

                session.tools[profile] = {"fake_db": SimpleNamespace(query=query, engine=SimpleNamespace(dispose=lambda: None))}

            with tempfile.TemporaryDirectory() as tmp, patch.object(base, "load_dotenv"), \
                    patch.object(base, "dotenv_values", return_value={}), \
                    patch.object(base.pilot, "public_model", return_value=manifest["model"]), \
                    patch.object(runtime.Session, "prepare_tools", prepare), \
                    patch.object(base.Session, "score", return_value=(1.0, None)), \
                    patch.object(base.asyncio, "sleep", new=AsyncMock()):
                root = Path(tmp)
                for _ in range(2):
                    session = collector.make_session(root, manifest, job)
                    try:
                        # Windows asyncio creates a local socketpair for its own wakeup pipe.
                        session.runner.get_loop()
                        with patch.object(socket.socket, "connect", side_effect=AssertionError("No network")):
                            session.evaluate(item, manifest["candidates"][benchmark], benchmark=benchmark)
                    finally:
                        session.close()
                self.assertEqual(len(seen), 1)
                budget = data.read(root / "budget.json")
                self.assertEqual(budget["examples"], 1)
                self.assertEqual(budget["teacher_calls"], 0)
                self.assertEqual(budget["task_calls"], 0)


if __name__ == "__main__":
    unittest.main()
