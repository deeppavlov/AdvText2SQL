import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from feature_presets import feature_flags
from research_metrics import ambrosia_report
import run_feature_combinations as suite


class CombinationTests(unittest.TestCase):
    def test_strict_scoring_never_rewards_errors_and_preserves_ambiguity_recall(self):
        items = [{"question_id": 1, "SQL": "SELECT 1"}, {"question_id": 2, "SQL": "ambiguous"},
                 {"question_id": 3, "SQL": "ambiguous"}, {"question_id": 4, "SQL": "SELECT 4"}]
        result = ambrosia_report(items, {"1": "error", "2": "error", "3": "ambiguous", "4": "SELECT 4"})
        self.assertEqual(result["accuracy"], 50)
        self.assertEqual(result["legacy_accuracy"], 75)
        self.assertEqual(result["generation_errors"], 2)
        metrics = result["ambiguity_metrics"]
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]), (1, 0, 1, 1))
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(metrics["errors_unambiguous"], 1)
        with self.assertRaises(ValueError):
            ambrosia_report(items, {"1": "error"})

    def test_single_ablation_guard_stays_strict_and_combinations_are_explicit(self):
        flags = feature_flags("combination", [2, 35])
        self.assertEqual([k for k, v in flags.items() if v], ["FEAT_2", "FEAT_35"])
        self.assertEqual(len(flags), 36)
        for values in ([2], [2, 2], [2, 22], []):
            with self.assertRaises(ValueError):
                feature_flags("combination", values)
        with self.assertRaises(ValueError):
            feature_flags("ablation", [2, 35])

    def test_frozen_controls_and_protocol_cannot_be_edited_silently(self):
        references = {"score": 40}
        protocol = {"references_sha256": suite.fingerprint(references)}
        manifest = {"references": references, "protocol": protocol,
                    "protocol_sha256": suite.fingerprint(protocol)}
        suite.validate_manifest(manifest)
        references["score"] = 50
        with self.assertRaisesRegex(ValueError, "snapshot"):
            suite.validate_manifest(manifest)
        protocol["references_sha256"] = suite.fingerprint(references)
        with self.assertRaisesRegex(ValueError, "protocol hash"):
            suite.validate_manifest(manifest)

    def test_stage_selection_can_reject_combined_features_and_rules(self):
        def candidate(features, score, tokens):
            return {"enabled": features, "benchmarks": {
                b: {"status": "completed", "accuracy": score, "tokens": tokens} for b in suite.BENCHMARKS}}

        rows = {"feat_2": candidate([2], 47, 2000), "feat_35": candidate([35], 47, 470),
                "schema_2_35": candidate([2, 35], 46, 2100)}
        rules = suite.next_case(2, rows)
        self.assertEqual(rules["enabled"], [14, 35])
        self.assertEqual(rules["parents"], ["feat_35"])
        rows["sql_rules"] = candidate([14, 35], 45, 600)
        self.assertEqual(suite.next_case(3, rows)["enabled"], [32, 35])
        rows["sql_rules"] = candidate([14, 35], 49, 600)
        self.assertEqual(suite.next_case(3, rows)["enabled"], [14, 32, 35])
        rows["schema_2_35"] = candidate([2, 35], 50, 2100)
        self.assertEqual(suite.next_case(2, rows)["enabled"], [2, 14, 35])
        rows["schema_2_35"]["benchmarks"]["ambrosia"]["status"] = "running"
        with self.assertRaises(ValueError):
            suite.next_case(2, rows)

    def test_reference_model_and_core_code_changes_are_rejected(self):
        original = {"model": "same", "files_sha256": {"prompts.py": "same"}}
        manifest = {"protocol": original, "protocol_sha256": suite.fingerprint(original)}
        with patch.object(suite, "read_json", side_effect=[manifest, {"status": "completed"}]):
            with self.assertRaisesRegex(ValueError, "settings differ"):
                suite.reference_snapshot(Path("unused"), {**original, "model": "different"})
        changed = copy.deepcopy(original)
        changed["files_sha256"]["prompts.py"] = "changed"
        with patch.object(suite, "read_json", side_effect=[manifest, {"status": "completed"}]):
            with self.assertRaisesRegex(ValueError, "Unreviewed"):
                suite.reference_snapshot(Path("unused"), changed)

    def test_three_stages_resume_without_repeating_paid_children(self):
        with tempfile.TemporaryDirectory(dir=suite.ROOT / "research_results", prefix=".combination_test_") as tmp:
            output = Path(tmp) / "series"
            config = {"LLM_MODEL_NAME": "test-model"}
            base = {"model": "test-model"}
            refs = {"source": "offline", "results": {}}
            for name, flags in suite.REFERENCES.items():
                refs["results"][name] = {"enabled": flags, "benchmarks": {}}
                for b in suite.BENCHMARKS:
                    refs["results"][name]["benchmarks"][b] = {
                        "status": "completed", "saved": 1, "accuracy": 40,
                        "f1": 0, "errors": 0, "tokens": 100,
                    }

            def result(folder, benchmark, enabled, protocol, accepted):
                return suite.read_json(folder / "fake_result.json", {
                    "status": "pending", "saved": 0, "accuracy": None,
                    "f1": None, "errors": 0, "tokens": None,
                })

            def child(command, cwd):
                self.assertIn("combination", command)
                self.assertEqual(command[command.index("--evaluation-policy") + 1], "strict")
                name = command[command.index("--benchmark") + 1]
                target = Path(command[command.index("--output") + 1]) / ("bird" if name == "bird" else "ambrosia_227")
                target.mkdir(parents=True, exist_ok=True)
                suite.save_json(target / "fake_result.json", {
                    "status": "completed", "saved": 1, "accuracy": 50,
                    "f1": 0.5, "errors": 0, "tokens": 200,
                })
                return object()

            argv = ["suite", "--output", str(output)]
            with patch.object(suite, "dotenv_values", return_value=config), \
                    patch.object(suite, "validate_connection"), \
                    patch.object(suite, "source_protocol", return_value=base), \
                    patch.object(suite, "reference_snapshot", return_value=refs), \
                    patch.object(suite, "result", side_effect=result), \
                    patch.object(suite, "monitor"), \
                    patch.object(suite.subprocess, "Popen", side_effect=child) as popen:
                with patch.object(sys, "argv", [*argv, "--dry-run"]):
                    suite.main()
                    self.assertFalse(output.exists())
                with patch.object(sys, "argv", [*argv, "--prepare"]):
                    suite.main()
                    popen.assert_not_called()
                with patch.object(sys, "argv", argv):
                    suite.main()
                    suite.main()
                    self.assertEqual(popen.call_count, 6)
                manifest = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
                self.assertEqual(len(manifest["cases"]), 3)
                self.assertEqual(suite.read_json(output / "suite_status.json")["status"], "completed")
                refs["source"] = "modified"
                with patch.object(sys, "argv", [*argv, "--dry-run"]):
                    with self.assertRaises(SystemExit):
                        suite.main()
                self.assertEqual(popen.call_count, 6)


if __name__ == "__main__":
    unittest.main()
