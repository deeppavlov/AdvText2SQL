import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from feature_presets import ABLATION_FEATURES, ablation_cases, feature_flags
import run_feature_ablation as suite


class FeatureAblationTests(unittest.TestCase):
    def test_every_feature_has_one_flag_and_is_compared_with_all_false_baseline(self):
        cases = ablation_cases(ABLATION_FEATURES)
        by_name = {case["name"]: case for case in cases}
        self.assertEqual(cases[0]["enabled"], [])
        self.assertEqual(by_name["feat_1"]["enabled"], [1])
        self.assertEqual(by_name["feat_2"]["enabled"], [2])
        self.assertEqual(len(cases), 34)
        for case in cases[1:]:
            self.assertEqual(len(case["enabled"]), 1)
            self.assertEqual(case["enabled"], case["targets"])
            self.assertEqual(case["control"], "baseline")
            flags = feature_flags("ablation", case["enabled"])
            self.assertEqual(sum(flags.values()), 1)
            self.assertEqual(len(flags), 36)
        for number in (3, 4, 7, 27, 33, 34, 36):
            self.assertTrue(by_name[f"feat_{number}"]["note"])
        self.assertEqual(len(cases), len({case["name"] for case in cases}))

    def test_multiple_flags_are_rejected_and_no_dependencies_are_enabled_automatically(self):
        for features in ([2, 3], [32, 33], [22], []):
            with self.assertRaises(ValueError):
                feature_flags("ablation", features)
        flags = feature_flags("ablation", [3])
        self.assertEqual({key for key, value in flags.items() if value}, {"FEAT_3"})
        self.assertEqual(ablation_cases([26])[1]["enabled"], [26])
        for number in (22, 24, 31, 37):
            with self.assertRaises(ValueError):
                ablation_cases([number])

    def test_health_detects_error_tail_and_stall_but_not_ambiguity(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            checkpoint = folder / "query_results.json"
            checkpoint.write_text(json.dumps({str(n): "error" for n in range(8)}))
            self.assertIn("generation errors", suite.health_reason(folder, "bird", time.time(), 900, 8))
            checkpoint.write_text(json.dumps({str(n): "ambiguous" for n in range(8)}))
            self.assertIsNone(suite.health_reason(folder, "bird", time.time(), 900, 8))
            checkpoint.unlink()
            self.assertIn("progress", suite.health_reason(folder, "bird", time.time() - 1000, 900, 8))

    def test_monitor_stops_only_its_child_on_failure(self):
        class Child:
            pid = 123
            terminated = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                return 0

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "baseline/bird"
            directory.mkdir(parents=True)
            child = Child()
            args = type("Args", (), {"stall_seconds": 900, "error_limit": 8})()
            with patch.object(suite, "health_reason", return_value="test failure"):
                with self.assertRaises(RuntimeError):
                    suite.monitor(child, directory, "bird", args)
            self.assertTrue(child.terminated)
            self.assertEqual(suite.read_json(Path(temporary) / "monitor_stop.json")["reason"], "test failure")

    def test_native_summary_does_not_label_cny_or_missing_snapshot_as_usd(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            directory = output / "baseline/bird"
            directory.mkdir(parents=True)
            suite.save_json(directory / "metadata.json", {"status": "completed",
                            "cost_usd_this_attempt": None, "balance_delta_this_attempt": {"CNY": 0.2}})
            suite.save_json(directory / "report.json", {"overall_accuracy": 50})
            manifest = {"protocol": {"model": "deepseek-flash"}, "cases": ablation_cases([]), "benchmarks": ["bird"]}
            rows = suite.summary(output, manifest)
            self.assertIsNone(rows[0]["key_usage_delta_usd"])
            self.assertEqual(rows[0]["account_balance_deltas"], {"CNY": 0.2})
            self.assertIn("0.2000 CNY", (output / "summary.md").read_text(encoding="utf-8"))

    def test_summary_compares_every_case_only_against_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for name, score in (("baseline", 40), ("feat_2", 50), ("feat_3", 45), ("feat_4", 35)):
                directory = output / name / "bird"
                directory.mkdir(parents=True)
                suite.save_json(directory / "metadata.json", {"status": "completed"})
                suite.save_json(directory / "report.json", {"overall_accuracy": score})
            manifest = {"protocol": {"model": "offline"}, "cases": ablation_cases([2, 3, 4]), "benchmarks": ["bird"]}
            rows = {row["case"]: row for row in suite.summary(output, manifest)}
            self.assertEqual(rows["feat_3"]["control"], "baseline")
            self.assertEqual(rows["feat_3"]["delta_pp"], 5)
            self.assertEqual(rows["feat_3"]["effect"], "improved")
            self.assertEqual(rows["feat_4"]["delta_pp"], -5)
            self.assertEqual(rows["feat_4"]["effect"], "worse")

    def test_dry_run_resume_and_changed_model_without_network(self):
        with tempfile.TemporaryDirectory(prefix=".ablation_test_", dir=suite.ROOT / "research_results") as temporary:
            folder = Path(temporary)
            env_file = folder / ".env.test"
            env_file.write_text("LLM_MODEL_NAME=deepseek/deepseek-v4-flash\n"
                                "LLM_API_KEY=sk-or-offline-placeholder\nLLM_BASE_URL=https://openrouter.ai/api/v1\n")
            output = folder / "series"
            argv = ["suite", "--env-file", str(env_file), "--output", str(output), "--features", "1", "2"]
            with patch.object(sys, "argv", [*argv, "--dry-run"]), patch.object(suite.subprocess, "Popen") as popen:
                suite.main()
                popen.assert_not_called()
                self.assertFalse(output.exists())

            class CompletedChild:
                returncode = 0

                def poll(self):
                    return 0

            def complete(command, cwd):
                case_output = Path(command[command.index("--output") + 1])
                benchmark = command[command.index("--benchmark") + 1]
                target = case_output / ("bird" if benchmark == "bird" else "ambrosia_227")
                target.mkdir(parents=True, exist_ok=True)
                digest = command[command.index("--protocol-sha256") + 1]
                suite.save_json(target / "metadata.json", {"status": "completed",
                                                          "configuration": {"protocol_sha256": digest}})
                suite.save_json(target / "report.json", {"accuracy": 50, "total": 227})
                return CompletedChild()

            with patch.object(sys, "argv", argv), patch.object(suite.subprocess, "Popen", side_effect=complete) as popen:
                suite.main()
                suite.main()
                self.assertEqual(popen.call_count, 6)  # second launch skips all six completed jobs
                first = popen.call_args_list[0].args[0]
                self.assertEqual(first[first.index("--model") + 1], "deepseek/deepseek-v4-flash")
                self.assertNotIn("--enabled-features", first)
                second_feature = popen.call_args_list[4].args[0]
                self.assertEqual(second_feature[second_feature.index("--enabled-features") + 1:], ["2"])
            env_file.write_text(env_file.read_text().replace("deepseek-v4-flash", "another-model"))
            with patch.object(sys, "argv", argv), patch.object(suite.subprocess, "Popen") as popen:
                with self.assertRaises(SystemExit):
                    suite.main()
                popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
