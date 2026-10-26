import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_hybrid_relationships as suite


class RelationshipsTests(unittest.TestCase):
    def test_reference_reuse_preserves_model_data_and_generator(self):
        original = {"model": "same", "temperature": 0, "evaluation_policy": "ambrosia-strict-v1",
                    "files_sha256": {"generator.py": "old", "data.json": "fixed"}}
        current = copy.deepcopy(original)
        current["files_sha256"]["scripts/run_hybrid_relationships.py"] = "new"
        self.assertEqual(suite.check_source_protocol(original, current), ["scripts/run_hybrid_relationships.py"])
        for kind in ("model", "temperature", "generator.py", "data.json"):
            changed = copy.deepcopy(current)
            if kind in changed:
                changed[kind] = "changed"
            else:
                changed["files_sha256"][kind] = "changed"
            with self.assertRaises(ValueError):
                suite.check_source_protocol(original, changed)

    def test_modified_frozen_controls_prevent_a_new_experiment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (root / "control.txt").write_text("modified")
            refs = {"files_sha256": {"control.txt": "original-hash"}}
            protocol = {"model": "same", "evaluation_policy": "ambrosia-strict-v1", "files_sha256": {},
                        "references_sha256": suite.fingerprint(refs)}
            manifest = {"protocol": protocol, "protocol_sha256": suite.fingerprint(protocol),
                        "references": refs, "cases": [{"name": "sql_ambiguity", "enabled": [14, 32, 35]}]}
            suite.save_json(source / "experiment.json", manifest)
            suite.save_json(source / "suite_status.json", {"status": "completed"})
            with patch.object(suite, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "Frozen control file changed"):
                    suite.snapshot(source, protocol)

    def test_two_children_resume_and_changed_model_are_guarded_without_network(self):
        with tempfile.TemporaryDirectory(dir=suite.ROOT / "research_results", prefix=".relationships_test_") as temporary:
            output = Path(temporary) / "experiment"
            settings = {"model": "same"}
            controls = {"source": "offline-controls"}

            def result(folder, benchmark, enabled, protocol, accepted):
                self.assertEqual(enabled, [1, 14, 32, 35])
                return suite.read_json(folder / "fake_result.json", {"status": "pending"})

            def child(command, cwd):
                start, end = command.index("--enabled-features") + 1, command.index("--benchmark")
                self.assertEqual(command[start:end], ["1", "14", "32", "35"])
                self.assertEqual(command[command.index("--evaluation-policy") + 1], "strict")
                benchmark = command[end + 1]
                folder = Path(command[command.index("--output") + 1]) / (
                    "bird" if benchmark == "bird" else "ambrosia_227")
                folder.mkdir(parents=True, exist_ok=True)
                suite.save_json(folder / "fake_result.json", {"status": "completed"})
                return object()

            argv = ["suite", "--output", str(output)]
            with patch.object(suite, "dotenv_values", return_value={"LLM_MODEL_NAME": "same"}), \
                    patch.object(suite, "validate_connection"), \
                    patch.object(suite, "protocol", return_value=settings), \
                    patch.object(suite, "snapshot", return_value=controls), \
                    patch.object(suite, "result", side_effect=result), \
                    patch.object(suite, "monitor"), \
                    patch.object(suite, "write_summary", return_value=[]), \
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
                    self.assertEqual(popen.call_count, 2)
                self.assertEqual(suite.read_json(output / "suite_status.json")["status"], "completed")
                settings["model"] = "different"
                with patch.object(sys, "argv", [*argv, "--dry-run"]):
                    with self.assertRaises(SystemExit):
                        suite.main()
                self.assertEqual(popen.call_count, 2)


if __name__ == "__main__":
    unittest.main()
