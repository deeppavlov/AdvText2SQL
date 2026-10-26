import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ablation_maintenance import align_protocol, approved_protocols, digest, io_revision_changes, prepare_repair, single_flag_revision_changes


class MaintenanceTests(unittest.TestCase):
    def test_single_flag_plan_reuses_runs_without_weakening_model_or_prompt_checks(self):
        original = {"model": "same", "files_sha256": {"scripts/feature_presets.py": "old", "prompts.py": "same"}}
        revised = copy.deepcopy(original)
        revised["ablation_design"] = "one_flag_vs_all_false"
        revised["files_sha256"]["scripts/feature_presets.py"] = "new"
        self.assertEqual(single_flag_revision_changes(original, revised), ["scripts/feature_presets.py"])
        manifest = {"protocol": original, "protocol_sha256": digest(original), "execution_revisions": [
            {"kind": "single_flag_plan", "protocol": revised, "protocol_sha256": digest(revised)}]}
        self.assertIn(digest(original), approved_protocols(manifest, revised))
        revised["files_sha256"]["prompts.py"] = "changed"
        with self.assertRaises(ValueError):
            single_flag_revision_changes(original, revised)

    def test_reviewed_io_revision_preserves_original_and_rejects_model_or_prompt_changes(self):
        original = {"model": "deepseek-flash", "files_sha256": {
            "scripts/run_openrouter_baseline.py": "old", "prompts.py": "same", "data.json": "same"}}
        revised = copy.deepcopy(original)
        revised["files_sha256"].update({"scripts/run_openrouter_baseline.py": "new", "scripts/atomic_artifacts.py": "added"})
        manifest = {"protocol": original, "protocol_sha256": digest(original)}
        with self.assertRaises(ValueError):
            approved_protocols(manifest, revised)
        manifest["execution_revisions"] = [{"protocol": revised, "protocol_sha256": digest(revised)}]
        self.assertEqual(approved_protocols(manifest, revised), {digest(original), digest(revised)})
        self.assertEqual(manifest["protocol"], original)
        for field in ("model", "prompts.py", "data.json"):
            changed = copy.deepcopy(revised)
            if field == "model":
                changed[field] = "another-model"
            else:
                changed["files_sha256"][field] = "changed"
            with self.assertRaises(ValueError):
                io_revision_changes(original, changed)

    def test_checkpoint_alignment_changes_only_approved_execution_hash(self):
        old = {"model": "same", "protocol_sha256": "old", "feature_flags": {"FEAT_2": True}}
        new = {**old, "protocol_sha256": "new"}
        self.assertEqual(align_protocol(old, new, {"old", "new"}), new)
        self.assertNotEqual(align_protocol(old, new, {"new"}), new)
        changed = {**new, "model": "different"}
        self.assertNotEqual(align_protocol(old, changed, {"old", "new"}), changed)
        self.assertEqual(align_protocol({"model": "legacy"}, {"model": "legacy"}, {None}), {"model": "legacy"})

    def test_repair_backs_up_run_and_removes_only_confirmed_technical_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            source = output / "feat_2/ambrosia_227"
            source.mkdir(parents=True)
            checkpoint = {"1": "SELECT 1", "167": "error", "168": "ambiguous"}
            (source / "query_results.json").write_text(json.dumps(checkpoint), encoding="utf-8")
            (source / "metadata.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            (source / "report.json").write_text('{"accuracy": 70.48}', encoding="utf-8")
            (source / "ambrosia.log").write_text(
                "Processing q_id=167\nPermissionError metadata.json\nProcessing q_id=168\n", encoding="utf-8")
            prepare_repair(output, "feat_2/ambrosia_227", "167")
            after = json.loads((source / "query_results.json").read_text())
            self.assertEqual(after, {"1": "SELECT 1", "168": "ambiguous"})
            metadata = json.loads((source / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "repair_pending")
            backup = output / metadata["technical_repair"]["backup"]
            self.assertEqual(json.loads((backup / "query_results.json").read_text()), checkpoint)
            self.assertTrue((backup / "backup_integrity.json").exists())
            self.assertEqual((source / "report.json").read_text(), '{"accuracy": 70.48}')
            prepare_repair(output, "feat_2/ambrosia_227", "167")  # idempotent after preparation

    def test_model_error_without_metadata_failure_cannot_be_selectively_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            source = output / "feat_2/bird"
            source.mkdir(parents=True)
            (source / "query_results.json").write_text('{"1":"error"}')
            (source / "metadata.json").write_text('{"status":"completed"}')
            (source / "bird.log").write_text('Processing q_id=1\nSQL error\n')
            with self.assertRaises(ValueError):
                prepare_repair(output, "feat_2/bird", "1")
            self.assertEqual(json.loads((source / "query_results.json").read_text()), {"1": "error"})
