import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import durable_prompt_store as store
import recover_prompt_disk_state as recovery
import prompt_comparison_durable as overlay


class DiskRecoveryTests(unittest.TestCase):
    def test_accepts_only_valid_prefix_and_null_tail(self):
        valid = b'{"a":1}\n'
        rows, prefix, tail = recovery.raw_prefix(valid + b"\0" * 12)
        self.assertEqual(rows, [{"a": 1}])
        self.assertEqual(prefix, valid)
        self.assertEqual(tail, b"\0" * 12)
        for invalid in (valid, valid + b"{broken", b"\0" * 12, valid + b"\0\n" + valid):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                recovery.raw_prefix(invalid)

    def test_sync_precedes_replace_and_append_is_synced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            events = []
            fsync, replace = store.os.fsync, store.os.replace

            def sync(fd):
                events.append("sync")
                return fsync(fd)

            def rename(source, target):
                events.append("replace")
                return replace(source, target)

            with (
                patch.object(store.os, "fsync", sync),
                patch.object(store.os, "replace", rename),
            ):
                store.write_json(path, {"preserved": True})
                store.append(Path(tmp) / "events.jsonl", {"answer": "cached"})
            self.assertEqual(events, ["sync", "replace", "sync"])
            self.assertEqual(json.loads(path.read_text()), {"preserved": True})

    def test_real_saved_messages_replay_without_network_or_db(self):
        root = (
            overlay.campaign.ROOT
            / "research_results/prompt_optimization_comparison_20261105"
        )
        manifest = json.loads((root / "experiment.json").read_bytes())
        raw = (root / "jobs/sql__mipro_few__r1/search/raw_responses.jsonl").read_bytes()
        pair = [json.loads(row) for row in raw.splitlines()[:2]]
        item = manifest["cohorts"]["sql"]["inner_val"][0]
        sql, issue = recovery.replay(item, pair)
        self.assertTrue(sql)
        self.assertIsNone(issue)
        changed = json.loads(json.dumps(pair))
        changed[1]["messages"][-1]["content"] = "Different request"
        with self.assertRaises(ValueError):
            recovery.replay(item, changed)

    def test_overlay_is_isolated_and_keeps_original_runtime_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiment.json").write_text("{}")
            original = overlay.runtime.write_json
            original_append = overlay.pilot_runtime.append
            seen = []

            def fake_main():
                seen.append(overlay.runtime.write_json is store.write_json)
                seen.append(overlay.budget.write_json is store.write_json)
                seen.append(overlay.pilot_runtime.append is store.append)

            with (
                patch.object(overlay.campaign.pilot, "output_path", return_value=root),
                patch.object(overlay.campaign, "main", fake_main),
                patch.object(
                    sys, "argv", ["prompt_comparison_durable.py", "--dry-run"]
                ),
            ):
                overlay.main()
                overlay.main()
            self.assertEqual(seen, [True] * 6)
            self.assertIs(overlay.runtime.write_json, original)
            self.assertIs(overlay.pilot_runtime.append, original_append)
            self.assertTrue((root / "io_overlay.json").exists())


if __name__ == "__main__":
    unittest.main()
