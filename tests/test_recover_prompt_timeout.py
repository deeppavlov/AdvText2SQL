import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from atomic_artifacts import write_json
from prompt_pilot_budget import Ledger
from prompt_comparison_runtime import DualLedger
import prompt_comparison as campaign
import recover_prompt_timeout as recovery


class RecoveryTests(unittest.TestCase):
    def budgets(self, root):
        (root / "local").mkdir()
        a = Ledger(root, campaign.GLOBAL_LIMITS)
        b = Ledger(root / "local", campaign.LIMITS)
        pair = DualLedger(b, a, "job")
        pair.reserve("task", ["uncertain request"], 4096)
        return copy.deepcopy(a.data), copy.deepcopy(b.data)

    def test_unknown_charged_without_fabricating_actual_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            original, _ = self.budgets(Path(tmp))
            expected = next(iter(original["pending"].values()))["reserved"]
            new = recovery.charge_unknown(original, "event")
            self.assertEqual(new["tokens"], original["tokens"] + expected)
            self.assertEqual(new["unknown_usage_tokens"], expected)
            self.assertFalse(new["pending"])
            for key in (
                "task_calls",
                "teacher_calls",
                "active_seconds",
                "examples",
                "prompt_tokens",
                "completion_tokens",
                "task_tokens",
                "teacher_tokens",
            ):
                self.assertEqual(new[key], original[key])
            self.assertTrue(original["pending"])

    def test_partial_write_can_be_completed_without_double_charge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = self.budgets(root)
            journal = {
                "state": "prepared",
                "unchanged_artifacts_sha256": {},
                "patches": [
                    {
                        "path": name,
                        "before": value,
                        "after": recovery.charge_unknown(value, "event"),
                    }
                    for name, value in (("budget.json", a), ("local/budget.json", b))
                ],
            }
            path = root / "recoveries/test.json"
            path.parent.mkdir()
            # Simulate a crash after the journal and first budget were written.
            write_json(path, journal)
            write_json(root / "budget.json", journal["patches"][0]["after"])
            recovery.apply_journal(root, path, journal)
            completed = json.loads(path.read_text(encoding="utf8"))
            self.assertFalse(recovery.apply_journal(root, path, completed))
            for entry in journal["patches"]:
                self.assertEqual(
                    json.loads((root / entry["path"]).read_text(encoding="utf8")),
                    entry["after"],
                )
            # Real unmodified ledger can resume; new API usage remains separate.
            pair = DualLedger(
                Ledger(root / "local", campaign.LIMITS),
                Ledger(root, campaign.GLOBAL_LIMITS),
                "job",
            )
            ticket = pair.reserve("task", ["retry"], 10)
            pair.settle(ticket, 20, 5)
            self.assertEqual(pair.data["prompt_tokens"], 20)
            self.assertEqual(
                pair.data["tokens"], pair.data["unknown_usage_tokens"] + 25
            )

    def test_rejects_other_outstanding_or_unexpected_role(self):
        for pending in (
            {},
            {"a": {"role": "teacher", "reserved": 100}},
            {
                "a": {"role": "task", "reserved": 100},
                "b": {"role": "task", "reserved": 100},
            },
        ):
            with self.subTest(pending=pending), self.assertRaises(ValueError):
                recovery.charge_unknown({"pending": pending}, "event")

    def test_concurrent_change_prevents_all_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = self.budgets(root)
            journal = {
                "state": "prepared",
                "unchanged_artifacts_sha256": {},
                "patches": [
                    {
                        "path": name,
                        "before": value,
                        "after": recovery.charge_unknown(value, "event"),
                    }
                    for name, value in (("budget.json", a), ("local/budget.json", b))
                ],
            }
            changed = {**b, "task_calls": 99}
            write_json(root / "local/budget.json", changed)
            with self.assertRaises(ValueError):
                recovery.apply_journal(root, root / "recoveries/test.json", journal)
            self.assertEqual(
                json.loads((root / "budget.json").read_text(encoding="utf8")), a
            )


if __name__ == "__main__":
    unittest.main()
