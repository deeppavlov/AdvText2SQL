from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bird_compact_budget_extension as extension


class BudgetExtensionTests(unittest.TestCase):
    def test_caps_preserve_usage_and_persistent_results(self):
        c = extension.compact
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            folder = root / "jobs/bird_compact__r2/confirm"
            folder.mkdir(parents=True)
            ledger = c.budget.Ledger(folder, c.LIMITS)
            ledger.data.update(tokens=2_976_793, task_calls=535, examples=275)
            ledger.save()
            result = folder.parent / "evaluations.json"
            result.write_text('{"saved": true}', encoding="utf-8")
            before = result.read_bytes()
            value = dict(
                original_limits=c.LIMITS,
                original_global_limits=c.GLOBAL_LIMITS,
                effective_limits={**c.LIMITS, "tokens": extension.LOCAL_TOKENS},
                effective_global_limits={
                    **c.GLOBAL_LIMITS,
                    "tokens": extension.GLOBAL_TOKENS,
                },
            )
            with extension.effective_ledgers(root, value):
                local = c.budget.Ledger(folder, c.LIMITS)
                global_ = c.budget.Ledger(root, c.GLOBAL_LIMITS)
                self.assertEqual(local.data["tokens"], 2_976_793)
                self.assertEqual(local.data["task_calls"], 535)
                self.assertEqual(local.data["examples"], 275)
                self.assertEqual(global_.limits["tokens"], 11_000_000)
                # Previously blocked 30K reservation now fits; teacher stays forbidden.
                local.reserve("task", [{"content": "x" * 25_000}], 4096)
                with self.assertRaises(c.budget.BudgetStop):
                    local.reserve("teacher", [], 1)
                with self.assertRaises(ValueError):
                    c.budget.Ledger(root / "foreign", c.LIMITS)
                with self.assertRaises(ValueError):
                    c.budget.Ledger(folder, {**c.LIMITS, "examples": 900})
            self.assertEqual(before, result.read_bytes())
            self.assertEqual(c.LIMITS["tokens"], 3_000_000)

    def test_preparation_is_offline_and_rejects_pending(self):
        c = extension.compact
        m = dict(sha256="frozen", limits=c.LIMITS, global_limits=c.GLOBAL_LIMITS)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c.store.write_json(root / "budget.json", {"tokens": 42, "pending": {}})
            before = (root / "budget.json").read_bytes()
            value = extension.proposal(root, m)
            extension.validate_extension(value, m)
            self.assertEqual(before, (root / "budget.json").read_bytes())
            changed = deepcopy(value)
            changed["effective_limits"]["teacher_calls"] = 1
            changed["sha256"] = c.source.search.fingerprint(
                {k: v for k, v in changed.items() if k != "sha256"}
            )
            with self.assertRaises(ValueError):
                extension.validate_extension(changed, m)
            c.store.write_json(root / "budget.json", {"pending": {"held": {}}})
            with self.assertRaises(ValueError):
                extension.proposal(root, m)

    def test_supervisor_child_uses_same_overlay_without_calling_network(self):
        c = extension.compact
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = dict(sha256="frozen", limits=c.LIMITS, global_limits=c.GLOBAL_LIMITS)
            c.budget.Ledger(root, c.GLOBAL_LIMITS)
            value = extension.proposal(root, m)
            c.store.write_json(root / "experiment.json", m)
            c.store.write_json(root / "budget_extension.json", value)

            def offline_main():
                self.assertEqual(
                    Path(c.__file__).resolve(), Path(extension.__file__).resolve()
                )
                self.assertEqual(sys.argv[1:], ["--dry-run", "--max-jobs", "3"])
                ledger = c.budget.Ledger(root, c.GLOBAL_LIMITS)
                self.assertEqual(ledger.limits["tokens"], extension.GLOBAL_TOKENS)
                return 0

            with (
                patch.object(c, "validate"),
                patch.object(c, "output_path", return_value=root),
                patch.object(c, "main", side_effect=offline_main),
            ):
                self.assertEqual(extension.main(["--dry-run", "--max-jobs", "3"]), 0)
            self.assertFalse((root / "resumption_events.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
