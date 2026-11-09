import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import durable_prompt_store as store
import prompt_comparison as campaign
from prompt_pilot_budget import Ledger
from prompt_comparison_runtime import DualLedger
import recover_prompt_transport as recovery


class TransportRecoveryTests(unittest.TestCase):
    def fixture(self, root, code=503):
        job = "sql__textgrad_zero__r2"
        phase = root / "jobs" / job / "search"
        phase.mkdir(parents=True)
        m = {"version": campaign.VERSION, "jobs": [{"name": job}]}
        m["sha256"] = campaign.search.fingerprint(m)
        store.write_json(root / "experiment.json", m)
        store.write_json(
            phase / "metadata.json",
            {
                "status": "stopped",
                "manifest_sha256": m["sha256"],
                "transport_failure": {
                    "reason": "client_exception",
                    "status_code": code,
                    "question_id": "260",
                    "checkpoint_written": False,
                },
            },
        )
        store.write_json(
            phase.parent / "evaluations.json", {"preserved": {"score": 1.0}}
        )
        local, global_ = (
            Ledger(phase, campaign.LIMITS),
            Ledger(root, campaign.GLOBAL_LIMITS),
        )
        pair = DualLedger(local, global_, job)
        pair.reserve("task", ["request"], 4096)
        return job

    def test_503_recovery_preserves_results_and_is_idempotent(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = self.fixture(root)
            original = (root / "jobs" / job / "evaluations.json").read_bytes()
            path, journal = recovery.build_journal(root, job)
            with patch.object(recovery.recovery, "write_json", store.write_json):
                self.assertTrue(recovery.recovery.apply_journal(root, path, journal))
                self.assertFalse(
                    recovery.recovery.apply_journal(
                        root, path, json.loads(path.read_bytes())
                    )
                )
            self.assertEqual(
                original, (root / "jobs" / job / "evaluations.json").read_bytes()
            )
            for entry in journal["patches"]:
                value = json.loads((root / entry["path"]).read_bytes())
                self.assertEqual(value["task_calls"], entry["before"]["task_calls"])
                self.assertEqual(
                    value["prompt_tokens"], entry["before"]["prompt_tokens"]
                )
                self.assertEqual(
                    value["tokens"],
                    entry["before"]["tokens"] + journal["unknown_usage_budget_tokens"],
                )
                self.assertFalse(value["pending"])
                self.assertEqual(value["unknown_task_calls"], 1)

    def test_auth_billing_and_bad_request_errors_are_not_auto_recovered(self):
        for code in (400, 401, 402, 403, 404, None):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                job = self.fixture(root, code)
                with self.assertRaises(ValueError):
                    recovery.build_journal(root, job)

    def test_mismatched_pending_ledgers_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = self.fixture(root)
            global_ = json.loads((root / "budget.json").read_bytes())
            next(iter(global_["pending"].values()))["reserved"] += 1
            store.write_json(root / "budget.json", global_)
            with self.assertRaises(ValueError):
                recovery.build_journal(root, job)

    def test_resume_preparation_is_offline_and_idempotent(self):
        import prepare_prompt_resume as resume

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = self.fixture(root)
            # Fields used only for matching the user's explicitly selected job.
            manifest = json.loads((root / "experiment.json").read_bytes())
            manifest["jobs"][0].update(slot="sql", method="textgrad_zero", seed=43)
            manifest["sha256"] = campaign.search.fingerprint(
                {k: v for k, v in manifest.items() if k != "sha256"}
            )
            store.write_json(root / "experiment.json", manifest)
            meta_path = root / "jobs" / job / "search/metadata.json"
            meta = json.loads(meta_path.read_bytes())
            meta["manifest_sha256"] = manifest["sha256"]
            store.write_json(meta_path, meta)
            original = (root / "jobs" / job / "evaluations.json").read_bytes()
            self.assertIn("ACCOUNTED", resume.prepare(root, slot="sql", repeat=2))
            budget = (root / "budget.json").read_bytes()
            self.assertIn("NO_PENDING", resume.prepare(root, slot="sql", repeat=2))
            self.assertEqual(budget, (root / "budget.json").read_bytes())
            self.assertEqual(
                original, (root / "jobs" / job / "evaluations.json").read_bytes()
            )

    def test_resume_preparation_rejects_billing_and_other_jobs(self):
        import prepare_prompt_resume as resume

        for code, repeat in ((402, 2), (503, 3)):
            with (
                self.subTest(code=code, repeat=repeat),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                self.fixture(root, code)
                before = (root / "budget.json").read_bytes()
                with self.assertRaises(ValueError):
                    resume.prepare(root, repeat=repeat)
                self.assertEqual(before, (root / "budget.json").read_bytes())

    def test_explicit_billing_accounting_preserves_measured_usage(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = self.fixture(root, 402)
            path, journal = recovery.build_journal(
                root, job, account_billing_error=True
            )
            self.assertTrue(journal["billing_error_accounting_only"])
            self.assertFalse(journal["paid_requests_started"])
            with patch.object(recovery.recovery, "write_json", store.write_json):
                recovery.recovery.apply_journal(root, path, journal)
                self.assertFalse(
                    recovery.recovery.apply_journal(
                        root, path, json.loads(path.read_bytes())
                    )
                )
            for entry in journal["patches"]:
                data = json.loads((root / entry["path"]).read_bytes())
                self.assertFalse(data["pending"])
                self.assertEqual(data["task_calls"], entry["before"]["task_calls"])
                self.assertEqual(
                    data["prompt_tokens"], entry["before"]["prompt_tokens"]
                )
                self.assertEqual(
                    data["tokens"],
                    entry["before"]["tokens"] + journal["unknown_usage_budget_tokens"],
                )
            with self.assertRaises(ValueError):
                recovery.build_journal(root, job)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = self.fixture(root, 401)
            with self.assertRaises(ValueError):
                recovery.build_journal(root, job, account_billing_error=True)


if __name__ == "__main__":
    unittest.main()
