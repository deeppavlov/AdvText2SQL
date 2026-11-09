from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import durable_prompt_store as store
import prompt_comparison_supervisor as supervisor


class SupervisorTests(unittest.TestCase):
    def root(self, tmp):
        root = Path(tmp)
        store.write_json(root / "experiment.json", {"sha256": "frozen"})
        return root

    def test_two_transient_resumes_are_bounded_and_delayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.root(tmp)
            with (
                patch.object(
                    supervisor.preparation,
                    "prepare",
                    side_effect=[
                        "NO_PENDING",
                        "ACCOUNTED: 500",
                        "ACCOUNTED: 503",
                        "ACCOUNTED: 500",
                    ],
                ),
                patch("builtins.print"),
                patch(
                    "socket.create_connection",
                    side_effect=AssertionError("No network in tests"),
                ),
            ):
                from unittest.mock import Mock

                child, sleep = Mock(return_value=1), Mock()
                code = supervisor.supervise(
                    root, filters={}, run_child=child, max_resumes=2, sleep=sleep
                )
                self.assertEqual(code, 1)
                self.assertEqual(child.call_count, 3)
                self.assertEqual([c.args[0] for c in sleep.call_args_list], [60, 120])

    def test_success_after_transient_failure_stops_restarting(self):
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as tmp:
            root = self.root(tmp)
            with (
                patch.object(
                    supervisor.preparation,
                    "prepare",
                    side_effect=["NO_PENDING", "ACCOUNTED: 500"],
                ),
                patch("builtins.print"),
            ):
                child, sleep = Mock(side_effect=[1, 0]), Mock()
                self.assertEqual(
                    supervisor.supervise(
                        root, filters={}, run_child=child, max_resumes=2, sleep=sleep
                    ),
                    0,
                )
                self.assertEqual(child.call_count, 2)
                sleep.assert_called_once_with(60)

    def test_billing_database_or_unidentified_failure_is_not_retried(self):
        from unittest.mock import Mock

        for outcome in (ValueError("402"), "NO_PENDING: no failed request"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as tmp:
                root = self.root(tmp)
                with (
                    patch.object(
                        supervisor.preparation,
                        "prepare",
                        side_effect=["NO_PENDING", outcome],
                    ),
                    patch("builtins.print"),
                ):
                    child, sleep = Mock(return_value=1), Mock()
                    self.assertEqual(
                        supervisor.supervise(
                            root,
                            filters={},
                            run_child=child,
                            max_resumes=2,
                            sleep=sleep,
                        ),
                        1,
                    )
                    child.assert_called_once()
                    sleep.assert_not_called()

    def test_default_does_not_restart(self):
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as tmp:
            root = self.root(tmp)
            with (
                patch.object(
                    supervisor.preparation,
                    "prepare",
                    side_effect=["NO_PENDING", "ACCOUNTED: 500"],
                ),
                patch("builtins.print"),
            ):
                child, sleep = Mock(return_value=1), Mock()
                self.assertEqual(
                    supervisor.supervise(
                        root, filters={}, run_child=child, sleep=sleep
                    ),
                    1,
                )
                child.assert_called_once()
                sleep.assert_not_called()

    def test_restarts_keep_original_job_cap(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as tmp:
            root = self.root(tmp)
            jobs = [
                {"name": f"sql__{method}__r3", "slot": "sql", "method": method}
                for method in ("gepa_zero", "mipro_zero", "textgrad_zero")
            ]
            filters = dict(phase="search", slot="sql", method="all", repeat=3)
            runner = supervisor.capped_runner(
                root, {"jobs": jobs}, filters, 2, ["frozen-launcher"]
            )
            with patch.object(
                supervisor.subprocess, "run", return_value=SimpleNamespace(returncode=0)
            ) as child:
                runner()
                self.assertEqual(child.call_args.args[0][-2:], ["--max-jobs", "2"])
                first = root / "jobs" / jobs[0]["name"]
                first.mkdir(parents=True)
                store.write_json(first / "selection.json", {})
                runner()
                self.assertEqual(child.call_args.args[0][-2:], ["--max-jobs", "1"])
                second = root / "jobs" / jobs[1]["name"]
                second.mkdir(parents=True)
                store.write_json(second / "selection.json", {})
                self.assertEqual(runner(), 0)
                self.assertEqual(child.call_count, 2)


if __name__ == "__main__":
    unittest.main()
