"""Connection supervisor never edits frozen checkpoints, SQL or scoring rules."""

from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from sqlalchemy.exc import DBAPIError  # noqa: E402
import resume_static_reevaluation as resume  # noqa: E402


def failure(code, message="private connection detail"):
    original = Exception(message)
    original.sqlstate = code
    return DBAPIError("SELECT 1", {}, original)


class ProbeTests(unittest.TestCase):
    def probe(self, error=None):
        connection = MagicMock()
        connection.execute.return_value.scalar.return_value = 1
        if error is not None:
            connection.execute.side_effect = error

        @contextmanager
        def boundary(engine, sql):
            self.assertEqual(sql, "SELECT 1")
            yield connection

        with (
            patch.dict("os.environ", {"DB_USER": "reader", "DB_PASS": "dummy"}),
            patch("baseline_connection.database_engine") as engine,
            patch("adv_text2sql.sql_safety.read_only_connection", boundary),
        ):
            resume.probe_database("db")
        return connection, engine

    def test_probe_executes_only_read_only_select_one(self):
        connection, engine = self.probe()
        self.assertEqual(str(connection.execute.call_args.args[0]), "SELECT 1")
        engine.return_value.dispose.assert_called_once()

    def test_only_connection_failures_are_retryable(self):
        for code, message, retryable in [
            ("08006", "connection lost", True),
            (None, "connection timed out", True),
            (None, "connection refused", True),
            ("57014", "statement timeout", False),
            ("28P01", "password authentication failed", False),
            ("42501", "permission denied", False),
            ("3D000", "database does not exist", False),
            (None, "unrecognized failure", False),
        ]:
            with self.subTest(code=code, message=message):
                with self.assertRaises(resume.AccessFailure) as caught:
                    self.probe(failure(code, message))
                self.assertEqual(caught.exception.retryable, retryable)
                self.assertEqual(caught.exception.sqlstate, code)
                self.assertNotIn("private", str(caught.exception))


class TunnelTests(unittest.TestCase):
    def test_external_listener_is_never_killed(self):
        tunnel = resume.Tunnel(MagicMock())
        with (
            patch.object(resume, "port_open", return_value=True),
            patch.object(resume.subprocess, "Popen") as create,
        ):
            tunnel.start()
            tunnel.close()
        self.assertTrue(tunnel.borrowed)
        create.assert_not_called()

    def test_own_ssh_uses_batch_mode_and_bounded_cleanup(self):
        process = MagicMock()
        process.poll.return_value = None
        tunnel = resume.Tunnel(MagicMock())
        with (
            patch.object(resume, "port_open", side_effect=[False, True]),
            patch.object(resume.shutil, "which", return_value="ssh"),
            patch.object(resume.subprocess, "Popen", return_value=process) as create,
        ):
            tunnel.start()
            tunnel.close()
        command = create.call_args.args[0]
        self.assertIn("BatchMode=yes", command)
        self.assertIn("ExitOnForwardFailure=yes", command)
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=5)


class SupervisorTests(unittest.TestCase):
    def run_case(
        self, *, probes, codes, borrowed=False, max_restarts=2, complete=False
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint-sentinel.json"
            checkpoint.write_text('{"saved":164,"score":0}')
            before = checkpoint.read_bytes()
            manifest = {
                "sha256": "fixture",
                "benchmarks": {
                    "bird": {
                        "items": [
                            {"question_id": 1, "db_id": "dummy"},
                            {"question_id": 2, "db_id": "dummy"},
                        ]
                    }
                },
            }
            records = {"1": {"status": "scored"}}
            if complete:
                records["2"] = {"status": "scored"}
            tunnel = MagicMock()
            tunnel.borrowed = borrowed
            tunnel.died.return_value = False
            with (
                patch.object(resume.audit, "rescore_records", return_value=records),
                patch.object(resume.audit, "validate"),
                patch.object(resume, "Tunnel", return_value=tunnel) as tunnels,
                patch.object(resume, "probe_database", side_effect=probes) as probe,
                patch.object(
                    resume.subprocess,
                    "run",
                    side_effect=[SimpleNamespace(returncode=c) for c in codes],
                ) as worker,
                patch.object(resume.time, "sleep") as sleep,
            ):
                code = resume.supervise(root, manifest, max_restarts=max_restarts)
                self.assertEqual(checkpoint.read_bytes(), before)
                journal = root / "connection_resume.json"
                state = json.loads(journal.read_text()) if journal.exists() else None
                return code, state, worker, probe, tunnel, tunnels, sleep

    def test_database_probe_precedes_original_reevaluation_worker(self):
        code, state, worker, probe, tunnel, _, _ = self.run_case(
            probes=[None], codes=[0]
        )
        self.assertEqual(code, 0)
        self.assertEqual(state["status"], "completed")
        command = worker.call_args.args[0]
        self.assertIn("reevaluate", command)
        self.assertNotIn("run", command)
        self.assertNotIn("--prepare", command)
        self.assertEqual(state["llm_calls"], 0)
        probe.assert_called_once_with("dummy")
        tunnel.close.assert_called_once()

    def test_worker_restarts_only_after_actual_connection_loss(self):
        broken = resume.AccessFailure("connection_closed", retryable=True)
        code, state, worker, probe, tunnel, _, sleep = self.run_case(
            probes=[None, broken, None], codes=[1, 0]
        )
        self.assertEqual(code, 0)
        self.assertEqual(worker.call_count, 2)
        self.assertEqual(len(state["attempts"]), 2)
        self.assertEqual(state["attempts"][0]["reason"], "connection_closed")
        self.assertEqual(tunnel.close.call_count, 2)
        sleep.assert_called_once_with(3)

    def test_healthy_db_after_worker_failure_never_retries_query(self):
        code, state, worker, _, _, _, sleep = self.run_case(
            probes=[None, None], codes=[1]
        )
        self.assertEqual(code, 1)
        self.assertEqual(state["status"], "worker_error_with_healthy_connection")
        worker.assert_called_once()
        sleep.assert_not_called()

    def test_bad_external_tunnel_is_not_replaced_or_killed(self):
        broken = resume.AccessFailure("connection_closed", retryable=True)
        code, state, worker, _, _, _, sleep = self.run_case(
            probes=[broken], codes=[], borrowed=True
        )
        self.assertEqual(code, 1)
        self.assertEqual(state["status"], "external_tunnel_unavailable")
        worker.assert_not_called()
        sleep.assert_not_called()

    def test_authentication_and_statement_timeout_never_loop(self):
        for reason, sqlstate in [
            ("authentication_failed", "28P01"),
            ("connection_or_statement_timeout", "57014"),
        ]:
            with self.subTest(sqlstate=sqlstate):
                code, state, worker, _, _, _, sleep = self.run_case(
                    probes=[
                        resume.AccessFailure(reason, retryable=False, sqlstate=sqlstate)
                    ],
                    codes=[],
                )
                self.assertEqual(code, 1)
                self.assertEqual(len(state["attempts"]), 1)
                worker.assert_not_called()
                sleep.assert_not_called()

    def test_restarts_are_bounded(self):
        broken = resume.AccessFailure("connection_closed", retryable=True)
        code, state, worker, _, tunnel, _, sleep = self.run_case(
            probes=[broken, broken, broken], codes=[]
        )
        self.assertEqual(code, 1)
        self.assertEqual(len(state["attempts"]), 3)
        self.assertEqual(tunnel.close.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        worker.assert_not_called()

    def test_completed_coverage_needs_no_network_or_worker(self):
        code, state, worker, probe, _, tunnels, _ = self.run_case(
            probes=[], codes=[], complete=True
        )
        self.assertEqual(code, 0)
        self.assertIsNone(state)
        worker.assert_not_called()
        probe.assert_not_called()
        tunnels.assert_not_called()


if __name__ == "__main__":
    unittest.main()
