from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")]

import resilient_hybrid_runtime as runtime  # noqa: E402
import resilient_hybrid_campaign as campaign  # noqa: E402
import static_hybrid_search as search  # noqa: E402


def manifest():
    return {
        "protocol": {
            "version": "guarded-hybrid-v3",
            "model": {},
            "connection_preflight": {"version": "resilient-catalog-v1"},
        },
        "protocol_sha256": "test",
        "cases": [{"name": "baseline_control__r1", "features": []}],
        "cohorts": {
            "train": {b: [{"question_id": 1, "db_id": "x"}] for b in search.BENCHMARKS}
        },
        "schema_audit": {
            "blocked_questions": {b: {} for b in search.BENCHMARKS},
            "schemas": {b: {} for b in search.BENCHMARKS},
        },
    }


class CatalogRetryTests(unittest.TestCase):
    def run_preflight(self, engine, *, schemas=None, attempts=3):
        inspector = MagicMock()
        inspector.get_table_names.return_value = ["things"]
        inspector.get_columns.return_value = [{"name": "id", "type": "INTEGER"}]
        progress, sleeps = [], []
        with (
            patch.dict("os.environ", {"DB_USER": "dummy", "DB_PASS": "fake-secret"}),
            patch("baseline_connection.database_engine", return_value=engine),
            patch("sqlalchemy.inspect", return_value=inspector),
            redirect_stdout(io.StringIO()),
        ):
            runtime.validate_live_schemas(
                [{"db_id": "x"}],
                schemas or {"x": {"rows": [["things", "id", "INTEGER"]]}},
                attempts=attempts,
                retry_delay=0,
                sleep=sleeps.append,
                progress=progress.append,
            )
        return progress, sleeps, inspector

    def test_transient_connection_failure_retries_and_uses_read_only_single_connection(
        self,
    ):
        from sqlalchemy.exc import OperationalError

        engine = MagicMock()
        connection = MagicMock()
        context = MagicMock()
        context.__enter__.return_value = connection
        engine.connect.side_effect = [
            OperationalError(None, None, Exception("timeout expired")),
            context,
        ]
        events, sleeps, inspector = self.run_preflight(engine)
        self.assertEqual(engine.connect.call_count, 2)
        self.assertEqual(engine.dispose.call_count, 2)
        self.assertEqual(sleeps, [0])
        self.assertEqual(events[-1]["event"], "schema_verified")
        self.assertTrue(any(e.get("will_retry") for e in events))
        connection.exec_driver_sql.assert_called_once_with("SET TRANSACTION READ ONLY")
        inspector.get_columns.assert_called_once_with("things", schema="public")

    def test_schema_mismatch_is_not_retried_or_ignored(self):
        engine = MagicMock()
        with self.assertRaisesRegex(ValueError, "Live schema changed"):
            self.run_preflight(
                engine, schemas={"x": {"rows": [["different", "id", "INTEGER"]]}}
            )
        self.assertEqual(engine.connect.call_count, 1)
        engine.dispose.assert_called_once()

    def test_auth_failure_is_sanitized_and_not_retried(self):
        from sqlalchemy.exc import OperationalError

        engine = MagicMock()
        engine.connect.side_effect = OperationalError(
            None, None, Exception("password authentication failed: fake-secret")
        )
        with self.assertRaisesRegex(
            RuntimeError, "authentication_failed after 1"
        ) as raised:
            self.run_preflight(engine)
        self.assertNotIn("fake-secret", str(raised.exception))
        self.assertEqual(engine.connect.call_count, 1)

    def test_retry_budget_is_finite(self):
        from sqlalchemy.exc import OperationalError

        engine = MagicMock()
        engine.connect.side_effect = OperationalError(
            None, None, Exception("timeout expired")
        )
        with self.assertRaisesRegex(RuntimeError, "after 3 attempt"):
            self.run_preflight(engine)
        self.assertEqual(engine.connect.call_count, 3)
        self.assertEqual(engine.dispose.call_count, 3)

    def test_unexpected_catalog_error_is_sanitized(self):
        engine = MagicMock()
        engine.connect.side_effect = RuntimeError("postgresql://fake-secret@host/db")
        with self.assertRaisesRegex(
            RuntimeError, "Catalog inspection failed"
        ) as raised:
            self.run_preflight(engine)
        self.assertNotIn("fake-secret", str(raised.exception))
        self.assertEqual(engine.connect.call_count, 1)

    def test_sqlstate_missing_db_and_permission_are_not_retryable(self):
        for code, expected in (
            ("3D000", "database_missing"),
            ("42501", "permission_denied"),
            ("28P01", "authentication_failed"),
        ):
            error = RuntimeError("secret text should never be returned")
            error.sqlstate = code
            self.assertEqual(runtime.connection_reason(error), (expected, False))


class MonitorRegressionTests(unittest.TestCase):
    def test_old_manifest_cannot_be_reused_by_the_new_launcher(self):
        with patch.object(campaign, "ORIGINAL_VALIDATE"):
            with self.assertRaisesRegex(ValueError, "old campaign folder"):
                campaign.validate({"protocol": {}})

    def test_wrapper_restores_frozen_campaign_functions_after_error(self):
        original = (
            campaign.campaign.DEFAULT_OUTPUT,
            campaign.campaign.make_manifest,
            campaign.campaign.run_jobs,
            campaign.campaign.validate,
        )
        with patch.object(campaign.campaign, "main", side_effect=RuntimeError("stop")):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                campaign.main()
        current = (
            campaign.campaign.DEFAULT_OUTPUT,
            campaign.campaign.make_manifest,
            campaign.campaign.run_jobs,
            campaign.campaign.validate,
        )
        self.assertEqual(current, original)

    def test_child_failure_before_metadata_preserves_original_error_and_monitor_diagnostics(
        self,
    ):
        m = manifest()
        args = SimpleNamespace(
            phase="controls",
            round=1,
            env_file="unused",
            stall_seconds=900,
            error_limit=8,
        )
        child = MagicMock()
        child.poll.return_value = 1
        child.returncode = 1
        with tempfile.TemporaryDirectory(
            dir=ROOT / "research_results", prefix=".retry_test_"
        ) as tmp:
            output = Path(tmp)
            self.assertTrue(output.resolve().is_relative_to(ROOT / "research_results"))

            def start(command, **kwargs):
                self.assertTrue(
                    search.run_folder(
                        output, "train", m["cases"][0]["name"], "bird"
                    ).is_dir()
                )
                self.assertTrue(
                    any("resilient_hybrid_runtime.py" in a for a in command)
                )
                return child

            with (
                patch.object(campaign.subprocess, "Popen", side_effect=start),
                patch.object(campaign.campaign, "campaign_report"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "Benchmark exited with code 1"
                ):
                    campaign.run_jobs(output, m, "train", m["cases"], args)
            status = json.loads((output / "suite_status.json").read_text())
            self.assertEqual(status["error_type"], "RuntimeError")
            self.assertTrue((output / "train/monitor_stop.json").exists())


class PreflightProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_preflight_failure_preserves_checkpoint_history_and_never_reaches_llm_runtime(
        self,
    ):
        m = manifest()
        with tempfile.TemporaryDirectory(
            dir=ROOT / "research_results", prefix=".retry_runtime_"
        ) as tmp:
            output = Path(tmp)
            self.assertTrue(output.resolve().is_relative_to(ROOT / "research_results"))
            (output / "experiment.json").write_text(json.dumps(m))
            case = m["cases"][0]
            folder = search.run_folder(output, "train", case["name"], "bird")
            folder.mkdir(parents=True)
            checkpoint = b'{"1": "SELECT 1"}'
            (folder / "query_results.json").write_bytes(checkpoint)
            existing = {
                "configuration": search.identity(m, "train", case, "bird"),
                "status": "failed",
                "phase": "evaluating",
                "balance_before": {"preserve": True},
            }
            (folder / "metadata.json").write_text(json.dumps(existing))
            args = SimpleNamespace(
                search=str(output),
                env_file="unused",
                scope="train",
                case=case["name"],
                benchmark="bird",
            )
            model_runtime = AsyncMock()
            with (
                patch.object(runtime.guarded, "validate"),
                patch.object(runtime.guarded, "load_dotenv"),
                patch.object(
                    runtime,
                    "validate_live_schemas",
                    side_effect=RuntimeError("preflight timeout"),
                ),
                patch.object(runtime.guarded.legacy, "run", model_runtime),
            ):
                with self.assertRaisesRegex(RuntimeError, "preflight timeout"):
                    await runtime.run(args)
            model_runtime.assert_not_awaited()
            self.assertEqual((folder / "query_results.json").read_bytes(), checkpoint)
            status = json.loads((folder / "metadata.json").read_text())
            self.assertEqual(status["status"], "preflight_failed")
            self.assertEqual(status["saved"], 1)
            history = [
                json.loads(s)
                for s in (folder / "attempt_history.jsonl").read_text().splitlines()
            ]
            self.assertEqual(history[0], existing)


if __name__ == "__main__":
    unittest.main()
