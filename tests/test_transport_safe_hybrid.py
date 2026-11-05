import asyncio
from contextvars import ContextVar
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]

import transport_safe_runtime as runtime  # noqa: E402
import transport_safe_campaign as campaign  # noqa: E402
import static_hybrid_runtime as legacy  # noqa: E402
from run_openrouter_baseline import ProviderUnavailable  # noqa: E402


class BoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_client_error_bypasses_generator_catch_and_checkpoint(self):
        for error in (
            TimeoutError("private-token"),
            AttributeError("null usage"),
            SimpleHTTPError(),
        ):
            with (
                self.subTest(error=type(error).__name__),
                tempfile.TemporaryDirectory() as tmp,
            ):
                context = ContextVar(
                    "question",
                    default={"phase": "question", "question_id": "31", "db_id": "db"},
                )
                meta = {}
                client = runtime.TransportRecordingClient(
                    SimpleNamespace(create=AsyncMock(side_effect=error)),
                    tmp,
                    meta,
                    context,
                )

                class SwallowingTool:
                    async def query(self, _):
                        try:
                            await client.create([])
                        except Exception:
                            return {"status": "error"}
                        return {"status": "success", "query": "SELECT 1"}

                predictions = {}
                with self.assertRaises(ProviderUnavailable):
                    await legacy.predict(
                        [{"question_id": 31, "db_id": "db", "question": "q"}],
                        {"db": SwallowingTool()},
                        predictions,
                        Path(tmp),
                        "ambrosia",
                        context,
                        lambda _: None,
                        sleep=AsyncMock(),
                    )
                self.assertEqual(predictions, {})
                self.assertFalse((Path(tmp) / "query_results.json").exists())
                event_text = (Path(tmp) / "transport_events.jsonl").read_text()
                event = json.loads(event_text)
                self.assertEqual(event["question_id"], "31")
                self.assertFalse(event["checkpoint_written"])
                self.assertNotIn("private-token", event_text)
                self.assertEqual(meta["consecutive_api_errors"], 1)

    async def test_null_and_missing_usage_are_terminal(self):
        for value in (None, SimpleNamespace(content="SELECT 1", usage=None)):
            with tempfile.TemporaryDirectory() as tmp:
                client = runtime.TransportRecordingClient(
                    SimpleNamespace(create=AsyncMock(return_value=value)),
                    tmp,
                    {},
                    ContextVar("q", default={"question_id": "7"}),
                )
                with self.assertRaises(ProviderUnavailable):
                    await client.create([])

    async def test_deadline_cancels_slow_call_without_checkpoint(self):
        async def slow():
            await asyncio.sleep(10)

        with tempfile.TemporaryDirectory() as tmp:
            boundary = runtime.ResponseBoundary(
                SimpleNamespace(create=slow),
                tmp,
                {},
                ContextVar("q", default={}),
                timeout=0.001,
            )
            with self.assertRaises(ProviderUnavailable):
                await boundary.create()
            event = json.loads((Path(tmp) / "transport_events.jsonl").read_text())
            self.assertEqual(event["reason"], "call_timeout")

    async def test_model_format_error_still_recorded_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            value = SimpleNamespace(
                content="{", usage=SimpleNamespace(prompt_tokens=7, completion_tokens=1)
            )
            sdk = SimpleNamespace(create=AsyncMock(return_value=value))
            client = runtime.TransportRecordingClient(
                sdk, tmp, {}, ContextVar("q", default={"phase": "question"})
            )
            self.assertIs(await client.create([]), value)
            self.assertEqual(sdk.create.await_count, 1)
            self.assertFalse((Path(tmp) / "transport_events.jsonl").exists())
            raw = json.loads((Path(tmp) / "raw_responses.jsonl").read_text())
            self.assertEqual(raw["raw_response"], "{")
            self.assertEqual(raw["prompt_tokens"], 7)

    async def test_empty_paid_text_preserves_usage_then_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            value = SimpleNamespace(
                content="", usage=SimpleNamespace(prompt_tokens=8, completion_tokens=0)
            )
            client = runtime.TransportRecordingClient(
                SimpleNamespace(create=AsyncMock(return_value=value)),
                tmp,
                {},
                ContextVar("q", default={}),
            )
            with self.assertRaises(ProviderUnavailable):
                await client.create([])
            raw = json.loads((Path(tmp) / "raw_responses.jsonl").read_text())
            self.assertEqual(raw["prompt_tokens"], 8)


class SimpleHTTPError(Exception):
    status_code = 503


class RecoveryTests(unittest.TestCase):
    def test_recovery_selects_transport_fallback_success_not_all_errors(self):
        log = (
            "12:00:00 [INFO] query() started: a\n"
            "openai.InternalServerError: private response\n[1/3] question=1 status=success\n"
            "12:00:01 [INFO] query() started: b\n"
            "ValueError: invalid SQL\n[2/3] question=2 status=error\n"
            "12:00:02 [INFO] query() started: c\n"
            "json.decoder.JSONDecodeError: bad model JSON\n[3/3] question=3 status=error\n"
        )
        self.assertEqual(
            set(
                campaign.affected_questions(
                    log, {"1": "SELECT 1", "2": "error", "3": "error"}
                )
            ),
            {"1"},
        )
        # Interrupted calls without a saved result are already eligible for resume.
        self.assertEqual(
            campaign.affected_questions(
                "12:00:00 [INFO] query() started: x\nTimeoutError: timeout", {}
            ),
            {},
        )

    def test_last_completed_attempt_controls_audit(self):
        log = (
            "12:00:00 [INFO] query() started: x\nTimeoutError: timeout\n[1/1] question=1 status=error\n"
            "12:01:00 [INFO] query() started: x\n[1/1] question=1 status=success\n"
        )
        self.assertEqual(campaign.affected_questions(log, {"1": "SELECT 1"}), {})

    def test_migration_preserves_original_and_only_invalidates_selected_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "research_results/old"
            output = root / "research_results/new"
            folder = source / "train/hybrid_control_repeat__r1/ambrosia_227"
            folder.mkdir(parents=True)
            original_manifest = {
                "protocol": {"files_sha256": {}},
                "protocol_sha256": "old",
            }
            (source / "experiment.json").write_text(json.dumps(original_manifest))
            (source / "suite_status.json").write_text('{"status":"stopped"}')
            frozen = {
                "protocol_sha256": "old",
                "source_sha256": {},
                "candidates": ["a", "b"],
                "sha256": "ignored",
            }
            (source / "finalists.json").write_text(json.dumps(frozen))
            (folder / "metadata.json").write_text(
                json.dumps(
                    {
                        "configuration": {"protocol_sha256": "old"},
                        "status": "completed",
                        "balance_delta_this_attempt": {"USD": 0.01},
                    }
                )
            )
            (folder / "query_results.json").write_text(
                '{"1":"error","2":"error","3":"SELECT 3"}'
            )
            (folder / "report.json").write_text('{"old":true}')
            (folder / "ambrosia.log").write_text(
                "12:00:00 [INFO] query() started: x\nopenai.InternalServerError: fail\n[1/3] question=1 status=error\n"
                "12:00:01 [INFO] query() started: y\nValueError: SQL\n[2/3] question=2 status=error\n"
            )
            before = {
                p.relative_to(source): p.read_bytes()
                for p in source.rglob("*")
                if p.is_file()
            }
            with (
                patch.object(campaign.campaign, "ROOT", root),
                patch.object(campaign.previous, "validate"),
                patch.object(campaign, "validate"),
                patch.object(campaign, "digest", return_value="hash"),
                patch.object(
                    campaign.campaign,
                    "finalists",
                    return_value=[
                        "baseline_control",
                        "hybrid_control_repeat",
                        "a",
                        "b",
                    ],
                ),
                patch.object(
                    campaign,
                    "finalists",
                    return_value=[
                        "baseline_control",
                        "hybrid_control_repeat",
                        "a",
                        "b",
                    ],
                ),
                patch.object(campaign.campaign, "campaign_report"),
                redirect_stdout(io.StringIO()),
            ):
                campaign.migrate(source, output)
            self.assertTrue((output / "migration_complete.json").exists())
            newfolder = output / folder.relative_to(source)
            self.assertEqual(
                json.loads((newfolder / "query_results.json").read_text()),
                {"2": "error", "3": "SELECT 3"},
            )
            self.assertEqual(
                json.loads((newfolder / "metadata.json").read_text())["status"],
                "recovery_pending",
            )
            self.assertTrue(
                (newfolder / "report.before_transport_recovery.json").exists()
            )
            self.assertEqual(
                json.loads((output / "recovery_plan.json").read_text())["questions"], 1
            )
            for rel, content in before.items():
                self.assertEqual((source / rel).read_bytes(), content)

    def test_no_overwrite_or_ancestor_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "research_results/old"
            source.mkdir(parents=True)
            with patch.object(campaign.campaign, "ROOT", root):
                for target in (source, source / "nested", source.parent):
                    with self.assertRaises(ValueError):
                        campaign.migrate(source, target)


if __name__ == "__main__":
    unittest.main()
