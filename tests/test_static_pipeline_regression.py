"""Audit controls use the real generator/scorer, with external services replaced."""

import json
import os
from pathlib import Path
from copy import deepcopy
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

import static_pipeline_regression as audit  # noqa: E402


def fixture():
    templates = audit.source.all_templates()
    slots = {s: audit.source.slot_parts(s)[1] for s in ("sql", "ambiguity")}
    model = audit.source.pilot.public_model({"LLM_MODEL_NAME": "deepseek-flash", "LLM_API_KEY": "dummy",
                                            "LLM_BASE_URL": "https://api.deepseek.com", "LLM_MODEL_VERSION": "fixture"})
    specs = {}
    for benchmark in audit.FEATURES:
        items = [{"question_id": 1, "db_id": "dummy", "question": "List employee names", "evidence": "",
                  "SQL": "SELECT 987654321 AS protected_gold"},
                 {"question_id": 2, "db_id": "dummy", "question": "List positions", "evidence": "",
                  "SQL": "ambiguous" if benchmark == "ambrosia" else "SELECT 2"}]
        value = audit.source.candidate(slots["sql" if benchmark == "bird" else "ambiguity"])
        specs[benchmark] = {
            "items": items, "features": audit.FEATURES[benchmark], "candidate": value,
            "frozen_prompts": {name: templates[name] for name in (
                "SQL_PROMPT_TEMPLATE", "SQL_PROMPT_BASIC", "TAXONOMY_DETECTION_PROMPT")},
            "cohorts": {s: {"fit": [], "inner_val": []} for s in slots}, "seed_instructions": slots,
            "schema_audit": {"schemas": {b: {"dummy": {"rows": []}} for b in audit.FEATURES}},
            "model": model, "source_files_sha256": {},
            "historical": {str(i["question_id"]): {"question_id": str(i["question_id"]), "db_id": "dummy",
                            "prediction": "SELECT 'DESC' AS value", "score": float(i["SQL"] != "ambiguous")} for i in items},
        }
    return {"sha256": "test-manifest", "env_file": ".env.test", "benchmarks": specs}


def fake_engine(rows=(("DESC",),)):
    engine = MagicMock()
    engine.dialect.name = "postgresql"
    connection = engine.connect.return_value.__enter__.return_value
    connection.execute.return_value.fetchall.return_value = list(rows)
    return engine, connection


def save_rescore(root, manifest, benchmark, records):
    folder = root / benchmark
    folder.mkdir(exist_ok=True)
    audit.store.write_json(audit.rescore_path(root, benchmark), {
        "manifest_sha256": manifest["sha256"], "records": records,
        "records_sha256": audit.fingerprint(records)})


class AuditRegressionTests(unittest.TestCase):
    def test_historical_profiles_are_copied_without_reopening_old_runtime(self):
        with patch.object(socket.socket, "connect", side_effect=AssertionError("No network")):
            for benchmark in audit.FEATURES:
                if not (ROOT / audit.SOURCES[benchmark] / "experiment.json").exists():
                    self.skipTest("Historical full source not present")
                spec = audit.source_snapshot(benchmark)
                self.assertEqual(len(spec["items"]), audit.COUNTS[benchmark])
                self.assertEqual(spec["features"], audit.FEATURES[benchmark])
                self.assertFalse(spec["candidate"]["demos"])
                self.assertEqual(sum(r["score"] for r in spec["historical"].values()),
                                 132 if benchmark == "bird" else 171)

    def test_ambrosia_reevaluation_never_calls_llm_and_errors_are_failures(self):
        m = fixture()
        m["benchmarks"]["ambrosia"]["historical"]["1"]["prediction"] = "error"
        with tempfile.TemporaryDirectory() as tmp, patch.object(socket.socket, "connect", side_effect=AssertionError("No network")), \
                patch("autogen_ext.models.openai.OpenAIChatCompletionClient") as client:
            root = Path(tmp)
            audit.reevaluate(root, m, "ambrosia")
            rows = audit.rescore_records(root, m, "ambrosia")
            self.assertEqual(rows["1"]["score"], 0)
            self.assertEqual(rows["2"]["score"], 0)
            audit.require_reevaluation(root, m, "ambrosia")
            audit.reevaluate(root, m, "ambrosia")
            self.assertEqual(audit.rescore_records(root, m, "ambrosia"), rows)
            client.assert_not_called()

    def test_bird_rescoring_uses_read_only_boundary_and_real_scorer(self):
        m = fixture()
        engine, connection = fake_engine()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, DB_USER="reader", DB_PASS="dummy"), \
                patch("baseline_connection.database_engine", return_value=engine), \
                patch("resilient_hybrid_runtime.validate_live_schemas"), \
                patch("autogen_ext.models.openai.OpenAIChatCompletionClient") as client:
            root = Path(tmp)
            audit.reevaluate(root, m, "bird")
            rows = audit.rescore_records(root, m, "bird")
            self.assertEqual([r["score"] for r in rows.values()], [1.0, 1.0])
            self.assertEqual(connection.begin.return_value.rollback.call_count, 4)
            client.assert_not_called()

    def test_bad_reference_keeps_denominator_and_blocks_paid_generation(self):
        m = fixture()
        m["benchmarks"]["bird"]["items"][0]["SQL"] = "SELECT 'unfinished"
        engine, _ = fake_engine()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, DB_USER="reader", DB_PASS="dummy"), \
                patch("baseline_connection.database_engine", return_value=engine), \
                patch("resilient_hybrid_runtime.validate_live_schemas"), patch.object(audit, "AuditSession") as session:
            root = Path(tmp)
            audit.reevaluate(root, m, "bird")
            rows = audit.rescore_records(root, m, "bird")
            self.assertIsNone(rows["1"]["score"])
            result = audit.summary(m["benchmarks"]["bird"]["items"], rows, "bird")
            self.assertEqual(result["total"], 2)
            self.assertIsNone(result["accuracy"])
            with self.assertRaisesRegex(ValueError, "all references valid"):
                audit.run(root, m, "bird")
            session.assert_not_called()

    def test_infrastructure_failure_is_not_persisted_as_model_error(self):
        from sqlalchemy.exc import OperationalError

        m = fixture()
        engine, connection = fake_engine()
        connection.execute.side_effect = OperationalError("SELECT 1", {}, Exception("connection lost"))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, DB_USER="reader", DB_PASS="dummy"), \
                patch("baseline_connection.database_engine", return_value=engine), \
                patch("resilient_hybrid_runtime.validate_live_schemas"), patch.object(audit.time, "sleep"):
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "no model label saved"):
                audit.reevaluate(root, m, "bird")
            self.assertEqual(audit.rescore_records(root, m, "bird"), {})
            progress = audit.read(root / "bird/reevaluation_progress.json")
            self.assertEqual(progress["failed_question"], "1")
            self.assertEqual(progress["exception_type"], "OperationalError")
            self.assertEqual(progress["stop_error_type"], "RuntimeError")
            self.assertEqual(progress["attempt"], 3)

    def test_connection_retry_is_bounded_and_diagnostics_do_not_expose_exception_text(self):
        from sqlalchemy.exc import OperationalError

        engine, connection = fake_engine()
        error = OperationalError("hidden SQL", {}, SimpleNamespace(sqlstate="08006"))
        connection.execute.side_effect = [error, SimpleNamespace(fetchall=lambda: [(1,)])]
        failures, delays = [], []
        rows = audit.reference_rows(engine, "SELECT 1", on_failure=failures.append, sleep=delays.append)
        self.assertEqual(rows, {(1,)})
        self.assertEqual(delays, [5])
        self.assertEqual(failures[0]["sqlstate"], "08006")
        self.assertTrue(failures[0]["will_retry"])
        self.assertNotIn("hidden SQL", str(failures))
        self.assertEqual(engine.connect.call_count, 2)
        self.assertEqual(connection.begin.return_value.rollback.call_count, 2)

    def test_query_and_statement_timeout_are_not_retried_as_connection_failures(self):
        from sqlalchemy.exc import DBAPIError

        for state in ("42601", "57014"):
            with self.subTest(state=state):
                engine, connection = fake_engine()
                connection.execute.side_effect = DBAPIError("hidden SQL", {}, SimpleNamespace(sqlstate=state))
                events, delays = [], []
                with self.assertRaises(DBAPIError):
                    audit.reference_rows(engine, "SELECT 1", on_failure=events.append, sleep=delays.append)
                self.assertEqual(delays, [])
                self.assertFalse(events[0]["will_retry"])
                self.assertEqual(engine.connect.call_count, 1)

    def test_recovery_preserves_old_files_and_only_carries_scored_answers(self):
        m = fixture()
        m["limits"] = audit.LIMITS
        m["benchmarks"]["bird"]["items"][0]["SQL"] = "SELECT `name` FROM `t`"
        previous = {**deepcopy(m), "version": "static-pipeline-audit-v1"}
        previous.pop("sha256")
        previous["sha256"] = audit.fingerprint(previous)
        rows = {"1": {**m["benchmarks"]["bird"]["historical"]["1"], "status": "scored"},
                "2": {**m["benchmarks"]["bird"]["historical"]["2"], "status": "reference_error",
                      "score": None, "error": "UnsafeSQL"}}
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as tmp:
            parent, target = Path(tmp) / "old", Path(tmp) / "new"
            parent.mkdir()
            target.mkdir()
            audit.store.write_json(parent / "experiment.json", previous)
            save_rescore(parent, previous, "bird", rows)
            before = audit.rescore_path(parent, "bird").read_bytes()
            with patch.object(audit, "verify_recovery_code", return_value={}), \
                    patch.object(socket.socket, "connect", side_effect=AssertionError("No network")):
                audit.recover_reevaluation(parent, target, m, parent)
            result = audit.rescore_records(target, m, "bird")
            self.assertEqual(set(result), {"1"})
            self.assertEqual(audit.rescore_path(parent, "bird").read_bytes(), before)
            journal = audit.read(target / "checkpoint_transfer.json")
            self.assertEqual(journal["benchmarks"]["bird"]["retry_ids"], ["2"])
            self.assertEqual(journal["paid_answers_transferred"], 0)
            self.assertTrue((target / "recovery_source/experiment.json").exists())
            with patch.object(audit, "verify_recovery_code", return_value={}), self.assertRaises(ValueError):
                audit.recover_reevaluation(parent, target, m, parent)

    def test_tampered_and_foreign_reevaluation_cannot_authorize_paid_calls(self):
        m = fixture()
        rows = {q: {**r, "status": "scored"} for q, r in m["benchmarks"]["bird"]["historical"].items()}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_rescore(root, m, "bird", rows)
            audit.require_reevaluation(root, m, "bird")
            rows["1"]["prediction"] = "SELECT 999"
            save_rescore(root, m, "bird", rows)
            with self.assertRaisesRegex(ValueError, "frozen prediction"):
                audit.require_reevaluation(root, m, "bird")
            value = audit.read(audit.rescore_path(root, "bird"))
            value["manifest_sha256"] = "foreign-model"
            audit.store.write_json(audit.rescore_path(root, "bird"), value)
            with self.assertRaisesRegex(ValueError, "identity/hash"):
                audit.require_reevaluation(root, m, "bird")

    def test_real_generation_keeps_features_prompt_budget_and_checkpoint(self):
        import baseline_connection
        from adv_text2sql.mcp_servers.text2sql_tool.src import text2sql_implementation as impl

        m = fixture()
        with tempfile.TemporaryDirectory() as tmp, audit.shared.durable_runtime(), patch.dict(os.environ):
            root = Path(tmp)
            env = root / "task.env"
            env.write_text("LLM_MODEL_NAME=deepseek-flash\nLLM_API_KEY=dummy\nLLM_BASE_URL=https://api.deepseek.com\nLLM_MODEL_VERSION=fixture\n")
            m["env_file"] = str(env)
            session = audit.AuditSession(root, audit.view(m, "bird"), audit.job_for("bird"), "confirm")
            session.runner.get_loop()  # Create the Windows self-pipe before network denial.
            engine, _ = fake_engine()
            session.engines = {"dummy": engine}
            session.golds = {"1": {("DESC",)}}
            raw = SimpleNamespace(create=AsyncMock(side_effect=[
                SimpleNamespace(content='{"is_ambiguous": false, "ambiguity_type": null, "reason": null}',
                                usage=SimpleNamespace(prompt_tokens=30, completion_tokens=10)),
                SimpleNamespace(content="SELECT 'DESC' AS value", usage=SimpleNamespace(prompt_tokens=30, completion_tokens=10))]),
                close=AsyncMock())

            async def build(items, generator, client, progress):
                self.assertEqual(items, m["benchmarks"]["bird"]["items"])
                self.assertEqual([n for n in range(1, 37) if getattr(impl, f"FEAT_{n}")], audit.FEATURES["bird"])
                tool = object.__new__(generator)
                tool.system_prompt = "PostgreSQL schema: t(value TEXT)"
                tool.db_schema = "t(value TEXT)"
                tool.relationships_str = tool.column_stats_str = ""
                tool.engine, tool.llm_client = engine, client
                return {"dummy": tool}

            try:
                with patch("autogen_ext.models.openai.OpenAIChatCompletionClient", return_value=raw), \
                        patch.object(baseline_connection, "build_tools", side_effect=build), \
                        patch.object(socket.socket, "connect", side_effect=AssertionError("No network")), \
                        patch.object(audit.runtime.asyncio, "sleep", AsyncMock()):
                    item = m["benchmarks"]["bird"]["items"][0]
                    value = m["benchmarks"]["bird"]["candidate"]
                    answer = session.evaluate(item, value, benchmark="bird")
                    self.assertEqual(answer["prediction"], "SELECT 'DESC' AS value")
                    self.assertEqual(answer["score"], 1)
                    self.assertEqual(session.evaluate(item, value, benchmark="bird"), answer)
                    self.assertEqual(raw.create.await_count, 2)
                    self.assertNotIn("987654321", str(raw.create.call_args_list))
                    self.assertEqual(session.ledger.data["teacher_calls"], 0)
                    self.assertEqual(session.ledger.data["task_calls"], 2)
                    with self.assertRaises(RuntimeError):
                        session.reflect("No teacher")
            finally:
                session.close()

    def test_report_separates_evaluator_and_generation_changes(self):
        m = fixture()
        spec = m["benchmarks"]["bird"]
        rescored = {q: {**r, "status": "scored", "score": 0.0 if q == "1" else 1.0}
                    for q, r in spec["historical"].items()}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_rescore(root, m, "bird", rescored)
            folder = root / "bird/jobs/static_control"
            (folder / "confirm").mkdir(parents=True)
            audit.store.write_json(folder / "confirm/metadata.json", {"manifest_sha256": m["sha256"]})
            audit.store.write_json(folder / "evaluations.json", {
                audit.key(spec["candidate"], i, "bird"): spec["historical"][str(i["question_id"])] for i in spec["items"]})
            result = audit.report(root, m)["benchmarks"]["bird"]
            self.assertEqual(result["paired"]["evaluator_B_vs_A"]["harmed_ids"], ["1"])
            self.assertEqual(result["paired"]["generation_C_vs_B"]["rescued_ids"], ["1"])
            self.assertFalse(json.loads((root / "comparison.json").read_text())["deployed"])


if __name__ == "__main__":
    unittest.main()
