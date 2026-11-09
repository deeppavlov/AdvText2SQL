import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prompt_pilot as pilot
from prompt_pilot_budget import AccountingError, BudgetStop, Ledger
import prompt_pilot_runtime as runtime


def mini_manifest():
    def item(q, db):
        return {
            "question_id": q,
            "db_id": db,
            "question": "How many entries exist?",
            "evidence": "",
            "SQL": "SELECT COUNT(*) FROM entries",
            "difficulty": "simple",
        }

    return {
        "sha256": "identity",
        "seed_instruction": "Use the supplied schema carefully.",
        "cohorts": {
            "fit": [item(1, "fit_db")],
            "inner_val": [item(2, "val_db")],
            "dev": {"bird": [item(3, "dev_db")], "ambrosia": [item(4, "amb_db")]},
        },
        "schema_audit": {
            "schemas": {"bird": {"fit_db": {"rows": [["entries", "id", "TEXT"]]}}}
        },
    }


class PromptContractTests(unittest.TestCase):
    def test_seed_is_exact_noop_and_question_is_protected(self):
        seed = pilot.slot_parts()[1]
        strict = pilot.templates()["SQL_PROMPT_TEMPLATE"]
        self.assertEqual(pilot.compile_template(seed, "hybrid"), strict)
        question = "Show {literal} values"
        rendered = pilot.compile_template(
            "Use only necessary columns.", "hybrid"
        ).format(user_query=question, sql_dialect="PostgreSQL")
        self.assertIn("Запрос: " + question, rendered)
        self.assertIn("PostgreSQL", rendered)
        self.assertTrue(rendered.endswith("SQL запрос:"))
        self.assertEqual(
            pilot.compile_template(seed, "baseline", original=True),
            pilot.templates()["SQL_PROMPT_BASIC"],
        )

    def test_protected_slots_and_database_specific_instructions_rejected(self):
        for value in ("", "x" * 6001, "Use {user_query}", "```sql", "Запрос: example"):
            with self.subTest(value=value[:20]), self.assertRaises(ValueError):
                pilot.check_candidate(value)
        with self.assertRaises(ValueError):
            pilot.check_candidate("Always use fit_db", mini_manifest())
        with self.assertRaises(ValueError):
            pilot.check_candidate(
                "For this problem use SELECT id FROM entries", mini_manifest()
            )

    def test_real_generator_seed_messages_unchanged_and_ambiguity_prompt_untouched(
        self,
    ):
        from unittest.mock import AsyncMock

        sys.path.insert(0, str(pilot.ROOT / "src"))
        from adv_text2sql.mcp_servers.text2sql_tool.src import (
            text2sql_implementation as impl,
        )

        fake = SimpleNamespace(
            system_prompt="Frozen schema",
            _llm_call_with_retry=AsyncMock(
                return_value=SimpleNamespace(content="SELECT 1")
            ),
            _strip_sql_comments=lambda s: s,
            _validate_sql=lambda s: True,
            _get_accessed_tables=lambda s: [],
        )
        seed = pilot.slot_parts()[1]
        taxonomy = impl.TAXONOMY_DETECTION_PROMPT
        with runtime.settings(impl, [14, 32]):
            result = asyncio.run(
                impl.Text2SQLGenerator.generate_sql(fake, "Question with {braces}")
            )
            self.assertEqual(result["status"], "success")
            old = fake._llm_call_with_retry.call_args.args[0]
        with runtime.settings(impl, [14, 32], seed):
            asyncio.run(
                impl.Text2SQLGenerator.generate_sql(fake, "Question with {braces}")
            )
            new = fake._llm_call_with_retry.call_args.args[0]
            self.assertEqual(impl.TAXONOMY_DETECTION_PROMPT, taxonomy)
        self.assertEqual(
            [(m.type, m.content) for m in old], [(m.type, m.content) for m in new]
        )

    def test_output_is_dedicated_and_bounded(self):
        for value in (
            "../outside",
            "research_results",
            "research_results/deepseek_full_static_20261103",
            "research_results/prompt_optimization_x/nested",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pilot.output_path(value)
        self.assertEqual(
            pilot.output_path(pilot.DEFAULT_OUTPUT).parent,
            pilot.ROOT / "research_results",
        )

    def test_same_unordered_result_normalization_as_bird(self):
        rows = [(1, [2, 3], {"a": 4}), (1, [2, 3], {"a": 4})]
        self.assertEqual(runtime.normalize(rows), {(1, (2, 3), (("a", 4),))})


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.time = 0.0
        self.limits = {
            "task_calls": 2,
            "teacher_calls": 1,
            "tokens": 10000,
            "active_seconds": 10,
            "examples": 1,
        }

    def tearDown(self):
        self.tmp.cleanup()

    def ledger(self):
        return Ledger(self.folder, self.limits, clock=lambda: self.time)

    def test_reservations_survive_failure_and_resume(self):
        a = self.ledger()
        a.reserve("task", ["hello"], 4000)
        b = self.ledger()
        b.reserve("task", ["hello"], 1)
        with self.assertRaises(BudgetStop):
            b.reserve("task", ["hello"], 1)
        self.assertEqual(b.data["task_calls"], 2)
        self.assertEqual(len(b.data["pending"]), 2)

    def test_tokens_checked_before_call_and_settled_from_usage(self):
        a = self.ledger()
        ticket = a.reserve("teacher", ["hello"], 10)
        a.settle(ticket, 30, 10)
        self.assertEqual(a.data["tokens"], 40)
        self.assertEqual(a.data["teacher_tokens"], 40)
        self.assertFalse(a.data["pending"])
        with self.assertRaises(BudgetStop):
            a.reserve("task", ["hello"], 10000)
        self.assertEqual(a.data["task_calls"], 0)

    def test_invalid_usage_keeps_reservation(self):
        a = self.ledger()
        ticket = a.reserve("task", [], 1)
        with self.assertRaises(AccountingError):
            a.settle(ticket, True, 1)
        self.assertIn(ticket, a.data["pending"])

    def test_unexpected_usage_overflow_is_not_a_normal_budget_completion(self):
        a = self.ledger()
        ticket = a.reserve("task", [], 1)
        with self.assertRaises(AccountingError):
            a.settle(ticket, 2000, 3000)
        self.assertEqual(a.data["tokens"], 5000)

    def test_question_retry_does_not_consume_new_example_slot(self):
        a = self.ledger()
        a.example("same-question-and-candidate")
        self.ledger().example("same-question-and-candidate")
        with self.assertRaises(BudgetStop):
            a.example("different-question")

    def test_active_time_does_not_reset_on_resume(self):
        a = self.ledger()
        self.time = 6
        a.save()
        b = self.ledger()
        self.time = 11
        with self.assertRaises(BudgetStop):
            b.reserve("task", [], 1)
        self.assertEqual(b.data["task_calls"], 0)


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pilot.load_gepa()

    def test_dev_or_mutated_question_never_reaches_task_model(self):
        m = mini_manifest()
        with patch.object(runtime.Session, "evaluate") as evaluate:
            adapter = runtime.Adapter(runtime.Session.__new__(runtime.Session), m)
            bad = copy.deepcopy(m["cohorts"]["fit"][0])
            bad["SQL"] = "SELECT 123"
            for batch in ([bad], m["cohorts"]["dev"]["bird"]):
                with self.assertRaises(ValueError):
                    adapter.evaluate(batch, {"sql_rules": m["seed_instruction"]})
            evaluate.assert_not_called()

    def test_inner_validation_never_reaches_teacher(self):
        from gepa.core.adapter import EvaluationBatch

        m = mini_manifest()
        adapter = runtime.Adapter(None, m)
        result = {"prediction": "SELECT 1", "score": 0, "error": None}
        batch = EvaluationBatch(
            [result], [0], [{"item": m["cohorts"]["inner_val"][0], "result": result}]
        )
        with self.assertRaises(ValueError):
            adapter.make_reflective_dataset({}, batch, ["sql_rules"])

    def test_real_gepa_engine_works_with_fake_models_and_no_network(self):
        m = mini_manifest()

        class Fake:
            def evaluate(self, item, instruction):
                return {
                    "prediction": "SELECT 1",
                    "score": float("Better" in instruction),
                    "error": None,
                }

        adapter = runtime.Adapter(Fake(), m)
        reflection_prompts = []

        def reflect(prompt):
            reflection_prompts.append(prompt)
            return "```\nBetter general SQL rules.\n```"

        gepa = pilot.load_gepa()
        with patch(
            "socket.socket.connect", side_effect=AssertionError("No network allowed")
        ):
            result = gepa.optimize(
                seed_candidate={"sql_rules": m["seed_instruction"]},
                trainset=m["cohorts"]["fit"],
                valset=m["cohorts"]["inner_val"],
                adapter=adapter,
                reflection_lm=reflect,
                reflection_prompt_template=runtime.REFLECTION,
                reflection_minibatch_size=1,
                max_metric_calls=8,
                stop_callbacks=lambda state: len(reflection_prompts) >= 1,
                use_merge=False,
                seed=42,
                raise_on_exception=True,
                display_progress_bar=False,
            )
        self.assertEqual(
            result.best_candidate["sql_rules"], "Better general SQL rules."
        )
        self.assertEqual(len(reflection_prompts), 1)
        self.assertIn("SELECT COUNT(*) FROM entries", reflection_prompts[0])
        self.assertNotIn("val_db", reflection_prompts[0])

    def test_invalid_candidate_is_failed_without_paid_call(self):
        m = mini_manifest()
        with patch.object(runtime.Session, "evaluate") as evaluate:
            adapter = runtime.Adapter(runtime.Session.__new__(runtime.Session), m)
            result = adapter.evaluate(
                m["cohorts"]["fit"], {"sql_rules": "{schema}"}, True
            )
            self.assertEqual(result.scores, [0.0])
            evaluate.assert_not_called()


class SelectionAndResumeTests(unittest.TestCase):
    def test_completed_question_cache_skips_budget_and_generation(self):
        m = mini_manifest()
        text = m["seed_instruction"]
        item = m["cohorts"]["fit"][0]
        session = runtime.Session.__new__(runtime.Session)
        session.manifest, session.phase = m, "search"
        row = {"score": 1, "prediction": "SELECT 1"}
        session.records = {f"search:{runtime.candidate_id(text, 'hybrid')}:bird:1": row}
        self.assertIs(session.evaluate(item, text), row)

    def test_database_failure_stops_but_sql_error_gets_zero(self):
        from sqlalchemy.exc import SQLAlchemyError
        from unittest.mock import Mock

        session = runtime.Session.__new__(runtime.Session)
        item = {"db_id": "db", "question_id": "1"}
        error = SQLAlchemyError("redacted")
        error.orig = SimpleNamespace(sqlstate=None)
        session.engines = {"db": Mock(connect=Mock(side_effect=error))}
        with self.assertRaises(runtime.ProviderUnavailable):
            session.score(item, "SELECT 1", "bird")
        error.orig.sqlstate = "42601"
        self.assertEqual(session.score(item, "SELECT 1", "bird")[0], 0)

    def test_selected_seed_skips_paid_confirmation(self):
        m = mini_manifest()
        selection = {
            "manifest_sha256": m["sha256"],
            "instruction": m["seed_instruction"],
        }
        selection["sha256"] = runtime.search.fingerprint(selection)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "selection.json").write_text(
                json.dumps(selection), encoding="utf-8"
            )
            with patch.object(runtime, "Session") as session:
                runtime.run(output, m, "confirm")
                session.assert_not_called()

    def test_incomplete_candidates_cannot_win_and_tie_keeps_seed(self):
        m = mini_manifest()
        seed = m["seed_instruction"]
        candidates = {}
        records = {}
        for text, completed in ((seed, True), ("Short", True), ("Incomplete", False)):
            cid = runtime.candidate_id(text, "hybrid")
            candidates[cid] = {
                "instruction": text,
                "profile": "hybrid",
                "original": False,
            }
            if completed:
                records[f"search:{cid}:bird:2"] = {"score": 1}
        with tempfile.TemporaryDirectory() as tmp:
            result = runtime.freeze_selection(Path(tmp), m, records, candidates)
            self.assertEqual(result["instruction"], seed)
            self.assertFalse(result["external_dev_used"])

    def test_completed_phase_skips_db_and_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "confirm").mkdir()
            (output / "confirm/metadata.json").write_text(
                json.dumps({"status": "completed", "manifest_sha256": "identity"}),
                encoding="utf-8",
            )
            with patch.object(runtime, "Session") as session:
                runtime.run(output, mini_manifest(), "confirm")
                session.assert_not_called()

    def test_preflight_failure_happens_before_gepa_or_model_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)

            class BrokenSession:
                def __init__(self, *args):
                    self.folder = output / "search"
                    self.folder.mkdir()
                    self.teacher_config = {}
                    self.metadata = {}

                def preflight(self):
                    raise RuntimeError("schema mismatch")

                def close(self):
                    pass

            with (
                patch.object(runtime, "Session", BrokenSession),
                patch.object(pilot, "load_gepa") as engine,
            ):
                with self.assertRaisesRegex(RuntimeError, "schema mismatch"):
                    runtime.run(output, mini_manifest(), "search")
                engine.assert_not_called()
            state = json.loads(
                (output / "search/metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["status"], "stopped")


if __name__ == "__main__":
    unittest.main()
