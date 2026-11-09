from copy import deepcopy
from pathlib import Path
import os
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ambrosia_prompt_full as full
import test_ambrosia_prompt_dev as fixtures


def fixture():
    m, _ = fixtures.fixture()
    item = deepcopy(m["dev"]["ambrosia"][0])
    item.update(question_id=901, SQL="SELECT 987654321 AS current_gold")
    other = deepcopy(item)
    other.update(question_id=902, SQL="ambiguous")
    m.update(
        full={"ambrosia": [item, other]},
        fixed_prompts=full.original_templates(),
        limits=full.LIMITS,
        global_limits=full.GLOBAL_LIMITS,
        historical_reference=dict(correct=1, accuracy=50),
    )
    m["jobs"] = [
        dict(
            name="ambrosia_full__control",
            slot="ambiguity",
            repeat=1,
            seed=42,
            arms={
                "original": full.source.candidate(m["seed_instructions"]["ambiguity"])
            },
        )
    ]
    return m


def runtime_fixture():
    from dotenv import dotenv_values

    m = fixture()
    public = full.source.pilot.public_model(
        dotenv_values(full.source.ROOT / ".env.deepseek")
    )
    m.update(
        env_file=".env.deepseek",
        teacher_env_file=".env.deepseek",
        model=public,
        teacher=public,
    )
    return m


class AmbrosiaFullTests(unittest.TestCase):
    def test_exact_original_prompts_and_mask_are_scoped_and_changed_candidates_rejected(
        self,
    ):
        m = fixture()
        s = object.__new__(full.FullSession)
        s.manifest = m
        s.slot = "ambiguity"
        s.fit = m["cohorts"]["ambiguity"]["fit"]
        s.impl = SimpleNamespace(
            SQL_PROMPT_TEMPLATE="foreign SQL",
            SQL_PROMPT_BASIC="foreign basic",
            TAXONOMY_DETECTION_PROMPT="before",
            **{f"FEAT_{i}": False for i in range(1, 37)},
        )
        value = m["jobs"][0]["arms"]["original"]
        with s.apply(value, m["full"]["ambrosia"][0], "hybrid", False) as demos:
            self.assertEqual(
                [i for i in range(1, 37) if getattr(s.impl, f"FEAT_{i}")], [14, 32]
            )
            for key, v in m["fixed_prompts"].items():
                self.assertEqual(getattr(s.impl, key), v)
            self.assertFalse(demos)
            self.assertNotIn(
                "987654321",
                s.impl.SQL_PROMPT_TEMPLATE + s.impl.TAXONOMY_DETECTION_PROMPT,
            )
        self.assertEqual(s.impl.SQL_PROMPT_TEMPLATE, "foreign SQL")
        self.assertEqual(s.impl.TAXONOMY_DETECTION_PROMPT, "before")
        with self.assertRaises(ValueError):
            with s.apply(
                full.source.candidate(value["instruction"] + " changed"),
                m["full"]["ambrosia"][0],
                "hybrid",
                False,
            ):
                pass

    def test_real_evaluation_full_scope_cache_and_teacher_forbidden(self):
        m = runtime_fixture()
        before = deepcopy(m["dev"])
        with tempfile.TemporaryDirectory() as tmp, full.shared.durable_runtime():
            s = full.FullSession(Path(tmp), m, m["jobs"][0])
            s.runner.get_loop()
            self.assertEqual(s.items, m["full"]["ambrosia"])
            self.assertEqual(before, m["dev"])
            self.assertEqual(s.metadata["evaluation_scope"], "full_retrospective")
            s.impl = SimpleNamespace(
                SQL_PROMPT_TEMPLATE="before",
                SQL_PROMPT_BASIC="before",
                TAXONOMY_DETECTION_PROMPT="before",
                **{f"FEAT_{i}": False for i in range(1, 37)},
            )
            calls = []

            async def query(question):
                calls.append(question)
                self.assertEqual(
                    s.impl.SQL_PROMPT_TEMPLATE,
                    m["fixed_prompts"]["SQL_PROMPT_TEMPLATE"],
                )
                return dict(status="ambiguous", query="")

            s.tools = {
                "hybrid": {
                    "dev_amb": SimpleNamespace(
                        query=query, engine=SimpleNamespace(dispose=lambda: None)
                    )
                }
            }
            value = m["jobs"][0]["arms"]["original"]
            with (
                patch.object(
                    socket.socket, "connect", side_effect=AssertionError("No network")
                ),
                patch.object(
                    socket.socket,
                    "connect_ex",
                    side_effect=AssertionError("No network"),
                ),
                patch.object(full.shared.runtime.asyncio, "sleep", AsyncMock()),
            ):
                a = s.evaluate(m["full"]["ambrosia"][0], value, benchmark="ambrosia")
                b = s.evaluate(m["full"]["ambrosia"][1], value, benchmark="ambrosia")
                self.assertEqual([a["score"], b["score"]], [0, 1])
                s.evaluate(m["full"]["ambrosia"][0], value, benchmark="ambrosia")
                self.assertEqual(len(calls), 2)
                with self.assertRaises(ValueError):
                    s.evaluate(m["dev"]["ambrosia"][0], value, benchmark="ambrosia")
                with self.assertRaises(RuntimeError):
                    s.reflect("No teacher")
                with self.assertRaises(full.shared.budget.BudgetStop):
                    s.ledger.reserve("teacher", [], 1)
            s.close()

    def test_real_tool_construction_receives_full_only_and_exact_mask(self):
        import autogen_ext.models.openai as api
        import baseline_connection

        m = runtime_fixture()
        with (
            tempfile.TemporaryDirectory() as tmp,
            full.shared.durable_runtime(),
            patch.dict(os.environ),
        ):
            s = full.FullSession(Path(tmp), m, m["jobs"][0])
            s.runner.get_loop()

            async def builder(items, generator, client, progress):
                self.assertEqual(items, m["full"]["ambrosia"])
                self.assertNotEqual(items, m["dev"]["ambrosia"])
                self.assertEqual(
                    [i for i in range(1, 37) if os.environ.get(f"FEAT_{i}") == "true"],
                    [14, 32],
                )
                self.assertEqual(
                    [i for i in range(1, 37) if getattr(s.impl, f"FEAT_{i}")], [14, 32]
                )
                return {}

            with (
                patch.object(
                    api,
                    "OpenAIChatCompletionClient",
                    return_value=SimpleNamespace(close=AsyncMock()),
                ),
                patch.object(baseline_connection, "build_tools", side_effect=builder),
                patch.object(
                    socket.socket, "connect", side_effect=AssertionError("No network")
                ),
                patch.object(
                    socket.socket,
                    "connect_ex",
                    side_effect=AssertionError("No network"),
                ),
            ):
                s.runner.run(s.prepare_tools("hybrid"))
            s.close()

    def test_preflight_checks_full_ambrosia_without_bird_or_reference_execution(self):
        import resilient_hybrid_runtime
        import baseline_connection

        m = runtime_fixture()
        with tempfile.TemporaryDirectory() as tmp, full.shared.durable_runtime():
            s = full.FullSession(Path(tmp), m, m["jobs"][0])
            with (
                patch.object(
                    resilient_hybrid_runtime, "validate_live_schemas"
                ) as check,
                patch.object(
                    baseline_connection,
                    "database_engine",
                    side_effect=AssertionError("No SQL reference execution"),
                ),
            ):
                s.preflight()
                self.assertEqual(check.call_args.args[0], m["full"]["ambrosia"])
                self.assertEqual(check.call_count, 1)
                self.assertEqual(s.ledger.data["task_calls"], 0)
            s.close()

    def test_report_never_counts_failed_unambiguous_answer_as_correct(self):
        m = fixture()
        job = m["jobs"][0]
        value = job["arms"]["original"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "jobs" / job["name"]
            (folder / "confirm").mkdir(parents=True)
            full.store.write_json(
                folder / "confirm/metadata.json",
                dict(
                    status="completed",
                    manifest_sha256=m["sha256"],
                    feature_mask=full.FEATURES,
                    full_prompt_version=full.VERSION,
                    evaluation_scope="full_retrospective",
                ),
            )
            full.report(root, m)
            self.assertFalse((root / "conclusion.json").exists())
            records = {
                full.dev.key(value, m["full"]["ambrosia"][0]): dict(
                    prediction="error", score=0
                ),
                full.dev.key(value, m["full"]["ambrosia"][1]): dict(
                    prediction="ambiguous", score=1
                ),
            }
            full.store.write_json(folder / "evaluations.json", records)
            full.report(root, m)
            conclusion = full.read(root / "conclusion.json")
            self.assertEqual(conclusion["row"]["accuracy"], 0.5)
            self.assertEqual(conclusion["row"]["metrics"]["generation_errors"], 1)
            self.assertFalse(conclusion["selection_on_full"])
            self.assertFalse(conclusion["deployed"])
            self.assertFalse((root / "selection.json").exists())

    def test_failed_preflight_stops_without_paid_or_model_error_checkpoint(self):
        m = runtime_fixture()
        with (
            tempfile.TemporaryDirectory() as tmp,
            full.shared.durable_runtime(),
            patch.object(
                full.FullSession,
                "preflight",
                side_effect=RuntimeError("DB unavailable"),
            ),
            patch.object(full.FullSession, "evaluate") as call,
        ):
            root = Path(tmp)
            with self.assertRaises(RuntimeError):
                full.run_job(root, m, m["jobs"][0])
            call.assert_not_called()
            self.assertEqual(full.read(root / "budget.json")["task_calls"], 0)
            self.assertFalse(
                (root / "jobs" / m["jobs"][0]["name"] / "evaluations.json").exists()
            )

    def test_foreign_scope_and_output_rejected(self):
        m = fixture()
        job = m["jobs"][0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "jobs" / job["name"] / "confirm/metadata.json"
            p.parent.mkdir(parents=True)
            full.store.write_json(
                p,
                dict(
                    status="completed",
                    manifest_sha256=m["sha256"],
                    feature_mask=full.FEATURES,
                    full_prompt_version=full.VERSION,
                    evaluation_scope="dev",
                ),
            )
            with self.assertRaises(ValueError):
                full.completed(root, job, m)
        for path in [
            full.dev.DEFAULT_OUTPUT,
            full.bird.DEFAULT_OUTPUT,
            "../ambrosia_prompt_full_escape",
        ]:
            with self.assertRaises(ValueError):
                full.output_path(path)

    def test_subprocess_reentry_keeps_full_adapter(self):
        previous = full.shared.__file__
        with full.runtime_context():
            self.assertEqual(
                Path(full.shared.__file__).resolve(), Path(full.__file__).resolve()
            )
            self.assertEqual(full.shared.FEATURES, [14, 32])
            self.assertIs(full.shared.run_job, full.run_job)
        self.assertEqual(full.shared.__file__, previous)


if __name__ == "__main__":
    unittest.main()
