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
import bird_prompt_full as full
import test_bird_compact_transfer as fixtures


def fixture():
    m, selections = fixtures.plan_fixture()
    m["jobs"] = full.shared.plan_jobs(m, selections)
    records = {}
    for j in m["jobs"]:
        j["arms"]["mipro_few"] = full.source.candidate(
            m["seed_instructions"]["sql"]
            + f"\nCheck column selection carefully, rule {j['repeat']}."
        )
        rows = {}
        for value in j["arms"].values():
            for item in m["dev"]["bird"]:
                rows[full.shared.key(value, item)] = dict(
                    score=float(value == j["arms"]["mipro_few"]), prediction="SELECT 1"
                )
        records[j["name"]] = rows
    selected = dict(
        manifest_sha256=m["sha256"], selected_arm="mipro_few", mean_dev_accuracy=1.0
    )
    choice = full.choose_candidate(m, records, selected)
    # New single-candidate policy ties by shorter instruction, then repeat.
    item = deepcopy(m["dev"]["bird"][0])
    item.update(question_id=901, SQL="SELECT 987654321 AS protected_gold")
    second = deepcopy(item)
    second["question_id"] = 902
    m.update(
        full={"bird": [item, second]}, feature_mask=full.FEATURES, frozen_choice=choice
    )
    m["jobs"] = [
        dict(
            name="bird_full__paired",
            slot="sql",
            repeat=1,
            seed=choice["source_optimizer_seed"],
            arms={
                "original": full.source.candidate(m["seed_instructions"]["sql"]),
                "selected_sql": choice["candidate"],
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
        limits=full.LIMITS,
        global_limits=full.GLOBAL_LIMITS,
    )
    return m


class FullPromptTests(unittest.TestCase):
    def test_real_tool_construction_targets_full_not_source_dev(self):
        import autogen_ext.models.openai as api
        import baseline_connection

        m = runtime_fixture()
        with (
            tempfile.TemporaryDirectory() as tmp,
            full.shared.durable_runtime(),
            patch.dict(os.environ),
        ):
            session = full.FullSession(Path(tmp), m, m["jobs"][0])
            session.runner.get_loop()

            async def builder(items, generator, client, progress):
                self.assertEqual(items, m["full"]["bird"])
                self.assertNotEqual(items, m["dev"]["bird"])
                self.assertEqual(
                    [i for i in range(1, 37) if os.environ.get(f"FEAT_{i}") == "true"],
                    full.FEATURES,
                )
                self.assertEqual(
                    [i for i in range(1, 37) if getattr(session.impl, f"FEAT_{i}")],
                    full.FEATURES,
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
                session.runner.run(session.prepare_tools("hybrid"))
            session.close()

    def test_real_preflight_checks_all_full_references_read_only(self):
        from contextlib import nullcontext
        import baseline_connection
        import resilient_hybrid_runtime

        m = runtime_fixture()
        before = deepcopy(m["dev"])

        class Connection:
            read_only = []
            queries = []

            def exec_driver_sql(self, sql):
                self.read_only.append(sql)

            def execute(self, sql):
                self.queries.append(str(sql))
                return SimpleNamespace(fetchall=lambda: [(1,)])

        conn = Connection()
        engine = SimpleNamespace(
            connect=lambda: nullcontext(conn), dispose=lambda: None
        )
        with tempfile.TemporaryDirectory() as tmp, full.shared.durable_runtime():
            session = full.FullSession(Path(tmp), m, m["jobs"][0])
            with (
                patch.object(
                    baseline_connection, "database_engine", return_value=engine
                ),
                patch.object(
                    resilient_hybrid_runtime, "validate_live_schemas"
                ) as schema_check,
                patch.object(
                    socket.socket, "connect", side_effect=AssertionError("No network")
                ),
            ):
                session.preflight()
                self.assertEqual(
                    schema_check.call_args_list[0].args[0], m["full"]["bird"]
                )
                self.assertEqual(set(session.golds), {"901", "902"})
                self.assertEqual(len(conn.queries), 2)
                self.assertEqual(conn.read_only, ["SET TRANSACTION READ ONLY"] * 2)
                self.assertEqual(session.ledger.data["task_calls"], 0)
            session.close()
        self.assertEqual(before, m["dev"])

    def test_candidate_choice_uses_complete_dev_and_keeps_zero_demo_identity(self):
        m, selections = fixtures.plan_fixture()
        m["jobs"] = full.shared.plan_jobs(m, selections)
        records = {}
        for j in m["jobs"]:
            value = full.source.candidate(
                m["seed_instructions"]["sql"]
                + f"\nCheck the requested projection carefully, iteration {j['repeat']}.",
                ["1"] if j["repeat"] != 2 else [],
            )
            j["arms"]["mipro_few"] = value
            records[j["name"]] = {
                full.shared.key(v, i): dict(
                    score=float(v == value and j["repeat"] == 2)
                )
                for v in j["arms"].values()
                for i in m["dev"]["bird"]
            }
        selected = dict(
            manifest_sha256=m["sha256"],
            selected_arm="mipro_few",
            mean_dev_accuracy=1 / 3,
        )
        choice = full.choose_candidate(m, records, selected)
        self.assertEqual(choice["source_repeat"], 2)
        self.assertEqual(choice["source_optimizer_seed"], 43)
        self.assertEqual(choice["effective_demo_ids"], [])
        self.assertEqual(choice["candidate"], m["jobs"][1]["arms"]["mipro_few"])
        with self.assertRaises(ValueError):
            full.choose_candidate(m, records, {**selected, "mean_dev_accuracy": 0.9})

    def test_pairs_keep_same_questions_and_alternate_prompt_order(self):
        m = fixture()
        job = m["jobs"][0]
        values = list(full.pairs(job, m["full"]["bird"]))
        self.assertEqual(
            [str(i["question_id"]) for i, v in values], ["901", "901", "902", "902"]
        )
        self.assertEqual(
            [v for i, v in values],
            [
                job["arms"]["original"],
                job["arms"]["selected_sql"],
                job["arms"]["selected_sql"],
                job["arms"]["original"],
            ],
        )

    def test_real_evaluate_uses_full_scope_without_relabeling_source_dev_or_gold_leak(
        self,
    ):
        m = runtime_fixture()
        before = deepcopy(m["dev"])
        with tempfile.TemporaryDirectory() as tmp, full.shared.durable_runtime():
            root = Path(tmp)
            session = full.FullSession(root, m, m["jobs"][0])
            session.runner.get_loop()
            self.assertEqual(m["dev"], before)
            self.assertEqual(session.items, m["full"]["bird"])
            self.assertEqual(session.metadata["evaluation_scope"], "full_retrospective")
            session.impl = SimpleNamespace(
                SQL_PROMPT_TEMPLATE="before",
                **{f"FEAT_{i}": False for i in range(1, 37)},
            )
            calls = []

            async def query(question):
                calls.append(question)
                self.assertEqual(
                    [i for i in range(1, 37) if getattr(session.impl, f"FEAT_{i}")],
                    full.FEATURES,
                )
                self.assertNotIn("987654321", session.impl.SQL_PROMPT_TEMPLATE)
                return dict(status="success", query="SELECT 1")

            session.tools = {
                "hybrid": {
                    "dev": SimpleNamespace(
                        query=query, engine=SimpleNamespace(dispose=lambda: None)
                    )
                }
            }
            session.score = lambda *args: (1.0, None)
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
                for item, value in full.pairs(m["jobs"][0], m["full"]["bird"]):
                    session.evaluate(item, value, benchmark="bird")
                session.evaluate(
                    m["full"]["bird"][0],
                    m["jobs"][0]["arms"]["original"],
                    benchmark="bird",
                )
                with self.assertRaises(ValueError):
                    session.evaluate(
                        m["dev"]["bird"][0],
                        m["jobs"][0]["arms"]["original"],
                        benchmark="bird",
                    )
                with self.assertRaises(RuntimeError):
                    session.reflect("no teacher")
                with self.assertRaises(full.shared.budget.BudgetStop):
                    session.ledger.reserve("teacher", [], 1)
            self.assertEqual(len(calls), 4)
            session.close()

    def test_full_report_preserves_fixed_candidate_when_it_loses(self):
        m = fixture()
        job = m["jobs"][0]
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
            records = {
                full.shared.key(v, i): dict(
                    score=float(arm == "original"),
                    prediction="SELECT 1",
                    usage={"prompt_tokens": 5, "completion_tokens": 2},
                )
                for arm, v in job["arms"].items()
                for i in m["full"]["bird"]
            }
            full.store.write_json(folder / "evaluations.json", records)
            full.report(root, m)
            c = full.read(root / "conclusion.json")
            self.assertFalse(c["selection_on_full"])
            self.assertFalse(c["deployed"])
            self.assertEqual(c["fixed_candidate"], m["frozen_choice"]["candidate_id"])
            self.assertEqual(c["rows"][1]["harmed"], 2)
            self.assertEqual(c["rows"][1]["delta"], -1)
            self.assertFalse((root / "selection.json").exists())

    def test_failed_preflight_stops_before_paid_generation_without_model_error(self):
        m = runtime_fixture()
        job = m["jobs"][0]
        with (
            tempfile.TemporaryDirectory() as tmp,
            full.shared.durable_runtime(),
            patch.object(
                full.FullSession,
                "preflight",
                side_effect=RuntimeError("DB unavailable"),
            ),
            patch.object(full.FullSession, "evaluate") as evaluate,
        ):
            root = Path(tmp)
            with self.assertRaises(RuntimeError):
                full.run_job(root, m, job)
            evaluate.assert_not_called()
            self.assertFalse(
                (root / "jobs" / job["name"] / "evaluations.json").exists()
            )
            self.assertEqual(full.read(root / "budget.json")["task_calls"], 0)
            self.assertEqual(
                full.read(root / "jobs" / job["name"] / "confirm/metadata.json")[
                    "status"
                ],
                "stopped",
            )

    def test_scope_and_output_isolation(self):
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
        for value in [
            full.shared.DEFAULT_OUTPUT,
            full.FULL_SOURCE,
            "../bird_prompt_full_escape",
        ]:
            with self.assertRaises(ValueError):
                full.output_path(value)
        with self.assertRaises(ValueError):
            full.full_source_path("../deepseek_full_static_escape")

    def test_subprocess_reentry_keeps_same_runtime_and_scope(self):
        original = full.shared.__file__
        with full.runtime_context():
            self.assertEqual(
                Path(full.shared.__file__).resolve(), Path(full.__file__).resolve()
            )
            self.assertIs(full.shared.run_job, full.run_job)
            self.assertIs(full.shared.completed, full.completed)
        self.assertEqual(full.shared.__file__, original)


if __name__ == "__main__":
    unittest.main()
