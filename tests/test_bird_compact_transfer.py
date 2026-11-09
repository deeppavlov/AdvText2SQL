from copy import deepcopy
from pathlib import Path
import socket
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bird_compact_transfer as compact
import test_prompt_comparison as fixtures


def plan_fixture():
    m = fixtures.manifest()
    selections = {}
    for repeat, seed in enumerate(compact.source.SEEDS, 1):
        for method in compact.source.METHODS:
            name = f"sql__{method}__r{repeat}"
            s = dict(
                manifest_sha256=m["sha256"],
                job=dict(name=name, slot="sql", method=method, seed=seed),
                candidate=compact.source.candidate(m["seed_instructions"]["sql"]),
                external_dev_used=False,
            )
            s["sha256"] = compact.source.search.fingerprint(s)
            selections[name] = s
    return m, selections


class CompactTests(unittest.TestCase):
    def test_completed_checkpoint_from_old_mask_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = dict(name="bird_compact__r1")
            path = root / "jobs/bird_compact__r1/confirm/metadata.json"
            path.parent.mkdir(parents=True)
            compact.store.write_json(
                path,
                dict(status="completed", manifest_sha256="old", feature_mask=[14, 32]),
            )
            with self.assertRaises(ValueError):
                compact.completed(root, job, dict(sha256="new"))

    def test_output_is_separate_and_cannot_escape_results(self):
        self.assertEqual(
            compact.output_path(compact.DEFAULT_OUTPUT).name,
            "bird_compact_prompt_transfer_20261107",
        )
        for path in [
            "research_results",
            compact.source.DEFAULT_OUTPUT,
            "../bird_compact_prompt_transfer_bad",
        ]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                compact.output_path(path)

    def test_plan_keeps_all_methods_seeds_and_deduplicates_only_within_repeat(self):
        m, selections = plan_fixture()
        jobs = compact.plan_jobs(m, selections)
        self.assertEqual([j["seed"] for j in jobs], [42, 43, 44])
        self.assertTrue(
            all(
                list(j["arms"]) == ["original", "fixed_few", *compact.source.METHODS]
                for j in jobs
            )
        )
        self.assertEqual([compact.count_evaluations(m, j) for j in jobs], [2, 2, 2])
        self.assertEqual(sum(compact.count_evaluations(m, j) for j in jobs), 6)

    def test_changed_or_dev_selected_candidate_rejected(self):
        m, selections = plan_fixture()
        for field, value in (("external_dev_used", True), ("manifest_sha256", "other")):
            changed = deepcopy(selections)
            s = changed["sql__gepa_zero__r1"]
            s[field] = value
            s["sha256"] = compact.source.search.fingerprint(
                {k: v for k, v in s.items() if k != "sha256"}
            )
            with self.assertRaises(ValueError):
                compact.plan_jobs(m, changed)

    def test_compact_flags_and_original_prompt_are_exact_and_scoped(self):
        m, _ = plan_fixture()
        session = object.__new__(compact.CompactSession)
        session.manifest, session.slot, session.fit = (
            m,
            "sql",
            m["cohorts"]["sql"]["fit"],
        )
        session.impl = SimpleNamespace(
            SQL_PROMPT_TEMPLATE="before", **{f"FEAT_{i}": False for i in range(1, 37)}
        )
        value = compact.source.candidate(m["seed_instructions"]["sql"])
        with session.apply(value, m["dev"]["bird"][0], "hybrid", False) as demos:
            self.assertEqual(
                [i for i in range(1, 37) if getattr(session.impl, f"FEAT_{i}")],
                compact.FEATURES,
            )
            self.assertEqual(
                session.impl.SQL_PROMPT_TEMPLATE,
                compact.source.pilot.templates()["SQL_PROMPT_TEMPLATE"],
            )
            self.assertFalse(demos)
        self.assertEqual(session.impl.SQL_PROMPT_TEMPLATE, "before")
        self.assertFalse(any(getattr(session.impl, f"FEAT_{i}") for i in range(1, 37)))

    def test_dev_gold_demo_rejected_and_current_gold_not_rendered(self):
        m, _ = plan_fixture()
        item = m["dev"]["bird"][0]
        item["SQL"] = "SELECT 987654321 AS protected_gold"
        value = compact.source.candidate(m["seed_instructions"]["sql"], ["1", "2"])
        template, demos = compact.source.render(
            value, m, "sql", m["cohorts"]["sql"]["fit"], item
        )
        self.assertNotIn("987654321", template)
        self.assertEqual(demos, ["1", "2"])
        with self.assertRaises(ValueError):
            compact.source.check(
                compact.source.candidate(
                    value["instruction"], [str(item["question_id"])]
                ),
                m,
                "sql",
                m["cohorts"]["sql"]["fit"],
            )

    def test_real_evaluator_checkpoint_reuse_and_no_cross_repeat_cache(self):
        from dotenv import dotenv_values

        m, selections = plan_fixture()
        m["jobs"] = compact.plan_jobs(m, selections)
        public = compact.source.pilot.public_model(
            dotenv_values(compact.source.ROOT / ".env.deepseek")
        )
        m.update(
            env_file=".env.deepseek",
            teacher_env_file=".env.deepseek",
            model=public,
            teacher=public,
            limits=compact.LIMITS,
            global_limits=compact.GLOBAL_LIMITS,
        )
        m["dev"]["ambrosia"] = []
        calls = []
        with tempfile.TemporaryDirectory() as tmp, compact.durable_runtime():
            root = Path(tmp)
            for job in m["jobs"][:2]:
                session = compact.CompactSession(root, m, job, "confirm")
                session.runner.get_loop()  # Windows self-pipe before socket blocking.
                session.impl = SimpleNamespace(
                    SQL_PROMPT_TEMPLATE="before",
                    **{f"FEAT_{i}": False for i in range(1, 37)},
                )

                async def query(question):
                    self.assertEqual(
                        [i for i in range(1, 37) if getattr(session.impl, f"FEAT_{i}")],
                        compact.FEATURES,
                    )
                    calls.append(question)
                    return dict(status="success", query="SELECT 1")

                session.tools = {
                    "hybrid": {
                        "dev": SimpleNamespace(
                            query=query, engine=SimpleNamespace(dispose=lambda: None)
                        )
                    }
                }
                session.score = lambda *args: (1.0, None)
                item = m["dev"]["bird"][0]
                value = job["arms"]["original"]
                with (
                    patch.object(
                        socket.socket,
                        "connect",
                        side_effect=AssertionError("No network"),
                    ),
                    patch.object(
                        socket.socket,
                        "connect_ex",
                        side_effect=AssertionError("No network"),
                    ),
                    patch.object(
                        socket,
                        "create_connection",
                        side_effect=AssertionError("No network"),
                    ),
                    patch.object(compact.runtime.asyncio, "sleep", AsyncMock()),
                ):
                    first = session.evaluate(item, value, benchmark="bird")
                    self.assertEqual(
                        first, session.evaluate(item, value, benchmark="bird")
                    )
                session.close()
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(list(root.glob("jobs/*/evaluations.json"))), 2)

    def test_real_tool_preparation_uses_compact_environment_and_flags(self):
        from dotenv import dotenv_values
        import autogen_ext.models.openai as api
        import baseline_connection

        m, selections = plan_fixture()
        job = compact.plan_jobs(m, selections)[0]
        public = compact.source.pilot.public_model(
            dotenv_values(compact.source.ROOT / ".env.deepseek")
        )
        m.update(
            env_file=".env.deepseek",
            teacher_env_file=".env.deepseek",
            model=public,
            teacher=public,
            limits=compact.LIMITS,
            global_limits=compact.GLOBAL_LIMITS,
        )
        m["dev"]["ambrosia"] = []
        with (
            tempfile.TemporaryDirectory() as tmp,
            compact.durable_runtime(),
            patch.dict(os.environ),
        ):
            session = compact.CompactSession(Path(tmp), m, job, "confirm")
            session.runner.get_loop()

            async def builder(items, generator, client, progress):
                self.assertEqual(items, m["dev"]["bird"])
                self.assertEqual(
                    [i for i in range(1, 37) if os.environ.get(f"FEAT_{i}") == "true"],
                    compact.FEATURES,
                )
                self.assertEqual(
                    [i for i in range(1, 37) if getattr(session.impl, f"FEAT_{i}")],
                    compact.FEATURES,
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

    def test_report_selects_only_after_all_three_paired_dev_repeats(self):
        m, selections = plan_fixture()
        jobs = compact.plan_jobs(m, selections)
        improved = compact.source.candidate(
            m["seed_instructions"]["sql"]
            + "\nCheck the requested projection carefully."
        )
        for job in jobs:
            job["arms"]["mipro_zero"] = improved
        m["jobs"] = jobs
        m["policy"] = {"selection": "Mean dev EX after all three repeats"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for job in jobs:
                folder = root / "jobs" / job["name"]
                (folder / "confirm").mkdir(parents=True)
                compact.store.write_json(
                    folder / "confirm/metadata.json",
                    dict(
                        status="completed",
                        manifest_sha256=m["sha256"],
                        feature_mask=compact.FEATURES,
                    ),
                )
                records = {}
                for arm, value in job["arms"].items():
                    for item in m["dev"]["bird"]:
                        records[compact.key(value, item)] = dict(
                            score=float(arm == "mipro_zero"),
                            prediction="SELECT 1",
                            error=None,
                        )
                compact.store.write_json(folder / "evaluations.json", records)
                compact.report(root, m)
                if job["repeat"] < 3:
                    self.assertFalse((root / "selection.json").exists())
            result = compact.read(root / "selection.json")
            self.assertEqual(result["selected_arm"], "mipro_zero")
            self.assertEqual(result["mean_dev_accuracy"], 1.0)
            self.assertFalse(result["deployed"])

    def recovery_fixture(self, root, code=500):
        job = "bird_compact__r1"
        folder = root / "jobs" / job / "confirm"
        folder.mkdir(parents=True)
        m = dict(sha256="compact", jobs=[dict(name=job)])
        compact.store.write_json(root / "experiment.json", m)
        for path, ticket in [
            (root / "budget.json", "global"),
            (folder / "budget.json", "local"),
        ]:
            compact.store.write_json(
                path,
                dict(
                    tokens=15,
                    task_calls=1,
                    teacher_calls=0,
                    prompt_tokens=10,
                    completion_tokens=5,
                    pending={ticket: dict(role="task", reserved=1234)},
                ),
            )
        compact.store.write_json(
            folder / "metadata.json",
            dict(
                status="stopped",
                manifest_sha256="compact",
                transport_failure=dict(
                    phase="confirm",
                    reason="client_exception",
                    checkpoint_written=False,
                    status_code=code,
                ),
            ),
        )
        compact.store.write_json(
            folder.parent / "evaluations.json", {"saved": dict(score=1.0)}
        )
        return m

    def test_transient_recovery_is_journaled_idempotent_and_preserves_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = self.recovery_fixture(root)
            saved = (root / "jobs/bird_compact__r1/evaluations.json").read_bytes()
            self.assertTrue(compact.recover_pending(root, m))
            self.assertFalse(compact.recover_pending(root, m))
            for path in [
                root / "budget.json",
                root / "jobs/bird_compact__r1/confirm/budget.json",
            ]:
                b = compact.read(path)
                self.assertFalse(b["pending"])
                self.assertEqual(b["tokens"], 1249)
                self.assertEqual(b["task_calls"], 1)
                self.assertEqual(b["prompt_tokens"] + b["completion_tokens"], 15)
            self.assertEqual(
                saved, (root / "jobs/bird_compact__r1/evaluations.json").read_bytes()
            )

    def test_billing_or_unselected_repeat_is_not_recovered(self):
        for code, allowed in [(402, None), (500, {"bird_compact__r2"})]:
            with (
                self.subTest(code=code, allowed=allowed),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                m = self.recovery_fixture(root, code)
                before = (root / "budget.json").read_bytes()
                with self.assertRaises(ValueError):
                    compact.recover_pending(root, m, allowed)
                self.assertEqual(before, (root / "budget.json").read_bytes())

    def test_teacher_is_forbidden(self):
        with self.assertRaises(RuntimeError):
            compact.CompactSession.reflect(None, "Do not call teacher")
        with tempfile.TemporaryDirectory() as tmp:
            ledger = compact.budget.Ledger(Path(tmp), compact.LIMITS)
            with self.assertRaises(compact.budget.BudgetStop):
                ledger.reserve("teacher", [], 4096)

    def test_partial_accounting_write_can_finish_without_double_charge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = self.recovery_fixture(root)
            local = root / "jobs/bird_compact__r1/confirm/budget.json"
            write = compact.store.write_json

            def interrupted(path, value):
                if Path(path) == local:
                    raise OSError("Simulated disk failure")
                write(path, value)

            with patch.object(compact.store, "write_json", side_effect=interrupted):
                with self.assertRaises(OSError):
                    compact.recover_pending(root, m)
            self.assertFalse(compact.read(root / "budget.json")["pending"])
            self.assertTrue(compact.read(local)["pending"])
            self.assertTrue(compact.recover_pending(root, m))
            self.assertFalse(compact.recover_pending(root, m))
            self.assertEqual(compact.read(root / "budget.json")["tokens"], 1249)
            self.assertEqual(compact.read(local)["tokens"], 1249)

    def test_external_resumes_are_bounded_and_preserve_authorized_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = dict(sha256="frozen")
            jobs = [dict(name="bird_compact__r1")]
            with (
                patch.object(
                    compact, "recover_pending", side_effect=[False, True, True, True]
                ),
                patch.object(
                    compact.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=1),
                ) as child,
                patch.object(compact.time, "sleep") as sleep,
                patch("builtins.print"),
            ):
                self.assertEqual(compact.supervise(root, m, jobs, 2), 1)
                self.assertEqual(child.call_count, 3)
                self.assertEqual([c.args[0] for c in sleep.call_args_list], [60, 120])
                self.assertEqual(
                    child.call_args.args[0][-2:], ["--jobs", "bird_compact__r1"]
                )


if __name__ == "__main__":
    unittest.main()
