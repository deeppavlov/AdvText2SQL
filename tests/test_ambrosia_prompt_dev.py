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
import ambrosia_prompt_dev as dev
import test_prompt_comparison as fixtures


def fixture():
    m = fixtures.manifest()
    selections = {}
    for name in dev.SOURCES:
        _, method, repeat = name.split("__")
        value = dev.source.candidate(m["seed_instructions"]["ambiguity"])
        if name == "ambiguity__mipro_zero__r1":
            value = dev.source.candidate(
                value["instruction"] + "\nResolve scope carefully."
            )
        if name == "ambiguity__mipro_zero__r2":
            value = dev.source.candidate(
                value["instruction"] + "\nCheck modifier attachment."
            )
        if name == "ambiguity__textgrad_zero__r1":
            value = dev.source.candidate(
                value["instruction"] + "\nCheck schema alternatives."
            )
        s = dict(
            manifest_sha256=m["sha256"],
            external_dev_used=False,
            job=dict(
                name=name,
                slot="ambiguity",
                method=method,
                seed=dev.source.SEEDS[int(repeat[1:]) - 1],
            ),
            candidate=value,
        )
        s["sha256"] = dev.source.search.fingerprint(s)
        selections[name] = s
    m["dev"]["bird"] = []
    m.update(jobs=dev.plan_jobs(m, selections), policy={"selection": "offline test"})
    return m, selections


class AmbrosiaDevTests(unittest.TestCase):
    def test_real_tool_construction_uses_ambrosia_items_and_exact_mask(self):
        from dotenv import dotenv_values
        import autogen_ext.models.openai as api
        import baseline_connection

        m, _ = fixture()
        public = dev.source.pilot.public_model(
            dotenv_values(dev.source.ROOT / ".env.deepseek")
        )
        m.update(
            env_file=".env.deepseek",
            teacher_env_file=".env.deepseek",
            model=public,
            teacher=public,
            limits=dev.LIMITS,
            global_limits=dev.GLOBAL_LIMITS,
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            dev.shared.durable_runtime(),
            patch.dict(os.environ),
        ):
            session = dev.DevSession(Path(tmp), m, m["jobs"][0], "confirm")
            session.runner.get_loop()

            async def builder(items, generator, client, progress):
                self.assertEqual(items, m["dev"]["ambrosia"])
                self.assertEqual(
                    [i for i in range(1, 37) if os.environ.get(f"FEAT_{i}") == "true"],
                    [14, 32],
                )
                self.assertEqual(
                    [i for i in range(1, 37) if getattr(session.impl, f"FEAT_{i}")],
                    [14, 32],
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
            self.assertEqual(session.ledger.data["task_calls"], 0)
            session.close()

    def test_schema_preflight_only_checks_frozen_ambrosia_cohort(self):
        import resilient_hybrid_runtime

        m, _ = fixture()
        session = object.__new__(dev.DevSession)
        session.manifest = m
        session.slot = "ambiguity"
        session.items = m["dev"]["ambrosia"]
        session.progress = lambda update: None
        with patch.object(resilient_hybrid_runtime, "validate_live_schemas") as check:
            session.preflight()
            self.assertEqual(
                check.call_args.args,
                (session.items, m["schema_audit"]["schemas"]["ambrosia"]),
            )
            check.side_effect = RuntimeError("DB unavailable")
            with self.assertRaises(RuntimeError):
                session.preflight()

    def test_shortlist_keeps_all_sources_and_rotates_unique_candidates(self):
        m, selections = fixture()
        jobs = dev.plan_jobs(m, selections)
        self.assertEqual(len(jobs), 3)
        self.assertEqual([len(j["arms"]) for j in jobs], [7] * 3)
        self.assertEqual([len(dev.unique_candidates(j)) for j in jobs], [4] * 3)
        self.assertTrue(all(j["arms"] == jobs[0]["arms"] for j in jobs))
        self.assertEqual(
            len({dev.source.cid(dev.unique_candidates(j)[0]) for j in jobs}), 3
        )
        for name in dev.SOURCES:
            s = deepcopy(selections)
            s[name]["external_dev_used"] = True
            s[name]["sha256"] = dev.source.search.fingerprint(
                {k: v for k, v in s[name].items() if k != "sha256"}
            )
            with self.assertRaises(ValueError):
                dev.plan_jobs(m, s)

    def test_apply_exact_taxonomy_and_flags_without_changing_sql_or_leaking_gold(self):
        m, _ = fixture()
        session = object.__new__(dev.DevSession)
        session.manifest, session.slot, session.fit = (
            m,
            "ambiguity",
            m["cohorts"]["ambiguity"]["fit"],
        )
        session.impl = SimpleNamespace(
            TAXONOMY_DETECTION_PROMPT="before",
            SQL_PROMPT_TEMPLATE="SQL unchanged",
            **{f"FEAT_{i}": False for i in range(1, 37)},
        )
        item = deepcopy(m["dev"]["ambrosia"][0])
        item["SQL"] = "SELECT 987654321 AS private_dev_gold"
        value = m["jobs"][0]["arms"]["original"]
        with session.apply(value, item, "hybrid", False):
            self.assertEqual(
                [i for i in range(1, 37) if getattr(session.impl, f"FEAT_{i}")],
                [14, 32],
            )
            self.assertEqual(
                session.impl.TAXONOMY_DETECTION_PROMPT,
                dev.source.all_templates()["TAXONOMY_DETECTION_PROMPT"],
            )
            self.assertEqual(session.impl.SQL_PROMPT_TEMPLATE, "SQL unchanged")
            self.assertNotIn("987654321", session.impl.TAXONOMY_DETECTION_PROMPT)
        self.assertEqual(session.impl.TAXONOMY_DETECTION_PROMPT, "before")
        with self.assertRaises(ValueError):
            dev.source.check(
                dev.source.candidate(value["instruction"], [str(item["question_id"])]),
                m,
                "ambiguity",
                session.fit,
            )

    def test_real_evaluate_reuses_only_same_repeat_and_rejects_outside_cohort(self):
        from dotenv import dotenv_values

        m, _ = fixture()
        public = dev.source.pilot.public_model(
            dotenv_values(dev.source.ROOT / ".env.deepseek")
        )
        m.update(
            env_file=".env.deepseek",
            teacher_env_file=".env.deepseek",
            model=public,
            teacher=public,
            limits=dev.LIMITS,
            global_limits=dev.GLOBAL_LIMITS,
        )
        calls = []
        with tempfile.TemporaryDirectory() as tmp, dev.shared.durable_runtime():
            root = Path(tmp)
            for job in m["jobs"][:2]:
                session = dev.DevSession(root, m, job, "confirm")
                session.runner.get_loop()
                session.impl = SimpleNamespace(
                    TAXONOMY_DETECTION_PROMPT="before",
                    **{f"FEAT_{i}": False for i in range(1, 37)},
                )

                async def query(question):
                    calls.append(question)
                    self.assertEqual(
                        [i for i in range(1, 37) if getattr(session.impl, f"FEAT_{i}")],
                        [14, 32],
                    )
                    return dict(status="ambiguous", query="")

                session.tools = {
                    "hybrid": {
                        "dev_amb": SimpleNamespace(
                            query=query, engine=SimpleNamespace(dispose=lambda: None)
                        )
                    }
                }
                item = m["dev"]["ambrosia"][0]
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
                    patch.object(dev.shared.runtime.asyncio, "sleep", AsyncMock()),
                ):
                    result = session.evaluate(item, value, benchmark="ambrosia")
                    self.assertEqual(result["score"], 0)
                    session.evaluate(item, value, benchmark="ambrosia")
                    with self.assertRaises(ValueError):
                        session.evaluate(
                            m["cohorts"]["ambiguity"]["fit"][0],
                            value,
                            benchmark="ambrosia",
                        )
                    with self.assertRaises(RuntimeError):
                        session.reflect("no teacher")
                    with self.assertRaises(dev.shared.budget.BudgetStop):
                        session.ledger.reserve("teacher", [], 1)
                session.close()
            self.assertEqual(len(calls), 2)

    def test_report_counts_generation_errors_and_selects_only_after_three_repeats(self):
        m, _ = fixture()
        # One ambiguous and one unambiguous dev question, generation error must
        # never become a correct unambiguous answer.
        first = m["dev"]["ambrosia"][0]
        second = deepcopy(first)
        second.update(question_id=302, SQL="ambiguous")
        m["dev"]["ambrosia"].append(second)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for job in m["jobs"]:
                folder = root / "jobs" / job["name"]
                (folder / "confirm").mkdir(parents=True)
                dev.store.write_json(
                    folder / "confirm/metadata.json",
                    dict(
                        status="completed",
                        manifest_sha256=m["sha256"],
                        feature_mask=dev.FEATURES,
                        dev_confirmation_version=dev.VERSION,
                    ),
                )
                records = {}
                for value in dev.unique_candidates(job):
                    original = value == job["arms"]["original"]
                    for item in m["dev"]["ambrosia"]:
                        pred = (
                            "error"
                            if original and item is first
                            else "ambiguous"
                            if item is second
                            else "SELECT 1"
                        )
                        records[dev.key(value, item)] = dict(
                            prediction=pred, score=float(pred != "error")
                        )
                dev.store.write_json(folder / "evaluations.json", records)
                dev.report(root, m)
                if job["repeat"] < 3:
                    self.assertFalse((root / "selection.json").exists())
            result = dev.read(root / "selection.json")
            self.assertEqual(result["selected_arm"], "mipro_zero__r1")
            self.assertFalse(result["deployed"])
            original = dev.read(root / "report.json")["rows"][0]
            self.assertEqual(original["accuracy"], 0.5)
            self.assertEqual(original["metrics"]["generation_errors"], 1)

    def test_foreign_results_and_output_paths_rejected(self):
        m, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = m["jobs"][0]
            path = root / "jobs" / job["name"] / "evaluations.json"
            path.parent.mkdir(parents=True)
            dev.store.write_json(path, {})
            with self.assertRaises(ValueError):
                dev.completed(root, job, m)
        for value in [
            dev.source.DEFAULT_OUTPUT,
            dev.shared.DEFAULT_OUTPUT,
            "../ambrosia_prompt_dev_escape",
        ]:
            with self.assertRaises(ValueError):
                dev.output_path(value)

    def test_worker_supervisor_child_is_this_entrypoint(self):
        old_file = dev.shared.__file__
        with dev.runtime_context():
            self.assertEqual(
                Path(dev.shared.__file__).resolve(), Path(dev.__file__).resolve()
            )
            self.assertEqual(dev.shared.FEATURES, [14, 32])
            self.assertIs(dev.shared.run_job, dev.run_job)
            self.assertIs(dev.shared.report, dev.report)
        self.assertEqual(dev.shared.__file__, old_file)


if __name__ == "__main__":
    unittest.main()
