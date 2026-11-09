"""Meaningful offline checks: leakage, actual library APIs, budgets and controls."""

import copy
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prompt_comparison as c
from prompt_comparison_runtime import DualLedger, confirmation_plan
from prompt_pilot_budget import BudgetStop, Ledger
import prompt_pilot_runtime as old


def item(q, db, prefix="fit"):
    return {
        "question_id": q,
        "db_id": db,
        "question": f"{prefix} question {q} asks how many records satisfy the requested condition",
        "evidence": "",
        "SQL": "SELECT COUNT(*) FROM entries",
        "difficulty": "simple",
    }


def manifest():
    fit = [item(i, "fit_" + str(i % 2)) for i in range(1, 9)]
    inner = [item(i, "inner_" + str(i % 2), "inner") for i in range(101, 109)]
    cohorts = {
        s: {"fit": copy.deepcopy(fit), "inner_val": copy.deepcopy(inner)}
        for s in c.SLOTS
    }
    for rows in cohorts["ambiguity"].values():
        for i, row in enumerate(rows):
            if i % 2:
                row["SQL"] = "ambiguous"
    schemas = {
        b: {i["db_id"]: {"rows": [["entries", "id", "INTEGER"]]} for i in fit + inner}
        for b in ("bird", "ambrosia")
    }
    return {
        "sha256": "test",
        "seed_instructions": {s: c.slot_parts(s)[1] for s in c.SLOTS},
        "cohorts": cohorts,
        "schema_audit": {"schemas": schemas},
        "teacher": {"model": "offline"},
        "limits": {**c.LIMITS, "examples": 80, "teacher_calls": 4},
        "dev": {"bird": [item(201, "dev")], "ambrosia": [item(301, "dev_amb")]},
    }


class Contracts(unittest.TestCase):
    def test_seed_compilation_exact_and_format_slots_preserved(self):
        m = manifest()
        for slot in c.SLOTS:
            fit = m["cohorts"][slot]["fit"]
            text, demos = c.render(
                c.candidate(m["seed_instructions"][slot]), m, slot, fit, fit[0]
            )
            expected = (
                c.all_templates()["TAXONOMY_DETECTION_PROMPT"]
                if slot == "ambiguity"
                else c.pilot.templates()["SQL_PROMPT_TEMPLATE"]
            )
            self.assertEqual(text, expected)
            self.assertEqual(demos, [])
            text.format(
                user_query="Question {literal}",
                db_schema="schema {data}",
                sql_dialect="PostgreSQL",
            )

    def test_fewshot_fit_only_no_self_gold_and_braces_are_data(self):
        m = manifest()
        fit = m["cohorts"]["sql"]["fit"]
        fit[0]["question"] = "Training {quoted} example"
        value = c.candidate(m["seed_instructions"]["sql"], ["1", "2"])
        text, demos = c.render(value, m, "sql", fit, fit[0])
        self.assertEqual(demos, ["2"])
        self.assertNotIn(fit[0]["question"], text)
        rendered = text.format(user_query="Current question", sql_dialect="PostgreSQL")
        self.assertIn(fit[1]["question"], rendered)
        self.assertIn("Current question", rendered)
        for forbidden in ("101", "201"):
            with self.assertRaises(ValueError):
                c.check(c.candidate(value["instruction"], [forbidden]), m, "sql", fit)

    def test_current_gold_never_added_in_real_generator_prompt(self):
        import asyncio
        from unittest.mock import AsyncMock

        sys.path.insert(0, str(c.ROOT / "src"))
        from adv_text2sql.mcp_servers.text2sql_tool.src import (
            text2sql_implementation as impl,
        )

        m = manifest()
        fit = m["cohorts"]["sql"]["fit"]
        fit[0]["SQL"] = "SELECT 987654321 AS secret_current_gold"
        template, _ = c.render(
            c.candidate(m["seed_instructions"]["sql"], ["1", "2"]),
            m,
            "sql",
            fit,
            fit[0],
        )
        fake = SimpleNamespace(
            system_prompt="Supplied schema",
            _llm_call_with_retry=AsyncMock(
                return_value=SimpleNamespace(content="SELECT 1")
            ),
            _strip_sql_comments=lambda s: s,
            _validate_sql=lambda s: True,
            _get_accessed_tables=lambda s: [],
        )
        with (
            old.settings(impl, [14, 32]),
            patch.object(impl, "SQL_PROMPT_TEMPLATE", template),
        ):
            asyncio.run(impl.Text2SQLGenerator.generate_sql(fake, fit[0]["question"]))
        messages = fake._llm_call_with_retry.call_args.args[0]
        self.assertNotIn("987654321", " ".join(m.content for m in messages))
        self.assertIn(fit[1]["question"], " ".join(m.content for m in messages))

    def test_split_keeps_same_database_and_question_families_together(self):
        rows = [item(1, "a"), item(2, "b"), item(3, "c"), item(4, "d")]
        rows[0]["question"] = "Show annual quantity for all active students at school"
        rows[1]["question"] = rows[0]["question"]
        rows[2]["question"] = "Which books were written before nineteen eighty"
        rows[3]["question"] = "Calculate total expenses by month and category"
        parts = c.group_split(rows, "bird")
        owner = {
            str(i["question_id"]): part
            for part, values in parts.items()
            for i in values
        }
        self.assertEqual(owner["1"], owner["2"])
        self.assertTrue(parts["fit"] and parts["inner_val"])

    def test_instruction_guards_and_unknown_labels(self):
        m = manifest()
        fit = m["cohorts"]["sql"]["fit"]
        for text in ("Use {user_query}", "SELECT id FROM entries"):
            with self.assertRaises(ValueError):
                c.check(c.candidate(text), m, "sql", fit)
        self.assertIsNone(c.complete_score({}, c.candidate("rules"), fit))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                confirmation_plan(Path(tmp), m)

    def test_zero_shot_cannot_select_fixed_fewshot_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = manifest()
            seed = c.candidate(m["seed_instructions"]["sql"])
            few = c.candidate(seed["instruction"], ["1"])
            records = {}
            for value, score in ((seed, 0.5), (few, 1.0)):
                for i in m["cohorts"]["sql"]["inner_val"]:
                    records[
                        f"search:{c.cid(value)}:hybrid:False:bird:{i['question_id']}"
                    ] = {"score": score}
            session = SimpleNamespace(
                manifest=m,
                slot="sql",
                job={"method": "gepa_zero"},
                benchmark="bird",
                inner=m["cohorts"]["sql"]["inner_val"],
                records=records,
                candidates={c.cid(v): v for v in (seed, few)},
                output=Path(tmp),
            )
            self.assertEqual(c.select(session)["candidate"], seed)


class Budgets(unittest.TestCase):
    def test_global_cap_blocks_network_reservation_and_keeps_local_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "local").mkdir()
            limits = {**c.LIMITS, "teacher_calls": 1}
            local, global_ = Ledger(root / "local", limits), Ledger(root, limits)
            ledger = DualLedger(local, global_, "job")
            ticket = ledger.reserve("teacher", ["x"], 10)
            ledger.settle(ticket, 20, 10)
            with self.assertRaises(BudgetStop):
                ledger.reserve("teacher", ["x"], 10)
            self.assertEqual(local.data["teacher_calls"], 1)
            self.assertEqual(global_.data["tokens"], 30)

    def test_pending_reservation_forbids_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "local").mkdir()
            local, global_ = (
                Ledger(root / "local", c.LIMITS),
                Ledger(root, c.GLOBAL_LIMITS),
            )
            global_.reserve("teacher", ["x"], 10)
            with self.assertRaises(RuntimeError):
                DualLedger(local, global_, "job")


class FakeSession:
    """No socket/DB/API: drives the real library optimizers and their parsers."""

    def __init__(self, folder, method, slot="sql"):
        self.output = folder
        self.folder = folder / "search"
        self.folder.mkdir()
        self.manifest = manifest()
        self.slot = slot
        self.fit = self.manifest["cohorts"][slot]["fit"]
        self.inner = self.manifest["cohorts"][slot]["inner_val"]
        self.benchmark = "bird" if slot == "sql" else "ambrosia"
        self.job = {"method": method, "seed": 42}
        self.ledger = SimpleNamespace(data={"teacher_calls": 0})
        self.records = {}
        self.candidates = {}
        self.prompts = []

    def evaluate(self, item, value):
        c.check(value, self.manifest, self.slot, self.fit)
        self.candidates[c.cid(value)] = value
        row = {
            "score": float(item["question_id"] % 2 == 0),
            "prediction": "SELECT 1",
            "error": None,
        }
        self.records[
            f"search:{c.cid(value)}:hybrid:False:{self.benchmark}:{item['question_id']}"
        ] = row
        return row

    def feedback(self, items, value):
        if any(i not in self.fit for i in items):
            raise AssertionError("Inner/dev reached teacher")
        return [
            {
                "question": i["question"],
                "reference": i["SQL"],
                "result": self.evaluate(i, value),
            }
            for i in items
        ]

    def reflect(self, prompt):
        self.prompts.append(prompt)
        self.ledger.data["teacher_calls"] += 1
        rules = "Use only necessary columns and valid PostgreSQL expressions."
        if self.slot == "ambiguity":
            rules = "Use [DATABASE_SCHEMA] to distinguish genuinely different interpretations."
        # DSPy ignores sections that are not output fields of its signature.
        return (
            "[[ ## observations ## ]]\nCOMPLETE\n[[ ## summary ## ]]\nSQL task with supplied schema.\n"
            "[[ ## proposed_instruction ## ]]\n" + rules + "\n[[ ## completed ## ]]\n"
            "<IMPROVED_VARIABLE>"
            + rules
            + "</IMPROVED_VARIABLE>\n```\n"
            + rules
            + "\n```"
        )


class RealLibrarySmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c.load_packages()
        import dspy  # noqa: F401
        import textgrad  # noqa: F401

    def test_actual_three_libraries_with_fake_provider_and_no_network(self):
        from prompt_comparison_methods import optimize

        for slot, method in ((s, m) for s in c.SLOTS for m in c.METHODS):
            with (
                self.subTest(method=method, slot=slot),
                tempfile.TemporaryDirectory() as tmp,
            ):
                session = FakeSession(Path(tmp), method, slot)
                with patch.object(
                    socket.socket,
                    "connect",
                    side_effect=AssertionError("No network in offline tests"),
                ):
                    optimize(session)
                self.assertTrue(session.records)
                self.assertTrue(session.prompts)
                for item in session.inner:
                    self.assertFalse(
                        any(item["question"] in p for p in session.prompts)
                    )
                if method.endswith("zero"):
                    self.assertTrue(
                        all(not v["demos"] for v in session.candidates.values())
                    )


if __name__ == "__main__":
    unittest.main()
