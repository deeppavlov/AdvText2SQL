import asyncio
from contextvars import ContextVar
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from sql_output_contract import guarded_response, sql_output_issue  # noqa:E402
from audit_question_schemas import reference_issues, sql_references  # noqa:E402
from guarded_hybrid_runtime import guarded_predict, validate_live_schemas  # noqa:E402
import guarded_hybrid_search as suite  # noqa:E402
from static_hybrid_runtime import checkpoint  # noqa:E402


class OutputContractTests(unittest.TestCase):
    def test_explicit_control_order_is_respected_in_new_protocol(self):
        manifest = {
            "cases": [
                {"name": "baseline_control"},
                {"name": "hybrid_plus_2"},
                {"name": "hybrid_control_repeat"},
            ]
        }
        names = ["baseline_control", "hybrid_control_repeat", "hybrid_plus_2"]
        self.assertEqual(
            [c["name"] for c in suite.select_cases(manifest, names, None)], names
        )

    def test_refusal_blank_comment_and_unstructured_responses_become_errors(self):
        for value in (
            "",
            "   ",
            "/* Cannot answer */",
            "-- No matching table",
            "Нет подходящей таблицы.",
            "The database does not contain this information.",
            "WITH this information I cannot answer",
            "ambiguous",
            None,
        ):
            response, issue = guarded_response({"status": "success", "query": value})
            self.assertEqual(response["status"], "error")
            self.assertIsNotNone(issue)
        for response in (None, {"status": "unknown"}, {"query": "SELECT 1"}):
            self.assertEqual(guarded_response(response)[1], "invalid_tool_response")

    def test_real_sql_literals_comments_ctes_and_values_are_preserved(self):
        for value in (
            "SELECT '-- not a comment' AS x",
            "SELECT '/* text */' AS x",
            "/* prefix */ SELECT name FROM things;",
            "WITH x AS (SELECT name FROM things) SELECT name FROM x",
            "WITH RECURSIVE x(n) AS (VALUES(1)) SELECT n FROM x",
            "VALUES (1)",
            "(SELECT 1)",
        ):
            response = {"status": "success", "query": value}
            self.assertEqual(guarded_response(response), (response, None))

    def test_multiple_statements_and_mutating_ctes_are_not_answers(self):
        for value in (
            "SELECT 1; SELECT 2",
            "DELETE FROM t",
            "WITH x AS (DELETE FROM t RETURNING id) SELECT id FROM x",
            "SELECT * INTO new_table FROM old_table",
        ):
            self.assertIsNotNone(sql_output_issue(value))

    def test_contract_does_not_absorb_the_optional_feat16_parser(self):
        # Deliberately malformed SQL still has SQL shape: FEAT_16/execution decides syntax.
        self.assertIsNone(sql_output_issue("SELECT name FROM"))
        self.assertEqual(
            guarded_response({"status": "ambiguous"}), ({"status": "ambiguous"}, None)
        )


class SchemaAuditTests(unittest.TestCase):
    def test_rental_properties_query_cannot_use_a_student_schema(self):
        result = reference_issues(
            "SELECT p.Street_Address,p.City FROM Properties p JOIN Transactions t ON p.id=t.property_id",
            [("students", "city", "TEXT")],
        )
        self.assertEqual(result["status"], "mismatch")
        self.assertTrue(any("properties" in i for i in result["issues"]))

    def test_nested_columns_are_resolved_in_their_own_scope(self):
        rows = [
            ("cards", "setcode", "TEXT"),
            ("sets", "code", "TEXT"),
            ("sets", "name", "TEXT"),
        ]
        result = reference_issues(
            "SELECT c.setCode FROM cards c WHERE c.setCode IN (SELECT code FROM sets WHERE name='X')",
            rows,
        )
        self.assertEqual(result["status"], "reference_consistent")

    def test_sqlite_backticks_and_double_quoted_values_are_not_missing_columns(self):
        result = reference_issues(
            'SELECT `School Code` FROM schools WHERE type="Hotel"',
            [("schools", "School Code", "TEXT"), ("schools", "type", "TEXT")],
        )
        self.assertEqual(result["status"], "reference_consistent")
        self.assertIn("sqlite_double_quoted_literal", result["notes"])
        result = reference_issues(
            'SELECT "Missing Column" FROM schools', [("schools", "name", "TEXT")]
        )
        self.assertEqual(result["status"], "mismatch")

    def test_missing_qualified_column_and_unknown_reference_remain_distinct(self):
        self.assertEqual(
            reference_issues(
                "SELECT t.unknown FROM things t", [("things", "name", "TEXT")]
            )["status"],
            "mismatch",
        )
        self.assertEqual(
            reference_issues("not a query", [("things", "name", "TEXT")])["status"],
            "review",
        )
        refs, issue = sql_references(
            {
                "SQL": "ambiguous",
                "ambig_queries": "['SELECT a FROM things', 'SELECT b FROM things']",
            }
        )
        self.assertEqual(len(refs), 2)
        self.assertIsNone(issue)


class GuardedRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_question_is_checkpointed_as_error_and_next_question_runs_without_retry(
        self,
    ):
        calls = []

        class Tool:
            async def query(self, question):
                calls.append(question)
                return {
                    "status": "success",
                    "query": "/* no SQL */"
                    if question == "Bad"
                    else "SELECT name FROM t",
                }

        items = [
            {"question_id": i, "question": q, "db_id": "x"}
            for i, q in [(1, "Bad"), (2, "Good")]
        ]
        with tempfile.TemporaryDirectory(
            dir=ROOT / "research_results", prefix=".guard_test_"
        ) as tmp:
            folder = Path(tmp)
            predictions = {}
            await guarded_predict(
                items,
                {"x": Tool()},
                predictions,
                folder,
                "ambrosia",
                ContextVar("q"),
                lambda _: None,
                sleep=lambda _: asyncio.sleep(0),
            )
            self.assertEqual(calls, ["Bad", "Good"])
            self.assertEqual(
                checkpoint(folder, {"1", "2"}),
                {"1": "error", "2": "SELECT name FROM t"},
            )
            self.assertTrue((folder / "invalid_outputs.jsonl").exists())
            # Resume never asks the failed completed question again.
            await guarded_predict(
                items,
                {},
                predictions,
                folder,
                "ambrosia",
                ContextVar("q2"),
                lambda _: None,
                sleep=lambda _: asyncio.sleep(0),
            )
            self.assertEqual(len(calls), 2)

    async def test_transport_failure_is_not_converted_into_a_model_label(self):
        from run_openrouter_baseline import ProviderUnavailable

        class Tool:
            async def query(self, q):
                raise ProviderUnavailable("Account unavailable")

        with tempfile.TemporaryDirectory(
            dir=ROOT / "research_results", prefix=".guard_test_"
        ) as tmp:
            with self.assertRaises(ProviderUnavailable):
                await guarded_predict(
                    [{"question_id": 1, "question": "Q", "db_id": "x"}],
                    {"x": Tool()},
                    {},
                    Path(tmp),
                    "ambrosia",
                    ContextVar("q"),
                    lambda _: None,
                )
            self.assertFalse((Path(tmp) / "query_results.json").exists())

    async def test_schema_gate_is_explicit_and_does_not_silently_drop_questions(self):
        manifest = {
            "cohorts": {"train": {"ambrosia": [{"question_id": 267}]}},
            "schema_audit": {"blocked_questions": {"ambrosia": {"267": {}}}},
        }
        with self.assertRaisesRegex(ValueError, "Schema gate"):
            suite.gate(manifest, "train", ["ambrosia"])
        self.assertEqual(len(manifest["cohorts"]["train"]["ambrosia"]), 1)

    async def test_live_catalog_mismatch_blocks_before_generation(self):
        from unittest.mock import MagicMock

        engine = MagicMock()
        inspector = MagicMock()
        inspector.get_table_names.return_value = ["students"]
        inspector.get_columns.return_value = [{"name": "name", "type": "TEXT"}]
        with (
            patch.dict(
                "os.environ", {"DB_USER": "dummy", "DB_PASS": "not-a-real-password"}
            ),
            patch("baseline_connection.database_engine", return_value=engine),
            patch("sqlalchemy.inspect", return_value=inspector),
        ):
            with self.assertRaisesRegex(ValueError, "Live schema changed"):
                validate_live_schemas(
                    [{"db_id": "x"}],
                    {"x": {"rows": [["properties", "address", "TEXT"]]}},
                )
            engine.dispose.assert_called_once()


if __name__ == "__main__":
    unittest.main()
