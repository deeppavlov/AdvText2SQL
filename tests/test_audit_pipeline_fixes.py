"""AUD-002/004/005/010/011/013: real pipeline boundaries with fake external services."""

import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import sqlglot

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from adv_text2sql.ambiguity import InvalidTaxonomy, parse_taxonomy  # noqa: E402
from adv_text2sql.sql_text import clean_sql  # noqa: E402
from adv_text2sql.mcp_servers.text2sql_tool.src import text2sql_implementation as impl  # noqa: E402
from benchmarks.bird import BenchmarkBIRD  # noqa: E402
from benchmarks.ambrosia import BenchmarkAmbrosia  # noqa: E402
from benchmarks.checkpoint import BenchmarkCheckpoint  # noqa: E402
from benchmarks.evaluate_bird import sqlite_to_postgres  # noqa: E402


@contextmanager
def working_directory(folder):
    previous = Path.cwd()
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(previous)


def generator():
    tool = object.__new__(impl.Text2SQLGenerator)
    tool.system_prompt = "PostgreSQL schema: employees(name TEXT)"
    tool.db_schema = "employees(name TEXT)"
    return tool


class SQLPipelineTests(unittest.TestCase):
    def setUp(self):
        self.flags = patch.multiple(impl, **{k: False for k in vars(impl) if k.startswith("FEAT_")})
        self.flags.start()
        self.addCleanup(self.flags.stop)

    def test_real_generate_sql_preserves_semantics_with_and_without_sanitizer(self):
        queries = [
            "SELECT 'DESC', 'ASC', 'a/*b*/c', 'a--b', 'ROUND(x, 2)' FROM t",
            'SELECT "CamelCase", "DESC", "a--b" FROM "MixedTable"',
            "SELECT TO_DATE(text_column, 'DD/MM/YYYY'), ROUND(amount, 2) FROM t ORDER BY amount DESC",
            "SELECT $$DESC /* data */ -- data$$, $tag$ROUND(x, 2)$tag$ FROM t",
            "SELECT E'a\\\'--b', 'it''s /*data*/' FROM t",
            "SELECT CAST(x AS REAL), NULLIF(x, 0) FROM t",
            "SELECT name$tag$, other$identifier FROM t",
        ]
        for sanitize in (False, True):
            for sql in queries:
                with self.subTest(sanitize=sanitize, sql=sql), patch.multiple(impl, FEAT_15=sanitize, FEAT_36=True):
                    tool = generator()
                    tool._llm_call_with_retry = AsyncMock(return_value=SimpleNamespace(content=sql))
                    result = asyncio.run(tool.generate_sql("List values"))
                    self.assertEqual(result["status"], "success")
                    self.assertEqual(result["sql_query"], sql)

    def test_comments_and_fences_are_removed_without_gluing_tokens(self):
        self.assertEqual(clean_sql("```sql\nSELECT/* outer /* inner */ end */name -- note\nFROM t\n```"),
                         "SELECT name  \nFROM t")
        self.assertEqual(clean_sql("SELECT 'a/*b*/c', 'x--y', \"Mixed--Case\" FROM t"),
                         "SELECT 'a/*b*/c', 'x--y', \"Mixed--Case\" FROM t")
        self.assertIn("\n", clean_sql("SELECT 'a'/*\ncomment*/'b' FROM t"))
        self.assertEqual(clean_sql("SELECT [column--name] FROM t", dialect="sqlite"),
                         "SELECT [column--name] FROM t")
        for value in ("SELECT /* unfinished", "SELECT 'unfinished", "SELECT $tag$unfinished"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                clean_sql(value)

    def test_full_answer_is_not_implicitly_limited(self):
        tool = generator()
        for sql in ("SELECT * FROM t", "SELECT 'LIMIT' FROM t", "SELECT * FROM (SELECT * FROM t LIMIT 10) x"):
            with self.subTest(sql=sql), patch.multiple(impl, FEAT_15=True, FEAT_36=True):
                self.assertEqual(tool.sanitize_sql(sql), sql)
        self.assertEqual(tool.sanitize_sql("SELECT * FROM t LIMIT 500"), "SELECT * FROM t LIMIT 500")

    def test_sqlite_reference_conversion_preserves_literals_and_identifiers(self):
        sql = sqlite_to_postgres('SELECT \'a/*b*/c\', \'DESC\', "CamelCase" FROM t')
        self.assertIn("'a/*b*/c'", sql)
        self.assertIn("'DESC'", sql)
        self.assertIn('"CamelCase"', sql)
        division = "SELECT numerator / denominator FROM t"
        self.assertEqual(sqlite_to_postgres(division), sqlglot.transpile(division, read="sqlite", write="postgres")[0])
        self.assertIn("TO_DATE", sqlite_to_postgres("SELECT TO_DATE(value, 'DD/MM/YYYY') FROM t"))
        with self.assertRaises(ValueError):
            sqlite_to_postgres("SELECT 1; SELECT 2")
        with self.assertRaises(ValueError):
            sqlite_to_postgres("SELECT 'unfinished")

    def test_sqlite_reference_is_checked_before_conversion_in_its_own_dialect(self):
        from adv_text2sql.sql_safety import UnsafeSQL, validate_read_query

        reference = "SELECT `Full Name` FROM `Student Records`"
        validate_read_query(reference, dialect="sqlite")
        with self.assertRaises(UnsafeSQL):
            validate_read_query(reference)
        converted = sqlite_to_postgres(reference)
        self.assertIn('"Full Name"', converted)
        validate_read_query(converted)
        for unsafe in ("SELECT 1; DELETE FROM `t`", "WITH x AS (DELETE FROM t RETURNING id) SELECT * FROM x",
                       "SELECT set_config('transaction_read_only','off',false)"):
            with self.subTest(unsafe=unsafe), self.assertRaises(UnsafeSQL):
                sqlite_to_postgres(unsafe)

    def test_full_generation_to_evaluation_keeps_more_than_101_rows(self):
        from benchmarks import evaluate_bird

        tool = generator()
        tool.relationships_str = tool.column_stats_str = ""
        tool._llm_call_with_retry = AsyncMock(side_effect=[SimpleNamespace(content="OK"),
                                                          SimpleNamespace(content="SELECT value FROM t")])
        with patch.multiple(impl, FEAT_15=True, FEAT_36=True):
            response = asyncio.run(tool.query("List all values"))
        engine = MagicMock()
        engine.dialect.name = "postgresql"
        connection = engine.connect.return_value.__enter__.return_value
        connection.execute.return_value.fetchall.return_value = [(n,) for n in range(150)]
        with tempfile.TemporaryDirectory() as tmp, working_directory(tmp):
            gold = Path("gold.json")
            gold.write_text(json.dumps([{"question_id": 1, "db_id": "db", "difficulty": "simple",
                                         "SQL": "SELECT value FROM t"}]))
            with patch.dict(os.environ, {"DB_USER": "reader", "DB_PASS": "dummy"}), \
                    patch.object(evaluate_bird, "create_engine", return_value=engine):
                report = evaluate_bird.run_evaluation({"1": response["query"]}, gold, "localhost:5432")
            self.assertEqual(report["overall_accuracy"], 100)
            self.assertEqual(len(json.loads(Path("all_predicted_results.json").read_text())["1"]), 150)
        self.assertTrue(all("LIMIT" not in str(call.args[0]) for call in connection.execute.call_args_list))


class TaxonomyPipelineTests(unittest.TestCase):
    def setUp(self):
        flags = {k: False for k in vars(impl) if k.startswith("FEAT_")}
        flags["FEAT_32"] = True
        self.flags = patch.multiple(impl, **flags)
        self.flags.start()
        self.addCleanup(self.flags.stop)

    def test_invalid_json_never_becomes_an_unambiguous_verdict(self):
        invalid = ["{}", "[]", "null", "broken", '{"is_ambiguous": "false", "ambiguity_type": null, "reason": ""}',
                   '{"is_ambiguous": 0, "ambiguity_type": null, "reason": ""}',
                   '{"is_ambiguous": true, "ambiguity_type": "A1", "reason": 123}',
                   '{"is_ambiguous": false, "ambiguity_type": "A1", "reason": ""}',
                   '{"is_ambiguous": true, "ambiguity_type": "A9", "reason": "x"}',
                   '{"is_ambiguous": false, "is_ambiguous": true, "ambiguity_type": null, "reason": ""}',
                   'Prose {"is_ambiguous": false, "ambiguity_type": null, "reason": ""}']
        for content in invalid:
            with self.subTest(content=content):
                tool = generator()
                tool._llm_call_with_retry = AsyncMock(return_value=SimpleNamespace(content=content))
                tool.generate_sql = AsyncMock()
                tool._check_ambiguity = AsyncMock()
                result = asyncio.run(tool.query("List employee names"))
                self.assertEqual(result["status"], "error")
                tool.generate_sql.assert_not_called()
                tool._check_ambiguity.assert_not_called()

    def test_valid_positive_verdict_stops_before_sql_generation(self):
        tool = generator()
        tool._llm_call_with_retry = AsyncMock(return_value=SimpleNamespace(
            content='{"is_ambiguous": true, "ambiguity_type": "A1", "reason": "Two plausible columns"}'))
        tool.generate_sql = AsyncMock()
        self.assertEqual(asyncio.run(tool.query("List employee names"))["status"], "ambiguous")
        tool.generate_sql.assert_not_called()

    def test_valid_negative_verdict_continues_through_real_generation(self):
        tool = generator()
        tool._llm_call_with_retry = AsyncMock(side_effect=[
            SimpleNamespace(content='```json\n{"is_ambiguous": false, "ambiguity_type": null, "reason": null}\n```'),
            SimpleNamespace(content='SELECT name FROM employees')])
        result = asyncio.run(tool.query("List employee names"))
        self.assertEqual(result, {"status": "success", "query": "SELECT name FROM employees"})
        self.assertEqual(tool._llm_call_with_retry.await_count, 2)

    def test_contract_rejects_missing_fields_and_accepts_null_negative_reason(self):
        with self.assertRaises(InvalidTaxonomy):
            parse_taxonomy('{"is_ambiguous": false}')
        self.assertEqual(parse_taxonomy('{"is_ambiguous": false, "ambiguity_type": null, "reason": null}')["reason"], "")
        self.assertEqual(parse_taxonomy('{"is_ambiguous": true, "ambiguity_type": "A1", "reason": null}')["reason"], "")


class CheckpointPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.dataset = self.folder / "questions.json"
        self.items = [{"question_id": 1, "db_id": "db", "question": "List names", "evidence": "", "SQL": "SELECT name FROM t"}]
        self.dataset.write_text(json.dumps(self.items))

    def benchmark(self, cls=BenchmarkBIRD, model="one"):
        return cls(db_url="unused", query_file=str(self.dataset), answer_file=str(self.dataset),
                   llm_client=SimpleNamespace(_raw_config={"model": model, "api_key": "dummy-secret"}),
                   checkpoint_context={"model": model})

    def test_checkpoint_rejects_model_task_prompt_flag_and_dataset_changes(self):
        checkpoint = BenchmarkCheckpoint(self.benchmark(), self.folder)
        checkpoint.save({"1": "SELECT name FROM t"})
        self.assertEqual(checkpoint.load(), {"1": "SELECT name FROM t"})
        for benchmark in (self.benchmark(model="two"), self.benchmark(BenchmarkAmbrosia)):
            with self.assertRaises(ValueError):
                BenchmarkCheckpoint(benchmark, self.folder).load()
        with patch.object(impl, "FEAT_14", not impl.FEAT_14), self.assertRaises(ValueError):
            BenchmarkCheckpoint(self.benchmark(), self.folder).load()
        with patch.object(impl, "SQL_PROMPT_TEMPLATE", "Different prompt"), self.assertRaises(ValueError):
            BenchmarkCheckpoint(self.benchmark(), self.folder).load()
        self.items[0]["question"] = "A different question with the same ID"
        self.dataset.write_text(json.dumps(self.items))
        with self.assertRaises(ValueError):
            BenchmarkCheckpoint(self.benchmark(), self.folder).load()
        self.assertNotIn("dummy-secret", checkpoint.envelope.read_text())

    def test_legacy_and_tampered_checkpoint_are_rejected(self):
        checkpoint = BenchmarkCheckpoint(self.benchmark(), self.folder)
        checkpoint.export.write_text('{"1": "SELECT 1"}')
        with self.assertRaisesRegex(ValueError, "no verified identity"):
            checkpoint.load()
        checkpoint.save({"1": "SELECT name FROM t"})
        checkpoint.export.write_text('{"1": "SELECT 2"}')
        with self.assertRaisesRegex(ValueError, "export differs"):
            checkpoint.load()
        value = json.loads(checkpoint.envelope.read_text())
        value["predictions"]["1"] = "SELECT 3"
        checkpoint.envelope.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "hash changed"):
            checkpoint.load()

    def test_interrupted_export_retains_answer_and_requires_explicit_repair(self):
        import benchmarks.checkpoint as storage

        checkpoint = BenchmarkCheckpoint(self.benchmark(), self.folder)
        write = storage.atomic_json

        def interrupted(path, value):
            if path == checkpoint.export:
                raise OSError("Interrupted compatibility export")
            return write(path, value)

        predictions = {"1": "SELECT name FROM t"}
        with patch.object(storage, "atomic_json", side_effect=interrupted), self.assertRaises(OSError):
            checkpoint.save(predictions)
        self.assertEqual(json.loads(checkpoint.envelope.read_text())["predictions"], predictions)
        with self.assertRaisesRegex(ValueError, "export differs"):
            checkpoint.load()
        write(checkpoint.export, predictions)
        self.assertEqual(checkpoint.load(), predictions)

    def test_base_predict_resumes_only_its_own_question_and_configuration(self):
        tool = SimpleNamespace(query=AsyncMock(return_value={"status": "success", "query": "SELECT name FROM t"}))
        with working_directory(self.folder), patch("benchmarks.bird.asyncio.sleep", new=AsyncMock()):
            benchmark = self.benchmark()
            first = asyncio.run(benchmark.predict({"db": tool}))
            self.assertEqual(asyncio.run(benchmark.predict({})), first)
            self.assertEqual(tool.query.await_count, 1)
            with self.assertRaises(ValueError):
                asyncio.run(self.benchmark(BenchmarkAmbrosia).predict({"db": tool}))
            self.assertEqual(tool.query.await_count, 1)


if __name__ == "__main__":
    unittest.main()
