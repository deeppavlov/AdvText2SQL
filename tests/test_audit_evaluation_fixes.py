"""AUD-007/008: exercise real evaluators and label importers with fake I/O."""

import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from adv_text2sql.evaluation import (
    ambrosia_report, database_failure, recorded_database_failure, sqlstate_failure_kind,
)
from benchmarks import evaluate_ambrosia, evaluate_bird
from benchmarks.ambrosia import BenchmarkAmbrosia
import build_configuration_dataset as labels
import build_feature_dataset as legacy_labels
from adv_text2sql.evaluation import validate_collected_observation
from sqlalchemy.exc import DBAPIError


def db_error(code, statement="SELECT 2", invalidated=False):
    original = Exception("private driver detail")
    original.sqlstate = code
    return DBAPIError(statement, {}, original, connection_invalidated=invalidated)


class AmbrosiaEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.items = [{"question_id": 1, "SQL": "SELECT 1", "db_id": "a", "question": "one"},
                      {"question_id": 2, "SQL": "ambiguous", "db_id": "b", "question": "two"}]
        self.gold = self.folder / "gold.json"
        self.gold.write_text(json.dumps(self.items), encoding="utf-8")

    def test_base_and_research_evaluators_share_strict_error_scoring(self):
        for predictions in ({"1": "error", "2": "ambiguous"}, {"1": "SELECT 1", "2": "error"}):
            with self.subTest(predictions=predictions):
                report = evaluate_ambrosia.run_evaluation(predictions, self.gold)
                self.assertEqual(report, ambrosia_report(self.items, predictions))
                self.assertEqual(report["accuracy"], 50)
                self.assertEqual(report["generation_errors"], 1)
        report = ambrosia_report(self.items, {"1": "error", "2": "error"})
        self.assertEqual(report["accuracy"], 0)
        self.assertEqual(report["ambiguity_metrics"]["tn"], 0)
        self.assertEqual(report["ambiguity_metrics"]["fn"], 1)

    def test_missing_foreign_duplicate_and_blank_predictions_are_rejected(self):
        for predictions in ({"1": "SELECT 1"}, {"1": "SELECT 1", "2": "ambiguous", "3": "SELECT 3"},
                            {"1": "", "2": "ambiguous"}):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                evaluate_ambrosia.run_evaluation(predictions, self.gold)
        with self.assertRaises(ValueError):
            ambrosia_report(self.items * 2, {"1": "SELECT 1", "2": "ambiguous"})

    def test_single_class_and_empty_datasets_have_defined_metrics(self):
        self.assertIsNone(ambrosia_report(self.items[:1], {"1": "SELECT 1"})["ambiguity_metrics"]["recall"])
        self.assertEqual(ambrosia_report([], {})["accuracy"], 0)

    def test_legacy_label_export_is_strict_and_requires_full_coverage(self):
        path = self.folder / "predictions.json"
        path.write_text(json.dumps({"1": "error", "2": "ambiguous"}))
        gold = {str(i["question_id"]): i for i in self.items}
        self.assertEqual(legacy_labels.parse_scores_from_query_results(path, gold), {"1": 0, "2": 1})
        path.write_text(json.dumps({"1": "error"}))
        with self.assertRaises(ValueError):
            legacy_labels.parse_scores_from_query_results(path, gold)

    def test_missing_database_stops_before_paid_generation(self):
        benchmark = BenchmarkAmbrosia(db_url="unused", query_file=str(self.gold), answer_file=str(self.gold))
        tool = SimpleNamespace(query=AsyncMock())
        with patch("benchmarks.ambrosia.os.path.exists", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "database tools unavailable"):
                asyncio.run(benchmark.predict({"a": tool}))
        tool.query.assert_not_called()


class DatabaseLabelTests(unittest.TestCase):
    def test_sqlstate_classification(self):
        for code in ("08006", "28P01", "3D000", "57P01", "57014", "42501", "53100", "40001"):
            with self.subTest(code=code):
                self.assertEqual(sqlstate_failure_kind(code), "infrastructure")
        for code in ("42601", "42703", "42P01", "22012", "22007"):
            with self.subTest(code=code):
                self.assertEqual(sqlstate_failure_kind(code), "query")
        for code in (None, "", "weird", "00000", "23505"):
            self.assertEqual(sqlstate_failure_kind(code), "unknown")

    def test_live_connection_invalidated_overrides_query_code(self):
        self.assertEqual(database_failure(db_error("42601", invalidated=True))["failure_kind"], "infrastructure")
        self.assertEqual(database_failure(db_error(None))["failure_kind"], "unknown")

    def test_archived_connection_failure_is_masked_despite_matching_prediction(self):
        items = [{"question_id": 1, "SQL": "SELECT 1"}]
        row = {"question_id": 1, "score": 0, "predicted_sql": "SELECT 2", "gold_sql": "SELECT 1",
               "error": "(psycopg.OperationalError) connection lost\n[SQL: SELECT 2]"}
        observed = labels.observations(items, {"1": "SELECT 2"}, {"results": [row]}, "bird")["1"]
        self.assertFalse(observed["observed"])
        self.assertIsNone(observed["correct"])
        self.assertEqual(observed["evaluation_error_kind"], "infrastructure")

    def test_unknown_archive_error_is_not_inferred_from_sql(self):
        row = {"error": "SQL error\n[SQL: SELECT 2]", "gold_sql": "SELECT 1", "predicted_sql": "SELECT 2"}
        self.assertEqual(labels.execution_failure_kind(row), "unknown")
        row["error"] = "SQL error\n[SQL: SELECT '(psycopg.errors.SyntaxError)']"
        self.assertEqual(recorded_database_failure(row), "unknown")

    def test_legacy_bird_export_rejects_infrastructure_instead_of_zero_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bird.log"
            row = {"question_id": 1, "score": 0, "gold_sql": "SELECT 1", "predicted_sql": "SELECT 2",
                   "error": "(psycopg.OperationalError) connection lost\n[SQL: SELECT 2]"}
            path.write_text(repr([row]))
            with self.assertRaisesRegex(ValueError, "Untrusted BIRD"):
                legacy_labels.parse_scores_from_log(path)

    def test_prompt_evaluator_also_stops_on_resource_and_permission_errors(self):
        import prompt_pilot_runtime as runtime
        from run_openrouter_baseline import ProviderUnavailable

        session = object.__new__(runtime.Session)
        session.engines = {"db": object()}
        session.golds = {"1": {(1,)}}
        item = {"question_id": 1, "db_id": "db"}
        conn = MagicMock()

        @contextmanager
        def connection(*args):
            yield conn

        for code in ("08006", "57014", "42501", None):
            conn.execute.side_effect = db_error(code)
            with self.subTest(code=code), patch("adv_text2sql.sql_safety.read_only_connection", connection):
                with self.assertRaises(ProviderUnavailable):
                    session.score(item, "SELECT 2", "bird")

    def test_known_historical_query_error_and_reference_error_are_distinguished(self):
        row = {"error": "(psycopg.errors.UndefinedColumn) missing\n[SQL: SELECT broken]",
               "gold_sql": "SELECT 1", "predicted_sql": "SELECT broken"}
        self.assertEqual(labels.execution_failure_kind(row), "prediction")
        row["gold_sql"] = row["predicted_sql"]
        self.assertEqual(labels.execution_failure_kind(row), "reference")

    def test_new_structured_errors_do_not_fall_back_to_text(self):
        row = {"error": "(psycopg.errors.SyntaxError) example", "failure_kind": "infrastructure", "sqlstate": "42601"}
        self.assertEqual(recorded_database_failure(row), "infrastructure")
        row.update(failure_kind="unknown", sqlstate=None)
        self.assertEqual(recorded_database_failure(row), "unknown")

    def test_v2_import_masks_infrastructure_and_preserves_model_errors(self):
        item = {"question_id": 1, "db_id": "db", "SQL": "SELECT 1"}
        row = {"question_id": 1, "db_id": "db", "score": 0, "prediction": "SELECT 2"}
        for code in ("08006", "57014", "42501", "53100"):
            row["error"] = f"postgresql:{code}:Error"
            self.assertIsNone(validate_collected_observation(row, item, "bird"))
        row["error"] = "postgresql:42703:UndefinedColumn"
        self.assertEqual(validate_collected_observation(row, item, "bird"), 0)


class BirdEvaluationTests(unittest.TestCase):
    def run_evaluator(self, failing_stage, code):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        gold = Path(tmp.name) / "gold.json"
        gold.write_text(json.dumps([{"question_id": 1, "SQL": "SELECT 1", "db_id": "db", "difficulty": "simple"}]))
        conn = MagicMock()
        conn.execute.side_effect = ([db_error(code, "SELECT 1")] if failing_stage == "reference" else
                                    [SimpleNamespace(fetchall=lambda: [(1,)]), db_error(code)])

        @contextmanager
        def connection(*args):
            yield conn

        with patch.dict("os.environ", {"DB_USER": "reader", "DB_PASS": "dummy"}), \
                patch.object(evaluate_bird, "create_engine"), \
                patch.object(evaluate_bird, "read_only_connection", connection), \
                patch("builtins.open", wraps=open) as files:
            result = evaluate_bird.run_evaluation({"1": "SELECT 2"}, gold, "unused")
            writes = [c for c in files.call_args_list if len(c.args) > 1 and c.args[1] == "w"]
            return result, writes

    def test_base_evaluator_stops_on_infrastructure_or_unknown_failure(self):
        for code in ("08006", "57014", None):
            with self.subTest(code=code), self.assertRaises(evaluate_bird.EvaluationUnavailable):
                self.run_evaluator("prediction", code)

    def test_base_evaluator_stops_on_reference_query_error(self):
        with self.assertRaises(evaluate_bird.EvaluationUnavailable):
            self.run_evaluator("reference", "42703")

    def test_base_evaluator_keeps_structured_prediction_sql_error(self):
        # Artifact files are intercepted; the real gold file remains readable.
        real_open = open

        def file_open(path, mode="r", *args, **kwargs):
            if mode == "w":
                return unittest.mock.mock_open()()
            return real_open(path, mode, *args, **kwargs)

        with patch("builtins.open", side_effect=file_open):
            report, _ = self.run_evaluator("prediction", "42703")
        row = report["results"][0]
        self.assertEqual(row["score"], 0)
        self.assertEqual(row["execution_stage"], "prediction")
        self.assertEqual(row["sqlstate"], "42703")
        self.assertNotIn("private driver detail", row["error"])


if __name__ == "__main__":
    unittest.main()
