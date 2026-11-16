"""BIRD upstream EX parity and real evaluator boundaries; no API/DB access.

Oracle: https://github.com/bird-bench/mini_dev/blob/main/evaluation/evaluation_ex.py
calculate_ex uses exact set equality (inspected 2026-11-16).
"""

from contextlib import contextmanager
from decimal import Decimal
import itertools
import json
import os
from pathlib import Path
import random
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from adv_text2sql.bird_results import (  # noqa: E402
    EX_POLICY,
    bird_column_types,
    compare_bird_results,
    normalize_bird_rows,
    prepare_bird_evaluation,
    _numeric_bijection,
)
from adv_text2sql.sql_safety import ReadOnlySetupError  # noqa: E402
from benchmarks import evaluate_bird  # noqa: E402
import prompt_pilot_runtime as runtime  # noqa: E402


class BirdExactContractTests(unittest.TestCase):
    def test_official_set_oracle_on_actual_sqlite_results(self):
        queries = [
            "SELECT 1",
            "SELECT 1.0",
            "SELECT '1'",
            "SELECT NULL",
            "SELECT 0.5",
            "SELECT 1,2",
            "SELECT 2,1",
            "SELECT 1 UNION ALL SELECT 1",
            "SELECT 1 UNION ALL SELECT 2",
            "SELECT 2 UNION ALL SELECT 1",
            "SELECT 1 WHERE 0",
            "SELECT CAST(139 AS REAL)/689",
            "SELECT 0.20174165",
            "SELECT 9007199254740992",
            "SELECT 9007199254740993",
        ]
        with sqlite3.connect(":memory:") as connection:
            results = [connection.execute(q).fetchall() for q in queries]
        for predicted, gold in itertools.product(results, repeat=2):
            with self.subTest(predicted=predicted, gold=gold):
                self.assertEqual(
                    compare_bird_results(predicted, gold)["exact_equal"],
                    set(predicted) == set(gold),
                )

    def test_decimal_is_not_lossily_converted_to_float(self):
        for a, b in [
            (Decimal("0.1"), 0.1),
            (Decimal("9007199254740993"), 9007199254740992.0),
            (Decimal("1.0000000000000000001"), Decimal("1")),
        ]:
            with self.subTest(a=a, b=b):
                self.assertFalse(compare_bird_results([(a,)], [(b,)])["exact_equal"])
        self.assertTrue(
            compare_bird_results([(Decimal("0.5"),)], [(0.5,)])["exact_equal"]
        )

    def test_rows_columns_null_and_strings_keep_upstream_semantics(self):
        for pred, gold, expected in [
            ([(1,), (1,)], [(1,)], True),
            ([(2,), (1,)], [(1,), (2,)], True),
            ([(1, 2)], [(2, 1)], False),
            ([(1, None)], [(1,)], False),
            ([(None,)], [], False),
            ([("A",)], [("a",)], False),
            ([("1",)], [(1,)], False),
            ([(1,), (2,)], [(1,)], False),
        ]:
            with self.subTest(pred=pred, gold=gold):
                self.assertEqual(
                    compare_bird_results(pred, gold)["exact_equal"], expected
                )

    def test_pg_array_json_extension_remains_exact(self):
        self.assertEqual(
            normalize_bird_rows([(1, [2, 3], {"a": 4})]), {(1, (2, 3), (("a", 4),))}
        )
        report = compare_bird_results(
            [([0.10000001],)], [([0.1],)], reference_types=(700,)
        )
        self.assertFalse(report["numeric_diagnostic_equal"])


class NumericWarningTests(unittest.TestCase):
    def test_observed_1473_difference_is_warning_not_a_correct_score(self):
        report = compare_bird_results(
            [(459.9562642112432,)], [(459.95626421124325,)], reference_types=(701,)
        )
        self.assertEqual(report["evaluation_policy"], EX_POLICY)
        self.assertFalse(report["exact_equal"])
        self.assertTrue(report["numeric_precision_warning"])

    def test_observed_34_difference_requires_real_metadata_and_never_awards_ex(self):
        pred, gold = [(Decimal("0.20174165457184325109"),)], [(0.20174165,)]
        report = compare_bird_results(
            pred, gold, predicted_types=(1700,), reference_types=(700,)
        )
        self.assertFalse(report["exact_equal"])
        self.assertTrue(report["numeric_precision_warning"])
        self.assertFalse(compare_bird_results(pred, gold)["numeric_precision_warning"])

    def test_close_but_wrong_numbers_are_never_accepted_as_ex(self):
        for value in [Decimal("0.201741655"), Decimal("0.201742"), Decimal("0.22")]:
            with self.subTest(value=value):
                self.assertFalse(
                    compare_bird_results(
                        [(value,)], [(0.20174165,)], reference_types=(700,)
                    )["exact_equal"]
                )

    def test_exact_integer_decimal_null_strings_zero_and_nonfinite_are_not_softened(
        self,
    ):
        pairs = [
            (1, 1.000000000000001),
            (2**60, 2**60 + 1),
            (9007199254740992.0, Decimal("9007199254740993")),
            (Decimal("0.123456789"), Decimal("0.123456788")),
            (None, 0.0),
            ("0.1", 0.1),
            (0.0, 1e-30),
            (float("nan"), 0.1),
            (float("inf"), 1e300),
            (0.2, 0.3),
        ]
        for a, b in pairs:
            with self.subTest(a=a, b=b):
                report = compare_bird_results(
                    [(a,)], [(b,)], predicted_types=(700,), reference_types=(700,)
                )
                self.assertFalse(report["numeric_diagnostic_equal"])
                self.assertFalse(report["exact_equal"])

    def test_matching_is_one_to_one_and_not_greedy(self):
        report = compare_bird_results(
            [(0.10000001,), (0.10000002,)],
            [(0.1,), (0.100000015,)],
            predicted_types=(700,),
            reference_types=(700,),
        )
        self.assertTrue(report["numeric_diagnostic_equal"])
        self.assertFalse(
            compare_bird_results(
                [(0.100000001,), (0.100000002,)],
                [(0.1,), (0.3,)],
                reference_types=(700,),
            )["numeric_diagnostic_equal"]
        )
        self.assertFalse(
            compare_bird_results(
                [(0.100000001,)], [(0.1,), (0.100000002,)], reference_types=(700,)
            )["numeric_diagnostic_equal"]
        )

    def test_matching_agrees_with_exhaustive_bijection_on_small_sets(self):
        rng = random.Random(42)
        for _ in range(25):
            a = {(0.1 + rng.randrange(8) * 1e-9,) for _ in range(4)}
            b = {(0.1 + rng.randrange(8) * 1e-9,) for _ in range(4)}
            from adv_text2sql.bird_results import _row_near

            expected = len(a) == len(b) and any(
                all(_row_near(x, y, (700,), (700,)) for x, y in zip(a, p))
                for p in itertools.permutations(b)
            )
            self.assertEqual(_numeric_bijection(a, b, (700,), (700,)), expected)

    def test_large_diagnostic_can_be_inconclusive_without_changing_exact_score(self):
        pred = [(i + 0.1,) for i in range(317)]
        gold = [(i + 0.2,) for i in range(317)]
        report = compare_bird_results(pred, gold)
        self.assertIsNone(report["numeric_diagnostic_equal"])
        self.assertFalse(report["exact_equal"])


class RealEvaluatorBoundaryTests(unittest.TestCase):
    def test_warning_does_not_turn_into_positive_classifier_label(self):
        from adv_text2sql.evaluation import validate_collected_observation

        item = {"question_id": 34, "db_id": "db", "SQL": "SELECT 0.20174165"}
        row = {
            "question_id": "34",
            "db_id": "db",
            "prediction": "SELECT 139::numeric/689",
            "score": 0.0,
            "error": None,
            "bird_comparison": {
                "numeric_diagnostic_equal": True,
                "numeric_precision_warning": True,
            },
        }
        self.assertEqual(validate_collected_observation(row, item, "bird"), 0)

    def test_research_setup_failure_never_creates_a_model_label(self):
        connection = MagicMock()
        connection.exec_driver_sql.side_effect = RuntimeError("private driver details")

        @contextmanager
        def boundary(*args):
            yield connection

        session = object.__new__(runtime.Session)
        session.engines = {"db": object()}
        session.golds = {"1": {(1,)}}
        with patch("adv_text2sql.sql_safety.read_only_connection", boundary):
            with self.assertRaises(ReadOnlySetupError):
                session.score({"question_id": 1, "db_id": "db"}, "SELECT 1", "bird")
        self.assertIsNone(session.last_bird_comparison)
        connection.execute.assert_not_called()

    def test_execution_setup_is_local_and_fails_before_any_select(self):
        connection = MagicMock()
        prepare_bird_evaluation(connection)
        self.assertEqual(
            [c.args[0] for c in connection.exec_driver_sql.call_args_list],
            [
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ",
                "SET LOCAL max_parallel_workers_per_gather = 0",
            ],
        )
        connection.exec_driver_sql.side_effect = RuntimeError(
            "private connection detail"
        )
        with self.assertRaisesRegex(
            ReadOnlySetupError, "BIRD evaluation setup failed"
        ) as caught:
            prepare_bird_evaluation(connection)
        self.assertNotIn("private", str(caught.exception))
        connection.execute.assert_not_called()

    def test_driver_types_are_read_without_float_coercion(self):
        result = SimpleNamespace(
            cursor=SimpleNamespace(
                description=[
                    SimpleNamespace(type_code=700),
                    SimpleNamespace(type_code=1700),
                ]
            )
        )
        self.assertEqual(bird_column_types(result), (700, 1700))
        self.assertEqual(bird_column_types(SimpleNamespace()), ())

    def test_base_and_research_evaluators_share_strict_score_and_diagnostic(self):
        gold = [(0.20174165,)]
        prediction = [(Decimal("0.20174165457184325109"),)]

        def result(rows, oid):
            return SimpleNamespace(
                fetchall=lambda: rows,
                cursor=SimpleNamespace(description=[SimpleNamespace(type_code=oid)]),
            )

        connection = MagicMock()

        @contextmanager
        def boundary(*args):
            yield connection

        item = {
            "question_id": 34,
            "db_id": "db",
            "SQL": "SELECT 0.20174165",
            "difficulty": "moderate",
        }
        with tempfile.TemporaryDirectory() as tmp:
            gold_file = Path(tmp) / "gold.json"
            gold_file.write_text(json.dumps([item]))
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                connection.execute.side_effect = [
                    result(gold, 700),
                    result(prediction, 1700),
                ]
                with (
                    patch.dict(os.environ, {"DB_USER": "reader", "DB_PASS": "dummy"}),
                    patch.object(evaluate_bird, "create_engine"),
                    patch.object(evaluate_bird, "read_only_connection", boundary),
                ):
                    report = evaluate_bird.run_evaluation(
                        {"34": "SELECT 139::numeric/689"}, gold_file, "unused"
                    )
            finally:
                os.chdir(previous)
        base_comparison = report["results"][0]["bird_comparison"]
        self.assertEqual(report["overall_accuracy"], 0)
        self.assertTrue(base_comparison["numeric_precision_warning"])
        connection.execute.side_effect = [result(prediction, 1700)]
        session = object.__new__(runtime.Session)
        session.engines = {"db": object()}
        session.golds = {"34": normalize_bird_rows(gold)}
        session.gold_type_oids = {"34": (700,)}
        with patch("adv_text2sql.sql_safety.read_only_connection", boundary):
            self.assertEqual(
                session.score(item, "SELECT 139::numeric/689", "bird"), (0, None)
            )
        self.assertEqual(session.last_bird_comparison, base_comparison)
        commands = [c.args[0] for c in connection.exec_driver_sql.call_args_list]
        self.assertEqual(
            commands.count("SET LOCAL max_parallel_workers_per_gather = 0"), 2
        )
        self.assertEqual(
            [str(c.args[0]) for c in connection.execute.call_args_list][-1],
            "SELECT 139::numeric/689",
        )


if __name__ == "__main__":
    unittest.main()
