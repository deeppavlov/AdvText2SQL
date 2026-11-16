import os
import json
import logging
import sqlglot
import decimal
from adv_text2sql.sql_text import clean_sql

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict

from adv_text2sql.sql_safety import UnsafeSQL, read_only_connection, validate_read_query
from adv_text2sql.evaluation import database_failure
from adv_text2sql.bird_results import (
    EX_POLICY, EXECUTION_POLICY, DIAGNOSTIC_POLICY, bird_column_types,
    compare_bird_results, prepare_bird_evaluation,
)

logger = logging.getLogger(__name__)


class EvaluationUnavailable(RuntimeError):
    """Evaluation is incomplete; no negative model label may be exported."""


def sqlite_to_postgres(query: str) -> str:
    """Translate one SQLite statement through an AST; never patch SQL with regex."""
    validate_read_query(query, dialect="sqlite")
    try:
        statements = sqlglot.transpile(clean_sql(query, dialect="sqlite"), read="sqlite", write="postgres",
                                      unsupported_level=sqlglot.ErrorLevel.RAISE)
    except (sqlglot.errors.SqlglotError, ValueError):
        raise ValueError("SQLite reference cannot be translated safely") from None
    if len(statements) != 1 or not statements[0].strip():
        raise ValueError("Expected exactly one SQLite reference statement")
    return statements[0]






def run_evaluation(predictions: Dict[str, str], answer_file: str, db_url: str):
    with open(answer_file, "r") as f:
        answer_file = json.load(f)

    gold_queries = {str(item["question_id"]): item for item in answer_file}

    results = []

    db_username = os.environ["DB_USER"]
    db_password = os.environ["DB_PASS"]

    all_predicted = {}
    all_gold = {}
    engines: Dict[str, any] = {}
    total = len(predictions)

    for idx, (question_id, predicted_sql) in enumerate(predictions.items(), 1):
        gold_query = gold_queries[question_id]

        gold_sql = gold_query["SQL"]
        db_id = gold_query["db_id"]
        difficulty = gold_query["difficulty"]

        print(f"[{idx}/{total}] q_id={question_id} db={db_id}", flush=True)

        # ---- ambiguous request processing ----
        if gold_sql == "ambiguous" or predicted_sql == "ambiguous":
            if gold_sql == "ambiguous" and predicted_sql == "ambiguous":
                score = 1
            else:
                score = 0

            results.append(
                {
                    "question_id": question_id,
                    "gold_sql": gold_sql,
                    "predicted_sql": predicted_sql,
                    "score": score,
                    "difficulty": difficulty,
                }
            )
            continue

        # ---- SQL execution and comparison with gold results ----
        db_uri = f"postgresql+psycopg://{db_username}:{db_password}@{db_url}/{db_id}"
        if db_id not in engines:
            engines[db_id] = create_engine(db_uri)

        stage = "reference"
        try:
            gold_sql = sqlite_to_postgres(gold_sql)
            engine = engines[db_id]

            # sqlite_to_postgres validated the original gold in its input dialect.
            validate_read_query(gold_sql)
            stage = "prediction"
            validate_read_query(predicted_sql)
            stage = "connection"
            with read_only_connection(engine, gold_sql, predicted_sql) as conn:
                prepare_bird_evaluation(conn)
                stage = "reference"
                gold_result = conn.execute(text(gold_sql))
                gold_types = bird_column_types(gold_result)
                gold_res = gold_result.fetchall()
                all_gold[question_id] = [list(row) for row in gold_res]

                stage = "prediction"
                pred_result = conn.execute(text(predicted_sql))
                pred_types = bird_column_types(pred_result)
                pred_res = pred_result.fetchall()
                stage = "connection"  # Transaction cleanup failures are not SQL mistakes.
                all_predicted[question_id] = [list(row) for row in pred_res]

            comparison = compare_bird_results(pred_res, gold_res, predicted_types=pred_types,
                                              reference_types=gold_types)
            score = comparison["exact_equal"]

            results.append(
                {
                    "question_id": question_id,
                    "gold_sql": gold_sql,
                    "predicted_sql": predicted_sql,
                    "score": score,
                    "difficulty": difficulty,
                    "bird_comparison": comparison,
                }
            )

        except (SQLAlchemyError, UnsafeSQL) as e:
            details = ({"failure_kind": "query", "sqlstate": None, "error_type": "UnsafeSQL"}
                       if isinstance(e, UnsafeSQL) else database_failure(e))
            if stage != "prediction" or details["failure_kind"] != "query":
                raise EvaluationUnavailable(
                    f"BIRD evaluation stopped at {stage} ({details['failure_kind']}, "
                    f"{details['error_type']}); no model label saved"
                ) from None
            logger.info(
                f"Failed to process sql query for question {question_id}: '{predicted_sql}'"
            )
            results.append(
                {
                    "question_id": question_id,
                    "gold_sql": gold_sql,
                    "predicted_sql": predicted_sql,
                    "score": 0,
                    "difficulty": difficulty,
                    "error": f"{details['error_type']} (SQLSTATE {details['sqlstate'] or 'none'})",
                    "execution_stage": stage,
                    **details,
                }
            )

    with open("all_predicted_results.json", "w", encoding="utf-8") as f:
        json.dump(all_predicted, f, ensure_ascii=False, indent=2, default=lambda x: float(x) if isinstance(x, decimal.Decimal) else str(x))

    with open("all_gold_results.json", "w", encoding="utf-8") as f:
        json.dump(all_gold, f, ensure_ascii=False, indent=2, default=lambda x: float(x) if isinstance(x, decimal.Decimal) else str(x))


    # ---- accuracy calculation ----

    def accuracy(rows):
        if not rows:
            return 0.0
        return 100.0 * sum(r["score"] for r in rows) / len(rows)

    total_acc = accuracy(results)

    by_difficulty = {}
    for diff in set(r["difficulty"] for r in results):
        subset = [r for r in results if r["difficulty"] == diff]
        by_difficulty[diff] = accuracy(subset)

    false_ambiguous = sum(
        1 for r in results if r["predicted_sql"] == "ambiguous"
    )

    false_ambiguous_rate = (
        100.0 * false_ambiguous / len(results) if results else 0.0
    )

    print(results)
    report = {
        "evaluation_policy": EX_POLICY,
        "postgresql_execution_policy": EXECUTION_POLICY,
        "numeric_diagnostic_policy": DIAGNOSTIC_POLICY,
        "overall_accuracy": total_acc,
        "accuracy_by_difficulty": by_difficulty,
        "false_ambiguous": false_ambiguous,
        "false_ambiguous_rate": false_ambiguous_rate,
        "total": len(results),
        "results": results,
    }

    return report

def print_evaluation_report(report: dict):
    print("\n================ BIRD Benchmark Results ====================\n")

    # ---- overall ----
    print(f"Overall accuracy: {report['overall_accuracy']:.2f}%")
    print(f" Total queries : {report['total']}")

    # ---- difficulty ----
    print()
    print("Accuracy by difficulty:")
    for diff, acc in sorted(report["accuracy_by_difficulty"].items()):
        print(f"  {diff:<12}: {acc:.2f}%")
    print()

    print(f"False ambiguous predicted : {report['false_ambiguous']}")
    print(f"False ambiguous rate      : {report['false_ambiguous_rate']:.2f}%")

    print("============================================================\n")
