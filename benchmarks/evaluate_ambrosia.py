import json
from typing import Dict

from adv_text2sql.evaluation import ambrosia_report

def run_evaluation(predictions: Dict[str, str], answer_file: str):
    with open(answer_file, "r") as f:
        gold_data = json.load(f)

    return ambrosia_report(gold_data, predictions)

def print_evaluation_report(report: dict):
    print("\n=== Ambiguity Detection (Ambrosia benchmark) ===")
    print(f"Accuracy: {report['accuracy']:.2f}%")
    print(f"Total questions: {report['total']}")
    print(f"Gold ambiguous: {report['gold_ambiguous']}")
    print(f"Predicted ambiguous: {report['pred_ambiguous']}")
    print(f"Correctly flagged ambiguous: {report['true_positive']}")
    print("===========================\n")
