import json
from typing import Dict

def run_evaluation(predictions: Dict[str, str], answer_file: str):
    with open(answer_file, "r") as f:
        gold_data = json.load(f)

    gold_map = {str(item["question_id"]): item for item in gold_data}

    total = 0
    correct = 0
    gold_ambiguous = 0
    pred_ambiguous = 0
    true_positive = 0

    for qid, pred in predictions.items():
        gold_is_amb = gold_map[qid]["SQL"] == "ambiguous"
        pred_is_amb = pred == "ambiguous"

        total += 1
        correct += gold_is_amb == pred_is_amb

        gold_ambiguous += gold_is_amb
        pred_ambiguous += pred_is_amb
        true_positive += gold_is_amb and pred_is_amb

    return {
        "accuracy": 100 * correct / total if total else 0,
        "total": total,
        "gold_ambiguous": gold_ambiguous,
        "pred_ambiguous": pred_ambiguous,
        "true_positive": true_positive,
    }

def print_evaluation_report(report: dict):
    print("\n=== Ambiguity Detection (Ambrosia benchmark) ===")
    print(f"Accuracy: {report['accuracy']:.2f}%")
    print(f"Total questions: {report['total']}")
    print(f"Gold ambiguous: {report['gold_ambiguous']}")
    print(f"Predicted ambiguous: {report['pred_ambiguous']}")
    print(f"Correctly flagged ambiguous: {report['true_positive']}")
    print("===========================\n")
