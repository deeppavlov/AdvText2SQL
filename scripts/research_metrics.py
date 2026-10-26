"""Offline ambiguity scoring: failed generation is never a correct answer."""


def ambrosia_report(items, predictions):
    gold = {str(item["question_id"]): item["SQL"] == "ambiguous" for item in items}
    if len(gold) != len(items) or set(predictions) != set(gold):
        raise ValueError("Ambrosia predictions must cover exactly the unique dataset IDs")
    if any(not isinstance(v, str) or not v.strip() for v in predictions.values()):
        raise ValueError("Predictions must be non-empty strings")
    tp = fp = fn = tn = errors_ambiguous = errors_unambiguous = 0
    for question_id, prediction in predictions.items():
        ambiguous = gold[question_id]
        if prediction == "error":
            if ambiguous:
                errors_ambiguous += 1
            else:
                errors_unambiguous += 1
        elif prediction == "ambiguous":
            if ambiguous:
                tp += 1
            else:
                fp += 1
        elif ambiguous:
            fn += 1
        else:
            tn += 1
    total = len(gold)
    ambiguous_count = sum(gold.values())
    unambiguous_count = total - ambiguous_count
    # Failed ambiguous questions count as missed detections. Errors on unambiguous
    # questions are abstentions, not true negatives or false ambiguity predictions.
    missed = fn + errors_ambiguous
    recall = tp / ambiguous_count if ambiguous_count else None
    specificity = tn / unambiguous_count if unambiguous_count else None
    return {
        "evaluation_policy": "ambrosia-strict-v1",
        "accuracy": 100 * (tp + tn) / total if total else 0,
        "legacy_accuracy": 100 * (tp + tn + errors_unambiguous) / total if total else 0,
        "total": total,
        "gold_ambiguous": ambiguous_count,
        "pred_ambiguous": tp + fp,
        "true_positive": tp,
        "generation_errors": errors_ambiguous + errors_unambiguous,
        "ambiguity_metrics": {
            "tp": tp, "fp": fp, "fn": missed, "tn": tn,
            "errors_ambiguous": errors_ambiguous,
            "errors_unambiguous": errors_unambiguous,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + missed) if 2 * tp + fp + missed else 0,
            "balanced_accuracy": (recall + specificity) / 2
            if recall is not None and specificity is not None else None,
        },
    }
