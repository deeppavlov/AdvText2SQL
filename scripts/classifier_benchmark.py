"""Compare a live classifier run with stored controls on exactly the submitted IDs."""

from build_multilabel_dataset import recorded_observations


def compare_with_controls(rows, benchmark, items, predictions, report):
    wanted = {str(item["question_id"]) for item in items}
    if set(predictions) != wanted:
        raise ValueError("Live prediction coverage differs from the submitted split")
    original = {row["question_id"]: row for row in rows if row["benchmark"] == benchmark and row["question_id"] in wanted}
    if set(original) != wanted:
        raise ValueError("Stored controls do not cover the submitted split")
    actual = recorded_observations(items, predictions, report, benchmark)
    if not all(value["observed"] for value in actual.values()):
        raise ValueError("Invalid output or reference/evaluation failure; do not claim a complete comparison")
    correct = sum(value["correct"] for value in actual.values())
    controls = {}
    for name, case in (("baseline", "baseline"), ("economical", "feat_35"), ("hybrid", "sql_ambiguity")):
        values = {key: original[key]["observations"][case] for key in wanted}
        if not all(v["observed"] for v in values.values()):
            raise ValueError("A stored control is incomplete")
        controls[name] = {
            "correct": sum(v["correct"] for v in values.values()),
            "accuracy": 100 * sum(v["correct"] for v in values.values()) / len(items),
            "rescued": sum(actual[key]["correct"] and not values[key]["correct"] for key in wanted),
            "lost": sum(not actual[key]["correct"] and values[key]["correct"] for key in wanted),
        }
    return {"benchmark": benchmark, "questions": len(items), "question_ids": sorted(wanted),
            "classifier": {"correct": correct, "accuracy": 100 * correct / len(items)}, "controls": controls,
            "same_question_ids": True, "retrospective_test": True,
            "control_model_calls_are_historical": True}
