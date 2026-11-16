"""Shared evaluation contracts; infrastructure failures never become model labels."""

import re


def ambrosia_decision_correct(item, prediction):
    """Score the ambiguity decision, never treating an error as a true negative."""
    return (isinstance(prediction, str) and bool(prediction.strip())
            and prediction != "error"
            and (prediction == "ambiguous") == (item["SQL"] == "ambiguous"))


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


# Exact historical driver class names, not a guess based on the failed SQL.
_QUERY_ERRORS = {
    "SyntaxError", "UndefinedColumn", "UndefinedTable", "UndefinedFunction",
    "AmbiguousColumn", "AmbiguousFunction", "GroupingError", "DatatypeMismatch",
    "InvalidDatetimeFormat", "DatetimeFieldOverflow", "InvalidTextRepresentation",
    "DivisionByZero", "NumericValueOutOfRange", "InvalidParameterValue",
    "InvalidColumnReference", "FeatureNotSupported", "InvalidRegularExpression", "WindowingError",
}
_INFRA_ERRORS = {
    "OperationalError", "InterfaceError", "ConnectionException", "ConnectionFailure",
    "ConnectionDoesNotExist", "SqlclientUnableToEstablishSqlconnection",
    "AdminShutdown", "CrashShutdown", "CannotConnectNow", "InvalidPassword",
    "InvalidAuthorizationSpecification", "InvalidCatalogName", "InsufficientPrivilege",
    "TooManyConnections", "DiskFull", "OutOfMemory", "QueryCanceled",
    "SerializationFailure", "DeadlockDetected", "LockNotAvailable",
}


def sqlstate_failure_kind(code):
    """Conservative allowlist: only known query errors are model failures."""
    if not isinstance(code, str) or not re.fullmatch(r"[0-9A-Z]{5}", code):
        return "unknown"
    if code == "42501" or code[:2] in {"08", "28", "3D", "40", "53", "54", "55", "57", "58", "XX"}:
        return "infrastructure"
    if code[:2] in {"22", "42"} or code == "0A000":
        return "query"
    return "unknown"


def database_failure(error):
    """Return sanitized structured failure details from a live DB exception."""
    original = getattr(error, "orig", error)
    code = getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)
    if not isinstance(code, str) or not re.fullmatch(r"[0-9A-Z]{5}", code):
        code = None
    kind = sqlstate_failure_kind(code)
    if getattr(error, "connection_invalidated", False):
        kind = "infrastructure"
    elif code is None and (type(original).__name__ in _INFRA_ERRORS
                           or type(error).__name__ in _INFRA_ERRORS):
        kind = "infrastructure"
    return {"failure_kind": kind, "sqlstate": code, "error_type": type(original).__name__}


def recorded_database_failure(entry):
    """Classify archives without executing SQL or trusting SQL-text attribution."""
    message = entry.get("error")
    if not message:
        return None
    if "sqlstate" in entry or "failure_kind" in entry:
        kind = sqlstate_failure_kind(entry.get("sqlstate"))
        if entry.get("failure_kind") in {"infrastructure", "unknown"}:
            return entry["failure_kind"]
        return kind
    text = str(message)
    # Inspect only the diagnostic prefix, never SQL literals or parameters.
    prefix = re.split(r"\[SQL:|\bLINE \d+:", text, maxsplit=1)[0]
    classes = set(re.findall(r"\((?:[\w.]+\.)?([A-Za-z]+)\)", prefix))
    if classes & _INFRA_ERRORS:
        return "infrastructure"
    codes = re.findall(r"\bSQLSTATE(?:\s*[:=]\s*|\s+)([0-9A-Z]{5})\b", prefix)
    if codes:
        return sqlstate_failure_kind(codes[0])
    if classes & _QUERY_ERRORS:
        return "query"
    return "unknown"


def validate_collected_observation(row, item, benchmark):
    """Transport failures remain unknown; model SQL failures are observed zeros."""
    if str(row.get("question_id")) != str(item["question_id"]) or row.get("db_id") != item["db_id"]:
        raise ValueError("Observation identity mismatch")
    score, prediction, error = row.get("score"), row.get("prediction"), row.get("error")
    if type(score) not in (int, float) or score not in (0, 1) or not isinstance(prediction, str) or not prediction.strip():
        raise ValueError("Invalid score/prediction")
    output_errors = {None, "generation_error", "false_ambiguity", "non_text_sql", "empty_sql",
                     "invalid_sql_tokens", "comments_only_sql", "multiple_statements", "non_query_text",
                     "non_read_query", "non_query_with_text", "invalid_tool_response"}
    if error not in output_errors:
        if not isinstance(error, str) or not error.startswith("postgresql:"):
            return None
        code = error.split(":")[1]
        if sqlstate_failure_kind(code) != "query":
            return None
    if benchmark == "ambrosia":
        expected = float(prediction != "error" and ((item["SQL"] == "ambiguous") == (prediction == "ambiguous")))
        if score != expected:
            raise ValueError("Ambrosia score differs from strict decision evaluator")
    if (prediction in {"error", "ambiguous"} and benchmark == "bird" or prediction == "error") and score:
        raise ValueError("A failed generation cannot be correct")
    return float(score)
