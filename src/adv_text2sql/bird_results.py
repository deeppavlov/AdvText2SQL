"""BIRD EX is exact set equality; PostgreSQL precision warnings never award EX.

Upstream: bird-bench/mini_dev/evaluation/evaluation_ex.py, calculate_ex,
and AlibabaResearch/DAMO-ConvAI/bird/llm/src/evaluation.py, execute_sql
(both inspected 2026-11-16). Neither defines a numeric tolerance.
"""

from collections import deque
from decimal import Decimal, localcontext
import math

from adv_text2sql.sql_safety import ReadOnlySetupError

EX_POLICY = "bird-exact-set-v1"
EXECUTION_POLICY = "bird-postgres-serial-repeatable-read-v1"
DIAGNOSTIC_POLICY = "postgres-numeric-warning-v1"
# A representation warning, not an error bound for arbitrary SQL arithmetic.
_RELATIVE_BOUNDS = {700: Decimal(2) ** -23, 701: Decimal(2) ** -48}
_MAX_PAIRS = 100_000


def normalize_bird_rows(rows):
    """Preserve upstream row-set semantics; retain the existing PG JSON/array support."""

    def hashable(value):
        if isinstance(value, list):
            return tuple(hashable(v) for v in value)
        if isinstance(value, dict):
            return tuple(sorted((k, hashable(v)) for k, v in value.items()))
        return value

    return {tuple(hashable(v) for v in row) for row in rows}


def prepare_bird_evaluation(connection):
    """Called inside the mandatory read-only boundary, before any SELECT.

    Settings are transaction-local. No server settings, generator SQL or other
    benchmarks are changed. Sequential aggregation removes the observed parallel
    AVG variance; it does not prove equivalence of arbitrary floating expressions.
    """
    try:
        connection.exec_driver_sql("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        connection.exec_driver_sql("SET LOCAL max_parallel_workers_per_gather = 0")
    except Exception as error:
        raise ReadOnlySetupError(
            f"BIRD evaluation setup failed ({type(error).__name__})"
        ) from None


def bird_column_types(result):
    """Read PostgreSQL OIDs before fetchall closes the cursor; absent metadata is OK."""
    description = getattr(getattr(result, "cursor", None), "description", None)
    if not isinstance(description, (list, tuple)):
        return ()
    return tuple(
        code if type(code := getattr(column, "type_code", None)) is int else None
        for column in description
    )


def _near_number(left, right, left_oid, right_oid):
    # Never coerce strings, bools, exact integers, containers or two DECIMALs.
    if not isinstance(left, (float, Decimal)) or not isinstance(
        right, (float, Decimal)
    ):
        return False
    if not isinstance(left, float) and not isinstance(right, float):
        return False
    if (
        isinstance(left, float)
        and not math.isfinite(left)
        or isinstance(right, float)
        and not math.isfinite(right)
        or isinstance(left, Decimal)
        and not left.is_finite()
        or isinstance(right, Decimal)
        and not right.is_finite()
    ):
        return False
    a = Decimal.from_float(left) if isinstance(left, float) else left
    b = Decimal.from_float(right) if isinstance(right, float) else right
    # Integral quantities stay exact even if a driver returned them as floats.
    if a == a.to_integral_value() or b == b.to_integral_value():
        return False
    oids = [
        oid
        for value, oid in ((left, left_oid), (right, right_oid))
        if isinstance(value, float)
    ]
    tolerance = max((_RELATIVE_BOUNDS.get(oid, _RELATIVE_BOUNDS[701]) for oid in oids))
    with localcontext() as context:
        context.prec = 80
        return abs(a - b) <= tolerance * max(abs(a), abs(b))


def _row_near(left, right, left_types, right_types):
    if len(left) != len(right):
        return False
    for index, (a, b) in enumerate(zip(left, right)):
        if a == b:
            continue
        lo = left_types[index] if index < len(left_types) else None
        ro = right_types[index] if index < len(right_types) else None
        if not _near_number(a, b, lo, ro):
            return False
    return True


def _numeric_bijection(predicted, reference, predicted_types, reference_types):
    """Bounded perfect matching, not greedy matching or rounded set keys."""
    if len(predicted) != len(reference):
        return False
    if len(predicted) * len(reference) > _MAX_PAIRS:
        return None
    left, right = list(predicted), list(reference)
    edges = [
        [
            j
            for j, b in enumerate(right)
            if _row_near(a, b, predicted_types, reference_types)
        ]
        for a in left
    ]
    # Augmenting paths prevent one near row from satisfying several gold rows.
    matched = {}
    for start in range(len(left)):
        queue, visited, parent = deque([start]), {start}, {}
        free = None
        while queue and free is None:
            node = queue.popleft()
            for target in edges[node]:
                if target in parent:
                    continue
                parent[target] = node
                if target not in matched:
                    free = target
                    break
                previous = matched[target]
                if previous not in visited:
                    visited.add(previous)
                    queue.append(previous)
        if free is None:
            return False
        while free is not None:
            node = parent[free]
            previous = next(
                (target for target, owner in matched.items() if owner == node), None
            )
            matched[free] = node
            if previous is not None:
                del matched[previous]
            free = previous
    return True


def compare_bird_results(
    predicted, reference, *, predicted_types=(), reference_types=()
):
    """One authoritative score, plus a separately named, non-scoring warning.

    EX exactly follows BIRD for SQLite scalar results. Row order and duplicate
    multiplicity are ignored, but column order, row width and values are not.
    The diagnostic preserves row-set cardinality with a one-to-one matching.
    Its precision bounds are PG-specific and must never be used as EX or labels.
    """
    pred, gold = normalize_bird_rows(predicted), normalize_bird_rows(reference)
    exact = pred == gold
    near = (
        True
        if exact
        else _numeric_bijection(pred, gold, predicted_types, reference_types)
    )
    return {
        "evaluation_policy": EX_POLICY,
        "execution_policy": EXECUTION_POLICY,
        "exact_equal": exact,
        "numeric_diagnostic_policy": DIAGNOSTIC_POLICY,
        "numeric_diagnostic_equal": near,
        "numeric_precision_warning": not exact and near is True,
    }
