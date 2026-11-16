"""Versioned classifier choices. Does not mutate the frozen v1 implementation."""

import math

VERSION = "feature-catalog-v2"
HEAD_FEATURES = (1, 2, 7, 12, 13, 14, 15, 16, 20, 21, 23, 25, 28, 29, 30, 32, 35)
DEPENDENCIES = {7: (2,)}
CONFLICTS = (frozenset((14, 29)), frozenset((20, 28)), frozenset((12, 32)))
MAX_FEATURES = 5
CONFIGURATIONS = {
    "baseline": (),
    "rules": (14,),
    "rules_taxonomy": (14, 32),
    "hybrid": (14, 32, 35),
    "statistics_hybrid": (2, 14, 32, 35),
    "compact": (2, 7, 14, 32, 35),
    "compact_without_taxonomy": (2, 7, 14, 35),
    "compact_without_rules": (2, 7, 32, 35),
}
# A bounded first collection, fixed without inspecting new configuration scores.
# Other masks/heads remain available for a later versioned extension, not fabricated labels.
TASK_CONFIGURATIONS = {
    "bird": ("baseline", "rules_taxonomy", "hybrid", "statistics_hybrid", "compact"),
    "ambrosia": ("baseline", "rules", "rules_taxonomy", "hybrid"),
}


def validate_features(features, max_features=MAX_FEATURES):
    values = tuple(features)
    if (type(max_features) is not int or not 0 <= max_features <= len(HEAD_FEATURES)
            or any(type(n) is not int for n in values) or len(set(values)) != len(values)
            or not set(values) <= set(HEAD_FEATURES) or len(values) > max_features):
        raise ValueError("Unsupported, repeated or over-budget feature mask")
    selected = set(values)
    if any(n in selected and not set(required) <= selected for n, required in DEPENDENCIES.items()):
        raise ValueError("FEAT_7 requires FEAT_2; dependencies are not implicit labels")
    if any(pair <= selected for pair in CONFLICTS):
        raise ValueError("Conflicting generator mechanisms")
    return tuple(sorted(values))


def feature_flags(features):
    selected = set(validate_features(features))
    return {f"FEAT_{n}": n in selected for n in range(1, 37)}


def catalog():
    return {"version": VERSION, "head_features": list(HEAD_FEATURES), "max_features": MAX_FEATURES,
            "dependencies": {str(k): list(v) for k, v in DEPENDENCIES.items()},
            "conflicts": [sorted(p) for p in CONFLICTS],
            "configurations": {k: list(validate_features(v)) for k, v in CONFIGURATIONS.items()},
            "task_configurations": {k: list(v) for k, v in TASK_CONFIGURATIONS.items()},
            "active_features_in_first_collection": sorted({n for names in TASK_CONFIGURATIONS.values()
                                                           for name in names for n in CONFIGURATIONS[name]})}


def constrained_features(probabilities, threshold=0.35, max_features=MAX_FEATURES):
    """Dependency-aware v2 decoding; missing labels must be handled during fitting."""
    if set(probabilities) != {str(n) for n in HEAD_FEATURES}:
        raise ValueError("Probabilities must match the v2 head order")
    if (not math.isfinite(threshold) or not 0 < threshold <= 1 or type(max_features) is not int
            or not 0 <= max_features <= MAX_FEATURES):
        raise ValueError("Invalid threshold/feature budget")
    if any(type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError("Invalid feature probability")
    raw = sorted((n for n in HEAD_FEATURES if probabilities[str(n)] >= threshold),
                 key=lambda n: (-probabilities[str(n)], n))
    selected, removed, added = set(), [], set()
    for n in raw:
        proposal = selected | {n} | set(DEPENDENCIES.get(n, ()))
        try:
            validate_features(proposal, max_features)
        except ValueError:
            removed.append(n)
            continue
        added |= proposal - selected - set(raw)
        selected = proposal
    return {"raw_features": sorted(raw), "features": sorted(selected),
            "dependency_added": sorted(added), "constraints_removed": removed,
            "feature_flags": feature_flags(selected), "catalog_version": VERSION}


def select_configuration(probabilities, benchmark):
    """Conservative deployment: only masks measured in this task's finite catalog."""
    constrained_features(probabilities)  # Validate the full vector before use.
    names = TASK_CONFIGURATIONS[benchmark]
    logits = {n: math.log(max(1e-7, probabilities[str(n)]) / max(1e-7, 1 - probabilities[str(n)]))
              for n in HEAD_FEATURES}
    # Additive baseline only; a joint-quality model must be compared separately.
    name = min(names, key=lambda k: (-sum(logits[n] for n in CONFIGURATIONS[k]), len(CONFIGURATIONS[k]), names.index(k)))
    return {"configuration": name, "features": list(CONFIGURATIONS[name]),
            "feature_flags": feature_flags(CONFIGURATIONS[name]), "policy": "finite_catalog_additive_v2"}
