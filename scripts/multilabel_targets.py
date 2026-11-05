"""Versioned eligibility and weak benefit targets for the recorded single-flag runs."""

FEATURES = tuple(n for n in range(1, 37) if n not in (22, 24, 31))
DEPENDENCIES = {3: [2], 4: [2], 7: [2], 27: [17], 33: [32], 34: [32, 33], 36: [15]}
INFRASTRUCTURE = (8, 9, 10, 11, 17, 18, 19)
EXCLUSIONS = {n: "dependency_not_enabled" for n in DEPENDENCIES}
EXCLUSIONS.update({n: "fixed_infrastructure" for n in INFRASTRUCTURE})
EXCLUSIONS.update({
    5: "no_effective_switch",
    6: "incomplete_schema_context_in_recorded_bird_run",
    26: "stateful_cache_without_question_history_features",
})
TARGET_FEATURES = tuple(n for n in FEATURES if n not in EXCLUSIONS)
CONDITIONAL_PAIRS = (
    {"name": "statistics_given_tsv", "feature": 2, "context": [35], "treatment": "schema_2_35", "control": "feat_35"},
    {"name": "tsv_given_statistics", "feature": 35, "context": [2], "treatment": "schema_2_35", "control": "feat_2"},
    {"name": "sql_rules_given_tsv", "feature": 14, "context": [35], "treatment": "sql_rules", "control": "feat_35"},
    {"name": "taxonomy_given_rules_tsv", "feature": 32, "context": [14, 35],
     "treatment": "sql_ambiguity", "control": "sql_rules"},
    {"name": "relationships_given_hybrid", "feature": 1, "context": [14, 32, 35],
     "treatment": "hybrid_with_relationships", "control": "sql_ambiguity"},
)


def feature_catalog():
    return {f"FEAT_{n}": {
        "number": n, "case": f"feat_{n}", "control": "baseline", "context_features": [],
        "required_features": DEPENDENCIES.get(n, []), "trainable": n in TARGET_FEATURES,
        "exclusion_reason": EXCLUSIONS.get(n),
        "eligibility": "independently available mechanism, not proof of execution or causality" if n in TARGET_FEATURES
        else "excluded from classifier targets",
    } for n in FEATURES}


def benefit_label(treatment, control, exclusion=None):
    """Correctness alone is not a positive target. Keep ties and absent values distinct."""
    for value in (treatment, control):
        if type(value.get("observed")) is not bool:
            raise ValueError("observed must be boolean")
        if value["observed"] and type(value.get("correct")) is not bool:
            raise ValueError("Observed correctness must be boolean")
        if not value["observed"] and value.get("correct") is not None:
            raise ValueError("Unobserved correctness must remain null")
    observed = treatment["observed"] and control["observed"]
    utility = int(treatment["correct"]) - int(control["correct"]) if observed else None
    reason = exclusion or ("missing_or_invalid_treatment" if not treatment["observed"] else
                           "missing_or_invalid_control" if not control["observed"] else None)
    usable = observed and reason is None
    return {
        "pair_observed": observed, "utility": utility,
        "benefit_target": int(utility > 0) if usable else None,
        "loss_mask": usable,
        "preference_target": int(utility > 0) if usable and utility != 0 else None,
        "preference_mask": usable and utility != 0,
        "exclusion_reason": reason,
        "outcome_class": "missing" if not observed else "helped" if utility > 0 else
        "harmed" if utility < 0 else "both_correct" if treatment["correct"] else "neither_correct",
    }


def single_feature_labels(observations):
    expected = {"baseline", *(f"feat_{n}" for n in FEATURES)}
    if not expected <= set(observations):
        raise ValueError("Keep every single-flag observation, including excluded ones")
    return {f"FEAT_{n}": benefit_label(observations[f"feat_{n}"], observations["baseline"], EXCLUSIONS.get(n))
            for n in FEATURES}


def target_vectors(labels):
    selected = [labels[f"FEAT_{n}"] for n in TARGET_FEATURES]
    return {"benefit_targets": [label["benefit_target"] for label in selected],
            "loss_mask": [label["loss_mask"] for label in selected],
            "preference_targets": [label["preference_target"] for label in selected],
            "preference_mask": [label["preference_mask"] for label in selected]}
