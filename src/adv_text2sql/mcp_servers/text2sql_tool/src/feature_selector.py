"""Cheap, deterministic configuration routing. No model, database, or environment access."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import unicodedata


RULE_VERSION = "configuration-rules-v1"
CONFIGURATIONS = {"economical": (35,), "hybrid": (14, 32, 35)}
INPUT_FIELDS = frozenset(("question", "evidence", "schema_stats"))
SCHEMA_FIELDS = frozenset(("table_count", "column_count", "text_column_count",
                           "numeric_column_count", "temporal_column_count", "boolean_column_count"))
# This finite catalog is fixed before looking at the held-out test outcomes.
POLICIES = ("always_economical", "always_hybrid", "ambiguity", "ambiguity_or_complexity",
            "ambiguity_or_large_schema", "strong_signals")
AMBIGUITY_PATTERNS = {
    "quantifier_scope": r"\b(?:every|each|all|any|both|either|neither)\b",
    "mixed_coordination": r"\band\b.*\bor\b|\bor\b.*\band\b",
    "modifier_attachment": r"\b(?:with|without|having)\b.*\b(?:and|or)\b",
    "underspecified_reference": r"\b(?:their|them|these|those|its)\b",
    "vague_comparison": r"\b(?:best|worst|good|bad|popular|successful|recent|nearby|large|small|high|low)\b",
}
COMPLEXITY_PATTERNS = {
    "aggregation": r"\b(?:average|avg|sum|total|count|how many|percentage|percent|ratio)\b",
    "ranking": r"\b(?:most|least|highest|lowest|maximum|minimum|top|bottom|rank)\b",
    "grouping": r"\b(?:per|for each|group(?:ed)? by)\b",
    "comparison": r"\b(?:more than|less than|greater than|fewer than|at least|at most|compared|difference)\b",
    "exclusion": r"\b(?:not|never|except|without|none)\b",
}


def _text(value):
    return unicodedata.normalize("NFKC", value).casefold()


def validate_inputs(inputs):
    if not isinstance(inputs, dict) or set(inputs) != INPUT_FIELDS:
        raise ValueError("Pass only question, evidence, schema_stats; identifiers and targets are forbidden")
    if not isinstance(inputs["question"], str) or not inputs["question"].strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(inputs["evidence"], str):
        raise ValueError("evidence must be a string")
    stats = inputs["schema_stats"]
    if not isinstance(stats, dict) or not set(stats) <= SCHEMA_FIELDS:
        raise ValueError("schema_stats must contain only structural counts")
    if any(type(value) is not int or value < 0 for value in stats.values()):
        raise ValueError("Schema counts must be non-negative integers")


@dataclass(frozen=True)
class Selection:
    configuration: str
    reasons: tuple[str, ...]
    policy: str

    @property
    def features(self):
        return CONFIGURATIONS[self.configuration]

    @property
    def feature_flags(self):
        return {f"FEAT_{number}": number in self.features for number in range(1, 37)}

    def as_dict(self):
        return {"configuration": self.configuration, "features": list(self.features),
                "reasons": list(self.reasons), "policy": self.policy, "rule_version": RULE_VERSION}


class RuleBasedFeatureSelector:
    def __init__(self, policy="ambiguity_or_complexity"):
        if policy not in POLICIES:
            raise ValueError(f"Unknown selector policy: {policy}")
        self.policy = policy

    @classmethod
    def from_file(cls, path):
        artifact = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if artifact.get("format_version") != "rule-selector-selection-v1" or artifact.get("rule_version") != RULE_VERSION:
            raise ValueError("Incompatible selector artifact")
        if artifact.get("selector_source_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError("Selector code differs from the frozen policy; create a new experiment")
        return cls(artifact["selected_policy"])

    def select_inputs(self, inputs):
        validate_inputs(inputs)
        question = _text(inputs["question"])
        # Quoted literal values are not evidence of scope or SQL complexity.
        question = re.sub(r'"[^"\n]*"|(?<!\w)\x27[^\x27\n]*\x27(?!\w)', " literal ", question)
        ambiguity = tuple(name for name, pattern in AMBIGUITY_PATTERNS.items() if re.search(pattern, question))
        complexity = tuple(name for name, pattern in COMPLEXITY_PATTERNS.items() if re.search(pattern, question))
        large = inputs["schema_stats"].get("table_count", 0) >= 10 or inputs["schema_stats"].get("column_count", 0) >= 100
        rules = {
            "always_economical": False,
            "always_hybrid": True,
            "ambiguity": bool(ambiguity),
            "ambiguity_or_complexity": bool(ambiguity) or len(complexity) >= 2,
            "ambiguity_or_large_schema": bool(ambiguity) or (large and bool(complexity)),
            "strong_signals": len(ambiguity) >= 2 or len(complexity) >= 2,
        }
        configuration = "hybrid" if rules[self.policy] else "economical"
        signals = (*ambiguity, *complexity, *(("large_schema",) if large else ()))
        return Selection(configuration, signals or ("no_rule_signal",), self.policy)

    def select(self, question, evidence="", schema_stats=None):
        return self.select_inputs({"question": question, "evidence": evidence, "schema_stats": schema_stats or {}})
