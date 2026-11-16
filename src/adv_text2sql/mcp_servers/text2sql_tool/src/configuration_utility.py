"""Anchor-relative utility and cheap schema grounding; independent of frozen v2."""

from collections import defaultdict
import re

import numpy as np
import torch
from torch import nn

from .configuration_classifier_v2 import ANCHORS, OUTPUTS, targets
from .feature_catalog_v2 import CONFIGURATIONS, TASK_CONFIGURATIONS
from .feature_selector import validate_inputs

UTILITY_OUTPUTS = tuple((task, name) for task, name in OUTPUTS if name != ANCHORS[task])
GROUND_FIELDS = (
    "question_tokens", "evidence_tokens", "has_number", "has_date", "has_quoted_literal",
    "count_cue", "sum_cue", "average_cue", "extremum_cue", "comparison_cue", "time_cue",
    "scope_cue", "conjunction_cue", "column_word_coverage", "table_word_coverage",
    "evidence_column_word_coverage", "matched_columns", "matched_tables",
    "cross_table_word_ratio", "repeated_column_name_ratio", "max_column_word_coverage",
    "matched_numeric_columns", "matched_temporal_columns", "matched_text_columns",
)
STOP = frozenset("a an the of in on at for to from with by and or is are was were be been what which who how "
                 "many much do does did please list show provide each all every their that this those these "
                 "as it its i me we our you your have has had id yn".split())


def words(text):
    """Small deterministic tokenizer, not a trained vocabulary or task-name parser."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    found = re.findall(r"[a-z]+|\d+", text.lower())
    result = set()
    for token in found:
        if token in STOP:
            continue
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        result.add(token)
    return result


def grounding_features(inputs, schema_rows):
    """Only question/evidence and schema names/types; no IDs, SQL, labels or DB values."""
    validate_inputs(inputs)
    if not schema_rows or any(not isinstance(r, (list, tuple)) or len(r) != 3
                              or any(not isinstance(v, str) for v in r) for r in schema_rows):
        raise ValueError("Expected non-empty schema rows [table, column, type]")
    query = inputs["question"].lower()
    q, e = words(inputs["question"]), words(inputs["evidence"])
    columns = [(table, words(name), type_.lower(), name.lower()) for table, name, type_ in schema_rows]
    table_words = set().union(*(words(table) for table, _, _, _ in columns))
    column_words = set().union(*(tokens for _, tokens, _, _ in columns))
    matched = [(table, tokens, type_, name) for table, tokens, type_, name in columns if tokens & q]
    token_tables, name_tables = defaultdict(set), defaultdict(set)
    for table, tokens, _, name in columns:
        name_tables[name].add(table)
        for token in tokens:
            token_tables[token].add(table)
    def fraction(tokens, available):
        return len(tokens & available) / max(len(tokens), 1)

    def cue(pattern):
        return float(bool(re.search(pattern, query)))
    values = (
        len(re.findall(r"\w+", query)), len(re.findall(r"\w+", inputs["evidence"])),
        cue(r"\d"), cue(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"),
        cue(r"['\"][^'\"]+['\"]"), cue(r"\b(count|number of|how many)\b"),
        cue(r"\b(sum|total)\b"), cue(r"\b(average|mean)\b"),
        cue(r"\b(most|least|fewest|highest|lowest|greatest|maximum|minimum|top)\b"),
        cue(r"\b(over|under|above|below|more than|less than|greater than|between)\b"),
        cue(r"\b(before|after|during|year|month|date|day|opened|closed)\b"),
        cue(r"\b(all|every|each|any|only|either|both)\b"), cue(r"\b(and|or)\b"),
        fraction(q, column_words), fraction(q, table_words), fraction(e, column_words),
        len(matched), len({table for table, _, _, _ in matched}),
        sum(len(token_tables[token]) > 1 for token in q) / max(len(q), 1),
        sum(len(tables) > 1 for tables in name_tables.values()) / max(len(name_tables), 1),
        max((len(tokens & q) / max(len(tokens), 1) for _, tokens, _, _ in columns), default=0),
        sum(bool(re.search(r"int|numeric|decimal|real|double|float", type_)) for _, _, type_, _ in matched),
        sum(bool(re.search(r"date|time", type_)) for _, _, type_, _ in matched),
        sum(bool(re.search(r"text|char|uuid", type_)) for _, _, type_, _ in matched),
    )
    result = np.asarray(values, dtype=np.float32)
    if len(result) != len(GROUND_FIELDS) or not np.isfinite(result).all():
        raise ValueError("Invalid grounding features")
    return result


class UtilityHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, len(UTILITY_OUTPUTS))

    def forward(self, x):
        return torch.tanh(self.linear(x))


def utility_targets(rows):
    quality, observed, tasks = targets(rows)
    values = torch.full((len(rows), len(UTILITY_OUTPUTS)), float("nan"))
    mask = torch.zeros_like(values, dtype=torch.bool)
    for i, row in enumerate(rows):
        task = row["benchmark"]
        anchor = OUTPUTS.index((task, ANCHORS[task]))
        for col, (target_task, name) in enumerate(UTILITY_OUTPUTS):
            if target_task != task:
                continue
            target = OUTPUTS.index((task, name))
            if observed[i, anchor] and observed[i, target]:
                values[i, col] = quality[i, target] - quality[i, anchor]
                mask[i, col] = True
    return values, mask, tasks


def utility_loss(prediction, values, mask, tasks):
    if (prediction.shape != values.shape or mask.shape != values.shape or mask.dtype != torch.bool
            or prediction.ndim != 2 or tasks.shape != (len(prediction),)
            or not torch.isfinite(prediction).all() or not torch.isfinite(values[mask]).all()
            or ((values[mask] != -1) & (values[mask] != 0) & (values[mask] != 1)).any()):
        raise ValueError("Invalid relative utility tensors")
    safe = torch.where(mask, values, torch.zeros_like(values))
    per_row = (((prediction - safe) ** 2) * mask).sum(1) / mask.sum(1).clamp_min(1)
    losses = [per_row[(tasks == i) & mask.any(1)].mean() for i in range(len(TASK_CONFIGURATIONS))
              if ((tasks == i) & mask.any(1)).any()]
    return torch.stack(losses).mean() if losses else prediction.sum() * 0


def choose_utility(values, benchmark, margin=0.15):
    names = TASK_CONFIGURATIONS.get(benchmark)
    if (names is None or set(values) != set(names) - {ANCHORS[benchmark]}
            or not np.isfinite(margin) or margin < 0
            or any(not np.isfinite(x) or not -1 <= x <= 1 for x in values.values())):
        raise ValueError("Invalid relative utility predictions")
    anchor = ANCHORS[benchmark]
    name = min(values, key=lambda n: (-values[n], len(CONFIGURATIONS[n]), names.index(n)))
    return name if values[name] > margin else anchor
