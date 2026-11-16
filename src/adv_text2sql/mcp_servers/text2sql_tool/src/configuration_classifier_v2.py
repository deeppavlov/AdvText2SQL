"""Frozen embeddings and task-routed joint configuration quality heads, v2.

Outputs describe complete measured configurations, never independent feature benefit.
The v1 selector and frozen collection runtime are intentionally untouched.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .feature_catalog_v2 import CONFIGURATIONS, TASK_CONFIGURATIONS, catalog, feature_flags
from .multilabel_classifier import FrozenTextEncoder, SchemaScaler, digest

FORMAT = "configuration-quality-classifier-v2"
ANCHORS = {"bird": "compact", "ambrosia": "rules_taxonomy"}
OUTPUTS = tuple((task, name) for task, names in TASK_CONFIGURATIONS.items() for name in names)


class ConfigurationHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, len(OUTPUTS))

    def forward(self, inputs):
        return self.linear(inputs)


def targets(rows):
    """Task routing and missing observations have distinct masks; no invented zeros."""
    values = torch.full((len(rows), len(OUTPUTS)), float("nan"))
    mask = torch.zeros_like(values, dtype=torch.bool)
    tasks = []
    for i, row in enumerate(rows):
        task = row["benchmark"]
        if task not in TASK_CONFIGURATIONS or row["configurations"] != list(TASK_CONFIGURATIONS[task]):
            raise ValueError("Unexpected task/configuration order")
        if not (len(row["quality_targets"]) == len(row["loss_mask"]) == len(row["configurations"])):
            raise ValueError("Target/mask lengths differ")
        tasks.append(tuple(TASK_CONFIGURATIONS).index(task))
        for name, value, observed in zip(row["configurations"], row["quality_targets"], row["loss_mask"]):
            if type(observed) is not bool or observed != (value is not None):
                raise ValueError("Unknown targets must have a false loss mask")
            if observed:
                if type(value) not in (float, int) or value not in (0, 1):
                    raise ValueError("Observed quality must be binary")
                col = OUTPUTS.index((task, name))
                values[i, col], mask[i, col] = value, True
    return values, mask, torch.tensor(tasks, dtype=torch.long)


def quality_loss(logits, values, mask, task_ids, pair_weight=0.0):
    """Masked BCE plus discordant-pair ranking; balance tasks, then questions.

    Equal outcomes impose no ranking. Unknowns and other tasks have zero gradient.
    Ranking is derived within each question, not treated as new independent data.
    """
    if (logits.shape != values.shape or mask.shape != values.shape or mask.dtype != torch.bool
            or logits.ndim != 2 or task_ids.shape != (len(logits),)):
        raise ValueError("Invalid loss shapes/masks")
    if (not torch.isfinite(logits).all() or not torch.isfinite(values[mask]).all()
            or ((values[mask] != 0) & (values[mask] != 1)).any()
            or not np.isfinite(pair_weight) or pair_weight < 0):
        raise ValueError("Invalid quality target/logit/ranking weight")
    safe = torch.where(mask, values, torch.zeros_like(values))
    pointwise = F.binary_cross_entropy_with_logits(logits, safe, reduction="none")
    per_row = (pointwise * mask).sum(1) / mask.sum(1).clamp_min(1)
    differences = logits.unsqueeze(2) - logits.unsqueeze(1)
    preferred = mask.unsqueeze(2) & mask.unsqueeze(1) & (safe.unsqueeze(2) > safe.unsqueeze(1))
    ranked = (F.softplus(-differences) * preferred).sum((1, 2)) / preferred.sum((1, 2)).clamp_min(1)
    per_row = per_row + pair_weight * ranked
    losses = [per_row[(task_ids == i) & mask.any(1)].mean() for i in range(len(TASK_CONFIGURATIONS))
              if ((task_ids == i) & mask.any(1)).any()]
    return torch.stack(losses).mean() if losses else logits.sum() * 0


def choose(scores, benchmark, margin=0.0):
    names = TASK_CONFIGURATIONS.get(benchmark)
    if (names is None or set(scores) != set(names) or not np.isfinite(margin) or margin < 0
            or any(not np.isfinite(p) or not 0 <= p <= 1 for p in scores.values())):
        raise ValueError("Invalid configuration quality scores/task/margin")
    anchor = ANCHORS[benchmark]
    best = min(names, key=lambda name: (-scores[name], name != anchor, len(CONFIGURATIONS[name]), names.index(name)))
    if scores[best] <= scores[anchor] + margin:
        best = anchor
    return {"configuration": best, "features": list(CONFIGURATIONS[best]),
            "feature_flags": feature_flags(CONFIGURATIONS[best]), "quality_scores": scores,
            "anchor": anchor, "switch_margin": margin, "policy": FORMAT}


class ConfigurationFeatureSelector:
    """Deployable local inference; benchmark routes to its measured catalog."""

    def __init__(self, folder, encoder_folder, *, encoder=None):
        folder = Path(folder)
        self.info = json.loads((folder / "model.json").read_text(encoding="utf-8"))
        if (self.info.get("format_version") != FORMAT or self.info.get("catalog") != catalog()
                or self.info.get("outputs") != [list(x) for x in OUTPUTS]
                or digest(folder / "head.npz") != self.info["head_sha256"]):
            raise ValueError("Invalid or changed configuration model")
        self.encoder = encoder or FrozenTextEncoder(encoder_folder)
        if self.encoder.info != self.info["encoder"]:
            raise ValueError("Encoder differs from training")
        scaler = self.info["schema_scaler"]
        if scaler.get("fields") != ["table_count", "column_count", "text_column_count", "numeric_column_count",
                                     "temporal_column_count", "boolean_column_count"] or scaler.get("transform") != "log1p":
            raise ValueError("Schema scaler order/transform changed")
        self.scaler = SchemaScaler(scaler["mean"], scaler["scale"])
        with np.load(folder / "head.npz", allow_pickle=False) as saved:
            weight, bias = saved["weight"].copy(), saved["bias"].copy()
        if (weight.shape != (len(OUTPUTS), self.info["input_size"]) or bias.shape != (len(OUTPUTS),)
                or not np.isfinite(weight).all() or not np.isfinite(bias).all()):
            raise ValueError("Invalid configuration head tensors")
        self.head = ConfigurationHead(self.info["input_size"]).eval()
        with torch.no_grad():
            self.head.linear.weight.copy_(torch.from_numpy(weight))
            self.head.linear.bias.copy_(torch.from_numpy(bias))
        self.head.requires_grad_(False)

    def select_many(self, inputs, benchmarks):
        if len(inputs) != len(benchmarks) or not inputs or any(b not in TASK_CONFIGURATIONS for b in benchmarks):
            raise ValueError("Provide non-empty inputs and a supported task for each input")
        x = np.concatenate((self.encoder.encode(inputs), self.scaler.transform(inputs)), axis=1)
        with torch.inference_mode():
            scores = self.head(torch.from_numpy(x)).sigmoid().numpy()
        return [choose({name: float(values[OUTPUTS.index((task, name))]) for name in TASK_CONFIGURATIONS[task]},
                       task, self.info["selected"]["margin"]) for values, task in zip(scores, benchmarks)]

    def select_inputs(self, inputs, benchmark):
        return self.select_many([inputs], [benchmark])[0]
