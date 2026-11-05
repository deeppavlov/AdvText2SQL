"""Frozen text encoder + train-standardized schema counts + a linear multilabel head."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .feature_selector import validate_inputs

HEAD_FEATURES = (1, 2, 12, 13, 14, 15, 16, 20, 21, 23, 25, 28, 29, 30, 32, 35)
SCHEMA_FIELDS = ("table_count", "column_count", "text_column_count", "numeric_column_count",
                 "temporal_column_count", "boolean_column_count")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schema_matrix(inputs):
    for item in inputs:
        validate_inputs(item)
        if set(item["schema_stats"]) != set(SCHEMA_FIELDS):
            raise ValueError("Classifier requires all six schema counts")
    return np.log1p(np.array([[row["schema_stats"][key] for key in SCHEMA_FIELDS] for row in inputs], dtype=np.float32))


@dataclass
class SchemaScaler:
    mean: list
    scale: list

    @classmethod
    def fit(cls, training_inputs):
        matrix = schema_matrix(training_inputs)
        if not len(matrix):
            raise ValueError("No training inputs")
        mean, scale = matrix.mean(0), matrix.std(0)
        return cls(mean.tolist(), np.where(scale > 1e-6, scale, 1).tolist())

    def transform(self, inputs):
        if len(self.mean) != 6 or len(self.scale) != 6 or any(value <= 0 for value in self.scale):
            raise ValueError("Invalid train-only schema scaler")
        return ((schema_matrix(inputs) - np.array(self.mean)) / np.array(self.scale)).astype(np.float32)

    def as_dict(self):
        return {"mean": self.mean, "scale": self.scale, "fields": list(SCHEMA_FIELDS), "transform": "log1p"}


class FrozenTextEncoder:
    """Loads only verified local safetensors. No Hub/network access during fit or inference."""

    def __init__(self, folder, *, model=None, tokenizer=None):
        self.folder = Path(folder)
        self.info = json.loads((self.folder / "encoder.json").read_text(encoding="utf-8"))
        if self.info.get("format_version") != "frozen-encoder-v1":
            raise ValueError("Unsupported encoder manifest")
        required = {"config.json", "model.safetensors", "tokenizer.json"}
        if not required <= set(self.info["files_sha256"]):
            raise ValueError("Incomplete frozen encoder")
        for name, expected in self.info["files_sha256"].items():
            if Path(name).name != name or digest(self.folder / name) != expected:
                raise ValueError("Frozen encoder hash changed")
        if model is None or tokenizer is None:
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.folder, local_files_only=True, trust_remote_code=False)
            model = AutoModel.from_pretrained(self.folder, local_files_only=True, trust_remote_code=False, use_safetensors=True)
        self.model, self.tokenizer = model.cpu(), tokenizer
        self.model.requires_grad_(False)
        self.model.eval()

    def encode(self, inputs, batch_size=32):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        texts = []
        for item in inputs:
            validate_inputs(item)
            texts.append(f"Question: {item['question']}\nEvidence: {item['evidence']}")
        arrays = []
        self.model.eval()
        with torch.inference_mode():
            for offset in range(0, len(texts), batch_size):
                tokens = self.tokenizer(texts[offset:offset + batch_size], padding=True, truncation=True,
                                        max_length=self.info["max_length"], return_tensors="pt")
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
                arrays.append(F.normalize(pooled, dim=1).cpu().numpy().copy())
        if not arrays:
            raise ValueError("No inputs to encode")
        return np.concatenate(arrays).astype(np.float32)


class LinearFeatureHead(nn.Module):
    def __init__(self, input_size, output_size=len(HEAD_FEATURES)):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, features):
        return self.linear(features)


def masked_bce(logits, targets, mask, pos_weight=None):
    if logits.shape != targets.shape or logits.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("Logits, targets and boolean mask must have identical shapes")
    if not torch.isfinite(logits).all() or not torch.isfinite(targets[mask]).all():
        raise ValueError("Non-finite observed logits/targets")
    if ((targets[mask] != 0) & (targets[mask] != 1)).any():
        raise ValueError("Observed targets must be binary")
    if pos_weight is not None and (not torch.isfinite(pos_weight).all() or (pos_weight <= 0).any()):
        raise ValueError("Invalid positive class weights")
    # NaN placeholders at unobserved positions must not poison the loss/gradient.
    safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
    loss = F.binary_cross_entropy_with_logits(logits, safe_targets, pos_weight=pos_weight, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def positive_weights(targets, mask, cap=20.0):
    positive = ((targets == 1) & mask).sum(0).float()
    negative = ((targets == 0) & mask).sum(0).float()
    return torch.where(positive > 0, (negative / positive.clamp_min(1)).sqrt().clamp(1, cap), 1.0)


def constrained_features(probabilities, threshold, max_features=3):
    if not 0 < threshold <= 1 or not 1 <= max_features <= len(HEAD_FEATURES):
        raise ValueError("Invalid threshold or feature budget")
    if set(probabilities) != {str(n) for n in HEAD_FEATURES}:
        raise ValueError("Probabilities must match the registered heads")
    if any(not np.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError("Invalid feature probabilities")
    ranked = sorted(HEAD_FEATURES, key=lambda n: (-probabilities[str(n)], n))
    raw = [n for n in ranked if probabilities[str(n)] >= threshold]
    selected, removed = [], []
    # Known overrides in the legacy generator, not labels learned from held-out data.
    conflicts = ({14, 29}, {20, 28}, {12, 32})
    for number in raw:
        reason = "mechanism_override" if any(number in pair and any(n in pair for n in selected) for pair in conflicts) else None
        if reason is None and len(selected) >= max_features:
            reason = "feature_budget"
        if reason:
            removed.append({"feature": number, "reason": reason})
        else:
            selected.append(number)
    return {"raw_features": raw, "features": sorted(selected), "constraints_removed": removed,
            "feature_flags": {f"FEAT_{n}": n in selected for n in range(1, 37)}}


class MultilabelFeatureSelector:
    def __init__(self, folder, encoder_folder, *, encoder=None):
        self.folder = Path(folder)
        self.info = json.loads((self.folder / "model.json").read_text(encoding="utf-8"))
        if self.info.get("format_version") != "multilabel-classifier-v1" or self.info["feature_numbers"] != list(HEAD_FEATURES):
            raise ValueError("Classifier feature order/format differs")
        if digest(self.folder / "head.npz") != self.info["head_sha256"]:
            raise ValueError("Classifier head changed")
        self.encoder = encoder or FrozenTextEncoder(encoder_folder)
        if self.encoder.info != self.info["encoder"]:
            raise ValueError("Encoder differs from the training encoder")
        scaler = self.info["schema_scaler"]
        if scaler["fields"] != list(SCHEMA_FIELDS) or scaler["transform"] != "log1p":
            raise ValueError("Schema feature order differs")
        self.scaler = SchemaScaler(scaler["mean"], scaler["scale"])
        with np.load(self.folder / "head.npz", allow_pickle=False) as saved:
            weight, bias = saved["weight"].copy(), saved["bias"].copy()
        if (weight.shape != (len(HEAD_FEATURES), self.info["input_size"]) or bias.shape != (len(HEAD_FEATURES),)
                or not np.isfinite(weight).all() or not np.isfinite(bias).all()):
            raise ValueError("Invalid classifier head shapes/values")
        self.head = LinearFeatureHead(self.info["input_size"]).eval()
        with torch.no_grad():
            self.head.linear.weight.copy_(torch.from_numpy(weight))
            self.head.linear.bias.copy_(torch.from_numpy(bias))
        self.head.requires_grad_(False)

    def probabilities(self, inputs):
        features = np.concatenate((self.encoder.encode(inputs), self.scaler.transform(inputs)), axis=1)
        with torch.inference_mode():
            return self.head(torch.from_numpy(features)).sigmoid().cpu().numpy()

    def select_inputs(self, inputs):
        probabilities = {str(n): float(p) for n, p in zip(HEAD_FEATURES, self.probabilities([inputs])[0])}
        decision = constrained_features(probabilities, self.info["threshold"], self.info["max_features"])
        return {**decision, "probabilities": probabilities, "threshold": self.info["threshold"],
                "configuration": "multilabel", "policy": "frozen_encoder_linear_head"}
