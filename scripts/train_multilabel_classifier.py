"""Train only a linear head on frozen local embeddings; evaluate labels separately."""

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from adv_text2sql.mcp_servers.text2sql_tool.src.multilabel_classifier import (  # noqa: E402
    HEAD_FEATURES, FrozenTextEncoder, LinearFeatureHead, MultilabelFeatureSelector, SchemaScaler,
    constrained_features, digest, masked_bce, positive_weights,
)
import build_multilabel_dataset as dataset_builder  # noqa: E402
from atomic_artifacts import write_json  # noqa: E402

DEFAULT_OUTPUT = "research_results/multilabel_classifier_deepseek_20261029"
DEFAULT_ENCODER = ".cache/classifier/minilm"
THRESHOLDS = (0.2, 0.35, 0.5, 0.65, 0.8)
SOURCES = (
    "scripts/train_multilabel_classifier.py", "scripts/prepare_classifier_encoder.py",
    "scripts/run_openrouter_baseline.py", "scripts/multilabel_targets.py",
    "scripts/classifier_benchmark.py",
    "src/adv_text2sql/mcp_servers/text2sql_tool/src/multilabel_classifier.py",
    "src/adv_text2sql/mcp_servers/text2sql_tool/src/multilabel_generator.py",
    "src/adv_text2sql/mcp_servers/text2sql_tool/src/feature_selector.py",
    "src/adv_text2sql/mcp_servers/text2sql_tool/src/selective_generator.py",
    "src/adv_text2sql/mcp_servers/text2sql_tool/src/text2sql_implementation.py",
    "src/adv_text2sql/mcp_servers/text2sql_tool/src/prompts.py",
)


def labels(rows):
    mask = np.array([r["multilabel"]["loss_mask"] for r in rows], dtype=bool)
    values = np.array([[np.nan if value is None else value for value in r["multilabel"]["benefit_targets"]]
                       for r in rows], dtype=np.float32)
    return torch.from_numpy(values), torch.from_numpy(mask)


def decisions(probabilities, threshold, max_features):
    selected = np.zeros(probabilities.shape, dtype=bool)
    for i, scores in enumerate(probabilities):
        decision = constrained_features({str(n): float(p) for n, p in zip(HEAD_FEATURES, scores)}, threshold, max_features)
        selected[i] = [n in decision["features"] for n in HEAD_FEATURES]
    return selected


def metrics(targets, mask, predictions):
    targets, mask, predictions = np.asarray(targets), np.asarray(mask), np.asarray(predictions)
    if targets.shape != mask.shape or predictions.shape != mask.shape or mask.dtype != bool:
        raise ValueError("Target/mask/prediction shapes differ")
    positive = targets == 1
    tp = int((predictions & positive & mask).sum())
    fp = int((predictions & ~positive & mask).sum())
    fn = int((~predictions & positive & mask).sum())
    errors = int(((predictions != positive) & mask).sum())
    observed = int(mask.sum())
    per_feature = {}
    for i, number in enumerate(HEAD_FEATURES):
        a = int((predictions[:, i] & positive[:, i] & mask[:, i]).sum())
        b = int((predictions[:, i] & ~positive[:, i] & mask[:, i]).sum())
        c = int((~predictions[:, i] & positive[:, i] & mask[:, i]).sum())
        per_feature[f"FEAT_{number}"] = {"tp": a, "fp": b, "fn": c,
                                        "f1": 2 * a / (2 * a + b + c) if 2 * a + b + c else 0.0}
    eligible_rows = mask.any(1)
    return {"observed_labels": observed, "tp": tp, "fp": fp, "fn": fn,
            "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
            "micro_recall": tp / (tp + fn) if tp + fn else 0.0,
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "macro_f1": sum(v["f1"] for v in per_feature.values()) / len(per_feature),
            "hamming_loss": errors / observed if observed else None,
            "subset_accuracy": float((((predictions == positive) | ~mask).all(1))[eligible_rows].mean())
            if eligible_rows.any() else None,
            "mean_selected_features": float(predictions.sum(1).mean()), "features": per_feature}


def choose_threshold(targets, mask, probabilities, max_features):
    candidates = []
    for threshold in THRESHOLDS:
        report = metrics(targets, mask, decisions(probabilities, threshold, max_features))
        candidates.append({"threshold": threshold, "metrics": report})
    best = min(candidates, key=lambda row: (-row["metrics"]["micro_f1"],
                                          row["metrics"]["mean_selected_features"], -row["threshold"]))
    return best["threshold"], candidates


def fit_head(train_x, train_y, train_mask, dev_x, dev_y, dev_mask, *, epochs=200, patience=30, seed=42):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    head = LinearFeatureHead(train_x.shape[1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=0.01)
    weights = positive_weights(train_y, train_mask)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        head.train()
        optimizer.zero_grad()
        loss = masked_bce(head(train_x), train_y, train_mask, weights)
        loss.backward()
        optimizer.step()
        head.eval()
        with torch.inference_mode():
            dev_loss = float(masked_bce(head(dev_x), dev_y, dev_mask, weights))
        history.append({"epoch": epoch, "train_weighted_loss": float(loss.detach()), "dev_weighted_loss": dev_loss})
        if dev_loss < best_loss - 1e-8:
            best_loss, best_state, best_epoch, stale = dev_loss, copy.deepcopy(head.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise ValueError("No finite training checkpoint")
    head.load_state_dict(best_state)
    return head.eval(), {"best_epoch": best_epoch, "best_dev_weighted_loss": best_loss,
                         "positive_weights_train_only": weights.tolist(), "history": history}


def load_rows(dataset):
    manifest = dataset_builder.verify(dataset)
    if manifest["target_features"] != [f"FEAT_{n}" for n in HEAD_FEATURES]:
        raise ValueError("Dataset target order differs from classifier")
    return manifest, dataset_builder.lines(dataset / "records.jsonl")


def train(dataset, encoder_folder, output, *, epochs=200, patience=30, max_features=3, seed=42):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output exists; use a new directory to keep the frozen experiment")
    manifest, rows = load_rows(dataset)
    training = [r for r in rows if r["split"] == "train"]
    development = [r for r in rows if r["split"] == "dev"]
    # No test rows/outcomes enter scaling, embeddings, training, threshold selection, or early stopping.
    train_inputs, dev_inputs = [r["inputs"] for r in training], [r["inputs"] for r in development]
    encoder = FrozenTextEncoder(encoder_folder)
    scaler = SchemaScaler.fit(train_inputs)
    train_x = torch.from_numpy(np.concatenate((encoder.encode(train_inputs), scaler.transform(train_inputs)), axis=1))
    dev_x = torch.from_numpy(np.concatenate((encoder.encode(dev_inputs), scaler.transform(dev_inputs)), axis=1))
    train_y, train_mask = labels(training)
    dev_y, dev_mask = labels(development)
    head, history = fit_head(train_x, train_y, train_mask, dev_x, dev_y, dev_mask, epochs=epochs, patience=patience, seed=seed)
    with torch.inference_mode():
        probabilities = head(dev_x).sigmoid().numpy()
    threshold, candidates = choose_threshold(dev_y.numpy(), dev_mask.numpy(), probabilities, max_features)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "head.npz", weight=head.linear.weight.detach().numpy(), bias=head.linear.bias.detach().numpy())
    info = {
        "format_version": "multilabel-classifier-v1", "feature_numbers": list(HEAD_FEATURES),
        "input_size": train_x.shape[1], "encoder": encoder.info, "encoder_trainable_parameters": 0,
        "head_trainable_parameters": sum(p.numel() for p in head.parameters()),
        "schema_scaler": scaler.as_dict(), "threshold": threshold, "threshold_candidates": list(THRESHOLDS),
        "max_features": max_features, "seed": seed, "epochs_limit": epochs, "patience": patience,
        "training": {"optimizer": "AdamW", "learning_rate": 0.01, "weight_decay": 0.01,
                     "class_weights": "sqrt(negative/positive) capped at 20; train only",
                     "loss": "masked BCEWithLogits; benefit targets; ties=0 means no observed gain",
                     "checkpoint": "minimum dev weighted loss", "threshold_selection": "maximum dev constrained micro F1; ties: fewer flags, higher threshold"},
        "train_ids": [r["id"] for r in training], "dev_ids": [r["id"] for r in development],
        "test_used_for_selection": False, "test_already_viewed_in_phase1": True,
        "dataset": dataset.relative_to(ROOT).as_posix(), "dataset_manifest_sha256": digest(dataset / "manifest.json"),
        "generator_model": manifest["model"], "head_sha256": digest(output / "head.npz"),
        "source_files_sha256": {name: digest(ROOT / name) for name in SOURCES},
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "safetensors")},
        "downstream_quality_measured": False,
    }
    write_json(output / "model.json", info)  # Frozen before any held-out evaluation.
    write_json(output / "training.json", history)
    write_json(output / "threshold_selection.json", candidates)
    return info


def validate_artifact(output):
    info = dataset_builder.parent.read(output / "model.json")
    if any(digest(ROOT / file) != expected for file, expected in info["source_files_sha256"].items()):
        raise ValueError("Classifier/generator code changed since training; create a new experiment")
    dataset = ROOT / info["dataset"]
    if digest(dataset / "manifest.json") != info["dataset_manifest_sha256"]:
        raise ValueError("Training dataset changed")
    return info, dataset


def evaluate(output, encoder_folder):
    info, dataset = validate_artifact(output)
    _, rows = load_rows(dataset)
    selector = MultilabelFeatureSelector(output, encoder_folder)
    report = {"format_version": "multilabel-label-evaluation-v1", "model_sha256": digest(output / "model.json"),
              "scope": "label prediction only; no SQL/F1 downstream claims for unobserved feature combinations",
              "retrospective_test": True, "live_llm_calls": 0, "splits": {}}
    predictions = []
    for split in ("train", "dev", "test"):
        subset = [r for r in rows if r["split"] == split]
        probabilities = selector.probabilities([r["inputs"] for r in subset])
        masks = decisions(probabilities, info["threshold"], info["max_features"])
        target, mask = labels(subset)
        fixed = np.array([[n in (14, 32, 35) for n in HEAD_FEATURES]] * len(subset), dtype=bool)
        report["splits"][split] = {
            "classifier": metrics(target.numpy(), mask.numpy(), masks),
            "all_off": metrics(target.numpy(), mask.numpy(), np.zeros_like(masks)),
            "static_14_32_35": metrics(target.numpy(), mask.numpy(), fixed),
        }
        for row, scores, selected in zip(subset, probabilities, masks):
            predictions.append({"id": row["id"], "split": split,
                                "probabilities": {str(n): float(p) for n, p in zip(HEAD_FEATURES, scores)},
                                "features": [n for n, enabled in zip(HEAD_FEATURES, selected) if enabled]})
    for name, value in (("evaluation.json", report), ("predictions.json", predictions)):
        content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        path = output / name
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError("Frozen evaluation differs; use a new experiment")
        path.write_text(content, encoding="utf-8", newline="\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "evaluate"))
    parser.add_argument("--dataset", default=dataset_builder.DEFAULT_OUTPUT)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--max-features", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    output, dataset, encoder = (ROOT / args.out).resolve(), (ROOT / args.dataset).resolve(), (ROOT / args.encoder).resolve()
    if not output.is_relative_to(ROOT / "research_results") or output == ROOT / "research_results":
        parser.error("Use a dedicated experiment directory inside research_results/")
    if not dataset.is_relative_to(ROOT / "data") or not encoder.is_relative_to(ROOT / ".cache/classifier"):
        parser.error("Use workspace data/ and .cache/classifier/ paths")
    if args.epochs < 1 or args.patience < 1 or not 1 <= args.max_features <= len(HEAD_FEATURES):
        parser.error("Invalid epochs, patience or feature budget")
    if args.dry_run:
        _, rows = load_rows(dataset)
        print(json.dumps({"rows": len(rows), "train": sum(r["split"] == "train" for r in rows),
                          "dev": sum(r["split"] == "dev" for r in rows), "heads": len(HEAD_FEATURES),
                          "encoder": str(encoder), "output": str(output), "paid_calls": 0}, indent=2))
    elif args.action == "train":
        info = train(dataset, encoder, output, epochs=args.epochs, patience=args.patience,
                     max_features=args.max_features, seed=args.seed)
        print(f"TRAINED: frozen encoder, {info['head_trainable_parameters']} head parameters; threshold={info['threshold']}: {output}")
    else:
        report = evaluate(output, encoder)
        for split, methods in report["splits"].items():
            print(split, {name: {key: round(value[key], 4) for key in ("micro_f1", "hamming_loss", "mean_selected_features")}
                          for name, value in methods.items()})


if __name__ == "__main__":
    main()
