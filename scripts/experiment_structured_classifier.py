"""Train/dev-only experiment: learn compatible masks from observed configuration quality."""

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from adv_text2sql.mcp_servers.text2sql_tool.src.multilabel_classifier import (  # noqa: E402
    FrozenTextEncoder, SchemaScaler, digest,
)
import build_multilabel_dataset as data  # noqa: E402
from atomic_artifacts import write_json  # noqa: E402
from research_metrics import ambrosia_report  # noqa: E402

PROFILES = (
    {"name": "economical", "case": "feat_35", "features": [35]},
    {"name": "rules", "case": "sql_rules", "features": [14, 35]},
    {"name": "hybrid", "case": "sql_ambiguity", "features": [14, 32, 35]},
)
CANDIDATES = (
    {"name": "text_schema_l2_001", "schema": True, "weight_decay": 0.01},
    {"name": "text_schema_l2_01", "schema": True, "weight_decay": 0.1},
    {"name": "text_l2_001", "schema": False, "weight_decay": 0.01},
    {"name": "text_l2_01", "schema": False, "weight_decay": 0.1},
)
DEFAULT_OUTPUT = "research_results/multilabel_structured_dev_20261029"


def profile_scores(flag_scores):
    if flag_scores.ndim != 2 or flag_scores.shape[1] != 2:
        raise ValueError("Expected two structured feature scores")
    # The joint profile can win even when the individual score for 14 is negative.
    return torch.stack((torch.zeros_like(flag_scores[:, 0]), flag_scores[:, 0], flag_scores.sum(1)), dim=1)


def set_quality_loss(scores, correct, complexity_weight=0.01):
    if scores.shape != correct.shape or correct.dtype != torch.bool or scores.shape[1] != len(PROFILES):
        raise ValueError("Expected scores and observed correctness for the three profiles")
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite profile scores")
    # All-correct ties provide no preference. All-wrong rows do not invent a correct profile.
    any_correct = correct.any(1)
    allowed = torch.where(any_correct[:, None], correct, torch.ones_like(correct))
    quality = torch.logsumexp(scores, dim=1) - torch.logsumexp(scores.masked_fill(~allowed, -torch.inf), dim=1)
    quality = quality * any_correct
    # A fixed complexity prior, not claimed as measured token cost.
    expected_extras = (scores.softmax(1) * scores.new_tensor([0, 1, 2])).sum(1)
    return quality.mean() + complexity_weight * expected_extras.mean()


def profile_correct(rows):
    result = []
    for row in rows:
        values = [row["observations"][profile["case"]] for profile in PROFILES]
        if any(not value["observed"] or type(value["correct"]) is not bool for value in values):
            raise ValueError("All compared configuration outcomes must be observed")
        result.append([value["correct"] for value in values])
    return torch.tensor(result, dtype=torch.bool)


def matrix(embeddings, rows, scaler, use_schema):
    structural = scaler.transform([r["inputs"] for r in rows])
    if not use_schema:
        structural = np.zeros_like(structural)
    return torch.from_numpy(np.concatenate((embeddings, structural), axis=1))


def fit(x, correct, *, weight_decay, epochs=120, seed=42):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    head = torch.nn.Linear(x.shape[1], 2)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=weight_decay)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = set_quality_loss(profile_scores(head(x)), correct)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return head.eval(), history


def predict(head, x):
    with torch.inference_mode():
        return profile_scores(head(x)).argmax(1).tolist()


def replay(rows, choices):
    if len(rows) != len(choices) or not rows:
        raise ValueError("Replay coverage differs")
    report = {"benchmarks": {}, "choices": dict(Counter(PROFILES[index]["name"] for index in choices))}
    for benchmark in ("bird", "ambrosia"):
        pairs = [(r, PROFILES[index]) for r, index in zip(rows, choices) if r["benchmark"] == benchmark]
        if not pairs:
            continue
        correct = sum(r["observations"][profile["case"]]["correct"] for r, profile in pairs)
        result = {"questions": len(pairs), "correct": correct, "accuracy": 100 * correct / len(pairs)}
        if benchmark == "ambrosia":
            items = [{"question_id": r["id"], "SQL": "ambiguous" if r["analysis"]["gold_ambiguous"] else "SELECT"}
                     for r, _ in pairs]
            predictions = {}
            for row, profile in pairs:
                value = row["observations"][profile["case"]]
                predictions[row["id"]] = "error" if value["generation_error"] else "ambiguous" if value["predicted_ambiguous"] else "SELECT"
            result["ambiguity_metrics"] = ambrosia_report(items, predictions)["ambiguity_metrics"]
        report["benchmarks"][benchmark] = result
    report["macro_accuracy"] = sum(v["accuracy"] for v in report["benchmarks"].values()) / len(report["benchmarks"])
    report["mean_extra_features"] = sum(len(PROFILES[i]["features"]) - 1 for i in choices) / len(rows)
    return report


def training_cv(train_rows, embeddings, folds, *, epochs=120):
    if {r["split"] for r in train_rows} != {"train"}:
        raise ValueError("CV accepts training rows only")
    positions = {r["id"]: i for i, r in enumerate(train_rows)}
    reports = []
    for candidate in CANDIDATES:
        oof = {}
        fold_reports = []
        for fold in folds:
            train_ids, valid_ids = fold["train_ids"], fold["validation_ids"]
            if (set(train_ids) & set(valid_ids) or set(train_ids) | set(valid_ids) != set(positions)
                    or not train_ids or not valid_ids):
                raise ValueError("CV includes outer dev/test, missing IDs, or overlapping rows")
            fitting = [train_rows[positions[key]] for key in train_ids]
            validation = [train_rows[positions[key]] for key in valid_ids]
            if {r["group_id"] for r in fitting} & {r["group_id"] for r in validation}:
                raise ValueError("CV crosses DB/family groups")
            scaler = SchemaScaler.fit([r["inputs"] for r in fitting])
            x = matrix(embeddings[[positions[key] for key in train_ids]], fitting, scaler, candidate["schema"])
            vx = matrix(embeddings[[positions[key] for key in valid_ids]], validation, scaler, candidate["schema"])
            head, _ = fit(x, profile_correct(fitting), weight_decay=candidate["weight_decay"], epochs=epochs)
            choices = predict(head, vx)
            for key, value in zip(valid_ids, choices):
                if key in oof:
                    raise ValueError("Repeated OOF validation row")
                oof[key] = value
            fold_reports.append({"fold": fold["fold"], "replay": replay(validation, choices)})
        if set(oof) != set(positions):
            raise ValueError("Incomplete OOF predictions")
        reports.append({"candidate": candidate, "oof": replay(train_rows, [oof[r["id"]] for r in train_rows]),
                        "folds": fold_reports})
    best = min(reports, key=lambda value: (-value["oof"]["macro_accuracy"], value["oof"]["mean_extra_features"],
                                          [r["name"] for r in CANDIDATES].index(value["candidate"]["name"])))
    return best["candidate"], reports


def markdown(report):
    result = ["# Structured classifier: train/dev only", "", f"CV-selected candidate: `{report['selected']['name']}`.", "",
              "| Part | Approach | BIRD correct | BIRD accuracy | Ambrosia correct | Ambrosia accuracy | Ambrosia F1 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for split, methods in report["replay"].items():
        for name, metrics in methods.items():
            bird, amb = metrics["benchmarks"]["bird"], metrics["benchmarks"]["ambrosia"]
            result.append(f"| {split} | {name} | {bird['correct']}/{bird['questions']} | {bird['accuracy']:.2f}% | "
                          f"{amb['correct']}/{amb['questions']} | {amb['accuracy']:.2f}% | {100 * amb['ambiguity_metrics']['f1']:.2f}% |")
    result += ["", "Predictions are restricted to observed profiles 35, 14+35, 14+32+35.",
               "This is a structured classifier over three masks, not independent selection of all 16 flags.",
               "Scores are replayed from saved single-pass outcomes, not a new SQL/LLM benchmark.",
               "Hyperparameters selected by four grouped folds inside train. Dev only reports transfer.",
               "No test predictions, test quality evaluation, or paid requests were performed.",
               "The existing test has already been viewed; future independent validation is still needed."]
    return "\n".join(result) + "\n"


def run(output, encoder_folder, *, epochs=120):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output folder; old experiments are immutable")
    dataset = ROOT / data.DEFAULT_OUTPUT
    manifest = data.verify(dataset)
    # Discard test immediately; it cannot reach embedding, fitting, selection, or replay below.
    rows = [r for r in data.lines(dataset / "records.jsonl") if r["split"] in ("train", "dev")]
    training = [r for r in rows if r["split"] == "train"]
    development = [r for r in rows if r["split"] == "dev"]
    encoder = FrozenTextEncoder(encoder_folder)
    train_embeddings = encoder.encode([r["inputs"] for r in training])
    dev_embeddings = encoder.encode([r["inputs"] for r in development])
    selected, cv = training_cv(training, train_embeddings, data.parent.read(dataset / "cv_folds.json"), epochs=epochs)
    scaler = SchemaScaler.fit([r["inputs"] for r in training])
    train_x = matrix(train_embeddings, training, scaler, selected["schema"])
    dev_x = matrix(dev_embeddings, development, scaler, selected["schema"])
    head, history = fit(train_x, profile_correct(training), weight_decay=selected["weight_decay"], epochs=epochs)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "head.npz", weight=head.weight.detach().numpy(), bias=head.bias.detach().numpy())
    source_files = ("scripts/experiment_structured_classifier.py",
                    "src/adv_text2sql/mcp_servers/text2sql_tool/src/multilabel_classifier.py")
    info = {"format_version": "structured-feature-experiment-v1", "profiles": list(PROFILES), "selected": selected,
            "candidate_catalog": list(CANDIDATES), "encoder": encoder.info, "schema_scaler": scaler.as_dict(),
            "epochs": epochs, "seed": 42, "complexity_prior": 0.01,
            "train_ids": [r["id"] for r in training], "dev_ids": [r["id"] for r in development],
            "head_sha256": digest(output / "head.npz"), "dataset_manifest_sha256": digest(dataset / "manifest.json"),
            "source_files_sha256": {name: digest(ROOT / name) for name in source_files},
            "generator_model": manifest["model"], "test_used": False, "paid_calls": 0,
            "selection": "grouped train OOF macro quality; ties: fewer optional features, catalog order"}
    write_json(output / "model.json", info)
    write_json(output / "train_cv.json", cv)
    write_json(output / "training_loss.json", history)
    report = {"selected": selected, "test_used": False, "paid_calls": 0, "replay": {}}
    saved = []
    for name, subset, x in (("train", training, train_x), ("dev", development, dev_x)):
        choices = predict(head, x)
        report["replay"][name] = {profile["name"]: replay(subset, [i] * len(subset)) for i, profile in enumerate(PROFILES)}
        report["replay"][name]["structured"] = replay(subset, choices)
        saved.extend({"id": row["id"], "split": name, "profile": PROFILES[index]["name"],
                      "features": PROFILES[index]["features"]} for row, index in zip(subset, choices))
    write_json(output / "report.json", report)
    write_json(output / "predictions_train_dev.json", saved)
    (output / "comparison.md").write_text(markdown(report), encoding="utf-8", newline="\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--encoder", default=".cache/classifier/minilm")
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()
    output, encoder = (ROOT / args.out).resolve(), (ROOT / args.encoder).resolve()
    if not output.is_relative_to(ROOT / "research_results") or output == ROOT / "research_results":
        parser.error("Use a dedicated research_results/ experiment folder")
    if not encoder.is_relative_to(ROOT / ".cache/classifier") or args.epochs < 1:
        parser.error("Use the workspace encoder cache and positive epochs")
    torch.set_num_threads(2)
    report = run(output, encoder, epochs=args.epochs)
    print("CV SELECTED:", report["selected"])
    for split, methods in report["replay"].items():
        print(split, {name: round(metrics["macro_accuracy"], 2) for name, metrics in methods.items()})
    print(f"SAVED: {output}; no test evaluation or paid calls")


if __name__ == "__main__":
    main()
