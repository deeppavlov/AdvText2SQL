"""Offline train/CV/dev for joint configuration quality. Never calls an LLM or DB."""

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adv_text2sql.mcp_servers.text2sql_tool.src.configuration_classifier_v2 import (  # noqa: E402
    ANCHORS, FORMAT, OUTPUTS, ConfigurationFeatureSelector, ConfigurationHead, choose, quality_loss, targets,
)
from adv_text2sql.mcp_servers.text2sql_tool.src.feature_catalog_v2 import (  # noqa: E402
    CONFIGURATIONS, TASK_CONFIGURATIONS, catalog,
)
from adv_text2sql.mcp_servers.text2sql_tool.src.multilabel_classifier import (  # noqa: E402
    FrozenTextEncoder, SchemaScaler,
)
import prepare_classifier_v2 as data  # noqa: E402
import collect_classifier_v2 as collection  # noqa: E402
from durable_prompt_store import write_json  # noqa: E402
from research_metrics import ambrosia_report  # noqa: E402

DEFAULT_OUTPUT = "research_results/classifier_v2_training_20261109"
# Fixed before training; margins are evaluated using the same OOF probabilities.
CANDIDATES = tuple({"weight_decay": decay, "pair_weight": pair}
                   for decay in (0.01, 0.1) for pair in (0.0, 0.5))
MARGINS = (0.0, 0.05, 0.15)
SOURCE_FILES = (
    "scripts/train_configuration_classifier_v2.py",
    "src/adv_text2sql/mcp_servers/text2sql_tool/src/configuration_classifier_v2.py",
    "src/adv_text2sql/mcp_servers/text2sql_tool/src/feature_catalog_v2.py",
    "src/adv_text2sql/mcp_servers/text2sql_tool/src/multilabel_classifier.py",
    "src/adv_text2sql/mcp_servers/text2sql_tool/src/feature_selector.py",
    "scripts/research_metrics.py",
)


def load_dataset():
    folder = ROOT / data.DATA_OUTPUT
    manifest = data.read(folder / "manifest.json")
    frozen = data.read(ROOT / data.DEFAULT_OUTPUT / "experiment.json")
    data.validate(frozen)
    if (not manifest["ready_for_training"] or manifest["test_included"]
            or manifest["collection_manifest_sha256"] != frozen["sha256"] or manifest["catalog"] != catalog()):
        raise ValueError("Incomplete, foreign or unsupported dataset")
    for name, digest in manifest["output_files_sha256"].items():
        if Path(name).name != name or data.digest(folder / name) != digest:
            raise ValueError("Dataset output changed")
    for name, digest in manifest["observation_files_sha256"].items():
        if data.digest(data.local_path(ROOT, name)) != digest:
            raise ValueError("Saved observation changed")
    if data.read(ROOT / data.DEFAULT_OUTPUT / "budget.json")["pending"]:
        raise ValueError("Unsettled collection API budget")
    if any(j["items"] and not collection.completed(ROOT / data.DEFAULT_OUTPUT, frozen, j) for j in frozen["jobs"]):
        raise ValueError("Collection jobs are not complete")
    rows = [json.loads(s) for s in (folder / "examples.jsonl").read_text(encoding="utf-8").splitlines()]
    _, _, _, originals = data.load_context()
    if len({r["id"] for r in rows}) != len(rows) or set(originals) != {r["id"] for r in rows}:
        raise ValueError("Repeated, missing or foreign question")
    truth, files = {}, {}
    for row in rows:
        reference = originals[row["id"]]
        if (row["split"] not in ("train", "dev") or row["split"] != reference["split"]
                or row["benchmark"] != reference["benchmark"] or row["group_id"] != reference["group_id"]
                or row["inputs"] != reference["inputs"] or not all(row["loss_mask"])):
            raise ValueError("Split, inputs, group or observation coverage changed")
        for i, name in enumerate(row["configurations"]):
            obs = row["observations"][name]
            path = obs["source"]
            if path not in manifest["observation_files_sha256"]:
                raise ValueError("Unaudited observation source")
            if path not in files:
                files[path] = data.read(data.local_path(ROOT, path))
            value = files[path][obs["key"]]
            score = data.validate_observation(value, reference["item"], row["benchmark"])
            if score is None or score != row["quality_targets"][i] or score != obs["score"]:
                raise ValueError("Labels differ from validated observations")
            truth[row["id"], name] = value
    targets(rows)
    training = [r for r in rows if r["split"] == "train"]
    development = [r for r in rows if r["split"] == "dev"]
    if not training or not development or {r["group_id"] for r in training} & {r["group_id"] for r in development}:
        raise ValueError("Empty or overlapping groups")
    return manifest, frozen, training, development, truth, originals


def grouped_folds(rows, folds=4, seed=42):
    if not rows or any(r["split"] != "train" for r in rows) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("CV accepts unique train rows only")
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    assigned, loads = {}, {b: [0] * folds for b in TASK_CONFIGURATIONS}
    for task in TASK_CONFIGURATIONS:
        candidates = [g for g, rs in groups.items() if rs[0]["benchmark"] == task]
        if len(candidates) < folds:
            raise ValueError("Too few train groups per task for frozen CV")
        for group in sorted(candidates, key=lambda g: (-len(groups[g]), hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())):
            if {r["benchmark"] for r in groups[group]} != {task}:
                raise ValueError("Mixed-task group requires a stratification review")
            index = min(range(folds), key=lambda i: (loads[task][i], i))
            assigned[group] = index
            loads[task][index] += len(groups[group])
    return [{"fold": i, "train_ids": [r["id"] for r in rows if assigned[r["group_id"]] != i],
             "validation_ids": [r["id"] for r in rows if assigned[r["group_id"]] == i]}
            for i in range(folds)]


def fit(x, rows, candidate, *, epochs=120, seed=42):
    if not rows or any(r["split"] != "train" for r in rows):
        raise ValueError("Fit accepts train rows only")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    values, mask, tasks = targets(rows)
    x = torch.from_numpy(x)
    head = ConfigurationHead(x.shape[1])
    with torch.no_grad():
        head.linear.weight.zero_()
        prevalence = torch.nan_to_num(values, nan=0).sum(0) / mask.sum(0).clamp_min(1)
        head.linear.bias.copy_(torch.logit(prevalence.clamp(0.05, 0.95)))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=candidate["weight_decay"])
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = quality_loss(head(x), values, mask, tasks, candidate["pair_weight"])
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch + 1, "loss": float(loss.detach())})
    return head.eval(), history


def probabilities(head, x):
    with torch.inference_mode():
        return head(torch.from_numpy(x)).sigmoid().numpy()


def decisions(rows, probabilities_, margin):
    return [choose({name: float(scores[OUTPUTS.index((r["benchmark"], name))])
                    for name in TASK_CONFIGURATIONS[r["benchmark"]]}, r["benchmark"], margin)["configuration"]
            for r, scores in zip(rows, probabilities_)]


def replay(rows, choices, truth, originals):
    if len(rows) != len(choices) or not rows:
        raise ValueError("Replay coverage differs")
    reports = {}
    usage = Counter()
    for row, name in zip(rows, choices):
        if name not in row["configurations"]:
            raise ValueError("Unmeasured configuration selected")
        saved = truth[row["id"], name]
        for key in ("task_calls", "prompt_tokens", "completion_tokens"):
            if key not in saved.get("usage", {}):
                raise ValueError("Replay cost lacks observed usage")
            usage[key] += saved["usage"][key]
    for task in TASK_CONFIGURATIONS:
        pairs = [(r, name) for r, name in zip(rows, choices) if r["benchmark"] == task]
        if not pairs:
            continue
        correct = int(sum(truth[r["id"], name]["score"] for r, name in pairs))
        out = {"questions": len(pairs), "correct": correct, "accuracy": 100 * correct / len(pairs),
               "selected_configurations": dict(Counter(name for _, name in pairs))}
        if task == "ambrosia":
            items = [originals[r["id"]]["item"] for r, _ in pairs]
            predictions = {str(originals[r["id"]]["item"]["question_id"]): truth[r["id"], name]["prediction"]
                           for r, name in pairs}
            out.update(ambrosia_report(items, predictions))
        reports[task] = out
    return {"benchmarks": reports, "macro_accuracy": sum(v["accuracy"] for v in reports.values()) / len(reports),
            "saved_response_usage": dict(usage), "mean_features": sum(len(CONFIGURATIONS[n]) for n in choices) / len(choices)}


def matrix(embeddings, rows, scaler):
    return np.concatenate((embeddings, scaler.transform([r["inputs"] for r in rows])), axis=1)


def train_cv(rows, embeddings, truth, originals, *, epochs=120):
    folds = grouped_folds(rows)
    positions = {r["id"]: i for i, r in enumerate(rows)}
    trials = []
    for candidate in CANDIDATES:
        oof = np.full((len(rows), len(OUTPUTS)), np.nan, dtype=np.float32)
        for fold in folds:
            ti, vi = ([positions[k] for k in fold[key]] for key in ("train_ids", "validation_ids"))
            fitting, validation = [rows[i] for i in ti], [rows[i] for i in vi]
            if {r["group_id"] for r in fitting} & {r["group_id"] for r in validation}:
                raise ValueError("CV split leaked a group")
            scaler = SchemaScaler.fit([r["inputs"] for r in fitting])
            head, _ = fit(matrix(embeddings[ti], fitting, scaler), fitting, candidate, epochs=epochs)
            oof[vi] = probabilities(head, matrix(embeddings[vi], validation, scaler))
        if not np.isfinite(oof).all():
            raise ValueError("Incomplete grouped OOF predictions")
        for margin in MARGINS:
            selected = dict(candidate, margin=margin)
            report = replay(rows, decisions(rows, oof, margin), truth, originals)
            trials.append({"candidate": selected, "oof": report})
        print("CV", candidate, "finished", flush=True)
    # Quality first; exact ties prefer retaining the anchor, then deterministic catalog order.
    best = min(enumerate(trials), key=lambda v: (-v[1]["oof"]["macro_accuracy"], -v[1]["candidate"]["margin"], v[0]))[1]
    return best["candidate"], {"folds": folds, "trials": trials,
                              "selection": "grouped train OOF macro accuracy; tie: larger anchor margin, catalog order"}


def markdown(report):
    lines = ["# Configuration classifier v2: saved-outcome replay", "",
             "| Split | Policy | BIRD correct | BIRD EX | Ambrosia correct | Ambrosia Accuracy | Ambrosia F1 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for split, methods in report["replay"].items():
        for policy, metrics in methods.items():
            b, a = (metrics["benchmarks"][k] for k in ("bird", "ambrosia"))
            lines.append(f"| {split} | {policy} | {b['correct']}/{b['questions']} | {b['accuracy']:.2f}% | "
                         f"{a['correct']}/{a['questions']} | {a['accuracy']:.2f}% | {100*a['ambiguity_metrics']['f1']:.2f}% |")
    lines += ["", "Train-only grouped CV chooses regularization, ranking loss and anchor margin.",
              "Dev is reported after the model is frozen; no test rows are loaded.",
              "This is replay of saved single-pass responses, not a fresh downstream LLM run.",
              "Outputs are complete measured configurations, not independent feature labels.",
              "Ambrosia scores ambiguity decisions; SQL execution accuracy is measured only for BIRD.",
              "No paid requests, database queries, encoder updates or LoRA training."]
    return "\n".join(lines) + "\n"


def run(output, encoder_folder, *, epochs=120):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output already exists; use a new directory")
    manifest, frozen, training, development, truth, originals = load_dataset()
    # Experiment specification exists before fitting or examining any replay result.
    output.mkdir(parents=True, exist_ok=True)
    spec = {"format": FORMAT, "candidate_grid": list(CANDIDATES), "margins": list(MARGINS),
            "epochs": epochs, "seed": 42, "anchors": ANCHORS,
            "dataset_manifest_sha256": data.digest(ROOT/data.DATA_OUTPUT/"manifest.json"),
            "source_files_sha256": {n: data.digest(ROOT/n) for n in SOURCE_FILES},
            "selection_uses": "four grouped folds inside train; no outer dev/test outcomes"}
    write_json(output/"experiment.json", spec)
    encoder = FrozenTextEncoder(encoder_folder)
    train_embeddings = encoder.encode([r["inputs"] for r in training])
    selected, cv = train_cv(training, train_embeddings, truth, originals, epochs=epochs)
    scaler = SchemaScaler.fit([r["inputs"] for r in training])
    head, history = fit(matrix(train_embeddings, training, scaler), training, selected, epochs=epochs)
    np.savez(output/"head.npz", weight=head.linear.weight.detach().numpy(), bias=head.linear.bias.detach().numpy())
    info = {"format_version": FORMAT, "catalog": catalog(), "outputs": [list(x) for x in OUTPUTS],
            "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "numpy", "safetensors")},
            "selected": selected, "input_size": head.linear.in_features,
            "head_trainable_parameters": sum(p.numel() for p in head.parameters()), "encoder_trainable_parameters": 0,
            "encoder": encoder.info, "schema_scaler": scaler.as_dict(), "head_sha256": data.digest(output/"head.npz"),
            "training": spec, "generator_model": frozen["model"],
            "generator_candidates": frozen["candidates"], "collection_manifest_sha256": frozen["sha256"],
            "train_ids": [r["id"] for r in training], "dev_ids": [r["id"] for r in development],
            "loss": "task-balanced per-question masked BCE + weighted discordant configuration-pair logistic loss",
            "live_llm_calls": 0, "test_used": False, "independent_holdout_claim": False}
    write_json(output/"model.json", info)  # Freeze before dev embedding/inference/replay.
    write_json(output/"train_cv.json", cv)
    write_json(output/"training_loss.json", history)
    selector = ConfigurationFeatureSelector(output, encoder_folder, encoder=encoder)
    report = {"format": FORMAT, "selected": selected, "live_llm_calls": 0, "test_used": False,
              "scope": "saved-response replay; not a new LLM/evaluator execution", "replay": {}}
    predictions = []
    for split, rows in (("train", training), ("dev", development)):
        chosen = selector.select_many([r["inputs"] for r in rows], [r["benchmark"] for r in rows])
        choices = [v["configuration"] for v in chosen]
        methods = {"all_false": ["baseline"] * len(rows), "task_anchor": [ANCHORS[r["benchmark"]] for r in rows],
                   "hybrid_14_32_35": ["hybrid"] * len(rows), "classifier_v2": choices}
        report["replay"][split] = {k: replay(rows, v, truth, originals) for k, v in methods.items()}
        for row, decision in zip(rows, chosen):
            predictions.append({"id": row["id"], "split": split, "benchmark": row["benchmark"], **decision})
    write_json(output/"report.json", report)
    write_json(output/"predictions_train_dev.json", predictions)
    (output/"comparison.md").write_text(markdown(report), encoding="utf-8", newline="\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--encoder", default=".cache/classifier/minilm")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output, encoder = (ROOT/args.out).resolve(), (ROOT/args.encoder).resolve()
    if (not output.is_relative_to(ROOT/"research_results") or output == ROOT/"research_results"
            or not encoder.is_relative_to(ROOT/".cache/classifier") or args.epochs < 1):
        parser.error("Use a new research_results subfolder, workspace encoder cache, positive epochs")
    torch.set_num_threads(2)
    if args.dry_run:
        _, _, train, dev, _, _ = load_dataset()
        print({"train": len(train), "dev": len(dev), "output_heads": len(OUTPUTS),
               "folds": len(grouped_folds(train)), "candidates": len(CANDIDATES)*len(MARGINS), "paid_calls": 0})
        return
    report = run(output, encoder, epochs=args.epochs)
    print(json.dumps({"selected": report["selected"], "dev": report["replay"]["dev"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
