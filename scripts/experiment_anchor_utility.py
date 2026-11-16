"""Fixed train-only grouped-CV experiment: quality vs anchor utility and schema grounding."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import torch

import prepare_classifier_v2 as data
import train_configuration_classifier_v2 as base
from durable_prompt_store import write_json

sys.path.insert(0, str(data.ROOT / "src"))
from adv_text2sql.mcp_servers.text2sql_tool.src.configuration_classifier_v2 import (  # noqa: E402
    ANCHORS, targets,
)
from adv_text2sql.mcp_servers.text2sql_tool.src.configuration_utility import (  # noqa: E402
    GROUND_FIELDS, UTILITY_OUTPUTS, UtilityHead, choose_utility, grounding_features, utility_loss, utility_targets,
)
from adv_text2sql.mcp_servers.text2sql_tool.src.multilabel_classifier import (  # noqa: E402
    FrozenTextEncoder, SchemaScaler,
)

DEFAULT_OUTPUT = "research_results/classifier_anchor_utility_20261109"
VARIANTS = (
    {"name": "quality_base", "objective": "quality", "grounded": False},
    {"name": "quality_grounded", "objective": "quality", "grounded": True},
    {"name": "utility_base", "objective": "utility", "grounded": False},
    {"name": "utility_grounded", "objective": "utility", "grounded": True},
)
EPOCHS, SEED, MARGIN = 120, 42, 0.15
FIT = {"weight_decay": 0.01, "pair_weight": 0.0}
SOURCE_FILES = (
    "scripts/experiment_anchor_utility.py",
    "src/adv_text2sql/mcp_servers/text2sql_tool/src/configuration_utility.py",
    *base.SOURCE_FILES,
)


def assert_train(rows):
    if not rows or any(r["split"] != "train" for r in rows):
        raise ValueError("This experiment accepts train rows only")


def load_train():
    """Verify artifacts but select train before looking up any observation outcomes.

    Source manifests/checkpoint files can contain outer-dev metadata and records;
    only train records are indexed, scored, embedded or used for model selection.
    """
    folder = data.ROOT / data.DATA_OUTPUT
    manifest = data.read(folder / "manifest.json")
    frozen = data.read(data.ROOT / data.DEFAULT_OUTPUT / "experiment.json")
    data.validate(frozen)
    if (not manifest["ready_for_training"] or manifest["test_included"]
            or manifest["collection_manifest_sha256"] != frozen["sha256"]):
        raise ValueError("Unsupported/incomplete dataset")
    for name, sha in manifest["output_files_sha256"].items():
        if Path(name).name != name or data.digest(folder / name) != sha:
            raise ValueError("Dataset changed")
    for name, sha in manifest["observation_files_sha256"].items():
        if data.digest(data.local_path(data.ROOT, name)) != sha:
            raise ValueError("Observation file changed")
    rows = [r for line in (folder / "examples.jsonl").read_text(encoding="utf-8").splitlines()
            if (r := json.loads(line))["split"] == "train"]
    assert_train(rows)
    _, _, _, all_originals = data.load_context()
    originals = {key: value for key, value in all_originals.items() if value["split"] == "train"}
    if len({r["id"] for r in rows}) != len(rows) or set(originals) != {r["id"] for r in rows}:
        raise ValueError("Train coverage changed")
    truth, files = {}, {}
    for r in rows:
        original = originals[r["id"]]
        if (r["inputs"] != original["inputs"] or r["group_id"] != original["group_id"]
                or r["benchmark"] != original["benchmark"] or not all(r["loss_mask"])):
            raise ValueError("Train identity/inputs/coverage changed")
        for i, name in enumerate(r["configurations"]):
            obs = r["observations"][name]
            if obs["source"] not in manifest["observation_files_sha256"]:
                raise ValueError("Unverified observation source")
            if obs["source"] not in files:
                files[obs["source"]] = data.read(data.local_path(data.ROOT, obs["source"]))
            value = files[obs["source"]][obs["key"]]
            score = data.validate_observation(value, original["item"], r["benchmark"])
            if score is None or score != r["quality_targets"][i] or score != obs["score"]:
                raise ValueError("Invalid training observation")
            truth[r["id"], name] = value
    targets(rows)
    return frozen, rows, truth, originals


def grounded_matrix(rows, originals, frozen):
    assert_train(rows)
    vectors = []
    for r in rows:
        # DB ID only retrieves the schema. No identifier or dataset family enters the feature vector.
        schema = frozen["runtime_base"]["schema_audit"]["schemas"][r["benchmark"]][originals[r["id"]]["db_id"]]["rows"]
        vectors.append(grounding_features(r["inputs"], schema))
    return np.stack(vectors)


def scale_grounding(fit_values, validation_values):
    if (fit_values.ndim != 2 or validation_values.ndim != 2 or not len(fit_values)
            or fit_values.shape[1] != len(GROUND_FIELDS) or validation_values.shape[1] != len(GROUND_FIELDS)
            or not np.isfinite(fit_values).all() or not np.isfinite(validation_values).all()
            or (fit_values < 0).any() or (validation_values < 0).any()):
        raise ValueError("Invalid grounding matrices")
    fitting, validation = np.log1p(fit_values), np.log1p(validation_values)
    mean, scale = fitting.mean(0), fitting.std(0)
    scale = np.where(scale > 1e-6, scale, 1)
    state = {"fields": list(GROUND_FIELDS), "transform": "log1p", "mean": mean.tolist(), "scale": scale.tolist()}
    return ((fitting-mean)/scale).astype(np.float32), ((validation-mean)/scale).astype(np.float32), state


def fit_utility(x, rows):
    assert_train(rows)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    y, mask, tasks = utility_targets(rows)
    head = UtilityHead(x.shape[1])
    with torch.no_grad():
        head.linear.weight.zero_()
        means = torch.nan_to_num(y, nan=0).sum(0) / mask.sum(0).clamp_min(1)
        head.linear.bias.copy_(torch.atanh(means.clamp(-0.95, 0.95)))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=FIT["weight_decay"])
    features = torch.from_numpy(x)
    history = []
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss = utility_loss(head(features), y, mask, tasks)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite utility loss")
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return head.eval(), history


def utility_decisions(rows, predictions):
    if predictions.shape != (len(rows), len(UTILITY_OUTPUTS)):
        raise ValueError("Utility prediction coverage differs")
    return [choose_utility({name: float(values[UTILITY_OUTPUTS.index((r["benchmark"], name))])
                            for name in base.TASK_CONFIGURATIONS[r["benchmark"]] if name != ANCHORS[r["benchmark"]]},
                           r["benchmark"], MARGIN) for r, values in zip(rows, predictions)]


def summarize(rows, choices, truth, originals):
    assert_train(rows)
    report = base.replay(rows, choices, truth, originals)
    for task, metric in report["benchmarks"].items():
        selected = [(r, name) for r, name in zip(rows, choices) if r["benchmark"] == task]
        metric["rescued_vs_anchor"] = sum(truth[r["id"], name]["score"] > truth[r["id"], ANCHORS[task]]["score"]
                                          for r, name in selected)
        metric["harmed_vs_anchor"] = sum(truth[r["id"], name]["score"] < truth[r["id"], ANCHORS[task]]["score"]
                                         for r, name in selected)
    return report


def eligibility(anchor, candidate, fold_deltas, task):
    """Predeclared practical filter, not a significance test or deployment approval."""
    net = candidate["correct"] - anchor["correct"]
    reasons = []
    if net < 2:
        reasons.append("fewer_than_two_net_rescues")
    if sum(d > 0 for d in fold_deltas) < 2:
        reasons.append("gain_not_seen_in_two_folds")
    if min(fold_deltas) < -1:
        reasons.append("a_fold_loses_more_than_one_answer")
    if task == "ambrosia":
        for key in ("f1", "recall"):
            if candidate["ambiguity_metrics"][key] < anchor["ambiguity_metrics"][key]:
                reasons.append(f"ambiguity_{key}_degraded")
    return {"eligible_for_next_validation": not reasons, "net_correct": net,
            "fold_net_correct": fold_deltas, "reasons": reasons}


def run(output, encoder_folder):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory; results are immutable")
    frozen, rows, truth, originals = load_train()
    source_model = data.read(data.ROOT / base.DEFAULT_OUTPUT / "model.json")
    for path, sha in source_model["training"]["source_files_sha256"].items():
        if data.digest(data.ROOT / path) != sha:
            raise ValueError("The control model's source code changed")
    folds = base.grouped_folds(rows)
    spec = {"version": "anchor-utility-experiment-v1", "variants": list(VARIANTS),
            "seed": SEED, "epochs": EPOCHS, "optimizer": "AdamW", "lr": 0.01,
            "fit": FIT, "margin": MARGIN, "ground_fields": list(GROUND_FIELDS),
            "source_sha256": {path: data.digest(data.ROOT / path) for path in SOURCE_FILES},
            "collection_manifest_sha256": frozen["sha256"],
            "data_manifest_sha256": data.digest(data.ROOT / data.DATA_OUTPUT / "manifest.json"),
            "control_model_sha256": data.digest(data.ROOT / base.DEFAULT_OUTPUT / "model.json"),
            "train_ids": [r["id"] for r in rows], "folds": folds,
            "gate": {"minimum_net_rescues": 2, "minimum_positive_folds": 2, "maximum_loss_per_fold": 1,
                     "ambrosia_f1_recall_must_not_drop": True, "statistical_significance_claim": False},
            "selection_scope": "per-task train OOF screening; no outer dev/test evaluation or deployment",
            "utility_loss": "task-balanced per-question masked MSE; target = config quality minus anchor quality",
            "encoder": "frozen local MiniLM", "live_llm_calls": 0,
            "versions": {n: importlib.metadata.version(n) for n in ("torch", "numpy", "transformers")}}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "experiment.json", spec)  # Before computing or scoring any new candidate.
    encoder = FrozenTextEncoder(encoder_folder)
    if encoder.info != source_model["encoder"]:
        raise ValueError("Encoder changed")
    embeddings = encoder.encode([r["inputs"] for r in rows])
    ground = grounded_matrix(rows, originals, frozen)
    write_json(output / "grounding_inputs.json", {"fields": list(GROUND_FIELDS),
               "vectors": {r["id"]: v.tolist() for r, v in zip(rows, ground)}})
    positions = {r["id"]: i for i, r in enumerate(rows)}
    anchor_choices = [ANCHORS[r["benchmark"]] for r in rows]
    anchor_report = summarize(rows, anchor_choices, truth, originals)
    report = {"version": spec["version"], "scope": spec["selection_scope"], "live_llm_calls": 0,
              "dev_evaluated": False, "test_used": False,
              "controls": {"task_anchor": anchor_report,
                           "all_false": summarize(rows, ["baseline"] * len(rows), truth, originals)}, "variants": {}}
    all_predictions, fold_models = {}, []
    for variant in VARIANTS:
        choices_by_id, fold_reports, deltas = {}, [], {b: [] for b in base.TASK_CONFIGURATIONS}
        for fold in folds:
            ti, vi = ([positions[key] for key in fold[name]] for name in ("train_ids", "validation_ids"))
            fit_rows, val_rows = [rows[i] for i in ti], [rows[i] for i in vi]
            if {r["group_id"] for r in fit_rows} & {r["group_id"] for r in val_rows}:
                raise ValueError("Grouped fold leaked")
            scaler = SchemaScaler.fit([r["inputs"] for r in fit_rows])
            x = base.matrix(embeddings[ti], fit_rows, scaler)
            vx = base.matrix(embeddings[vi], val_rows, scaler)
            extra_scaler = None
            if variant["grounded"]:
                extra_x, extra_vx, extra_scaler = scale_grounding(ground[ti], ground[vi])
                x, vx = np.concatenate((x, extra_x), axis=1), np.concatenate((vx, extra_vx), axis=1)
            if variant["objective"] == "quality":
                head, _ = base.fit(x, fit_rows, FIT, epochs=EPOCHS, seed=SEED)
                choices = base.decisions(val_rows, base.probabilities(head, vx), MARGIN)
            else:
                head, _ = fit_utility(x, fit_rows)
                with torch.inference_mode():
                    predictions = head(torch.from_numpy(vx)).numpy()
                choices = utility_decisions(val_rows, predictions)
            for row, name in zip(val_rows, choices):
                if row["id"] in choices_by_id:
                    raise ValueError("Repeated validation prediction")
                choices_by_id[row["id"]] = name
            observed = summarize(val_rows, choices, truth, originals)
            control = summarize(val_rows, [ANCHORS[r["benchmark"]] for r in val_rows], truth, originals)
            for task in deltas:
                deltas[task].append(observed["benchmarks"][task]["correct"] - control["benchmarks"][task]["correct"])
            fold_reports.append({"fold": fold["fold"], "observed": observed, "anchor": control})
            name = f"{variant['name']}__fold{fold['fold']}"
            np.savez(output / f"{name}.npz", weight=head.linear.weight.detach().numpy(), bias=head.linear.bias.detach().numpy())
            fold_models.append({"name": name, "objective": variant["objective"],
                                "input_size": x.shape[1], "schema_scaler": scaler.as_dict(),
                                "ground_scaler": extra_scaler, "head_sha256": data.digest(output / f"{name}.npz")})
        if set(choices_by_id) != set(positions):
            raise ValueError("Incomplete OOF coverage")
        overall = summarize(rows, [choices_by_id[r["id"]] for r in rows], truth, originals)
        gates = {task: eligibility(anchor_report["benchmarks"][task], overall["benchmarks"][task], deltas[task], task)
                 for task in base.TASK_CONFIGURATIONS}
        report["variants"][variant["name"]] = {"oof": overall, "folds": fold_reports, "gates": gates}
        all_predictions[variant["name"]] = choices_by_id
        print("Finished", variant["name"], {b: v["correct"] for b, v in overall["benchmarks"].items()}, flush=True)
    nominations = {}
    for task in base.TASK_CONFIGURATIONS:
        eligible = [v["name"] for v in VARIANTS if report["variants"][v["name"]]["gates"][task]["eligible_for_next_validation"]]
        nominations[task] = (max(eligible, key=lambda n: report["variants"][n]["oof"]["benchmarks"][task]["correct"])
                             if eligible else "task_anchor")
    report["nominations"] = nominations
    report["adopt_as_default"] = False  # CV screening alone cannot establish a deployment win.
    write_json(output / "report.json", report)
    write_json(output / "oof_choices.json", all_predictions)
    write_json(output / "fold_models.json", fold_models)
    lines = ["# Anchor utility: fixed train-only experiment", "",
             "| Method | BIRD OOF | Ambrosia OOF | BIRD gate | Ambrosia gate |",
             "| --- | ---: | ---: | --- | --- |"]
    for method, metric in report["controls"].items():
        b, a = (metric["benchmarks"][t] for t in ("bird", "ambrosia"))
        lines.append(f"| {method} | {b['correct']}/{b['questions']} | {a['correct']}/{a['questions']} | control | control |")
    for method, values in report["variants"].items():
        b, a = (values["oof"]["benchmarks"][t] for t in ("bird", "ambrosia"))
        bg, ag = (values["gates"][t]["eligible_for_next_validation"] for t in ("bird", "ambrosia"))
        lines.append(f"| {method} | {b['correct']}/{b['questions']} | {a['correct']}/{a['questions']} | {bg} | {ag} |")
    lines += ["", f"Next validation nominations: {nominations}", "",
              "No outer-dev/test evaluation, paid requests, synthetic questions or source-label edits.",
              "The gate is a predeclared practical screening rule, not statistical significance.",
              "Fold heads are research artifacts, not a full-train production model."]
    (output / "comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8", newline="\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--encoder", default=".cache/classifier/minilm")
    args = parser.parse_args()
    output, encoder = ((data.ROOT / v).resolve() for v in (args.out, args.encoder))
    if (not output.is_relative_to(data.ROOT / "research_results") or output == data.ROOT / "research_results"
            or not encoder.is_relative_to(data.ROOT / ".cache/classifier")):
        parser.error("Use a research_results subfolder and the local encoder cache")
    torch.set_num_threads(2)
    report = run(output, encoder)
    print("Nominations:", report["nominations"], "paid calls: 0; no dev/test evaluation")


if __name__ == "__main__":
    main()
