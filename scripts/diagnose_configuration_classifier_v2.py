"""Post-hoc frozen model diagnosis and train-only input ablations. No API/DB calls."""

import argparse
from collections import defaultdict
import json
import sys

import numpy as np
import torch

import train_configuration_classifier_v2 as training
import prepare_classifier_v2 as data
from durable_prompt_store import write_json

sys.path.insert(0, str(training.ROOT / "src"))
from adv_text2sql.mcp_servers.text2sql_tool.src.configuration_classifier_v2 import ANCHORS, OUTPUTS  # noqa: E402
from adv_text2sql.mcp_servers.text2sql_tool.src.multilabel_classifier import (  # noqa: E402
    FrozenTextEncoder, SCHEMA_FIELDS, SchemaScaler,
)

DEFAULT_OUTPUT = "research_results/classifier_v2_diagnosis_20261109"
MODES = ("text_schema", "text_only", "schema_only", "intercept_only")


def feature_view(x, mode, text_size=384):
    if mode not in MODES or x.ndim != 2 or x.shape[1] != text_size + 6:
        raise ValueError("Unknown ablation or feature shape")
    result = x.copy()
    if mode in ("schema_only", "intercept_only"):
        result[:, :text_size] = 0
    if mode in ("text_only", "intercept_only"):
        result[:, text_size:] = 0
    return result


def gap_parts(weight, bias, x, chosen, anchor, text_size=384):
    """Exact linear logit-gap decomposition, not a probability or causal attribution."""
    delta = weight[chosen] - weight[anchor]
    return {"text_logit_gap": float(delta[:text_size] @ x[:text_size]),
            "schema_logit_gap": float(delta[text_size:] @ x[text_size:]),
            "bias_logit_gap": float(bias[chosen] - bias[anchor]),
            "total_logit_gap": float(delta @ x + bias[chosen] - bias[anchor])}


def schema_shift(train_rows, dev_rows):
    report = {}
    for task in training.TASK_CONFIGURATIONS:
        fitting = [r for r in train_rows if r["benchmark"] == task]
        development = [r for r in dev_rows if r["benchmark"] == task]
        fields = {}
        for field in SCHEMA_FIELDS:
            values = [r["inputs"]["schema_stats"][field] for r in fitting]
            low, high = min(values), max(values)
            fields[field] = {"train_min": low, "train_max": high,
                             "dev_min": min(r["inputs"]["schema_stats"][field] for r in development),
                             "dev_max": max(r["inputs"]["schema_stats"][field] for r in development),
                             "dev_outside_train_count": sum(not low <= r["inputs"]["schema_stats"][field] <= high
                                                            for r in development)}
        report[task] = fields
    return report


def task_summary(rows, choices, truth, originals):
    out = training.replay(rows, choices, truth, originals)
    for task, metrics in out["benchmarks"].items():
        pairs = [(r, name) for r, name in zip(rows, choices) if r["benchmark"] == task]
        metrics["rescued_vs_anchor"] = sum(truth[r["id"], name]["score"] > truth[r["id"], ANCHORS[task]]["score"]
                                           for r, name in pairs)
        metrics["harmed_vs_anchor"] = sum(truth[r["id"], name]["score"] < truth[r["id"], ANCHORS[task]]["score"]
                                          for r, name in pairs)
    return out


def ablate_train(rows, embeddings, model, truth, originals):
    """Fixed diagnostics, not a new hyperparameter search or outer-dev selection."""
    if any(r["split"] != "train" for r in rows):
        raise ValueError("Ablation accepts train only")
    folds = training.grouped_folds(rows)
    positions = {r["id"]: i for i, r in enumerate(rows)}
    result = {}
    for mode in MODES:
        oof = np.full((len(rows), len(OUTPUTS)), np.nan, dtype=np.float32)
        per_fold = []
        for fold in folds:
            ti, vi = ([positions[key] for key in fold[name]] for name in ("train_ids", "validation_ids"))
            fitting, validation = [rows[i] for i in ti], [rows[i] for i in vi]
            scaler = SchemaScaler.fit([r["inputs"] for r in fitting])
            x = feature_view(training.matrix(embeddings[ti], fitting, scaler), mode)
            vx = feature_view(training.matrix(embeddings[vi], validation, scaler), mode)
            head, _ = training.fit(x, fitting, model["selected"], epochs=model["training"]["epochs"])
            scores = training.probabilities(head, vx)
            oof[vi] = scores
            choices = training.decisions(validation, scores, model["selected"]["margin"])
            per_fold.append({"fold": fold["fold"], "validation_ids": fold["validation_ids"],
                             "replay": task_summary(validation, choices, truth, originals)})
        if not np.isfinite(oof).all():
            raise ValueError("Incomplete OOF diagnostic")
        choices = training.decisions(rows, oof, model["selected"]["margin"])
        result[mode] = {"oof": task_summary(rows, choices, truth, originals), "folds": per_fold,
                        "choices": {r["id"]: name for r, name in zip(rows, choices)}}
        print("Train input ablation:", mode, flush=True)
    return result


def repeat_consistency(frozen, originals):
    """Audit same candidate/mask/question across compatible saved records.

    This measures observed archive disagreements, not independent provider trial rates.
    No best-of-repeat selection or edits to the training dataset are performed.
    """
    summary = data.read(data.ROOT / data.STAGE_SUMMARY)
    manifests = {b: data.read(data.local_path(data.ROOT, p["source_manifest_path"]))
                 for b, p in summary["profiles"].items()}
    points, evidence = defaultdict(list), {}
    for source in data.SOURCES:
        folder = data.ROOT / "research_results" / source
        m = data.read(folder / "experiment.json")
        if m["sha256"] != data.fingerprint({k: v for k, v in m.items() if k != "sha256"}):
            raise ValueError("Archive manifest changed")
        target = manifests["bird"]
        if (m["model"] != target["model"] or m["runtime_environment"] != target["runtime_environment"]
                or any(m["files_sha256"].get(n) != target["files_sha256"].get(n) for n in data.CORE_FILES)):
            continue
        for path in sorted(folder.glob("jobs/*/evaluations.json")):
            records = data.read(path)
            for key, value in records.items():
                parts = key.split(":", 5)
                if len(parts) != 6:
                    raise ValueError("Unknown checkpoint key")
                phase, cid, profile, original, task, qid = parts
                identity = f"{task}:{qid}"
                if identity not in originals or phase not in ("search", "confirm") or profile != "hybrid" or original != "False":
                    continue
                if cid != data.fingerprint(frozen["candidates"][task]):
                    continue
                features = tuple(m.get("feature_mask", [14, 32]))
                name = next((n for n in training.TASK_CONFIGURATIONS[task] if training.CONFIGURATIONS[n] == features), None)
                meta_path = path.parent / phase / "metadata.json"
                if name is None or not meta_path.exists():
                    continue
                meta = data.read(meta_path)
                if meta.get("status") != "completed" or meta.get("manifest_sha256") != m["sha256"]:
                    continue
                if m["schema_audit"]["schemas"][task] != manifests[task]["schema_audit"]["schemas"][task]:
                    raise ValueError("Archive schema changed")
                slot = "sql" if task == "bird" else "ambiguity"
                items = (m["cohorts"][slot]["fit"] + m["cohorts"][slot]["inner_val"] if phase == "search"
                         else m.get("full", m["dev"])[task])
                item = next((i for i in items if str(i["question_id"]) == qid), None)
                if item is None or any(item.get(k, "") != originals[identity]["item"].get(k, "")
                                       for k in ("question", "db_id", "evidence", "SQL")):
                    raise ValueError("Archive input/reference differs")
                score = data.validate_observation(value, item, task)
                if score is None:
                    continue
                points[identity, name].append({"source": path.relative_to(data.ROOT).as_posix(), "key": key, "score": score})
                for p in (path, meta_path, folder / "experiment.json"):
                    evidence[p.relative_to(data.ROOT).as_posix()] = data.digest(p)
    # The new collection never repeats cached question/configuration points.
    by_task = {}
    for task in training.TASK_CONFIGURATIONS:
        repeated = {k: v for k, v in points.items() if k[0].startswith(task + ":") and len(v) > 1}
        by_task[task] = {"repeated_points": len(repeated),
                         "points_with_score_disagreement": sum(len({v["score"] for v in rs}) > 1 for rs in repeated.values()),
                         "records": sum(len(v) for v in repeated.values())}
    return {"scope": "same compatible archived candidate/mask; not established independent trials",
            "benchmarks": by_task, "repeated_points": [
                {"id": key[0], "configuration": key[1], "records": values}
                for key, values in sorted(points.items()) if len(values) > 1], "source_sha256": evidence}


def run(output, model_folder, encoder_folder):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new diagnosis directory")
    _, frozen, train, dev, truth, originals = training.load_dataset()
    info = data.read(model_folder / "model.json")
    for name, expected in info["training"]["source_files_sha256"].items():
        if data.digest(data.ROOT / name) != expected:
            raise ValueError("Frozen model code changed")
    if data.digest(model_folder / "head.npz") != info["head_sha256"]:
        raise ValueError("Frozen model head changed")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "experiment.json", {"model_sha256": data.digest(model_folder / "model.json"),
        "data_sha256": data.digest(data.ROOT / data.DATA_OUTPUT / "manifest.json"),
        "source_sha256": data.digest(__file__), "modes": list(MODES), "fit_selection": info["selected"],
        "scope": "post-hoc diagnosis; fixed input ablations within train; no deployment selection or new dev tuning",
        "paid_calls": 0, "test_used": False})
    encoder = FrozenTextEncoder(encoder_folder)
    if encoder.info != info["encoder"]:
        raise ValueError("Diagnostic encoder differs")
    emb = encoder.encode([r["inputs"] for r in train])
    dev_emb = encoder.encode([r["inputs"] for r in dev])
    scaler = SchemaScaler(info["schema_scaler"]["mean"], info["schema_scaler"]["scale"])
    dx = training.matrix(dev_emb, dev, scaler)
    with np.load(model_folder / "head.npz", allow_pickle=False) as saved:
        w, bias = saved["weight"].copy(), saved["bias"].copy()
    probabilities = 1 / (1 + np.exp(-(dx @ w.T + bias)))
    choices = training.decisions(dev, probabilities, info["selected"]["margin"])
    saved_predictions = {p["id"]: p for p in data.read(model_folder / "predictions_train_dev.json")}
    if any(saved_predictions[r["id"]]["configuration"] != name for r, name in zip(dev, choices)):
        raise ValueError("Frozen model predictions do not reproduce")
    switches, per_db = [], defaultdict(list)
    for i, (r, name) in enumerate(zip(dev, choices)):
        task, anchor = r["benchmark"], ANCHORS[r["benchmark"]]
        per_db[task, originals[r["id"]]["db_id"]].append((r, name))
        if name == anchor:
            continue
        parts = gap_parts(w, bias, dx[i], OUTPUTS.index((task, name)), OUTPUTS.index((task, anchor)))
        contributions = {field: float((w[OUTPUTS.index((task, name)), 384+j]
                                      - w[OUTPUTS.index((task, anchor)), 384+j]) * dx[i, 384+j])
                         for j, field in enumerate(SCHEMA_FIELDS)}
        same_task = [j for j, tr in enumerate(train) if tr["benchmark"] == task]
        nearest = sorted(same_task, key=lambda j: -float(emb[j] @ dev_emb[i]))[:3]
        switches.append({"id": r["id"], "benchmark": task, "db_id": originals[r["id"]]["db_id"],
                         "question": r["inputs"]["question"], "evidence": r["inputs"]["evidence"],
                         "chosen": name, "anchor": anchor, "chosen_score": truth[r["id"], name]["score"],
                         "anchor_score": truth[r["id"], anchor]["score"], **parts,
                         "schema_field_logit_gaps": contributions,
                         "quality_scores": saved_predictions[r["id"]]["quality_scores"],
                         "nearest_train": [{"id": train[j]["id"], "db_id": originals[train[j]["id"]]["db_id"],
                                            "question": train[j]["inputs"]["question"],
                                            "cosine": float(emb[j] @ dev_emb[i])} for j in nearest],
                         "chosen_saved_sql": truth[r["id"], name]["prediction"],
                         "anchor_saved_sql": truth[r["id"], anchor]["prediction"]})
    db_report = []
    for (task, db), pairs in sorted(per_db.items()):
        rs, selected = zip(*pairs)
        metric = task_summary(list(rs), list(selected), truth, originals)["benchmarks"][task]
        db_report.append({"benchmark": task, "db_id": db, **metric,
                          "all_configuration_correct": {n: sum(truth[r["id"], n]["score"] for r in rs)
                                                        for n in training.TASK_CONFIGURATIONS[task]}})
    ablations = ablate_train(train, emb, info, truth, originals)
    # No ablation variant is evaluated on outer dev or adopted as a new model.
    repeats = repeat_consistency(frozen, originals)
    report = {"scope": "post-hoc diagnostics, not an independent new evaluation", "paid_calls": 0, "test_used": False,
              "schema_shift": schema_shift(train, dev), "dev_switches": switches, "dev_databases": db_report,
              "train_input_ablations": ablations, "compatible_archive_repeats": repeats["benchmarks"]}
    write_json(output / "report.json", report)
    write_json(output / "archive_repeat_audit.json", repeats)
    lines = ["# Configuration v2: diagnostic input ablations", "",
             "Fixed selected hyperparameters; 4 train-group folds; no new dev-model selection.", "",
             "| Input | BIRD OOF correct | Ambrosia OOF correct | Macro accuracy |",
             "| --- | ---: | ---: | ---: |"]
    for mode, v in ablations.items():
        b, a = (v["oof"]["benchmarks"][k] for k in ("bird", "ambrosia"))
        lines.append(f"| {mode} | {b['correct']}/{b['questions']} | {a['correct']}/{a['questions']} | {v['oof']['macro_accuracy']:.2f}% |")
    lines += ["", "These are post-hoc diagnostic comparisons, not evidence for a new deployed winner.",
              "Original model, dataset, checkpoints and generator are unchanged. No paid calls or test evaluation."]
    (output / "comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8", newline="\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=training.DEFAULT_OUTPUT)
    parser.add_argument("--encoder", default=".cache/classifier/minilm")
    args = parser.parse_args()
    output, model, encoder = ((training.ROOT / v).resolve() for v in (args.out, args.model, args.encoder))
    if (not output.is_relative_to(training.ROOT / "research_results") or output == training.ROOT / "research_results"
            or not model.is_relative_to(training.ROOT / "research_results")
            or not encoder.is_relative_to(training.ROOT / ".cache/classifier")):
        parser.error("Use workspace research results and local encoder cache")
    torch.set_num_threads(2)
    report = run(output, model, encoder)
    print(json.dumps({"archive_repeats": report["compatible_archive_repeats"],
                      "dev_switches": len(report["dev_switches"])}, indent=2))


if __name__ == "__main__":
    main()
