"""Freeze rules on dev, then replay held-out saved outcomes. No API/DB calls."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adv_text2sql.mcp_servers.text2sql_tool.src.feature_selector import (  # noqa: E402
    CONFIGURATIONS, POLICIES, RULE_VERSION, RuleBasedFeatureSelector,
)
from build_configuration_dataset import DEFAULT_OUTPUT as DATASET, read, sha, verify_bundle  # noqa: E402
from research_metrics import ambrosia_report  # noqa: E402

DEFAULT_OUTPUT = "research_results/rule_selector_deepseek_20261028"
SOURCE_FILES = (
    "src/adv_text2sql/mcp_servers/text2sql_tool/src/feature_selector.py",
    "src/adv_text2sql/mcp_servers/text2sql_tool/src/selective_generator.py",
    "scripts/evaluate_rule_selector.py",
    "scripts/run_openrouter_baseline.py",
    "src/adv_text2sql/mcp_servers/text2sql_tool/src/text2sql_implementation.py",
    "src/adv_text2sql/mcp_servers/text2sql_tool/src/prompts.py",
)


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def frozen_write(path, text):
    """Repeated evaluation is allowed only with byte-identical artifacts."""
    if path.exists():
        if path.read_bytes() != text.encode("utf-8"):
            raise ValueError(f"Frozen artifact differs: {path.name}; use a new output directory")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def evaluate_policy(rows, policy, registry):
    """Only the allowlisted inputs reach the selector. Outcomes are read afterwards."""
    selector = RuleBasedFeatureSelector(policy)
    groups = {}
    decisions = []
    for row in rows:
        selection = selector.select_inputs(row["inputs"])
        outcome = row["outcomes"][selection.configuration]
        if any(not row["outcomes"][c]["observed"] or type(row["outcomes"][c]["correct"]) is not bool
               for c in CONFIGURATIONS):
            raise ValueError("Replay requires observed outcomes for both candidates")
        decisions.append({"id": row["id"], "selection": selection.as_dict()})
        groups.setdefault(row["benchmark"], []).append((row, selection, outcome))
    for benchmark, entries in groups.items():
        total = len(entries)
        choices = Counter(s.configuration for _, s, _ in entries)
        correct = sum(o["correct"] for _, _, o in entries)
        estimated_tokens = sum(
            registry[name]["benchmarks"][benchmark]["recorded_run_cost"]["mean_tokens_per_question"] * count
            for name, count in choices.items())
        estimated_calls = sum(
            registry[name]["benchmarks"][benchmark]["recorded_run_cost"]["recorded_calls"]
            / registry[name]["benchmarks"][benchmark]["questions"] * count
            for name, count in choices.items())
        result = {
            "questions": total, "correct": correct, "accuracy": 100 * correct / total,
            "choices": {c: choices[c] for c in CONFIGURATIONS},
            "oracle_correct": sum(any(r["outcomes"][c]["correct"] for c in CONFIGURATIONS) for r, _, _ in entries),
            "generation_errors": sum(bool(o["generation_error"]) for _, _, o in entries),
            "execution_errors": sum(bool(o["execution_error"]) for _, _, o in entries),
            "estimated_tokens_from_run_means": estimated_tokens,
            "estimated_calls_from_run_means": estimated_calls,
            "exact_selected_cost_available": False,
            "vs_static": {},
        }
        for configuration in CONFIGURATIONS:
            gains = sum(o["correct"] and not r["outcomes"][configuration]["correct"] for r, _, o in entries)
            losses = sum(not o["correct"] and r["outcomes"][configuration]["correct"] for r, _, o in entries)
            result["vs_static"][configuration] = {"rescued": gains, "lost": losses, "net_correct": gains - losses}
        if benchmark == "ambrosia":
            items = [{"question_id": r["id"], "SQL": "ambiguous" if r["analysis"]["gold_ambiguous"] else "SELECT"}
                     for r, _, _ in entries]
            predictions = {r["id"]: "error" if o["generation_error"] else
                           "ambiguous" if o["predicted_ambiguous"] else "SELECT" for r, _, o in entries}
            strict = ambrosia_report(items, predictions)
            if strict["accuracy"] != result["accuracy"]:
                raise ValueError("Ambrosia outcomes disagree with strict class scoring")
            result["ambiguity_metrics"] = strict["ambiguity_metrics"]
        groups[benchmark] = result
    if not groups:
        raise ValueError("No rows to evaluate")
    return {
        "policy": policy, "benchmarks": groups,
        "macro_accuracy": sum(v["accuracy"] for v in groups.values()) / len(groups),
        "estimated_total_tokens_from_run_means": sum(v["estimated_tokens_from_run_means"] for v in groups.values()),
    }, decisions


def choose_policy(train, dev, registry):
    """Fixed rules; choose on dev only. No fitting, ID memorization, or test argument."""
    if {r["split"] for r in train} != {"train"} or {r["split"] for r in dev} != {"dev"}:
        raise ValueError("Policy selection accepts only train and dev rows")
    candidates = []
    for policy in POLICIES:
        training, _ = evaluate_policy(train, policy, registry)
        development, _ = evaluate_policy(dev, policy, registry)
        candidates.append({"policy": policy, "train": training, "dev": development})
    best = min(candidates, key=lambda c: (
        -c["dev"]["macro_accuracy"], c["dev"]["estimated_total_tokens_from_run_means"], POLICIES.index(c["policy"])))
    return best["policy"], candidates


def load_dataset(dataset, root=ROOT):
    manifest = verify_bundle(dataset, root)
    for name, features in CONFIGURATIONS.items():
        if manifest["configurations"][name]["features"] != list(features):
            raise ValueError("Dataset candidate flags differ from selector candidates")
    rows = [json.loads(line) for line in (dataset / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    partitions = read(dataset / "splits.json")["partitions"]
    for name, ids in partitions.items():
        if set(ids) != {r["id"] for r in rows if r["split"] == name}:
            raise ValueError("Row split differs from the frozen split manifest")
    return manifest, rows


def prepare(dataset, output, root=ROOT):
    manifest, rows = load_dataset(dataset, root)
    registry = manifest["configurations"]
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    selected, candidates = choose_policy(train, dev, registry)
    sources = {file: sha(root / file) for file in SOURCE_FILES}
    selection = {
        "format_version": "rule-selector-selection-v1", "rule_version": RULE_VERSION,
        "selected_policy": selected, "candidate_policies": list(POLICIES),
        "configurations": {c: list(v) for c, v in CONFIGURATIONS.items()},
        "selector_source_sha256": sources[SOURCE_FILES[0]], "source_files_sha256": sources,
        "dataset_manifest_sha256": sha(dataset / "manifest.json"),
        "model": manifest["model"],
        "selection_objective": "maximize dev macro accuracy across BIRD/Ambrosia; tie: lower run-mean token proxy, then catalog order",
        "train_ids": sorted(r["id"] for r in train), "dev_ids": sorted(r["id"] for r in dev),
        "test_used_for_selection": False, "retrospective_test": True,
        "exact_selected_cost_available": False,
    }
    # Freeze before any held-out report can be produced.
    frozen_write(output / "selection.json", json_text(selection))
    frozen_write(output / "development.json", json_text(candidates))
    folds = []
    for fold in read(dataset / "cv_folds.json"):
        ids = set(fold["validation_ids"])
        validation = [r for r in train if r["id"] in ids]
        if {r["id"] for r in validation} != ids:
            raise ValueError("CV includes non-training IDs")
        folds.append({"fold": fold["fold"], "evaluation": evaluate_policy(validation, selected, registry)[0]})
    frozen_write(output / "train_cv.json", json_text({"purpose": "diagnostic only; fixed rules, no fitting", "folds": folds}))
    return selection


def report_markdown(report):
    lines = ["# Phase 1 — офлайн-проверка селектора", "",
             f"Выбранная по dev политика: `{report['selected_policy']}`. Новых LLM/SQL-вызовов: **0**.", "",
             "| Часть | Бенчмарк | Подход | Верно / всего | Accuracy | F1 неоднозначности | Hybrid-вызовов* | Токены, оценка* |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for split, methods in report["splits"].items():
        for method, evaluation in methods.items():
            for benchmark, metrics in evaluation["benchmarks"].items():
                f1 = metrics.get("ambiguity_metrics", {}).get("f1")
                f1_text = f"{100 * f1:.2f}%" if f1 is not None else "—"
                lines.append(f"| {split} | {benchmark} | {method} | {metrics['correct']}/{metrics['questions']} | "
                             f"{metrics['accuracy']:.2f}% | {f1_text} | {metrics['choices']['hybrid']} | "
                             f"{metrics['estimated_tokens_from_run_means']:,.0f} |")
    lines += ["", "*Hybrid-вызовов — число вопросов, направленных в конфигурацию, а не число API-запросов.",
              "Токены — приближение по средним расходам полных исходных прогонов; точные затраты по вопросам неизвестны.",
              "В исходных средних присутствуют probe/build. Экономия должна быть подтверждена отдельным live-прогоном.", "",
              "Выбор заморожен до оценки test. Test ретроспективный: статические конфигурации ранее выбирались на этих же наборах.",
              "Ambrosia измеряет детекцию неоднозначности, BIRD — локальную EX-подобную правильность SQL.",
              "Оракул — только верхняя граница по сохранённым ответам, не результат селектора."]
    return "\n".join(lines) + "\n"


def evaluate(dataset, output, root=ROOT):
    manifest, rows = load_dataset(dataset, root)
    selection = read(output / "selection.json")
    selector = RuleBasedFeatureSelector.from_file(output / "selection.json")
    if selection["dataset_manifest_sha256"] != sha(dataset / "manifest.json"):
        raise ValueError("Frozen policy belongs to a different dataset")
    if any(sha(root / file) != expected for file, expected in selection["source_files_sha256"].items()):
        raise ValueError("Code changed after policy selection; create a new experiment")
    if selection["test_used_for_selection"]:
        raise ValueError("Invalid selection protocol")
    report = {
        "format_version": "rule-selector-replay-v1", "selected_policy": selector.policy,
        "selection_sha256": sha(output / "selection.json"), "dataset_manifest_sha256": sha(dataset / "manifest.json"),
        "retrospective_test": True, "live_llm_calls": 0, "splits": {},
        "cost_policy": "run-mean proxy only; exact per-question costs are unavailable",
    }
    decisions = []
    for split in ("train", "dev", "test"):
        subset = [r for r in rows if r["split"] == split]
        methods = {}
        for method, policy in (("economical", "always_economical"), ("hybrid", "always_hybrid"), ("selector", selector.policy)):
            metrics, selected = evaluate_policy(subset, policy, manifest["configurations"])
            methods[method] = metrics
            if method == "selector":
                decisions.extend({**d, "split": split} for d in selected)
        report["splits"][split] = methods
    report["possible_prompt_example_overlap"] = {
        split: {"questions": sum(r["analysis"]["question_example_overlap"] for r in rows if r["split"] == split),
                "family_rows": sum(r["analysis"]["family_example_overlap"] for r in rows if r["split"] == split)}
        for split in ("train", "dev", "test")}
    frozen_write(output / "report.json", json_text(report))
    frozen_write(output / "comparison.md", report_markdown(report))
    frozen_write(output / "decisions.jsonl", "".join(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n" for d in decisions))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "evaluate"))
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dataset, output = (ROOT / args.dataset).resolve(), (ROOT / args.out).resolve()
    if not dataset.is_relative_to(ROOT / "data"):
        parser.error("Dataset must be inside data/")
    if not output.is_relative_to(ROOT / "research_results") or output == ROOT / "research_results":
        parser.error("Use a dedicated output folder inside research_results/")
    if output.exists() and any(output.iterdir()) and not (output / "selection.json").exists():
        parser.error("Output contains unrelated files")
    if args.action == "prepare":
        artifact = prepare(dataset, output)
        print(f"FROZEN: {artifact['selected_policy']}; selection used train/dev only: {output}")
    else:
        report = evaluate(dataset, output)
        print(f"REPLAYED: {report['selected_policy']}; 0 paid calls: {output / 'comparison.md'}")
        for benchmark, metrics in report["splits"]["test"]["selector"]["benchmarks"].items():
            print(f"test {benchmark}: {metrics['correct']}/{metrics['questions']} ({metrics['accuracy']:.2f}%)")


if __name__ == "__main__":
    main()
