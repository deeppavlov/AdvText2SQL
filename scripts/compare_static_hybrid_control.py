"""Offline comparison against a fresh control; preserves the frozen search protocol."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from atomic_artifacts import write_json
import static_hybrid_search as search


def taxonomy_messages(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if any(m["content"] == "Classify the query." for m in r["messages"])]


def prompt_comparison(previous, current):
    def signature(row):
        return hashlib.sha256(json.dumps(row["messages"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def verdict(row):
        match = re.search(r'"is_ambiguous"\s*:\s*(true|false)', row["raw_response"], re.I)
        if not match:
            raise ValueError("Unknown taxonomy response format; inspect it rather than assuming a verdict")
        return match.group(1).lower() == "true"

    index = defaultdict(list)
    for row in previous:
        index[signature(row)].append(row)
    matched = [(r, index[signature(r)]) for r in current if signature(r) in index]
    return {"current_calls": len(current), "identical_messages": len(matched),
            "changed_verdicts": [{"question_id": r["question_id"],
                                  "previous_verdicts": [verdict(p) for p in matches], "current_verdict": verdict(r)}
                                 for r, matches in matched if verdict(r) not in [verdict(p) for p in matches]]}


def compare(output, scope="train", control_name="hybrid_control_repeat"):
    manifest = search.read(output / "experiment.json")
    search.validate_manifest(manifest)
    cases = {c["name"]: c for c in manifest["cases"]}
    if control_name not in cases:
        raise ValueError("Unknown control name")
    controls = {b: search.result(output, manifest, scope, cases[control_name], b) for b in search.BENCHMARKS}
    if any(r["status"] != "completed" for r in controls.values()):
        raise ValueError("Both control benchmarks must be completed")
    results = []
    sources = [Path(__file__).resolve(), output / "experiment.json"]
    for case in cases.values():
        values = {b: search.result(output, manifest, scope, case, b) for b in search.BENCHMARKS}
        if any(r["status"] != "completed" for r in values.values()):
            continue
        item = {"case": case["name"], "features": case["features"], "benchmarks": {}}
        for b, value in values.items():
            control = controls[b]
            paired = {
                "rescued": [q for q, v in value["outcomes"].items() if v and not control["outcomes"][q]],
                "harmed": [q for q, v in value["outcomes"].items() if not v and control["outcomes"][q]],
            }
            item["benchmarks"][b] = {**{k: v for k, v in value.items() if k != "outcomes"},
                                      "delta_current_control_pp": value["accuracy"] - control["accuracy"],
                                      "paired_current_control": paired}
            folder = search.run_folder(output, scope, case["name"], b)
            sources.extend(folder / f for f in ("metadata.json", "report.json", "query_results.json", "raw_responses.jsonl"))
        item["macro_accuracy"] = sum(v["accuracy"] for v in values.values()) / 2
        results.append(item)
    results.sort(key=lambda r: (-r["macro_accuracy"], len(r["features"]), r["case"]))
    old_path = search.ROOT / "research_results/deepseek_combinations_large_20261026/sql_ambiguity/ambrosia_227/raw_responses.jsonl"
    control_path = search.run_folder(output, scope, control_name, "ambrosia") / "raw_responses.jsonl"
    current = taxonomy_messages(control_path)
    comparison = {"historical_H_vs_fresh_H": prompt_comparison(taxonomy_messages(old_path), current)}
    sources.append(old_path)
    for case in results:
        if case["case"] == control_name:
            continue
        path = search.run_folder(output, scope, case["case"], "ambrosia") / "raw_responses.jsonl"
        comparison[case["case"] + "_vs_fresh_H"] = prompt_comparison(taxonomy_messages(path), current)
    report = {"scope": scope, "current_control": control_name, "results": results,
              "taxonomy_message_comparisons": comparison,
              "limitations": ["Historical and fresh H are not independent replicated trials under guaranteed identical server conditions.",
                              "One fresh run per candidate cannot establish statistically reliable ranking.",
                              "The frozen suite summary still uses historical H; this report uses the explicit fresh control."],
              "source_sha256": {p.relative_to(search.ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    lines = ["# Comparison with fresh control", "", f"Scope: {scope}; control: {control_name}.",
             "Historical H remains unchanged in the frozen suite summary; this table uses the fresh control.", "",
             "| Case | Flags | BIRD correct | BIRD accuracy | Delta fresh H pp | Ambrosia correct | Ambrosia accuracy | Delta fresh H pp | Ambrosia F1 | Mean accuracy |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in results:
        b, a = r["benchmarks"]["bird"], r["benchmarks"]["ambrosia"]
        lines.append(f"| {r['case']} | {','.join(map(str,r['features']))} | {b['correct']}/{b['questions']} | {b['accuracy']:.2f} | "
                     f"{b['delta_current_control_pp']:+.2f} | {a['correct']}/{a['questions']} | {a['accuracy']:.2f} | "
                     f"{a['delta_current_control_pp']:+.2f} | {100*a['ambiguity_metrics']['f1']:.2f} | {r['macro_accuracy']:.2f} |")
    lines += ["", "## Identical taxonomy messages and changed verdicts", "",
              "| Comparison | Identical / current messages | Changed verdicts |", "| --- | ---: | ---: |"]
    for name, value in comparison.items():
        lines.append(f"| {name} | {value['identical_messages']}/{value['current_calls']} | {len(value['changed_verdicts'])} |")
    lines += ["", "No causal attribution to FEAT_14: it changes the SQL prompt, not the taxonomy prompt.",
              "One trial per new mask is insufficient to establish a stable optimum. Question-level paired changes and source hashes are in the JSON report.", ""]
    write_json(output / f"current_control_comparison_{scope}.json", report)
    (output / f"current_control_comparison_{scope}.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", default=search.DEFAULT_OUTPUT)
    parser.add_argument("--scope", choices=search.SCOPES, default="train")
    args = parser.parse_args()
    output = (search.ROOT / args.search).resolve()
    if not output.is_relative_to(search.ROOT / "research_results"):
        parser.error("Search must be inside research_results")
    report = compare(output, args.scope)
    print(f"Offline comparison saved: {len(report['results'])} completed configurations; no API/DB calls.")


if __name__ == "__main__":
    main()
