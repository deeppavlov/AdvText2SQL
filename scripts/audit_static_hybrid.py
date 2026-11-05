"""Offline evidence and proposed probes for static hybrid search. Never calls an API."""

import argparse
import hashlib
import json
from pathlib import Path

import build_multilabel_dataset as dataset_builder


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/multilabel_utility/deepseek_20261029"
BASE = (14, 32, 35)
FEATURES = tuple(n for n in range(1, 37) if n not in (22, 24, 31))
KNOWN_COMBINATIONS = {
    "schema_2_35": (2, 35), "sql_rules": (14, 35),
    "sql_ambiguity": BASE, "hybrid_with_relationships": (1, *BASE),
}


def paired(rows, treatment, control):
    """Missing comparisons remain missing; invalid output fails the fixed-size score."""
    observed = [r for r in rows if all(r["observations"][c]["observed"] for c in (treatment, control))]
    correct = sum(r["observations"][treatment]["correct"] is True for r in rows)
    baseline_correct = sum(r["observations"][control]["correct"] is True for r in rows)
    return {
        "questions": len(rows), "correct": correct,
        "accuracy": 100 * correct / len(rows),
        "delta_pp": 100 * (correct - baseline_correct) / len(rows),
        "observed_pairs": len(observed), "missing_pairs": len(rows) - len(observed),
        "rescued": sum(r["observations"][treatment]["correct"] and
                       not r["observations"][control]["correct"] for r in observed),
        "harmed": sum(not r["observations"][treatment]["correct"] and
                      r["observations"][control]["correct"] for r in observed),
        "generation_errors": sum(r["observations"][treatment]["generation_error"] is True for r in rows),
        "invalid_predictions": sum(r["observations"][treatment]["invalid_prediction"] is True for r in rows),
    }


def proposed_probes():
    """Explicit masks, not automatic prerequisite expansion or a paid execution queue."""
    result = []

    def add(name, stage, flags, controls):
        result.append({"name": name, "stage": stage, "features": sorted(flags),
                       "controls": controls, "status": "proposed_not_run",
                       "benchmarks": ["bird", "ambrosia"],
                       "selection_scope": "train_then_dev; old full cohorts exploratory only"})

    add("core_without_tsv", "core_factorial", (14, 32), ["feat_14", "feat_32", "sql_ambiguity"])
    add("core_without_rules", "core_factorial", (32, 35), ["feat_32", "feat_35", "sql_ambiguity"])
    for number in (2, 15, 16, 21, 23, 25, 28, 29, 30):
        add(f"hybrid_plus_{number}", "one_mechanism", (*BASE, number), ["sql_ambiguity"])
    add("statistics_compact", "statistics_bundle", (*BASE, 2, 7), ["hybrid_plus_2"])
    add("statistics_types", "statistics_bundle", (*BASE, 2, 3, 7), ["statistics_compact"])
    add("statistics_counts", "statistics_bundle", (*BASE, 2, 3, 4, 7), ["statistics_types"])
    add("sanitize_validate", "repair_bundle", (*BASE, 15, 16), ["hybrid_plus_15", "hybrid_plus_16"])
    add("validated_retry", "repair_bundle", (*BASE, 15, 16, 17), ["sanitize_validate"])
    add("validated_feedback", "repair_bundle", (*BASE, 15, 16, 17, 27), ["validated_retry"])
    add("judged_feedback", "repair_bundle", (*BASE, 15, 16, 17, 20, 27), ["validated_feedback"])
    add("validated_voting", "workflow_interaction", (*BASE, 15, 16, 28),
        ["hybrid_plus_28", "sanitize_validate"])
    add("routed_decomposition", "workflow_interaction", (*BASE, 29, 30),
        ["hybrid_plus_29", "hybrid_plus_30"])
    masks = [tuple(p["features"]) for p in result]
    known = {(), *((n,) for n in FEATURES), *KNOWN_COMBINATIONS.values()}
    if len(set(masks)) != len(masks) or set(masks) & known:
        raise ValueError("Probes must be distinct and not already measured masks")
    return result


def build_audit(rows):
    if len(rows) != 468 or len({r["id"] for r in rows}) != 468:
        raise ValueError("Expected the complete frozen 468-question cohort")
    expected = {"baseline", *(f"feat_{n}" for n in FEATURES), *KNOWN_COMBINATIONS}
    if any(set(r["observations"]) != expected for r in rows):
        raise ValueError("The source observation registry changed")
    singles = {}
    for n in FEATURES:
        singles[f"FEAT_{n}"] = {
            b: paired([r for r in rows if r["benchmark"] == b], f"feat_{n}", "baseline")
            for b in ("bird", "ambrosia")
        }
    # These are separate single-run answers, not predictions of H+feature quality.
    complement = {
        f"FEAT_{n}": {b: paired([r for r in rows if r["split"] == "train" and r["benchmark"] == b],
                                f"feat_{n}", "sql_ambiguity") for b in ("bird", "ambrosia")}
        for n in (1, 2, 15, 16, 20, 21, 23, 25, 28, 29, 30)
    }
    return {"format": "static-hybrid-audit-v1", "paid_calls": 0,
            "known_configurations": len(expected), "single_effects": singles,
            "train_single_vs_hybrid_complement": complement,
            "proposed_probes": proposed_probes(),
            "limitations": ["Single recorded trials; effects are not causal estimates.",
                            "FEAT_6 BIRD schema preparation was incomplete.",
                            "FEAT_25 Ambrosia blank output fails accuracy but is missing in paired utility.",
                            "Probes are hypotheses, not a global-optimality certificate or execution queue.",
                            "Historical full cohorts and old test were already examined."]}


def markdown(audit):
    lines = ["# Static hybrid: offline audit", "", "No API, SQL execution or new training.", "",
             "Accuracy uses fixed denominators 241/227; blank/error output fails.",
             "Rescued/harmed uses observed pairs only. Scores are single recorded outcomes.", "",
             "| Feature | BIRD correct | Delta pp | Rescued / harmed | Ambrosia correct | Delta pp | Rescued / harmed |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, scores in audit["single_effects"].items():
        b, a = scores["bird"], scores["ambrosia"]
        lines.append(f"| {name} | {b['correct']}/241 | {b['delta_pp']:+.2f} | {b['rescued']} / {b['harmed']} | "
                     f"{a['correct']}/227 | {a['delta_pp']:+.2f} | {a['rescued']} / {a['harmed']} |")
    lines += ["", "FEAT_6: incomplete BIRD schema context. FEAT_25: one invalid Ambrosia output, excluded from paired utility.",
              "", "## Proposed probes (not run)", "",
              "| Stage | Name | Flags | Controls |", "| --- | --- | --- | --- |"]
    for p in audit["proposed_probes"]:
        lines.append(f"| {p['stage']} | {p['name']} | {','.join(map(str, p['features']))} | {', '.join(p['controls'])} |")
    lines += ["", "Twenty proposed masks; running all on both full cohorts would be 40 benchmark runs / 9,360 question answers.",
              "This is a maximum initial menu, not a recommendation to execute all. Stage and filter on train/dev.",
              "Semantic audit and interpretation: ../../docs/10_static_hybrid_search.md", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="research_results/static_hybrid_audit_20261029")
    args = parser.parse_args()
    output = (ROOT / args.out).resolve()
    if not output.is_relative_to(ROOT / "research_results") or output == ROOT / "research_results":
        parser.error("Use a new dedicated folder inside research_results")
    if output.exists():
        parser.error("Output already exists; use a new folder")
    dataset_builder.verify(DATASET)
    rows = [json.loads(s) for s in (DATASET / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    audit = build_audit(rows)
    sources = [Path(__file__).resolve(), DATASET / "records.jsonl", DATASET / "manifest.json",
               ROOT / "practices_doc.md", ROOT / "scripts/run_openrouter_baseline.py",
               ROOT / "src/adv_text2sql/mcp_servers/text2sql_tool/src/text2sql_implementation.py",
               ROOT / "src/adv_text2sql/mcp_servers/text2sql_tool/src/prompts.py"]
    audit["source_sha256"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    output.mkdir(parents=True)
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "comparison.md").write_text(markdown(audit), encoding="utf-8", newline="\n")
    print(f"Offline audit: {audit['known_configurations']} known configurations; "
          f"{len(audit['proposed_probes'])} proposed masks, none executed. Output: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
