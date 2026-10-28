"""Build configuration utility labels and grouped splits from saved runs; no network."""

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from ablation_maintenance import approved_protocols
from atomic_artifacts import write_json
from configuration_splits import (
    FAMILY_POLICY, PARTITIONS, audit_partitions, family_keys, family_stress_folds,
    make_groups, normalize_family, partition, training_folds,
)
from research_metrics import ambrosia_report
from run_feature_combinations import validate_manifest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = "data/configuration_utility/deepseek_20261028"
FORMAT = "configuration-utility-v1"
DATASETS = {"bird": "data/bird_large.json", "ambrosia": "data/ambrosia_large_available_227.json"}
EXPERIMENTS = {
    "ablation": "research_results/deepseek_direct_ablation_large_20261024",
    "combinations": "research_results/deepseek_combinations_large_20261026",
    "relationships": "research_results/deepseek_relationships_large_20261026",
}
CONFIGURATIONS = {
    "baseline": {"experiment": "ablation", "case": "baseline", "features": [], "role": "control"},
    "economical": {"experiment": "ablation", "case": "feat_35", "features": [35], "role": "candidate"},
    "hybrid": {"experiment": "combinations", "case": "sql_ambiguity", "features": [14, 32, 35], "role": "candidate"},
    "relationships": {"experiment": "relationships", "case": "hybrid_with_relationships",
                      "features": [1, 14, 32, 35], "role": "auxiliary"},
}
PRIMARY = ("economical", "hybrid")
GENERATION_KEYS = (
    "model", "model_version", "base_url", "api_backend", "provider", "request_options",
    "temperature", "max_output_tokens", "reasoning", "max_retries", "base_interval_seconds",
    "min_interval_seconds", "ambiguity_prompt",
)
MODEL_INPUT_FIELDS = ("question", "evidence", "schema_stats")
PROMPT_TEMPLATES = ("SYSTEM_PROMPT_TEMPLATE", "SQL_PROMPT_TEMPLATE", "SQL_PROMPT_BASIC",
                    "AMBIGUITY_PROMPT_SIMPLE", "TAXONOMY_DETECTION_PROMPT")
OUTPUT_FILES = ("records.jsonl", "inputs.jsonl", "splits.json", "cv_folds.json",
                "ambrosia_family_holdout.json", "summary.json", "summary.md")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_questions(path):
    items = read(path)
    ids = [str(row["question_id"]) for row in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs in the source dataset")
    if any(not isinstance(row.get("question"), str) or not row["question"].strip() for row in items):
        raise ValueError("Source questions must be non-empty strings")
    return sorted(items, key=lambda row: str(row["question_id"]))


def execution_failure_kind(entry):
    if not entry.get("error"):
        return None
    statement = re.search(r"\[SQL: (.*?)\]\s*(?:\[parameters:|\(Background|$)", entry["error"], re.S)
    if not statement:
        return "unknown"
    def normalized(sql):
        return re.sub(r"\s+", " ", sql.strip()).replace("%%", "%")
    sql = normalized(statement.group(1))
    # The reference executes first. An identical failing reference/prediction
    # must never become a negative training label against the model.
    if sql == normalized(entry.get("gold_sql", "")):
        return "reference"
    return "prediction" if sql == normalized(entry.get("predicted_sql", "")) else "unknown"


def active_prompt_catalog(path):
    values = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in PROMPT_TEMPLATES:
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, str):
                        raise ValueError("Prompt template must be a literal string")
                    values[target.id] = value
    if set(values) != set(PROMPT_TEMPLATES):
        raise ValueError("Selected prompt catalog cannot be reconstructed")
    return "\n".join(values[name] for name in PROMPT_TEMPLATES)


def observations(items, predictions, report, benchmark):
    """Missing observations stay null; errors are observed failures, not missing data."""
    ids = {str(row["question_id"]) for row in items}
    if not set(predictions) <= ids:
        raise ValueError("Predictions contain IDs outside the source dataset")
    if any(not isinstance(v, str) or not v.strip() for v in predictions.values()):
        raise ValueError("Invalid prediction value")
    scores = {}
    if benchmark == "bird":
        for row in report.get("results", []):
            key, score = str(row["question_id"]), row["score"]
            if key not in ids or key in scores or type(score) not in (bool, int) or score not in (0, 1):
                raise ValueError("BIRD report contains duplicate/invalid scores or unexpected IDs")
            scores[key] = row
    result = {}
    for item in items:
        key = str(item["question_id"])
        prediction = predictions.get(key)
        observed = prediction is not None and (benchmark == "ambrosia" or key in scores)
        error = prediction == "error" if prediction is not None else None
        failure_kind = execution_failure_kind(scores[key]) if benchmark == "bird" and key in scores else None
        correct = None
        if observed:
            correct = (not error and (prediction == "ambiguous") == (item["SQL"] == "ambiguous")) \
                if benchmark == "ambrosia" else bool(scores[key]["score"])
            if error and correct:
                raise ValueError("Generation error cannot have a successful BIRD score")
            if benchmark == "bird" and scores[key].get("predicted_sql") != prediction:
                raise ValueError("BIRD prediction differs from the scored SQL")
            if benchmark == "bird" and scores[key].get("error") and correct:
                raise ValueError("Execution error cannot have a successful score")
            if failure_kind in ("reference", "unknown"):
                observed, correct = False, None
        result[key] = {
            "observed": observed, "correct": correct, "generation_error": error,
            "execution_error": bool(scores[key].get("error")) if benchmark == "bird" and key in scores else None,
            "evaluation_error_kind": failure_kind,
            "predicted_ambiguous": prediction == "ambiguous" if observed and not error else None,
            "response_sha256": hashlib.sha256(prediction.encode("utf-8")).hexdigest() if prediction is not None else None,
            "question_cost_observed": False, "prompt_tokens": None, "completion_tokens": None,
        }
    return result


def utility(first, control):
    if not first["observed"] or not control["observed"]:
        return None
    return int(first["correct"]) - int(control["correct"])


def selection_labels(outcomes):
    a, b = outcomes["economical"], outcomes["hybrid"]
    difference = utility(b, a)
    if difference is None:
        return {"utility_hybrid_vs_economical": None, "pair_observed": False,
                "quality_preference": None, "quality_preference_observed": False,
                "cost_aware_preference": None, "outcome_class": "unobserved"}
    if difference > 0:
        outcome = "hybrid_only_correct"
    elif difference < 0:
        outcome = "economical_only_correct"
    else:
        outcome = "both_correct" if a["correct"] else "neither_correct"
    return {
        "utility_hybrid_vs_economical": difference, "pair_observed": True,
        "quality_preference": ("hybrid" if difference > 0 else "economical") if difference else None,
        "quality_preference_observed": difference != 0,
        "cost_aware_preference": "hybrid" if difference > 0 else "economical",
        "outcome_class": outcome,
    }


def stream_raw(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if any(type(row.get(k)) is not int or row[k] < 0 for k in ("prompt_tokens", "completion_tokens")):
                    raise ValueError("Invalid recorded token counts")
                yield row


def tsv_schemas(messages):
    for message in messages:
        lines = message["content"].splitlines()
        for i, line in enumerate(lines):
            if line.strip() != "table\tcolumn\ttype":
                continue
            rows = []
            for line in lines[i + 1:]:
                parts = line.split("\t")
                if len(parts) != 3 or not all(p.strip() for p in parts):
                    break
                rows.append(tuple(parts))
            if rows:
                if len({(r[0], r[1]) for r in rows}) != len(rows):
                    raise ValueError("Duplicate columns in a TSV schema block")
                yield tuple(sorted(rows))


def schema_statistics(rows):
    types = [r[2].casefold() for r in rows]
    return {
        "table_count": len({r[0] for r in rows}), "column_count": len(rows),
        "text_column_count": sum(bool(re.search(r"text|char|clob", t)) for t in types),
        "numeric_column_count": sum(bool(re.search(r"int|numeric|decimal|real|float|double", t)) for t in types),
        "temporal_column_count": sum(bool(re.search(r"date|time", t)) for t in types),
        "boolean_column_count": sum("bool" in t for t in types),
    }


def extract_schemas(items, raw_path):
    candidates = defaultdict(set)
    for row in stream_raw(raw_path):
        blocks = set(tsv_schemas(row["messages"]))
        if not blocks:
            continue
        messages = [m["content"] for m in row["messages"]]
        dbs = {item["db_id"] for item in items if any(item["question"] in m for m in messages)}
        # Never guess an identity when repeated question text matches multiple DBs.
        if len(dbs) == 1:
            candidates[next(iter(dbs))].update(blocks)
    result = {}
    for db in sorted({item["db_id"] for item in items}):
        versions = candidates[db]
        if len(versions) != 1:
            raise ValueError(f"Missing or inconsistent saved TSV schema for database {db}")
        rows = next(iter(versions))
        content = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
        result[db] = {"sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                      "statistics": schema_statistics(rows)}
    return result


def validate_protocols(manifests):
    first = next(iter(manifests.values()))["protocol"]
    for manifest in manifests.values():
        current = manifest["protocol"]
        if any(first.get(key) != current.get(key) for key in GENERATION_KEYS):
            raise ValueError("Experiments mix models or generation settings")
        protected = {p for p in first["files_sha256"] if p.startswith(("src/", "benchmarks/", "data/"))}
        protected |= {"scripts/llm_endpoints.py", "scripts/baseline_connection.py"}
        if any(first["files_sha256"].get(p) != current["files_sha256"].get(p) for p in protected):
            raise ValueError("Experiments mix generator, prompt, evaluator or dataset versions")


def load_experiments(root):
    manifests, accepted, sources = {}, {}, {}
    for name, relative in EXPERIMENTS.items():
        folder = root / relative
        manifest = read(folder / "experiment.json")
        if read(folder / "suite_status.json")["status"] != "completed":
            raise ValueError("All source experiments must be completed")
        if "references" in manifest:
            validate_manifest(manifest)
            accepted[name] = {manifest["protocol_sha256"]}
            for file, expected in manifest["references"]["files_sha256"].items():
                if sha(root / file) != expected:
                    raise ValueError(f"Frozen reference changed: {file}")
                sources[file] = expected
        else:
            revisions = manifest.get("execution_revisions", [])
            latest = revisions[-1]["protocol"] if revisions else manifest["protocol"]
            accepted[name] = approved_protocols(manifest, latest)
        manifests[name] = manifest
        for file in ("experiment.json", "suite_status.json"):
            sources[(folder / file).relative_to(root).as_posix()] = sha(folder / file)
    validate_protocols(manifests)
    return manifests, accepted, sources


def load_run(root, folder, benchmark, items, config, manifest, accepted):
    meta = read(folder / "metadata.json")
    if meta.get("status") != "completed":
        raise ValueError("A configuration run is not completed")
    identity = meta["configuration"]
    expected = {
        "model": manifest["protocol"]["model"], "model_version": manifest["protocol"].get("model_version"),
        "api_backend": manifest["protocol"]["api_backend"], "base_url": manifest["protocol"]["base_url"],
        "provider": manifest["protocol"]["provider"], "request_options": manifest["protocol"]["request_options"],
        "temperature": 0, "max_output_tokens": 4096, "reasoning_effort": "none", "max_retries": 5,
        "use_evidence": benchmark == "bird", "dataset_sha256": sha(root / DATASETS[benchmark]),
        "feature_flags": {f"FEAT_{n}": n in config["features"] for n in range(1, 37)},
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Incompatible configuration metadata: {folder.relative_to(root)}")
    if identity.get("protocol_sha256") not in accepted:
        raise ValueError("Unknown run protocol")
    report, predictions = read(folder / "report.json"), read(folder / "query_results.json")
    per_question = observations(items, predictions, report, benchmark)
    if set(predictions) != {str(i["question_id"]) for i in items} or not all(o["observed"] for o in per_question.values()):
        raise ValueError("Completed configuration has incomplete scores or an invalid/unclassified evaluation")
    correct = sum(o["correct"] for o in per_question.values())
    if benchmark == "bird":
        if abs(report["overall_accuracy"] - 100 * correct / len(items)) > 1e-8:
            raise ValueError("BIRD aggregate score disagrees with question-level scores")
        metrics = {"accuracy": 100 * correct / len(items)}
    else:
        metrics = ambrosia_report(items, predictions)
        if report.get("evaluation_policy") == "ambrosia-strict-v1" and abs(report["accuracy"] - metrics["accuracy"]) > 1e-8:
            raise ValueError("Strict Ambrosia aggregate disagrees with saved predictions")
    cost = Counter()
    raw = folder / "raw_responses.jsonl"
    for row in stream_raw(raw):
        cost.update({k: row[k] for k in ("prompt_tokens", "completion_tokens")})
        cost["recorded_calls"] += 1
    total = cost["prompt_tokens"] + cost["completion_tokens"]
    files = {p.relative_to(root).as_posix(): sha(p) for p in
             (folder / "metadata.json", folder / "query_results.json", folder / "report.json", raw)}
    return per_question, {
        "path": folder.relative_to(root).as_posix(), "protocol_sha256": identity["protocol_sha256"],
        "questions": len(items), "correct": correct, "accuracy": 100 * correct / len(items),
        "generation_errors": sum(bool(o["generation_error"]) for o in per_question.values()),
        "execution_errors": sum(bool(o["execution_error"]) for o in per_question.values()),
        "ambiguity_metrics": metrics.get("ambiguity_metrics"),
        "recorded_run_cost": {**dict(cost), "total_tokens": total, "mean_tokens_per_question": total / len(items),
                              "includes_probe_and_build": True, "exact_question_cost_available": False},
    }, files


def build_records(questions, outcomes, schemas, prompt_text):
    records = []
    # Keep quoted example sentences: replacing them with a literal placeholder
    # would hide precisely the prompt exposure this diagnostic is meant to flag.
    prompt_normalized = normalize_family(prompt_text, replace_literals=False)
    for benchmark, items in sorted(questions.items()):
        for item in items:
            key = str(item["question_id"])
            values = {name: dict(outcomes[name][benchmark][key]) for name in CONFIGURATIONS}
            for value in values.values():
                value["utility_vs_baseline"] = utility(value, values["baseline"])
                value["utility_vs_economical"] = utility(value, values["economical"])
            question_template = normalize_family(item["question"], replace_literals=False)
            family_template = normalize_family(item.get("ambig_question", item["question"]), replace_literals=False)
            schema = schemas[benchmark][item["db_id"]]
            records.append({
                "id": f"{benchmark}:{key}", "benchmark": benchmark,
                "task": "sql_execution" if benchmark == "bird" else "ambiguity_detection",
                "question_id": key, "db_id": item["db_id"],
                "inputs": {"question": item["question"], "evidence": item.get("evidence", ""),
                           "schema_stats": schema["statistics"]},
                "family_keys": family_keys(item, benchmark),
                "analysis": {
                    "difficulty": item.get("difficulty"),
                    "gold_ambiguous": item["SQL"] == "ambiguous" if benchmark == "ambrosia" else None,
                    "construction_family": item["db_id"].split("_", 1)[0] if benchmark == "ambrosia" else None,
                    "schema_sha256": schema["sha256"],
                    "question_example_overlap": len(question_template.split()) >= 6 and question_template in prompt_normalized,
                    "family_example_overlap": len(family_template.split()) >= 6 and family_template in prompt_normalized,
                },
                "outcomes": values, "labels": selection_labels(values),
            })
    return sorted(records, key=lambda row: row["id"])


def summary_data(records, registry, groups, splits, audit, near_edges):
    result = {
        "records": len(records), "databases": len({(r["benchmark"], r["db_id"]) for r in records}),
        "groups": len(groups), "near_duplicate_edges": len(near_edges), "leakage_audit": audit,
        "independent_final_test": False, "benchmarks": {}, "splits": {},
    }
    for benchmark in DATASETS:
        rows = [r for r in records if r["benchmark"] == benchmark]
        classes = Counter(r["labels"]["outcome_class"] for r in rows)
        oracle = sum(any(r["outcomes"][c]["correct"] for c in PRIMARY) for r in rows)
        result["benchmarks"][benchmark] = {
            "records": len(rows), "databases": len({r["db_id"] for r in rows}),
            "outcome_classes": dict(sorted(classes.items())),
            "quality_preference_observed": sum(r["labels"]["quality_preference_observed"] for r in rows),
            "oracle_correct": oracle, "oracle_accuracy": 100 * oracle / len(rows),
            "question_example_overlap": sum(r["analysis"]["question_example_overlap"] for r in rows),
            "family_example_overlap": sum(r["analysis"]["family_example_overlap"] for r in rows),
            "configurations": {name: registry[name]["benchmarks"][benchmark] for name in CONFIGURATIONS},
        }
    by_id = {r["id"]: r for r in records}
    for split, ids in splits.items():
        result["splits"][split] = {}
        for benchmark in DATASETS:
            rows = [by_id[i] for i in ids if by_id[i]["benchmark"] == benchmark]
            result["splits"][split][benchmark] = {
                "records": len(rows), "databases": sorted({r["db_id"] for r in rows}),
                "outcome_classes": dict(sorted(Counter(r["labels"]["outcome_class"] for r in rows).items())),
                "quality_preference_observed": sum(r["labels"]["quality_preference_observed"] for r in rows),
                "gold_ambiguous": sum(bool(r["analysis"]["gold_ambiguous"]) for r in rows),
                "prompt_family_overlap": sum(r["analysis"]["family_example_overlap"] for r in rows),
            }
    return result


def markdown_summary(summary):
    lines = ["# Configuration utility dataset", "", "Offline retrospective dataset; not an independent final test.", "",
             "| Benchmark | Rows | DBs | Hybrid only correct | Economical only correct | Both correct | Neither correct | Quality labels | Oracle |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for benchmark, row in summary["benchmarks"].items():
        c = row["outcome_classes"]
        lines.append(f"| {benchmark} | {row['records']} | {row['databases']} | {c.get('hybrid_only_correct', 0)} | "
                     f"{c.get('economical_only_correct', 0)} | {c.get('both_correct', 0)} | {c.get('neither_correct', 0)} | "
                     f"{row['quality_preference_observed']} | {row['oracle_accuracy']:.2f}% |")
    lines += ["", "| Split | Benchmark | Rows | DBs | Quality labels | Gold ambiguous | Prompt-exposed families, rows |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for split, benchmarks in summary["splits"].items():
        for benchmark, row in benchmarks.items():
            lines.append(f"| {split} | {benchmark} | {row['records']} | {len(row['databases'])} | "
                         f"{row['quality_preference_observed']} | {row['gold_ambiguous']} | {row['prompt_family_overlap']} |")
    lines += ["", "Only the inputs object is a model feature source. DB IDs, benchmark difficulty, gold labels,",
              "families, split IDs and outcomes are audit/target fields, not classifier inputs.",
              "utility = configuration_correct - control_correct; missing observations remain null.",
              "Quality preference is null for ties. Cost-aware ties prefer the economical configuration",
              "using the fixed run-level cost policy, not fabricated per-question costs.",
              "All exact question token costs are unknown/masked. Recorded run totals include probe/build calls.",
              "Source ambiguity labels use strict-v1: generation error is always incorrect.",
              "Small sets were not appended. These observations do not provide independent multilabel feature targets.",
              "Family normalization and high-similarity grouping are conservative heuristics, not semantic guarantees.",
              "Configuration selection and prompt examples have already exposed this material; test is retrospective.",
              "Family stress folds are separate diagnostics over Ambrosia, not part of main train/dev/test tuning."]
    return "\n".join(lines) + "\n"


def audit_folds(records, groups, splits, folds, stress):
    by_id = {r["id"]: r for r in records}
    train = set(splits["train"])
    validation_seen = Counter()
    for fold in folds:
        a, b = set(fold["train_ids"]), set(fold["validation_ids"])
        if a & b or a | b != train:
            raise ValueError("CV includes outer dev/test rows or overlaps its validation set")
        validation_seen.update(b)
        if any(set(ids) & a and set(ids) & b for ids in groups.values()):
            raise ValueError("CV splits a database/family group")
    if set(validation_seen) != train or any(n != 1 for n in validation_seen.values()):
        raise ValueError("Every training row must be validated exactly once in CV")
    for fold in stress:
        a, b = set(fold["train_ids"]), set(fold["test_ids"])
        expected = {r["id"] for r in records if r["benchmark"] == "ambrosia"}
        if a & b or a | b != expected:
            raise ValueError("Ambrosia family holdout has invalid coverage")
        if {by_id[i]["db_id"] for i in a} & {by_id[i]["db_id"] for i in b}:
            raise ValueError("Database leakage in construction-family holdout")
        if {f for i in a for f in by_id[i]["family_keys"]} & {f for i in b for f in by_id[i]["family_keys"]}:
            raise ValueError("Question-family leakage in construction-family holdout")


def verify_bundle(output, root=ROOT):
    manifest = read(output / "manifest.json")
    if manifest.get("format_version") != FORMAT:
        raise ValueError("Unknown dataset format")
    for relative, expected in manifest["output_files_sha256"].items():
        if relative not in OUTPUT_FILES or sha(output / relative) != expected:
            raise ValueError(f"Derived artifact hash changed: {relative}")
    if set(manifest["output_files_sha256"]) != set(OUTPUT_FILES):
        raise ValueError("Incomplete dataset bundle")
    for relative, expected in manifest["source_files_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"Source artifact hash changed: {relative}")
    records = [json.loads(line) for line in (output / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    split = read(output / "splits.json")
    audit_partitions(records, split["groups"], split["partitions"])
    audit_folds(records, split["groups"], split["partitions"], read(output / "cv_folds.json"),
                read(output / "ambrosia_family_holdout.json"))
    inputs = [json.loads(line) for line in (output / "inputs.jsonl").read_text(encoding="utf-8").splitlines()]
    if inputs != [{"id": r["id"], "inputs": r["inputs"]} for r in records]:
        raise ValueError("Inputs-only export differs from the records")
    for row in records:
        if set(row["inputs"]) != set(MODEL_INPUT_FIELDS) or row["labels"] != selection_labels(row["outcomes"]):
            raise ValueError("Unsafe model features or inconsistent selection labels")
        for value in row["outcomes"].values():
            if (not value["observed"] and value["correct"] is not None) or (
                value["generation_error"] and value["correct"] is not False
            ):
                raise ValueError("Missing/failed observation was treated as successful")
    return manifest


def build(root=ROOT, output=None, seed=42, ratios=(0.6, 0.2, 0.2)):
    root = Path(root).resolve()
    output = Path(output or root / DEFAULT_OUTPUT).resolve()
    if not output.is_relative_to(root / "data") or output == root / "data":
        raise ValueError("Output must be a dedicated folder inside data/")
    if output.exists() and any(output.iterdir()) and not (output / "manifest.json").exists():
        raise ValueError("Output has unrelated files without a dataset manifest")
    if (output / "manifest.json").exists() and read(output / "manifest.json").get("format_version") != FORMAT:
        raise ValueError("Refusing to overwrite a different dataset format")
    questions = {b: load_questions(root / path) for b, path in DATASETS.items()}
    manifests, accepted, sources = load_experiments(root)
    registry = {name: {**config, "benchmarks": {}} for name, config in CONFIGURATIONS.items()}
    outcomes = {name: {} for name in CONFIGURATIONS}
    for name, config in CONFIGURATIONS.items():
        for benchmark, items in questions.items():
            folder = root / EXPERIMENTS[config["experiment"]] / config["case"] / (
                "bird" if benchmark == "bird" else "ambrosia_227")
            obs, info, files = load_run(root, folder, benchmark, items, config,
                                        manifests[config["experiment"]], accepted[config["experiment"]])
            outcomes[name][benchmark] = obs
            registry[name]["benchmarks"][benchmark] = info
            sources.update(files)
    schemas = {}
    for benchmark, items in questions.items():
        folder = root / registry["economical"]["benchmarks"][benchmark]["path"]
        schemas[benchmark] = extract_schemas(items, folder / "raw_responses.jsonl")
    prompt_path = root / "src/adv_text2sql/mcp_servers/text2sql_tool/src/prompts.py"
    if sha(prompt_path) != manifests["ablation"]["protocol"]["files_sha256"][prompt_path.relative_to(root).as_posix()]:
        raise ValueError("Working prompt catalog differs from the historical source snapshot")
    records = build_records(questions, outcomes, schemas, active_prompt_catalog(prompt_path))
    assignments, groups, near_edges = make_groups(records)
    splits = partition(records, groups, seed, ratios)
    owner = {i: name for name, ids in splits.items() for i in ids}
    for row in records:
        row.update(group_id=assignments[row["id"]], split=owner[row["id"]])
    audit = audit_partitions(records, groups, splits)
    folds = training_folds(records, groups, splits)
    stress = family_stress_folds(records)
    audit_folds(records, groups, splits, folds, stress)
    stats = summary_data(records, registry, groups, splits, audit, near_edges)
    small = {}
    for benchmark, path in DATASETS.items():
        sources[path] = sha(root / path)
        small_path = root / "data" / f"{benchmark}_small.json"
        small_items = load_questions(small_path)
        ids = {str(r["question_id"]) for r in questions[benchmark]}
        small[benchmark] = {"rows": len(small_items), "overlap_with_current_large": sum(
            str(r["question_id"]) in ids for r in small_items), "appended": False}
        sources[small_path.relative_to(root).as_posix()] = sha(small_path)
    sources[prompt_path.relative_to(root).as_posix()] = sha(prompt_path)
    output.mkdir(parents=True, exist_ok=True)
    for file, data in (("records.jsonl", records), ("inputs.jsonl", [{"id": r["id"], "inputs": r["inputs"]} for r in records])):
        text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in data)
        (output / file).write_text(text, encoding="utf-8", newline="\n")
    write_json(output / "splits.json", {
        "seed": seed, "ratios": dict(zip(PARTITIONS, ratios)), "family_policy": FAMILY_POLICY,
        "partition_method": "database/family connected components; balance row counts only",
        "partitions": splits, "groups": groups, "near_duplicate_edges": near_edges,
        "independent_final_test": False,
    })
    write_json(output / "cv_folds.json", folds)
    write_json(output / "ambrosia_family_holdout.json", stress)
    write_json(output / "summary.json", stats)
    (output / "summary.md").write_text(markdown_summary(stats), encoding="utf-8", newline="\n")
    manifest = {
        "format_version": FORMAT, "primary_candidates": list(PRIMARY), "configurations": registry,
        "model": {k: manifests["ablation"]["protocol"].get(k) for k in GENERATION_KEYS},
        "model_input_fields": list(MODEL_INPUT_FIELDS), "independent_multilabel_targets_available": False,
        "label_policy": {
            "utility": "correct(configuration) - correct(control), observed single-run outcomes, not causal effects",
            "missing": "null + observed=false", "generation_error": "observed failure, never successful",
            "ambiguity": "ambrosia-strict-v1; class detection only, not SQL execution quality",
            "quality_preference": "observed only when primary candidates differ in correctness",
            "cost_aware_tie": "prefer economical by fixed recorded run-level policy, not measured question cost",
            "question_cost": "unavailable; null + question_cost_observed=false",
        },
        "grouping": {"family_policy": FAMILY_POLICY, "seed": seed, "ratios": list(ratios),
                     "near_duplicate_threshold": {"token_jaccard": 0.8, "sequence_ratio": 0.9, "min_unique_tokens": 8}},
        "prompt_overlap_policy": {"templates": list(PROMPT_TEMPLATES),
                                  "method": "normalized substring in selected static templates; possible exposure only"},
        "evaluation_scope": {"retrospective_test": True, "independent_final_test_available": False,
                             "configuration_selection_already_used_all_questions": True,
                             "family_stress_is_separate_diagnostic": True},
        "small_sets": small, "source_files_sha256": dict(sorted(sources.items())),
        "builder_files_sha256": {"scripts/" + p: sha(Path(__file__).parent / p) for p in
                                 ("build_configuration_dataset.py", "configuration_splits.py", "research_metrics.py")},
        "output_files_sha256": {file: sha(output / file) for file in OUTPUT_FILES},
    }
    write_json(output / "manifest.json", manifest)
    verify_bundle(output, root)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.6, 0.2, 0.2))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = (ROOT / args.out).resolve()
    if not output.is_relative_to(ROOT / "data") or output == ROOT / "data":
        parser.error("Use a dedicated output folder inside data/")
    if args.verify_only:
        verify_bundle(output)
        print(f"VERIFIED: source hashes, labels, grouped splits and CV: {output}")
    else:
        result = build(output=output, seed=args.seed, ratios=tuple(args.ratios))
        print(f"BUILT: {result['records']} records, {result['databases']} DBs, {result['groups']} groups: {output}")
        for benchmark, data in result["benchmarks"].items():
            print(f"{benchmark}: rows={data['records']}, quality preference labels={data['quality_preference_observed']}, "
                  f"classes={data['outcome_classes']}")
        for split, benchmarks in result["splits"].items():
            print(split, {b: v["records"] for b, v in benchmarks.items()})


if __name__ == "__main__":
    main()
