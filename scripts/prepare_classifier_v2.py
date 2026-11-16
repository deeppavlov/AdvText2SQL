"""Freeze a bounded v2 collection and export masked configuration observations offline."""

import argparse
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adv_text2sql.mcp_servers.text2sql_tool.src.feature_catalog_v2 import (  # noqa: E402
    CONFIGURATIONS, TASK_CONFIGURATIONS, catalog,
)
from audit_classifier_refresh import (  # noqa: E402
    OLD_DATASET, STAGE_SUMMARY, audit, digest, fingerprint, local_path, read,
)
from durable_prompt_store import write_json, write_bytes  # noqa: E402
from adv_text2sql.evaluation import validate_collected_observation  # noqa: E402

DEFAULT_OUTPUT = "research_results/classifier_v2_20261109"
DATA_OUTPUT = "data/multilabel_utility/deepseek_20261109_v2"
VERSION = "classifier-v2-collection-v1"
RUNTIME = "scripts/prompt_pilot_runtime.py"
CORE_FILES = (
    RUNTIME, "scripts/prompt_comparison_runtime.py", "scripts/sql_output_contract.py",
    "src/adv_text2sql/mcp_servers/text2sql_tool/src/prompts.py",
    "src/adv_text2sql/mcp_servers/text2sql_tool/src/text2sql_implementation.py",
)
NEW_FILES = (
    "scripts/prepare_classifier_v2.py", "scripts/collect_classifier_v2.py",
    "src/adv_text2sql/mcp_servers/text2sql_tool/src/feature_catalog_v2.py",
    "local/run_classifier_v2.sh",
)
SOURCES = (
    "bird_prompt_full_20261107", "ambrosia_prompt_full_20261107",
    "prompt_optimization_comparison_20261105", "bird_compact_prompt_transfer_20261107",
    "ambrosia_prompt_dev_20261107",
)


def client_options(path):
    """Resolve literal client kwargs from pinned source, never execute historical code."""
    calls = [n for n in ast.walk(ast.parse(Path(path).read_text(encoding="utf-8")))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "OpenAIChatCompletionClient"]
    if len(calls) != 1:
        raise ValueError("Review runtime: expected exactly one native client constructor")
    kwargs = {k.arg: k.value for k in calls[0].keywords}
    return {name: ast.literal_eval(kwargs[name]) for name in ("temperature", "max_tokens", "max_retries", "timeout")}


def selected_candidate(benchmark, summary, manifests):
    return (summary["profiles"][benchmark]["sql_candidate"] if benchmark == "bird"
            else manifests[benchmark]["jobs"][0]["arms"]["original"])


validate_observation = validate_collected_observation


def load_context(root=ROOT):
    readiness = audit(root)
    summary = read(root / STAGE_SUMMARY)
    manifests = {b: read(local_path(root, p["source_manifest_path"])) for b, p in summary["profiles"].items()}
    old = [json.loads(line) for line in (root / OLD_DATASET / "records.jsonl").read_text(encoding="utf-8").splitlines() if line]
    old = {r["id"]: r for r in old}
    rows = {}
    for benchmark, m in manifests.items():
        for item in m["full"][benchmark]:
            identity = f"{benchmark}:{item['question_id']}"
            if identity not in set(readiness["partitions"]["train"] + readiness["partitions"]["dev"]):
                continue
            r = old[identity]
            rows[identity] = {"id": identity, "benchmark": benchmark, "split": r["split"],
                              "db_id": r["db_id"], "group_id": r["group_id"], "family_keys": r["family_keys"],
                              "inputs": r["inputs"], "item": item}
    return readiness, summary, manifests, rows


def scan_archives(summary, manifests, rows, root=ROOT):
    observations, audit_rows, source_hashes = {}, [], {}
    target = manifests["bird"]
    actual = client_options(root / RUNTIME)
    for source in SOURCES:  # Predetermined priority; never choose the best saved answer.
        folder = root / "research_results" / source
        m = read(folder / "experiment.json")
        if m["sha256"] != fingerprint({k: v for k, v in m.items() if k != "sha256"}):
            raise ValueError("Archive manifest changed")
        compatible = (m["model"] == target["model"] and m["runtime_environment"] == target["runtime_environment"]
                      and all(m["files_sha256"].get(n) == target["files_sha256"].get(n) == digest(root / n) for n in CORE_FILES))
        entry = {"source": source, "accepted_observations": 0, "repeated_observations_not_selected": 0,
                 "reason": "matched_native_prompt_runtime" if compatible else "model_runtime_or_generator_mismatch"}
        audit_rows.append(entry)
        if not compatible:
            continue
        source_hashes[(folder / "experiment.json").relative_to(root).as_posix()] = digest(folder / "experiment.json")
        for path in sorted(folder.glob("jobs/*/evaluations.json")):
            values = read(path)
            for key, value in values.items():
                parts = key.split(":", 5)
                if len(parts) != 6:
                    raise ValueError("Unknown archive key format")
                phase, cid, profile, original, benchmark, qid = parts
                identity = f"{benchmark}:{qid}"
                if identity not in rows or profile != "hybrid" or original != "False" or phase not in ("search", "confirm"):
                    continue
                candidate = selected_candidate(benchmark, summary, manifests)
                if cid != fingerprint(candidate):
                    continue
                features = tuple(m.get("feature_mask", [14, 32]))
                name = next((n for n in TASK_CONFIGURATIONS[benchmark] if CONFIGURATIONS[n] == features), None)
                if name is None:
                    continue
                metadata_path = path.parent / phase / "metadata.json"
                if not metadata_path.exists():
                    continue
                meta = read(metadata_path)
                if meta.get("status") != "completed" or meta.get("manifest_sha256") != m["sha256"]:
                    continue
                if m["schema_audit"]["schemas"][benchmark] != manifests[benchmark]["schema_audit"]["schemas"][benchmark]:
                    raise ValueError("Archive schema differs")
                slot = "sql" if benchmark == "bird" else "ambiguity"
                items = (m["cohorts"][slot]["fit"] + m["cohorts"][slot]["inner_val"]
                         if phase == "search" else m.get("full", m["dev"])[benchmark])
                source_item = next((i for i in items if str(i["question_id"]) == qid), None)
                if source_item is None or any(source_item.get(k, "") != rows[identity]["item"].get(k, "")
                                              for k in ("question", "db_id", "evidence", "SQL")):
                    raise ValueError("Archive question/reference differs")
                score = validate_observation(value, rows[identity]["item"], benchmark)
                if score is None:
                    continue
                point = (identity, name)
                if point in observations:
                    entry["repeated_observations_not_selected"] += 1
                    continue
                observations[point] = {"score": score, "source": path.relative_to(root).as_posix(), "key": key,
                                       "usage": value.get("usage"), "archive_phase": phase}
                entry["accepted_observations"] += 1
                for p in (path, metadata_path):
                    source_hashes[p.relative_to(root).as_posix()] = digest(p)
    # Review legacy static outputs too, but do not silently waive a transport-policy change.
    legacy_path = root / "research_results/deepseek_full_static_20261103/experiment.json"
    legacy = read(legacy_path)
    if legacy["protocol_sha256"] != fingerprint(legacy["protocol"]):
        raise ValueError("Legacy static protocol changed")
    old_options = client_options(root / "scripts/static_hybrid_runtime.py")
    audit_rows.append({"source": "deepseek_full_static_20261103", "accepted_observations": 0,
                       "reason": "BIRD_prompt_mismatch; native_client_transport_policy_differs",
                       "legacy_client": old_options, "current_client": actual,
                       "ambrosia_requires_explicit_transport_compatibility_review": True})
    source_hashes[legacy_path.relative_to(root).as_posix()] = digest(legacy_path)
    audit_rows.append({"source": "original_single_flag_dataset", "accepted_observations": 0,
                       "reason": "not_automatically_reused; different_pipeline_and_BIRD_prompt; requires_separate_audit"})
    return observations, audit_rows, source_hashes


def make_plan(root=ROOT):
    readiness, summary, manifests, rows = load_context(root)
    observations, archives, sources = scan_archives(summary, manifests, rows, root)
    options = client_options(root / RUNTIME)
    model = {**manifests["bird"]["model"], "max_tokens": options["max_tokens"]}
    runtime_files = {p: expected for p, expected in manifests["bird"]["files_sha256"].items()
                     if p.endswith(".py")}
    for p, expected in runtime_files.items():
        if digest(root / p) != expected:
            raise ValueError(f"Runtime differs from archived observations: {p}")
    jobs = []
    for split in ("train", "dev"):
        for benchmark, names in TASK_CONFIGURATIONS.items():
            items = [r["item"] for r in rows.values() if r["split"] == split and r["benchmark"] == benchmark]
            for name in names:
                missing = [i for i in items if (f"{benchmark}:{i['question_id']}", name) not in observations]
                jobs.append({"name": f"{split}__{benchmark}__{name}", "split": split, "benchmark": benchmark,
                             "configuration": name, "features": list(CONFIGURATIONS[name]),
                             "items": missing, "cached_questions": len(items) - len(missing)})
    manifest = {"version": VERSION, "env_file": ".env.deepseek", "catalog": catalog(), "model": model,
                "cached_observations": [{"id": k[0], "configuration": k[1], "observation": v}
                                        for k, v in sorted(observations.items())],
                "declared_source_model": manifests["bird"]["model"], "effective_client": options,
                "source_max_tokens_discrepancy": {"declared": manifests["bird"]["model"]["max_tokens"],
                                                  "effective": options["max_tokens"], "evidence": RUNTIME},
                "runtime_base": manifests["bird"], "task_manifests": manifests,
                "candidates": {b: selected_candidate(b, summary, manifests) for b in manifests},
                "jobs": jobs, "readiness": readiness, "archive_audit": archives,
                "sources_sha256": {**readiness["source_files_sha256"], **sources,
                                   **runtime_files,
                                   **{p: digest(root / p) for p in NEW_FILES},
                                   "scripts/audit_classifier_refresh.py": digest(root / "scripts/audit_classifier_refresh.py"),
                                   "scripts/durable_prompt_store.py": digest(root / "scripts/durable_prompt_store.py"),
                                   "local/start_tunnel.sh": digest(root / "local/start_tunnel.sh"),
                                   "scripts/static_hybrid_runtime.py": digest(root / "scripts/static_hybrid_runtime.py")},
                "limits": {"examples": 200, "task_calls": 1000, "teacher_calls": 0,
                           "tokens": 5_000_000, "active_seconds": 7200},
                "global_limits": {"examples": 2000, "task_calls": 6000, "teacher_calls": 0,
                                  "tokens": 20_000_000, "active_seconds": 72000},
                "policy": {"holdout_used": False, "independent_test_available": False,
                           "archive_priority": list(SOURCES), "label_scope": "configuration_quality_and_conditional_differences",
                           "unknown": "null_score_false_mask", "automatic_resume": False}}
    manifest["sha256"] = fingerprint(manifest)
    return manifest, rows, observations


def validate(manifest):
    if manifest.get("version") != VERSION or manifest["sha256"] != fingerprint({k: v for k, v in manifest.items() if k != "sha256"}):
        raise ValueError("Collection manifest changed")
    if manifest["catalog"] != catalog():
        raise ValueError("Feature catalog changed; create a new protocol")
    for p, expected in manifest["sources_sha256"].items():
        if digest(local_path(ROOT, p)) != expected:
            raise ValueError(f"Pinned collection source changed: {p}")


def export_dataset(root, manifest, rows, observations):
    """Only train/dev scores; do not invent a feature target from an entire mask."""
    records = []
    for identity, r in sorted(rows.items()):
        if r["split"] not in ("train", "dev") or set(r["inputs"]) != {"question", "evidence", "schema_stats"}:
            raise ValueError("Export accepts train/dev model inputs only; no holdout/targets in inputs")
        names = TASK_CONFIGURATIONS[r["benchmark"]]
        scores = [observations.get((identity, n), {}).get("score") for n in names]
        records.append({k: r[k] for k in ("id", "benchmark", "split", "group_id", "inputs")} | {
            "configurations": list(names), "quality_targets": scores,
            "loss_mask": [s is not None for s in scores],
            "observations": {n: observations[(identity, n)] for n in names if (identity, n) in observations}})
    pairs = []
    for row in records:
        for j, left in enumerate(row["configurations"]):
            for right in row["configurations"][j + 1:]:
                if left not in row["observations"] or right not in row["observations"]:
                    continue
                a, b = set(CONFIGURATIONS[left]), set(CONFIGURATIONS[right])
                if not (a < b or b < a):
                    continue
                control, treatment = (left, right) if a < b else (right, left)
                pairs.append({"id": row["id"], "split": row["split"], "control": control, "treatment": treatment,
                              "context_features": list(CONFIGURATIONS[control]),
                              "added_features": sorted(set(CONFIGURATIONS[treatment]) - set(CONFIGURATIONS[control])),
                              "utility": row["observations"][treatment]["score"] - row["observations"][control]["score"],
                              "individual_feature_target_available": len(set(CONFIGURATIONS[treatment]) - set(CONFIGURATIONS[control])) == 1})
    counts = {s: {b: {"questions": sum(r["split"] == s and r["benchmark"] == b for r in records),
                      "observed_scores": sum(sum(r["loss_mask"]) for r in records if r["split"] == s and r["benchmark"] == b),
                      "fully_observed_questions": sum(all(r["loss_mask"]) for r in records if r["split"] == s and r["benchmark"] == b)}
                  for b in TASK_CONFIGURATIONS} for s in ("train", "dev")}
    ready = all(v["questions"] > 0 and v["questions"] == v["fully_observed_questions"]
                for splits in counts.values() for v in splits.values())
    output = root / DATA_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    for name, values in (("examples.jsonl", records), ("conditional_pairs.jsonl", pairs)):
        write_bytes(output / name, ("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values)).encode("utf-8"))
    result = {"version": "configuration-utility-v2-draft", "collection_manifest_sha256": manifest["sha256"],
              "catalog": manifest["catalog"], "coverage": counts, "ready_for_training": ready,
              "training_blockers": [] if ready else ["Incomplete fixed train/dev configuration coverage; do not train on fabricated negatives"],
              "conditional_pairs": len(pairs), "test_included": False,
              "source_files_sha256": manifest["sources_sha256"],
              "observation_files_sha256": {p: digest(local_path(root, p))
                                           for p in sorted({v["source"] for v in observations.values()})},
              "output_files_sha256": {p: digest(output / p) for p in ("examples.jsonl", "conditional_pairs.jsonl")}}
    write_json(output / "manifest.json", result)
    return result


def status(manifest, dataset):
    print("OFFLINE classifier v2; training ready:", dataset["ready_for_training"])
    print("Effective generation max_tokens:", manifest["effective_client"]["max_tokens"],
          "; old prompt manifests declared:", manifest["declared_source_model"]["max_tokens"])
    for split in ("train", "dev"):
        jobs = [j for j in manifest["jobs"] if j["split"] == split]
        remaining = sum(v["questions"] * len(TASK_CONFIGURATIONS[b]) - v["observed_scores"]
                        for b, v in dataset["coverage"][split].items())
        print(split, "planned configurations:", len(jobs), "cached:", sum(j["cached_questions"] for j in jobs),
              "remaining question evaluations:", remaining)
    print("Conditional pairs:", dataset["conditional_pairs"], "; paid calls during preparation: 0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = local_path(ROOT, args.output)
    if not output.is_relative_to(ROOT / "research_results") or output == ROOT / "research_results":
        parser.error("Use a dedicated research_results collection folder")
    manifest, rows, observations = make_plan()
    if not args.prepare:
        print("OFFLINE:", sum(len(j["items"]) for j in manifest["jobs"]), "missing question evaluations; use --prepare to freeze")
        return
    path = output / "experiment.json"
    if path.exists() and read(path) != manifest:
        raise ValueError("Existing collection differs; use a new folder")
    output.mkdir(parents=True, exist_ok=True)
    from run_feature_ablation import suite_lock
    from collect_classifier_v2 import collected_observations
    with suite_lock(output):
        write_json(path, manifest)
        write_json(output / "archive_audit.json", manifest["archive_audit"])
        new = collected_observations(output, manifest, rows)
        if set(new) & set(observations):
            raise ValueError("Collected observations duplicate archived ones")
        status(manifest, export_dataset(ROOT, manifest, rows, observations | new))


if __name__ == "__main__":
    main()
