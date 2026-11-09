"""Offline preparation and manual, bounded comparison of real prompt optimizers."""

import argparse
import ast
import importlib.metadata as metadata
import json
import os
import statistics
import sys

from dotenv import dotenv_values

from atomic_artifacts import write_json
import prompt_pilot as pilot
import prompt_pilot_guardfix as guard
import static_hybrid_search as search

ROOT = pilot.ROOT
PACKAGES = ROOT / ".cache/prompt_comparison"
DEFAULT_OUTPUT = "research_results/prompt_optimization_comparison_20261105"
VERSION = "prompt-comparison-v1"
SEEDS = (42, 43, 44)
METHODS = ("gepa_zero", "mipro_zero", "textgrad_zero", "mipro_few")
SLOTS = ("sql", "ambiguity")
LIMITS = dict(
    examples=800,
    task_calls=1800,
    teacher_calls=48,
    tokens=6_000_000,
    active_seconds=14400,
)
GLOBAL_LIMITS = dict(
    examples=20000,
    task_calls=45000,
    teacher_calls=1400,
    tokens=60_000_000,
    active_seconds=172800,
)
FILES = (
    "scripts/prompt_comparison.py",
    "scripts/prompt_comparison_runtime.py",
    "scripts/prompt_comparison_methods.py",
    "local/run_prompt_comparison.ps1",
    "local/run_prompt_comparison.sh",
    "requirements-prompt-comparison.txt",
)


def load_packages():
    # New directory, never the old, hash-pinned .cache/prompt_optimizer.
    if str(PACKAGES) not in sys.path:
        sys.path.append(str(PACKAGES))
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    os.environ.setdefault("DSPY_CACHEDIR", str(ROOT / ".cache/prompt_comparison_state"))
    os.environ.setdefault("TEXTGRAD_LOG_DIR", str(ROOT / ".cache/textgrad_logs"))
    expected = {}
    for line in (ROOT / "requirements-prompt-comparison.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            if metadata.version(name) != version:
                raise ValueError(
                    f"Install pinned comparison dependency: {name}=={version}"
                )
            expected[name] = version
    return expected


def all_templates():
    values = {}
    for node in ast.parse((ROOT / pilot.PROMPTS).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for name in node.targets:
                if isinstance(name, ast.Name) and isinstance(node.value.value, str):
                    values[name.id] = node.value.value
    return values


def slot_parts(slot):
    if slot == "sql":
        return pilot.slot_parts()
    template = all_templates()["TAXONOMY_DETECTION_PROMPT"]
    head, rules = template.split("\n---\n", 1)
    rules, footer = rules.rsplit("\n---\n\n## Ответ", 1)
    return (
        head + "\n---\n",
        rules.replace("{db_schema}", "[DATABASE_SCHEMA]"),
        "\n---\n\n## Ответ" + footer,
    )


def candidate(instruction, demos=()):
    return {"instruction": instruction, "demos": list(demos)}


def cid(value):
    return search.fingerprint(value)


def item_id(item):
    return str(item["question_id"])


def check(value, manifest, slot, fit):
    if set(value) != {"instruction", "demos"}:
        raise ValueError("Only instruction and train demonstration IDs are mutable")
    text = value["instruction"]
    if not isinstance(text, str) or not 1 <= len(text) <= 12000:
        raise ValueError("Instruction length outside 1..12000")
    seed = manifest["seed_instructions"][slot]
    if text != seed:
        if slot == "sql":
            guard.corrected_check(
                text,
                {
                    "seed_instruction": seed,
                    "cohorts": {
                        "fit": fit,
                        "inner_val": manifest["cohorts"][slot]["inner_val"],
                    },
                },
            )
        else:
            if text.count("[DATABASE_SCHEMA]") != 1 or any(
                x in text for x in ("{", "}", "```", "{user_query}")
            ):
                raise ValueError(
                    "Ambiguity schema marker/output contract must be preserved"
                )
            folded = " ".join(text.lower().split())
            for item in fit + manifest["cohorts"][slot]["inner_val"]:
                words = item["question"].lower().split()
                if any(
                    " ".join(words[j : j + 8]) in folded
                    and " ".join(words[j : j + 8]) not in " ".join(seed.lower().split())
                    for j in range(max(0, len(words) - 7))
                ):
                    raise ValueError("Instruction copied a dataset question")
    demos = value["demos"]
    allowed = {item_id(i) for i in fit}
    if (
        not isinstance(demos, list)
        or len(demos) > 2
        or len(set(demos)) != len(demos)
        or any(not isinstance(q, str) or q not in allowed for q in demos)
    ):
        raise ValueError(
            "Demonstrations must be up to two distinct optimization-fit IDs"
        )
    return value


def fixed_demos(fit, slot, seed):
    ordered = sorted(fit, key=lambda i: search.fingerprint([seed, item_id(i)]))
    if slot == "ambiguity":
        positives = [i for i in ordered if i["SQL"] == "ambiguous"]
        negatives = [i for i in ordered if i["SQL"] != "ambiguous"]
        if not positives or not negatives:
            raise ValueError("Ambiguity train must include both labels")
        return [item_id(positives[0]), item_id(negatives[0])]
    first = ordered[0]
    second = next(i for i in ordered if i["db_id"] != first["db_id"])
    return [item_id(first), item_id(second)]


def render(value, manifest, slot, fit, item, *, profile="hybrid", original=False):
    check(value, manifest, slot, fit)
    if original:
        if value["demos"]:
            raise ValueError("Original control cannot include demonstrations")
        if slot == "sql":
            return pilot.compile_template(
                value["instruction"], profile, original=True
            ), []
        return all_templates()["TAXONOMY_DETECTION_PROMPT"], []
    head, _, footer = slot_parts(slot)
    rules = value["instruction"]
    if slot == "ambiguity":
        rules = rules.replace("[DATABASE_SCHEMA]", "{db_schema}")
    lookup = {item_id(i): i for i in fit}
    # Never reveal the current item's reference during fit evaluation.
    demos = [q for q in value["demos"] if q != item_id(item)]
    examples = []
    for q in demos:
        row = lookup[q]
        schema = manifest["schema_audit"]["schemas"][
            "bird" if slot == "sql" else "ambrosia"
        ][row["db_id"]]["rows"]
        answer = (
            row["SQL"]
            if slot == "sql"
            else ("ambiguous" if row["SQL"] == "ambiguous" else "unambiguous")
        )
        examples.append(
            {
                "question": row["question"],
                "evidence": row.get("evidence", ""),
                "schema": schema,
                "answer": answer,
            }
        )
    demo_text = (
        "\n\nTraining demonstrations (different questions; use current schema for current question):\n"
        + json.dumps(examples, ensure_ascii=False)
        if examples
        else ""
    )
    if len(demo_text.encode("utf-8")) > 40000:
        raise ValueError("Demonstrations exceed fixed 40 KB context cap")
    # These braces are data, not format slots of the runtime template.
    demo_text = demo_text.replace("{", "{{").replace("}", "}}")
    return head + rules + demo_text + footer, demos


def grouping_rows(items, benchmark):
    from configuration_splits import family_keys

    return [
        {
            "id": str(i["question_id"]),
            "benchmark": benchmark,
            "db_id": i["db_id"],
            "family_keys": family_keys(i, benchmark),
            "inputs": {"question": i["question"]},
        }
        for i in items
    ]


def group_split(items, benchmark):
    from configuration_splits import make_groups

    _, groups, _ = make_groups(grouping_rows(items, benchmark))
    if len(groups) < 2:
        raise ValueError("Need two independent database/question-family components")
    parts = {"fit": set(), "inner_val": set()}
    # Balance counts only; questions, labels and observed outcomes do not score splits.
    for group, ids in sorted(groups.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        part = min(parts, key=lambda p: (len(parts[p]), p))
        parts[part].update(ids)
    return {k: [i for i in items if item_id(i) in ids] for k, ids in parts.items()}


def make_manifest(env_file, teacher_env_file):
    versions = load_packages()
    parent = ROOT / pilot.source.DEFAULT_OUTPUT
    old = search.read(parent / "experiment.json")
    pilot.source.validate(old)
    model = pilot.public_model(dotenv_values(ROOT / env_file))
    if any(
        model[k] != old["protocol"]["model"][k]
        for k in ("model", "model_version", "base_url", "api_backend")
    ):
        raise ValueError("Keep the task model of the frozen DeepSeek controls")
    cohorts = {
        slot: group_split(old["cohorts"]["train"][b], b)
        for slot, b in (("sql", "bird"), ("ambiguity", "ambrosia"))
    }
    manifest = {
        "version": VERSION,
        "created_at_utc": search.now(),
        "env_file": env_file,
        "teacher_env_file": teacher_env_file,
        "model": model,
        "teacher": pilot.public_model(dotenv_values(ROOT / teacher_env_file)),
        "source": {
            "path": pilot.source.DEFAULT_OUTPUT,
            "sha256": pilot.digest(parent / "experiment.json"),
        },
        "files_sha256": {
            **old["protocol"]["files_sha256"],
            "scripts/configuration_splits.py": pilot.digest(
                ROOT / "scripts/configuration_splits.py"
            ),
            **{
                p: pilot.digest(ROOT / p)
                for p in (
                    *FILES,
                    "scripts/prompt_pilot.py",
                    "scripts/prompt_pilot_guardfix.py",
                    "scripts/prompt_pilot_runtime.py",
                    "scripts/prompt_pilot_budget.py",
                )
            },
        },
        "packages": versions,
        "runtime_environment": {
            **pilot.environment(),
            "openai": metadata.version("openai"),
        },
        "package_files_sha256": {
            str(p.relative_to(PACKAGES)): pilot.digest(p)
            for p in PACKAGES.rglob("*.py")
        },
        "cohorts": cohorts,
        "dev": old["cohorts"]["dev"],
        "schema_audit": old["schema_audit"],
        "seed_instructions": {s: slot_parts(s)[1] for s in SLOTS},
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "limits": LIMITS,
        "global_limits": GLOBAL_LIMITS,
        "jobs": [
            {"name": f"{s}__{m}__r{r}", "slot": s, "method": m, "seed": seed}
            for s in SLOTS
            for r, seed in enumerate(SEEDS, 1)
            for m in METHODS
        ],
        "policy": {
            "split": "All frozen train questions, connected DB/question-family components; balanced counts only; 3 optimizer seeds, same split",
            "search": "Fixed 14+32; only fit feedback/gold/demos reach teacher; inner_val scores select candidates",
            "fewshot": "MIPRO labeled fit demos only, max 2, no bootstrap; leave current fit question out",
            "ambiguity": "Optimize taxonomy rule body; existing hand-written examples retained in seed; zero means no added train demonstrations",
            "confirm": "Separate manual stage: original, fixed few-shot and 4 selected methods per seed/slot on primary external dev, fixed 14+32; no feedback to teacher",
            "confirmation_mask": [14, 32],
            "diagnostic": "Separate optional all-false matrix; ambiguity changes are inactive when FEAT_32=false",
            "selection": "Complete inner_val only, ties prefer seed, fewer demos, shorter text, hash",
            "cache": "Within job only; no task response sharing across optimizer repeats or methods",
            "resume": "Completed evaluations and teacher responses replayed; budgets persist; pending API reservations block resume",
            "data_status": "Previously viewed cohorts, not an independent final test",
            "search_design": {
                "gepa_minibatch": 4,
                "textgrad_steps": 12,
                "mipro_candidates": 8,
                "mipro_trials": 12,
            },
            "task_output_limit": 4096,
            "teacher_output_limits": {"sql": 4096, "ambiguity": 8192},
        },
    }
    manifest["sha256"] = search.fingerprint(manifest)
    validate(manifest)
    return manifest


def validate(manifest):
    if (
        manifest.get("version") != VERSION
        or manifest["sha256"]
        != search.fingerprint({k: v for k, v in manifest.items() if k != "sha256"})
        or manifest["limits"] != LIMITS
        or manifest["global_limits"] != GLOBAL_LIMITS
    ):
        raise ValueError("Frozen comparison protocol changed")
    if load_packages() != manifest["packages"]:
        raise ValueError("Pinned packages changed")
    if manifest["runtime_environment"] != {
        **pilot.environment(),
        "openai": metadata.version("openai"),
    }:
        raise ValueError("Task runtime package environment changed")
    for name, sha in manifest["files_sha256"].items():
        if pilot.digest(ROOT / name) != sha:
            raise ValueError(f"Frozen source changed: {name}; create a new experiment")
    for name, sha in manifest["package_files_sha256"].items():
        if pilot.digest(PACKAGES / name) != sha:
            raise ValueError("Optimizer package source changed")
    if (
        pilot.digest(ROOT / manifest["source"]["path"] / "experiment.json")
        != manifest["source"]["sha256"]
    ):
        raise ValueError("Source cohorts changed")
    from configuration_splits import make_groups

    for slot, c in manifest["cohorts"].items():
        fit = {i["db_id"] for i in c["fit"]}
        val = {i["db_id"] for i in c["inner_val"]}
        external = {i["db_id"] for items in manifest["dev"].values() for i in items}
        if fit & val or (fit | val) & external:
            raise ValueError("Train/inner/dev databases overlap")
        benchmark = "bird" if slot == "sql" else "ambrosia"
        _, groups, _ = make_groups(
            grouping_rows(
                c["fit"] + c["inner_val"] + manifest["dev"][benchmark], benchmark
            )
        )
        owner = {
            item_id(i): part
            for part, items in {**c, "dev": manifest["dev"][benchmark]}.items()
            for i in items
        }
        if any(len({owner[q] for q in ids}) > 1 for ids in groups.values()):
            raise ValueError("Database/question-family component crosses fit/inner/dev")
        for rows in c.values():
            if len({item_id(i) for i in rows}) != len(rows):
                raise ValueError("Duplicate question IDs")


def complete_score(
    records,
    value,
    items,
    phase="search",
    profile="hybrid",
    original=False,
    benchmark="bird",
):
    rows = [
        records.get(
            f"{phase}:{cid(value)}:{profile}:{original}:{benchmark}:{item_id(i)}"
        )
        for i in items
    ]
    return (
        sum(r["score"] for r in rows) / len(rows)
        if rows and all(r is not None for r in rows)
        else None
    )


def select(session):
    seed = candidate(session.manifest["seed_instructions"][session.slot])
    eligible = []
    for value in session.candidates.values():
        if session.job["method"] != "mipro_few" and value["demos"]:
            continue
        score = complete_score(
            session.records, value, session.inner, benchmark=session.benchmark
        )
        if score is not None:
            eligible.append(
                (
                    score,
                    value == seed,
                    -len(value["demos"]),
                    -len(value["instruction"]),
                    cid(value),
                    value,
                )
            )
    if (
        complete_score(
            session.records, seed, session.inner, benchmark=session.benchmark
        )
        is None
    ):
        raise ValueError("Original inner validation incomplete; selection forbidden")
    best = max(eligible, key=lambda r: r[:5])
    value = {
        "manifest_sha256": session.manifest["sha256"],
        "job": session.job,
        "candidate": best[-1],
        "inner_accuracy": best[0],
        "complete_candidates": len(eligible),
        "external_dev_used": False,
        "selected_at_utc": search.now(),
    }
    value["sha256"] = search.fingerprint(value)
    write_json(session.output / "selection.json", value)
    return value


def report(output, manifest):
    rows, summary = [], {}
    lines = [
        "# Prompt optimizer comparison",
        "",
        "Search selection uses grouped train only. Existing dev is retrospective; no global optimum claim.",
        "",
        "| Slot | Method | Repeat | Status | Original inner | Selected inner | Tokens | Calls task / teacher |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for job in manifest["jobs"]:
        folder = output / "jobs" / job["name"]
        meta = (
            search.read(folder / "search/metadata.json")
            if (folder / "search/metadata.json").exists()
            else {}
        )
        selection = (
            search.read(folder / "selection.json")
            if (folder / "selection.json").exists()
            else {}
        )
        records = (
            search.read(folder / "evaluations.json")
            if (folder / "evaluations.json").exists()
            else {}
        )
        budget = (
            search.read(folder / "search/budget.json")
            if (folder / "search/budget.json").exists()
            else {}
        )
        b = "bird" if job["slot"] == "sql" else "ambrosia"
        original = complete_score(
            records,
            candidate(manifest["seed_instructions"][job["slot"]]),
            manifest["cohorts"][job["slot"]]["inner_val"],
            benchmark=b,
        )
        selected = selection.get("inner_accuracy")
        if selected is not None:
            summary.setdefault((job["slot"], job["method"]), []).append(selected)

        def percent(v):
            return "—" if v is None else f"{100 * v:.2f}%"

        lines.append(
            f"| {job['slot']} | {job['method']} | {job['seed']} | {meta.get('status', 'pending')} | {percent(original)} | {percent(selected)} | {budget.get('tokens', 0)} | {budget.get('task_calls', 0)} / {budget.get('teacher_calls', 0)} |"
        )
        rows.append(
            {
                **job,
                "status": meta.get("status", "pending"),
                "original_inner": original,
                "selected_inner": selected,
                "tokens": budget.get("tokens", 0),
            }
        )
    lines += [
        "",
        "| Slot | Method | Completed repeats | Mean inner | SD across repeats |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for (slot, method), values in sorted(summary.items()):
        sd = f"{100 * statistics.stdev(values):.2f} pp" if len(values) > 1 else "—"
        lines.append(
            f"| {slot} | {method} | {len(values)}/3 | {100 * statistics.mean(values):.2f}% | {sd} |"
        )
    lines += [
        "",
        "SD describes optimizer/run variability on one fixed split, not uncertainty over new databases.",
        "Use confirmation_report.md after the separately launched dev phase.",
    ]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        output / "comparison.json",
        {"manifest_sha256": manifest["sha256"], "rows": rows},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", default=".env.deepseek")
    parser.add_argument("--teacher-env-file", default=".env.deepseek")
    parser.add_argument("--phase", choices=("search", "confirm"), default="search")
    parser.add_argument("--slot", choices=(*SLOTS, "all"), default="all")
    parser.add_argument("--method", choices=(*METHODS, "all"), default="all")
    parser.add_argument("--repeat", type=int, choices=(1, 2, 3))
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=1,
        help="New jobs per invocation; 1 is the deliberate first-run checkpoint, 24 runs the full search",
    )
    actions = parser.add_mutually_exclusive_group()
    for name in ("prepare", "dry-run", "status", "report"):
        actions.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    if args.max_jobs < 1:
        parser.error("--max-jobs must be positive")
    output = pilot.output_path(args.output)
    path = output / "experiment.json"
    if path.exists():
        manifest = search.read(path)
        validate(manifest)
    else:
        if output.exists() and any(output.iterdir()):
            parser.error("Output is non-empty without a manifest; choose a new folder")
        manifest = make_manifest(args.env_file, args.teacher_env_file)
    jobs = [
        j
        for j in manifest["jobs"]
        if (args.slot == "all" or args.slot == j["slot"])
        and (args.method == "all" or args.method == j["method"])
        and (not args.repeat or j["seed"] == SEEDS[args.repeat - 1])
    ]
    if args.dry_run or args.status:
        print(
            f"OFFLINE: {len(jobs)} matching jobs; run cap {args.max_jobs}; phase {args.phase}"
        )
        for slot, c in manifest["cohorts"].items():
            print(
                slot,
                {k: len(v) for k, v in c.items()},
                "databases",
                {k: len({i["db_id"] for i in v}) for k, v in c.items()},
            )
        print("Per job limits:", LIMITS, "\nWhole experiment limits:", GLOBAL_LIMITS)
        if args.status:
            for j in jobs:
                p = output / "jobs" / j["name"] / args.phase / "metadata.json"
                m = search.read(p) if p.exists() else {}
                print(
                    j["name"],
                    {
                        k: m.get(k)
                        for k in ("status", "phase", "saved_evaluations", "error_type")
                    },
                )
        return
    from run_feature_ablation import suite_lock

    output.mkdir(parents=True, exist_ok=True)
    with suite_lock(output):
        if not path.exists():
            write_json(path, manifest)
        if args.prepare or args.report:
            report(output, manifest)
            print("Offline preparation/report complete:", output)
            return
        from prompt_comparison_runtime import run_job, confirm

        try:
            if args.phase == "confirm":
                confirm(output, manifest)
            else:
                count = 0
                for job in jobs:
                    folder = output / "jobs" / job["name"]
                    if (folder / "selection.json").exists():
                        continue
                    if count >= args.max_jobs:
                        break
                    run_job(output, manifest, job)
                    count += 1
                print(
                    "Search checkpoint reached. Inspect comparison.md; rerun manually to continue."
                )
        finally:
            report(output, manifest)


if __name__ == "__main__":
    main()
