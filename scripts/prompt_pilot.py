"""Frozen, manually launched GEPA zero-shot SQL pilot; preparation is offline."""

import argparse
import ast
import hashlib
import importlib.metadata
from pathlib import Path
import re
import sys

from dotenv import dotenv_values

from atomic_artifacts import write_json
import static_hybrid_search as search
import transport_safe_campaign as source
from static_hybrid_campaign import environment

ROOT = search.ROOT
DEFAULT_OUTPUT = "research_results/prompt_optimization_20261105"
GEPA_VERSION = "0.0.27"
GEPA_PATH = ROOT / ".cache/prompt_optimizer"
FILES = (
    "scripts/prompt_pilot.py",
    "scripts/prompt_pilot_runtime.py",
    "scripts/prompt_pilot_budget.py",
    "local/run_prompt_pilot.ps1",
    "local/run_prompt_pilot.sh",
    "requirements-prompt-optimization.txt",
)
PROMPTS = "src/adv_text2sql/mcp_servers/text2sql_tool/src/prompts.py"
PROFILES = {"baseline": [], "hybrid": [14, 32]}
LIMITS = {
    "search": {
        "examples": 160,
        "task_calls": 400,
        "teacher_calls": 4,
        "tokens": 2_000_000,
        "active_seconds": 5400,
    },
    "confirm": {
        "examples": 404,
        "task_calls": 1000,
        "teacher_calls": 0,
        "tokens": 4_000_000,
        "active_seconds": 7200,
    },
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def templates():
    tree = ast.parse((ROOT / PROMPTS).read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for name in node.targets:
                if isinstance(name, ast.Name) and name.id in (
                    "SQL_PROMPT_TEMPLATE",
                    "SQL_PROMPT_BASIC",
                ):
                    values[name.id] = ast.literal_eval(node.value)
    return values


def slot_parts():
    strict = templates()["SQL_PROMPT_TEMPLATE"]
    head, tail = strict.split("\n\nТребования:\n", 1)
    dialect, tail = tail.split("\n", 1)
    rules, footer = tail.rsplit("\nНЕ ОФОРМЛЯЙ", 1)
    return head + "\n\nТребования:\n" + dialect + "\n", rules, "\nНЕ ОФОРМЛЯЙ" + footer


def check_candidate(text, manifest=None):
    if not isinstance(text, str) or not text.strip() or len(text) > 6000:
        raise ValueError("Instruction must contain 1..6000 characters")
    if any(x in text for x in ("{", "}", "```", "Запрос:", "SQL запрос:")):
        raise ValueError(
            "Instruction cannot change protected slots or contain fenced examples"
        )
    if manifest and text != manifest["seed_instruction"]:
        if re.search(r"\bSELECT\s+[\s\S]{0,300}?\bFROM\s+[\w\"]", text, re.I):
            raise ValueError("Zero-shot instruction must not contain SQL examples")
        rows = manifest["cohorts"]["fit"] + manifest["cohorts"]["inner_val"]
        if any(
            re.search(r"\b" + re.escape(i["db_id"]) + r"\b", text, re.I) for i in rows
        ):
            raise ValueError("Instruction must not name training databases")
        folded = " ".join(text.lower().split())
        for item in rows:
            words = item["question"].lower().split()
            if any(
                " ".join(words[j : j + 8]) in folded
                for j in range(max(0, len(words) - 7))
            ):
                raise ValueError("Instruction must not copy training questions")
    return text


def compile_template(instruction, profile, *, original=False):
    check_candidate(instruction)
    values = templates()
    if original:
        return values[
            "SQL_PROMPT_TEMPLATE" if profile == "hybrid" else "SQL_PROMPT_BASIC"
        ]
    head, _, footer = slot_parts()
    return head + instruction + footer


def output_path(value):
    path = (ROOT / value).resolve()
    research = (ROOT / "research_results").resolve()
    if path.parent != research or not path.name.startswith("prompt_optimization_"):
        raise ValueError(
            "Use a dedicated research_results/prompt_optimization_<name> folder"
        )
    return path


def load_gepa():
    if not GEPA_PATH.is_dir():
        raise ValueError(
            "Install requirements-prompt-optimization.txt into .cache/prompt_optimizer first"
        )
    sys.path.insert(0, str(GEPA_PATH))
    if importlib.metadata.version("gepa") != GEPA_VERSION:
        raise ValueError("Unexpected GEPA version")
    import gepa

    return gepa


def public_model(config):
    from llm_endpoints import validate_connection, client_options

    backend = validate_connection(config)
    return {
        "model": config["LLM_MODEL_NAME"],
        "model_version": config.get("LLM_MODEL_VERSION"),
        "base_url": config["LLM_BASE_URL"].rstrip("/"),
        "api_backend": backend,
        "request_options": client_options(backend),
        "temperature": 0,
        "max_tokens": 2048,
    }


def make_manifest(env_file, teacher_env_file):
    parent = ROOT / source.DEFAULT_OUTPUT
    old = search.read(parent / "experiment.json")
    source.validate(old)
    if not (parent / "static_hybrid_conclusion.json").exists():
        raise ValueError("Complete static confirmation first")
    conclusion = search.read(parent / "static_hybrid_conclusion.json")
    if (
        conclusion["sha256"]
        != search.fingerprint({k: v for k, v in conclusion.items() if k != "sha256"})
        or conclusion["best_observed_static"]["features"] != PROFILES["hybrid"]
    ):
        raise ValueError("The selected static control differs from the agreed 14+32")
    model = old["protocol"]["model"]
    task = public_model(dotenv_values(ROOT / env_file))
    if any(
        task[k] != model[k]
        for k in ("model", "model_version", "base_url", "api_backend")
    ):
        raise ValueError("Task model must match the frozen DeepSeek controls")
    train = old["cohorts"]["train"]["bird"]
    dbs = sorted({i["db_id"] for i in train})
    if len(dbs) != 4:
        raise ValueError("Review pilot grouping: expected four BIRD training databases")
    # Select without consulting outcomes; spread the 12 questions/DB over difficulties.
    selected = []
    for db in dbs:
        groups = {}
        for item in train:
            if item["db_id"] == db:
                groups.setdefault(item["difficulty"], []).append(item)
        groups = {
            k: sorted(v, key=lambda i: search.fingerprint([42, i["question_id"]]))
            for k, v in groups.items()
        }
        for _ in range(12):
            key = sorted(
                (k for k, v in groups.items() if v),
                key=lambda k: (
                    sum(i["difficulty"] == k and i["db_id"] == db for i in selected),
                    k,
                ),
            )[0]
            selected.append(groups[key].pop(0))
    cohorts = {
        "fit": [i for i in selected if i["db_id"] in dbs[:2]],
        "inner_val": [i for i in selected if i["db_id"] in dbs[2:]],
        "dev": old["cohorts"]["dev"],
    }
    code = dict(old["protocol"]["files_sha256"])
    code.update({p: digest(ROOT / p) for p in FILES})
    package = {
        str(p.relative_to(GEPA_PATH)): digest(p)
        for p in GEPA_PATH.rglob("*.py")
        if "gepa" in p.parts
    }
    manifest = {
        "version": "gepa-sql-pilot-v1",
        "created_at_utc": search.now(),
        "source": {
            "path": source.DEFAULT_OUTPUT,
            "manifest_sha256": digest(parent / "experiment.json"),
            "conclusion_sha256": digest(parent / "static_hybrid_conclusion.json"),
        },
        "model": model,
        "teacher": public_model(dotenv_values(ROOT / teacher_env_file)),
        "env_file": env_file,
        "teacher_env_file": teacher_env_file,
        "gepa_version": GEPA_VERSION,
        "environment": {
            **environment(),
            "openai": importlib.metadata.version("openai"),
        },
        "gepa_files_sha256": package,
        "files_sha256": code,
        "cohorts": cohorts,
        "profiles": PROFILES,
        "seed": 42,
        "seed_instruction": slot_parts()[1],
        "limits": LIMITS,
        "schema_audit": old["schema_audit"],
        "policy": {
            "mutable": "SQL rules only; question, dialect, schema, output carrier protected",
            "search": "GEPA Pareto, minibatch 3, 24 fit + 24 inner validation, DB-disjoint; only fit reflection",
            "selection": "max complete inner-val EX; ties prefer seed, shorter instruction, SHA",
            "confirmation": "manual separate 2x2 old/new prompt x all-false/14+32 on dev; no reflection",
            "data_status": "retrospective viewed cohorts, not independent final test",
            "sdk_retries": 0,
            "task_output_limit": 4096,
            "teacher_output_limit": 2048,
            "budget": "reserve conservative UTF-8 bytes + output cap + 1024 before call; unknown failed calls keep reservation",
            "cache": "first completed candidate/profile/question result reused; failed model answers included",
            "gold": "schema and all BIRD references checked read-only before any paid call in phase",
        },
    }
    manifest["sha256"] = search.fingerprint(manifest)
    return manifest


def validate(manifest):
    if manifest["sha256"] != search.fingerprint(
        {k: v for k, v in manifest.items() if k != "sha256"}
    ):
        raise ValueError("Pilot manifest changed")
    if (
        manifest["version"] != "gepa-sql-pilot-v1"
        or manifest["profiles"] != PROFILES
        or manifest["limits"] != LIMITS
    ):
        raise ValueError("Pilot policy changed")
    for name, sha in manifest["files_sha256"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"Frozen file changed: {name}; use a new pilot folder")
    if (
        digest(ROOT / manifest["source"]["path"] / "experiment.json")
        != manifest["source"]["manifest_sha256"]
    ):
        raise ValueError("Frozen source experiment changed")
    if (
        digest(ROOT / manifest["source"]["path"] / "static_hybrid_conclusion.json")
        != manifest["source"]["conclusion_sha256"]
    ):
        raise ValueError("Frozen static conclusion changed")
    if manifest["environment"] != {
        **environment(),
        "openai": importlib.metadata.version("openai"),
    }:
        raise ValueError("Python/package environment changed")
    for name, sha in manifest["gepa_files_sha256"].items():
        if digest(GEPA_PATH / name) != sha:
            raise ValueError("GEPA installation changed")
    c = manifest["cohorts"]
    sets = [{i["db_id"] for i in c[k]} for k in ("fit", "inner_val")]
    if len(c["fit"]) != 24 or len(c["inner_val"]) != 24 or sets[0] & sets[1]:
        raise ValueError("Pilot must use 24+24 DB-disjoint training examples")
    train_dbs = sets[0] | sets[1]
    if any(train_dbs & {i["db_id"] for i in items} for items in c["dev"].values()):
        raise ValueError("External dev overlaps training databases")


def report(output, manifest):
    from research_metrics import ambrosia_report

    selection = (
        search.read(output / "selection.json")
        if (output / "selection.json").exists()
        else None
    )
    records = (
        search.read(output / "evaluations.json")
        if (output / "evaluations.json").exists()
        else {}
    )
    states = {
        phase: search.read(output / phase / "metadata.json").get("status", "pending")
        if (output / phase / "metadata.json").exists()
        else "pending"
        for phase in LIMITS
    }
    rows_out = []
    lines = [
        "# GEPA: zero-shot SQL pilot",
        "",
        f"Search: {states['search']}; confirmation: {states['confirm']}.",
        "",
        "Fit: 24 questions / 2 DB; inner validation: 24 / 2 other DB. External dev: BIRD 55 / Ambrosia 46.",
        "",
        "| Prompt | Flags | BIRD dev | Ambrosia dev | F1 Ambrosia | Task tokens |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for original in (True, False):
        for profile in PROFILES:
            text = (
                manifest["seed_instruction"]
                if original or not selection
                else selection["instruction"]
            )
            cid = search.fingerprint(
                {"instruction": text, "profile": profile, "original": original}
            )
            values = []
            metrics = None
            tokens = 0
            results = {}
            for b, items in manifest["cohorts"]["dev"].items():
                rows = [
                    records.get(f"confirm:{cid}:{b}:{i['question_id']}") for i in items
                ]
                values.append(
                    f"{100 * sum(r['score'] for r in rows) / len(rows):.2f}%"
                    if all(r is not None for r in rows)
                    else f"{sum(r is not None for r in rows)}/{len(rows)}, pending"
                )
                tokens += sum(
                    sum(
                        r.get("usage", {}).get(k, 0)
                        for k in ("prompt_tokens", "completion_tokens")
                    )
                    for r in rows
                    if r is not None
                )
                results[b] = {
                    "saved": sum(r is not None for r in rows),
                    "questions": len(rows),
                    "accuracy": 100 * sum(r["score"] for r in rows) / len(rows)
                    if all(r is not None for r in rows)
                    else None,
                    "generation_errors": sum(
                        r["prediction"] == "error" for r in rows if r is not None
                    ),
                }
                if b == "ambrosia" and all(r is not None for r in rows):
                    metrics = ambrosia_report(
                        items, {r["question_id"]: r["prediction"] for r in rows}
                    )["ambiguity_metrics"]
            f1 = f"{100 * metrics['f1']:.2f}%" if metrics is not None else "—"
            lines.append(
                f"| {'original' if original else 'selected'} | {profile} | {' | '.join(values)} | {f1} | {tokens} |"
            )
            rows_out.append(
                {
                    "original": original,
                    "profile": profile,
                    "candidate_id": cid,
                    "benchmarks": results,
                    "ambiguity_metrics": metrics,
                    "task_tokens": tokens,
                }
            )
    if selection:
        lines += [
            "",
            f"Selected on internal train validation only: {selection['inner_val_correct']}/24; seed: {selection['seed_correct']}/24.",
            "External dev is never sent to reflection. Deployment and utility labels are unchanged.",
        ]
    for phase in LIMITS:
        path = output / phase / "budget.json"
        if path.exists():
            b = search.read(path)
            lines += [
                "",
                f"{phase}: task calls {b['task_calls']}; teacher calls {b['teacher_calls']}; actual tokens {b['tokens']}; unreconciled reservations {sum(x['reserved'] for x in b['pending'].values())}.",
            ]
    (output / "pilot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        output / "pilot_report.json",
        {"manifest_sha256": manifest["sha256"], "states": states, "rows": rows_out},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", default=".env.deepseek")
    parser.add_argument("--teacher-env-file", default=".env.deepseek")
    parser.add_argument("--phase", choices=("search", "confirm"), default="search")
    actions = parser.add_mutually_exclusive_group()
    for name in ("prepare", "dry-run", "status", "report"):
        actions.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    output = output_path(args.output)
    path = output / "experiment.json"
    if path.exists():
        manifest = search.read(path)
        validate(manifest)
    else:
        load_gepa()
        if output.exists() and any(output.iterdir()):
            parser.error("Output has files but no manifest; choose a new pilot folder")
        manifest = make_manifest(args.env_file, args.teacher_env_file)
    if args.dry_run or args.status:
        print(
            "OFFLINE: GEPA SQL pilot; phase",
            args.phase,
            "; 24 fit / 24 inner-val; dev 55/46, two masks x two prompts",
        )
        print("Limits:", manifest["limits"][args.phase])
        print(
            "Selection:",
            "frozen" if (output / "selection.json").exists() else "not started",
        )
        if args.status:
            for phase in LIMITS:
                meta = output / phase / "metadata.json"
                budget = output / phase / "budget.json"
                if meta.exists():
                    m = search.read(meta)
                    print(
                        phase,
                        {
                            k: m.get(k)
                            for k in (
                                "status",
                                "phase",
                                "question_id",
                                "saved_evaluations",
                            )
                        },
                    )
                if budget.exists():
                    b = search.read(budget)
                    print(
                        phase,
                        {
                            k: b[k]
                            for k in (
                                "examples",
                                "task_calls",
                                "teacher_calls",
                                "tokens",
                                "active_seconds",
                            )
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
        from prompt_pilot_runtime import run

        try:
            run(output, manifest, args.phase)
        finally:
            report(output, manifest)


if __name__ == "__main__":
    main()
