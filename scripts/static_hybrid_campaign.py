"""Bounded manual campaign over the guarded runtime; never advances phases automatically."""

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import random
import subprocess
import sys

from dotenv import dotenv_values

from atomic_artifacts import write_json
import guarded_hybrid_search as guarded
import static_hybrid_search as search
from run_feature_ablation import monitor, now, suite_lock

ROOT = search.ROOT
DEFAULT_OUTPUT = (
    "research_results/deepseek_combinations_large_20261026/final_search_20261030"
)
CONTROL = "hybrid_control_repeat"
BASELINE = "baseline_control"
ROUNDS = (1, 2, 3)
# Fifteen unique masks. Repeats have distinct checkpoint identities, not new masks.
GROUPS = {
    "controls": [BASELINE, CONTROL, "hybrid_plus_2"],
    "schema": ["core_without_tsv"],
    "statistics": ["statistics_compact", "statistics_types", "statistics_counts"],
    "repair": [
        "sanitize_validate",
        "validated_retry",
        "validated_feedback",
        "judged_feedback",
    ],
    "interactions": ["validated_voting", "routed_decomposition"],
    "combine": ["statistics_repair", "statistics_routing"],
}
SCREEN = [name for group in GROUPS.values() for name in group]


def environment():
    packages = {}
    for name in ("sqlglot", "SQLAlchemy", "pydantic", "autogen-core", "autogen-ext"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "packages": packages}


def alias(name, round_number):
    return f"{name}__r{round_number}"


def registry():
    source = {c["name"]: c for c in search.case_registry()}
    source[BASELINE] = {"features": []}
    source["statistics_repair"] = {"features": [2, 14, 15, 16, 17, 20, 27, 32, 35]}
    source["statistics_routing"] = {"features": [2, 14, 29, 30, 32, 35]}
    parents = {
        "statistics_compact": ["hybrid_plus_2"],
        "statistics_types": ["statistics_compact"],
        "statistics_counts": ["statistics_types"],
        "validated_retry": ["sanitize_validate"],
        "validated_feedback": ["validated_retry"],
        "judged_feedback": ["validated_feedback"],
        "validated_voting": ["sanitize_validate"],
        "statistics_repair": ["hybrid_plus_2", "judged_feedback"],
        "statistics_routing": ["hybrid_plus_2", "routed_decomposition"],
    }
    cases = []
    for round_number in ROUNDS:
        for name in SCREEN:
            flags = source[name]["features"]
            if flags:
                search.validate_features(flags)
            cases.append(
                {
                    "name": alias(name, round_number),
                    "features": flags,
                    "stage": "bounded_campaign",
                    "configuration": name,
                    "round": round_number,
                    "controls": [alias(p, round_number) for p in parents.get(name, [])],
                }
            )
    return cases


def make_manifest(config, audit_path, policy):
    manifest = guarded.make_manifest(config, audit_path, policy)
    manifest["cases"] = registry()
    manifest["cases_sha256"] = search.fingerprint(manifest["cases"])
    manifest["protocol"]["bounded_campaign"] = {
        "version": "bounded-static-v1",
        "groups": GROUPS,
        "rounds": list(ROUNDS),
        "screen_scope": "train",
        "finalist_limit": 3,
        "selection": "union of train macro/BIRD/Ambrosia leaders; fill to two by train macro",
        "tie_break": "exact score ties: fewer recorded tokens, fewer flags, name",
        "confirmation": "same frozen finalists and controls, three train and three dev runs",
        "test_policy": "no full/test scope; previously viewed test remains exploratory",
        "inference": "repeated paired deltas and descriptive database/repeat bootstrap; no global optimum claim",
    }
    manifest["protocol"]["environment"] = environment()
    for name in (
        "scripts/static_hybrid_campaign.py",
        "local/run_static_hybrid_campaign.ps1",
        "local/run_static_hybrid_campaign.sh",
    ):
        manifest["protocol"]["files_sha256"][name] = hashlib.sha256(
            (ROOT / name).read_bytes()
        ).hexdigest()
    manifest["protocol_sha256"] = search.fingerprint(manifest["protocol"])
    return manifest


def validate(manifest, *, code=True):
    guarded.validate(manifest, code=code)
    if (
        manifest["protocol"].get("bounded_campaign", {}).get("version")
        != "bounded-static-v1"
    ):
        raise ValueError("This folder is not a bounded campaign")
    if code and manifest["protocol"]["environment"] != environment():
        raise ValueError(
            "Python/package versions changed; do not mix campaign checkpoints"
        )


def case_for(manifest, name, round_number):
    return next(c for c in manifest["cases"] if c["name"] == alias(name, round_number))


def measurement(output, manifest, scope, name, round_number):
    case = case_for(manifest, name, round_number)
    values = {
        b: search.result(output, manifest, scope, case, b) for b in search.BENCHMARKS
    }
    complete = all(v["status"] == "completed" for v in values.values())
    return {
        "configuration": name,
        "round": round_number,
        "scope": scope,
        "features": case["features"],
        "completed": complete,
        "benchmarks": values,
        "macro_accuracy": sum(v["accuracy"] for v in values.values()) / 2
        if complete
        else None,
        "tokens": sum(
            v.get("prompt_tokens", 0) + v.get("completion_tokens", 0)
            for v in values.values()
        ),
        "calls": sum(v.get("recorded_calls", 0) for v in values.values()),
    }


def require_complete(output, manifest, scope, names, rounds=(1,)):
    for r in rounds:
        for name in names:
            value = measurement(output, manifest, scope, name, r)
            if not value["completed"]:
                raise ValueError(
                    f"Complete {scope}/{alias(name, r)} on BOTH benchmarks first"
                )


def screen_hashes(output, manifest):
    hashes = {}
    for name in SCREEN:
        for b in search.BENCHMARKS:
            folder = search.run_folder(output, "train", alias(name, 1), b)
            for filename in (
                "metadata.json",
                "report.json",
                "query_results.json",
                "raw_responses.jsonl",
            ):
                path = folder / filename
                hashes[path.relative_to(output).as_posix()] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    return hashes


def choose_finalists(rows):
    candidates = [
        r
        for r in rows
        if r["completed"] and r["configuration"] not in (BASELINE, CONTROL)
    ]
    if len(candidates) < 2:
        raise ValueError("At least two complete train candidates required")

    def key(row, metric):
        score = (
            row["macro_accuracy"]
            if metric == "macro"
            else row["benchmarks"][metric]["accuracy"]
        )
        return -score, row["tokens"], len(row["features"]), row["configuration"]

    winners = []
    for metric in ("macro", "bird", "ambrosia"):
        name = min(candidates, key=lambda r: key(r, metric))["configuration"]
        if name not in winners:
            winners.append(name)
    for row in sorted(candidates, key=lambda r: key(r, "macro")):
        if len(winners) >= 2:
            break
        if row["configuration"] not in winners:
            winners.append(row["configuration"])
    return winners


def finalists(output, manifest):
    path = output / "finalists.json"
    if not path.exists():
        raise ValueError("Finish the train screen, then run --freeze-finalists offline")
    saved = search.read(path)
    payload = {k: v for k, v in saved.items() if k != "sha256"}
    if (
        saved.get("sha256") != search.fingerprint(payload)
        or saved["protocol_sha256"] != manifest["protocol_sha256"]
    ):
        raise ValueError("Frozen finalist identity changed")
    require_complete(output, manifest, "train", SCREEN)
    rows = [measurement(output, manifest, "train", name, 1) for name in SCREEN]
    if saved["candidates"] != choose_finalists(rows) or saved[
        "source_sha256"
    ] != screen_hashes(output, manifest):
        raise ValueError(
            "Screen results changed after finalist freeze; do not reuse dev selection"
        )
    return [BASELINE, CONTROL, *saved["candidates"]]


def freeze_finalists(output, manifest):
    if (output / "finalists.json").exists():
        return finalists(output, manifest)
    # Dev cannot precede selection, including artifacts copied in by hand.
    if (output / "dev").exists():
        raise ValueError("Dev artifacts already exist before finalist freeze")
    require_complete(output, manifest, "train", SCREEN)
    rows = [measurement(output, manifest, "train", name, 1) for name in SCREEN]
    saved = {
        "created_at_utc": now(),
        "protocol_sha256": manifest["protocol_sha256"],
        "selection_scope": "train",
        "candidates": choose_finalists(rows),
        "source_sha256": screen_hashes(output, manifest),
    }
    saved["sha256"] = search.fingerprint(saved)
    write_json(output / "finalists.json", saved)
    return [BASELINE, CONTROL, *saved["candidates"]]


def jobs_for(output, manifest, phase, round_number):
    if phase in GROUPS:
        if round_number != 1:
            raise ValueError(
                "Screen phases use round 1; use --phase repeat for round 2/3"
            )
        if phase != "controls":
            require_complete(output, manifest, "train", GROUPS["controls"])
        if phase == "interactions":
            require_complete(output, manifest, "train", GROUPS["repair"])
        if phase == "combine":
            require_complete(
                output,
                manifest,
                "train",
                [n for n in SCREEN if n not in GROUPS["combine"]],
            )
        scope, names = "train", GROUPS[phase]
    else:
        names = finalists(output, manifest)
        if phase == "repeat":
            if round_number not in (2, 3):
                raise ValueError("Train repeats are round 2 and 3")
            require_complete(
                output, manifest, "train", names, rounds=range(1, round_number)
            )
            scope = "train"
        else:
            require_complete(output, manifest, "train", names, rounds=ROUNDS)
            require_complete(
                output, manifest, "dev", names, rounds=range(1, round_number)
            )
            scope = "dev"
        # Rotate and reverse the fixed list to reduce a fixed time/order advantage.
        names = list(names)
        if round_number % 2 == 0:
            names.reverse()
        offset = (round_number - 1) % len(names)
        names = names[offset:] + names[:offset]
    return scope, [case_for(manifest, n, round_number) for n in names]


def paired_repeats(values, controls, cohorts, *, samples=2000):
    """Descriptive bootstrap over paired repeat blocks and database clusters, not pooled rows."""
    if len(values) != len(controls) or len(values) < 3:
        raise ValueError("Three matched rounds are required")
    rounds = []
    dbs = {b: sorted({i["db_id"] for i in cohorts[b]}) for b in search.BENCHMARKS}
    for value, control in zip(values, controls):
        per_benchmark = {}
        for b in search.BENCHMARKS:
            a, c = (
                value["benchmarks"][b]["outcomes"],
                control["benchmarks"][b]["outcomes"],
            )
            expected = {str(i["question_id"]) for i in cohorts[b]}
            if set(a) != expected or set(c) != expected:
                raise ValueError(
                    "Paired comparison requires exactly the frozen cohort IDs"
                )
            per_benchmark[b] = {
                "delta_pp": 100
                * sum(int(a[q]) - int(c[q]) for q in expected)
                / len(expected),
                "rescued": sum(a[q] and not c[q] for q in expected),
                "harmed": sum(not a[q] and c[q] for q in expected),
                "by_database": {
                    db: [
                        int(a[str(i["question_id"])]) - int(c[str(i["question_id"])])
                        for i in cohorts[b]
                        if i["db_id"] == db
                    ]
                    for db in dbs[b]
                },
            }
        rounds.append(per_benchmark)
    rng = random.Random(20261030)
    distribution = []
    for _ in range(samples):
        picked_rounds = [rng.randrange(len(rounds)) for _ in rounds]
        means = []
        for b in search.BENCHMARKS:
            picked_dbs = rng.choices(dbs[b], k=len(dbs[b]))
            deltas = [
                d
                for r in picked_rounds
                for db in picked_dbs
                for d in rounds[r][b]["by_database"][db]
            ]
            means.append(100 * sum(deltas) / len(deltas))
        distribution.append(sum(means) / 2)
    distribution.sort()
    deltas = [sum(r[b]["delta_pp"] for b in search.BENCHMARKS) / 2 for r in rounds]
    return {
        "rounds": rounds,
        "mean_macro_delta_pp": sum(deltas) / len(deltas),
        "per_round_macro_delta_pp": deltas,
        "positive_rounds": sum(d > 0 for d in deltas),
        "descriptive_95_interval_pp": [
            distribution[int(0.025 * (samples - 1))],
            distribution[int(0.975 * (samples - 1))],
        ],
        "database_clusters": {b: len(v) for b, v in dbs.items()},
        "bootstrap_samples": samples,
        "limitation": "Few databases and three runs; descriptive interval, no multiple-comparison correction or global optimum proof.",
    }


def campaign_report(output, manifest, *, write=True):
    screen = [measurement(output, manifest, "train", name, 1) for name in SCREEN]
    by_name = {r["configuration"]: r for r in screen}
    for row in screen:
        row["comparisons"] = {}
        for control_name in (BASELINE, CONTROL):
            control = by_name[control_name]
            if not row["completed"] or not control["completed"]:
                continue
            row["comparisons"][control_name] = {
                "macro_delta_pp": row["macro_accuracy"] - control["macro_accuracy"],
                "benchmarks": {
                    b: {
                        "delta_pp": row["benchmarks"][b]["accuracy"]
                        - control["benchmarks"][b]["accuracy"],
                        "rescued": sum(
                            v and not control["benchmarks"][b]["outcomes"][q]
                            for q, v in row["benchmarks"][b]["outcomes"].items()
                        ),
                        "harmed": sum(
                            not v and control["benchmarks"][b]["outcomes"][q]
                            for q, v in row["benchmarks"][b]["outcomes"].items()
                        ),
                    }
                    for b in search.BENCHMARKS
                },
            }
    selected = (
        finalists(output, manifest) if (output / "finalists.json").exists() else []
    )
    report = {
        "protocol_sha256": manifest["protocol_sha256"],
        "screen": screen,
        "finalists_and_controls": selected,
        "replicated": {},
        "cohorts": {
            s: {b: len(v) for b, v in bs.items()}
            for s, bs in manifest["cohorts"].items()
            if s != "full"
        },
        "selection_status": "screening" if not selected else "confirmation_pending",
    }
    lines = [
        "# Ограниченная серия статического гибрида",
        "",
        "Только свежие guarded-прогоны; исторические ответы в рейтинг не входят.",
        "",
        "| Train, повтор 1 | Фичи | BIRD | Ambrosia | Средняя Accuracy | Δ к свежему H | F1 Ambrosia | Токены | Готово |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for value in screen:

        def cell(b):
            v = value["benchmarks"][b]
            return (
                f"{v['correct']}/{v['questions']} ({v['accuracy']:.2f}%)"
                if v["status"] == "completed"
                else f"{v['saved']}/{v['questions']}, {v['status']}"
            )

        score = (
            "—"
            if value["macro_accuracy"] is None
            else f"{value['macro_accuracy']:.2f}%"
        )
        delta = value["comparisons"].get(CONTROL, {}).get("macro_delta_pp")
        delta = "—" if delta is None else f"{delta:+.2f}"
        f1 = value["benchmarks"]["ambrosia"].get("ambiguity_metrics", {}).get("f1")
        f1 = "—" if f1 is None else f"{100 * f1:.2f}%"
        lines.append(
            f"| {value['configuration']} | {'+'.join(map(str, value['features'])) or 'все false'} | {cell('bird')} | {cell('ambrosia')} | {score} | {delta} | {f1} | {value['tokens']} | {value['completed']} |"
        )
    for scope in ("train", "dev"):
        rows = []
        for name in selected:
            values = [measurement(output, manifest, scope, name, r) for r in ROUNDS]
            if not all(v["completed"] for v in values):
                continue
            controls = [
                measurement(output, manifest, scope, CONTROL, r) for r in ROUNDS
            ]
            value = {
                "configuration": name,
                "features": values[0]["features"],
                "macro_accuracy_mean": sum(v["macro_accuracy"] for v in values) / 3,
                "macro_accuracy_range": [
                    min(v["macro_accuracy"] for v in values),
                    max(v["macro_accuracy"] for v in values),
                ],
                "tokens_per_round": sum(v["tokens"] for v in values) / 3,
                "benchmarks": {
                    b: {
                        "accuracy_mean": sum(
                            v["benchmarks"][b]["accuracy"] for v in values
                        )
                        / 3,
                        "f1_mean": sum(
                            v["benchmarks"][b].get("ambiguity_metrics", {}).get("f1", 0)
                            for v in values
                        )
                        / 3,
                    }
                    for b in search.BENCHMARKS
                },
            }
            if all(c["completed"] for c in controls):
                value["paired_vs_H"] = paired_repeats(
                    values, controls, manifest["cohorts"][scope]
                )
            rows.append(value)
        rows.sort(
            key=lambda v: (
                -v["macro_accuracy_mean"],
                v["tokens_per_round"],
                len(v["features"]),
                v["configuration"],
            )
        )
        report["replicated"][scope] = rows
        lines += [
            "",
            f"## {scope}: только три завершённых повтора",
            "",
            "| Конфигурация | BIRD, среднее | Ambrosia, среднее | F1 Ambrosia | Средняя Accuracy | Δ к H | Положительных повторов | Описательный 95% интервал Δ | Токены/повтор |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
        for value in rows:
            pair = value.get("paired_vs_H")
            delta = f"{pair['mean_macro_delta_pp']:+.2f}" if pair else "—"
            interval = (
                ", ".join(f"{x:+.2f}" for x in pair["descriptive_95_interval_pp"])
                if pair
                else "—"
            )
            positive = f"{pair['positive_rounds']}/3" if pair else "—"
            lines.append(
                f"| {value['configuration']} | {value['benchmarks']['bird']['accuracy_mean']:.2f}% | {value['benchmarks']['ambrosia']['accuracy_mean']:.2f}% | {100 * value['benchmarks']['ambrosia']['f1_mean']:.2f}% | {value['macro_accuracy_mean']:.2f}% | {delta} | {positive} | {interval} | {value['tokens_per_round']:.0f} |"
            )
    if selected and len(report["replicated"]["dev"]) == len(selected):
        report["selection_status"] = "bounded_confirmation_completed"
        report["best_observed_dev"] = report["replicated"]["dev"][0]["configuration"]
    lines += [
        "",
        f"Состояние: {report['selection_status']}.",
        "Отсутствие устойчивого выигрыша — допустимый итог. Лучший наблюдаемый набор не является глобальным максимумом.",
        "Dev уже используется для сравнения финалистов; независимый итоговый test этой серией не создаётся.",
        "Интервалы учитывают пары повторов и кластеры БД, но при малом числе БД остаются описательными.",
        "Сравнение включает итоговые ошибки pipeline; F1/FP/ошибки и стоимость нужно разбирать вместе с Accuracy.",
        "Папка ambrosia_227 — имя из переиспользуемого runtime; фактические ID/размеры заданы в experiment.json.",
        "",
    ]
    if write:
        write_json(output / "campaign_report.json", report)
        (output / "campaign_report.md").write_text(
            "\n".join(lines), encoding="utf-8", newline="\n"
        )
    return report


def run_jobs(output, manifest, scope, cases, args):
    write_json(
        output / "suite_status.json",
        {
            "status": "running",
            "phase": args.phase,
            "round": args.round,
            "at_utc": now(),
        },
    )
    try:
        for case in cases:
            for b in search.BENCHMARKS:
                if (
                    search.result(output, manifest, scope, case, b)["status"]
                    == "completed"
                ):
                    continue
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(ROOT / "scripts/guarded_hybrid_runtime.py"),
                        "--search",
                        str(output),
                        "--env-file",
                        args.env_file,
                        "--scope",
                        scope,
                        "--case",
                        case["name"],
                        "--benchmark",
                        b,
                    ],
                    cwd=ROOT,
                )
                monitor(
                    child, search.run_folder(output, scope, case["name"], b), b, args
                )
                if (
                    search.result(output, manifest, scope, case, b)["status"]
                    != "completed"
                ):
                    raise RuntimeError("Incomplete guarded child run")
                campaign_report(output, manifest)
        write_json(
            output / "suite_status.json",
            {
                "status": "phase_completed",
                "phase": args.phase,
                "round": args.round,
                "at_utc": now(),
            },
        )
    except BaseException as error:
        write_json(
            output / "suite_status.json",
            {
                "status": "stopped",
                "phase": args.phase,
                "error_type": type(error).__name__,
                "at_utc": now(),
            },
        )
        raise
    finally:
        campaign_report(output, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", default=".env.deepseek")
    parser.add_argument("--audit", default=guarded.DEFAULT_AUDIT)
    parser.add_argument(
        "--schema-policy", choices=("require_all", "exclude_mismatches")
    )
    parser.add_argument(
        "--phase", choices=(*GROUPS, "repeat", "dev"), default="controls"
    )
    parser.add_argument("--round", type=int, choices=ROUNDS, default=1)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--report", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--freeze-finalists", action="store_true")
    parser.add_argument("--stall-seconds", type=int, default=900)
    parser.add_argument("--error-limit", type=int, default=8)
    args = parser.parse_args()
    output, audit_path = (ROOT / args.output).resolve(), (ROOT / args.audit).resolve()
    if (
        not output.is_relative_to(ROOT / "research_results")
        or output == ROOT / "research_results"
        or output == (ROOT / DEFAULT_OUTPUT).parent
        or not audit_path.is_relative_to(ROOT / "research_results")
        or args.stall_seconds < 60
        or args.error_limit < 1
    ):
        parser.error(
            "Use a dedicated campaign folder/audit inside research_results; stall>=60, error-limit>=1"
        )
    path = output / "experiment.json"
    try:
        if path.exists():
            manifest = search.read(path)
            validate(manifest, code=not args.status)
            if (
                args.schema_policy
                and args.schema_policy != manifest["schema_audit"]["policy"]
            ):
                raise ValueError("Schema policy differs from frozen campaign")
        elif args.report or args.status or args.freeze_finalists:
            raise ValueError("Prepare the campaign first")
        else:
            manifest = make_manifest(
                dotenv_values(ROOT / args.env_file),
                audit_path,
                args.schema_policy or "require_all",
            )
        if args.status:
            report = campaign_report(output, manifest, write=False)
            print(
                json.dumps(
                    {
                        "state": report["selection_status"],
                        "completed_screen": sum(
                            r["completed"] for r in report["screen"]
                        ),
                        "screen_total": len(SCREEN),
                        "replicated": {
                            s: len(v) for s, v in report["replicated"].items()
                        },
                    }
                )
            )
            return
        if args.report or args.freeze_finalists:
            with suite_lock(output):
                if args.freeze_finalists:
                    print(
                        "Frozen candidates and controls:",
                        freeze_finalists(output, manifest),
                    )
                campaign_report(output, manifest)
            print("Offline report:", output / "campaign_report.md")
            return
        config = dotenv_values(ROOT / args.env_file)
        search.validate_connection(config)
        model = manifest["protocol"]["model"]
        if (
            config["LLM_MODEL_NAME"] != model["model"]
            or config.get("LLM_MODEL_VERSION") != model["model_version"]
            or config["LLM_BASE_URL"].rstrip("/") != model["base_url"]
        ):
            raise ValueError("Env model differs from the frozen campaign")
        scope, cases = jobs_for(output, manifest, args.phase, args.round)
        guarded.gate(manifest, scope, search.BENCHMARKS)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    counts = {b: len(manifest["cohorts"][scope][b]) for b in search.BENCHMARKS}
    pending = sum(
        v["questions"] - v["saved"]
        for c in cases
        for b in search.BENCHMARKS
        if (v := search.result(output, manifest, scope, c, b))["status"] != "completed"
    )
    print(
        f"Phase={args.phase}; round={args.round}; scope={scope}; schema-policy={manifest['schema_audit']['policy']}"
    )
    print("Questions:", counts, "remaining question predictions:", pending)
    print("Cases:", [(c["name"], c["features"]) for c in cases])
    if args.dry_run:
        print("DRY RUN: no files, SSH, DB or API calls. No automatic next phase.")
        return
    output.mkdir(parents=True, exist_ok=True)
    with suite_lock(output):
        if path.exists() and search.read(path) != manifest:
            raise ValueError("Manifest changed while acquiring lock")
        if not path.exists():
            if any(p.name != "suite.lock" for p in output.iterdir()):
                raise ValueError(
                    "Output contains artifacts without a campaign manifest"
                )
            write_json(path, manifest)
        campaign_report(output, manifest)
        if args.prepare:
            print("PREPARED offline; no paid calls. Next: manual controls launch.")
            return
        run_jobs(output, manifest, scope, cases, args)


if __name__ == "__main__":
    main()
