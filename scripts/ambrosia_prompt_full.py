"""Manual fresh full Ambrosia control: 211 questions, original prompts, 14+32."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import sys
from unittest.mock import patch

import ambrosia_prompt_dev as dev
import bird_prompt_full as bird

shared, source, store, read, digest = (
    dev.shared,
    dev.source,
    dev.store,
    dev.read,
    dev.digest,
)
VERSION = "ambrosia-prompt-full-v1"
DEFAULT_OUTPUT = "research_results/ambrosia_prompt_full_20261107"
FEATURES = [14, 32]
LIMITS = dict(
    examples=250, task_calls=800, teacher_calls=0, tokens=2_000_000, active_seconds=7200
)
GLOBAL_LIMITS = dict(
    examples=250, task_calls=800, teacher_calls=0, tokens=3_000_000, active_seconds=7200
)
FILES = (
    "scripts/ambrosia_prompt_full.py",
    "local/run_ambrosia_prompt_full.ps1",
    "local/run_ambrosia_prompt_full.sh",
    *dev.FILES,
    *bird.FILES,
)


def output_path(value):
    p = Path(value)
    p = (source.ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    if p.parent != (
        source.ROOT / "research_results"
    ).resolve() or not p.name.startswith("ambrosia_prompt_full_"):
        raise ValueError(
            "Use a separate research_results/ambrosia_prompt_full_<name> folder"
        )
    return p


def original_templates():
    sql = source.pilot.templates()
    return {
        "SQL_PROMPT_TEMPLATE": sql["SQL_PROMPT_TEMPLATE"],
        "SQL_PROMPT_BASIC": sql["SQL_PROMPT_BASIC"],
        "TAXONOMY_DETECTION_PROMPT": source.all_templates()[
            "TAXONOMY_DETECTION_PROMPT"
        ],
    }


def load_dev(parent):
    old = read(parent / "experiment.json")
    dev.validate(old)
    selection = read(parent / "selection.json")
    seed = source.candidate(old["seed_instructions"]["ambiguity"])
    if (
        selection.get("manifest_sha256") != old["sha256"]
        or selection.get("selected_arm") != "original"
        or selection.get("candidate") != seed
    ):
        raise ValueError("Ambrosia dev must retain the exact original taxonomy")
    hashes = {"selection.json": digest(parent / "selection.json")}
    means = {}
    for job in old["jobs"]:
        if not dev.completed(parent, job, old):
            raise ValueError("Finish all three Ambrosia dev repeats first")
        p = parent / "jobs" / job["name"] / "evaluations.json"
        records = read(p)
        expected = {
            dev.key(v, i) for v in job["arms"].values() for i in old["dev"]["ambrosia"]
        }
        if set(records) != expected:
            raise ValueError("Ambrosia dev question IDs differ")
        for arm, value in job["arms"].items():
            rows = [records[dev.key(value, i)] for i in old["dev"]["ambrosia"]]
            metrics = dev.ambrosia_report(
                old["dev"]["ambrosia"],
                {
                    source.item_id(i): r["prediction"]
                    for i, r in zip(old["dev"]["ambrosia"], rows)
                },
            )
            means.setdefault(arm, []).append(metrics["accuracy"] / 100)
        for path in (p, p.parent / "confirm/metadata.json"):
            hashes[str(path.relative_to(parent))] = digest(path)
    averages = {arm: sum(values) / len(values) for arm, values in means.items()}
    if (
        max(averages.values()) > averages["original"] + 1e-9
        or abs(averages["original"] - selection["mean_dev_accuracy"]) > 1e-9
    ):
        raise ValueError("Dev choice differs from strict scores")
    return old, hashes


def load_full(parent, old):
    full = read(parent / "experiment.json")
    for name in ("protocol", "cohorts", "cases", "references"):
        if full[name + "_sha256"] != source.search.fingerprint(full[name]):
            raise ValueError("Full source fingerprints changed")
    if full["protocol"]["schema_audit_sha256"] != source.search.fingerprint(
        full["schema_audit"]
    ):
        raise ValueError("Full schema audit changed")
    items = full["cohorts"]["full"]["ambrosia"]
    if len(items) != 211 or len({source.item_id(i) for i in items}) != 211:
        raise ValueError("Expected exactly the agreed 211 Ambrosia questions")
    if (
        full["schema_audit"]["schemas"]["ambrosia"]
        != old["schema_audit"]["schemas"]["ambrosia"]
    ):
        raise ValueError("Full/dev schema audits differ")
    if any(
        full["protocol"]["model"][k] != old["model"][k]
        for k in ("model", "model_version", "base_url", "api_backend")
    ):
        raise ValueError("Historical model/provider differs")
    report = read(parent / "full_report.json")
    if report["protocol_sha256"] != full["protocol_sha256"]:
        raise ValueError("Historical report belongs to a different full protocol")
    matches = [r for r in report["rows"] if r["features"] == FEATURES]
    if len(matches) != 1:
        raise ValueError("Missing historical 14+32 control")
    history = matches[0]["benchmarks"]["ambrosia"]
    if history["status"] != "completed" or history["questions"] != 211:
        raise ValueError("Historical Ambrosia control incomplete")
    return items, {
        k: history[k] for k in ("correct", "questions", "accuracy", "ambiguity_metrics")
    }


def make_manifest(parent, full_parent, env_file):
    old, hashes = load_dev(parent)
    items, history = load_full(full_parent, old)
    m = deepcopy(old)
    m.pop("sha256")
    m.update(
        version=VERSION,
        created_at_utc=source.search.now(),
        dev_source=dict(
            path=str(parent.relative_to(source.ROOT)),
            manifest_file_sha256=digest(parent / "experiment.json"),
            files_sha256=hashes,
        ),
        full_source=dict(
            path=str(full_parent.relative_to(source.ROOT)),
            files_sha256={
                name: digest(full_parent / name)
                for name in ("experiment.json", "full_report.json")
            },
        ),
        full={"ambrosia": deepcopy(items)},
        historical_reference=history,
        fixed_prompts=original_templates(),
        env_file=env_file,
        teacher_env_file=env_file,
        feature_mask=FEATURES,
        limits=LIMITS,
        global_limits=GLOBAL_LIMITS,
        jobs=[
            dict(
                name="ambrosia_full__control",
                slot="ambiguity",
                repeat=1,
                seed=42,
                arms={
                    "original": source.candidate(old["seed_instructions"]["ambiguity"])
                },
            )
        ],
        files_sha256={
            **old["files_sha256"],
            **{name: digest(source.ROOT / name) for name in FILES},
        },
        policy=dict(
            task="Fresh single full Ambrosia control at 14+32; original SQL and taxonomy, no BIRD optimized instruction",
            metric="Strict ambiguity decision accuracy, F1/precision/recall/FP/FN/generation errors; not SQL EX",
            selection="Original retained by completed dev; no optimizer, full-driven selection or automatic deployment",
            data_status="Same 211 curated previously viewed full questions; historical 227 minus 16 schema/SQL inconsistencies; not independent test",
            runtime="Internal confirm phase is transport/budget compatibility only; full scope is mandatory and separate from unchanged dev cohort",
            preflight="Check all live Ambrosia schemas before paid calls; no BIRD connection or execution of reference SQL for ambiguity labels",
            cache="Fresh root only; never reuse historical full/search/dev responses",
            budget="New bounded ledgers, teacher forbidden; optional max two conservative journaled transient task resumes; no billing/auth/timeout auto-retry",
        ),
    )
    m["sha256"] = source.search.fingerprint(m)
    return m


def validate(m):
    if m.get("version") != VERSION or m.get("sha256") != source.search.fingerprint(
        {k: v for k, v in m.items() if k != "sha256"}
    ):
        raise ValueError("Ambrosia full manifest changed")
    parent = dev.output_path(m["dev_source"]["path"])
    old, hashes = load_dev(parent)
    full_parent = bird.full_source_path(m["full_source"]["path"])
    items, history = load_full(full_parent, old)
    job = dict(
        name="ambrosia_full__control",
        slot="ambiguity",
        repeat=1,
        seed=42,
        arms={"original": source.candidate(old["seed_instructions"]["ambiguity"])},
    )
    if (
        m["feature_mask"] != FEATURES
        or m["limits"] != LIMITS
        or m["global_limits"] != GLOBAL_LIMITS
        or m["dev_source"]["manifest_file_sha256"] != digest(parent / "experiment.json")
        or m["dev_source"]["files_sha256"] != hashes
        or m["full_source"]["files_sha256"]
        != {
            name: digest(full_parent / name)
            for name in ("experiment.json", "full_report.json")
        }
        or m["full"] != {"ambrosia": items}
        or m["historical_reference"] != history
        or m["fixed_prompts"] != original_templates()
        or m["jobs"] != [job]
        or any(
            m[k] != old[k]
            for k in (
                "cohorts",
                "dev",
                "schema_audit",
                "seed_instructions",
                "model",
                "teacher",
                "runtime_environment",
            )
        )
    ):
        raise ValueError("Frozen full cohort/prompts/protocol changed")
    for name, expected in m["files_sha256"].items():
        if digest(source.ROOT / name) != expected:
            raise ValueError(
                f"Pinned full runtime changed: {name}; prepare a new output"
            )


class FullSession(dev.DevSession):
    def __init__(self, root, m, job):
        view = {**m, "dev": {"bird": [], "ambrosia": m["full"]["ambrosia"]}}
        super().__init__(root, view, job, "confirm")
        self.metadata.pop("dev_confirmation_version", None)
        self.metadata.update(
            feature_mask=FEATURES,
            full_prompt_version=VERSION,
            evaluation_scope="full_retrospective",
        )

    @contextmanager
    def apply(self, value, item, profile, original):
        if value != source.candidate(self.manifest["seed_instructions"]["ambiguity"]):
            raise ValueError("Only the fixed original taxonomy is authorized")
        with super().apply(value, item, profile, original) as demos:
            with patch.multiple(
                self.impl,
                **{
                    k: self.manifest["fixed_prompts"][k]
                    for k in ("SQL_PROMPT_TEMPLATE", "SQL_PROMPT_BASIC")
                },
            ):
                yield demos


def completed(root, job, m):
    p = root / "jobs" / job["name"] / "confirm/metadata.json"
    if not p.exists():
        if (p.parent.parent / "evaluations.json").exists():
            raise ValueError("Checkpoint lacks full provenance")
        return False
    meta = read(p)
    if (
        meta.get("manifest_sha256") != m["sha256"]
        or meta.get("feature_mask") != FEATURES
        or meta.get("full_prompt_version") != VERSION
        or meta.get("evaluation_scope") != "full_retrospective"
    ):
        raise ValueError("Foreign checkpoint/scope")
    return meta.get("status") == "completed"


def run_job(root, m, job):
    if completed(root, job, m):
        return
    session = FullSession(root, m, job)
    try:
        with (session.folder / "comparison.log").open("a", encoding="utf-8") as log:
            secrets = [os.environ.get("LLM_API_KEY", ""), os.environ.get("DB_PASS", "")]
            with patch.multiple(
                sys,
                stdout=shared.Tee(sys.stdout, log, secrets),
                stderr=shared.Tee(sys.stderr, log, secrets),
            ):
                session.preflight()
                print(
                    "[AMBROSIA FULL] 211 fresh questions; original SQL/taxonomy; features=14+32",
                    flush=True,
                )
                for item in m["full"]["ambrosia"]:
                    session.evaluate(
                        item, job["arms"]["original"], benchmark="ambrosia"
                    )
                session.metadata.update(
                    status="completed", finished_at_utc=source.search.now()
                )
    except BaseException as error:
        session.metadata.update(
            status="stopped",
            error_type=type(error).__name__,
            finished_at_utc=source.search.now(),
        )
        raise
    finally:
        store.write_json(session.folder / "metadata.json", session.metadata)
        session.close()
        report(root, m)


def report(root, m):
    job = m["jobs"][0]
    completed(root, job, m)
    p = root / "jobs" / job["name"] / "evaluations.json"
    records = read(p) if p.exists() else {}
    value = job["arms"]["original"]
    items = m["full"]["ambrosia"]
    if not set(records) <= {dev.key(value, i) for i in items}:
        raise ValueError("Unexpected full checkpoint keys")
    results = [records.get(dev.key(value, i)) for i in items]
    metrics = (
        dev.ambrosia_report(
            items, {source.item_id(i): r["prediction"] for i, r in zip(items, results)}
        )
        if all(results)
        else None
    )
    if (
        metrics
        and abs(
            metrics["accuracy"] / 100 - sum(r["score"] for r in results) / len(items)
        )
        > 1e-9
    ):
        raise ValueError("Checkpoint scores differ from strict ambiguity metrics")
    usage = read(root / "budget.json") if (root / "budget.json").exists() else {}
    amb = metrics["ambiguity_metrics"] if metrics else {}

    def fmt(v):
        return "—" if v is None else f"{100 * v:.2f}%"

    row = dict(
        arm="original",
        saved=sum(r is not None for r in results),
        questions=len(items),
        accuracy=metrics["accuracy"] / 100 if metrics else None,
        metrics=metrics,
    )
    lines = [
        "# Ambrosia full: fresh original 14+32 control",
        "",
        "Same curated full cohort; original SQL and taxonomy, no teacher or prompt optimization.",
        "Accuracy is strict ambiguity decision accuracy, not SQL EX.",
        "",
        "| Saved | Accuracy | F1 | Precision | Recall | TP / FP / FN / TN | Generation errors |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {row['saved']}/{len(items)} | {fmt(row['accuracy'])} | {fmt(amb.get('f1'))} | {fmt(amb.get('precision'))} | {fmt(amb.get('recall'))} | {' / '.join(str(amb.get(k, '—')) for k in ('tp', 'fp', 'fn', 'tn'))} | {metrics['generation_errors'] if metrics else '—'} |",
        "",
        f"Historical same-profile full: {m['historical_reference']['correct']}/{len(items)} ({m['historical_reference']['accuracy']:.2f}%), separate pass; no old predictions reused.",
        f"Measured tokens: {usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)}; unknown allowance: {usage.get('unknown_usage_tokens', 0)}; accounted: {usage.get('tokens', 0)}.",
        "",
        "Retrospective single full control, not an independent test or optimized-vs-original comparison. No full-driven selection or automatic deployment.",
    ]
    store.write_bytes(root / "report.md", ("\n".join(lines) + "\n").encode("utf-8"))
    result = dict(
        manifest_sha256=m["sha256"],
        evaluation_scope="full_retrospective",
        row=row,
        historical_reference=m["historical_reference"],
        authoritative_budget=usage,
    )
    store.write_json(root / "report.json", result)
    if metrics:
        store.write_json(
            root / "conclusion.json",
            dict(
                **result,
                status="completed_descriptive",
                selection_on_full=False,
                deployed=False,
                independent_test=False,
            ),
        )


@contextmanager
def runtime_context():
    with patch.multiple(
        shared,
        VERSION=VERSION,
        FEATURES=FEATURES,
        __file__=str(Path(__file__).resolve()),
        completed=completed,
        run_job=run_job,
        report=report,
    ):
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--dev-source", default=dev.DEFAULT_OUTPUT)
    parser.add_argument("--full-source", default=bird.FULL_SOURCE)
    parser.add_argument("--env-file", default=".env.deepseek")
    for flag in ("prepare", "dry-run", "status", "report"):
        parser.add_argument("--" + flag, action="store_true")
    parser.add_argument("--transient-resumes", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--jobs", nargs="+", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = output_path(args.output)
    path = root / "experiment.json"
    if path.exists():
        m = read(path)
    elif args.prepare or args.dry_run:
        m = make_manifest(
            dev.output_path(args.dev_source),
            bird.full_source_path(args.full_source),
            args.env_file,
        )
    else:
        raise ValueError("Prepare separate full Ambrosia protocol offline first")
    validate(m)
    if not (args.status or args.report):
        shared.check_current_model(m)
    jobs = m["jobs"]
    if args.jobs and (not args.worker or set(args.jobs) != {j["name"] for j in jobs}):
        raise ValueError("Unexpected worker job authorization")
    if args.dry_run:
        print(
            f"OFFLINE Ambrosia full: {len(m['full']['ambrosia'])} fresh evaluations; features={FEATURES}; original SQL/taxonomy; teacher=0"
        )
        print("Local limits:", LIMITS, "Global:", GLOBAL_LIMITS)
        return 0
    root.mkdir(parents=True, exist_ok=True)
    if args.prepare or args.status or args.report:
        with shared.suite_lock(root):
            if not path.exists():
                store.write_json(path, m)
            if args.prepare or args.report:
                report(root, m)
            print(
                jobs[0]["name"],
                "completed" if completed(root, jobs[0], m) else "pending",
            )
        return 0
    with runtime_context():
        if args.worker:
            shared.run_worker(root, m, jobs)
            return 0
        return (
            shared.supervise(root, m, jobs, args.transient_resumes)
            if not completed(root, jobs[0], m)
            else 0
        )


if __name__ == "__main__":
    sys.exit(main())
