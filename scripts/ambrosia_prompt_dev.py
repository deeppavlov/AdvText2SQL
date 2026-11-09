"""Manual frozen Ambrosia dev confirmation of the completed prompt shortlist."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import statistics
import sys
from unittest.mock import patch

import bird_compact_transfer as shared
from research_metrics import ambrosia_report

source, store, read, digest = shared.source, shared.store, shared.read, shared.digest
VERSION = "ambrosia-prompt-dev-v1"
DEFAULT_OUTPUT = "research_results/ambrosia_prompt_dev_20261107"
FEATURES = [14, 32]
LIMITS = dict(
    examples=220, task_calls=800, teacher_calls=0, tokens=2_000_000, active_seconds=7200
)
GLOBAL_LIMITS = dict(
    examples=660,
    task_calls=2400,
    teacher_calls=0,
    tokens=5_000_000,
    active_seconds=21600,
)
SOURCES = (
    "ambiguity__mipro_zero__r1",
    "ambiguity__mipro_zero__r2",
    "ambiguity__mipro_zero__r3",
    "ambiguity__gepa_zero__r1",
    "ambiguity__textgrad_zero__r1",
    "ambiguity__mipro_few__r1",
)
FILES = (
    "scripts/ambrosia_prompt_dev.py",
    "local/run_ambrosia_prompt_dev.ps1",
    "local/run_ambrosia_prompt_dev.sh",
    *shared.FILES,
)


def output_path(value):
    path = Path(value)
    path = (source.ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if path.parent != (
        source.ROOT / "research_results"
    ).resolve() or not path.name.startswith("ambrosia_prompt_dev_"):
        raise ValueError(
            "Use a separate research_results/ambrosia_prompt_dev_<name> folder"
        )
    return path


def plan_jobs(old, selections):
    arms = {"original": source.candidate(old["seed_instructions"]["ambiguity"])}
    for name in SOURCES:
        selected = selections[name]
        _, method, repeat = name.split("__")
        expected = dict(
            name=name,
            slot="ambiguity",
            method=method,
            seed=source.SEEDS[int(repeat[1:]) - 1],
        )
        if (
            selected.get("manifest_sha256") != old["sha256"]
            or selected.get("external_dev_used") is not False
            or selected.get("job") != expected
            or selected.get("sha256")
            != source.search.fingerprint(
                {k: v for k, v in selected.items() if k != "sha256"}
            )
        ):
            raise ValueError("Unexpected source selection provenance")
        value = deepcopy(selected["candidate"])
        source.check(value, old, "ambiguity", old["cohorts"]["ambiguity"]["fit"])
        arms[f"{method}__{repeat}"] = value
    # Same fixed prompts in all repeats: these measure task variability, not
    # three optimizer restarts for methods that were searched only once.
    return [
        dict(
            name=f"ambrosia_dev__r{r}",
            slot="ambiguity",
            repeat=r,
            seed=seed,
            arms=deepcopy(arms),
        )
        for r, seed in enumerate(source.SEEDS, 1)
    ]


def load_selections(parent, old):
    selections, hashes = {}, {}
    for name in SOURCES:
        folder = parent / "jobs" / name
        meta = read(folder / "search/metadata.json")
        if (
            meta.get("status") != "completed"
            or meta.get("manifest_sha256") != old["sha256"]
        ):
            raise ValueError("Finish the six shortlisted source searches first")
        path = folder / "selection.json"
        selections[name] = read(path)
        for p in (path, folder / "search/metadata.json"):
            hashes[str(p.relative_to(parent))] = digest(p)
    return selections, hashes


def make_manifest(parent, env_file):
    old = shared.check_source(parent)
    selections, hashes = load_selections(parent, old)
    m = deepcopy(old)
    m.pop("sha256")
    m.pop("package_files_sha256", None)
    m.update(
        version=VERSION,
        created_at_utc=source.search.now(),
        source_search=dict(
            path=str(parent.relative_to(source.ROOT)),
            file_sha256=digest(parent / "experiment.json"),
            manifest_sha256=old["sha256"],
            files_sha256=hashes,
        ),
        env_file=env_file,
        teacher_env_file=env_file,
        feature_mask=FEATURES,
        limits=LIMITS,
        global_limits=GLOBAL_LIMITS,
        dev={"bird": [], "ambrosia": deepcopy(old["dev"]["ambrosia"])},
        jobs=plan_jobs(old, selections),
        files_sha256={
            **old["files_sha256"],
            **{name: digest(source.ROOT / name) for name in FILES},
        },
        policy=dict(
            task="Ambrosia strict ambiguity decision accuracy, F1/precision/recall/errors; not SQL execution accuracy",
            control="Original taxonomy prompt at static 14+32, not all-false",
            shortlist="All six completed ambiguity selections: three MIPRO-zero seeds and one seed for each other method; no dev-based inclusion",
            repeats="The same fixed candidates on the same 46 dev questions, three independent task repeats; not additional optimizer runs",
            cache="Identical candidates share answers only within one repeat; no search/BIRD/other-repeat response reuse",
            order="Unique candidates rotate cyclically across repeats, control is not always first",
            selection="After all three repeats: mean strict dev accuracy; ties prefer original, then frozen arm order. Deploy false; no feedback to teacher",
            data_status="Previously viewed grouped dev; retrospective, not an independent final test",
            generation="Original SQL generator/retries/schema context unchanged; only frozen taxonomy instruction differs; no teacher calls",
            recovery="Inherited fsync IO and bounded conservative transient task recovery; no budget resets, automatic billing/auth/timeout resumes",
        ),
    )
    m["sha256"] = source.search.fingerprint(m)
    return m


def validate(m):
    if m.get("version") != VERSION or m.get("sha256") != source.search.fingerprint(
        {k: v for k, v in m.items() if k != "sha256"}
    ):
        raise ValueError("Ambrosia dev manifest changed")
    if (
        m["feature_mask"] != FEATURES
        or m["limits"] != LIMITS
        or m["global_limits"] != GLOBAL_LIMITS
    ):
        raise ValueError("Frozen feature mask/budgets changed")
    parent = source.ROOT / m["source_search"]["path"]
    old = shared.check_source(parent)
    selections, hashes = load_selections(parent, old)
    if (
        digest(parent / "experiment.json") != m["source_search"]["file_sha256"]
        or old["sha256"] != m["source_search"]["manifest_sha256"]
        or hashes != m["source_search"]["files_sha256"]
        or m["jobs"] != plan_jobs(old, selections)
        or m["dev"] != {"bird": [], "ambrosia": old["dev"]["ambrosia"]}
        or any(
            m[field] != old[field]
            for field in (
                "cohorts",
                "schema_audit",
                "seed_instructions",
                "model",
                "teacher",
                "runtime_environment",
            )
        )
    ):
        raise ValueError("Frozen candidates/cohort changed")
    for name, expected in m["files_sha256"].items():
        if digest(source.ROOT / name) != expected:
            raise ValueError(
                f"Pinned runtime changed: {name}; prepare a separate output"
            )


def unique_candidates(job):
    values = list({source.cid(v): v for v in job["arms"].values()}.values())
    shift = (job["repeat"] - 1) % len(values)
    return values[shift:] + values[:shift]


class DevSession(shared.runtime.Session):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata.update(feature_mask=FEATURES, dev_confirmation_version=VERSION)

    def preflight(self):
        from resilient_hybrid_runtime import validate_live_schemas

        if self.manifest["dev"]["bird"] or self.slot != "ambiguity":
            raise ValueError("Only Ambrosia dev is authorized")
        validate_live_schemas(
            self.items,
            self.manifest["schema_audit"]["schemas"]["ambrosia"],
            progress=self.progress,
        )
        print(
            f"[PREFLIGHT] {len(self.items)} Ambrosia questions: live grouped schemas verified; no paid calls",
            flush=True,
        )

    async def prepare_tools(self, profile):
        with patch.dict(shared.base.pilot.PROFILES, {"hybrid": FEATURES}):
            await super().prepare_tools(profile)

    @contextmanager
    def apply(self, value, item, profile, original):
        with patch.dict(shared.base.pilot.PROFILES, {"hybrid": FEATURES}):
            with super().apply(value, item, profile, original) as demos:
                yield demos

    def reflect(self, prompt):
        raise RuntimeError("Teacher calls forbidden in dev confirmation")


def completed(root, job, m):
    path = root / "jobs" / job["name"] / "confirm/metadata.json"
    if not path.exists():
        if (path.parent.parent / "evaluations.json").exists():
            raise ValueError("Existing answers lack Ambrosia dev provenance")
        return False
    meta = read(path)
    if (
        meta.get("manifest_sha256") != m["sha256"]
        or meta.get("feature_mask") != FEATURES
        or meta.get("dev_confirmation_version") != VERSION
    ):
        raise ValueError("Foreign checkpoint/protocol")
    return meta.get("status") == "completed"


def run_job(root, m, job):
    if completed(root, job, m):
        return
    session = DevSession(root, m, job, "confirm")
    try:
        with (session.folder / "comparison.log").open("a", encoding="utf-8") as log:
            secrets = [os.environ.get("LLM_API_KEY", ""), os.environ.get("DB_PASS", "")]
            with patch.multiple(
                sys,
                stdout=shared.Tee(sys.stdout, log, secrets),
                stderr=shared.Tee(sys.stderr, log, secrets),
            ):
                session.preflight()
                for value in unique_candidates(job):
                    print(
                        f"[AMBROSIA DEV] {job['name']} candidate={source.cid(value)[:12]} features={FEATURES}",
                        flush=True,
                    )
                    for item in m["dev"]["ambrosia"]:
                        session.evaluate(item, value, benchmark="ambrosia")
                session.metadata.update(
                    status="completed",
                    phase="confirm",
                    finished_at_utc=source.search.now(),
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


def key(value, item):
    return f"confirm:{source.cid(value)}:hybrid:False:ambrosia:{source.item_id(item)}"


def report(root, m):
    rows, summary = [], []
    items = m["dev"]["ambrosia"]
    lines = [
        "# Ambrosia frozen prompt dev confirmation",
        "",
        "14+32; same fixed prompts and questions in three task repeats; no teacher.",
        "Accuracy measures ambiguity decisions, not SQL execution. Identical arms share results within a repeat.",
        "",
        "| Repeat | Arm | Saved | Accuracy | Delta, pp | Rescued / harmed | F1 | Recall | FP / FN | Errors |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def fmt(v):
        return "—" if v is None else f"{100 * v:.2f}%"

    for job in m["jobs"]:
        completed(root, job, m)
        path = root / "jobs" / job["name"] / "evaluations.json"
        records = read(path) if path.exists() else {}
        control = [records.get(key(job["arms"]["original"], i)) for i in items]
        baseline = (
            sum(r["score"] for r in control) / len(items) if all(control) else None
        )
        for arm, value in job["arms"].items():
            answers = [records.get(key(value, i)) for i in items]
            full = all(answers)
            metrics = (
                ambrosia_report(
                    items,
                    {
                        source.item_id(i): r["prediction"]
                        for i, r in zip(items, answers)
                    },
                )
                if full
                else None
            )
            accuracy = metrics["accuracy"] / 100 if full else None
            if (
                full
                and abs(accuracy - sum(r["score"] for r in answers) / len(items)) > 1e-9
            ):
                raise ValueError(
                    "Checkpoint scores differ from strict Ambrosia metrics"
                )
            amb = metrics["ambiguity_metrics"] if full else {}
            delta = accuracy - baseline if full and baseline is not None else None
            rescued = sum(
                r["score"] > c["score"] for r, c in zip(answers, control) if r and c
            )
            harmed = sum(
                r["score"] < c["score"] for r, c in zip(answers, control) if r and c
            )
            row = dict(
                repeat=job["repeat"],
                arm=arm,
                candidate_id=source.cid(value),
                saved=sum(r is not None for r in answers),
                accuracy=accuracy,
                delta=delta,
                rescued=rescued,
                harmed=harmed,
                metrics=metrics,
            )
            rows.append(row)
            lines.append(
                f"| {job['repeat']} | {arm} | {row['saved']}/{len(items)} | {fmt(accuracy)} | {'—' if delta is None else f'{100 * delta:+.2f}'} | {rescued} / {harmed} | {fmt(amb.get('f1'))} | {fmt(amb.get('recall'))} | {amb.get('fp', '—')} / {amb.get('fn', '—')} | {metrics['generation_errors'] if full else '—'} |"
            )
    lines += [
        "",
        "| Arm | Complete repeats | Mean accuracy | SD, pp | Mean delta, pp | Mean F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in m["jobs"][0]["arms"]:
        values = [r for r in rows if r["arm"] == arm and r["accuracy"] is not None]
        if not values:
            continue
        avg = statistics.mean(r["accuracy"] for r in values)
        sd = (
            statistics.stdev(r["accuracy"] for r in values) if len(values) > 1 else None
        )
        deltas = [r["delta"] for r in values if r["delta"] is not None]
        gain = statistics.mean(deltas) if deltas else None
        f1 = statistics.mean(r["metrics"]["ambiguity_metrics"]["f1"] for r in values)
        summary.append(
            dict(
                arm=arm,
                candidate_id=values[0]["candidate_id"],
                complete_repeats=len(values),
                mean_accuracy=avg,
                sd=sd,
                mean_delta=gain,
                mean_f1=f1,
            )
        )
        lines.append(
            f"| {arm} | {len(values)}/3 | {fmt(avg)} | {'—' if sd is None else f'{100 * sd:.2f}'} | {'—' if gain is None else f'{100 * gain:+.2f}'} | {fmt(f1)} |"
        )
    usage = read(root / "budget.json") if (root / "budget.json").exists() else {}
    lines += [
        "",
        f"Measured tokens: {usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)}; unknown usage allowance: {usage.get('unknown_usage_tokens', 0)}; accounted: {usage.get('tokens', 0)}.",
        "",
        "Retrospective dev; no independent-test claim or automatic deployment. Shared arms must not be counted as independent observations or duplicate costs.",
    ]
    store.write_bytes(root / "report.md", ("\n".join(lines) + "\n").encode("utf-8"))
    store.write_json(
        root / "report.json",
        dict(
            manifest_sha256=m["sha256"],
            rows=rows,
            summary=summary,
            authoritative_budget=usage,
        ),
    )
    if len(summary) == len(m["jobs"][0]["arms"]) and all(
        r["complete_repeats"] == 3 for r in summary
    ):
        order = list(m["jobs"][0]["arms"])
        winner = max(
            summary, key=lambda r: (r["mean_accuracy"], -order.index(r["arm"]))
        )
        store.write_json(
            root / "selection.json",
            dict(
                manifest_sha256=m["sha256"],
                selected_arm=winner["arm"],
                candidate=m["jobs"][0]["arms"][winner["arm"]],
                mean_dev_accuracy=winner["mean_accuracy"],
                policy=m["policy"]["selection"],
                deployed=False,
                independent_test=False,
            ),
        )


@contextmanager
def runtime_context():
    # Reuse the hash-pinned, tested accounting/worker supervisor without editing
    # either historical experiment. Children import this same entry point.
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
    parser.add_argument("--source", default=source.DEFAULT_OUTPUT)
    parser.add_argument("--env-file", default=".env.deepseek")
    for flag in ("prepare", "dry-run", "status", "report"):
        parser.add_argument("--" + flag, action="store_true")
    parser.add_argument("--repeat", type=int, choices=(1, 2, 3))
    parser.add_argument("--max-jobs", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--transient-resumes", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--jobs", nargs="+", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = output_path(args.output)
    path = root / "experiment.json"
    if path.exists():
        m = read(path)
    elif args.prepare or args.dry_run:
        m = make_manifest(source.pilot.output_path(args.source), args.env_file)
    else:
        raise ValueError("Prepare the separate dev manifest offline first")
    validate(m)
    if not (args.status or args.report):
        shared.check_current_model(m)
    jobs = [j for j in m["jobs"] if args.repeat is None or j["repeat"] == args.repeat]
    if args.jobs:
        if not args.worker or not set(args.jobs) <= {j["name"] for j in jobs}:
            raise ValueError("Unexpected worker job authorization")
        jobs = [j for j in jobs if j["name"] in args.jobs]
    if args.dry_run:
        print(
            f"OFFLINE Ambrosia dev: features={FEATURES}; questions={len(m['dev']['ambrosia'])}; teacher=0"
        )
        for j in jobs:
            print(
                j["name"],
                "arms:",
                len(j["arms"]),
                "unique evaluations:",
                len(unique_candidates(j)) * len(m["dev"]["ambrosia"]),
            )
        print("Per repeat:", LIMITS, "Whole NEW series:", GLOBAL_LIMITS)
        return 0
    root.mkdir(parents=True, exist_ok=True)
    if args.prepare or args.status or args.report:
        with shared.suite_lock(root):
            if not path.exists():
                store.write_json(path, m)
            if args.prepare or args.report:
                report(root, m)
            for j in jobs:
                print(j["name"], "completed" if completed(root, j, m) else "pending")
        return 0
    with runtime_context():
        if args.worker:
            shared.run_worker(root, m, jobs)
            return 0
        jobs = [j for j in jobs if not completed(root, j, m)][: args.max_jobs]
        return shared.supervise(root, m, jobs, args.transient_resumes) if jobs else 0


if __name__ == "__main__":
    sys.exit(main())
