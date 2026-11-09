"""Manual paired full BIRD comparison of original and a dev-frozen SQL prompt."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import sys
from unittest.mock import patch

import bird_compact_transfer as shared

source, store, read, digest = shared.source, shared.store, shared.read, shared.digest
VERSION = "bird-prompt-full-v1"
DEFAULT_OUTPUT = "research_results/bird_prompt_full_20261107"
FULL_SOURCE = "research_results/deepseek_full_static_20261103"
FEATURES = [2, 7, 14, 32, 35]
LIMITS = dict(
    examples=550,
    task_calls=2000,
    teacher_calls=0,
    tokens=8_000_000,
    active_seconds=14400,
)
GLOBAL_LIMITS = dict(
    examples=550,
    task_calls=2000,
    teacher_calls=0,
    tokens=9_000_000,
    active_seconds=14400,
)
FILES = (
    "scripts/bird_prompt_full.py",
    "local/run_bird_prompt_full.ps1",
    "local/run_bird_prompt_full.sh",
    *shared.FILES,
)


def output_path(value):
    p = Path(value)
    p = (source.ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    if p.parent != (
        source.ROOT / "research_results"
    ).resolve() or not p.name.startswith("bird_prompt_full_"):
        raise ValueError(
            "Use a separate research_results/bird_prompt_full_<name> folder"
        )
    return p


def full_source_path(value):
    p = Path(value)
    p = (source.ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    if p.parent != (
        source.ROOT / "research_results"
    ).resolve() or not p.name.startswith("deepseek_full_static_"):
        raise ValueError(
            "Use the frozen research_results/deepseek_full_static_<name> manifest"
        )
    return p


def choose_candidate(m, records, selection):
    """Freeze one concrete candidate using only completed compact dev evidence."""
    if (
        selection.get("manifest_sha256") != m["sha256"]
        or selection.get("selected_arm") != "mipro_few"
    ):
        raise ValueError("Expected completed compact dev winner MIPRO few mode")
    order = list(m["jobs"][0]["arms"])
    means = {}
    for arm in order:
        scores = []
        for j in m["jobs"]:
            values = [
                records[j["name"]][shared.key(j["arms"][arm], i)]
                for i in m["dev"]["bird"]
            ]
            scores.append(sum(v["score"] for v in values) / len(values))
        means[arm] = sum(scores) / len(scores)
    winner = max(order, key=lambda arm: (means[arm], -order.index(arm)))
    if (
        winner != selection["selected_arm"]
        or abs(means[winner] - selection["mean_dev_accuracy"]) > 1e-9
    ):
        raise ValueError("Compact dev selection differs from checkpoints")
    ranked = []
    for j in m["jobs"]:
        value = j["arms"][winner]
        results = [records[j["name"]][shared.key(value, i)] for i in m["dev"]["bird"]]
        correct = sum(r["score"] for r in results)
        ranked.append(
            (
                correct,
                -len(value["demos"]),
                -len(value["instruction"]),
                -j["repeat"],
                j,
                value,
            )
        )
    correct, _, _, _, job, value = max(ranked, key=lambda row: row[:4])
    return dict(
        policy="New decision frozen before full: keep dev-winning arm by three-source mean, then choose its concrete candidate by highest compact dev EX; ties fewer demos, shorter instruction, earlier source repeat. Full outcomes never consulted.",
        source_arm=winner,
        source_repeat=job["repeat"],
        source_optimizer_seed=job["seed"],
        candidate=deepcopy(value),
        candidate_id=source.cid(value),
        dev_correct=correct,
        dev_questions=len(m["dev"]["bird"]),
        arm_mean_dev_accuracy=means[winner],
        effective_demo_ids=value["demos"],
        limitation="Each concrete SQL candidate had one compact dev measurement; strongest observed candidate is not proved best. Method name 'few' permits zero demonstrations.",
    )


def load_dev(parent):
    m = read(parent / "experiment.json")
    shared.validate(m)
    hashes, records = {}, {}
    for j in m["jobs"]:
        if not shared.completed(parent, j, m):
            raise ValueError("Finish all three compact dev jobs first")
        p = parent / "jobs" / j["name"] / "evaluations.json"
        records[j["name"]] = read(p)
        expected = {
            shared.key(v, i) for v in j["arms"].values() for i in m["dev"]["bird"]
        }
        if set(records[j["name"]]) != expected:
            raise ValueError("Compact dev checkpoint IDs differ")
        for path in (p, p.parent / "confirm/metadata.json"):
            hashes[str(path.relative_to(parent))] = digest(path)
    hashes["selection.json"] = digest(parent / "selection.json")
    return m, choose_candidate(m, records, read(parent / "selection.json")), hashes


def load_full(parent, dev):
    full = read(parent / "experiment.json")
    for name in ("protocol", "cohorts", "cases", "references"):
        if full[name + "_sha256"] != source.search.fingerprint(full[name]):
            raise ValueError("Full source fingerprints changed")
    if full["protocol"]["schema_audit_sha256"] != source.search.fingerprint(
        full["schema_audit"]
    ):
        raise ValueError("Full source schema audit changed")
    items = full["cohorts"]["full"]["bird"]
    if len(items) != 241 or len({source.item_id(i) for i in items}) != 241:
        raise ValueError("Expected exactly the agreed 241 full BIRD questions")
    if (
        full["schema_audit"]["schemas"]["bird"]
        != dev["schema_audit"]["schemas"]["bird"]
    ):
        raise ValueError("Full/dev schema audit differs")
    if any(
        full["protocol"]["model"][k] != dev["model"][k]
        for k in ("model", "model_version", "base_url", "api_backend")
    ):
        raise ValueError("Full source used a different model/provider")
    return items


def make_manifest(parent, full_parent, env_file):
    old, choice, hashes = load_dev(parent)
    items = load_full(full_parent, old)
    m = deepcopy(old)
    m.pop("sha256")
    m.update(
        version=VERSION,
        created_at_utc=source.search.now(),
        compact_dev_source=dict(
            path=str(parent.relative_to(source.ROOT)),
            manifest_file_sha256=digest(parent / "experiment.json"),
            files_sha256=hashes,
        ),
        full_source=dict(
            path=str(full_parent.relative_to(source.ROOT)),
            manifest_file_sha256=digest(full_parent / "experiment.json"),
        ),
        frozen_choice=choice,
        full={"bird": deepcopy(items)},
        env_file=env_file,
        teacher_env_file=env_file,
        limits=LIMITS,
        global_limits=GLOBAL_LIMITS,
        feature_mask=FEATURES,
        jobs=[
            dict(
                name="bird_full__paired",
                slot="sql",
                repeat=1,
                seed=choice["source_optimizer_seed"],
                arms={
                    "original": source.candidate(old["seed_instructions"]["sql"]),
                    "selected_sql": deepcopy(choice["candidate"]),
                },
            )
        ],
        files_sha256={
            **old["files_sha256"],
            **{name: digest(source.ROOT / name) for name in FILES},
        },
        policy=dict(
            task="Descriptive fresh BIRD full EX: two fixed SQL prompts, same 241 questions, compact features",
            control="Original compact SQL instruction, not all-false baseline; original taxonomy unchanged in both arms",
            selection="No prompt selection, optimization or deployment from full results; concrete candidate already frozen from compact dev",
            order="Paired by question; alternate original/selected order across successive questions",
            data_status="Retrospective full includes viewed train/inner/dev; not an independent generalization test",
            runtime="Internal confirm phase reused solely for budget/transport compatibility; evaluation_scope=full_retrospective is mandatory. Full cohort is separately named and recorded, source dev remains unchanged.",
            gold="All 241 references executable read-only before paid generation; current reference never inserted into prompt; fit demos self-excluded by original renderer",
            cache="Only this series checkpoints; never reuse old full or compact dev answers",
            budget="New bounded ledger, teacher forbidden; actual usage preserved, no counter resets",
        ),
    )
    m["sha256"] = source.search.fingerprint(m)
    return m


def validate(m):
    if m.get("version") != VERSION or m.get("sha256") != source.search.fingerprint(
        {k: v for k, v in m.items() if k != "sha256"}
    ):
        raise ValueError("Full prompt manifest changed")
    dev_parent = shared.output_path(m["compact_dev_source"]["path"])
    old, choice, hashes = load_dev(dev_parent)
    full_parent = full_source_path(m["full_source"]["path"])
    full_items = load_full(full_parent, old)
    expected_job = dict(
        name="bird_full__paired",
        slot="sql",
        repeat=1,
        seed=choice["source_optimizer_seed"],
        arms={
            "original": source.candidate(old["seed_instructions"]["sql"]),
            "selected_sql": choice["candidate"],
        },
    )
    if (
        m["feature_mask"] != FEATURES
        or m["limits"] != LIMITS
        or m["global_limits"] != GLOBAL_LIMITS
        or m["compact_dev_source"]["manifest_file_sha256"]
        != digest(dev_parent / "experiment.json")
        or m["compact_dev_source"]["files_sha256"] != hashes
        or m["full_source"]["manifest_file_sha256"]
        != digest(full_parent / "experiment.json")
        or m["frozen_choice"] != choice
        or m["full"] != {"bird": full_items}
        or m["jobs"] != [expected_job]
        or any(
            m[field] != old[field]
            for field in (
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
        raise ValueError("Frozen candidate, full cohort or protocol differs")
    for name, expected in m["files_sha256"].items():
        if digest(source.ROOT / name) != expected:
            raise ValueError(f"Pinned full runtime changed: {name}; prepare new output")


class FullSession(shared.CompactSession):
    def __init__(self, root, manifest, job):
        # Only the runtime view uses the historical confirm key. Persisted full
        # manifest keeps the old dev cohort distinct from the full cohort.
        view = {**manifest, "dev": {"bird": manifest["full"]["bird"], "ambrosia": []}}
        super().__init__(root, view, job, "confirm")
        self.metadata.update(
            feature_mask=FEATURES,
            full_prompt_version=VERSION,
            evaluation_scope="full_retrospective",
        )

    def progress(self, update):
        shared.runtime.Session.progress(
            self,
            {
                **update,
                "feature_mask": FEATURES,
                "full_prompt_version": VERSION,
                "evaluation_scope": "full_retrospective",
            },
        )


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


def pairs(job, items):
    values = list(job["arms"].values())
    for index, item in enumerate(items):
        for value in values if index % 2 == 0 else reversed(values):
            yield item, value


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
                    f"[BIRD FULL] 241 paired questions; features={FEATURES}; candidate={m['frozen_choice']['candidate_id'][:12]}",
                    flush=True,
                )
                for item, value in pairs(job, m["full"]["bird"]):
                    session.evaluate(item, value, benchmark="bird")
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
    items = m["full"]["bird"]
    rows = []
    control = [records.get(shared.key(job["arms"]["original"], i)) for i in items]
    base = sum(r["score"] for r in control) / len(items) if all(control) else None
    lines = [
        "# BIRD full: frozen compact SQL prompt comparison",
        "",
        f"Same {len(items)} questions, features {FEATURES}; fresh paired calls; one descriptive repeat, teacher=0.",
        "Selected SQL candidate fixed before full from compact dev; source method permits zero demonstrations.",
        "",
        "| Arm | Saved | Correct | EX | Delta, pp | Rescued / harmed | Generation / SQL errors | False ambiguity | Question tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, value in job["arms"].items():
        results = [records.get(shared.key(value, i)) for i in items]
        full = all(results)
        correct = sum(r["score"] for r in results if r)
        ex = correct / len(items) if full else None
        gain = ex - base if full and base is not None else None
        rescued = sum(
            r["score"] > b["score"] for r, b in zip(results, control) if r and b
        )
        harmed = sum(
            r["score"] < b["score"] for r, b in zip(results, control) if r and b
        )
        gen = sum(r["prediction"] == "error" for r in results if r)
        false_ambiguity = sum(r["prediction"] == "ambiguous" for r in results if r)
        sql = sum(
            str(r.get("error", "")).startswith("postgresql:") for r in results if r
        )
        question_tokens = sum(
            r.get("usage", {}).get("prompt_tokens", 0)
            + r.get("usage", {}).get("completion_tokens", 0)
            for r in results
            if r
        )
        rows.append(
            dict(
                arm=arm,
                candidate_id=source.cid(value),
                saved=sum(r is not None for r in results),
                correct=correct,
                ex=ex,
                delta=gain,
                rescued=rescued,
                harmed=harmed,
                generation_errors=gen,
                false_ambiguity=false_ambiguity,
                sql_errors=sql,
                question_tokens=question_tokens,
            )
        )
        lines.append(
            f"| {arm} | {rows[-1]['saved']}/{len(items)} | {correct:g} | {'—' if ex is None else f'{100 * ex:.2f}%'} | {'—' if gain is None else f'{100 * gain:+.2f}'} | {rescued} / {harmed} | {gen} / {sql} | {false_ambiguity} | {question_tokens} |"
        )
    usage = read(root / "budget.json") if (root / "budget.json").exists() else {}
    lines += [
        "",
        f"Measured total tokens: {usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)}; unknown allowance: {usage.get('unknown_usage_tokens', 0)}; accounted: {usage.get('tokens', 0)}. Root ledger is authoritative, arm usage excludes build/unknown overhead.",
        "",
        "No full-driven prompt selection or automatic deployment. Full overlaps viewed training/dev; not an independent test. One paired pass does not establish run-to-run robustness.",
    ]
    store.write_bytes(root / "report.md", ("\n".join(lines) + "\n").encode("utf-8"))
    store.write_json(
        root / "report.json",
        dict(
            manifest_sha256=m["sha256"],
            evaluation_scope="full_retrospective",
            rows=rows,
            authoritative_budget=usage,
            frozen_choice=m["frozen_choice"],
        ),
    )
    if all(row["ex"] is not None for row in rows):
        store.write_json(
            root / "conclusion.json",
            dict(
                manifest_sha256=m["sha256"],
                status="completed_descriptive",
                fixed_candidate=m["frozen_choice"]["candidate_id"],
                rows=rows,
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
    parser.add_argument("--dev-source", default=shared.DEFAULT_OUTPUT)
    parser.add_argument("--full-source", default=FULL_SOURCE)
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
            shared.output_path(args.dev_source),
            full_source_path(args.full_source),
            args.env_file,
        )
    else:
        raise ValueError("Prepare separate full protocol offline first")
    validate(m)
    if not (args.status or args.report):
        shared.check_current_model(m)
    jobs = m["jobs"]
    if args.jobs and (not args.worker or set(args.jobs) != {j["name"] for j in jobs}):
        raise ValueError("Unexpected worker job authorization")
    if args.dry_run:
        choice = m["frozen_choice"]
        print(
            f"OFFLINE BIRD full: {len(m['full']['bird'])} questions x 2 prompts = 482 fresh evaluations; features={FEATURES}; teacher=0"
        )
        print(
            f"Fixed candidate: {choice['candidate_id']}; source optimizer seed={choice['source_optimizer_seed']}; demonstrations={choice['effective_demo_ids']}; dev={choice['dev_correct']:g}/{choice['dev_questions']}"
        )
        print("Local limits:", LIMITS, "Global:", GLOBAL_LIMITS)
        return 0
    root.mkdir(parents=True, exist_ok=True)
    if args.prepare or args.status or args.report:
        with shared.suite_lock(root):
            if not path.exists():
                store.write_json(path, m)
                store.write_json(
                    root / "frozen_candidate.json",
                    dict(manifest_sha256=m["sha256"], **m["frozen_choice"]),
                )
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
