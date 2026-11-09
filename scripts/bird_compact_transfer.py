"""Frozen, manual BIRD dev transfer of train-selected SQL prompts to compact."""

import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import importlib.metadata as metadata
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

import atomic_artifacts
import durable_prompt_store as store
import prompt_comparison as source
import prompt_comparison_runtime as runtime
import prompt_pilot_budget as budget
import prompt_pilot_runtime as base
import recover_prompt_timeout as recovery
import static_hybrid_runtime as recording
import transport_safe_runtime as transport
from run_feature_ablation import suite_lock
from run_openrouter_baseline import Tee

VERSION = "bird-compact-prompt-transfer-v1"
DEFAULT_OUTPUT = "research_results/bird_compact_prompt_transfer_20261107"
FEATURES = [2, 7, 14, 32, 35]
LIMITS = dict(
    examples=400,
    task_calls=1500,
    teacher_calls=0,
    tokens=3_000_000,
    active_seconds=7200,
)
GLOBAL_LIMITS = dict(
    examples=1200,
    task_calls=4500,
    teacher_calls=0,
    tokens=6_000_000,
    active_seconds=21600,
)
FILES = (
    "scripts/bird_compact_transfer.py",
    "local/run_bird_compact_transfer.ps1",
    "local/run_bird_compact_transfer.sh",
    "scripts/durable_prompt_store.py",
    "scripts/recover_prompt_timeout.py",
)
read = source.search.read
digest = source.pilot.digest


def output_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = source.ROOT / path
    path = path.resolve()
    if path.parent != (
        source.ROOT / "research_results"
    ).resolve() or not path.name.startswith("bird_compact_prompt_transfer_"):
        raise ValueError(
            "Use a dedicated research_results/bird_compact_prompt_transfer_<name> folder"
        )
    return path


def check_source(parent):
    m = read(parent / "experiment.json")
    if m.get("version") != source.VERSION or m.get(
        "sha256"
    ) != source.search.fingerprint({k: v for k, v in m.items() if k != "sha256"}):
        raise ValueError("Source experiment manifest changed")
    for name, sha in m["files_sha256"].items():
        if digest(source.ROOT / name) != sha:
            raise ValueError(f"Frozen source changed: {name}")
    if {**source.pilot.environment(), "openai": metadata.version("openai")} != m[
        "runtime_environment"
    ]:
        raise ValueError("Task runtime environment differs from source searches")
    if (
        digest(source.ROOT / m["source"]["path"] / "experiment.json")
        != m["source"]["sha256"]
    ):
        raise ValueError("Source dataset protocol changed")
    return m


def plan_jobs(m, selections):
    """Select no candidate on dev outcomes: preserve all four methods/seeds."""
    jobs = []
    for repeat, seed in enumerate(source.SEEDS, 1):
        original = source.candidate(m["seed_instructions"]["sql"])
        arms = {
            "original": original,
            "fixed_few": source.candidate(
                original["instruction"],
                source.fixed_demos(m["cohorts"]["sql"]["fit"], "sql", seed),
            ),
        }
        for method in source.METHODS:
            selection = selections[f"sql__{method}__r{repeat}"]
            if (
                selection.get("manifest_sha256") != m["sha256"]
                or selection.get("external_dev_used") is not False
                or selection.get("sha256")
                != source.search.fingerprint(
                    {k: v for k, v in selection.items() if k != "sha256"}
                )
            ):
                raise ValueError(
                    "SQL selection provenance changed or used external dev"
                )
            if selection["job"] != dict(
                name=f"sql__{method}__r{repeat}", slot="sql", method=method, seed=seed
            ):
                raise ValueError("Unexpected optimizer selection identity")
            value = deepcopy(selection["candidate"])
            source.check(value, m, "sql", m["cohorts"]["sql"]["fit"])
            arms[method] = value
        jobs.append(
            dict(
                name=f"bird_compact__r{repeat}",
                slot="sql",
                seed=seed,
                repeat=repeat,
                arms=arms,
            )
        )
    return jobs


def make_manifest(parent, env_file):
    old = check_source(parent)
    selections, hashes = {}, {}
    for repeat in (1, 2, 3):
        for method in source.METHODS:
            name = f"sql__{method}__r{repeat}"
            p = parent / "jobs" / name
            if read(p / "search/metadata.json").get("status") != "completed":
                raise ValueError("Complete all 12 SQL searches before compact transfer")
            selections[name] = read(p / "selection.json")
            hashes[str((p / "selection.json").relative_to(parent))] = digest(
                p / "selection.json"
            )
    m = deepcopy(old)
    m.pop("sha256")
    m.update(
        version=VERSION,
        created_at_utc=source.search.now(),
        source_search=dict(
            path=str(parent.relative_to(source.ROOT)),
            file_sha256=digest(parent / "experiment.json"),
            manifest_sha256=old["sha256"],
            selection_files_sha256=hashes,
        ),
        env_file=env_file,
        teacher_env_file=env_file,
        limits=LIMITS,
        global_limits=GLOBAL_LIMITS,
        feature_mask=FEATURES,
        dev={"bird": deepcopy(old["dev"]["bird"]), "ambrosia": []},
        jobs=plan_jobs(old, selections),
        files_sha256={
            **old["files_sha256"],
            **{name: digest(source.ROOT / name) for name in FILES},
        },
        policy=dict(
            task="BIRD SQL execution dev, compact features only; no optimizer/teacher calls",
            control="Original static compact SQL prompt; not all-false baseline",
            candidates="Four methods selected on grouped train inner validation for each of three optimizer seeds; no best-seed cherry picking",
            cache="Only identical candidate+question within the same repeat; never across repeats or from old 14+32 responses",
            selection="After all three repeats: mean dev EX, ties prefer original then fixed arm order; no automatic deployment",
            data_status="55 previously viewed dev questions, retrospective development evidence; full/test are not used to choose candidates",
            features="All flags false except 2,7,14,32,35, for tool construction and each query",
            generation="Inherited source generator, taxonomy, retry, evidence, SQL guard, evaluation and 3s question pacing",
            recovery="Conservative journaled accounting; optional max two external resumes for confirmed transient task failures; no billing/auth/unknown retries",
            io="flush+fsync before JSON replace and after response/journal append",
        ),
    )
    # Optimizer packages are not imported or needed for this task-only evaluator.
    m.pop("package_files_sha256", None)
    m["sha256"] = source.search.fingerprint(m)
    return m


def validate(m):
    if m.get("version") != VERSION or m.get("sha256") != source.search.fingerprint(
        {k: v for k, v in m.items() if k != "sha256"}
    ):
        raise ValueError("Compact manifest changed")
    if (
        m["feature_mask"] != FEATURES
        or m["limits"] != LIMITS
        or m["global_limits"] != GLOBAL_LIMITS
    ):
        raise ValueError("Compact feature mask or budgets changed")
    parent = source.ROOT / m["source_search"]["path"]
    old = check_source(parent)
    if digest(parent / "experiment.json") != m["source_search"]["file_sha256"]:
        raise ValueError("Source manifest bytes changed")
    selections = {}
    for name, expected in m["source_search"]["selection_files_sha256"].items():
        if digest(parent / name) != expected:
            raise ValueError("Source SQL selection changed")
        selections[Path(name).parent.name] = read(parent / name)
    if m["jobs"] != plan_jobs(old, selections) or m["dev"] != {
        "bird": old["dev"]["bird"],
        "ambrosia": [],
    }:
        raise ValueError("Candidate plan or dev questions changed")
    for name, expected in m["files_sha256"].items():
        if digest(source.ROOT / name) != expected:
            raise ValueError(
                f"Pinned compact/source file changed: {name}; prepare a new output"
            )


def check_current_model(m):
    from dotenv import dotenv_values

    actual = source.pilot.public_model(dotenv_values(source.ROOT / m["env_file"]))
    if any(
        actual[k] != m["model"][k]
        for k in ("model", "model_version", "base_url", "api_backend")
    ):
        raise ValueError("Env model/provider differs from the frozen compact protocol")


def count_evaluations(m, job):
    return len({source.cid(v) for v in job["arms"].values()}) * len(m["dev"]["bird"])


@contextmanager
def durable_runtime():
    with ExitStack() as stack:
        for module in (
            atomic_artifacts,
            runtime,
            base,
            budget,
            recording,
            transport,
            recovery,
        ):
            stack.enter_context(patch.object(module, "write_json", store.write_json))
        for module in (runtime, base, recording):
            stack.enter_context(patch.object(module, "append", store.append))
        yield


class CompactSession(runtime.Session):
    async def prepare_tools(self, profile):
        with patch.dict(base.pilot.PROFILES, {"hybrid": FEATURES}):
            await super().prepare_tools(profile)

    @contextmanager
    def apply(self, value, item, profile, original):
        with patch.dict(base.pilot.PROFILES, {"hybrid": FEATURES}):
            with super().apply(value, item, profile, original) as demos:
                yield demos

    def progress(self, update):
        super().progress(
            {**update, "feature_mask": FEATURES, "compact_version": VERSION}
        )

    def reflect(self, prompt):
        raise RuntimeError(
            "Teacher/optimization calls are forbidden in compact transfer"
        )


def run_job(root, m, job):
    if completed(root, job, m):
        return
    session = CompactSession(root, m, job, "confirm")
    try:
        with (session.folder / "comparison.log").open("a", encoding="utf-8") as log:
            secrets = [os.environ.get("LLM_API_KEY", ""), os.environ.get("DB_PASS", "")]
            with patch.multiple(
                sys,
                stdout=Tee(sys.stdout, log, secrets),
                stderr=Tee(sys.stderr, log, secrets),
            ):
                session.preflight()
                for arm, value in job["arms"].items():
                    print(
                        f"[COMPACT] {job['name']} arm={arm}, features={FEATURES}",
                        flush=True,
                    )
                    for item in m["dev"]["bird"]:
                        session.evaluate(item, value, benchmark="bird")
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
    return f"confirm:{source.cid(value)}:hybrid:False:bird:{source.item_id(item)}"


def report(root, m):
    rows = []
    lines = [
        "# BIRD compact prompt transfer — retrospective dev",
        "",
        f"Features: {FEATURES}. {len(m['dev']['bird'])} fixed dev questions, three repeats, no teacher.",
        "Original means the static compact control, not all-false. Identical arms share measurements within a repeat.",
        "",
        "| Repeat | Arm | Saved | EX | Delta | Rescued / harmed | Errors |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for job in m["jobs"]:
        p = root / "jobs" / job["name"] / "evaluations.json"
        if p.exists():
            completed(root, job, m)  # Refuse copied 14+32 checkpoints, even in reports.
            if not (p.parent / "confirm/metadata.json").exists():
                raise ValueError(
                    "Existing evaluations lack compact provenance metadata"
                )
        records = read(p) if p.exists() else {}
        items = m["dev"]["bird"]
        control = [records.get(key(job["arms"]["original"], i)) for i in items]
        baseline = (
            sum(r["score"] for r in control) / len(items) if all(control) else None
        )
        for arm, value in job["arms"].items():
            results = [records.get(key(value, i)) for i in items]
            score = (
                sum(r["score"] for r in results) / len(items) if all(results) else None
            )
            delta = (
                score - baseline if score is not None and baseline is not None else None
            )
            rescued = sum(
                r["score"] > b["score"] for r, b in zip(results, control) if r and b
            )
            harmed = sum(
                r["score"] < b["score"] for r, b in zip(results, control) if r and b
            )
            row = dict(
                repeat=job["repeat"],
                arm=arm,
                candidate_id=source.cid(value),
                saved=sum(r is not None for r in results),
                accuracy=score,
                delta=delta,
                rescued=rescued,
                harmed=harmed,
                generation_errors=sum(r["prediction"] == "error" for r in results if r),
                sql_errors=sum(
                    str(r.get("error", "")).startswith("postgresql:")
                    for r in results
                    if r
                ),
            )
            rows.append(row)

            def fmt(n):
                return "—" if n is None else f"{100 * n:.2f}%"

            gain = "—" if delta is None else f"{100 * delta:+.2f} pp"
            lines.append(
                f"| {job['repeat']} | {arm} | {row['saved']}/{len(items)} | {fmt(score)} | {gain} | {rescued} / {harmed} | {row['generation_errors']} / {row['sql_errors']} |"
            )
    lines += [
        "",
        "| Arm | Complete repeats | Mean EX | SD | Mean paired delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    summary = []
    for arm in m["jobs"][0]["arms"]:
        complete = [r for r in rows if r["arm"] == arm and r["accuracy"] is not None]
        if not complete:
            continue
        mean = statistics.mean(r["accuracy"] for r in complete)
        sd = (
            statistics.stdev(r["accuracy"] for r in complete)
            if len(complete) > 1
            else None
        )
        deltas = [r["delta"] for r in complete if r["delta"] is not None]
        gain = statistics.mean(deltas) if deltas else None
        summary.append(
            dict(
                arm=arm,
                complete_repeats=len(complete),
                mean_accuracy=mean,
                sd=sd,
                mean_paired_delta=gain,
            )
        )
        lines.append(
            f"| {arm} | {len(complete)}/3 | {100 * mean:.2f}% | {'—' if sd is None else f'{100 * sd:.2f} pp'} | {'—' if gain is None else f'{100 * gain:.2f} pp'} |"
        )
    usage = read(root / "budget.json") if (root / "budget.json").exists() else {}
    lines += [
        "",
        f"Measured tokens: {usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)}; conservative unknown usage: {usage.get('unknown_usage_tokens', 0)}; accounted budget: {usage.get('tokens', 0)}.",
        "",
        "No full/test selection, independent-test claim or automatic deployment. SQL execution errors remain model outcomes; provider/DB outages stop the job.",
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
    if len(summary) == 6 and all(r["complete_repeats"] == 3 for r in summary):
        order = list(m["jobs"][0]["arms"])
        winner = max(
            summary, key=lambda r: (r["mean_accuracy"], -order.index(r["arm"]))
        )
        store.write_json(
            root / "selection.json",
            dict(
                manifest_sha256=m["sha256"],
                selected_arm=winner["arm"],
                mean_dev_accuracy=winner["mean_accuracy"],
                policy=m["policy"]["selection"],
                deployed=False,
                independent_test=False,
            ),
        )


def recover_pending(root, m, allowed_jobs=None):
    """Only confirmed transient, uncheckpointed task failures can be resumed."""
    prepared = [
        (p, read(p))
        for p in (root / "recoveries").glob("transport_*.json")
        if read(p).get("state") == "prepared"
    ]
    if prepared:
        if len(prepared) != 1:
            raise ValueError(
                "Multiple unfinished accounting journals; inspect manually"
            )
        path, journal = prepared[0]
        if journal.get("manifest_sha256") != m["sha256"] or (
            allowed_jobs is not None and journal["job"] not in allowed_jobs
        ):
            raise ValueError(
                "Unfinished accounting journal belongs to another protocol/repeat"
            )
        with patch.object(recovery, "write_json", store.write_json):
            recovery.apply_journal(root, path, journal)
        print(
            "[RECOVERY] Finished a partially applied accounting journal; no API calls.",
            flush=True,
        )
        return True
    global_path = root / "budget.json"
    locals_ = [p for p in root.glob("jobs/*/confirm/budget.json") if read(p)["pending"]]
    if not global_path.exists():
        if locals_:
            raise ValueError("Missing global budget with pending task usage")
        return False
    global_ = read(global_path)
    if not global_["pending"] and not locals_:
        return False
    if len(global_["pending"]) != 1 or len(locals_) != 1:
        raise ValueError("Ambiguous pending budgets; never reset counters")
    local_path = locals_[0]
    local = read(local_path)
    if list(global_["pending"].values()) != list(local["pending"].values()):
        raise ValueError("Global and local reservations differ")
    folder = local_path.parent
    job_name = folder.parent.name
    if allowed_jobs is not None and job_name not in allowed_jobs:
        raise ValueError("Pending reservation belongs to another selected repeat")
    meta = read(folder / "metadata.json")
    event = meta.get("transport_failure", {})
    if (
        job_name not in {j["name"] for j in m["jobs"]}
        or meta.get("status") != "stopped"
        or meta.get("manifest_sha256") != m["sha256"]
        or event.get("checkpoint_written") is not False
        or event.get("phase") != "confirm"
        or event.get("reason") != "client_exception"
        or event.get("status_code") not in (429, 500, 502, 503, 504)
    ):
        raise ValueError(
            "Pending call is not a confirmed transient task failure; inspect billing/auth/timeout manually"
        )
    event_id = source.search.fingerprint(
        dict(manifest=m["sha256"], job=job_name, event=event)
    )
    path = root / "recoveries" / f"transport_{event_id[:20]}.json"
    if path.exists():
        journal = read(path)
    else:
        patches = []
        for p, before in ((global_path, global_), (local_path, local)):
            after = recovery.charge_unknown(before, event_id)
            after["unknown_task_calls"] = after.get("unknown_task_calls", 0) + 1
            patches.append(
                dict(path=str(p.relative_to(root)), before=before, after=after)
            )
        artifacts = [
            root / "experiment.json",
            folder.parent / "evaluations.json",
            folder / "raw_responses.jsonl",
        ]
        journal = dict(
            version="compact-transient-accounting-v1",
            manifest_sha256=m["sha256"],
            state="prepared",
            event_id=event_id,
            job=job_name,
            transport_failure=event,
            created_at_utc=source.search.now(),
            actual_provider_usage=None,
            recovery_script_sha256=digest(Path(__file__)),
            patches=patches,
            unchanged_artifacts_sha256={
                str(p.relative_to(root)): digest(p) for p in artifacts if p.exists()
            },
        )
    if journal["state"] == "applied":
        raise ValueError("Previously applied recovery still has pending usage")
    with patch.object(recovery, "write_json", store.write_json):
        recovery.apply_journal(root, path, journal)
    print(
        f"[RECOVERY] {job_name}: HTTP {event['status_code']}, reservation accounted conservatively; measured tokens/calls/results preserved.",
        flush=True,
    )
    return True


def completed(root, job, m):
    p = root / "jobs" / job["name"] / "confirm/metadata.json"
    if not p.exists():
        return False
    meta = read(p)
    if (
        meta.get("manifest_sha256") != m["sha256"]
        or meta.get("feature_mask", FEATURES) != FEATURES
    ):
        raise ValueError("Existing checkpoint belongs to another protocol/feature mask")
    return meta.get("status") == "completed"


def run_worker(root, m, jobs):
    with suite_lock(root), durable_runtime():
        recover_pending(root, m, {j["name"] for j in jobs})
        for job in jobs:
            run_job(root, m, job)


def supervise(root, m, jobs, max_resumes):
    with suite_lock(root):
        recover_pending(root, m, {j["name"] for j in jobs})
        store.append(
            root / "resumption_events.jsonl",
            dict(
                version=VERSION,
                event="manual_launch",
                at_utc=source.search.now(),
                manifest_sha256=m["sha256"],
                jobs=[j["name"] for j in jobs],
                max_transient_resumes=max_resumes,
            ),
        )
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--output",
        str(root),
        "--worker",
        "--jobs",
        *[j["name"] for j in jobs],
    ]
    for attempt in range(max_resumes + 1):
        code = subprocess.run(command, cwd=source.ROOT).returncode
        store.append(
            root / "resumption_events.jsonl",
            dict(
                event="child_exit",
                at_utc=source.search.now(),
                attempt=attempt + 1,
                exit_code=code,
            ),
        )
        if code == 0:
            return 0
        with suite_lock(root):
            try:
                recovered = recover_pending(root, m, {j["name"] for j in jobs})
            except ValueError:
                print(
                    "[STOP] Not an eligible transient task failure; inspect saved metadata. No automatic retry.",
                    flush=True,
                )
                return code
        if not recovered or attempt == max_resumes:
            print(
                "[STOP] No transient resume available; inspect provider/database state.",
                flush=True,
            )
            return code
        wait = 60 * 2**attempt
        print(
            f"[RESUME] Waiting {wait}s before bounded external resume {attempt + 1}/{max_resumes}.",
            flush=True,
        )
        store.append(
            root / "resumption_events.jsonl",
            dict(event="wait", at_utc=source.search.now(), delay_seconds=wait),
        )
        time.sleep(wait)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--source", default=source.DEFAULT_OUTPUT)
    parser.add_argument("--env-file", default=".env.deepseek")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--repeat", type=int, choices=(1, 2, 3))
    parser.add_argument("--max-jobs", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--transient-resumes", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--jobs", nargs="+", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = output_path(args.output)
    if root == source.pilot.output_path(args.source):
        raise ValueError("Compact output must differ from the source campaign")
    path = root / "experiment.json"
    if not path.exists():
        if not (args.prepare or args.dry_run):
            raise ValueError(
                "Prepare the separate compact protocol offline with --prepare first"
            )
        m = make_manifest(source.pilot.output_path(args.source), args.env_file)
    else:
        m = read(path)
    validate(m)
    if not (args.report or args.status):
        check_current_model(m)
    jobs = [j for j in m["jobs"] if args.repeat is None or j["repeat"] == args.repeat]
    if args.jobs:
        if not args.worker or not set(args.jobs) <= {j["name"] for j in jobs}:
            raise ValueError("Invalid worker job authorization")
        jobs = [j for j in jobs if j["name"] in args.jobs]
    if args.dry_run:
        print(
            "OFFLINE compact BIRD:",
            FEATURES,
            "dev questions:",
            len(m["dev"]["bird"]),
            "no teacher",
        )
        for j in jobs:
            print(
                j["name"],
                "arms:",
                len(j["arms"]),
                "unique evaluations:",
                count_evaluations(m, j),
            )
        print("Per repeat limits:", LIMITS, "Whole series:", GLOBAL_LIMITS)
        return 0
    root.mkdir(parents=True, exist_ok=True)
    if args.prepare or args.report or args.status:
        with suite_lock(root):
            if not path.exists():
                store.write_json(path, m)
            if args.prepare or args.report:
                report(root, m)
            for j in jobs:
                print(
                    j["name"],
                    "completed" if completed(root, j, m) else "pending",
                    count_evaluations(m, j),
                )
        return 0
    if args.worker:
        run_worker(root, m, jobs)
        return 0
    jobs = [j for j in jobs if not completed(root, j, m)][: args.max_jobs]
    return supervise(root, m, jobs, args.transient_resumes) if jobs else 0


if __name__ == "__main__":
    sys.exit(main())
