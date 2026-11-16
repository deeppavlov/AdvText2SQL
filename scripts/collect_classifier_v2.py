"""Manual bounded collection of missing train/dev configuration scores; no teacher."""

import argparse
from contextlib import contextmanager
import copy
import importlib.metadata
import os
from pathlib import Path
import sys
from unittest.mock import patch

import prepare_classifier_v2 as data
from durable_prompt_store import write_json


def check_packages(manifest):
    environment = manifest["runtime_base"]["runtime_environment"]
    expected = environment["packages"] | {"openai": environment["openai"]}
    for name, version in expected.items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Runtime package changed: {name}; restore the pinned environment before collection")


def expected_key(manifest, job, item):
    return f"confirm:{data.fingerprint(manifest['candidates'][job['benchmark']])}:hybrid:False:{job['benchmark']}:{item['question_id']}"


def completed(root, manifest, job):
    folder = root / "jobs" / job["name"]
    path = folder / "confirm/metadata.json"
    if not path.exists():
        if (folder / "evaluations.json").exists():
            raise ValueError("Collection checkpoint lacks provenance")
        return False
    meta = data.read(path)
    if (meta.get("manifest_sha256") != manifest["sha256"] or meta.get("collection_job") != job["name"]
            or meta.get("feature_mask") != job["features"]):
        raise ValueError("Foreign v2 checkpoint")
    if meta.get("status") == "completed":
        values = data.read(folder / "evaluations.json")
        if set(values) != {expected_key(manifest, job, i) for i in job["items"]}:
            raise ValueError("Completed job has incomplete or foreign evaluations")
    return meta.get("status") == "completed"


def collected_observations(root, manifest, rows):
    observations = {}
    for job in manifest["jobs"]:
        folder = root / "jobs" / job["name"]
        path = folder / "evaluations.json"
        if not path.exists():
            continue
        completed(root, manifest, job)  # Check ownership also for partial checkpoints.
        values = data.read(path)
        allowed = {expected_key(manifest, job, i): i for i in job["items"]}
        if not set(values) <= set(allowed):
            raise ValueError("Question outside frozen collection job")
        for key, value in values.items():
            item = allowed[key]
            identity = f"{job['benchmark']}:{item['question_id']}"
            if identity not in rows or rows[identity]["split"] != job["split"]:
                raise ValueError("Collection crossed a split boundary")
            score = data.validate_observation(value, item, job["benchmark"])
            if score is None:
                raise ValueError("Unaudited infrastructure error in collected checkpoint; restore it before training")
            if score is not None:
                observations[(identity, job["configuration"])] = {
                    "score": score, "source": path.relative_to(data.ROOT).as_posix(), "key": key,
                    "usage": value.get("usage"), "archive_phase": "new_frozen_collection"}
    return observations


def refresh_dataset(root, manifest):
    _, _, _, rows = data.load_context()
    cached = {(r["id"], r["configuration"]): r["observation"] for r in manifest["cached_observations"]}
    new = collected_observations(root, manifest, rows)
    if set(cached) & set(new):
        raise ValueError("New calls unexpectedly duplicate an archived observation")
    return data.export_dataset(data.ROOT, manifest, rows, cached | new)


def make_session(root, manifest, job):
    # Heavy runtime and environment loading happen only after explicit --run.
    import prompt_comparison_runtime as runtime
    import prompt_pilot_runtime as base
    import prompt_comparison as campaign

    class CollectionSession(runtime.Session):
        def __init__(self):
            view = copy.deepcopy(manifest["runtime_base"])
            view.update(sha256=manifest["sha256"], model=manifest["model"], teacher=manifest["model"],
                        env_file=manifest["env_file"], teacher_env_file=manifest["env_file"],
                        limits=manifest["limits"], global_limits=manifest["global_limits"],
                        dev={b: job["items"] if b == job["benchmark"] else [] for b in ("bird", "ambrosia")})
            worker = {**job, "slot": "sql" if job["benchmark"] == "bird" else "ambiguity", "seed": 42}
            super().__init__(root, view, worker, "confirm")
            self.metadata.update(collection_job=job["name"], collection_version=data.VERSION,
                                 feature_mask=job["features"], collection_split=job["split"])

        async def prepare_tools(self, profile):
            with patch.dict(base.pilot.PROFILES, {"hybrid": job["features"]}):
                await super().prepare_tools(profile)

        @contextmanager
        def apply(self, value, item, profile, original):
            if value != manifest["candidates"][job["benchmark"]] or profile != "hybrid" or original:
                raise ValueError("Only the frozen task prompt is authorized")
            if job["benchmark"] == "bird":
                template, demos = campaign.render(value, self.manifest, "sql", self.fit, item,
                                                   profile=profile, original=False)
                prompts = {"SQL_PROMPT_TEMPLATE": template}
                # When FEAT_14 is off, the generator intentionally uses the original BASIC template.
            else:
                prompts = manifest["task_manifests"]["ambrosia"]["fixed_prompts"]
                demos = []
            with base.settings(self.impl, job["features"]):
                with patch.multiple(self.impl, **prompts):
                    yield demos

        def reflect(self, prompt):
            raise RuntimeError("Teacher calls are forbidden in classifier data collection")

    return CollectionSession()


def run_job(root, manifest, job):
    if completed(root, manifest, job):
        return
    # Do not construct another session while an unknown API charge is reserved.
    for p in (root / "budget.json", root / "jobs" / job["name"] / "confirm/budget.json"):
        if p.exists() and data.read(p).get("pending"):
            raise RuntimeError("Unsettled API reservation; inspect accounting before manual resume")
    import bird_compact_transfer as durable
    import prompt_comparison_runtime as runtime

    with durable.durable_runtime():
        session = make_session(root, manifest, job)
        try:
            with (session.folder / "collection.log").open("a", encoding="utf-8") as log:
                secrets = [os.environ.get("LLM_API_KEY", ""), os.environ.get("DB_PASS", "")]
                with patch.multiple(sys, stdout=runtime.Tee(sys.stdout, log, secrets),
                                    stderr=runtime.Tee(sys.stderr, log, secrets)):
                    session.preflight()
                    for item in job["items"]:
                        session.evaluate(item, manifest["candidates"][job["benchmark"]], benchmark=job["benchmark"])
                    session.metadata.update(status="completed", finished_at_utc=runtime.search.now())
        except BaseException as error:
            session.metadata.update(status="stopped", error_type=type(error).__name__)
            raise
        finally:
            write_json(session.folder / "metadata.json", session.metadata)
            session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=data.DEFAULT_OUTPUT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--export", action="store_true")
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    parser.add_argument("--benchmark", choices=("bird", "ambrosia", "both"), default="both")
    parser.add_argument("--configuration", choices=tuple(data.CONFIGURATIONS))
    parser.add_argument("--max-jobs", type=int, default=1)
    args = parser.parse_args()
    if args.max_jobs < 1:
        parser.error("max-jobs must be positive")
    root = data.local_path(data.ROOT, args.output)
    if not root.is_relative_to(data.ROOT / "research_results"):
        parser.error("Use a collection folder under research_results")
    manifest = data.read(root / "experiment.json")
    data.validate(manifest)
    if args.export:
        from run_feature_ablation import suite_lock
        with suite_lock(root):
            data.status(manifest, refresh_dataset(root, manifest))
        return
    jobs = [j for j in manifest["jobs"] if j["split"] == args.split
            and (args.benchmark == "both" or j["benchmark"] == args.benchmark)
            and (not args.configuration or j["configuration"] == args.configuration) and j["items"]]
    jobs = [j for j in jobs if not completed(root, manifest, j)]
    jobs = jobs[:args.max_jobs]
    for job in jobs:
        path = root / "jobs" / job["name"] / "evaluations.json"
        saved = len(data.read(path)) if path.exists() else 0
        print(f"{job['name']}: {saved}/{len(job['items'])} saved; features={job['features']}", flush=True)
    if args.dry_run or not jobs:
        print("OFFLINE: no paid calls; selected jobs:", len(jobs))
        return
    check_packages(manifest)
    from run_feature_ablation import suite_lock
    with suite_lock(root):
        for job in jobs:
            run_job(root, manifest, job)
            data.status(manifest, refresh_dataset(root, manifest))


if __name__ == "__main__":
    main()
