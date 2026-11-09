"""Shared real generator, grouped feedback and persistent budgets for all methods."""

import asyncio
from contextlib import contextmanager
import os
import sys
from unittest.mock import patch

from atomic_artifacts import write_json
import prompt_comparison as campaign
from prompt_pilot_budget import BudgetStop, Ledger
import prompt_pilot_runtime as base
import static_hybrid_search as search
from run_openrouter_baseline import Tee
from run_openrouter_baseline import ProviderUnavailable
from static_hybrid_runtime import append


class DualLedger:
    """Reserve both caps before IO. Partial reservations stay held on any failure."""

    def __init__(self, local, global_, identity):
        self.local, self.global_, self.identity = local, global_, identity
        self.data = local.data
        if local.data["pending"] or global_.data["pending"]:
            raise RuntimeError(
                "Unsettled API reservation: inspect budget files before any resume; never reset counters"
            )

    def save(self):
        self.local.save()
        self.global_.save()

    def check_time(self):
        self.local.check_time()
        self.global_.check_time()

    def example(self, key):
        self.global_.example(self.identity + ":" + key)
        self.local.example(key)

    def reserve(self, role, messages, cap):
        a = self.global_.reserve(role, messages, cap)
        try:
            b = self.local.reserve(role, messages, cap)
        except BudgetStop:
            # No HTTP occurred: release only this unused global reservation.
            self.global_.data["pending"].pop(a)
            self.global_.data[role + "_calls"] -= 1
            self.global_.save()
            raise
        return a, b

    def settle(self, ticket, prompt, completion):
        a, b = ticket
        self.global_.settle(a, prompt, completion)
        self.local.settle(b, prompt, completion)


class Session(base.Session):
    def __init__(self, root, manifest, job, phase="search"):
        self.root, self.job, self.slot = root, job, job["slot"]
        self.fit = manifest["cohorts"][self.slot]["fit"]
        self.inner = manifest["cohorts"][self.slot]["inner_val"]
        self.benchmark = "bird" if self.slot == "sql" else "ambrosia"
        output = root / "jobs" / job["name"]
        output.mkdir(parents=True, exist_ok=True)
        # The old Session supplies transport, native API client and evaluator;
        # it never validates a comparison manifest as an old pilot.
        proxy = {
            **manifest,
            "cohorts": {**manifest["cohorts"][self.slot], "dev": manifest["dev"]},
            "limits": {
                "search": manifest["limits"],
                "confirm": {**manifest["limits"], "teacher_calls": 0},
            },
        }
        super().__init__(output, proxy, phase)
        self.manifest = manifest
        global_ = Ledger(root, manifest["global_limits"])
        self.ledger = DualLedger(self.ledger, global_, job["name"] + ":" + phase)
        self.items = (
            self.fit + self.inner
            if phase == "search"
            else [i for v in manifest["dev"].values() for i in v]
        )
        self.bird_items = (
            (self.items if self.slot == "sql" else [])
            if phase == "search"
            else manifest["dev"]["bird"]
        )
        self.teacher_cache = (
            search.read(self.folder / "teacher_cache.json")
            if (self.folder / "teacher_cache.json").exists()
            else {}
        )

    def preflight(self):
        if self.phase == "search" and self.slot == "ambiguity":
            from resilient_hybrid_runtime import validate_live_schemas

            validate_live_schemas(
                self.items,
                self.manifest["schema_audit"]["schemas"]["ambrosia"],
                progress=self.progress,
            )
            print(
                "[PREFLIGHT] Ambrosia grouped schemas verified; no paid calls",
                flush=True,
            )
        else:
            # Base uses a phase-specific proxy only during preflight.
            original = self.manifest
            self.manifest = {**original, "cohorts": {"dev": original["dev"]}}
            try:
                super().preflight()
            finally:
                self.manifest = original

    @contextmanager
    def apply(self, value, item, profile, original):
        with base.settings(self.impl, base.pilot.PROFILES[profile]):
            template, demos = campaign.render(
                value,
                self.manifest,
                self.slot,
                self.fit,
                item,
                profile=profile,
                original=original,
            )
            # With FEAT_32 disabled the taxonomy prompt is intentionally unused.
            field = (
                ("SQL_PROMPT_TEMPLATE" if profile == "hybrid" else "SQL_PROMPT_BASIC")
                if self.slot == "sql"
                else "TAXONOMY_DETECTION_PROMPT"
            )
            with patch.object(self.impl, field, template):
                yield demos

    def evaluate(self, item, value, profile="hybrid", original=False, benchmark=None):
        campaign.check(value, self.manifest, self.slot, self.fit)
        benchmark = benchmark or self.benchmark
        allowed = (
            (self.fit + self.inner)
            if self.phase == "search"
            else self.manifest["dev"][benchmark]
        )
        if not any(i == item for i in allowed):
            raise ValueError("Question outside the frozen phase cohort")
        key = f"{self.phase}:{campaign.cid(value)}:{profile}:{original}:{benchmark}:{campaign.item_id(item)}"
        if key in self.records:
            return self.records[key]
        self.ledger.example(key)
        before = {
            k: self.ledger.data[k]
            for k in ("task_calls", "prompt_tokens", "completion_tokens")
        }
        self.candidates[campaign.cid(value)] = value
        write_json(self.output / "candidates.json", self.candidates)

        async def query():
            await self.prepare_tools(profile)
            self.progress(
                {
                    "phase": "predicting",
                    "question_id": campaign.item_id(item),
                    "candidate": campaign.cid(value),
                    "profile": profile,
                }
            )
            question = (
                f"question: {item['question']}, evidence (may be empty): {item['evidence']}"
                if benchmark == "bird"
                else item["question"]
            )
            token = self.context.set(
                {
                    "phase": self.phase,
                    "question_id": campaign.item_id(item),
                    "db_id": item["db_id"],
                }
            )
            try:
                with self.apply(value, item, profile, original) as demos:
                    response = await self.tools[profile][item["db_id"]].query(question)
            finally:
                self.context.reset(token)
            from sql_output_contract import guarded_response

            response, issue = guarded_response(response)
            answer = (
                response["query"]
                if response["status"] == "success"
                else "ambiguous"
                if response["status"] == "ambiguous"
                else "error"
            )
            return answer, issue, demos

        answer, issue, demos = self.runner.run(query())
        score, error = self.score(item, answer, benchmark)
        result = {
            "question_id": campaign.item_id(item),
            "db_id": item["db_id"],
            "prediction": answer,
            "score": score,
            "error": issue or error,
            "effective_demo_ids": demos,
            "usage": {k: self.ledger.data[k] - before[k] for k in before},
        }
        self.records[key] = result
        write_json(self.output / "evaluations.json", self.records)
        self.progress({"saved_evaluations": len(self.records)})
        print(
            f"[{self.job['name']}/{self.phase}] {benchmark} {item['question_id']}: {score:g}",
            flush=True,
        )
        self.runner.run(asyncio.sleep(3))
        return result

    def feedback(self, items, value):
        if any(not any(i == row for row in self.fit) for i in items):
            raise ValueError("Only optimization-fit examples may reach reflection")
        return [
            {
                "question": i["question"],
                "evidence": i.get("evidence", ""),
                "schema": self.manifest["schema_audit"]["schemas"][self.benchmark][
                    i["db_id"]
                ]["rows"],
                "reference": i["SQL"]
                if self.slot == "sql"
                else i["SQL"] == "ambiguous",
                "result": self.evaluate(i, value),
            }
            for i in items
        ]

    def reflect(self, prompt):
        # Replay exact teacher requests on restart without extra charges.
        key = search.fingerprint(prompt)
        if key in self.teacher_cache:
            return self.teacher_cache[key]
        from openai import OpenAI

        model = self.manifest["teacher"]
        cap = 4096 if self.slot == "sql" else 8192
        messages = [{"role": "user", "content": prompt}]
        ticket = self.ledger.reserve("teacher", messages, cap)
        try:
            with OpenAI(
                api_key=self.teacher_config["LLM_API_KEY"],
                base_url=model["base_url"],
                max_retries=0,
                timeout=120,
            ) as client:
                result = client.chat.completions.create(
                    model=model["model"],
                    messages=messages,
                    temperature=0,
                    max_tokens=cap,
                    **model["request_options"],
                )
        except Exception as error:
            append(
                self.folder / "transport_events.jsonl",
                {"role": "teacher", "error_type": type(error).__name__},
            )
            raise ProviderUnavailable(
                "Teacher unavailable; reservation preserved, inspect before resume"
            ) from None
        if result.usage is None:
            raise ProviderUnavailable("Teacher usage missing; reservation preserved")
        self.ledger.settle(
            ticket, result.usage.prompt_tokens, result.usage.completion_tokens
        )
        answer = result.choices[0].message.content if result.choices else None
        append(
            self.folder / "raw_responses.jsonl",
            {
                "phase": "teacher",
                "messages": messages,
                "raw_response": answer,
                "usage": result.usage.model_dump(),
            },
        )
        if (
            not isinstance(answer, str)
            or not answer.strip()
            or result.choices[0].finish_reason == "length"
        ):
            raise ProviderUnavailable(
                "Teacher reply empty/truncated; costs retained, inspect response"
            )
        self.teacher_cache[key] = answer
        write_json(self.folder / "teacher_cache.json", self.teacher_cache)
        return answer


def run_job(root, manifest, job):
    folder = root / "jobs" / job["name"]
    if (folder / "selection.json").exists():
        return
    session = Session(root, manifest, job)
    stdout, stderr = sys.stdout, sys.stderr
    try:
        with (session.folder / "comparison.log").open("a", encoding="utf-8") as log:
            secrets = [
                os.environ.get("LLM_API_KEY", ""),
                os.environ.get("DB_PASS", ""),
                session.teacher_config.get("LLM_API_KEY", ""),
            ]
            sys.stdout, sys.stderr = (
                Tee(stdout, log, secrets),
                Tee(stderr, log, secrets),
            )
            session.preflight()
            seed = campaign.candidate(manifest["seed_instructions"][session.slot])
            for item in session.inner:
                session.evaluate(item, seed)
            fixed = campaign.candidate(
                seed["instruction"],
                campaign.fixed_demos(session.fit, session.slot, job["seed"]),
            )
            for item in session.inner:
                session.evaluate(item, fixed)
            try:
                from prompt_comparison_methods import optimize

                optimize(session)
            except BudgetStop:
                session.metadata["budget_exhausted"] = True
            campaign.select(session)
            session.metadata.update(
                status="completed", phase="search", finished_at_utc=search.now()
            )
    except BaseException as error:
        session.metadata.update(
            status="stopped",
            error_type=type(error).__name__,
            finished_at_utc=search.now(),
        )
        raise
    finally:
        sys.stdout, sys.stderr = stdout, stderr
        write_json(session.folder / "metadata.json", session.metadata)
        session.close()


def confirmation_plan(root, manifest):
    plan = []
    for slot in campaign.SLOTS:
        for r, seed in enumerate(campaign.SEEDS, 1):
            fit = manifest["cohorts"][slot]["fit"]
            original = campaign.candidate(manifest["seed_instructions"][slot])
            arms = {
                "original": original,
                "fixed_few": campaign.candidate(
                    original["instruction"], campaign.fixed_demos(fit, slot, seed)
                ),
            }
            for method in campaign.METHODS:
                p = root / "jobs" / f"{slot}__{method}__r{r}" / "selection.json"
                if not p.exists():
                    raise ValueError(
                        "Finish all 24 search jobs before freezing external dev confirmation"
                    )
                selection = search.read(p)
                if selection["manifest_sha256"] != manifest["sha256"] or selection[
                    "sha256"
                ] != search.fingerprint(
                    {k: v for k, v in selection.items() if k != "sha256"}
                ):
                    raise ValueError("Search selection provenance changed")
                arms[method] = selection["candidate"]
            plan.append(
                {
                    "name": f"{slot}__confirm__r{r}",
                    "slot": slot,
                    "seed": seed,
                    "arms": arms,
                }
            )
    return plan


def confirm(root, manifest):
    plan = confirmation_plan(root, manifest)
    frozen = {"manifest_sha256": manifest["sha256"], "plan": plan}
    path = root / "confirmation_plan.json"
    if path.exists() and search.read(path) != frozen:
        raise ValueError("Confirmation plan already frozen with different candidates")
    write_json(path, frozen)
    # Deliberately primary task + fixed hybrid only; no 2x2 explosion here.
    # Identical arms in one repeat reuse the same measured response, never across repeats.
    for job in plan:
        folder = root / "jobs" / job["name"] / "confirm"
        if (folder / "metadata.json").exists() and search.read(
            folder / "metadata.json"
        ).get("status") == "completed":
            continue
        session = Session(root, manifest, job, "confirm")
        try:
            session.preflight()
            for value in job["arms"].values():
                for item in manifest["dev"][session.benchmark]:
                    session.evaluate(item, value, benchmark=session.benchmark)
            session.metadata.update(status="completed", phase="confirm")
        except BaseException as error:
            session.metadata.update(status="stopped", error_type=type(error).__name__)
            raise
        finally:
            write_json(session.folder / "metadata.json", session.metadata)
            session.close()
            confirmation_report(root, manifest, plan)


def confirmation_report(root, manifest, plan):
    from research_metrics import ambrosia_report

    lines = [
        "# External dev confirmation (retrospective)",
        "",
        "| Slot | Repeat | Arm | Saved | Accuracy | Delta to original | Rescued / harmed | Amb F1 | Errors | Task tokens |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    output = []
    for job in plan:
        p = root / "jobs" / job["name"] / "evaluations.json"
        records = search.read(p) if p.exists() else {}
        benchmark = "bird" if job["slot"] == "sql" else "ambrosia"
        items = manifest["dev"][benchmark]
        baseline = campaign.complete_score(
            records, job["arms"]["original"], items, "confirm", benchmark=benchmark
        )
        for arm, value in job["arms"].items():
            rows = [
                records.get(
                    f"confirm:{campaign.cid(value)}:hybrid:False:{benchmark}:{campaign.item_id(i)}"
                )
                for i in items
            ]
            score = campaign.complete_score(
                records, value, items, "confirm", benchmark=benchmark
            )
            f1 = None
            if score is not None and benchmark == "ambrosia":
                f1 = ambrosia_report(
                    items, {r["question_id"]: r["prediction"] for r in rows}
                )["ambiguity_metrics"]["f1"]
            delta = None if score is None or baseline is None else score - baseline
            baseline_rows = [
                records.get(
                    f"confirm:{campaign.cid(job['arms']['original'])}:hybrid:False:{benchmark}:{campaign.item_id(i)}"
                )
                for i in items
            ]
            rescued = sum(
                r["score"] > b["score"] for r, b in zip(rows, baseline_rows) if r and b
            )
            harmed = sum(
                r["score"] < b["score"] for r, b in zip(rows, baseline_rows) if r and b
            )
            errors = sum(r["prediction"] == "error" for r in rows if r)
            tokens = sum(
                r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"]
                for r in rows
                if r
            )

            def fmt(v):
                return "—" if v is None else f"{100 * v:.2f}"

            lines.append(
                f"| {job['slot']} | {job['seed']} | {arm} | {sum(r is not None for r in rows)}/{len(items)} | {fmt(score)} | {fmt(delta)} pp | {rescued} / {harmed} | {fmt(f1)} | {errors} | {tokens} |"
            )
            output.append(
                {
                    "slot": job["slot"],
                    "seed": job["seed"],
                    "arm": arm,
                    "accuracy": score,
                    "delta": delta,
                    "f1": f1,
                    "tokens": tokens,
                    "rescued": rescued,
                    "harmed": harmed,
                    "generation_errors": errors,
                    "ambiguity_metrics": ambrosia_report(
                        items, {r["question_id"]: r["prediction"] for r in rows}
                    )["ambiguity_metrics"]
                    if score is not None and benchmark == "ambrosia"
                    else None,
                }
            )
    import statistics

    lines += [
        "",
        "| Slot | Arm | Complete repeats | Mean accuracy | SD | Mean delta |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for slot in campaign.SLOTS:
        for arm in ("original", "fixed_few", *campaign.METHODS):
            measured = [
                r
                for r in output
                if r["slot"] == slot and r["arm"] == arm and r["accuracy"] is not None
            ]
            if measured:
                mean = statistics.mean(r["accuracy"] for r in measured)
                sd = (
                    f"{100 * statistics.stdev(r['accuracy'] for r in measured):.2f}"
                    if len(measured) > 1
                    else "—"
                )
                deltas = [r["delta"] for r in measured if r["delta"] is not None]
                delta = f"{100 * statistics.mean(deltas):.2f}" if deltas else "—"
                lines.append(
                    f"| {slot} | {arm} | {len(measured)}/3 | {100 * mean:.2f}% | {sd} pp | {delta} pp |"
                )
    lines += [
        "",
        "Tokens per arm describe inference use; identical arms share cached measurements within a repeat.",
        "Repeated dev questions do not constitute independent new test samples. No prompt is deployed automatically.",
    ]
    (root / "confirmation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    write_json(
        root / "confirmation_report.json",
        {"manifest_sha256": manifest["sha256"], "rows": output},
    )
