"""GEPA adapter and persistent, budgeted execution of the unchanged SQL generator."""

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import os
import sys
from unittest.mock import patch

from dotenv import dotenv_values, load_dotenv

from atomic_artifacts import write_json
from prompt_pilot_budget import BudgetStop, Ledger
import prompt_pilot as pilot
import static_hybrid_search as search
from static_hybrid_runtime import append
from transport_safe_runtime import TransportRecordingClient
from run_openrouter_baseline import ProviderUnavailable, Tee

REFLECTION = """Improve general zero-shot SQL rules for PostgreSQL.
Only edit the rule text below. Question/evidence, schema, SQL dialect, the
SQL-only output contract, features, checker and retry algorithm are immutable.
Do not add examples, ready answers, database/table/column names, literal values
from the feedback, or rules for particular questions. Infer transferable rules.
No placeholders or braces. Keep the rule text under 6000 characters.
The feedback contains only optimization-fit questions, never external dev.
Current rules:
<curr_instructions>
Training feedback:
<inputs_outputs_feedback>
Return only the new rule text within one ``` block.
"""


def candidate_id(text, profile, original=False):
    return search.fingerprint(
        {"instruction": text, "profile": profile, "original": original}
    )


@contextmanager
def settings(module, features, instruction=None, profile="hybrid", original=False):
    updates = {f"FEAT_{n}": n in features for n in range(1, 37)}
    if instruction is not None:
        updates[
            "SQL_PROMPT_TEMPLATE" if profile == "hybrid" else "SQL_PROMPT_BASIC"
        ] = pilot.compile_template(instruction, profile, original=original)
    with patch.multiple(module, create=True, **updates):
        yield


def normalize(rows):
    # Same unordered set semantics as benchmarks.evaluate_bird, including arrays/JSON.
    def hashable(v):
        if isinstance(v, list):
            return tuple(hashable(x) for x in v)
        if isinstance(v, dict):
            return tuple(sorted((k, hashable(x)) for k, x in v.items()))
        return v

    return {tuple(hashable(v) for v in row) for row in rows}


class Session:
    def __init__(self, output, manifest, phase):
        self.output, self.manifest, self.phase = output, manifest, phase
        self.folder = output / phase
        self.folder.mkdir(exist_ok=True)
        previous = self.folder / "metadata.json"
        if previous.exists():
            old = search.read(previous)
            if old.get("manifest_sha256") != manifest["sha256"]:
                raise ValueError("Existing phase belongs to a different pilot")
            append(self.folder / "attempt_history.jsonl", old)
        self.ledger = Ledger(self.folder, manifest["limits"][phase])
        self.records = (
            search.read(output / "evaluations.json")
            if (output / "evaluations.json").exists()
            else {}
        )
        self.candidates = (
            search.read(output / "candidates.json")
            if (output / "candidates.json").exists()
            else {}
        )
        self.runner = asyncio.Runner()
        self.tools, self.engines, self.golds = {}, {}, {}
        self.context = ContextVar("pilot_question", default={"phase": phase})
        self.client = None
        self.metadata = {
            "status": "starting",
            "phase": phase,
            "manifest_sha256": manifest["sha256"],
        }
        load_dotenv(pilot.ROOT / manifest["env_file"], override=True)
        self.teacher_config = dotenv_values(pilot.ROOT / manifest["teacher_env_file"])
        for config, expected in (
            (os.environ, manifest["model"]),
            (self.teacher_config, manifest["teacher"]),
        ):
            actual = pilot.public_model(config)
            if any(
                actual[k] != expected[k]
                for k in ("model", "model_version", "base_url", "api_backend")
            ):
                raise ValueError("Task/teacher env differs from frozen identity")
        self.items = (
            manifest["cohorts"]["fit"] + manifest["cohorts"]["inner_val"]
            if phase == "search"
            else [i for values in manifest["cohorts"]["dev"].values() for i in values]
        )
        self.bird_items = (
            self.items if phase == "search" else manifest["cohorts"]["dev"]["bird"]
        )

    def progress(self, update):
        self.ledger.check_time()
        self.metadata.update(update)
        write_json(self.folder / "metadata.json", self.metadata)

    def preflight(self):
        from resilient_hybrid_runtime import validate_live_schemas
        from sqlalchemy import text
        from sqlalchemy.engine import URL
        from baseline_connection import database_engine

        sys.path[:0] = [str(pilot.ROOT), str(pilot.ROOT / "src")]
        from benchmarks.evaluate_bird import sqlite_to_postgres

        for b, items in (
            ("bird", self.bird_items),
            (
                "ambrosia",
                []
                if self.phase == "search"
                else self.manifest["cohorts"]["dev"]["ambrosia"],
            ),
        ):
            validate_live_schemas(
                items,
                self.manifest["schema_audit"]["schemas"][b],
                progress=self.progress,
            )
        for item in self.bird_items:
            db = item["db_id"]
            self.progress(
                {
                    "phase": "reference_preflight",
                    "question_id": str(item["question_id"]),
                }
            )
            if db not in self.engines:
                self.engines[db] = database_engine(
                    URL.create(
                        "postgresql+psycopg",
                        username=os.environ["DB_USER"],
                        password=os.environ["DB_PASS"],
                        host="127.0.0.1",
                        port=5444,
                        database=db,
                    )
                )
            try:
                with self.engines[db].connect() as conn:
                    conn.exec_driver_sql("SET TRANSACTION READ ONLY")
                    rows = conn.execute(
                        text(sqlite_to_postgres(item["SQL"]))
                    ).fetchall()
                self.golds[str(item["question_id"])] = normalize(rows)
            except Exception as error:
                raise RuntimeError(
                    f"Reference preflight failed for question {item['question_id']} ({type(error).__name__}); no paid calls started"
                ) from None
        print(
            f"[PREFLIGHT] {len(self.bird_items)} BIRD references executable; no paid calls",
            flush=True,
        )

    async def prepare_tools(self, profile):
        if profile in self.tools:
            return
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from baseline_connection import build_tools, database_engine

        for n in range(1, 37):
            os.environ[f"FEAT_{n}"] = str(n in pilot.PROFILES[profile]).lower()
        os.environ.update(
            MAX_RETRIES="5",
            OPTIMISTIC_AMBIGUITY_FALLBACK="false",
            LLM_BASE_INTERVAL="3",
            LLM_MIN_INTERVAL="3",
        )
        from adv_text2sql.mcp_servers.text2sql_tool.src import (
            text2sql_implementation as impl,
        )

        self.impl = impl
        model = self.manifest["model"]
        if self.client is None:
            raw = OpenAIChatCompletionClient(
                model=model["model"],
                base_url=model["base_url"],
                api_key=os.environ["LLM_API_KEY"],
                temperature=0,
                max_tokens=4096,
                max_retries=0,
                timeout=90,
                model_info={
                    "json_output": False,
                    "function_calling": True,
                    "vision": False,
                    "family": "unknown",
                    "structured_output": False,
                },
                **model["request_options"],
            )
            session = self

            class BudgetedClient(TransportRecordingClient):
                async def create(self, *args, **kwargs):
                    messages = args[0] if args else kwargs["messages"]
                    data = [{"type": m.type, "content": m.content} for m in messages]
                    ticket = session.ledger.reserve("task", data, 4096)
                    response = await super().create(*args, **kwargs)
                    session.ledger.settle(
                        ticket,
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens,
                    )
                    return response

            self.client = BudgetedClient(raw, self.folder, self.metadata, self.context)

        class Generator(impl.Text2SQLGenerator):
            def __init__(self, db_uri, llm_client):
                super().__init__(db_uri, llm_client)
                self.engine.dispose()
                self.engine = database_engine(db_uri)

        with settings(impl, pilot.PROFILES[profile]):
            self.tools[profile] = await build_tools(
                self.items, Generator, self.client, self.progress
            )

    def score(self, item, prediction, benchmark):
        if prediction == "error":
            return 0.0, "generation_error"
        if benchmark == "ambrosia":
            return float(
                (item["SQL"] == "ambiguous") == (prediction == "ambiguous")
            ), None
        if prediction == "ambiguous":
            return 0.0, "false_ambiguity"
        from sqlalchemy import text
        from sqlalchemy.exc import SQLAlchemyError

        try:
            with self.engines[item["db_id"]].connect() as conn:
                conn.exec_driver_sql("SET TRANSACTION READ ONLY")
                predicted = normalize(conn.execute(text(prediction)).fetchall())
            return float(predicted == self.golds[str(item["question_id"])]), None
        except SQLAlchemyError as error:
            original = getattr(error, "orig", error)
            code = getattr(original, "sqlstate", None)
            if (
                not code
                or code.startswith(("08", "28"))
                or code in ("3D000", "57P01", "57P02", "57P03")
            ):
                raise ProviderUnavailable(
                    "Database unavailable during evaluation; no model label saved"
                ) from None
            # SQLSTATE gives actionable feedback without including connection strings.
            return 0.0, f"postgresql:{code}:{type(original).__name__}"

    def evaluate(
        self, item, instruction, profile="hybrid", original=False, benchmark="bird"
    ):
        pilot.check_candidate(instruction, self.manifest)
        cid = candidate_id(instruction, profile, original)
        key = f"{self.phase}:{cid}:{benchmark}:{item['question_id']}"
        if key in self.records:
            return self.records[key]
        self.ledger.example(key)
        before = {
            k: self.ledger.data[k]
            for k in ("task_calls", "prompt_tokens", "completion_tokens")
        }
        self.candidates[cid] = {
            "instruction": instruction,
            "profile": profile,
            "original": original,
        }
        write_json(self.output / "candidates.json", self.candidates)

        async def query():
            await self.prepare_tools(profile)
            self.progress(
                {
                    "phase": "predicting",
                    "question_id": str(item["question_id"]),
                    "candidate": cid,
                    "profile": profile,
                }
            )
            value = (
                f"question: {item['question']}, evidence (may be empty): {item['evidence']}"
                if benchmark == "bird"
                else item["question"]
            )
            token = self.context.set(
                {
                    "phase": self.phase,
                    "question_id": str(item["question_id"]),
                    "db_id": item["db_id"],
                }
            )
            try:
                with settings(
                    self.impl, pilot.PROFILES[profile], instruction, profile, original
                ):
                    response = await self.tools[profile][item["db_id"]].query(value)
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
            return answer, issue

        answer, issue = self.runner.run(query())
        score, execution_issue = self.score(item, answer, benchmark)
        result = {
            "question_id": str(item["question_id"]),
            "db_id": item["db_id"],
            "prediction": answer,
            "score": score,
            "error": issue or execution_issue,
            "usage": {k: self.ledger.data[k] - before[k] for k in before},
        }
        self.records[key] = result
        write_json(self.output / "evaluations.json", self.records)
        self.progress({"saved_evaluations": len(self.records)})
        print(
            f"[{self.phase}] {profile}/{benchmark} question={item['question_id']} score={score:g}",
            flush=True,
        )
        self.runner.run(asyncio.sleep(3))
        return result

    def reflect(self, prompt):
        from openai import OpenAI

        model = self.manifest["teacher"]
        messages = [{"role": "user", "content": prompt}]
        ticket = self.ledger.reserve("teacher", messages, 2048)
        try:
            with OpenAI(
                api_key=self.teacher_config["LLM_API_KEY"],
                base_url=model["base_url"],
                max_retries=0,
                timeout=90,
            ) as client:
                result = client.chat.completions.create(
                    model=model["model"],
                    messages=messages,
                    temperature=0,
                    max_tokens=2048,
                    **model["request_options"],
                )
        except Exception as error:
            append(
                self.folder / "transport_events.jsonl",
                {
                    "role": "teacher",
                    "error_type": type(error).__name__,
                    "at_utc": search.now(),
                },
            )
            raise ProviderUnavailable(
                "Teacher unavailable; reservation preserved, no candidate accepted"
            ) from None
        if result.usage is None:
            raise ProviderUnavailable("Teacher usage absent; reservation preserved")
        self.ledger.settle(
            ticket, result.usage.prompt_tokens, result.usage.completion_tokens
        )
        if not result.choices:
            raise ProviderUnavailable("Teacher choices absent; inspect usage ledger")
        content = result.choices[0].message.content
        append(
            self.folder / "raw_responses.jsonl",
            {
                "phase": "teacher",
                "timestamp_utc": search.now(),
                "messages": messages,
                "raw_response": content,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
            },
        )
        if (
            not isinstance(content, str)
            or not content.strip()
            or result.choices[0].finish_reason == "length"
        ):
            raise ProviderUnavailable(
                "Teacher response empty/truncated; inspect preserved response"
            )
        return content

    def close(self):
        for tools in self.tools.values():
            for tool in tools.values():
                tool.engine.dispose()
        for engine in self.engines.values():
            engine.dispose()
        if self.client is not None:
            self.runner.run(self.client.client.client.close())
        self.runner.close()
        self.ledger.save()


class Adapter:
    propose_new_texts = None

    def __init__(self, session, manifest):
        self.session, self.manifest = session, manifest
        self.fit = {str(i["question_id"]) for i in manifest["cohorts"]["fit"]}
        self.allowed = {
            str(i["question_id"]): i
            for i in manifest["cohorts"]["fit"] + manifest["cohorts"]["inner_val"]
        }

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch

        if set(candidate) != {"sql_rules"} or any(
            self.allowed.get(str(i["question_id"])) != i for i in batch
        ):
            raise ValueError("GEPA may evaluate only the frozen training pilot")
        try:
            pilot.check_candidate(candidate["sql_rules"], self.manifest)
        except ValueError:
            rows = [
                {"prediction": "error", "score": 0.0, "error": "invalid_instruction"}
                for i in batch
            ]
        else:
            rows = [
                self.session.evaluate(item, candidate["sql_rules"]) for item in batch
            ]
        traces = (
            [{"item": item, "result": row} for item, row in zip(batch, rows)]
            if capture_traces
            else None
        )
        return EvaluationBatch(
            outputs=rows, scores=[r["score"] for r in rows], trajectories=traces
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        if components_to_update != ["sql_rules"]:
            raise ValueError("Only SQL instructions are mutable")
        feedback = []
        for trace in eval_batch.trajectories or []:
            item, result = trace["item"], trace["result"]
            if str(item["question_id"]) not in self.fit:
                raise ValueError(
                    "Inner validation/dev feedback must never reach the teacher"
                )
            schema = self.manifest["schema_audit"]["schemas"]["bird"][item["db_id"]][
                "rows"
            ]
            feedback.append(
                {
                    "Inputs": {
                        "question": item["question"],
                        "evidence": item["evidence"],
                        "schema": schema,
                    },
                    "Generated Outputs": result["prediction"],
                    "Feedback": {
                        "correct": bool(result["score"]),
                        "error": result["error"],
                        "reference_SQL": item["SQL"],
                    },
                }
            )
        return {"sql_rules": feedback}


def freeze_selection(output, manifest, records, candidates):
    ids = [str(i["question_id"]) for i in manifest["cohorts"]["inner_val"]]
    eligible = []
    for cid, candidate in candidates.items():
        if candidate["profile"] != "hybrid" or candidate["original"]:
            continue
        rows = [records.get(f"search:{cid}:bird:{q}") for q in ids]
        if all(row is not None for row in rows):
            text = candidate["instruction"]
            eligible.append(
                (
                    sum(r["score"] for r in rows),
                    text == manifest["seed_instruction"],
                    -len(text),
                    cid,
                    text,
                )
            )
    seed = next((r for r in eligible if r[1]), None)
    if seed is None:
        raise ValueError("Seed inner validation incomplete; cannot freeze a selection")
    best = sorted(eligible, key=lambda r: (-r[0], -r[1], -r[2], r[3]))[0]
    selection = {
        "manifest_sha256": manifest["sha256"],
        "instruction": best[4],
        "candidate_id": best[3],
        "inner_val_correct": best[0],
        "seed_correct": seed[0],
        "selected_at_utc": search.now(),
        "external_dev_used": False,
    }
    selection["sha256"] = search.fingerprint(selection)
    write_json(output / "selection.json", selection)
    return selection


def run(output, manifest, phase):
    status = output / phase / "metadata.json"
    if status.exists() and search.read(status).get("status") == "completed":
        if search.read(status).get("manifest_sha256") != manifest["sha256"]:
            raise ValueError("Completed phase identity differs from manifest")
        print("Phase already completed; no DB or API calls.")
        return
    if phase == "search" and (output / "selection.json").exists():
        print("Search already frozen; no calls. Launch --phase confirm separately.")
        return
    selection = None
    if phase == "confirm":
        selection = search.read(output / "selection.json")
        if selection["manifest_sha256"] != manifest["sha256"] or selection[
            "sha256"
        ] != search.fingerprint({k: v for k, v in selection.items() if k != "sha256"}):
            raise ValueError("Selected prompt identity changed")
        if selection["instruction"] == manifest["seed_instruction"]:
            print(
                "No improved prompt selected on internal validation; retain original. No paid confirmation needed."
            )
            return
    session = Session(output, manifest, phase)
    secrets = [
        os.environ.get("DB_PASS", ""),
        os.environ.get("LLM_API_KEY", ""),
        session.teacher_config.get("LLM_API_KEY", ""),
    ]
    stdout, stderr = sys.stdout, sys.stderr
    try:
        with (session.folder / "pilot.log").open("a", encoding="utf-8") as log:
            sys.stdout, sys.stderr = (
                Tee(stdout, log, secrets),
                Tee(stderr, log, secrets),
            )
            session.preflight()
            if phase == "search":
                gepa = pilot.load_gepa()
                adapter = Adapter(session, manifest)
                try:
                    result = gepa.optimize(
                        seed_candidate={"sql_rules": manifest["seed_instruction"]},
                        trainset=manifest["cohorts"]["fit"],
                        valset=manifest["cohorts"]["inner_val"],
                        adapter=adapter,
                        reflection_lm=session.reflect,
                        reflection_prompt_template=REFLECTION,
                        reflection_minibatch_size=3,
                        max_metric_calls=160,
                        stop_callbacks=lambda state: session.ledger.data[
                            "teacher_calls"
                        ]
                        >= manifest["limits"]["search"]["teacher_calls"],
                        use_merge=False,
                        run_dir=str(output / "gepa"),
                        seed=42,
                        raise_on_exception=True,
                        display_progress_bar=False,
                    )
                    write_json(output / "gepa_result.json", result.to_dict())
                except BudgetStop:
                    session.metadata["budget_exhausted"] = True
                freeze_selection(output, manifest, session.records, session.candidates)
            else:
                for original in (True, False):
                    for profile in pilot.PROFILES:
                        instruction = (
                            manifest["seed_instruction"]
                            if original
                            else selection["instruction"]
                        )
                        for b, items in manifest["cohorts"]["dev"].items():
                            for item in items:
                                session.evaluate(
                                    item, instruction, profile, original, b
                                )
            session.metadata.update(
                status="completed", phase=phase, finished_at_utc=search.now()
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
