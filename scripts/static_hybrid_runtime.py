"""One frozen search case. Started by static_hybrid_search.py after manual launch."""

import argparse
import asyncio
from contextvars import ContextVar
import json
import logging
import os
from pathlib import Path
import sys
from urllib.parse import quote

from dotenv import load_dotenv

from atomic_artifacts import ArtifactWriteError, write_json
from llm_endpoints import account_snapshot, balance_deltas, validate_connection
from run_openrouter_baseline import ProviderUnavailable, Tee
from static_hybrid_search import ROOT, identity, now, read, run_folder, scored_predictions, validate_manifest


def append(path, value):
    try:
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            stream.flush()
    except OSError as error:
        raise ArtifactWriteError("Cannot preserve search response/usage; stopped before continuing") from error


class RecordingClient:
    def __init__(self, client, folder, metadata, context):
        self.client, self.folder, self.metadata, self.context = client, folder, metadata, context

    async def create(self, *args, **kwargs):
        try:
            response = await self.client.create(*args, **kwargs)
        except Exception as error:
            count = self.metadata.get("consecutive_api_errors", 0) + 1
            self.metadata.update(consecutive_api_errors=count, last_api_error_type=type(error).__name__)
            write_json(self.folder / "metadata.json", self.metadata)
            if count >= 3 or getattr(error, "status_code", None) in (401, 402, 403, 404):
                raise ProviderUnavailable("Provider unavailable; inspect account/model before resuming") from None
            raise
        self.metadata.update(consecutive_api_errors=0, last_llm_activity_utc=now())
        messages = args[0] if args else kwargs.get("messages", [])
        context = self.context.get()
        usage = response.usage
        record = {"timestamp_utc": now(), **{k: context[k] for k in ("phase", "question_id", "db_id") if k in context},
                  "messages": [{"type": m.type, "content": m.content} for m in messages],
                  "raw_response": response.content, "prompt_tokens": usage.prompt_tokens,
                  "completion_tokens": usage.completion_tokens}
        append(self.folder / "raw_responses.jsonl", record)
        for key in ("prompt_tokens", "completion_tokens"):
            context[key] = context.get(key, 0) + record[key]
        context["recorded_calls"] = context.get("recorded_calls", 0) + 1
        write_json(self.folder / "metadata.json", self.metadata)
        if not isinstance(response.content, str) or not response.content.strip():
            raise ProviderUnavailable("Empty/non-text paid response preserved; stopped for inspection")
        return response


def checkpoint(folder, expected):
    path = folder / "query_results.json"
    predictions = read(path) if path.exists() else {}
    if not set(predictions) <= expected or any(not isinstance(p, str) or not p.strip() for p in predictions.values()):
        raise ValueError("Invalid or foreign question in checkpoint; do not regenerate the whole run")
    return predictions


async def predict(items, tools, predictions, folder, benchmark, context, progress, *, sleep=asyncio.sleep):
    from benchmarks.response import ToolResponse
    for index, item in enumerate(items, 1):
        qid = str(item["question_id"])
        if qid in predictions:
            continue
        progress({"phase": "predicting", "current_question": qid, "saved": len(predictions)})
        query = (f"question: {item['question']}, evidence (may be empty): {item['evidence']}"
                 if benchmark == "bird" else item["question"])
        usage = {"phase": "question", "question_id": qid, "db_id": item["db_id"]}
        token = context.set(usage)
        try:
            response = ToolResponse.model_validate(await tools[item["db_id"]].query(query))
        finally:
            context.reset(token)
        answer = ("ambiguous" if response.status == "ambiguous" else
                  response.query if response.status == "success" else "error")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Generator produced an empty SQL result; inspect saved responses")
        predictions[qid] = answer
        write_json(folder / "query_results.json", predictions)
        append(folder / "question_usage.jsonl", usage)
        progress({"saved": len(predictions)})
        print(f"[{index}/{len(items)}] question={qid} status={response.status}", flush=True)
        # All search masks keep FEAT_19=false; preserve the historical 3s base interval.
        await sleep(3)
    return predictions


async def run(args):
    output = Path(args.search).resolve()
    if not output.is_relative_to(ROOT / "research_results"):
        raise ValueError("Search must be inside research_results")
    manifest = read(output / "experiment.json")
    validate_manifest(manifest)
    case = next(c for c in manifest["cases"] if c["name"] == args.case)
    items = manifest["cohorts"][args.scope][args.benchmark]
    folder = run_folder(output, args.scope, args.case, args.benchmark)
    config = identity(manifest, args.scope, case, args.benchmark)
    load_dotenv(ROOT / args.env_file, override=True)
    model = manifest["protocol"]["model"]
    validate_connection(os.environ)
    if (os.environ["LLM_MODEL_NAME"] != model["model"] or
            os.environ["LLM_BASE_URL"].rstrip("/") != model["base_url"] or
            os.environ.get("LLM_MODEL_VERSION") != model["model_version"]):
        raise ValueError("Env differs from frozen model identity")
    for flag, enabled in config["feature_flags"].items():
        os.environ[flag] = str(enabled).lower()
    os.environ.update(MAX_RETRIES="5", OPTIMISTIC_AMBIGUITY_FALLBACK="false",
                      LLM_BASE_INTERVAL="3", LLM_MIN_INTERVAL="3")
    sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
    from autogen_core.models import UserMessage
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from baseline_connection import build_tools, database_engine
    from adv_text2sql.mcp_servers.text2sql_tool.src import text2sql_implementation as implementation

    class Generator(implementation.Text2SQLGenerator):
        def __init__(self, db_uri, llm_client):
            super().__init__(db_uri, llm_client)
            self.engine.dispose()
            self.engine = database_engine(db_uri)

    folder.mkdir(parents=True, exist_ok=True)
    previous = read(folder / "metadata.json") if (folder / "metadata.json").exists() else None
    if previous:
        if previous["configuration"] != config:
            raise ValueError("Frozen case identity differs from existing metadata")
        if previous["status"] == "completed":
            print("Already completed; no calls made.")
            return
        append(folder / "attempt_history.jsonl", previous)
    elif (folder / "query_results.json").exists():
        raise ValueError("Checkpoint without metadata")
    predictions = checkpoint(folder, {str(i["question_id"]) for i in items})
    dataset = folder / "dataset.json"
    if dataset.exists() and read(dataset) != items:
        raise ValueError("Saved cohort differs from frozen questions")
    write_json(dataset, items)
    metadata = {"configuration": config, "status": "running", "questions": len(items),
                "started_at_utc": now(), "phase": "starting", "saved": len(predictions)}
    write_json(folder / "metadata.json", metadata)

    def progress(update):
        metadata.update(update)
        write_json(folder / "metadata.json", metadata)

    client, tools = None, {}
    stdout, stderr, previous_cwd = sys.stdout, sys.stderr, Path.cwd()
    secrets = [os.environ.get(k, "") for k in ("DB_PASS", "LLM_API_KEY")]
    secrets += [quote(v, safe="") for v in secrets if v]
    with (folder / f"{args.benchmark}.log").open("a", encoding="utf-8") as log:
        sys.stdout, sys.stderr = Tee(stdout, log, secrets), Tee(stderr, log, secrets)
        try:
            os.chdir(folder)
            implementation.setup_logging()
            context = ContextVar("search_question", default={"phase": "probe"})
            remaining = [i for i in items if str(i["question_id"]) not in predictions]
            if remaining:
                metadata["balance_before"] = account_snapshot(model["api_backend"], os.environ["LLM_API_KEY"])
                if not metadata["balance_before"].get("is_available", True):
                    raise RuntimeError("Account balance unavailable")
                client = OpenAIChatCompletionClient(
                    model=model["model"], base_url=model["base_url"], api_key=os.environ["LLM_API_KEY"],
                    temperature=0, max_tokens=4096, max_retries=2, timeout=90,
                    model_info={"json_output": False, "function_calling": True, "vision": False,
                                "family": "unknown", "structured_output": False}, **model["request_options"])
                recording = RecordingClient(client, folder, metadata, context)
                probe = await recording.create([UserMessage(content="Reply with exactly OK.", source="user")],
                                               extra_create_args={"max_tokens": 16})
                if probe.content.strip().upper().rstrip(".") != "OK":
                    raise RuntimeError("Provider probe did not return OK")
                context.set({"phase": "build"})
                tools = await build_tools(remaining, Generator, recording, progress)
                await predict(items, tools, predictions, folder, args.benchmark, context, progress)
            progress({"phase": "evaluating"})
            if args.benchmark == "bird":
                from benchmarks.evaluate_bird import run_evaluation
                report = run_evaluation(predictions, str(dataset), "127.0.0.1:5444")
            else:
                from research_metrics import ambrosia_report
                report = ambrosia_report(items, predictions)
            report["generation_errors"] = sum(p == "error" for p in predictions.values())
            write_json(folder / "report.json", report)
            scored_predictions(items, predictions, report, args.benchmark)
            metadata.update(status="completed", phase="completed")
            print(f"Completed {args.scope}/{args.case}/{args.benchmark}; questions={len(items)}", flush=True)
        except BaseException as error:
            metadata.update(status="failed", error_type=type(error).__name__)
            raise
        finally:
            metadata["finished_at_utc"] = now()
            if metadata.get("balance_before"):
                try:
                    metadata["balance_after"] = account_snapshot(model["api_backend"], os.environ["LLM_API_KEY"])
                    metadata["balance_delta_this_attempt"] = balance_deltas(metadata["balance_before"], metadata["balance_after"])
                except Exception as error:
                    metadata["balance_snapshot_error"] = type(error).__name__
            try:
                write_json(folder / "metadata.json", metadata)
            finally:
                for tool in tools.values():
                    tool.engine.dispose()
                if client is not None:
                    await client.close()
                logger = logging.getLogger("text2sql_tool")
                for handler in list(logger.handlers):
                    logger.removeHandler(handler)
                    handler.close()
                os.chdir(previous_cwd)
                sys.stdout, sys.stderr = stdout, stderr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", required=True)
    parser.add_argument("--env-file", default=".env.deepseek")
    parser.add_argument("--scope", choices=("train", "dev", "full"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--benchmark", choices=("bird", "ambrosia"), required=True)
    asyncio.run(run(parser.parse_args()))
