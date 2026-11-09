"""Recover the interrupted seed control from verified raw responses + score log.

Offline only. Replays the real generator against already-paid responses with
exact message equality and no network, then restores measured scores from log.
All corrupted bytes are archived; unknown usage is charged conservatively.
"""

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch

import durable_prompt_store as store
import prompt_comparison as campaign
import prompt_pilot_runtime as runtime
from run_feature_ablation import suite_lock
from sql_output_contract import guarded_response

JOB = "sql__mipro_few__r1"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def raw_prefix(data):
    rows = []
    prefix = b""
    bad = b""
    for line in data.splitlines(keepends=True):
        if bad:
            bad += line
            continue
        try:
            value = json.loads(line)
        except (ValueError, UnicodeError):
            bad = line
            continue
        rows.append(value)
        prefix += line
    if not rows or not bad or any(bad):
        raise ValueError("Expected a valid response prefix followed only by null bytes")
    return rows, prefix, bad


def replay(item, rows):
    sys.path[:0] = [str(campaign.ROOT), str(campaign.ROOT / "src")]
    from adv_text2sql.mcp_servers.text2sql_tool.src import (
        text2sql_implementation as impl,
    )

    question = (
        f"question: {item['question']}, evidence (may be empty): {item['evidence']}"
    )
    if len(rows) != 2 or any(
        str(r["question_id"]) != campaign.item_id(item) or r["db_id"] != item["db_id"]
        for r in rows
    ):
        raise ValueError("Expected one complete taxonomy + SQL response pair")
    template = campaign.all_templates()["TAXONOMY_DETECTION_PROMPT"]
    head, tail = template.split("{db_schema}")
    head, tail = head.format(user_query=question), tail.format(user_query=question)
    taxonomy = rows[0]["messages"][0]["content"]
    if not taxonomy.startswith(head) or not taxonomy.endswith(tail):
        raise ValueError("Saved taxonomy prompt differs from frozen generator")
    schema = taxonomy[len(head) : -len(tail)]
    tool = object.__new__(impl.Text2SQLGenerator)
    tool.db_schema = schema
    tool.system_prompt = rows[1]["messages"][0]["content"]
    cursor = 0

    async def saved(messages, *args, **kwargs):
        nonlocal cursor
        if cursor >= len(rows):
            raise AssertionError("Replay attempted an unsaved model call")
        record = rows[cursor]
        if [{"type": m.type, "content": m.content} for m in messages] != record[
            "messages"
        ]:
            raise AssertionError(
                "Replay messages differ from the recorded paid request"
            )
        cursor += 1
        return SimpleNamespace(content=record["raw_response"])

    tool._llm_call_with_retry = saved
    with (
        asyncio.Runner() as runner,
        runtime.settings(impl, [14, 32]),
        patch.object(impl.logger, "disabled", True),
        patch.object(
            socket.socket,
            "connect",
            side_effect=AssertionError("Offline replay cannot access network"),
        ),
    ):
        response = runner.run(tool.query(question))
    if cursor != len(rows):
        raise ValueError("Saved calls were not consumed by the real generator")
    response, issue = guarded_response(response)
    if response["status"] != "success":
        raise ValueError(
            "This recovery supports the recorded successful SQL responses only"
        )
    return response["query"], issue


def plan(root):
    manifest = json.loads((root / "experiment.json").read_bytes())
    if manifest["sha256"] != campaign.search.fingerprint(
        {k: v for k, v in manifest.items() if k != "sha256"}
    ):
        raise ValueError("Frozen manifest changed")
    for name, expected in manifest["files_sha256"].items():
        if digest((campaign.ROOT / name).read_bytes()) != expected:
            raise ValueError("Frozen runtime source changed")
    folder = root / "jobs" / JOB
    phase = folder / "search"
    targets = [
        root / "budget.json",
        folder / "candidates.json",
        folder / "evaluations.json",
        phase / "budget.json",
        phase / "metadata.json",
    ]
    for p in targets:
        data = p.read_bytes()
        if not data or any(data):
            raise ValueError(
                f"Recovery requires confirmed all-null corruption: {p.name}"
            )
    raw = phase / "raw_responses.jsonl"
    rows, prefix, bad = raw_prefix(raw.read_bytes())
    score_pairs = re.findall(
        r"\[" + re.escape(JOB) + r"/search\] bird (\d+): ([01])\s*",
        (phase / "comparison.log").read_text(encoding="utf8"),
    )
    inner = manifest["cohorts"]["sql"]["inner_val"]
    if len(rows) != 2 * len(score_pairs) or len(score_pairs) >= len(inner):
        raise ValueError("Response pairs and completed score lines disagree")
    if [q for q, _ in score_pairs] != [
        campaign.item_id(i) for i in inner[: len(score_pairs)]
    ]:
        raise ValueError("Only the initial seed control prefix may be reconstructed")
    seed = campaign.candidate(manifest["seed_instructions"]["sql"])
    cid = campaign.cid(seed)
    records = {}
    for n, (qid, score) in enumerate(score_pairs):
        pair = rows[2 * n : 2 * n + 2]
        answer, issue = replay(inner[n], pair)
        key = f"search:{cid}:hybrid:False:bird:{qid}"
        records[key] = {
            "question_id": qid,
            "db_id": inner[n]["db_id"],
            "prediction": answer,
            "score": float(score),
            "error": issue,
            "effective_demo_ids": [],
            "usage": {
                "task_calls": 2,
                "prompt_tokens": sum(r["prompt_tokens"] for r in pair),
                "completion_tokens": sum(r["completion_tokens"] for r in pair),
            },
            "recovery": {
                "prediction": "exact real-generator replay of saved responses",
                "score": "original score printed after evaluation, not rejudged",
                "execution_diagnostic_available": False,
            },
        }
    completed = []
    for job in manifest["jobs"]:
        if job["name"] == JOB:
            continue
        p = root / "jobs" / job["name"] / "search"
        if not (p / "budget.json").exists():
            continue
        meta = json.loads((p / "metadata.json").read_bytes())
        budget = json.loads((p / "budget.json").read_bytes())
        if meta.get("status") != "completed" or budget["pending"]:
            raise ValueError("Unexpected additional unfinished job")
        completed.append((job["name"], meta, budget))
    if len(completed) != 3:
        raise ValueError("Expected the three completed SQL controls")
    latest = max(
        datetime.fromisoformat(meta["finished_at_utc"]).timestamp()
        for _, meta, _ in completed
    )
    damaged_at = max(p.stat().st_mtime for p in targets)
    local = {
        k: 0
        for k in (
            "task_calls",
            "teacher_calls",
            "tokens",
            "examples",
            "active_seconds",
            "prompt_tokens",
            "completion_tokens",
            "task_tokens",
            "teacher_tokens",
        )
    }
    # A conservative allowance for one interrupted question (including fallback
    # and retry possibilities). Actual usage of the torn tail is not invented.
    max_context = max(
        len(json.dumps(r["messages"], ensure_ascii=False).encode("utf8")) for r in rows
    )
    templates = sum(len(v.encode("utf8")) for v in campaign.all_templates().values())
    max_question = max(len(str(i).encode("utf8")) for i in inner)
    cap = max_context + templates + max_question + len(bad) + 40000 + 4096 + 1024
    unknown_calls = 6
    unknown_tokens = unknown_calls * cap
    local.update(
        task_calls=len(rows) + unknown_calls,
        active_seconds=math.ceil(damaged_at - latest) + 1,
        prompt_tokens=sum(r["prompt_tokens"] for r in rows),
        completion_tokens=sum(r["completion_tokens"] for r in rows),
    )
    local["task_tokens"] = local["prompt_tokens"] + local["completion_tokens"]
    local["tokens"] = local["task_tokens"] + unknown_tokens
    local["unknown_usage_tokens"] = unknown_tokens
    local["unknown_task_calls"] = unknown_calls
    local["pending"] = {}
    local["example_keys"] = [
        f"search:{cid}:hybrid:False:bird:{campaign.item_id(i)}"
        for i in inner[: len(records) + 1]
    ]
    local["examples"] = len(local["example_keys"])
    local["unknown_usage_events"] = [
        {
            "role": "task",
            "actual_usage": None,
            "budget_tokens": unknown_tokens,
            "budget_calls": unknown_calls,
            "per_request_budget_allowance": cap,
            "policy": "Conservative interrupted-question allowance after disk corruption; not measured usage",
        }
    ]
    if (
        local["tokens"] > manifest["limits"]["tokens"]
        or local["active_seconds"] > manifest["limits"]["active_seconds"]
    ):
        raise ValueError("Conservative reconstruction exhausts the original budget")
    global_ = deepcopy(local)
    global_["example_keys"] = [JOB + ":search:" + k for k in local["example_keys"]]
    for name, meta, budget in completed:
        for k in (
            "task_calls",
            "teacher_calls",
            "tokens",
            "active_seconds",
            "prompt_tokens",
            "completion_tokens",
            "task_tokens",
            "teacher_tokens",
        ):
            global_[k] += budget[k]
        global_["unknown_usage_tokens"] += budget.get("unknown_usage_tokens", 0)
        global_["unknown_task_calls"] += budget.get("unknown_task_calls", 0)
        global_["unknown_usage_events"] += budget.get("unknown_usage_events", [])
        global_["example_keys"] += [
            name + ":search:" + k for k in budget["example_keys"]
        ]
        if meta.get("budget_exhausted"):
            global_["example_keys"].append(
                "disk_recovery:conservative_rejected_example:" + name
            )
    global_["examples"] = len(global_["example_keys"])
    global_["active_seconds"] += len(completed)
    metadata = {
        "status": "stopped",
        "phase": "recovered_from_disk_corruption",
        "manifest_sha256": manifest["sha256"],
        "saved_evaluations": len(records),
        "error_type": "StateCorruptionRecovered",
        "question_id": campaign.item_id(inner[len(records)]),
        "recovery_at_utc": campaign.search.now(),
        "unknown_usage_tokens": unknown_tokens,
        "calls_not_replayed_for_payment": len(rows),
    }
    outputs = {
        root / "budget.json": global_,
        folder / "candidates.json": {cid: seed},
        folder / "evaluations.json": records,
        phase / "budget.json": local,
        phase / "metadata.json": metadata,
    }
    return (
        outputs,
        raw,
        prefix,
        {
            "manifest_sha256": manifest["sha256"],
            "recovered_questions": len(records),
            "known_api_calls": len(rows),
            "known_tokens": local["task_tokens"],
            "unknown_budget_tokens": unknown_tokens,
            "unknown_budget_calls": unknown_calls,
            "next_question": metadata["question_id"],
            "script_sha256": digest(Path(__file__).read_bytes()),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=campaign.DEFAULT_OUTPUT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = campaign.pilot.output_path(args.output)
    journal = root / "recoveries/disk_recovery_20261106"
    with suite_lock(root):
        if (journal / "applied.json").exists():
            print("Already restored; no writes or paid calls.")
            return
        outputs, raw, prefix, summary = plan(root)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if not args.apply:
            print("DRY RUN: no artifact writes, no DB/API calls.")
            return
        if journal.exists():
            raise ValueError(
                "Partial recovery exists; inspect backups before proceeding"
            )
        journal.mkdir(parents=True)
        backup = {}
        for p in [*outputs, raw]:
            rel = p.relative_to(root)
            target = journal / "original" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            data = p.read_bytes()
            store.write_bytes(target, data)
            backup[rel.as_posix()] = digest(data)
        store.write_json(
            journal / "plan.json",
            {
                **summary,
                "original_sha256": backup,
                "policy": "Original frozen runtime and completed jobs unchanged; scores restored only from original score log",
            },
        )
        for p, value in outputs.items():
            store.write_json(p, value)
        store.write_bytes(raw, prefix)
        store.write_json(
            journal / "applied.json",
            {
                **summary,
                "state": "applied",
                "at_utc": campaign.search.now(),
                "restored_sha256": {
                    p.relative_to(root).as_posix(): digest(p.read_bytes())
                    for p in [*outputs, raw]
                },
            },
        )
        print("Restored. Corrupted bytes archived; no paid calls made.")


if __name__ == "__main__":
    main()
