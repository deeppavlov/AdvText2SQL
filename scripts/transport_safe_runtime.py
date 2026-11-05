"""Stop client failures before generator fallbacks; preserve the frozen SQL behavior."""

import argparse
import asyncio
from pathlib import Path

from atomic_artifacts import write_json
from run_openrouter_baseline import ProviderUnavailable
import resilient_hybrid_runtime as resilient
import static_hybrid_runtime as legacy
import static_hybrid_search as search

VERSION = "fail-fast-transport-v1"
CALL_TIMEOUT = 180
ORIGINAL_RECORDING_CLIENT = legacy.RecordingClient


class ResponseBoundary:
    def __init__(self, client, folder, metadata, context, timeout=CALL_TIMEOUT):
        self.client, self.folder = client, Path(folder)
        self.metadata, self.context, self.timeout = metadata, context, timeout

    def abort(self, reason, error_type, status_code=None):
        current = self.context.get()
        event = {
            "at_utc": search.now(),
            "transport_version": VERSION,
            **{
                k: current[k] for k in ("phase", "question_id", "db_id") if k in current
            },
            "reason": reason,
            "error_type": error_type,
            "status_code": status_code,
            "checkpoint_written": False,
            "automatic_question_retries": 0,
        }
        self.metadata.update(
            last_api_error_type=error_type,
            consecutive_api_errors=1,
            transport_failure=event,
        )
        # The exception must bypass broad `except Exception` in the SQL generator.
        # Even an unexpected journal failure must not turn into a model label.
        try:
            legacy.append(self.folder / "transport_events.jsonl", event)
            write_json(self.folder / "metadata.json", self.metadata)
        finally:
            raise ProviderUnavailable(
                f"Client boundary stopped ({reason}); question is not checkpointed. "
                "Resume manually after provider recovery."
            ) from None

    async def create(self, *args, **kwargs):
        try:
            result = await asyncio.wait_for(
                self.client.create(*args, **kwargs), timeout=self.timeout
            )
        except Exception as error:
            code = getattr(error, "status_code", None)
            code = code if isinstance(code, int) else None
            reason = (
                "call_timeout"
                if isinstance(error, TimeoutError)
                else "client_exception"
            )
            self.abort(reason, type(error).__name__, code)
        if result is None:
            self.abort("null_response", "InvalidClientResponse")
        usage = getattr(result, "usage", None)
        if usage is None or any(
            not isinstance(getattr(usage, key, None), int)
            or isinstance(getattr(usage, key, None), bool)
            or getattr(usage, key) < 0
            for key in ("prompt_tokens", "completion_tokens")
        ):
            self.abort("invalid_usage", "InvalidClientResponse")
        if not hasattr(result, "content"):
            self.abort("missing_content", "InvalidClientResponse")
        return result


class TransportRecordingClient(ORIGINAL_RECORDING_CLIENT):
    def __init__(self, client, folder, metadata, context):
        boundary = ResponseBoundary(client, folder, metadata, context)
        super().__init__(boundary, Path(folder), metadata, context)

    async def create(self, *args, **kwargs):
        # Empty paid text is still recorded by the original recorder before its
        # terminal ProviderUnavailable; JSON/SQL errors remain model outcomes.
        try:
            return await super().create(*args, **kwargs)
        except Exception as error:
            self.client.abort("recorder_exception", type(error).__name__)


async def run(args):
    from transport_safe_campaign import validate

    manifest = search.read(Path(args.search) / "experiment.json")
    validate(manifest)
    original = legacy.RecordingClient
    legacy.RecordingClient = TransportRecordingClient
    try:
        await resilient.run(args)
    finally:
        legacy.RecordingClient = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", required=True)
    parser.add_argument("--env-file", default=".env.deepseek")
    parser.add_argument("--scope", choices=("train", "dev"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--benchmark", choices=search.BENCHMARKS, required=True)
    asyncio.run(run(parser.parse_args()))
