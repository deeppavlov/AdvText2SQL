"""Check the selected OpenRouter or direct DeepSeek endpoint (one billable probe)."""

import argparse
import asyncio
from pathlib import Path

from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import dotenv_values
from llm_endpoints import client_options, validate_connection


async def check(env_path: Path) -> int:
    if not env_path.is_file():
        print(f"Configuration file not found: {env_path}")
        return 2

    config = dotenv_values(env_path)
    key = config.get("LLM_API_KEY") or ""
    if not key:
        print(f"Set LLM_API_KEY in {env_path} to your OpenRouter API key.")
        print("Do not paste the key into chat or commit it to Git.")
        return 2
    try:
        backend = validate_connection(config)
    except ValueError as error:
        print(f"{error}; no request sent.")
        return 2

    model = config.get("LLM_MODEL_NAME") or "openai/gpt-5.4-mini"
    parameters = {
        "model": model,
        "base_url": config["LLM_BASE_URL"],
        "api_key": key,
        "temperature": 0,
        "max_tokens": 16,
        "max_retries": 0,
        "timeout": 30,
        "model_info": {
            "json_output": False,
            "function_calling": True,
            "vision": False,
            "family": "unknown",
            "structured_output": False,
        },
        **client_options(backend, protocol=True),
    }

    client = OpenAIChatCompletionClient(**parameters)
    try:
        response = await client.create(
            [UserMessage(content="Reply with exactly OK.", source="user")]
        )
        print(f"Connected to {backend}: {model}")
        print("Response:", str(response.content).replace(key, "[REDACTED]"))
        print("Usage:", response.usage)
        return 0
    except Exception as error:
        message = str(error).replace(key, "[REDACTED]")
        print(f"Connection failed: {type(error).__name__}: {message}")
        return 1
    finally:
        await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env.openrouter"))
    arguments = parser.parse_args()
    raise SystemExit(asyncio.run(check(arguments.env_file)))
