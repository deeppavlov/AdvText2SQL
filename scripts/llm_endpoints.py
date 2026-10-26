"""Provider-specific request and accounting settings without exposing credentials."""

from decimal import Decimal
import json
import urllib.request


def api_backend(base_url):
    normalized = base_url.rstrip("/")
    if normalized == "https://openrouter.ai/api/v1":
        return "openrouter"
    if normalized in ("https://api.deepseek.com", "https://api.deepseek.com/v1"):
        return "deepseek"
    raise ValueError("LLM_BASE_URL must point to OpenRouter or the official DeepSeek API")


def validate_connection(config, provider=None):
    backend = api_backend(config.get("LLM_BASE_URL", ""))
    model = config.get("LLM_MODEL_NAME", "")
    key = config.get("LLM_API_KEY", "")
    if not model or not key:
        raise ValueError("Set LLM_MODEL_NAME and LLM_API_KEY in the selected env file")
    if backend == "deepseek":
        if provider:
            raise ValueError("--provider is only supported with OpenRouter")
        if "/" in model:
            raise ValueError("Direct DeepSeek uses LLM_MODEL_NAME=deepseek-flash, without deepseek/ prefix")
        if key.startswith("sk-or-"):
            raise ValueError("The direct DeepSeek API needs a DeepSeek key, not an OpenRouter key")
    elif not key.startswith("sk-or-"):
        raise ValueError("OpenRouter needs an sk-or- key; use --env-file .env.deepseek for a direct DeepSeek key")
    return backend


def client_options(backend, protocol=False, provider=None):
    if backend == "deepseek":
        # DeepSeek enables thinking by default. reasoning_effort=none is not its native toggle.
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    options = {"reasoning_effort": "none"}
    if protocol:
        body = {"reasoning": {"enabled": False}}
        if provider:
            body["provider"] = {"only": [provider], "allow_fallbacks": False}
        options["extra_body"] = body
    return options


def account_snapshot(backend, api_key):
    url = ("https://api.deepseek.com/user/balance" if backend == "deepseek"
           else "https://openrouter.ai/api/v1/key")
    request = urllib.request.Request(url, headers={"Authorization": "Bearer " + api_key})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)
    if backend == "deepseek":
        return {"backend": backend, "is_available": data["is_available"],
                "balance_infos": [{key: item.get(key) for key in
                                   ("currency", "total_balance", "granted_balance", "topped_up_balance")}
                                  for item in data["balance_infos"]]}
    return {key: data["data"].get(key) for key in ("usage", "limit", "limit_remaining")}


def balance_deltas(before, after):
    """Positive means balance consumed. Keep currencies separate; do not assume CNY=USD."""
    if before is None or after is None:
        return {}
    if "balance_infos" not in before:
        if before.get("usage") is None or after.get("usage") is None:
            return {}
        return {"USD": float(Decimal(str(after["usage"])) - Decimal(str(before["usage"])))}
    initial = {item["currency"]: Decimal(item["total_balance"]) for item in before["balance_infos"]}
    return {item["currency"]: float(initial[item["currency"]] - Decimal(item["total_balance"]))
            for item in after.get("balance_infos", []) if item["currency"] in initial}
