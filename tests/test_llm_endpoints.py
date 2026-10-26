import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from llm_endpoints import account_snapshot, api_backend, balance_deltas, client_options, validate_connection


class EndpointTests(unittest.TestCase):
    def test_validate_native_model_and_key_pairing(self):
        config = {"LLM_BASE_URL": "https://api.deepseek.com", "LLM_MODEL_NAME": "deepseek-flash",
                  "LLM_API_KEY": "sk-offline-placeholder"}
        self.assertEqual(validate_connection(config), "deepseek")
        self.assertEqual(api_backend("https://api.deepseek.com/v1/"), "deepseek")
        for override in ({"LLM_MODEL_NAME": "deepseek/deepseek-v4.1-flash-1017"},
                         {"LLM_API_KEY": "sk-or-offline-placeholder"},
                         {"LLM_BASE_URL": "https://openrouter.ai/api/v1"}):
            with self.assertRaises(ValueError):
                validate_connection({**config, **override})
        with self.assertRaises(ValueError):
            validate_connection(config, provider="another-provider")

    def test_native_balance_uses_deepseek_endpoint_and_preserves_currency(self):
        response = {"is_available": True, "balance_infos": [
            {"currency": "CNY", "total_balance": "20.00", "granted_balance": "0", "topped_up_balance": "20"}]}
        with patch("llm_endpoints.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as request:
            before = account_snapshot("deepseek", "sk-offline-placeholder")
        self.assertEqual(request.call_args.args[0].full_url, "https://api.deepseek.com/user/balance")
        self.assertNotIn("usage", before)
        after = {**before, "balance_infos": [{"currency": "CNY", "total_balance": "19.80"}]}
        self.assertEqual(balance_deltas(before, after), {"CNY": 0.2})
        self.assertNotIn("USD", balance_deltas(before, after))
        self.assertEqual(balance_deltas(None, after), {})
        self.assertEqual(balance_deltas({"usage": 1}, {"usage": 1.25}), {"USD": 0.25})

    def test_openrouter_options_preserve_legacy_and_optional_pinning(self):
        self.assertEqual(client_options("openrouter"), {"reasoning_effort": "none"})
        options = client_options("openrouter", protocol=True, provider="offline-provider")
        self.assertEqual(options["extra_body"]["reasoning"], {"enabled": False})
        self.assertFalse(options["extra_body"]["provider"]["allow_fallbacks"])


class NativeTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_client_serializes_native_toggle_without_network(self):
        import httpx
        from autogen_core.models import UserMessage
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        captured = []

        def respond(request):
            captured.append((str(request.url), json.loads(request.content)))
            return httpx.Response(200, json={
                "id": "offline", "object": "chat.completion", "created": 0, "model": "deepseek-flash",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}})

        client = OpenAIChatCompletionClient(
            model="deepseek-flash", base_url="https://api.deepseek.com", api_key="sk-offline-placeholder",
            model_info={"json_output": False, "function_calling": True, "vision": False,
                        "family": "unknown", "structured_output": False},
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
            **client_options("deepseek", protocol=True))
        try:
            response = await client.create([UserMessage(content="OK", source="user")])
            self.assertEqual(response.content, "OK")
            self.assertEqual(captured[0][0], "https://api.deepseek.com/chat/completions")
            self.assertEqual(captured[0][1]["thinking"], {"type": "disabled"})
            self.assertNotIn("reasoning_effort", captured[0][1])
            self.assertNotIn("provider", captured[0][1])
        finally:
            await client.close()


if __name__ == "__main__":
    unittest.main()
