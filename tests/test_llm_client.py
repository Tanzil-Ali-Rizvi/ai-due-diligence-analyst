"""LLM client behaviour without network: rate-limit retries, error surfacing, reasoning-effort payload."""

import httpx
import pytest

from ddagent import llm as llm_mod
from ddagent.llm import JSONChatLLM, LLMError


def _resp(status, body, headers=None):
    return httpx.Response(status, json=body, headers=headers or {}, request=httpx.Request("POST", "https://x/y"))


def test_waits_on_429_then_succeeds(monkeypatch):
    calls, sleeps = [], []
    replies = [
        _resp(429, {"error": {"message": "slow down"}}, {"retry-after": "3"}),
        _resp(200, {"choices": [{"message": {"content": '{"ok": true}'}}], "usage": {"prompt_tokens": 5}}),
    ]
    monkeypatch.setattr(llm_mod.httpx, "post", lambda *a, **k: (calls.append(k["json"]), replies.pop(0))[1])
    monkeypatch.setattr(llm_mod.time, "sleep", sleeps.append)
    client = JSONChatLLM("https://x", "openai/gpt-oss-120b", "key", reasoning_effort="low")
    assert client.json("s", "u") == {"ok": True}
    assert sleeps == [3.0] and client.usage["rate_limit_waits"] == 1
    assert calls[0]["reasoning_effort"] == "low"


def test_non_retryable_error_shows_provider_message(monkeypatch):
    monkeypatch.setattr(
        llm_mod.httpx, "post", lambda *a, **k: _resp(404, {"error": {"message": "model does not exist"}})
    )
    with pytest.raises(LLMError, match="model does not exist"):
        JSONChatLLM("https://x", "some-model", "key").json("s", "u")


def test_reasoning_effort_only_sent_to_reasoning_models():
    assert JSONChatLLM("https://x", "llama-3.1-8b-instant", "k", reasoning_effort="low")._reasoning_effort is None
