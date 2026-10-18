"""Minimal OpenAI-compatible JSON chat client (Groq, OpenAI, Gemini, Ollama, vLLM ...)."""

from __future__ import annotations

import json
import re
import time

import httpx


class LLMError(RuntimeError):
    pass


class JSONChatLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float = 0.0,
        timeout_s: float = 90.0,
        reasoning_effort: str | None = None,
        max_wait_s: float = 60.0,
    ):
        if not api_key:
            raise LLMError("DD_LLM_API_KEY is not set")
        self.name = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._model, self._temperature, self._timeout = model, temperature, timeout_s
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "rate_limit_waits": 0}
        # Reasoning models (e.g. gpt-oss) accept an effort level; "low" saves many hidden tokens.
        self._reasoning_effort = reasoning_effort if reasoning_effort and "gpt-oss" in model else None
        self._max_wait = max_wait_s

    def json(self, system: str, user: str, retries: int = 6) -> dict:
        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                r = httpx.post(self._url, json=payload, headers=self._headers, timeout=self._timeout)
                if r.status_code in (429, 503) and attempt < retries:
                    # Free tiers cap tokens per minute: wait as long as the provider asks, then retry.
                    wait = min(float(r.headers.get("retry-after") or 2 ** (attempt + 2)), self._max_wait)
                    self.usage["rate_limit_waits"] += 1
                    time.sleep(wait)
                    continue
                if r.status_code >= 400:
                    # Surface the provider's own message (e.g. "model not found", "invalid API key").
                    try:
                        detail = r.json().get("error", {}).get("message", r.text)
                    except ValueError:
                        detail = r.text
                    if r.status_code in (400, 401, 403, 404):
                        raise LLMError(f"{self._model}: HTTP {r.status_code}: {str(detail)[:300]}")
                    r.raise_for_status()
                data = r.json()
                for k in ("prompt_tokens", "completion_tokens"):
                    self.usage[k] += int((data.get("usage") or {}).get(k, 0))
                self.usage["calls"] += 1
                return _parse_json(data["choices"][0]["message"]["content"])
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                last = exc
        raise LLMError(f"LLM call failed: {last}")


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError("no JSON object in LLM output") from None
        return json.loads(m.group(0))


def build_llm(settings) -> JSONChatLLM | None:
    """LLM client for "openai" mode; None (offline rules) otherwise or when no key is configured."""
    if settings.llm_provider == "openai" and settings.llm_api_key:
        return JSONChatLLM(
            settings.llm_base_url,
            settings.llm_model,
            settings.llm_api_key,
            settings.llm_temperature,
            settings.llm_timeout_s,
            settings.llm_reasoning_effort,
        )
    return None
