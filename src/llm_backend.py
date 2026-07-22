"""Pluggable LLM backend for the MarxGraph pipeline.

Set MARXGRAPH_BACKEND=groq to use Groq's cheap Llama-3.3-70B instead of the
Anthropic API. Both are called via plain HTTP so no extra SDK is required.

    export MARXGRAPH_BACKEND=groq
    export GROQ_API_KEY=...
    export MARXGRAPH_MODEL=openai/gpt-oss-120b          # optional override

    export MARXGRAPH_BACKEND=anthropic                 # default
    export ANTHROPIC_API_KEY=...
    export MARXGRAPH_MODEL=claude-sonnet-4-6            # optional override
"""

import json
import os
import time

import requests

BACKEND = os.environ.get("MARXGRAPH_BACKEND", "anthropic").lower()

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    # Groq deprecated llama-3.3-70b-versatile; gpt-oss-120b is their current
    # general-purpose/reasoning model. Check https://console.groq.com/docs/models
    # (and the deprecations page) before a long run, as this list moves fast.
    "groq": "openai/gpt-oss-120b",
}
MODEL = os.environ.get("MARXGRAPH_MODEL", DEFAULT_MODELS[BACKEND])

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


class LLMError(Exception):
    pass


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        raw = raw.rsplit("```", 1)[0]
    return raw.strip()


def _call_groq(system: str, user: str, max_tokens: int) -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY not set")
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=60,
    )
    if resp.status_code == 429:
        raise LLMError("rate_limited")
    if not resp.ok:
        body = resp.text[:500]
        if "json_validate_failed" in body or "json_generate_failed" in body:
            raise LLMError(f"json_truncated: {body}")
        raise LLMError(f"groq {resp.status_code}: {body}")
    return resp.json()["choices"][0]["message"]["content"]


def _call_anthropic(system: str, user: str, max_tokens: int) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise LLMError("ANTHROPIC_API_KEY not set")
    resp = requests.post(
        ANTHROPIC_URL,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={
            "model": MODEL,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        timeout=60,
    )
    if resp.status_code == 429:
        raise LLMError("rate_limited")
    if not resp.ok:
        raise LLMError(f"anthropic {resp.status_code}: {resp.text[:500]}")
    blocks = resp.json()["content"]
    return "".join(b["text"] for b in blocks if b.get("type") == "text")


def call_json(system: str, user: str, max_tokens: int = 2000, retries: int = 6) -> dict:
    """Call the configured backend and parse a JSON object from the response.

    On a truncated/invalid-JSON response (common with smaller models when the
    output is long — the passage produced more claims than max_tokens allowed
    for), retries with a larger token budget rather than giving up, since a
    plain retry at the same budget would just fail the same way again.

    Rate-limit retries get their own budget (up to 4 of the `retries` attempts)
    with longer backoff, since a burst of concurrent workers can trip a
    short-window rate limit even when daily/TPM budgets are nowhere near used.
    """
    fn = _call_groq if BACKEND == "groq" else _call_anthropic
    budget = max_tokens
    rate_limit_attempts = 0
    last_exc = None
    for attempt in range(retries):
        try:
            raw = fn(system, user, budget)
            return json.loads(_strip_fences(raw))
        except LLMError as exc:
            msg = str(exc)
            if msg.startswith("rate_limited"):
                rate_limit_attempts += 1
                time.sleep(min(10 * rate_limit_attempts, 60))
                last_exc = exc
                continue
            if msg.startswith("json_truncated"):
                budget = min(int(budget * 1.75), 8000)
                last_exc = exc
                continue
            raise
        except json.JSONDecodeError as exc:
            last_exc = exc
            continue
    raise last_exc or LLMError("failed after retries")
