"""Single choke point for every LLM call.

Everything goes through `call()`. That gives us one place to enforce routing,
throttling, retries, and usage logging — instead of scattering `client.models.
generate_content` across the codebase where none of it can be measured.

Uses the current `google-genai` SDK. The older `google-generativeai` package is
deprecated; most tutorials online still show it. Do not copy those.
"""

from __future__ import annotations

import json
import time
from typing import Type, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from config import (
    GEMINI_API_KEY,
    METRICS_LOG,
    RATE_LIMITS,
    ROUTES,
)
from core.ratelimit import bucket_for, with_retry

T = TypeVar("T", bound=BaseModel)

# A degenerate repetition loop ("Arg-m\n\nArg-m\n\n...") occasionally breaks
# schema-constrained decoding even on a normal 200 response -- a different
# failure class from the 429/503s core.ratelimit.with_retry handles, and
# resampling (not backing off) is the fix. Rare, but real at scale: it
# surfaced during a 60-call repertory eval run.
MAX_PARSE_ATTEMPTS = 2

_client: genai.Client | None = None


def client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it as a Space secret or in .env")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _log(record: dict) -> None:
    record["ts"] = time.time()
    with METRICS_LOG.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def call(
    task: str,
    prompt: str,
    schema: Type[T],
    system: str,
    temperature: float = 0.3,
    path: str = "generative",
) -> T:
    """Route a task to its model, enforce structured output, return a parsed object.

    `task` must be a key in config.ROUTES — that's deliberate. Adding a new LLM
    call forces you to make an explicit cost decision about which model runs it.

    `path` tags the metrics log with "generative" or "deterministic" so
    latency and cost can be split by which pipeline actually produced the
    question (see evaluation/run_eval.py) — the deterministic path only ever
    calls this for task="coaching" (mnemonic + exam tip), never for the
    question/options/answer itself.
    """
    model = ROUTES[task]
    rpm = RATE_LIMITS[model]["rpm"]

    if not bucket_for(model, rpm).acquire():
        raise RuntimeError(f"Local rate limit exceeded for {model}; try again shortly.")

    cfg = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=schema,
        # Flash-Lite thinks by default on some tasks; we don't need it for
        # extraction-shaped work and it inflates output-token cost.
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    resp = None
    for attempt in range(MAX_PARSE_ATTEMPTS):
        started = time.monotonic()
        resp = with_retry(
            lambda: client().models.generate_content(model=model, contents=prompt, config=cfg)
        )
        latency_ms = int((time.monotonic() - started) * 1000)

        usage = getattr(resp, "usage_metadata", None)
        _log(
            {
                "task": task,
                "model": model,
                "path": path,
                "latency_ms": latency_ms,
                "input_tokens": getattr(usage, "prompt_token_count", None),
                "output_tokens": getattr(usage, "candidates_token_count", None),
            }
        )

        parsed = getattr(resp, "parsed", None)
        if parsed is not None:
            return parsed

    # Schema-constrained decoding failed on every attempt. Surface it loudly
    # rather than limping on with half-parsed prose.
    raise ValueError(
        f"Model returned unparseable output for task={task} after {MAX_PARSE_ATTEMPTS} attempts: {resp.text[:400]}"
    )
