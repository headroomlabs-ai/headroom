"""5xx accounting for the OpenAI and Gemini handler paths.

The >=500 routing in emit_request_outcome is covered provider-agnostically by
tests/test_outcome_records_5xx_as_failed.py. This file pins the per-provider
contract: every retry-fed handler site now threads the real upstream status
onto RequestOutcome (mirroring the Anthropic sites), so an exhausted 5xx is
recorded as failed and skips the savings/cost funnel rather than inflating the
save-rate. Each case below corresponds to one wired call site.
"""

import asyncio

import pytest

from headroom.proxy.outcome import RequestOutcome, emit_request_outcome


class _Metrics:
    def __init__(self):
        self.failed = []
        self.requested = []

    async def record_failed(self, provider):
        self.failed.append(provider)

    async def record_request(self, **kwargs):
        self.requested.append(kwargs)


class _Handler:
    # Minimal stub: if emit_request_outcome escapes the >=500 guard it reaches
    # the success funnel and AttributeErrors on cost_tracker/logger.
    def __init__(self):
        self.metrics = _Metrics()


# (label, provider, status) — one entry per retry-fed site wired in this change.
WIRED_SITES = [
    ("openai.chat", "openai", 529),
    ("openai.chat", "openai", 503),
    ("openai.responses", "openai", 503),
    ("openai.passthrough", "anthropic", 503),
    ("gemini.generateContent", "gemini", 529),
    ("gemini.generateContent", "gemini", 503),
    ("gemini.allNonText", "gemini", 503),
    ("gemini.countTokens", "gemini", 503),
]


@pytest.mark.parametrize("label,provider,status", WIRED_SITES)
def test_exhausted_5xx_recorded_as_failed(label, provider, status):
    handler = _Handler()
    outcome = RequestOutcome(
        request_id=f"req-{label}",
        provider=provider,
        model="test-model",
        status_code=status,
        original_tokens=100,
        optimized_tokens=100,
        output_tokens=0,
        tokens_saved=0,
        attempted_input_tokens=100,
    )
    asyncio.run(emit_request_outcome(handler, outcome))
    assert handler.metrics.failed == [provider]
    assert handler.metrics.requested == []


# --- handler-level wiring test -------------------------------------------
#
# The parametrized cases above hand-build RequestOutcome, so they would not
# catch a regression where a handler stops passing status_code=response.
# status_code. This test invokes a real handler entry method end to end with a
# transport that exhausts retries on 503, then asserts the outcome the handler
# emitted carries that 503 — i.e. deleting the wiring would fail here.
#
# gemini countTokens is the cheapest retry-fed path: it never streams, has no
# cache, and skips the compression pipeline under optimize=False.
#
# Status note: _retry_request (server.py) returns a 429/529 *verbatim* on
# exhaustion (RETRYABLE_OVERLOAD_STATUSES) but re-raises a generic 503 as
# HTTPStatusError. So the response-fed RequestOutcome is only reachable for the
# overload statuses — 529 is the live scenario this change targets ("529
# Overloaded surfaced after retry exhaustion").


class _Always529Transport:
    """httpx transport returning 529 every call (overload, exhausts retries)."""

    def __init__(self):
        self.calls = 0

    async def handle_async_request(self, request):
        self.calls += 1
        async for _ in request.stream:  # drain body
            pass
        return _httpx().Response(529, json={"error": {"message": "overloaded"}})

    async def aclose(self):  # let AsyncClient.aclose() tear down cleanly
        pass


def _httpx():
    import httpx

    return httpx


def _count_tokens_request(body_bytes: bytes):
    """Real Starlette Request over an ASGI scope (exercises body parsing)."""
    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1beta/models/gemini-2.0-flash:countTokens",
        "raw_path": b"/v1beta/models/gemini-2.0-flash:countTokens",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
    }
    sent = {"done": False}

    async def receive():
        if sent["done"]:
            return {"type": "http.disconnect"}
        sent["done"] = True
        return {"type": "http.request", "body": body_bytes, "more_body": False}

    return Request(scope, receive)


