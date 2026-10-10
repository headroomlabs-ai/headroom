"""Input volume must be the provider's own billed count, not Headroom's estimate.

Headroom's local tokenizer is a stand-in for Claude's private one (15-25% under
Claude 4.x, 35-38% under Claude 5.x on real transcripts) and only ever sees the
forwarded messages, never the system prompt or tool definitions. Every streamed
Anthropic turn (all Claude Code traffic) used to fall back to that estimate for
"input after Headroom", the dashboard input total and the licence usage report,
so none of them reconciled with the Anthropic Console.

The Anthropic fixture is a real recorded stream: a second call on a cached
~8.9K-token system prompt. Anthropic billed 19 uncached + 8,857 cache-read
tokens; the messages Headroom counts locally are a single short user turn.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest

from headroom.proxy.cost import CostTracker
from headroom.proxy.handlers.openai import _passthrough_usage_from_json
from headroom.proxy.outcome import RequestOutcome, emit_request_outcome
from headroom.proxy.prometheus_metrics import PrometheusMetrics
from headroom.proxy.provider_usage import (
    billed_input_for_provider,
    billed_input_from_usage,
    usage_int,
)
from headroom.proxy.server import HeadroomProxy
from headroom.telemetry.reporter import UsageReporter

FIXTURE = Path(__file__).parent / "fixtures" / "anthropic" / "stream_cached_turn.sse"
# From the fixture's message_start usage: input 19 + cache_read 8857 + cache_write 0.
ANTHROPIC_BILLED = 19 + 8_857
# What Headroom's tokenizer would see: one short user message.
LOCAL_ESTIMATE = 17


# ── provider shape semantics ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "provider,payload,expected",
    [
        # Anthropic: input_tokens is the uncached tail; cache buckets are disjoint.
        (
            "anthropic",
            {
                "usage": {
                    "input_tokens": 19,
                    "cache_read_input_tokens": 8_857,
                    "cache_creation_input_tokens": 100,
                }
            },
            8_976,
        ),
        # OpenAI chat: prompt_tokens already includes cached_tokens.
        (
            "openai",
            {"usage": {"prompt_tokens": 5_000, "prompt_tokens_details": {"cached_tokens": 4_000}}},
            5_000,
        ),
        # OpenAI Responses: input_tokens already includes cached_tokens.
        (
            "openai",
            {"usage": {"input_tokens": 5_000, "input_tokens_details": {"cached_tokens": 4_000}}},
            5_000,
        ),
        # Gemini: promptTokenCount includes cachedContentTokenCount.
        (
            "gemini",
            {"usageMetadata": {"promptTokenCount": 7_000, "cachedContentTokenCount": 6_000}},
            7_000,
        ),
        # OpenAI-compatible gateways (LiteLLM) mirror top-level Anthropic-named
        # cache keys beside an INCLUSIVE input_tokens. Keys alone must not flip
        # the dialect, or the cached part is counted twice.
        ("openai", {"usage": {"input_tokens": 5_000, "cache_read_input_tokens": 4_000}}, 5_000),
        (
            "vertex:anthropic",
            {"usage": {"input_tokens": 50, "cache_read_input_tokens": 4_000}},
            4_050,
        ),
        # Nothing reported: 0 means "unknown", never "zero tokens billed".
        ("anthropic", {"id": "x"}, 0),
        ("openai", None, 0),
    ],
)
def test_billed_input_from_usage_per_provider_shape(provider, payload, expected) -> None:  # noqa: ANN001
    assert billed_input_from_usage(payload, provider) == expected


def test_billed_input_for_provider_adds_anthropic_cache_buckets_only() -> None:
    assert billed_input_for_provider("anthropic", 19, cache_read=8_857, cache_write=0) == 8_876
    assert billed_input_for_provider("bedrock", 2, cache_read=360_949, cache_write=840) == 361_791
    # OpenAI/Gemini headline figures are inclusive: never add the cache split.
    assert billed_input_for_provider("openai", 5_000, cache_read=4_000) == 5_000
    assert billed_input_for_provider("gemini", 7_000, cache_read=6_000) == 7_000
    assert billed_input_for_provider("anthropic", None) == 0


def test_passthrough_normaliser_returns_anthropic_total_not_uncached_tail() -> None:
    usage = _passthrough_usage_from_json(
        {"usage": {"input_tokens": 19, "output_tokens": 14, "cache_read_input_tokens": 8_857}},
        "anthropic",
    )
    # Callers derive uncached as input - read - write, which is only right on the total.
    assert usage["input_tokens"] == ANTHROPIC_BILLED
    assert usage["input_tokens"] - usage["cache_read_input_tokens"] == 19


# ── recorded Anthropic stream through the real SSE parser and finalizer ──


def _replay_recorded_stream(provider: str, raw: bytes, chunk_size: int = 97) -> dict:
    """Feed recorded SSE bytes through the proxy's own usage parser, in odd-sized
    chunks so events split across reads the way they do on the wire."""
    proxy = object.__new__(HeadroomProxy)
    state = {
        "input_tokens": None,
        "output_tokens": None,
        "cache_read_input_tokens": None,
        "cache_creation_input_tokens": None,
        "cache_creation_ephemeral_5m_input_tokens": None,
        "cache_creation_ephemeral_1h_input_tokens": None,
        "total_bytes": 0,
        "sse_buffer": bytearray(),
        "ttfb_ms": 3.0,
    }
    for i in range(0, len(raw), chunk_size):
        chunk = raw[i : i + chunk_size]
        state["total_bytes"] += len(chunk)
        state["sse_buffer"].extend(chunk)
        usage = proxy._parse_sse_usage_from_buffer(state, provider)
        for key, value in (usage or {}).items():
            if key in state:
                state[key] = value
    return state


def _finalize(provider: str, stream_state: dict, optimized_tokens: int) -> RequestOutcome:
    handler = object.__new__(HeadroomProxy)
    handler.config = SimpleNamespace(log_full_messages=False)
    outcomes: list[RequestOutcome] = []

    async def record(outcome):  # noqa: ANN001, ANN202
        outcomes.append(outcome)

    handler._record_request_outcome = record
    asyncio.run(
        handler._finalize_stream_response(
            body={"messages": [{"role": "user", "content": "Which job reruns target t17?"}]},
            provider=provider,
            model="claude-sonnet-4-6" if provider == "anthropic" else "gpt-5",
            request_id=f"req_{provider}_billed",
            original_tokens=optimized_tokens,
            optimized_tokens=optimized_tokens,
            tokens_saved=0,
            transforms_applied=[],
            optimization_latency=1.0,
            stream_state=stream_state,
            start_time=0.0,
        )
    )
    (outcome,) = outcomes
    return outcome


def test_recorded_anthropic_stream_records_anthropics_billed_input() -> None:
    state = _replay_recorded_stream("anthropic", FIXTURE.read_bytes())
    assert state["input_tokens"] == 19
    assert state["cache_read_input_tokens"] == 8_857

    outcome = _finalize("anthropic", state, optimized_tokens=LOCAL_ESTIMATE)

    assert outcome.provider_input_tokens == ANTHROPIC_BILLED
    assert outcome.uncached_input_tokens == 19
    assert outcome.cache_read_tokens == 8_857
    # The local estimate stays the local estimate: savings math must never mix
    # it with the provider's scale.
    assert outcome.optimized_tokens == LOCAL_ESTIMATE


def test_anthropic_stream_without_usage_stays_estimated() -> None:
    # An error before message_start: no provider count exists, so the outcome
    # must not invent one (provider_input_tokens 0 => counted as estimated).
    state = _replay_recorded_stream("anthropic", b"")
    outcome = _finalize("anthropic", state, optimized_tokens=LOCAL_ESTIMATE)
    assert outcome.provider_input_tokens == 0


def test_openai_stream_records_inclusive_prompt_tokens() -> None:
    frame = {
        "id": "c1",
        "choices": [],
        "usage": {
            "prompt_tokens": 12_000,
            "completion_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 11_000},
        },
    }
    raw = f"data: {json.dumps(frame)}\n\ndata: [DONE]\n\n".encode()
    state = _replay_recorded_stream("openai", raw)

    outcome = _finalize("openai", state, optimized_tokens=9_000)

    assert outcome.provider_input_tokens == 12_000


# ── the full chain: outcome -> metrics/cost tracker -> licence usage report ──


def _handler() -> SimpleNamespace:
    cost = CostTracker()
    return SimpleNamespace(
        metrics=PrometheusMetrics(cost_tracker=cost, stateless=True),
        cost_tracker=cost,
        logger=None,
    )


def _reporter_payload(cost_tracker: CostTracker) -> dict:
    reporter = object.__new__(UsageReporter)
    reporter._proxy = SimpleNamespace(cost_tracker=cost_tracker)
    reporter._last_report_time = None
    reporter._last_tokens_saved_by_model = {}
    reporter._last_tokens_sent_by_model = {}
    reporter._last_requests_by_model = {}
    reporter._last_provider_counters = {}
    reporter._license_key = "k"
    reporter._cloud_url = "https://cloud.example"
    reporter._license_info = None
    sent: list[dict] = []

    class _Client:
        async def post(self, url, json, **kwargs):  # noqa: A002, ANN001, ANN003, ANN202
            sent.append(json)
            return SimpleNamespace(status_code=200, json=lambda: {})

    async def _get_client():  # noqa: ANN202
        return _Client()

    reporter._get_client = _get_client
    anyio.run(reporter._report_usage)
    (payload,) = sent
    return payload


def test_streamed_claude_turn_reaches_dashboard_and_licence_report_as_billed() -> None:
    state = _replay_recorded_stream("anthropic", FIXTURE.read_bytes())
    outcome = _finalize("anthropic", state, optimized_tokens=LOCAL_ESTIMATE)
    handler = _handler()

    asyncio.run(emit_request_outcome(handler, outcome))

    m = handler.metrics
    assert m.tokens_input_total == ANTHROPIC_BILLED
    assert m.tokens_input_provider_reported_total == ANTHROPIC_BILLED
    assert m.tokens_input_estimated_total == 0
    assert m.requests_input_provider_reported == 1

    payload = _reporter_payload(handler.cost_tracker)
    # "Input after Headroom" now reconciles with the Anthropic Console
    # (input + cache read + cache write) for this turn.
    assert payload["tokens_after"] == ANTHROPIC_BILLED
    assert payload["tokens_after_provider_reported"] == ANTHROPIC_BILLED
    assert payload["tokens_after_estimated"] == 0
    assert payload["requests_provider_reported"] == 1
    assert payload["cache_read_tokens"] == 8_857
    assert payload["uncached_input_tokens"] == 19


def test_estimated_input_is_reported_as_estimated() -> None:
    handler = _handler()
    outcome = RequestOutcome(
        request_id="req_no_usage",
        provider="bedrock",
        model="anthropic.claude-sonnet-4-6",
        original_tokens=500,
        optimized_tokens=400,
        output_tokens=0,
        tokens_saved=100,
        attempted_input_tokens=500,
    )

    asyncio.run(emit_request_outcome(handler, outcome))

    assert handler.metrics.tokens_input_estimated_total == 400
    assert handler.metrics.requests_input_estimated == 1
    payload = _reporter_payload(handler.cost_tracker)
    assert payload["tokens_after"] == 400
    assert payload["tokens_after_provider_reported"] == 0
    assert payload["tokens_after_estimated"] == 400


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [
        (12, 0, 12),
        ("7", 0, 7),
        (-3, 0, 0),
        (None, 0, 0),
        (None, 41, 41),
        ("n/a", 5, 5),
        (True, 0, 0),
    ],
)
def test_usage_int_coerces_upstream_counts(value, default, expected) -> None:
    assert usage_int(value, default) == expected
