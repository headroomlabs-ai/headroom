"""Regression tests for #3019 — a response-cache hit must not hand the client
an unusable HTTP 200.

Three separate defects met to produce the reported failure:

1. The cached entry stores the *producing* upstream's response headers
   verbatim. Replaying ``transfer-encoding: chunked`` onto a fresh
   fixed-length response makes the client parse plain JSON as chunked frames
   (RFC 9112 §6.1: Transfer-Encoding overrides Content-Length), so it reads an
   empty body out of a 200.
2. The Anthropic ``cache.set`` had no ``stream`` gate while ``cache.get`` did,
   and the cache key has no ``stream`` component — so a buffered-CCR turn
   (client asked for ``stream: true``, upstream forced to ``stream: false``)
   could store a response that a later non-streaming caller was served.
3. Nothing logged the hit, and the PERF line rendered no ``cached=`` field, so
   a served-from-cache turn was indistinguishable from a turn that died.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.cache.backends import InMemoryBackend  # noqa: E402
from headroom.cache.compression_store import (  # noqa: E402
    get_compression_store,
    reset_compression_store,
)
from headroom.ccr.tool_injection import create_ccr_tool_definition  # noqa: E402
from headroom.proxy.helpers import sanitize_forwarded_response_headers  # noqa: E402
from headroom.proxy.models import CacheEntry  # noqa: E402
from headroom.proxy.outcome import RequestOutcome, emit_request_outcome  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


class _CapturingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self) -> list[str]:
        return [record.getMessage() for record in self.records]


@pytest.fixture
def proxy_log_capture():
    """Capture ``headroom.proxy`` records.

    ``_setup_file_logging`` sets ``propagate = False`` on this logger, so
    ``caplog`` (which hangs off the root) never sees them — the same reason
    ``tests/test_anthropic_stage_timings.py`` attaches its own handler.
    """
    target = logging.getLogger("headroom.proxy")
    handler = _CapturingHandler()
    previous_level = target.level
    target.addHandler(handler)
    target.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        target.removeHandler(handler)
        target.setLevel(previous_level)


# --------------------------------------------------------------------------
# 1. The shared header sanitiser
# --------------------------------------------------------------------------


class TestSanitizeForwardedResponseHeaders:
    def test_drops_every_wire_framing_header(self):
        cleaned = sanitize_forwarded_response_headers(
            {
                "content-encoding": "gzip",
                "content-length": "412",
                "transfer-encoding": "chunked",
                "connection": "keep-alive",
                "keep-alive": "timeout=5",
                "server": "cloudflare",
                "request-id": "req_abc",
                "anthropic-ratelimit-requests-remaining": "42",
            }
        )
        assert cleaned == {
            "request-id": "req_abc",
            "anthropic-ratelimit-requests-remaining": "42",
        }

    def test_matches_case_insensitively_but_preserves_surviving_casing(self):
        cleaned = sanitize_forwarded_response_headers(
            {"Transfer-Encoding": "chunked", "Request-Id": "req_abc"}
        )
        assert cleaned == {"Request-Id": "req_abc"}

    def test_extra_names_are_dropped_too(self):
        cleaned = sanitize_forwarded_response_headers(
            {"content-type": "text/event-stream", "request-id": "req_abc"},
            "content-type",
        )
        assert cleaned == {"request-id": "req_abc"}

    def test_accepts_httpx_headers(self):
        cleaned = sanitize_forwarded_response_headers(
            httpx.Headers({"transfer-encoding": "chunked", "request-id": "req_abc"})
        )
        assert "transfer-encoding" not in cleaned
        assert cleaned["request-id"] == "req_abc"


# --------------------------------------------------------------------------
# 2. Replaying a poisoned cache entry
# --------------------------------------------------------------------------


def _cache_config() -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=True,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        image_optimize=False,
    )


_CACHED_BODY = json.dumps(
    {
        "id": "msg_cached",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "served from cache"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
).encode()


def _poisoned_entry() -> CacheEntry:
    """A cache entry carrying the producing upstream's wire framing."""
    return CacheEntry(
        response_body=_CACHED_BODY,
        response_headers={
            "transfer-encoding": "chunked",
            "content-length": "999999",
            "content-encoding": "gzip",
            "connection": "keep-alive",
            "content-type": "text/event-stream",
            "request-id": "req_from_the_producing_turn",
        },
        created_at=datetime.now(),
        ttl_seconds=3600,
    )


def test_cache_hit_replays_a_body_the_client_can_actually_read(proxy_log_capture):
    """The replayed 200 must carry no stale framing and an intact JSON body."""
    with patch("headroom.proxy.server.AnyLLMBackend"):
        app = create_app(_cache_config())
        with TestClient(app) as client:
            proxy = client.app.state.proxy
            proxy.cache.get = AsyncMock(return_value=_poisoned_entry())
            proxy._retry_request = AsyncMock(
                side_effect=AssertionError("a cache hit must not contact the upstream")
            )

            resp = client.post(
                "/v1/messages",
                headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert resp.status_code == 200
    # The body survived intact — this is what an empty 200 looked like.
    assert resp.json()["content"][0]["text"] == "served from cache"

    replayed = {key.lower(): value for key, value in resp.headers.items()}
    # None of the producing turn's framing may ride along.
    assert "transfer-encoding" not in replayed
    assert "content-encoding" not in replayed
    assert "connection" not in replayed
    # content-type is the caller's, not the producing turn's (#2952).
    assert replayed["content-type"] == "application/json"
    # content-length describes THIS body, not the stored one.
    assert replayed["content-length"] == str(len(_CACHED_BODY))
    # Non-framing upstream metadata still passes through.
    assert replayed["request-id"] == "req_from_the_producing_turn"

    # The hit is no longer silent, and the PERF line marks it as cache-served.
    messages = proxy_log_capture.messages()
    assert any("RESPONSE-CACHE-HIT" in message for message in messages)
    assert any(" PERF " in message and "cached=1" in message for message in messages)


# --------------------------------------------------------------------------
# 3. A buffered-CCR turn must not populate the cache
# --------------------------------------------------------------------------


def _ccr_cache_config() -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=True,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=True,
        ccr_handle_responses=True,
        ccr_context_tracking=False,
        ccr_event_streaming=False,
        image_optimize=False,
    )


def test_buffered_ccr_turn_does_not_write_the_response_cache():
    """A client ``stream: true`` turn is converted to a buffered ``stream:
    false`` upstream call. Its reply is shaped by that flip plus CCR tool
    injection, and the cache key has no ``stream`` component — so storing it
    would let a later non-streaming caller be served a response built for a
    request it never made (#3019).
    """
    upstream_response = {
        "id": "msg_buffered",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "buffered reply"}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }

    # The buffered conversion needs a marker retrieval could actually expand;
    # a resident `headroom_retrieve` alone keeps the request streaming (#3071).
    reset_compression_store()
    store = get_compression_store(backend=InMemoryBackend())
    marker = store.store(
        original=json.dumps({"earlier": "tool output"}),
        compressed="{}",
        original_item_count=1,
    )

    with patch("headroom.proxy.server.AnyLLMBackend"):
        app = create_app(_ccr_cache_config())
        with TestClient(app) as client:
            proxy = client.app.state.proxy
            proxy._stream_response = AsyncMock(
                side_effect=AssertionError("buffered CCR must not take the live stream path")
            )
            proxy.cache.set = AsyncMock()
            forwarded_bodies: list[dict] = []

            async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
                forwarded_bodies.append(json.loads(json.dumps(body)))
                return httpx.Response(200, json=upstream_response)

            proxy._retry_request = _fake_retry  # type: ignore[assignment]

            resp = client.post(
                "/v1/messages",
                headers={
                    "x-api-key": "test-key",
                    "anthropic-version": "2023-06-01",
                    "accept": "text/event-stream",
                },
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 64,
                    "stream": True,
                    "tools": [create_ccr_tool_definition("anthropic")],
                    "messages": [
                        {"role": "user", "content": f"hello (earlier at <<ccr:{marker}>>)"}
                    ],
                },
            )

    assert resp.status_code == 200, resp.text
    # The conversion really happened — otherwise this test proves nothing.
    assert forwarded_bodies and forwarded_bodies[0]["stream"] is False
    # ...and nothing was written to the response cache.
    proxy.cache.set.assert_not_awaited()
    reset_compression_store()


# --------------------------------------------------------------------------
# 4. The PERF line marks a cache-served turn
# --------------------------------------------------------------------------


class _Metrics:
    async def record_request(self, **kwargs):
        return None

    async def record_failed(self, provider):
        return None


class _Handler:
    def __init__(self):
        self.metrics = _Metrics()
        self.cost_tracker = None
        self.logger = None


def _perf_line(capture: _CapturingHandler) -> str:
    for message in capture.messages():
        if " PERF " in message:
            return message
    raise AssertionError("no PERF log line captured")


def _outcome(*, from_response_cache: bool) -> RequestOutcome:
    return RequestOutcome(
        request_id="req-1",
        provider="anthropic",
        model="claude-sonnet-4-6",
        original_tokens=0,
        optimized_tokens=0,
        output_tokens=0,
        tokens_saved=0,
        attempted_input_tokens=0,
        from_response_cache=from_response_cache,
    )


def test_perf_line_marks_a_response_cache_hit(proxy_log_capture):
    asyncio.run(emit_request_outcome(_Handler(), _outcome(from_response_cache=True)))
    assert "cached=1" in _perf_line(proxy_log_capture)


def test_perf_line_is_unchanged_for_an_ordinary_turn(proxy_log_capture):
    """Appended only on a hit, so existing PERF parsers see no new field."""
    asyncio.run(emit_request_outcome(_Handler(), _outcome(from_response_cache=False)))
    assert "cached=" not in _perf_line(proxy_log_capture)


def test_perf_analyzer_reads_the_cached_field():
    from headroom.perf.analyzer import _parse_kv

    parsed = _parse_kv("model=claude-sonnet-4-6 transforms=none client=claude cached=1")
    assert parsed["cached"] == "1"
    # ``transforms=`` is parsed last and swallows the rest of the line, so the
    # new trailing field has to survive that split the way ``client=`` does.
    assert parsed["client"] == "claude"
    assert parsed["transforms"] == "none"
    assert parsed["model"] == "claude-sonnet-4-6"
