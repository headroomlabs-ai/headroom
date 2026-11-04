"""Regression tests: a non-streaming caller must never receive an SSE body.

The buffered Anthropic path copies the upstream response headers wholesale,
``content-type`` included. When the upstream answers a ``stream``-less request
with ``text/event-stream``, that body reached the caller as a ``200`` it could
not parse — the reply was complete, just in the wrong wire format, and the turn
was lost.

The buffered-stream (CCR) path already refused this shape (#2952). These tests
pin the same protection on the plain non-streaming path, plus the recovery that
turns a lost turn into a normal reply.
"""

from __future__ import annotations

import json

import httpx
import pytest

from headroom.proxy.nonstream_sse_policy import (
    is_event_stream,
    media_type,
    should_recover_sse_reply,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SSE_REPLY = (
    "event: message_start\n"
    'data: {"type":"message_start","message":{"id":"msg_sse_recovered",'
    '"type":"message","role":"assistant","model":"claude-sonnet-4-6",'
    '"content":[],"usage":{"input_tokens":11,"output_tokens":0}}}\n'
    "\n"
    "event: content_block_start\n"
    'data: {"type":"content_block_start","index":0,'
    '"content_block":{"type":"text","text":""}}\n'
    "\n"
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"text_delta","text":"recovered body"}}\n'
    "\n"
    "event: content_block_stop\n"
    'data: {"type":"content_block_stop","index":0}\n'
    "\n"
    "event: message_delta\n"
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    '"usage":{"output_tokens":4}}\n'
    "\n"
    "event: message_stop\n"
    'data: {"type":"message_stop"}\n'
    "\n"
)

# Upstream headers as they actually arrive through Anthropic's edge — the
# correlation headers here are what a client uses to report and dedup a turn,
# so the fix must not drop them while correcting the content-type.
_UPSTREAM_SSE_HEADERS = {
    "content-type": "text/event-stream; charset=utf-8",
    "request-id": "req_011CeC1JTMS8egPL3FBteQay",
    "anthropic-ratelimit-requests-remaining": "49",
    "cf-ray": "9a1b2c3d4e5f6789-GRU",
    "server": "cloudflare",
}


# ---------------------------------------------------------------------------
# Pure policy
# ---------------------------------------------------------------------------


class TestMediaTypeParsing:
    @pytest.mark.parametrize(
        ("header", "expected"),
        [
            ("text/event-stream", "text/event-stream"),
            ("text/event-stream; charset=utf-8", "text/event-stream"),
            ("Text/Event-Stream", "text/event-stream"),
            ("  text/event-stream  ", "text/event-stream"),
            ("application/json", "application/json"),
            (None, ""),
            ("", ""),
        ],
    )
    def test_parameters_and_case_are_normalized(self, header, expected) -> None:
        assert media_type(header) == expected

    def test_is_event_stream_only_matches_sse(self) -> None:
        assert is_event_stream("text/event-stream; charset=utf-8") is True
        assert is_event_stream("application/json") is False
        assert is_event_stream(None) is False


class TestShouldRecoverSseReply:
    """The gate has three deliberate negative arms; each is a separate risk."""

    def test_recovers_sse_200_for_a_non_streaming_caller(self) -> None:
        assert (
            should_recover_sse_reply(
                client_requested_stream=False,
                status_code=200,
                content_type="text/event-stream",
            )
            is True
        )

    def test_streaming_caller_is_untouched(self) -> None:
        """A streaming caller asked for SSE — rewriting it would break the turn."""
        assert (
            should_recover_sse_reply(
                client_requested_stream=True,
                status_code=200,
                content_type="text/event-stream",
            )
            is False
        )

    def test_json_reply_is_untouched(self) -> None:
        assert (
            should_recover_sse_reply(
                client_requested_stream=False,
                status_code=200,
                content_type="application/json",
            )
            is False
        )

    @pytest.mark.parametrize("status", [429, 500, 529])
    def test_error_status_is_passed_through(self, status) -> None:
        """A non-200 carries an upstream error payload the client should see."""
        assert (
            should_recover_sse_reply(
                client_requested_stream=False,
                status_code=status,
                content_type="text/event-stream",
            )
            is False
        )


# ---------------------------------------------------------------------------
# Handler end-to-end — the wiring is where the bug lived
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def _make_proxy_client() -> TestClient:
    config = ProxyConfig(
        optimize=True,
        mode="token",
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    return TestClient(create_app(config))


def _post_non_streaming(client: TestClient):
    return client.post(
        "/v1/messages",
        headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )


def _stub_upstream(proxy, response: httpx.Response) -> None:
    async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
        return response

    proxy._retry_request = _fake_retry


class TestNonStreamingCallerNeverGetsAnEventStream:
    def test_sse_reply_is_recovered_as_json(self) -> None:
        """Before the fix this returned text/event-stream and the SDK reported
        an empty or malformed response despite a complete reply."""
        with _make_proxy_client() as client:
            _stub_upstream(
                client.app.state.proxy,
                httpx.Response(
                    200,
                    headers=_UPSTREAM_SSE_HEADERS,
                    content=_SSE_REPLY.encode(),
                ),
            )
            response = _post_non_streaming(client)

        assert response.status_code == 200
        assert "event-stream" not in response.headers["content-type"]
        assert response.headers["content-type"].startswith("application/json")

        payload = response.json()
        assert payload["id"] == "msg_sse_recovered"
        assert payload["content"][0]["text"] == "recovered body"

    def test_upstream_correlation_headers_survive_recovery(self) -> None:
        with _make_proxy_client() as client:
            _stub_upstream(
                client.app.state.proxy,
                httpx.Response(
                    200,
                    headers=_UPSTREAM_SSE_HEADERS,
                    content=_SSE_REPLY.encode(),
                ),
            )
            response = _post_non_streaming(client)

        assert response.headers["request-id"] == "req_011CeC1JTMS8egPL3FBteQay"

    def test_unrecoverable_event_stream_is_refused_not_forwarded(self) -> None:
        """No message_start means no message. Refuse loudly rather than hand
        the caller a 200 it cannot parse."""
        with _make_proxy_client() as client:
            _stub_upstream(
                client.app.state.proxy,
                httpx.Response(
                    200,
                    headers=_UPSTREAM_SSE_HEADERS,
                    content=b'event: ping\ndata: {"type":"ping"}\n\n',
                ),
            )
            response = _post_non_streaming(client)

        assert response.status_code == 502
        assert "event-stream" not in response.headers["content-type"]
        assert response.json()["error"]["type"] == "upstream_protocol_error"

    def test_ordinary_json_reply_is_unaffected(self) -> None:
        """Control: the fix must be inert on the overwhelmingly common path."""
        with _make_proxy_client() as client:
            _stub_upstream(
                client.app.state.proxy,
                httpx.Response(
                    200,
                    json={
                        "id": "msg_plain",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "ok"}],
                        "usage": {"input_tokens": 10, "output_tokens": 3},
                    },
                ),
            )
            response = _post_non_streaming(client)

        assert response.status_code == 200
        assert json.loads(response.content)["id"] == "msg_plain"
