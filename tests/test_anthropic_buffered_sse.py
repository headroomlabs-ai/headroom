"""A non-streaming turn must never be answered with an event stream (#3130).

Claude Code retries a failed streaming turn as ``stream: false``. The buffered
Anthropic path forwarded the upstream response headers wholesale, so when the
upstream answered that JSON request with ``content-type: text/event-stream``
the SDK got a wire format it never asked for and lost a complete, already-paid
turn:

    API returned an empty or malformed response (HTTP 200) ... content-type
    event-stream, body is an event stream (the non-streaming request was
    answered with a stream), 8756 bytes

Two defects, fixed on both sides:

* the request went out contradicting itself — ``stream: false`` in the body,
  ``Accept: text/event-stream`` in the headers (the narrow CCR-only rewrite
  from #3078 never covered a client-originated non-stream turn), and
* the response was relayed verbatim instead of being adapted to the JSON the
  caller asked for.

Reconstruction is deliberately strict: a partial stream must fail loudly as a
502 rather than be handed back as a successful — and silently truncated —
message.
"""

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

COMPLETE_SSE = (
    "event: message_start\n"
    'data: {"type":"message_start","message":{"id":"msg_1","type":"message",'
    '"role":"assistant","model":"claude-sonnet-4-6","content":[],'
    '"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,'
    '"output_tokens":1,"cache_read_input_tokens":2,'
    '"cache_creation_input_tokens":3}}}\n\n'
    "event: content_block_start\n"
    'data: {"type":"content_block_start","index":0,'
    '"content_block":{"type":"text","text":""}}\n\n'
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"text_delta","text":"hello"}}\n\n'
    "event: content_block_stop\n"
    'data: {"type":"content_block_stop","index":0}\n\n'
    "event: message_delta\n"
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn",'
    '"stop_sequence":null},"usage":{"output_tokens":5}}\n\n'
    "event: message_stop\n"
    'data: {"type":"message_stop"}\n\n'
)

# Everything up to — but not including — the terminal event.
TRUNCATED_SSE = COMPLETE_SSE.split("event: message_delta")[0]

ERROR_SSE = (
    COMPLETE_SSE.split("event: message_delta")[0] + "event: error\n"
    'data: {"type":"error","error":{"type":"overloaded_error",'
    '"message":"Overloaded"}}\n\n'
)

JSON_REPLY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-6",
    "content": [{"type": "text", "text": "hello"}],
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    },
}


def _config() -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        memory_enabled=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )


def _drive(
    *,
    upstream: httpx.Response,
    accept: str | None = "text/event-stream",
    stream: bool = False,
) -> tuple[httpx.Response, dict[str, object]]:
    """Run one turn against a canned upstream reply.

    Returns the client-facing response and what went upstream.
    """
    seen: dict[str, object] = {}
    app = create_app(_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            sent = json.loads(body) if isinstance(body, (str, bytes)) else body
            seen["stream"] = sent.get("stream")
            seen["headers"] = dict(headers or {})
            return upstream

        proxy._retry_request = _fake_retry  # type: ignore[assignment]

        headers = {"x-api-key": "test-key", "anthropic-version": "2023-06-01"}
        if accept is not None:
            headers["accept"] = accept
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "stream": stream,
                "messages": [{"role": "user", "content": "go"}],
            },
            headers=headers,
        )
    return resp, seen


def _sse_response(body: str, **extra_headers: str) -> httpx.Response:
    headers = {"content-type": "text/event-stream", **extra_headers}
    return httpx.Response(200, content=body.encode(), headers=headers)


def _accepts(headers: dict) -> list[str]:
    return [v for k, v in headers.items() if k.lower() == "accept"]


# --------------------------------------------------------------------------- #
# Request side: a stream:false body must not carry an SSE-only Accept
# --------------------------------------------------------------------------- #
def test_non_stream_turn_asks_upstream_for_json() -> None:
    _, seen = _drive(upstream=httpx.Response(200, json=JSON_REPLY))

    assert seen["stream"] is False
    assert _accepts(seen["headers"]) == ["application/json"]  # type: ignore[arg-type]


def test_non_stream_turn_replaces_rather_than_appends_accept() -> None:
    _, seen = _drive(upstream=httpx.Response(200, json=JSON_REPLY), accept="TEXT/EVENT-STREAM")

    values = _accepts(seen["headers"])  # type: ignore[arg-type]
    assert values == ["application/json"]
    assert not any("event-stream" in v.lower() for v in values)


def test_non_stream_turn_without_client_accept_still_asks_for_json() -> None:
    _, seen = _drive(upstream=httpx.Response(200, json=JSON_REPLY), accept=None)

    assert _accepts(seen["headers"]) == ["application/json"]  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Response side: SSE at 200 for a JSON request is adapted, not relayed
# --------------------------------------------------------------------------- #
def test_event_stream_answer_is_adapted_to_json() -> None:
    resp, _ = _drive(upstream=_sse_response(COMPLETE_SSE))

    assert resp.status_code == 200
    assert "application/json" in resp.headers["content-type"]
    body = resp.json()
    assert body["type"] == "message"
    assert body["content"] == [{"type": "text", "text": "hello"}]
    assert body["stop_reason"] == "end_turn"


def test_adapted_reply_preserves_usage_for_accounting() -> None:
    resp, _ = _drive(upstream=_sse_response(COMPLETE_SSE))

    usage = resp.json()["usage"]
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 5
    assert usage["cache_read_input_tokens"] == 2
    assert usage["cache_creation_input_tokens"] == 3


def test_adapted_reply_carries_no_streaming_only_index() -> None:
    """``index`` is a response-delta field; Anthropic rejects it on replay."""
    resp, _ = _drive(upstream=_sse_response(COMPLETE_SSE))

    assert all("index" not in block for block in resp.json()["content"])


def test_adapted_reply_drops_cdn_and_framing_headers() -> None:
    resp, _ = _drive(
        upstream=_sse_response(
            COMPLETE_SSE,
            **{
                "server": "cloudflare",
                "cf-ray": "abc123",
                "cf-cache-status": "DYNAMIC",
                "request-id": "req_011CeC1JTMS8egPL3FBteQay",
                "anthropic-ratelimit-requests-remaining": "42",
            },
        )
    )

    lowered = {k.lower() for k in resp.headers}
    assert "server" not in lowered
    assert not any(k.startswith("cf-") for k in lowered)
    # Provenance the caller legitimately needs survives.
    assert resp.headers["request-id"] == "req_011CeC1JTMS8egPL3FBteQay"
    assert resp.headers["anthropic-ratelimit-requests-remaining"] == "42"


def test_truncated_event_stream_fails_loudly() -> None:
    """A partial stream is not a successful short answer."""
    resp, _ = _drive(upstream=_sse_response(TRUNCATED_SSE))

    assert resp.status_code == 502
    assert "application/json" in resp.headers["content-type"]
    assert resp.json()["error"]["type"] == "upstream_protocol_error"


def test_error_event_is_not_reported_as_success() -> None:
    resp, _ = _drive(upstream=_sse_response(ERROR_SSE))

    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "upstream_protocol_error"


def test_plain_json_reply_is_untouched() -> None:
    resp, _ = _drive(upstream=httpx.Response(200, json=JSON_REPLY))

    assert resp.status_code == 200
    assert "application/json" in resp.headers["content-type"]
    assert resp.json()["content"] == [{"type": "text", "text": "hello"}]


# --------------------------------------------------------------------------- #
# Strict reconstruction, exercised directly
# --------------------------------------------------------------------------- #
@pytest.fixture()
def proxy():
    from headroom.proxy.server import HeadroomProxy

    return HeadroomProxy(_config())


def test_strict_mode_requires_a_terminal_event(proxy) -> None:
    assert proxy._parse_sse_to_response(TRUNCATED_SSE, "anthropic", require_complete=True) is None


def test_strict_mode_rejects_an_error_event(proxy) -> None:
    assert proxy._parse_sse_to_response(ERROR_SSE, "anthropic", require_complete=True) is None


def test_strict_mode_rejects_an_unclosed_block(proxy) -> None:
    unclosed = COMPLETE_SSE.replace(
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n', ""
    )

    assert proxy._parse_sse_to_response(unclosed, "anthropic", require_complete=True) is None


def test_strict_mode_rejects_an_unknown_delta_type(proxy) -> None:
    """A future delta Headroom cannot replay must not pass as complete."""
    unknown = COMPLETE_SSE.replace('"type":"text_delta","text":"hello"', '"type":"future_delta"')

    assert proxy._parse_sse_to_response(unknown, "anthropic", require_complete=True) is None


def test_strict_mode_reads_crlf_framed_events(proxy) -> None:
    parsed = proxy._parse_sse_to_response(
        COMPLETE_SSE.replace("\n", "\r\n"), "anthropic", require_complete=True
    )

    assert parsed is not None
    assert parsed["content"] == [{"type": "text", "text": "hello"}]


def test_strict_mode_keeps_stop_sequence_and_type(proxy) -> None:
    parsed = proxy._parse_sse_to_response(COMPLETE_SSE, "anthropic", require_complete=True)

    assert parsed is not None
    assert parsed["type"] == "message"
    assert parsed["stop_sequence"] is None


def test_permissive_mode_is_unchanged_for_existing_callers(proxy) -> None:
    """Streaming callers keep the lenient reconstruction they rely on."""
    parsed = proxy._parse_sse_to_response(TRUNCATED_SSE, "anthropic")

    assert parsed is not None
    assert parsed["content"][0]["text"] == "hello"


def test_event_stream_under_a_vague_content_type_is_still_adapted() -> None:
    """A gateway may relay the stream without declaring it (#3130)."""
    resp, _ = _drive(
        upstream=httpx.Response(
            200,
            content=COMPLETE_SSE.encode(),
            headers={"content-type": "application/octet-stream"},
        )
    )

    assert resp.status_code == 200
    assert "application/json" in resp.headers["content-type"]
    assert resp.json()["content"] == [{"type": "text", "text": "hello"}]
