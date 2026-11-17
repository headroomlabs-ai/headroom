"""The buffered CCR request streams from upstream, and the proxy folds the reply.

The buffered CCR path flips a streaming turn to ``stream: false`` so the whole
reply is in hand before server-side retrieval. Sent that way, upstream sends
nothing until the generation is complete, and a network that drops HTTP
connections idle for ~60s (TLS-inspecting antivirus, VPN, corporate proxy)
cut every long turn: three attempts failed at ~60s each and the client got a
502 it resent forever. The request now goes out as a stream (as do the CCR,
memory and turn-hook continuations built from it) and the event stream is
folded back into the Message the buffered path reads.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

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
from headroom.proxy.anthropic_wire import fold_anthropic_stream  # noqa: E402
from headroom.proxy.server import HeadroomProxy, ProxyConfig, create_app  # noqa: E402

URL = "https://api.anthropic.com/v1/messages"


def _sse(*events: dict[str, Any]) -> bytes:
    return b"".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events
    )


def _message_start(model: str = "claude-opus-5") -> dict[str, Any]:
    return {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 25,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 100,
                "output_tokens": 1,
            },
        },
    }


def _text_turn(text: str) -> list[dict[str, Any]]:
    return [
        _message_start(),
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 7},
        },
        {"type": "message_stop"},
    ]


# A turn exercising every delta the fold accumulates, with the members only a
# stream carries: a thinking block's ``estimated_tokens`` placeholder, a delta's
# progress count, interleaved pings, and a null usage count in message_delta.
EVERY_DELTA = [
    _message_start(),
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "thinking",
            "thinking": "",
            "signature": "",
            "estimated_tokens": None,
        },
    },
    {"type": "ping"},
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "Let me ", "estimated_tokens": 3},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "check."},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "signature_delta", "signature": "sig-1"},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "content_block_start",
        "index": 1,
        "content_block": {"type": "redacted_thinking", "data": "opaque"},
    },
    {"type": "content_block_stop", "index": 1},
    {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": "Paris"}},
    {
        "type": "content_block_delta",
        "index": 2,
        "delta": {
            "type": "citations_delta",
            "citation": {
                "type": "char_location",
                "cited_text": "Paris",
                "document_index": 0,
                "document_title": "Atlas",
                "start_char_index": 0,
                "end_char_index": 5,
            },
        },
    },
    {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": " it is."}},
    {"type": "content_block_stop", "index": 2},
    {
        "type": "content_block_start",
        "index": 3,
        "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}},
    },
    {
        "type": "content_block_delta",
        "index": 3,
        "delta": {"type": "input_json_delta", "partial_json": '{"file_pa'},
    },
    {
        "type": "content_block_delta",
        "index": 3,
        "delta": {"type": "input_json_delta", "partial_json": 'th": "a.py"}'},
    },
    {"type": "content_block_stop", "index": 3},
    {"type": "ping"},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        "usage": {"output_tokens": 42, "input_tokens": None},
    },
    {"type": "message_stop"},
]


# --------------------------------------------------------------------------- #
# The fold
# --------------------------------------------------------------------------- #
def test_fold_matches_the_anthropic_sdk_accumulator() -> None:
    streaming = pytest.importorskip("anthropic.lib.streaming._messages")
    accumulate = streaming.accumulate_event
    # anthropic 1.x keeps each tool block's partial JSON in a buffer the caller
    # owns (``json_bufs``); 0.x kept it on the block. Same accumulation either way.
    buffers: dict[str, Any] = (
        {"json_bufs": {}} if "json_bufs" in inspect.signature(accumulate).parameters else {}
    )
    snapshot = None
    for event in EVERY_DELTA:
        snapshot = accumulate(event=event, current_snapshot=snapshot, **buffers)
    assert snapshot is not None

    folded = fold_anthropic_stream(_sse(*EVERY_DELTA))

    assert folded is not None
    # The SDK keeps nulls (``stop_sequence``, the thinking placeholder); the
    # non-streaming Message drops the placeholder, so compare without them.
    assert {k: v for k, v in folded.items() if v is not None} == snapshot.model_dump(
        exclude_none=True
    )
    assert folded["usage"]["input_tokens"] == 25, "a null delta count must not erase it"


def test_fold_keeps_stream_only_members_out_of_blocks() -> None:
    """The proxy resends folded blocks; Anthropic 400s on any extra member."""
    folded = fold_anthropic_stream(_sse(*EVERY_DELTA))

    assert folded is not None
    assert folded["content"][0] == {
        "type": "thinking",
        "thinking": "Let me check.",
        "signature": "sig-1",
    }
    assert folded["content"][1] == {"type": "redacted_thinking", "data": "opaque"}
    assert "estimated_tokens" not in json.dumps(folded)


def test_fold_reports_a_truncated_stream() -> None:
    assert fold_anthropic_stream(_sse(*EVERY_DELTA[:-2])) is None
    assert fold_anthropic_stream(_sse(*EVERY_DELTA[:5])) is None  # a block left open


def test_fold_returns_an_in_band_error_event() -> None:
    error = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}

    assert fold_anthropic_stream(_sse(*EVERY_DELTA[:7], error)) == error


def test_fold_refuses_a_delta_it_does_not_know() -> None:
    unknown = {"type": "content_block_delta", "index": 2, "delta": {"type": "future_delta"}}

    with pytest.raises(ValueError):
        fold_anthropic_stream(_sse(*EVERY_DELTA[:10], unknown, *EVERY_DELTA[10:]))


# --------------------------------------------------------------------------- #
# The send helper
# --------------------------------------------------------------------------- #
@pytest.fixture()
def proxy() -> HeadroomProxy:
    return HeadroomProxy(ProxyConfig(optimize=False, cache_enabled=False, image_optimize=False))


class _Upstream:
    """Answers a stream:true request with ``stream_reply``, anything else with JSON."""

    def __init__(self, stream_reply: httpx.Response) -> None:
        self.stream_reply = stream_reply
        self.calls: list[tuple[dict[str, str], dict[str, Any]]] = []

    async def __call__(self, headers: dict[str, str], body: dict[str, Any]) -> httpx.Response:
        self.calls.append((dict(headers), json.loads(json.dumps(body))))
        if body.get("stream"):
            return self.stream_reply
        return httpx.Response(200, json={"type": "message", "content": [], "via": "json"})


def _stream_reply(*events: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, content=_sse(*events), headers={"content-type": "text/event-stream"})


async def _send(
    proxy: HeadroomProxy, upstream: _Upstream, *, url: str = URL, body: dict | None = None
) -> httpx.Response:
    return await proxy._send_buffered_as_stream(
        upstream,
        {"Accept": "application/json", "x-api-key": "k"},
        body if body is not None else {"model": "m", "stream": False, "messages": []},
        url=url,
        request_id="t",
    )


@pytest.mark.asyncio
async def test_a_buffered_request_streams_and_folds(proxy) -> None:
    upstream = _Upstream(_stream_reply(*_text_turn("hello")))
    body = {"model": "m", "stream": False, "messages": []}

    response = await _send(proxy, upstream, body=body)

    assert response.status_code == 200
    assert response.json()["content"] == [{"type": "text", "text": "hello"}]
    assert len(upstream.calls) == 1
    headers, sent = upstream.calls[0]
    assert sent["stream"] is True
    assert [v for k, v in headers.items() if k.lower() == "accept"] == ["text/event-stream"]
    assert body["stream"] is False, "the caller's body is not mutated"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("env", "url", "stream"),
    [
        ("0", URL, False),  # kill switch
        ("1", "https://gateway.example/v1/messages", False),  # not first-party
        ("1", URL, True),  # already a stream
    ],
)
async def test_requests_it_should_not_touch_go_out_unchanged(
    proxy, monkeypatch, env: str, url: str, stream: bool
) -> None:
    monkeypatch.setenv("HEADROOM_BUFFERED_UPSTREAM_STREAM", env)
    upstream = _Upstream(_stream_reply(*_text_turn("hello")))
    body = {"model": "m", "stream": stream, "messages": []}

    await _send(proxy, upstream, url=url, body=body)

    assert upstream.calls == [({"Accept": "application/json", "x-api-key": "k"}, body)]


@pytest.mark.asyncio
async def test_a_propagated_kill_switch_keeps_the_request_non_streaming(proxy) -> None:
    # `headroom wrap` pushes this knob to a reused proxy, so the rollback works
    # without the proxy's own environment or a restart.
    from headroom.proxy import runtime_env

    assert runtime_env.explicit_env({"HEADROOM_BUFFERED_UPSTREAM_STREAM": "0"}) == {
        "HEADROOM_BUFFERED_UPSTREAM_STREAM": "0"
    }
    upstream = _Upstream(_stream_reply(*_text_turn("hello")))
    body = {"model": "m", "stream": False, "messages": []}
    runtime_env.set_overrides({"HEADROOM_BUFFERED_UPSTREAM_STREAM": "0"})
    try:
        response = await _send(proxy, upstream, body=body)
    finally:
        runtime_env.clear_overrides()

    assert upstream.calls == [({"Accept": "application/json", "x-api-key": "k"}, body)]
    assert response.json()["via"] == "json"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(529, json={"type": "error", "error": {"type": "overloaded_error"}}),
        httpx.Response(200, json={"type": "message", "content": []}),  # answered JSON anyway
    ],
)
async def test_a_reply_that_is_not_an_event_stream_is_returned_as_is(proxy, reply) -> None:
    upstream = _Upstream(reply)

    assert await _send(proxy, upstream) is reply


@pytest.mark.asyncio
async def test_an_in_band_error_comes_back_with_its_status(proxy) -> None:
    error = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    upstream = _Upstream(_stream_reply(_message_start(), error))

    response = await _send(proxy, upstream)

    assert response.status_code == 529
    assert response.json() == error


@pytest.mark.asyncio
async def test_a_truncated_stream_is_a_502(proxy) -> None:
    upstream = _Upstream(_stream_reply(*_text_turn("hello")[:-2]))

    response = await _send(proxy, upstream)

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_protocol_error"
    assert len(upstream.calls) == 1


@pytest.mark.asyncio
async def test_a_stream_it_cannot_fold_is_sent_again_as_stream_false(proxy) -> None:
    unknown = {"type": "content_block_delta", "index": 0, "delta": {"type": "future_delta"}}
    events = _text_turn("hello")
    upstream = _Upstream(_stream_reply(*events[:3], unknown, *events[3:]))

    response = await _send(proxy, upstream)

    assert response.json()["via"] == "json"
    assert [sent["stream"] for _, sent in upstream.calls] == [True, False]
    assert upstream.calls[1][0]["Accept"] == "application/json"


# --------------------------------------------------------------------------- #
# Through the handler
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _store():
    reset_compression_store()
    get_compression_store(backend=InMemoryBackend())
    try:
        yield
    finally:
        reset_compression_store()


def _config() -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        memory_enabled=False,
        ccr_inject_tool=True,
        ccr_handle_responses=True,
        ccr_context_tracking=False,
        image_optimize=False,
    )


class _ContinuationClient:
    """Stands in for ``http_client``; the CCR continuation posts through it."""

    def __init__(self, reply: httpx.Response) -> None:
        self.reply = reply
        self.calls: list[tuple[dict[str, str], dict[str, Any]]] = []

    async def post(self, url, *, content=None, headers=None, timeout=None):  # noqa: ANN001
        self.calls.append((dict(headers or {}), json.loads(content)))
        return self.reply

    async def aclose(self) -> None:
        return None


def _accept(headers: dict[str, str]) -> list[str]:
    return [v for k, v in headers.items() if k.lower() == "accept"]


def test_the_buffered_turn_and_its_ccr_continuation_stream_upstream() -> None:
    store = get_compression_store()
    marker = store.store(
        original=json.dumps({"secret": "retrieved answer"}),
        compressed="{}",
        original_item_count=1,
    )
    retrieve_turn = [
        _message_start(),
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "thinking",
                "thinking": "",
                "signature": "",
                "estimated_tokens": None,
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "I need it."},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_ccr",
                "name": "headroom_retrieve",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps({"hash": marker})},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 9},
        },
        {"type": "message_stop"},
    ]
    sent: list[tuple[dict[str, str], dict[str, Any]]] = []

    app = create_app(_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            sent.append((dict(headers), json.loads(json.dumps(body))))
            return _stream_reply(*retrieve_turn)

        proxy._retry_request = _fake_retry  # type: ignore[assignment]
        continuation = _ContinuationClient(_stream_reply(*_text_turn("the answer")))
        proxy.http_client = continuation

        resp = client.post(
            "/v1/messages",
            headers={"x-api-key": "k", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-opus-5",
                "max_tokens": 64,
                "stream": True,
                "tools": [create_ccr_tool_definition("anthropic")],
                "messages": [{"role": "user", "content": f"use <<ccr:{marker}>>"}],
            },
        )

    assert resp.status_code == 200, resp.text
    assert "the answer" in resp.text
    assert "headroom_retrieve" not in resp.text
    [(headers, body)] = sent
    assert body["stream"] is True
    assert _accept(headers) == ["text/event-stream"]
    [(cont_headers, cont_body)] = continuation.calls
    assert cont_body["stream"] is True
    assert _accept(cont_headers) == ["text/event-stream"]
    # The folded turn is resent as history: no stream-only member may ride along.
    assert cont_body["messages"][-2]["content"][0] == {
        "type": "thinking",
        "thinking": "I need it.",
        "signature": "sig",
    }


class _FailingContinuationClient:
    """A CCR continuation that dies on the wire or comes back non-2xx."""

    def __init__(self, reply: httpx.Response | None) -> None:
        self.reply = reply

    async def post(self, url, *, content=None, headers=None, timeout=None):  # noqa: ANN001
        if self.reply is None:
            raise httpx.ConnectError("peer said CONTENT_CANARY")
        return self.reply

    async def aclose(self) -> None:
        return None


@pytest.mark.parametrize("failure", ["transport", "status"])
def test_a_failed_ccr_continuation_logs_only_safe_metadata(monkeypatch, failure) -> None:
    import logging

    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)
    marker = get_compression_store().store(
        original=json.dumps({"secret": "retrieved answer"}), compressed="{}", original_item_count=1
    )
    retrieve_turn = [
        _message_start(),
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "t1",
                "name": "headroom_retrieve",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps({"hash": marker})},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 9},
        },
        {"type": "message_stop"},
    ]
    reply = (
        None
        if failure == "transport"
        else httpx.Response(
            503,
            content=b"CONTENT_CANARY",
            headers={"request-id": "req_upstream_1", "x-private": "HEADER_CANARY"},
        )
    )
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    app = create_app(_config())
    with TestClient(app) as client:
        # Attached after startup: startup reconfigures headroom.* logging.
        logger = logging.getLogger("headroom.proxy")
        handler = _Collect(level=logging.DEBUG)
        logger.addHandler(handler)
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            return _stream_reply(*retrieve_turn)

        proxy._retry_request = _fake_retry  # type: ignore[assignment]
        proxy.http_client = _FailingContinuationClient(reply)
        try:
            client.post(
                "/v1/messages",
                headers={"x-api-key": "k", "anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-opus-5",
                    "max_tokens": 64,
                    "stream": True,
                    "tools": [create_ccr_tool_definition("anthropic")],
                    "messages": [{"role": "user", "content": f"use <<ccr:{marker}>>"}],
                },
            )
        finally:
            logger.removeHandler(handler)

    [failure_log] = [
        logging.Formatter().format(r)
        for r in records
        if "CCR continuation failed" in r.getMessage()
    ]
    if failure == "transport":
        assert "ConnectError" in failure_log
        assert "status=None" in failure_log
    else:
        assert "status=503" in failure_log
        assert "upstream_request_id='req_upstream_1'" in failure_log
    for record in records:
        rendered = logging.Formatter().format(record)
        assert "CONTENT_CANARY" not in rendered
        assert "HEADER_CANARY" not in rendered


def test_a_client_stream_false_request_is_left_alone() -> None:
    marker = get_compression_store().store(
        original=json.dumps({"earlier": "tool output"}),
        compressed="{}",
        original_item_count=1,
    )
    sent: list[tuple[dict[str, str], dict[str, Any], dict[str, Any]]] = []
    app = create_app(_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            sent.append((dict(headers), json.loads(json.dumps(body)), kwargs))
            return httpx.Response(200, json=fold_anthropic_stream(_sse(*_text_turn("ok"))))

        proxy._retry_request = _fake_retry  # type: ignore[assignment]
        resp = client.post(
            "/v1/messages",
            headers={"x-api-key": "k", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-opus-5",
                "max_tokens": 64,
                "stream": False,
                "tools": [create_ccr_tool_definition("anthropic")],
                "messages": [{"role": "user", "content": f"use <<ccr:{marker}>>"}],
            },
        )

    assert resp.status_code == 200, resp.text
    [(headers, body, kwargs)] = sent
    assert body["stream"] is False
    assert _accept(headers) == ["application/json"]
    assert kwargs["original_body_bytes"] is not None
