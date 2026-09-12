"""Regression tests for Anthropic streaming CCR retrieval interception."""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.responses import StreamingResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.requests import Request  # noqa: E402

from headroom.cache.compression_store import get_compression_store  # noqa: E402
from headroom.ccr.tool_injection import create_ccr_tool_definition  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def _make_config() -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=True,
        ccr_handle_responses=True,
        ccr_event_streaming=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )


@pytest.fixture(autouse=True)
def _fresh_compression_store():
    """Each test gets its own store, so seeded markers cannot leak between them."""
    from headroom.cache.backends import InMemoryBackend
    from headroom.cache.compression_store import get_compression_store, reset_compression_store

    reset_compression_store()
    get_compression_store(backend=InMemoryBackend())
    try:
        yield
    finally:
        reset_compression_store()


def _buffered(text: str) -> str:
    """User content carrying a marker this proxy owns.

    The buffered path engages only when retrieval has something to expand
    (#3071); a resident ``headroom_retrieve`` with no redeemable marker in the
    request keeps streaming. Every test below that is *about* the buffered path
    therefore has to earn it with a real marker rather than the tool alone.
    """
    store = get_compression_store()
    hash_key = store.store(
        original=json.dumps({"earlier": "tool output"}),
        compressed="{}",
        original_item_count=1,
    )
    return f"{text} (earlier output at <<ccr:{hash_key}>>)"


def _message_response(content: list[dict], *, stop_reason: str = "end_turn") -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": content,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }


class _ContinuationClient:
    def __init__(self, response_json: dict) -> None:
        self.response_json = response_json
        self.post_calls: list[dict] = []

    async def post(self, url, *, content=None, headers=None, timeout=None):  # noqa: ANN001
        self.post_calls.append(
            {
                "url": url,
                "content": content,
                "headers": dict(headers or {}),
                "timeout": timeout,
            }
        )
        return httpx.Response(200, json=self.response_json)

    async def aclose(self) -> None:
        return None


def test_event_level_ccr_keeps_upstream_streaming_and_wires_interceptor() -> None:
    config = _make_config()
    config.ccr_event_streaming = True

    async def _result_stream():
        yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    with patch("headroom.proxy.server.AnyLLMBackend"):
        app = create_app(config)
        with TestClient(app) as client:
            proxy = client.app.state.proxy
            proxy._stream_response = AsyncMock(
                return_value=StreamingResponse(_result_stream(), media_type="text/event-stream")
            )
            proxy._retry_request = AsyncMock(
                side_effect=AssertionError("event-level path must not pre-buffer upstream")
            )
            response = client.post(
                "/v1/messages",
                headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 64,
                    "stream": True,
                    "tools": [create_ccr_tool_definition("anthropic")],
                    "messages": [{"role": "user", "content": _buffered("use retrieval")}],
                },
            )

    assert response.status_code == 200
    call = proxy._stream_response.await_args
    assert call.args[2]["stream"] is True
    assert call.kwargs["ccr_stream_provider"] == "anthropic"


def test_streaming_headroom_retrieve_is_intercepted_and_returned_as_sse() -> None:
    config = _make_config()
    store = get_compression_store()
    hash_key = store.store(
        original=json.dumps({"secret": "retrieved answer"}),
        compressed="{}",
        original_item_count=1,
    )
    initial_response = _message_response(
        [
            {
                "type": "tool_use",
                "id": "toolu_ccr",
                "name": "headroom_retrieve",
                "input": {"hash": hash_key},
            }
        ],
        stop_reason="tool_use",
    )
    final_response = _message_response(
        [{"type": "text", "text": "retrieved answer is now available"}]
    )

    with patch("headroom.proxy.server.AnyLLMBackend"):
        app = create_app(config)
        with TestClient(app) as client:
            proxy = client.app.state.proxy
            proxy._stream_response = AsyncMock(
                side_effect=AssertionError("live streaming path should not be used")
            )
            continuation_client = _ContinuationClient(final_response)
            proxy.http_client = continuation_client
            initial_bodies: list[dict] = []

            async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
                initial_bodies.append(json.loads(json.dumps(body)))
                assert stream is False
                assert body["stream"] is False
                return httpx.Response(200, json=initial_response)

            proxy._retry_request = _fake_retry  # type: ignore[assignment]

            resp = client.post(
                "/v1/messages",
                headers={
                    "x-api-key": "test-key",
                    "anthropic-version": "2023-06-01",
                    "accept": "text/event-stream",
                    "content-encoding": "identity",
                },
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 64,
                    "stream": True,
                    "tools": [create_ccr_tool_definition("anthropic")],
                    "messages": [{"role": "user", "content": _buffered("retrieve it")}],
                },
            )

    assert resp.status_code == 200, resp.text
    assert "text/event-stream" in resp.headers["content-type"]
    assert "retrieved answer is now available" in resp.text
    assert "headroom_retrieve" not in resp.text
    assert initial_bodies and initial_bodies[0]["stream"] is False
    assert len(continuation_client.post_calls) == 1
    continuation_body = json.loads(continuation_client.post_calls[0]["content"].decode())
    assert continuation_body["stream"] is False
    continuation_headers = {
        key.lower(): value for key, value in continuation_client.post_calls[0]["headers"].items()
    }
    assert "content-length" not in continuation_headers
    assert "content-encoding" not in continuation_headers
    assert "transfer-encoding" not in continuation_headers
    assert "accept-encoding" not in continuation_headers


def test_streaming_without_headroom_retrieve_uses_normal_streaming_path() -> None:
    config = _make_config()

    with patch("headroom.proxy.server.AnyLLMBackend"):
        app = create_app(config)
        with TestClient(app) as client:
            proxy = client.app.state.proxy

            async def _fake_stream_response(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
                async def _gen():
                    yield b"event: message_stop\n"
                    yield b'data: {"type":"message_stop"}\n\n'

                return StreamingResponse(_gen(), media_type="text/event-stream")

            proxy._stream_response = AsyncMock(side_effect=_fake_stream_response)

            resp = client.post(
                "/v1/messages",
                headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 64,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert resp.status_code == 200, resp.text
    assert "text/event-stream" in resp.headers["content-type"]
    assert '"message_stop"' in resp.text
    proxy._stream_response.assert_awaited_once()


def test_streaming_with_headroom_retrieve_available_but_unused_returns_sse() -> None:
    config = _make_config()
    text_response = _message_response([{"type": "text", "text": "plain answer"}])

    with patch("headroom.proxy.server.AnyLLMBackend"):
        app = create_app(config)
        with TestClient(app) as client:
            proxy = client.app.state.proxy
            proxy._stream_response = AsyncMock(
                side_effect=AssertionError("live streaming path should not be used")
            )
            continuation_client = _ContinuationClient(_message_response([]))
            proxy.http_client = continuation_client
            initial_bodies: list[dict] = []

            async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
                initial_bodies.append(json.loads(json.dumps(body)))
                assert stream is False
                assert body["stream"] is False
                return httpx.Response(200, json=text_response)

            proxy._retry_request = _fake_retry  # type: ignore[assignment]

            resp = client.post(
                "/v1/messages",
                headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 64,
                    "stream": True,
                    "tools": [create_ccr_tool_definition("anthropic")],
                    "messages": [{"role": "user", "content": _buffered("hello")}],
                },
            )

    assert resp.status_code == 200, resp.text
    assert "text/event-stream" in resp.headers["content-type"]
    assert "plain answer" in resp.text
    assert "headroom_retrieve" not in resp.text
    assert initial_bodies and initial_bodies[0]["stream"] is False
    assert continuation_client.post_calls == []
    proxy._stream_response.assert_not_awaited()


def test_mixed_ccr_and_client_tool_suppresses_private_call_and_scrubs_markers() -> None:
    """LEGAL mixed turn (#839, #2089): headroom_retrieve emitted alongside a
    client tool. The proxy cannot synthesize the client tool_result, so it must
    hand the client-owned tool back as a 200 SSE stream. Headroom's injected
    call is not part of the client's tool contract and markers in the remaining
    arguments must not cross the client boundary. It must NOT 502 and must NOT
    issue a continuation request."""
    config = _make_config()
    initial_response = _message_response(
        [
            {
                "type": "tool_use",
                "id": "toolu_ccr",
                "name": "headroom_retrieve",
                "input": {"hash": "abc123"},
            },
            {
                "type": "tool_use",
                "id": "toolu_client",
                "name": "client_tool",
                "input": {"value": "<<ccr:deadbeefdeadbeef,text,1KB>>"},
            },
        ],
        stop_reason="tool_use",
    )

    with patch("headroom.proxy.server.AnyLLMBackend"):
        app = create_app(config)
        with TestClient(app) as client:
            proxy = client.app.state.proxy
            proxy._stream_response = AsyncMock(
                side_effect=AssertionError("live streaming path should not be used")
            )
            continuation_client = _ContinuationClient(_message_response([]))
            proxy.http_client = continuation_client

            async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
                assert body["stream"] is False
                return httpx.Response(200, json=initial_response)

            proxy._retry_request = _fake_retry  # type: ignore[assignment]

            resp = client.post(
                "/v1/messages",
                headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 64,
                    "stream": True,
                    "tools": [
                        create_ccr_tool_definition("anthropic"),
                        {
                            "name": "client_tool",
                            "description": "Client-owned tool",
                            "input_schema": {"type": "object", "properties": {}},
                        },
                    ],
                    "messages": [{"role": "user", "content": _buffered("use tools")}],
                },
            )

    assert resp.status_code == 200, resp.text
    assert "text/event-stream" in resp.headers["content-type"]
    # Preserve only the tool the client actually declared. Internal CCR syntax
    # must not leak through its arguments.
    assert "headroom_retrieve" not in resp.text
    assert "client_tool" in resp.text
    assert "toolu_ccr" not in resp.text
    assert "toolu_client" in resp.text
    assert "<<ccr:" not in resp.text
    assert "compressed content unavailable" in resp.text
    assert "Unable to safely complete streamed CCR retrieval" not in resp.text
    # No continuation is issued — the client resolves all tool calls.
    assert continuation_client.post_calls == []


def test_unresolved_ccr_only_fails_closed_without_private_tool_leak() -> None:
    """CCR-only turn that never resolves: the model keeps re-emitting
    headroom_retrieve so the continuation exhausts its retrieval rounds with a
    residual call and no accompanying client tool. The client never declared
    Headroom's private tool, so the stream must fail closed instead of exposing
    an impossible call contract."""
    config = _make_config()
    persistent_ccr = _message_response(
        [
            {
                "type": "tool_use",
                "id": "toolu_ccr",
                "name": "headroom_retrieve",
                "input": {"hash": "deadbeefdead"},
            },
        ],
        stop_reason="tool_use",
    )

    with patch("headroom.proxy.server.AnyLLMBackend"):
        app = create_app(config)
        with TestClient(app) as client:
            proxy = client.app.state.proxy
            proxy._stream_response = AsyncMock(
                side_effect=AssertionError("live streaming path should not be used")
            )
            # Every continuation re-emits headroom_retrieve, so it never resolves.
            continuation_client = _ContinuationClient(persistent_ccr)
            proxy.http_client = continuation_client

            async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
                return httpx.Response(200, json=persistent_ccr)

            proxy._retry_request = _fake_retry  # type: ignore[assignment]

            resp = client.post(
                "/v1/messages",
                headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 64,
                    "stream": True,
                    "tools": [create_ccr_tool_definition("anthropic")],
                    "messages": [{"role": "user", "content": _buffered("use tools")}],
                },
            )

    assert resp.status_code == 502, resp.text
    assert "text/event-stream" in resp.headers["content-type"]
    assert "headroom_retrieve" not in resp.text
    assert "Unable to safely complete streamed CCR retrieval" in resp.text


@pytest.mark.asyncio
async def test_buffered_ccr_withholds_output_until_delayed_upstream_resolves() -> None:
    """Nothing is sent — no status, no body — until the buffered result exists.

    The response used to commit ``200 text/event-stream`` on a 1s keepalive
    timer, which made every later failure unreportable: the client saw a 200
    with no ``message_start`` and the real status was gone. See
    ``test_buffered_ccr_preserves_late_failure_status_and_headers``.
    """
    config = _make_config()
    final_response = _message_response([{"type": "text", "text": "done"}])
    started = asyncio.Event()
    release = asyncio.Event()
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "stream": True,
        "tools": [create_ccr_tool_definition("anthropic")],
        "messages": [{"role": "user", "content": _buffered("wait")}],
    }
    request_delivered = False

    async def receive():
        # Mirror a real ASGI server: the body arrives once, then the channel
        # stays open because the client is still connected. Returning instantly
        # on every call spins `StreamingResponse.listen_for_disconnect`, which
        # never yields, so the response body would never be scheduled.
        nonlocal request_delivered
        if request_delivered:
            await asyncio.Event().wait()
        request_delivered = True
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/messages",
        "raw_path": b"/v1/messages",
        "query_string": b"",
        "headers": [(b"x-api-key", b"test-key"), (b"anthropic-version", b"2023-06-01")],
        "server": ("testserver", 80),
        "client": ("testclient", 123),
        "root_path": "",
    }

    with patch("headroom.proxy.server.AnyLLMBackend"):
        app = create_app(config)
        with TestClient(app):
            proxy = app.state.proxy

            async def delayed_retry(*args, **kwargs):  # noqa: ANN002, ANN003
                started.set()
                await release.wait()
                return httpx.Response(200, json=final_response)

            proxy._retry_request = delayed_retry
            task = asyncio.create_task(proxy.handle_anthropic_messages(Request(scope, receive)))
            await started.wait()
            response = await asyncio.wait_for(asyncio.shield(task), 1)
            events: list[dict] = []

            async def send(message):  # noqa: ANN001
                events.append(message)

            response_task = asyncio.create_task(response(scope, receive, send))
            # Longer than the deleted 1.0s keepalive deadline: an unresolved
            # upstream must still have produced no ASGI message at all.
            await asyncio.sleep(1.1)
            assert events == []
            release.set()
            await response_task

    start = next(event for event in events if event["type"] == "http.response.start")
    assert start["status"] == 200
    bodies = b"".join(event["body"] for event in events if event["type"] == "http.response.body")
    assert b"event: ping" not in bodies
    assert b"done" in bodies


@pytest.mark.asyncio
async def test_buffered_ccr_preserves_early_failure_status_and_headers() -> None:
    config = _make_config()
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "stream": True,
        "tools": [create_ccr_tool_definition("anthropic")],
        "messages": [{"role": "user", "content": _buffered("fail early")}],
    }

    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/messages",
        "raw_path": b"/v1/messages",
        "query_string": b"",
        "headers": [(b"x-api-key", b"test-key"), (b"anthropic-version", b"2023-06-01")],
        "server": ("testserver", 80),
        "client": ("testclient", 123),
        "root_path": "",
    }

    with patch("headroom.proxy.server.AnyLLMBackend"):
        app = create_app(config)
        with TestClient(app):
            proxy = app.state.proxy

            async def early_failure(*args, **kwargs):  # noqa: ANN002, ANN003
                await asyncio.sleep(0.05)
                return httpx.Response(
                    429,
                    headers={"retry-after": "7"},
                    json={"error": {"message": "slow down"}},
                )

            proxy._retry_request = early_failure
            response = await proxy.handle_anthropic_messages(Request(scope, receive))
            events: list[dict] = []

            async def send(message):  # noqa: ANN001
                events.append(message)

            await response(scope, receive, send)

    start = next(event for event in events if event["type"] == "http.response.start")
    headers = dict(start["headers"])
    assert start["status"] == 429
    assert headers[b"retry-after"] == b"7"
    assert b": headroom-keepalive\n\n" not in b"".join(
        event["body"] for event in events if event["type"] == "http.response.body"
    )


@pytest.mark.asyncio
async def test_buffered_ccr_preserves_late_failure_status_and_headers() -> None:
    """The reported failure: a non-200 landing after the old keepalive deadline.

    Headroom had already committed ``200 text/event-stream`` by then, so the 429
    reached Claude Code as a 200 whose body carried no ``message_start`` — shown
    as "API returned an empty or malformed response (HTTP 200) — check for a
    proxy or gateway intercepting the request" — and ``retry-after`` was dropped,
    so the client never backed off.
    """
    config = _make_config()
    started = asyncio.Event()
    release = asyncio.Event()
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "stream": True,
        "tools": [create_ccr_tool_definition("anthropic")],
        "messages": [{"role": "user", "content": _buffered("fail late")}],
    }

    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/messages",
        "raw_path": b"/v1/messages",
        "query_string": b"",
        "headers": [(b"x-api-key", b"test-key"), (b"anthropic-version", b"2023-06-01")],
        "server": ("testserver", 80),
        "client": ("testclient", 123),
        "root_path": "",
    }

    with patch("headroom.proxy.server.AnyLLMBackend"):
        app = create_app(config)
        with TestClient(app):
            proxy = app.state.proxy

            async def delayed_failure(*args, **kwargs):  # noqa: ANN002, ANN003
                started.set()
                await release.wait()
                return httpx.Response(
                    429,
                    headers={"retry-after": "7"},
                    json={"error": {"message": "slow down"}},
                )

            proxy._retry_request = delayed_failure
            task = asyncio.create_task(proxy.handle_anthropic_messages(Request(scope, receive)))
            await started.wait()
            response = await asyncio.wait_for(asyncio.shield(task), 1)
            events: list[dict] = []

            async def send(message):  # noqa: ANN001
                events.append(message)

            response_task = asyncio.create_task(response(scope, receive, send))
            # Past the deleted 1.0s keepalive deadline before the upstream fails.
            await asyncio.sleep(1.1)
            assert events == []
            release.set()
            await response_task

    start = next(event for event in events if event["type"] == "http.response.start")
    assert start["status"] == 429
    assert dict(start["headers"])[b"retry-after"] == b"7"
    assert b"slow down" in b"".join(
        event["body"] for event in events if event["type"] == "http.response.body"
    )


def test_buffered_ccr_rejects_malformed_success_as_502() -> None:
    """A non-SSE, non-JSON 200 is an upstream protocol error, not success."""
    config = _make_config()
    with patch("headroom.proxy.server.AnyLLMBackend"):
        app = create_app(config)
        with TestClient(app) as client:
            proxy = app.state.proxy
            proxy._retry_request = AsyncMock(
                return_value=httpx.Response(
                    200,
                    content=b"<html>gateway timeout</html>",
                    headers={"content-type": "text/html"},
                )
            )
            response = client.post(
                "/v1/messages",
                headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 64,
                    "stream": True,
                    "tools": [create_ccr_tool_definition("anthropic")],
                    "messages": [{"role": "user", "content": _buffered("fail safely")}],
                },
            )

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_protocol_error"
    assert b"gateway timeout" not in response.content


@pytest.mark.asyncio
async def test_buffered_ccr_late_failure_returns_sanitized_json_error() -> None:
    """A slow crash gets the same 502 the fast one does, not a downgraded 200."""
    config = _make_config()
    started = asyncio.Event()
    release = asyncio.Event()
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "stream": True,
        "tools": [create_ccr_tool_definition("anthropic")],
        "messages": [{"role": "user", "content": _buffered("wait")}],
    }

    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/messages",
        "raw_path": b"/v1/messages",
        "query_string": b"",
        "headers": [(b"x-api-key", b"test-key"), (b"anthropic-version", b"2023-06-01")],
        "server": ("testserver", 80),
        "client": ("testclient", 123),
        "root_path": "",
    }

    with patch("headroom.proxy.server.AnyLLMBackend"):
        app = create_app(config)
        with TestClient(app):
            proxy = app.state.proxy
            proxy_logger = logging.getLogger("headroom.proxy")
            error_records: list[logging.LogRecord] = []
            log_handler = logging.Handler()
            log_handler.setLevel(logging.ERROR)
            log_handler.emit = error_records.append
            proxy_logger.addHandler(log_handler)

            async def delayed_failure(*args, **kwargs):  # noqa: ANN002, ANN003
                started.set()
                await release.wait()
                raise RuntimeError("boom")

            with patch.object(
                proxy.metrics, "record_failed", new_callable=AsyncMock
            ) as record_failed:
                proxy._retry_request = delayed_failure
                task = asyncio.create_task(proxy.handle_anthropic_messages(Request(scope, receive)))
                await started.wait()
                response = await asyncio.wait_for(asyncio.shield(task), 1)
                events: list[dict] = []

                async def send(message):  # noqa: ANN001
                    events.append(message)

                response_task = asyncio.create_task(response(scope, receive, send))
                await asyncio.sleep(0)
                assert events == []
                release.set()
                await response_task
                record_failed.assert_awaited_once_with(provider="anthropic")
            proxy_logger.removeHandler(log_handler)

    start = next(event for event in events if event["type"] == "http.response.start")
    assert start["status"] == 502
    bodies = b"".join(event["body"] for event in events if event["type"] == "http.response.body")
    assert b"An error occurred while processing your request." in bodies
    assert b"boom" not in bodies
    assert events[-1]["more_body"] is False
    assert any(
        record.levelno == logging.ERROR and "RuntimeError: boom" in record.getMessage()
        for record in error_records
    )


@pytest.mark.asyncio
async def test_buffered_ccr_pre_keepalive_exception_returns_json_error() -> None:
    config = _make_config()
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "stream": True,
        "tools": [create_ccr_tool_definition("anthropic")],
        "messages": [{"role": "user", "content": _buffered("fail before keepalive")}],
    }

    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/messages",
        "raw_path": b"/v1/messages",
        "query_string": b"",
        "headers": [(b"x-api-key", b"test-key"), (b"anthropic-version", b"2023-06-01")],
        "server": ("testserver", 80),
        "client": ("testclient", 123),
        "root_path": "",
    }

    with patch("headroom.proxy.server.AnyLLMBackend"):
        app = create_app(config)
        with TestClient(app):
            proxy = app.state.proxy

            async def early_exception(*args, **kwargs):  # noqa: ANN002, ANN003
                raise RuntimeError("boom")

            proxy._retry_request = early_exception
            response = await proxy.handle_anthropic_messages(Request(scope, receive))
            events: list[dict] = []

            async def send(message):  # noqa: ANN001
                events.append(message)

            await response(scope, receive, send)

    start = next(event for event in events if event["type"] == "http.response.start")
    bodies = [event["body"] for event in events if event["type"] == "http.response.body"]
    assert start["status"] == 502
    assert dict(start["headers"])[b"content-type"] == b"application/json"
    payload = json.loads(bodies[-1].decode())
    assert (
        payload["error"]["message"]
        == "An error occurred while processing your request. Please try again."
    )
