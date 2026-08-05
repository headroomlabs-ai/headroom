"""HTTP-level tests for CCR retrieve-tool interception on /v1/responses (#1877).

`handle_openai_responses` previously had zero CCR/headroom_retrieve wiring:
a `headroom_retrieve` function_call in a Responses API reply passed straight
through to the client, which typically can't resolve it (see issue #1877).
These tests exercise the new interception mirrored from the chat-completions
backend path (`handle_openai_chat` ~2775-2848) — see
tests/test_proxy/test_openai_backend_path.py for that precedent.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from starlette.requests import Request  # noqa: E402

from headroom.cache.compression_store import reset_compression_store  # noqa: E402
from headroom.ccr.tool_injection import CCR_TOOL_NAME  # noqa: E402
from headroom.proxy.loopback_guard import require_loopback  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

_RETRIEVE_TOOL = {
    "type": "function",
    "name": CCR_TOOL_NAME,
    "description": "Retrieve original content.",
    "parameters": {
        "type": "object",
        "properties": {"hash": {"type": "string"}},
        "required": ["hash"],
    },
}


@pytest.fixture(autouse=True)
def _reset_store():
    reset_compression_store()
    yield
    reset_compression_store()


def _make_app():
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    app.dependency_overrides[require_loopback] = lambda: None
    return app


def _tool_call_response(url: str, hash_key: str = "abc123def456abc123def456") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "resp_1",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": CCR_TOOL_NAME,
                    "arguments": json.dumps({"hash": hash_key}),
                },
            ],
            "usage": {"input_tokens": 50, "output_tokens": 10},
        },
        request=httpx.Request("POST", url),
    )


def _final_response(url: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "resp_2",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Resolved!"}],
                }
            ],
            "usage": {"input_tokens": 60, "output_tokens": 5},
        },
        request=httpx.Request("POST", url),
    )


def _install_two_call_retry(app, hash_key: str = "abc123def456abc123def456"):
    """First upstream call returns a headroom_retrieve function_call, second the resolved reply."""
    server = app.state.proxy
    calls: list[dict] = []

    async def fake_retry(method, url, headers, body, stream=False, **kwargs):
        calls.append({"method": method, "url": url, "headers": dict(headers), "body": body})
        if len(calls) == 1:
            return _tool_call_response(url, hash_key)
        return _final_response(url)

    server._retry_request = fake_retry
    return calls


def test_non_streaming_ccr_tool_call_is_intercepted_and_resolved():
    """A headroom_retrieve function_call in a non-streaming reply is resolved server-side."""
    app = _make_app()
    with TestClient(app) as client:
        server = app.state.proxy
        calls = _install_two_call_retry(app)

        recording_handler = MagicMock()
        recording_handler.has_ccr_tool_calls = MagicMock(return_value=True)
        recording_handler.handle_response = AsyncMock(
            return_value={
                "id": "resp_2",
                "object": "response",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Resolved!"}],
                    }
                ],
            }
        )
        server.ccr_response_handler = recording_handler

        resp = client.post(
            "/v1/responses",
            json={
                "model": "gpt-5-codex",
                "input": "please look this up",
                "tools": [_RETRIEVE_TOOL],
                "stream": False,
            },
            headers={"Authorization": "Bearer sk-test"},
        )

    assert resp.status_code == 200, resp.text
    # Only the initial upstream call happened — continuation is owned by
    # the (mocked) handle_response, not a second real _retry_request call.
    assert len(calls) == 1
    recording_handler.handle_response.assert_awaited_once()
    _args, kwargs = recording_handler.handle_response.call_args
    assert kwargs.get("provider") == "openai_responses"
    body = resp.json()
    assert body["output"][0]["content"][0]["text"] == "Resolved!"
    # The unresolved function_call must not leak to the client.
    assert not any(item.get("type") == "function_call" for item in body["output"])


def test_ccr_intercept_exception_is_reraised_not_swallowed():
    """CCR resolution failure -> 502, NOT a silent fallback to the unresolved tool_call body."""
    app = _make_app()
    with TestClient(app) as client:
        server = app.state.proxy
        _install_two_call_retry(app)

        failing_handler = MagicMock()
        failing_handler.has_ccr_tool_calls = MagicMock(return_value=True)
        failing_handler.handle_response = AsyncMock(side_effect=RuntimeError("ccr-store-blew-up"))
        server.ccr_response_handler = failing_handler

        resp = client.post(
            "/v1/responses",
            json={
                "model": "gpt-5-codex",
                "input": "please look this up",
                "tools": [_RETRIEVE_TOOL],
                "stream": False,
            },
            headers={"Authorization": "Bearer sk-test"},
        )

    failing_handler.handle_response.assert_awaited_once()
    assert resp.status_code == 502, resp.text
    body = resp.json()
    assert "function_call" not in json.dumps(body)


def test_streaming_request_with_retrieve_tool_buffers_upstream_and_streams_final_result():
    """stream:true + headroom_retrieve in tools -> forced buffered stream:false upstream."""
    app = _make_app()
    with TestClient(app) as client:
        server = app.state.proxy
        calls = _install_two_call_retry(app)

        async def _unexpected_stream_response(*args, **kwargs):
            raise AssertionError(
                "_stream_response should not be called when headroom_retrieve "
                "forces the buffered CCR path"
            )

        server._stream_response = _unexpected_stream_response

        resp = client.post(
            "/v1/responses",
            json={
                "model": "gpt-5-codex",
                "input": "please look this up",
                "tools": [_RETRIEVE_TOOL],
                "stream": True,
            },
            headers={"Authorization": "Bearer sk-test"},
        )

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    # Both upstream calls (initial + CCR continuation) went out with
    # stream forced False so the retrieval round-trip could complete.
    assert len(calls) == 2
    assert calls[0]["body"]["stream"] is False
    assert calls[1]["body"]["stream"] is False
    assert "response.completed" in resp.text
    assert "Resolved!" in resp.text


def test_streaming_request_without_retrieve_tool_uses_normal_stream_path():
    """No headroom_retrieve in tools -> unaffected, still goes through _stream_response."""
    app = _make_app()
    with TestClient(app) as client:
        server = app.state.proxy

        stream_called = {"value": False}

        async def fake_stream_response(*args, **kwargs):
            stream_called["value"] = True
            from fastapi.responses import StreamingResponse

            async def _gen():
                yield b"data: {}\n\n"

            return StreamingResponse(_gen(), media_type="text/event-stream")

        server._stream_response = fake_stream_response

        resp = client.post(
            "/v1/responses",
            json={
                "model": "gpt-5-codex",
                "input": "hello",
                "stream": True,
            },
            headers={"Authorization": "Bearer sk-test"},
        )

    assert resp.status_code == 200, resp.text
    assert stream_called["value"] is True


@pytest.mark.asyncio
async def test_buffered_responses_ccr_emits_keepalive_before_delayed_upstream():
    app = _make_app()
    body = {
        "model": "gpt-5-codex",
        "input": "please wait",
        "tools": [_RETRIEVE_TOOL],
        "stream": True,
    }
    started = asyncio.Event()
    release = asyncio.Event()

    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
        "query_string": b"",
        "headers": [(b"authorization", b"Bearer sk-test")],
        "server": ("testserver", 80),
        "client": ("testclient", 123),
        "root_path": "",
    }

    with TestClient(app):
        server = app.state.proxy

        async def delayed_retry(*args, **kwargs):  # noqa: ANN002, ANN003
            started.set()
            await release.wait()
            return _final_response("https://api.openai.com/v1/responses")

        server._retry_request = delayed_retry
        task = asyncio.create_task(server.handle_openai_responses(Request(scope, receive)))
        await started.wait()
        response = await asyncio.wait_for(asyncio.shield(task), 1)
        events: list[dict] = []
        first_body = asyncio.Event()

        async def send(message):  # noqa: ANN001
            events.append(message)
            if message["type"] == "http.response.body" and message["body"]:
                first_body.set()

        response_task = asyncio.create_task(response(scope, receive, send))
        await asyncio.wait_for(first_body.wait(), 2)
        assert not release.is_set()
        release.set()
        await response_task

    bodies = [event["body"] for event in events if event["type"] == "http.response.body"]
    assert bodies[0] == b'event: ping\ndata: {"type":"ping"}\n\n'
    assert b"Resolved!" in b"".join(bodies)


@pytest.mark.asyncio
async def test_buffered_responses_ccr_preserves_early_failure_status_and_headers():
    app = _make_app()
    body = {
        "model": "gpt-5-codex",
        "input": "fail early",
        "tools": [_RETRIEVE_TOOL],
        "stream": True,
    }

    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
        "query_string": b"",
        "headers": [(b"authorization", b"Bearer sk-test")],
        "server": ("testserver", 80),
        "client": ("testclient", 123),
        "root_path": "",
    }

    with TestClient(app):
        server = app.state.proxy

        async def early_failure(*args, **kwargs):  # noqa: ANN002, ANN003
            await asyncio.sleep(0.05)
            return httpx.Response(
                429,
                headers={"retry-after": "7"},
                json={"error": {"message": "slow down"}},
                request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
            )

        server._retry_request = early_failure
        response = await server.handle_openai_responses(Request(scope, receive))
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
async def test_buffered_responses_ccr_late_failure_emits_sanitized_error_event():
    app = _make_app()
    body = {
        "model": "gpt-5-codex",
        "input": "please wait",
        "tools": [_RETRIEVE_TOOL],
        "stream": True,
    }
    started = asyncio.Event()
    release = asyncio.Event()

    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
        "query_string": b"",
        "headers": [(b"authorization", b"Bearer sk-test")],
        "server": ("testserver", 80),
        "client": ("testclient", 123),
        "root_path": "",
    }

    with TestClient(app):
        server = app.state.proxy
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

        with patch.object(server.metrics, "record_failed", new_callable=AsyncMock) as record_failed:
            server._retry_request = delayed_failure
            task = asyncio.create_task(server.handle_openai_responses(Request(scope, receive)))
            await started.wait()
            response = await asyncio.wait_for(asyncio.shield(task), 1)
            events: list[dict] = []
            first_body = asyncio.Event()

            async def send(message):  # noqa: ANN001
                events.append(message)
                if message["type"] == "http.response.body" and message["body"]:
                    first_body.set()

            response_task = asyncio.create_task(response(scope, receive, send))
            await asyncio.wait_for(first_body.wait(), 2)
            release.set()
            await response_task
            record_failed.assert_awaited_once_with(provider="openai")
        proxy_logger.removeHandler(log_handler)

    bodies = [event["body"] for event in events if event["type"] == "http.response.body"]
    assert bodies[0] == b'event: ping\ndata: {"type":"ping"}\n\n'
    assert b"An error occurred while processing the request." in bodies[-1]
    assert b"boom" not in bodies[-1]
    assert events[-1]["more_body"] is False
    assert any(
        record.levelno == logging.ERROR and "RuntimeError: boom" in record.getMessage()
        for record in error_records
    )


@pytest.mark.asyncio
async def test_buffered_responses_ccr_pre_keepalive_exception_returns_json_error():
    app = _make_app()
    body = {
        "model": "gpt-5-codex",
        "input": "fail before keepalive",
        "tools": [_RETRIEVE_TOOL],
        "stream": True,
    }

    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
        "query_string": b"",
        "headers": [(b"authorization", b"Bearer sk-test")],
        "server": ("testserver", 80),
        "client": ("testclient", 123),
        "root_path": "",
    }

    with TestClient(app):
        server = app.state.proxy

        async def early_exception(*args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("boom")

        server._retry_request = early_exception
        response = await server.handle_openai_responses(Request(scope, receive))
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
