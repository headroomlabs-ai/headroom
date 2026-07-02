"""Regression tests for Anthropic streaming CCR retrieval interception."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.responses import StreamingResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

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
        ccr_context_tracking=False,
        image_optimize=False,
    )


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
                    "messages": [{"role": "user", "content": "retrieve it"}],
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
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert resp.status_code == 200, resp.text
    assert "text/event-stream" in resp.headers["content-type"]
    assert "plain answer" in resp.text
    assert "headroom_retrieve" not in resp.text
    assert initial_bodies and initial_bodies[0]["stream"] is False
    assert continuation_client.post_calls == []
    proxy._stream_response.assert_not_awaited()


def test_mixed_ccr_and_client_tool_does_not_issue_continuation() -> None:
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
                "input": {"value": 1},
            },
        ],
        stop_reason="tool_use",
    )

    with patch("headroom.proxy.server.AnyLLMBackend"):
        app = create_app(config)
        with TestClient(app) as client:
            proxy = client.app.state.proxy
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
                    "messages": [{"role": "user", "content": "use tools"}],
                },
            )

    assert resp.status_code == 502, resp.text
    assert "text/event-stream" in resp.headers["content-type"]
    assert "headroom_retrieve" not in resp.text
    assert "client_tool" not in resp.text
    assert "Unable to safely complete streamed CCR retrieval" in resp.text
    assert continuation_client.post_calls == []
