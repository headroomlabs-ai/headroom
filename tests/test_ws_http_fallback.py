"""Tests for WebSocket HTTP fallback in the OpenAI handler.

When the upstream WebSocket connection to OpenAI fails (HTTP 500),
the proxy should transparently fall back to HTTP POST streaming
and relay SSE events over the client WebSocket.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx


class FakeWebSocket:
    """Minimal WebSocket mock for testing."""

    def __init__(self):
        self.sent_texts: list[str] = []
        self.closed = False

    async def send_text(self, data: str) -> None:
        self.sent_texts.append(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = True


class FakeStreamResponse:
    """Mock httpx streaming response."""

    def __init__(self, status_code: int = 200, sse_events: list[str] | None = None):
        self.status_code = status_code
        self._events = sse_events or []

    async def aiter_text(self):
        for event in self._events:
            yield event

    async def aiter_bytes(self):
        yield b"error body"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeHttpClient:
    """Mock httpx.AsyncClient with stream support."""

    def __init__(self, response: FakeStreamResponse):
        self._response = response

    def stream(self, method, url, **kwargs):
        return self._response


def _make_handler():
    """Create a minimal OpenAIHandlerMixin-like object."""
    from headroom.proxy.handlers.openai import OpenAIHandlerMixin

    obj = object.__new__(OpenAIHandlerMixin)
    obj.OPENAI_API_URL = "https://api.openai.com"
    obj.http_client = None
    obj.config = SimpleNamespace(
        retry_max_attempts=3,
        retry_base_delay_ms=0,
        retry_max_delay_ms=0,
    )
    return obj


class TestWsHttpFallback:
    def test_fallback_relays_sse_events(self):
        """HTTP fallback should relay SSE data lines as WS text messages."""
        handler = _make_handler()
        ws = FakeWebSocket()
        sse_lines = [
            'event: response.created\ndata: {"type":"response.created","response":{"id":"r1"}}\n\n',
            'event: response.output_item.added\ndata: {"type":"response.output_item.added"}\n\n',
            'event: response.completed\ndata: {"type":"response.completed"}\n\n',
            "data: [DONE]\n\n",
        ]
        response = FakeStreamResponse(200, sse_lines)
        handler.http_client = FakeHttpClient(response)

        body = {"model": "gpt-5.4", "input": "hi"}
        first_msg_raw = json.dumps({"type": "response.create", "response": body})

        asyncio.run(
            handler._ws_http_fallback(
                ws, body, first_msg_raw, {"Authorization": "Bearer test"}, "req_1"
            )
        )

        assert len(ws.sent_texts) == 3  # 3 data events, [DONE] skipped
        assert '"response.created"' in ws.sent_texts[0]
        assert '"response.output_item.added"' in ws.sent_texts[1]
        assert '"response.completed"' in ws.sent_texts[2]
        assert ws.closed

    def test_fallback_sends_error_on_non_200(self):
        """HTTP fallback should send error event on non-200 response."""
        handler = _make_handler()
        ws = FakeWebSocket()
        response = FakeStreamResponse(status_code=401)
        handler.http_client = FakeHttpClient(response)

        body = {"model": "gpt-5.4", "input": "hi"}
        asyncio.run(
            handler._ws_http_fallback(
                ws, body, json.dumps(body), {"Authorization": "Bearer bad"}, "req_2"
            )
        )

        assert len(ws.sent_texts) == 1
        event = json.loads(ws.sent_texts[0])
        assert event["type"] == "error"
        assert "401" in event["error"]["message"]

    def test_fallback_sets_stream_true(self):
        """HTTP fallback should force stream=True in request body.

        After PR-A3 (byte-faithful Python forwarders) the fallback sends
        the request body as raw bytes via `content=`, not via the `json=`
        kwarg. The test extracts the posted JSON from the captured bytes.
        """
        handler = _make_handler()
        ws = FakeWebSocket()
        captured_kwargs: dict = {}

        class CapturingClient:
            def stream(self, method, url, **kwargs):
                captured_kwargs.update(kwargs)
                return FakeStreamResponse(200, ["data: [DONE]\n\n"])

        handler.http_client = CapturingClient()

        body = {"model": "gpt-5.4", "input": "test", "stream": False}
        asyncio.run(handler._ws_http_fallback(ws, body, json.dumps(body), {}, "req_3"))

        posted = json.loads(captured_kwargs["content"])
        assert posted["stream"] is True

    def test_fallback_unwraps_response_create_envelope(self):
        """HTTP fallback should unwrap WS response.create wrapper for HTTP POST."""
        handler = _make_handler()
        ws = FakeWebSocket()
        captured_kwargs: dict = {}

        class CapturingClient:
            def stream(self, method, url, **kwargs):
                captured_kwargs.update(kwargs)
                return FakeStreamResponse(200, ["data: [DONE]\n\n"])

        handler.http_client = CapturingClient()

        # WS sends wrapped format: {"type": "response.create", "response": {...}}
        inner = {
            "model": "gpt-5.4",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        }
        ws_msg = {"type": "response.create", "response": inner}
        asyncio.run(handler._ws_http_fallback(ws, ws_msg, json.dumps(ws_msg), {}, "req_unwrap"))

        posted = json.loads(captured_kwargs["content"])
        # Should be the inner response, not the wrapper
        assert "type" not in posted  # no "response.create" type field
        assert posted["model"] == "gpt-5.4"
        assert posted["stream"] is True
        assert "input" in posted

    def test_fallback_strips_top_level_response_create_type(self):
        """HTTP fallback should strip top-level response.create metadata."""
        handler = _make_handler()
        ws = FakeWebSocket()
        captured_kwargs: dict = {}

        class CapturingClient:
            def stream(self, method, url, **kwargs):
                captured_kwargs.update(kwargs)
                return FakeStreamResponse(200, ["data: [DONE]\n\n"])

        handler.http_client = CapturingClient()

        body = {"type": "response.create", "model": "gpt-5.4", "input": "hi"}
        asyncio.run(handler._ws_http_fallback(ws, body, json.dumps(body), {}, "req_type_strip"))

        posted = json.loads(captured_kwargs["content"])
        assert posted["model"] == "gpt-5.4"
        assert posted["stream"] is True
        assert "type" not in posted

    def test_fallback_handles_http_exception(self):
        """HTTP fallback should send error event when HTTP request fails."""
        handler = _make_handler()
        ws = FakeWebSocket()

        class FailingClient:
            def stream(self, method, url, **kwargs):
                raise ConnectionError("upstream unreachable")

        handler.http_client = FailingClient()

        body = {"model": "gpt-5.4", "input": "test"}
        asyncio.run(handler._ws_http_fallback(ws, body, json.dumps(body), {}, "req_4"))

        assert len(ws.sent_texts) == 1
        event = json.loads(ws.sent_texts[0])
        assert event["type"] == "error"
        assert "unreachable" in event["error"]["message"]

    def test_fallback_retries_connect_timeout(self):
        """HTTP fallback should retry transient connect timeouts."""
        handler = _make_handler()
        ws = FakeWebSocket()
        attempts = {"count": 0}

        class FlakyClient:
            def stream(self, method, url, **kwargs):
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise httpx.ConnectTimeout("timed out")
                return FakeStreamResponse(200, ['data: {"type":"response.completed"}\n\n'])

        handler.http_client = FlakyClient()

        body = {"model": "gpt-5.4", "input": "test"}
        asyncio.run(handler._ws_http_fallback(ws, body, json.dumps(body), {}, "req_retry"))

        assert attempts["count"] == 2
        assert len(ws.sent_texts) == 1
        assert json.loads(ws.sent_texts[0])["type"] == "response.completed"

    def test_fallback_routes_chatgpt_auth_to_chatgpt_domain(self):
        """ChatGPT session auth should route to chatgpt.com, not api.openai.com."""
        handler = _make_handler()
        ws = FakeWebSocket()
        captured_url = {}

        class CapturingClient:
            def stream(self, method, url, **kwargs):
                captured_url["url"] = url
                return FakeStreamResponse(200, ["data: [DONE]\n\n"])

        handler.http_client = CapturingClient()

        body = {"model": "gpt-5.4", "input": "test"}
        # ChatGPT session auth includes this header
        headers = {
            "Authorization": "Bearer chatgpt-session-token",
            "ChatGPT-Account-ID": "acct_abc123",
        }
        asyncio.run(handler._ws_http_fallback(ws, body, json.dumps(body), headers, "req_5"))

        assert "chatgpt.com" in captured_url["url"]
        assert "api.openai.com" not in captured_url["url"]

    def test_fallback_routes_api_key_to_openai(self):
        """API key auth should route to api.openai.com."""
        handler = _make_handler()
        ws = FakeWebSocket()
        captured_url = {}

        class CapturingClient:
            def stream(self, method, url, **kwargs):
                captured_url["url"] = url
                return FakeStreamResponse(200, ["data: [DONE]\n\n"])

        handler.http_client = CapturingClient()

        body = {"model": "gpt-5.4", "input": "test"}
        # API key auth — no ChatGPT-Account-ID header
        headers = {"Authorization": "Bearer sk-abc123"}
        asyncio.run(handler._ws_http_fallback(ws, body, json.dumps(body), headers, "req_6"))

        assert "api.openai.com" in captured_url["url"]
