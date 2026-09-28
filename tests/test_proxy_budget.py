"""Integration tests for proxy budget enforcement (Issue #3374).

Verifies that `--budget` limits are enforced on these generation routes:
- OpenAI chat completions (`/v1/chat/completions`)
- OpenAI responses (`/v1/responses` HTTP and WebSocket: handshake, first frame,
  later `response.create` frames, and the HTTP fallback)
- Gemini generate content (`/v1beta/models/{model}:generateContent`)
- Gemini stream generate content (`/v1beta/models/{model}:streamGenerateContent`)
- Google Cloud Code Assist stream (`/v1internal:streamGenerateContent`)
- Anthropic messages (`/v1/messages`) for parity
"""

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from headroom.proxy.server import ProxyConfig, create_app


class _CountingMockTransport(httpx.AsyncBaseTransport):
    """Mock transport that tracks calls and returns valid mock LLM responses."""

    def __init__(self) -> None:
        self.call_count = 0
        self.captured_requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        self.captured_requests.append(request)

        url_str = str(request.url)

        if "chat/completions" in url_str:
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hello!"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            )

        if "responses" in url_str:
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={
                    "id": "resp-mock",
                    "object": "response",
                    "created": 1234567890,
                    "model": "gpt-4o",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Hello!"}],
                        }
                    ],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            )

        if "generateContent" in url_str:
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "Hello from Gemini!"}],
                                "role": "model",
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 5,
                        "totalTokenCount": 15,
                    },
                },
            )

        # Anthropic messages
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "id": "msg_mock",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello from Claude!"}],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                },
            },
        )


def _build_proxy_client(
    *,
    budget_limit_usd: float | None = None,
    budget_period: str = "daily",
    cost_tracking_enabled: bool = True,
) -> tuple[TestClient, _CountingMockTransport]:
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=cost_tracking_enabled,
        budget_limit_usd=budget_limit_usd,
        budget_period=budget_period,  # type: ignore[arg-type]
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        anthropic_api_url="https://api.anthropic.test",
        openai_api_url="https://api.openai.test",
        gemini_api_url="https://api.gemini.test",
    )
    app = create_app(config)
    transport = _CountingMockTransport()
    proxy = app.state.proxy
    proxy.http_client = httpx.AsyncClient(transport=transport)
    return TestClient(app), transport


def test_zero_budget_blocks_openai_chat_completions() -> None:
    client, transport = _build_proxy_client(budget_limit_usd=0.0)

    response = client.post(
        "/v1/chat/completions",
        headers={"authorization": "Bearer sk-test"},
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 429
    assert response.json()["detail"] == "Budget exceeded for daily period"
    assert transport.call_count == 0


def test_zero_budget_blocks_openai_responses() -> None:
    client, transport = _build_proxy_client(budget_limit_usd=0.0)

    response = client.post(
        "/v1/responses",
        headers={"authorization": "Bearer sk-test"},
        json={
            "model": "gpt-4o",
            "input": "Hello",
        },
    )

    assert response.status_code == 429
    assert response.json()["detail"] == "Budget exceeded for daily period"
    assert transport.call_count == 0


def test_zero_budget_blocks_gemini_generate_content() -> None:
    client, transport = _build_proxy_client(budget_limit_usd=0.0)

    response = client.post(
        "/v1beta/models/gemini-1.5-pro:generateContent",
        headers={"x-goog-api-key": "test-gemini-key"},
        json={
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        },
    )

    assert response.status_code == 429
    assert response.json()["detail"] == "Budget exceeded for daily period"
    assert transport.call_count == 0


def test_zero_budget_blocks_gemini_stream_generate_content() -> None:
    client, transport = _build_proxy_client(budget_limit_usd=0.0)

    response = client.post(
        "/v1beta/models/gemini-1.5-pro:streamGenerateContent",
        headers={"x-goog-api-key": "test-gemini-key"},
        json={
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        },
    )

    assert response.status_code == 429
    assert response.json()["detail"] == "Budget exceeded for daily period"
    assert transport.call_count == 0


def test_zero_budget_blocks_google_cloudcode_stream() -> None:
    client, transport = _build_proxy_client(budget_limit_usd=0.0)

    response = client.post(
        "/v1internal:streamGenerateContent",
        json={
            "model": "gemini-2.5-pro",
            "request": {"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
        },
    )

    assert response.status_code == 429
    assert response.json()["detail"] == "Budget exceeded for daily period"
    assert transport.call_count == 0


def test_zero_budget_blocks_anthropic_messages_parity() -> None:
    client, transport = _build_proxy_client(budget_limit_usd=0.0)

    response = client.post(
        "/v1/messages",
        headers={
            "x-api-key": "sk-ant-test",
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        },
    )

    assert response.status_code == 429
    assert response.json()["detail"] == "Budget exceeded for daily period"
    assert transport.call_count == 0


def test_unlimited_budget_allows_all_generation_routes() -> None:
    client, transport = _build_proxy_client(budget_limit_usd=None)

    # OpenAI chat
    resp_chat = client.post(
        "/v1/chat/completions",
        headers={"authorization": "Bearer sk-test"},
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert resp_chat.status_code == 200

    # OpenAI responses
    resp_resp = client.post(
        "/v1/responses",
        headers={"authorization": "Bearer sk-test"},
        json={"model": "gpt-4o", "input": "Hello"},
    )
    assert resp_resp.status_code == 200

    # Gemini generateContent
    resp_gem = client.post(
        "/v1beta/models/gemini-1.5-pro:generateContent",
        headers={"x-goog-api-key": "test-key"},
        json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
    )
    assert resp_gem.status_code == 200

    assert transport.call_count == 3


def test_zero_budget_websocket_preflight_rejected() -> None:
    client, _ = _build_proxy_client(budget_limit_usd=0.0)

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/v1/responses"):
            pass

    assert exc_info.value.code == 1008


def test_dynamic_budget_exhaustion_blocks_all_providers() -> None:
    client, transport = _build_proxy_client(budget_limit_usd=0.05)

    # First request succeeds
    resp1 = client.post(
        "/v1/chat/completions",
        headers={"authorization": "Bearer sk-test"},
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert resp1.status_code == 200
    assert transport.call_count == 1

    # Simulate accumulated spend that exceeds the $0.05 budget
    proxy = client.app.state.proxy  # type: ignore[attr-defined]
    assert proxy.cost_tracker is not None
    proxy.cost_tracker.record_tokens(
        model="gpt-4o",
        tokens_saved=0,
        tokens_sent=100_000,
        output_tokens=50_000,
    )

    allowed, _ = proxy.cost_tracker.check_budget()
    assert not allowed

    # Now all provider generation routes must reject with 429
    routes_and_payloads = [
        (
            "/v1/chat/completions",
            {"authorization": "Bearer sk-test"},
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
        ),
        (
            "/v1/responses",
            {"authorization": "Bearer sk-test"},
            {"model": "gpt-4o", "input": "Hi"},
        ),
        (
            "/v1beta/models/gemini-1.5-pro:generateContent",
            {"x-goog-api-key": "test-key"},
            {"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]},
        ),
        (
            "/v1/messages",
            {"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"},
            {
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
            },
        ),
    ]

    for route, headers, body in routes_and_payloads:
        resp = client.post(route, headers=headers, json=body)
        assert resp.status_code == 429
        assert "Budget exceeded for daily period" in resp.json()["detail"]

    # Upstream should not have received any additional calls after budget exhaustion
    assert transport.call_count == 1


class _FakeWebSocketDisconnect(Exception):
    """Exception matching WebSocketDisconnect type-name check."""


_FakeWebSocketDisconnect.__name__ = "WebSocketDisconnect_Fake"


class _FakeUpstream:
    """Fake upstream connection that records frames sent by Headroom."""

    def __init__(self, events: list[str]) -> None:
        self._events = list(events)
        self.sent: list[str] = []
        self.closed = False
        self.response = SimpleNamespace(headers=[])

    async def __aenter__(self) -> _FakeUpstream:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.closed = True

    async def send(self, payload: str) -> None:
        self.sent.append(payload)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for ev in self._events:
            yield ev
        await asyncio.Event().wait()


class _ScriptedClientWS:
    """Scripted client WebSocket delivering frames and tracking close state."""

    def __init__(self, frames: list[str], *, on_frame2_callback=None) -> None:
        self.headers = {"authorization": "Bearer test"}
        self._frames = list(frames)
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []
        self.closed = False
        self.close_code: int | None = None
        self.close_reason: str | None = None
        self.client = SimpleNamespace(host="127.0.0.1", port=12345)
        self._on_frame2_callback = on_frame2_callback

    async def accept(self, subprotocol=None, headers=None) -> None:
        pass

    async def receive_text(self) -> str:
        if not self._frames:
            raise _FakeWebSocketDisconnect("client closed")
        frame = self._frames.pop(0)
        if len(self._frames) == 0 and self._on_frame2_callback is not None:
            self._on_frame2_callback()
        return frame

    async def send_text(self, text: str) -> None:
        self.sent_text.append(text)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.closed = True
        if code is not None or self.close_code is None:
            self.close_code = code
        if reason is not None or self.close_reason is None:
            self.close_reason = reason


def test_websocket_per_turn_budget_enforcement_blocks_late_response_create() -> None:
    """A long-lived /v1/responses WebSocket opened under budget must reject subsequent

    response.create frames once the budget is exhausted, and not forward them upstream.
    """

    async def _run() -> None:
        config = ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=True,
            budget_limit_usd=0.05,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            image_optimize=False,
        )
        app = create_app(config)
        proxy = app.state.proxy
        assert proxy.cost_tracker is not None

        # Verify initial state opens under budget
        allowed, _ = proxy.cost_tracker.check_budget()
        assert allowed

        first_frame = json.dumps(
            {
                "type": "response.create",
                "response": {"model": "gpt-4o", "input": "first turn"},
            }
        )
        second_frame = json.dumps(
            {
                "type": "response.create",
                "response": {"model": "gpt-4o", "input": "second turn"},
            }
        )

        upstream_events = [
            json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
            json.dumps(
                {
                    "type": "response.completed",
                    "response": {
                        "id": "r_1",
                        "model": "gpt-4o",
                        "usage": {"input_tokens": 10, "output_tokens": 1},
                    },
                }
            ),
        ]
        upstream = _FakeUpstream(upstream_events)

        mod = MagicMock()

        async def _fake_connect(*args, **kwargs):
            return upstream

        mod.connect = _fake_connect
        mod.Subprotocol = str

        def _exhaust_budget() -> None:
            proxy.cost_tracker.record_tokens(
                model="gpt-4o",
                tokens_saved=0,
                tokens_sent=100_000,
                output_tokens=50_000,
            )

        client_ws = _ScriptedClientWS(
            [first_frame, second_frame],
            on_frame2_callback=_exhaust_budget,
        )
        deregister_spy = MagicMock(side_effect=proxy.ws_sessions.deregister_and_count)
        proxy.ws_sessions.deregister_and_count = deregister_spy

        with patch.dict(sys.modules, {"websockets": mod}):
            await asyncio.wait_for(
                proxy.handle_openai_responses_ws(client_ws),
                timeout=3.0,
            )

        # Proves:
        # 1. First frame was allowed and forwarded upstream
        assert len(upstream.sent) == 1
        assert json.loads(upstream.sent[0])["response"]["input"] == "first turn"

        # 2. Budget is exhausted after turn 1 spend
        allowed, _ = proxy.cost_tracker.check_budget()
        assert not allowed

        # 3. Subsequent response.create on same socket was rejected with 1008
        assert client_ws.closed is True
        assert client_ws.close_code == 1008
        assert "Budget exceeded for daily period" in (client_ws.close_reason or "")

        # 4. Subsequent response.create was NOT forwarded upstream
        assert len(upstream.sent) == 1

        # 5. Session teardown records the budget rejection as the termination cause
        deregister_spy.assert_called_once()
        assert deregister_spy.call_args.kwargs["cause"] == "budget_exceeded"

    asyncio.run(_run())


def test_websocket_turn_completion_usage_exhausts_budget_and_blocks_next_turn() -> None:
    """When turn 1 usage naturally exhausts the daily budget via the outcome funnel,

    a subsequent response.create turn on the same socket is rejected and not forwarded.
    """

    async def _run() -> None:
        config = ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=True,
            budget_limit_usd=0.05,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            image_optimize=False,
        )
        app = create_app(config)
        proxy = app.state.proxy
        assert proxy.cost_tracker is not None

        allowed, _ = proxy.cost_tracker.check_budget()
        assert allowed

        first_frame = json.dumps(
            {
                "type": "response.create",
                "response": {"model": "gpt-4o", "input": "turn 1"},
            }
        )
        second_frame = json.dumps(
            {
                "type": "response.create",
                "response": {"model": "gpt-4o", "input": "turn 2"},
            }
        )

        upstream_events = [
            json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
            json.dumps(
                {
                    "type": "response.completed",
                    "response": {
                        "id": "r_1",
                        "model": "gpt-4o",
                        "usage": {"input_tokens": 100_000, "output_tokens": 50_000},
                    },
                }
            ),
        ]
        upstream = _FakeUpstream(upstream_events)

        mod = MagicMock()

        async def _fake_connect(*args, **kwargs):
            return upstream

        mod.connect = _fake_connect
        mod.Subprotocol = str

        class _DelayedClientWS(_ScriptedClientWS):
            async def receive_text(self) -> str:
                if not self._frames:
                    raise _FakeWebSocketDisconnect("client closed")
                # Hold turn 2 until turn 1's response.completed usage is booked,
                # rather than sleeping for a fixed wall-clock interval.
                if len(self._frames) == 1:
                    for _ in range(400):
                        if not proxy.cost_tracker.check_budget()[0]:
                            break
                        await asyncio.sleep(0.005)
                return self._frames.pop(0)

        client_ws = _DelayedClientWS([first_frame, second_frame])

        with patch.dict(sys.modules, {"websockets": mod}):
            await asyncio.wait_for(
                proxy.handle_openai_responses_ws(client_ws),
                timeout=3.0,
            )

        assert client_ws.closed is True
        assert client_ws.close_code == 1008
        assert "Budget exceeded" in (client_ws.close_reason or "")
        assert len(upstream.sent) == 1
        assert json.loads(upstream.sent[0])["response"]["input"] == "turn 1"

        allowed, _ = proxy.cost_tracker.check_budget()
        assert not allowed

    asyncio.run(_run())


def _ws_budget_proxy(**overrides):
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=True,
        budget_limit_usd=0.05,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        **overrides,
    )
    proxy = create_app(config).state.proxy
    assert proxy.cost_tracker is not None
    assert proxy.cost_tracker.check_budget()[0]
    return proxy


def _spend_past_budget(proxy) -> None:
    proxy.cost_tracker.record_tokens(
        model="gpt-4o",
        tokens_saved=0,
        tokens_sent=100_000,
        output_tokens=50_000,
    )


def _fake_websockets_module(connect):
    mod = MagicMock()
    mod.connect = connect
    mod.Subprotocol = str
    return mod


def _response_create_frame(text: str) -> str:
    return json.dumps({"type": "response.create", "response": {"model": "gpt-4o", "input": text}})


def test_websocket_first_frame_blocked_when_budget_exhausted_after_handshake() -> None:
    """Budget spent between the handshake preflight and the first frame must stop
    that frame before it reaches the already-connected upstream socket.
    """

    async def _run() -> None:
        proxy = _ws_budget_proxy()
        upstream = _FakeUpstream([])

        async def _fake_connect(*args, **kwargs):
            _spend_past_budget(proxy)
            return upstream

        client_ws = _ScriptedClientWS([_response_create_frame("hi")])

        with patch.dict(sys.modules, {"websockets": _fake_websockets_module(_fake_connect)}):
            await asyncio.wait_for(
                proxy.handle_openai_responses_ws(client_ws),
                timeout=3.0,
            )

        assert upstream.sent == []
        assert upstream.closed is True
        assert client_ws.close_code == 1008
        assert "Budget exceeded for daily period" in (client_ws.close_reason or "")

    asyncio.run(_run())


def test_websocket_http_fallback_blocked_when_budget_exhausted() -> None:
    """When the upstream WebSocket upgrade fails, the HTTP fallback must not
    dispatch a paid request once the budget is exhausted.
    """

    async def _run() -> None:
        proxy = _ws_budget_proxy(retry_max_attempts=1)
        proxy._ws_http_fallback = AsyncMock(return_value=(0, 0, 0, 0, 0))

        async def _fake_connect(*args, **kwargs):
            _spend_past_budget(proxy)
            raise OSError("upstream websocket unavailable")

        client_ws = _ScriptedClientWS([_response_create_frame("hi")])

        with patch.dict(sys.modules, {"websockets": _fake_websockets_module(_fake_connect)}):
            await asyncio.wait_for(
                proxy.handle_openai_responses_ws(client_ws),
                timeout=3.0,
            )

        proxy._ws_http_fallback.assert_not_awaited()
        assert client_ws.close_code == 1008
        assert "Budget exceeded for daily period" in (client_ws.close_reason or "")

    asyncio.run(_run())
