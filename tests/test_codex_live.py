from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import socket
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import uvicorn
import websockets
from starlette.websockets import WebSocket

from headroom.providers.codex.live import (
    CODEX_LIVE_ROUTE_PATHS,
    DEFAULT_CODEX_LIVE_WS_PATH,
    _close_info,
    _ensure_live_authorization,
    _forward_headers,
    codex_live_websocket_url,
    codex_live_ws_path,
    handle_codex_live_websocket,
)
from headroom.providers.codex.runtime import resolve_codex_routing
from headroom.proxy.server import ProxyConfig, create_app


def _jwt(payload: dict[str, Any]) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


def test_live_aliases_are_registered_as_websocket_routes(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_REQUIRE_RUST_CORE", "false")
    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            openai_api_url="https://api.openai.test",
        )
    )

    for path in CODEX_LIVE_ROUTE_PATHS:
        matching = [route for route in app.routes if route.path == path]
        assert any("live_websocket" in route.name for route in matching)


def test_live_auth_modes_and_derived_paths(monkeypatch) -> None:
    subscription_headers = {
        "authorization": "Bearer "
        + _jwt(
            {
                "https://api.openai.com/auth": {"chatgpt_account_id": "acct-live"},
            }
        ),
    }
    subscription = _forward_headers(subscription_headers)
    decision = resolve_codex_routing(subscription)
    assert (
        codex_live_websocket_url(
            subscription=True,
            base_url="https://api.openai.test",
            path="/backend-api/live",
            query="model=gpt-5.4",
        )
        == "wss://chatgpt.com/backend-api/codex/live?model=gpt-5.4"
    )
    assert subscription["authorization"] == subscription_headers["authorization"]
    assert decision.headers["ChatGPT-Account-ID"] == "acct-live"

    monkeypatch.setenv("OPENAI_API_KEY", "env-live-key")
    assert _ensure_live_authorization({}) == {"Authorization": "Bearer env-live-key"}
    assert _ensure_live_authorization({"authorization": "Bearer client-key"}) == {
        "authorization": "Bearer client-key"
    }
    monkeypatch.delenv("OPENAI_API_KEY")
    existing_headers = {"X-Trace": "keep"}
    assert _ensure_live_authorization(existing_headers) == existing_headers

    assert (
        codex_live_websocket_url(
            subscription=False,
            base_url="http://127.0.0.1:9000",
            path="/backend-api/live",
            query="mode=live",
        )
        == "ws://127.0.0.1:9000/backend-api/live?mode=live"
    )
    assert DEFAULT_CODEX_LIVE_WS_PATH == "/live"
    monkeypatch.setenv("HEADROOM_CODEX_LIVE_WS_PATH", "/custom/live")
    assert codex_live_ws_path() == "/custom/live"
    assert (
        codex_live_websocket_url(
            subscription=True,
            base_url="https://api.openai.test",
            query="mode=live",
        )
        == "wss://chatgpt.com/backend-api/codex/custom/live?mode=live"
    )


def test_live_headers_strip_internal_and_handshake_headers_without_beta_injection() -> None:
    headers = _forward_headers(
        {
            "Authorization": "Bearer live-token",
            "ChatGPT-Account-ID": "acct-live",
            "OpenAI-Beta": "client-beta",
            "X-Headroom-User-ID": "private",
            "Connection": "keep-alive",
            "Sec-WebSocket-Key": "nonce",
            "Sec-WebSocket-Protocol": "codex.live.v1",
            "X-Trace": "keep",
        },
    )
    lowered = {key.lower(): value for key, value in headers.items()}
    assert lowered == {
        "authorization": "Bearer live-token",
        "chatgpt-account-id": "acct-live",
        "openai-beta": "client-beta",
        "x-trace": "keep",
    }


def test_live_close_info_preserves_wire_codes_and_reasons() -> None:
    assert _close_info({"code": 1008, "reason": "origin not allowed"}, 1011) == (
        1008,
        "origin not allowed",
    )
    assert _close_info(
        SimpleNamespace(rcvd=SimpleNamespace(code=1013, reason="upstream busy")),
        1011,
    ) == (1013, "upstream busy")
    assert _close_info({"code": 1006, "reason": "abnormal"}, 1011) == (1011, "abnormal")


class _LoopbackLiveUpstream:
    def __init__(self) -> None:
        self.server: Any = None
        self.port = 0
        self.headers: dict[str, str] = {}
        self.paths: list[str] = []
        self.received: list[str | bytes] = []

    async def handler(self, connection: Any) -> None:
        request = getattr(connection, "request", None)
        request_headers = getattr(request, "headers", None)
        if request_headers is None:
            request_headers = getattr(connection, "request_headers", {})
        self.headers = dict(request_headers)
        request_target = getattr(request, "path", None) or getattr(request, "target", None)
        if request_target is not None:
            self.paths.append(str(request_target))
        async for message in connection:
            assert isinstance(message, (str, bytes))
            self.received.append(message)
            await connection.send(message)

    async def start(self) -> None:
        self.port = _free_port()
        self.server = await websockets.serve(
            self.handler,
            "127.0.0.1",
            self.port,
            subprotocols=["codex.live.v1"],
        )

    async def stop(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()


class _ProxyThread:
    def __init__(self, port: int, upstream_port: int) -> None:
        app = create_app(
            ProxyConfig(
                host="127.0.0.1",
                port=port,
                optimize=False,
                cache_enabled=False,
                rate_limit_enabled=False,
                openai_api_url=f"http://127.0.0.1:{upstream_port}",
            )
        )
        self.server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                log_level="warning",
                loop="asyncio",
                lifespan="on",
                ws="websockets",
            )
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        deadline = time.perf_counter() + 15.0
        while time.perf_counter() < deadline:
            if self.server.started:
                return
            time.sleep(0.05)
        raise RuntimeError("uvicorn proxy failed to start")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=15.0)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.asyncio
async def test_live_handler_propagates_close_metadata_and_cleans_tasks(monkeypatch, caplog) -> None:
    class _Client:
        headers = {"authorization": "Bearer sk-live"}
        url = SimpleNamespace(query="turn=1")

        def __init__(self, message: dict[str, Any], *, fail_accept: bool = False) -> None:
            self.message = message
            self.fail_accept = fail_accept
            self.accepted = False
            self.closed: list[tuple[int | None, str | None]] = []
            self.cancelled = False

        async def accept(self, **kwargs: Any) -> None:
            if self.fail_accept:
                raise RuntimeError("accept failed")
            self.accepted = True

        async def close(self, code=None, reason=None) -> None:
            self.closed.append((code, reason))

        async def receive(self) -> dict[str, Any]:
            if self.message.get("type") == "wait":
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
            return self.message

        async def send_text(self, message: str) -> None:
            del message

        async def send_bytes(self, message: bytes) -> None:
            del message

    class _Upstream:
        subprotocol = None

        def __init__(self, error: Exception | None = None) -> None:
            self.error = error
            self.close_calls: list[tuple[int | None, str | None]] = []
            self.sent: list[str | bytes] = []
            self.cancelled = False

        async def send(self, message: str | bytes) -> None:
            self.sent.append(message)

        async def close(self, code=None, reason=None) -> None:
            self.close_calls.append((code, reason))

        def __aiter__(self):
            async def _events():
                if self.error is not None:
                    raise self.error
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
                if False:
                    yield b""

            return _events()

    proxy = SimpleNamespace(
        config=SimpleNamespace(openai_extra_headers=None, connect_timeout_seconds=10),
    )

    disconnect_client = _Client(
        {"type": "websocket.disconnect", "code": 1008, "reason": "client stopped"}
    )
    disconnect_upstream = _Upstream()
    monkeypatch.setattr(websockets, "connect", lambda *args, **kwargs: _await(disconnect_upstream))
    await handle_codex_live_websocket(
        disconnect_client,
        proxy,
        "https://api.openai.test",
        "/v1/live",
    )
    assert disconnect_upstream.close_calls[0] == (1008, "client stopped")

    subscription_client = _Client({"type": "websocket.disconnect", "code": 1000, "reason": "done"})
    subscription_client.headers = {
        "authorization": "Bearer "
        + _jwt(
            {
                "https://api.openai.com/auth": {
                    "chatgpt_account_id": "acct-live",
                }
            }
        )
    }
    subscription_upstream = _Upstream()
    subscription_connect: dict[str, Any] = {}

    async def connect_subscription(*args: Any, **kwargs: Any) -> Any:
        subscription_connect["url"] = args[0]
        subscription_connect["headers"] = kwargs["additional_headers"]
        return subscription_upstream

    monkeypatch.setattr(websockets, "connect", connect_subscription)
    await handle_codex_live_websocket(
        subscription_client,
        proxy,
        "https://api.openai.test",
        "/backend-api/live",
    )
    assert subscription_connect["url"] == "wss://chatgpt.com/backend-api/codex/live?turn=1"
    assert subscription_connect["headers"]["ChatGPT-Account-ID"] == "acct-live"

    fallback_client = _Client({"type": "websocket.disconnect", "code": 1000, "reason": "done"})
    fallback_client.headers = {}
    fallback_connect: dict[str, Any] = {}
    fallback_upstream = _Upstream()

    async def connect_fallback(*args: Any, **kwargs: Any) -> Any:
        fallback_connect["headers"] = kwargs["additional_headers"]
        return fallback_upstream

    monkeypatch.setenv("OPENAI_API_KEY", "env-live-key")
    monkeypatch.setattr(websockets, "connect", connect_fallback)
    await handle_codex_live_websocket(
        fallback_client,
        proxy,
        "https://api.openai.test",
        "/v1/live",
    )
    assert fallback_connect["headers"]["Authorization"] == "Bearer env-live-key"

    no_auth_client = _Client({"type": "websocket.disconnect", "code": 1000, "reason": "done"})
    no_auth_client.headers = {}
    no_auth_connect: dict[str, Any] = {}
    no_auth_upstream = _Upstream()

    async def connect_without_auth(*args: Any, **kwargs: Any) -> Any:
        no_auth_connect["headers"] = kwargs["additional_headers"]
        return no_auth_upstream

    monkeypatch.delenv("OPENAI_API_KEY")
    caplog.set_level(logging.WARNING, logger="headroom.providers.codex.live")
    monkeypatch.setattr(websockets, "connect", connect_without_auth)
    await handle_codex_live_websocket(
        no_auth_client,
        proxy,
        "https://api.openai.test",
        "/v1/live",
    )
    assert no_auth_connect["headers"] == {}
    assert "Codex Live has no Authorization header or OPENAI_API_KEY" in caplog.text

    class _QueuedClient(WebSocket):
        def __init__(self, messages: list[dict[str, Any]]) -> None:
            scope = {
                "type": "websocket",
                "path": "/v1/live",
                "raw_path": b"/v1/live",
                "scheme": "ws",
                "query_string": b"",
                "headers": [],
                "client": ("127.0.0.1", 1234),
                "server": ("127.0.0.1", 8788),
                "subprotocols": [],
            }
            super().__init__(scope, self._receive_asgi, self._send_asgi)
            self._messages = iter([{"type": "websocket.connect"}, *messages])
            self.sent: list[dict[str, Any]] = []

        async def _receive_asgi(self) -> dict[str, Any]:
            return {"type": "websocket.connect"}

        async def _send_asgi(self, message: dict[str, Any]) -> None:
            self.sent.append(message)

        async def receive(self) -> dict[str, Any]:
            # Starlette rejects unknown connected-state messages before the
            # handler can reach its defensive branch, so inject it at this seam.
            try:
                return next(self._messages)
            except StopIteration:
                return {"type": "websocket.disconnect", "code": 1000, "reason": "done"}

    queued_client = _QueuedClient(
        [
            {"type": "websocket.other"},
            {"type": "websocket.receive"},
            {"type": "websocket.disconnect", "code": 1000, "reason": "done"},
        ]
    )
    queued_upstream = _Upstream()
    monkeypatch.setattr(websockets, "connect", lambda *args, **kwargs: _await(queued_upstream))
    await handle_codex_live_websocket(
        queued_client,
        proxy,
        "https://api.openai.test",
        "/v1/live",
    )
    assert queued_upstream.sent == []
    assert queued_upstream.close_calls[0] == (1000, "done")

    cancellation_client = _Client({"type": "wait"})
    cancellation_upstream = _Upstream()
    monkeypatch.setattr(
        websockets, "connect", lambda *args, **kwargs: _await(cancellation_upstream)
    )
    handler_task = asyncio.create_task(
        handle_codex_live_websocket(
            cancellation_client,
            proxy,
            "https://api.openai.test",
            "/v1/live",
        )
    )
    while not cancellation_client.accepted:
        await asyncio.sleep(0)
    handler_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handler_task
    assert cancellation_client.cancelled
    assert cancellation_upstream.cancelled
    assert cancellation_upstream.close_calls

    class _UpstreamFailure(Exception):
        rcvd = SimpleNamespace(code=1013, reason="upstream busy")

    failure_client = _Client({"type": "wait"})
    failure_upstream = _Upstream(_UpstreamFailure("busy"))
    monkeypatch.setattr(websockets, "connect", lambda *args, **kwargs: _await(failure_upstream))
    await handle_codex_live_websocket(
        failure_client,
        proxy,
        "https://api.openai.test",
        "/v1/live",
    )
    assert failure_client.closed[-1] == (1013, "upstream busy")

    handshake_client = _Client({"type": "wait"})

    async def fail_handshake(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("handshake failed")

    monkeypatch.setattr(websockets, "connect", fail_handshake)
    await handle_codex_live_websocket(
        handshake_client,
        proxy,
        "https://api.openai.test",
        "/v1/live",
    )
    assert handshake_client.closed[-1] == (1011, "upstream connection failed")

    class _ReceiveFailureClient(_Client):
        async def receive(self) -> dict[str, Any]:
            raise RuntimeError("client receive failed")

    task_failure_client = _ReceiveFailureClient({"type": "wait"})
    task_failure_upstream = _Upstream()
    monkeypatch.setattr(
        websockets, "connect", lambda *args, **kwargs: _await(task_failure_upstream)
    )
    await handle_codex_live_websocket(
        task_failure_client,
        proxy,
        "https://api.openai.test",
        "/v1/live",
    )
    assert task_failure_client.closed[-1] == (1011, "")

    accept_client = _Client({"type": "wait"}, fail_accept=True)
    accept_upstream = _Upstream()
    monkeypatch.setattr(websockets, "connect", lambda *args, **kwargs: _await(accept_upstream))
    await handle_codex_live_websocket(
        accept_client,
        proxy,
        "https://api.openai.test",
        "/v1/live",
    )
    assert accept_upstream.close_calls

    missing_client = _Client({"type": "wait"})
    with patch.dict(sys.modules, {"websockets": None}):
        await handle_codex_live_websocket(
            missing_client,
            proxy,
            "https://api.openai.test",
            "/v1/live",
        )
    assert missing_client.accepted
    assert missing_client.closed[-1] == (
        1011,
        "websockets package not installed; pip install websockets",
    )


async def _await(value: Any) -> Any:
    return value


@pytest.mark.asyncio
async def test_live_rg4_real_uvicorn_and_websockets_binary_round_trip() -> None:
    previous = os.environ.get("HEADROOM_REQUIRE_RUST_CORE")
    os.environ["HEADROOM_REQUIRE_RUST_CORE"] = "false"
    upstream = _LoopbackLiveUpstream()
    await upstream.start()
    proxy = _ProxyThread(_free_port(), upstream.port)
    proxy.start()
    payload = b"\x00\x01live-voice\xff"
    expected: list[str | bytes] = []
    try:
        with pytest.raises(websockets.exceptions.InvalidStatus):
            async with websockets.connect(
                f"ws://127.0.0.1:{proxy.server.config.port}/v1/live",
                additional_headers={
                    "Authorization": "Bearer sk-live",
                    "Origin": "https://remote.example",
                },
            ):
                raise AssertionError("disallowed origin unexpectedly connected")
        assert upstream.received == []

        for index, path in enumerate(CODEX_LIVE_ROUTE_PATHS):
            text_payload = f"live-text-{index}"
            binary_payload = payload + bytes([index])
            async with websockets.connect(
                f"ws://127.0.0.1:{proxy.server.config.port}{path}?turn={index}",
                additional_headers={
                    "Authorization": "Bearer sk-live",
                    "OpenAI-Beta": "client-beta",
                    "X-Headroom-User-ID": "private",
                },
                subprotocols=["codex.live.v1"],
            ) as client:
                response = getattr(client, "response", None)
                assert response is not None
                assert response.status_code == 101
                assert client.subprotocol == "codex.live.v1"
                await client.send(text_payload)
                assert await client.recv() == text_payload
                await client.send(binary_payload)
                assert await client.recv() == binary_payload
            expected.extend([text_payload, binary_payload])

        assert upstream.received == expected
        assert upstream.paths == [
            f"{path}?turn={index}" for index, path in enumerate(CODEX_LIVE_ROUTE_PATHS)
        ]
        assert upstream.headers["authorization"] == "Bearer sk-live"
        assert upstream.headers["openai-beta"] == "client-beta"
        assert "x-headroom-user-id" not in upstream.headers
    finally:
        proxy.stop()
        await upstream.stop()
        if previous is None:
            os.environ.pop("HEADROOM_REQUIRE_RUST_CORE", None)
        else:
            os.environ["HEADROOM_REQUIRE_RUST_CORE"] = previous
