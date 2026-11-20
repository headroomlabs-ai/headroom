from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Event

import pytest
import websockets
from fastapi.testclient import TestClient

from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app

EXAMPLE = (
    Path(__file__).parents[2]
    / "docs"
    / "proposals"
    / "unified-api-gateway"
    / "examples"
    / "gateway.api-keys.json"
)


def test_websocket_disconnect_closes_owned_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    sent = Event()
    closed = Event()

    class FakeUpstream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            closed.set()

        async def send(self, _frame: str) -> None:
            sent.set()

        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.Future()

    monkeypatch.setattr(websockets, "connect", lambda *_args, **_kwargs: FakeUpstream())
    app = create_app(ProxyConfig(gateway=GatewayConfigSnapshot.load(EXAMPLE)))

    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/responses",
            headers={"host": "127.0.0.1:8787", "authorization": "Bearer client-secret"},
        ) as socket:
            socket.send_json(
                {
                    "type": "response.create",
                    "response": {
                        "model": "REPLACE_WITH_ENABLED_OPENAI_MODEL",
                        "input": "one",
                    },
                }
            )
            assert sent.wait(2), "disconnect only after the upstream is owned"

        assert closed.wait(2), "disconnect must close the owned upstream"
