"""Unit: event-loop callback handling for Codex WS disconnect regressions."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def _known_loop_callback_context() -> dict[str, object]:
    return {
        "message": "Exception in callback Connection.connection_lost(ConnectionResetError())",
        "exception": AttributeError("'ClientConnection' object has no attribute 'recv_messages'"),
    }


def _make_client(app):
    return TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345))


def test_livez_reports_known_websockets_callback_degradation():
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)

    with _make_client(app) as client:
        before = client.get("/livez")
        assert before.status_code == 200
        assert before.json()["status"] == "healthy"
        assert before.json()["alive"] is True

        assert app.state.loop_exception_handler is not None
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        app.state.loop_exception_handler(mock_loop, _known_loop_callback_context())

        after = client.get("/livez")
        assert after.status_code == 503
        payload = after.json()
        assert payload["status"] == "unhealthy"
        assert payload["alive"] is False
        loop_health = payload["loop_health"]
        assert loop_health["status"] == "unhealthy"
        assert loop_health["known_failures"] == 1
        assert (
            loop_health["last_known_failure"]["exception"]
            == "'ClientConnection' object has no attribute 'recv_messages'"
        )


def test_unrelated_loop_callback_is_delegated_to_previous_handler():
    delegate_calls: list[dict[str, object]] = []
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)

    with _make_client(app) as client:
        client.get("/livez")
        assert app.state.loop_exception_handler is not None

        def _previous(_loop: object, context: dict[str, object]) -> None:
            delegate_calls.append(dict(context))

        app.state.previous_loop_exception_handler = _previous

        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        app.state.loop_exception_handler(
            mock_loop,
            {
                "message": "random callback failed",
                "exception": RuntimeError("not known failure"),
            },
        )

        assert len(delegate_calls) == 1
        assert delegate_calls[0]["message"] == "random callback failed"
        assert app.state.loop_callback_health["status"] == "healthy"
        assert app.state.loop_callback_health["known_failures"] == 0

        health = client.get("/livez").json()
        assert health["status"] == "healthy"
        assert health["alive"] is True
