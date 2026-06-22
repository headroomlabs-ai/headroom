"""Fail-closed enforcement for HEADROOM_UPSTREAM_ROUTES BearerAuth routes.

When a request matches a ``auth: env:VARNAME`` route whose env var is
unset/empty, the inbound Authorization/x-api-key has already been stripped
and there is no replacement token. The proxy MUST fail closed -- return a
502 (or a WS error event) *before* contacting the upstream -- rather than
forward the request unauthenticated.

These tests drive the actual handler call sites end-to-end and assert the
upstream HTTP client is never invoked. The pre-existing
``test_multi_upstream_routes.py`` exercises ``resolve_upstream`` in
isolation; that level cannot catch a call site that ignores the result,
which is exactly the gap these tests close.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from headroom.providers.registry import UpstreamAuthUnavailable

# A single BearerAuth route whose token env var is intentionally left unset.
_ROUTES = json.dumps(
    [{"model_prefix": "glm-", "upstream": "https://ollama.test", "auth": "env:FAILCLOSED_TEST_KEY"}]
)


@pytest.fixture
def _routed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure a BearerAuth route with a missing token env var."""
    monkeypatch.setenv("HEADROOM_UPSTREAM_ROUTES", _ROUTES)
    monkeypatch.delenv("FAILCLOSED_TEST_KEY", raising=False)


def _config() -> Any:
    from headroom.proxy.server import ProxyConfig

    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        anthropic_api_url="https://api.anthropic.test",
        openai_api_url="https://api.openai.test",
        gemini_api_url="https://api.gemini.test",
        cloudcode_api_url="https://cloudcode.test",
        vertex_api_url="https://vertex.test",
    )


# --- resolve_upstream raises (unit-level guard for the new exception) ---


def test_resolve_upstream_raises_on_empty_bearer_env(_routed_env: None) -> None:
    from headroom.proxy.server import HeadroomProxy

    proxy = HeadroomProxy(_config())
    with pytest.raises(UpstreamAuthUnavailable) as exc:
        proxy.resolve_upstream(
            protocol="openai",
            model="glm-4.6",
            headers={"authorization": "Bearer inbound-secret"},
        )
    # Carries the offending env var; the inbound secret is never echoed.
    assert exc.value.env_var == "FAILCLOSED_TEST_KEY"
    assert "inbound-secret" not in str(exc.value)


def test_resolve_upstream_passthrough_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no routes, the legacy passthrough path must not raise."""
    from headroom.proxy.server import HeadroomProxy

    monkeypatch.delenv("HEADROOM_UPSTREAM_ROUTES", raising=False)
    proxy = HeadroomProxy(_config())
    base_url, headers = proxy.resolve_upstream(
        protocol="openai",
        model="glm-4.6",
        headers={"authorization": "Bearer keep-me"},
    )
    # Byte-identical legacy fallback: headers pass through unchanged.
    assert headers.get("authorization") == "Bearer keep-me"
    assert isinstance(base_url, str) and base_url


# --- HTTP call sites: 502 before upstream is contacted ---


class _SendGuard:
    """Records any attempt to send over an httpx client."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, _self: Any, request: httpx.Request, *a: Any, **k: Any) -> Any:
        self.calls.append(str(request.url))
        raise AssertionError(f"upstream was contacted: {request.url}")


def _client(_routed_env: None) -> Any:
    from fastapi.testclient import TestClient

    from headroom.proxy.server import create_app

    return TestClient(create_app(_config()), raise_server_exceptions=False)


def test_chat_completions_fails_closed(_routed_env: None) -> None:
    guard = _SendGuard()
    with patch.object(httpx.AsyncClient, "send", guard):
        resp = _client(_routed_env).post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer inbound"},
            json={"model": "glm-4.6", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 502, resp.text
    assert resp.json()["error"]["code"] == "upstream_auth_unavailable"
    assert guard.calls == []


def test_responses_fails_closed(_routed_env: None) -> None:
    guard = _SendGuard()
    with patch.object(httpx.AsyncClient, "send", guard):
        resp = _client(_routed_env).post(
            "/v1/responses",
            headers={"authorization": "Bearer inbound"},
            json={"model": "glm-4.6", "input": "hi"},
        )
    assert resp.status_code == 502, resp.text
    assert resp.json()["error"]["code"] == "upstream_auth_unavailable"
    assert guard.calls == []


def test_anthropic_messages_fails_closed(_routed_env: None) -> None:
    guard = _SendGuard()
    with patch.object(httpx.AsyncClient, "send", guard):
        resp = _client(_routed_env).post(
            "/v1/messages",
            headers={"x-api-key": "inbound", "anthropic-version": "2023-06-01"},
            json={
                "model": "glm-4.6",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert resp.status_code == 502, resp.text
    assert resp.json()["error"]["type"] == "server_error"
    assert guard.calls == []


# --- WS-to-HTTP fallback call site: error event, no upstream contact ---


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent_texts: list[str] = []
        self.closed = False

    async def send_text(self, data: str) -> None:
        self.sent_texts.append(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = True


def test_ws_fallback_fails_closed(_routed_env: None) -> None:
    from headroom.proxy.server import HeadroomProxy

    proxy = HeadroomProxy(_config())
    ws = _FakeWebSocket()
    body = {"model": "glm-4.6", "input": "hi"}
    first_msg_raw = json.dumps({"type": "response.create", "response": body})

    guard = _SendGuard()
    with patch.object(httpx.AsyncClient, "send", guard):
        asyncio.run(
            proxy._ws_http_fallback(
                ws,
                body,
                first_msg_raw,
                {"authorization": "Bearer inbound"},
                "req_failclosed",
            )
        )

    # Exactly one WS error event relayed, and no upstream contact.
    assert len(ws.sent_texts) == 1, ws.sent_texts
    event = json.loads(ws.sent_texts[0])
    assert event["type"] == "error"
    assert event["error"]["type"] == "server_error"
    assert guard.calls == []
