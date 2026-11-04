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
import base64
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
        subscription_tracking_enabled=False,
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


@pytest.mark.parametrize("wrapped", [False, True])
def test_ws_fallback_fails_closed(_routed_env: None, wrapped: bool) -> None:
    from headroom.proxy.server import HeadroomProxy

    proxy = HeadroomProxy(_config())
    ws = _FakeWebSocket()
    body = {"model": "glm-4.6", "input": "hi"}
    first_msg_raw = json.dumps({"type": "response.create", "response": body})

    ws_body = {"type": "response.create", "response": body} if wrapped else body
    guard = _SendGuard()
    with patch.object(httpx.AsyncClient, "send", guard):
        usage = asyncio.run(
            proxy._ws_http_fallback(
                ws,
                ws_body,
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
    assert usage == (0, 0, 0, 0, 0)


@pytest.mark.parametrize(
    ("path", "payload", "expected_path"),
    [
        (
            "/v1/chat/completions",
            {"model": "glm-4.6", "messages": [{"role": "user", "content": "hi"}]},
            "/v1/chat/completions",
        ),
        ("/v1/responses", {"model": "glm-4.6", "input": "hi"}, "/v1/responses"),
        (
            "/v1/messages",
            {"model": "glm-4.6", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
            "/v1/messages",
        ),
    ],
)
@pytest.mark.parametrize("routed", [True, False])
def test_route_forwarding_keeps_url_auth_and_extra_header_scope_together(
    _routed_env: None,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    payload: dict[str, Any],
    expected_path: str,
    routed: bool,
) -> None:
    from fastapi.testclient import TestClient

    from headroom.proxy.server import create_app

    monkeypatch.setenv("FAILCLOSED_TEST_KEY", "route-secret")
    payload = dict(payload, model="glm-4.6" if routed else "unmatched-model")
    config = _config()
    config.openai_extra_headers = {"X-Operator-Secret": "openai-secret"}
    config.anthropic_extra_headers = {"X-Operator-Secret": "anthropic-secret"}
    sent: list[httpx.Request] = []

    async def capture_send(client: Any, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        sent.append(request)
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "test",
                "model": "glm-4.6",
                "content": [],
                "choices": [],
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        )

    with TestClient(create_app(config), raise_server_exceptions=False) as client:
        with patch.object(httpx.AsyncClient, "send", capture_send):
            response = client.post(
                path,
                json=payload,
                headers={
                    "Authorization": "Bearer inbound-secret",
                    "x-api-key": "inbound-api-key",
                    "Cookie": "session=inbound-cookie",
                    "Proxy-Authorization": "Basic inbound-proxy-credential",
                    "X-Goog-Api-Key": "inbound-google-key",
                },
            )
    assert response.status_code == 200, response.text
    assert len(sent) == 1
    if routed:
        assert str(sent[0].url) == "https://ollama.test" + expected_path
        assert sent[0].headers["authorization"] == "Bearer route-secret"
        assert "x-api-key" not in sent[0].headers
        assert "cookie" not in sent[0].headers
        assert "proxy-authorization" not in sent[0].headers
        assert "x-goog-api-key" not in sent[0].headers
        assert "x-operator-secret" not in sent[0].headers
    else:
        provider = "anthropic" if path == "/v1/messages" else "openai"
        assert str(sent[0].url) == f"https://api.{provider}.test" + expected_path
        assert sent[0].headers["authorization"] == "Bearer inbound-secret"
        assert sent[0].headers["x-api-key"] == "inbound-api-key"
        assert sent[0].headers["cookie"] == "session=inbound-cookie"
        assert sent[0].headers["proxy-authorization"] == "Basic inbound-proxy-credential"
        assert sent[0].headers["x-goog-api-key"] == "inbound-google-key"
        assert sent[0].headers["x-operator-secret"] == provider + "-secret"


@pytest.mark.parametrize("token_available", [False, True])
def test_responses_oauth_bypasses_env_route_before_token_substitution(
    _routed_env: None, monkeypatch: pytest.MonkeyPatch, token_available: bool
) -> None:
    from fastapi.testclient import TestClient

    from headroom.proxy.server import create_app

    if token_available:
        monkeypatch.setenv("FAILCLOSED_TEST_KEY", "route-secret")
    claims = {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-from-jwt"}}
    encoded = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    bearer = f"Bearer e30.{encoded}."
    sent: list[httpx.Request] = []

    async def capture_send(client: Any, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, request=request, json={"id": "resp_test", "output": []})

    with TestClient(create_app(_config()), raise_server_exceptions=False) as client:
        with patch.object(httpx.AsyncClient, "send", capture_send):
            response = client.post(
                "/v1/responses",
                json={"model": "glm-4.6", "input": "hi"},
                headers={"Authorization": bearer},
            )
    assert response.status_code == 200, response.text
    assert len(sent) == 1
    assert str(sent[0].url) == "https://chatgpt.com/backend-api/codex/responses"
    assert sent[0].headers["authorization"] == bearer
    assert sent[0].headers["chatgpt-account-id"] == "acct-from-jwt"


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/chat/completions", {"model": "glm-4.6", "messages": []}),
        ("/v1/responses", {"model": "glm-4.6", "input": "hi"}),
        ("/v1/messages", {"model": "glm-4.6", "messages": [], "max_tokens": 1}),
    ],
)
def test_env_route_rate_limits_inbound_clients_separately(
    _routed_env: None, monkeypatch: pytest.MonkeyPatch, path: str, payload: dict[str, Any]
) -> None:
    from fastapi.testclient import TestClient

    from headroom.proxy.server import create_app

    monkeypatch.setenv("FAILCLOSED_TEST_KEY", "route-secret")
    config = _config()
    config.rate_limit_enabled = True
    app = create_app(config)
    limiter = app.state.proxy.rate_limiter
    sent: list[httpx.Request] = []

    async def capture_send(client: Any, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        sent.append(request)
        return httpx.Response(
            200, request=request, json={"id": "test", "output": [], "content": [], "choices": []}
        )

    with TestClient(app, client=("127.0.0.1", 55123), raise_server_exceptions=False) as client:
        with (
            patch.object(httpx.AsyncClient, "send", capture_send),
            patch.object(limiter, "check_request", wraps=limiter.check_request) as check_request,
        ):
            for credential in ("inbound-one", "inbound-two"):
                response = client.post(
                    path, json=payload, headers={"Authorization": f"Bearer {credential}"}
                )
                assert response.status_code == 200, response.text
    keys = [call.args[0] for call in check_request.await_args_list]
    assert len(keys) == 2
    assert all(key.startswith("peer:127.0.0.1|cred:") for key in keys)
    assert keys[0] != keys[1]
    assert len(sent) == 2
    assert all(request.headers["authorization"] == "Bearer route-secret" for request in sent)


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("designated", [False, True])
def test_responses_client_override_keeps_existing_operator_host_policy(
    monkeypatch: pytest.MonkeyPatch, routed: bool, designated: bool
) -> None:
    from fastapi.testclient import TestClient

    from headroom.proxy.server import create_app

    if routed:
        monkeypatch.setenv("HEADROOM_UPSTREAM_ROUTES", _ROUTES)
    else:
        monkeypatch.delenv("HEADROOM_UPSTREAM_ROUTES", raising=False)
    # The route token stays absent: a client override must bypass its auth.
    monkeypatch.delenv("FAILCLOSED_TEST_KEY", raising=False)
    monkeypatch.delenv("HEADROOM_UPSTREAM_ALLOWED_HOSTS", raising=False)
    override = "https://api.anthropic.test" if designated else "https://override.test"
    monkeypatch.setenv("HEADROOM_ALLOWED_BASE_URLS", override)
    config = _config()
    config.openai_extra_headers = {"X-Operator-Secret": "openai-secret"}
    sent: list[httpx.Request] = []

    async def capture_send(client: Any, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, request=request, json={"id": "test", "output": []})

    with TestClient(create_app(config), raise_server_exceptions=False) as client:
        with patch.object(httpx.AsyncClient, "send", capture_send):
            response = client.post(
                "/v1/responses",
                json={"model": "glm-4.6", "input": "hi"},
                headers={
                    "Authorization": "Bearer inbound-secret",
                    "X-Headroom-Base-Url": override,
                },
            )
    assert response.status_code == 200, response.text
    assert len(sent) == 1
    assert str(sent[0].url) == override + "/v1/responses"
    assert sent[0].headers["authorization"] == "Bearer inbound-secret"
    assert ("x-operator-secret" in sent[0].headers) is designated
