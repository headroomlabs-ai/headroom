"""Tests for the Tier-2 pilot hardening features:

- 2.1 optional inbound auth token (HEADROOM_PROXY_TOKEN) on the data plane
- 3.1 response security headers
- 2.4 admin/state-mutating audit log
- 2.2 air-gap master switch (HEADROOM_OFFLINE)
"""

from __future__ import annotations

import contextlib
import logging

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from headroom.cache.compression_store import reset_compression_store
from headroom.offline import apply_offline_env, is_offline
from headroom.proxy.audit import is_auditable_path
from headroom.proxy.handlers.openai import OpenAIHandlerMixin
from headroom.proxy.server import (
    ProxyConfig,
    WebSocketAuthMiddleware,
    create_app,
    scrub_proxy_token_headers,
)

NONLOOPBACK = ("203.0.113.5", 44444)  # TEST-NET-3, never loopback
LOOPBACK = ("127.0.0.1", 12345)


def _make_app(**overrides):
    reset_compression_store()
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        **overrides,
    )
    return create_app(config)


def test_create_app_accepts_test_net_host_without_transport_validation() -> None:
    app = _make_app(host="203.0.113.5", proxy_token=None)
    assert app is not None


# ───────────────────────────── 2.1 inbound auth token ─────────────────────


class TestInboundAuthToken:
    def test_no_token_configured_leaves_data_plane_open(self):
        """Default (no token): non-loopback callers are not challenged."""
        app = _make_app()
        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            assert c.get("/livez").status_code == 200

    def test_token_set_rejects_nonloopback_without_credential(self):
        app = _make_app(proxy_token="s3cr3t-token")
        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            resp = c.get("/stats")
            assert resp.status_code == 401

    def test_token_set_accepts_correct_bearer(self):
        app = _make_app(proxy_token="s3cr3t-token")
        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            resp = c.get("/stats", headers={"Authorization": "Bearer s3cr3t-token"})
            assert resp.status_code != 401

    def test_token_set_accepts_custom_header(self):
        app = _make_app(proxy_token="s3cr3t-token")
        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            resp = c.get("/stats", headers={"X-Headroom-Proxy-Token": "s3cr3t-token"})
            assert resp.status_code != 401

    def test_token_set_accepts_custom_header_with_upstream_oauth(self):
        """The upstream OAuth bearer must not override the proxy credential."""
        app = _make_app(proxy_token="s3cr3t-token")
        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            resp = c.get(
                "/stats",
                headers={
                    "Authorization": "Bearer oauth-subscription-token",
                    "X-Headroom-Proxy-Token": "s3cr3t-token",
                },
            )
            assert resp.status_code != 401

    def test_token_set_rejects_wrong_token(self):
        app = _make_app(proxy_token="s3cr3t-token")
        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            resp = c.get("/stats", headers={"Authorization": "Bearer wrong"})
            assert resp.status_code == 401

    def test_loopback_is_exempt_from_token(self):
        """Loopback callers (same trust boundary as admin routes) skip the token."""
        app = _make_app(proxy_token="s3cr3t-token")
        with TestClient(app, base_url="http://127.0.0.1", client=LOOPBACK) as c:
            assert c.get("/stats").status_code != 401

    def test_health_endpoints_exempt_even_nonloopback(self):
        """Orchestrator health probes must work without the token."""
        app = _make_app(proxy_token="s3cr3t-token")
        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            assert c.get("/livez").status_code == 200
            assert c.get("/readyz").status_code in (200, 503)  # ready/not-ready, never 401

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("GET", "/healthz"),  # no Python route: lands on the catch-all
            ("POST", "/health"),
            ("PUT", "/livez"),
            ("DELETE", "/readyz"),
            ("HEAD", "/livez"),  # @app.get does not register HEAD
        ],
    )
    def test_health_exemption_does_not_reach_passthrough(self, monkeypatch, method, path):
        """Only GET probes are exempt from the token.

        Any other request on a health path is not served by the health handler
        but by the catch-all passthrough, which relays it upstream. Exempting
        those turned the proxy into an unauthenticated relay.
        """
        relayed: list[tuple[str, str]] = []

        async def _spy_passthrough(self, request, base_url, *args, **kwargs):
            relayed.append((request.method, request.url.path))
            return JSONResponse({"relayed_to": base_url})

        monkeypatch.setattr(OpenAIHandlerMixin, "handle_passthrough", _spy_passthrough)
        app = _make_app(proxy_token="s3cr3t-token")
        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            resp = c.request(method, path)
        assert resp.status_code == 401
        assert relayed == []


# ──────────────────── 2.1b inbound auth token over WebSocket ──────────────


WS_PATHS = ("/v1/responses", "/v1/live")


class _SpyApp:
    """Downstream ASGI app that records whether it was ever reached."""

    def __init__(self) -> None:
        self.called = False

    async def __call__(self, scope, receive, send) -> None:
        self.called = True


def _ws_scope(*, client=NONLOOPBACK, headers=(), path="/v1/responses"):
    return {
        "type": "websocket",
        "path": path,
        "client": client,
        "headers": [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers],
    }


async def _drive(middleware, scope):
    """Run one connection through the middleware, returning (sent, downstream)."""
    inbox = [{"type": "websocket.connect"}]
    sent: list[dict] = []

    async def receive():
        return inbox.pop(0) if inbox else {"type": "websocket.disconnect"}

    async def send(message):
        sent.append(message)

    await middleware(scope, receive, send)
    return sent


def _closed_with_policy_violation(sent) -> bool:
    return any(m.get("type") == "websocket.close" and m.get("code") == 1008 for m in sent)


class TestWebSocketAuthMiddleware:
    """The middleware itself, driven directly over ASGI.

    Asserted at this layer because a pre-accept close surfaces through
    ``TestClient`` as a bare ``AttributeError`` — indistinguishable from any
    other handshake failure — so an exception-shape assertion would pass for
    the wrong reason.
    """

    async def test_rejects_missing_credential(self):
        downstream = _SpyApp()
        mw = WebSocketAuthMiddleware(downstream, proxy_token="s3cr3t-token")

        sent = await _drive(mw, _ws_scope())

        assert downstream.called is False
        assert _closed_with_policy_violation(sent)

    async def test_rejects_wrong_credential(self):
        downstream = _SpyApp()
        mw = WebSocketAuthMiddleware(downstream, proxy_token="s3cr3t-token")

        sent = await _drive(mw, _ws_scope(headers=[("authorization", "Bearer wrong")]))

        assert downstream.called is False
        assert _closed_with_policy_violation(sent)

    async def test_accepts_correct_bearer(self):
        downstream = _SpyApp()
        mw = WebSocketAuthMiddleware(downstream, proxy_token="s3cr3t-token")

        sent = await _drive(mw, _ws_scope(headers=[("authorization", "Bearer s3cr3t-token")]))

        assert downstream.called is True
        assert not _closed_with_policy_violation(sent)

    async def test_accepts_custom_header(self):
        downstream = _SpyApp()
        mw = WebSocketAuthMiddleware(downstream, proxy_token="s3cr3t-token")

        sent = await _drive(mw, _ws_scope(headers=[("x-headroom-proxy-token", "s3cr3t-token")]))

        assert downstream.called is True
        assert not _closed_with_policy_violation(sent)

    async def test_accepts_custom_header_with_upstream_oauth(self):
        """The upstream OAuth bearer must not override the proxy credential."""
        downstream = _SpyApp()
        mw = WebSocketAuthMiddleware(downstream, proxy_token="s3cr3t-token")

        sent = await _drive(
            mw,
            _ws_scope(
                headers=[
                    ("authorization", "Bearer oauth-subscription-token"),
                    ("x-headroom-proxy-token", "s3cr3t-token"),
                ]
            ),
        )

        assert downstream.called is True
        assert not _closed_with_policy_violation(sent)

    async def test_loopback_is_exempt(self):
        """Same trust boundary the HTTP gate already grants loopback."""
        downstream = _SpyApp()
        mw = WebSocketAuthMiddleware(downstream, proxy_token="s3cr3t-token")

        sent = await _drive(mw, _ws_scope(client=LOOPBACK))

        assert downstream.called is True
        assert not _closed_with_policy_violation(sent)

    async def test_unknown_client_is_treated_as_loopback(self):
        """Mirrors is_loopback_host(None) -> True, as the HTTP gate does."""
        downstream = _SpyApp()
        mw = WebSocketAuthMiddleware(downstream, proxy_token="s3cr3t-token")

        sent = await _drive(mw, _ws_scope(client=None))

        assert downstream.called is True
        assert not _closed_with_policy_violation(sent)

    async def test_repeated_header_resolves_like_the_http_gate(self):
        """A duplicated Authorization must mean the same thing on both transports.

        Starlette's Headers (what the HTTP gate reads) returns the FIRST
        occurrence. A hand-built dict returns the last, which would let the two
        paths disagree about which credential counted.
        """
        downstream = _SpyApp()
        mw = WebSocketAuthMiddleware(downstream, proxy_token="s3cr3t-token")

        sent = await _drive(
            mw,
            _ws_scope(
                headers=[
                    ("authorization", "Bearer s3cr3t-token"),
                    ("authorization", "Bearer wrong"),
                ]
            ),
        )

        # First header wins → authenticated, same as the HTTP gate.
        assert downstream.called is True
        assert not _closed_with_policy_violation(sent)

    async def test_no_token_configured_is_a_passthrough(self):
        """Default deployment must gain no new challenge."""
        downstream = _SpyApp()
        mw = WebSocketAuthMiddleware(downstream, proxy_token=None)

        sent = await _drive(mw, _ws_scope())

        assert downstream.called is True
        assert not _closed_with_policy_violation(sent)

    async def test_http_scope_is_left_to_the_http_gate(self):
        downstream = _SpyApp()
        mw = WebSocketAuthMiddleware(downstream, proxy_token="s3cr3t-token")

        sent = await _drive(mw, {**_ws_scope(), "type": "http"})

        assert downstream.called is True
        assert not _closed_with_policy_violation(sent)


class TestWebSocketRoutesAreGatedInTheApp:
    """The middleware is actually wired into ``create_app``.

    Asserts the security property directly — the route handler must never run
    for an unauthenticated handshake — rather than inspecting the exception the
    client happens to see.
    """

    @pytest.mark.parametrize("path", WS_PATHS)
    def test_unauthenticated_handshake_never_reaches_the_handler(self, path, monkeypatch):
        app = _make_app(proxy_token="s3cr3t-token")
        reached = _record_ws_handler_reached(app, monkeypatch)

        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            try:
                with c.websocket_connect(path):
                    pass
            except Exception:  # noqa: BLE001 - the refusal shape is asserted above
                pass

        assert reached() is False

    @pytest.mark.parametrize("path", WS_PATHS)
    def test_authenticated_handshake_reaches_the_handler(self, path, monkeypatch):
        app = _make_app(proxy_token="s3cr3t-token")
        reached = _record_ws_handler_reached(app, monkeypatch)

        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            try:
                with c.websocket_connect(path, headers={"X-Headroom-Proxy-Token": "s3cr3t-token"}):
                    pass
            except Exception:  # noqa: BLE001 - route may fail with no upstream
                pass

        assert reached() is True


def _record_ws_handler_reached(app, monkeypatch):
    """Spy both WebSocket route families; returns a callable reporting arrival."""
    from headroom.providers import proxy_routes

    seen: list[str] = []

    # Each spy must terminate the handshake itself: a handler that returns
    # without accepting or closing leaves the client waiting forever.
    async def _responses_spy(websocket):
        seen.append("responses")
        await websocket.close(code=1000)

    async def _live_spy(websocket, *args, **kwargs):
        seen.append("live")
        await websocket.close(code=1000)

    monkeypatch.setattr(app.state.proxy, "handle_openai_responses_ws", _responses_spy)
    monkeypatch.setattr(proxy_routes, "handle_codex_live_websocket", _live_spy)
    return lambda: bool(seen)


# ─────────────── 2.1c the proxy token never reaches an upstream ───────────


TOKEN = "s3cr3t-token"


class _CapturingTransport(httpx.AsyncBaseTransport):
    """Upstream stand-in that records every request the proxy sends it."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if "chat/completions" in str(request.url):
            body = {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        elif "/v1/messages" in str(request.url):
            body = {
                "id": "msg_mock",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        else:
            body = {"object": "list", "data": []}
        return httpx.Response(200, headers={"content-type": "application/json"}, json=body)


@contextlib.contextmanager
def _upstream_capturing_client(*, client=NONLOOPBACK, base_url="http://testserver"):
    reset_compression_store()
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        anthropic_api_url="https://api.anthropic.test",
        openai_api_url="https://api.openai.test",
        proxy_token=TOKEN,
    )
    app = create_app(config)
    transport = _CapturingTransport()
    with TestClient(app, base_url=base_url, client=client) as c:
        # Startup builds the real upstream client, so swap it only after
        # entering the lifespan or requests would leave the test process.
        app.state.proxy.http_client = httpx.AsyncClient(transport=transport)
        yield c, transport


def _anthropic_request(c, headers):
    return c.post(
        "/v1/messages",
        headers={"anthropic-version": "2023-06-01", **headers},
        json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 16,
        },
    )


def _only_upstream_request(transport) -> httpx.Request:
    assert len(transport.requests) == 1, transport.requests
    return transport.requests[0]


class TestProxyTokenIsNotForwardedUpstream:
    """The proxy credential authenticates to Headroom only.

    Handlers build upstream headers from the inbound request and strip only
    ``x-headroom-*``, so a token sent as ``Authorization: Bearer`` used to be
    relayed to the provider verbatim. These tests capture what actually leaves
    the proxy.
    """

    def test_bearer_token_is_dropped_and_provider_key_kept(self):
        with _upstream_capturing_client() as (c, transport):
            resp = _anthropic_request(
                c, {"x-api-key": "sk-ant-test", "Authorization": f"Bearer {TOKEN}"}
            )
        assert resp.status_code == 200, resp.text

        upstream = _only_upstream_request(transport)
        assert "authorization" not in upstream.headers
        assert upstream.headers["x-api-key"] == "sk-ant-test"
        assert TOKEN not in str(upstream.headers.raw)

    def test_lowercase_scheme_is_dropped_too(self):
        """The gate accepts ``bearer`` in any case, so the scrub must as well."""
        with _upstream_capturing_client() as (c, transport):
            resp = _anthropic_request(
                c, {"x-api-key": "sk-ant-test", "Authorization": f"bearer {TOKEN}"}
            )
        assert resp.status_code == 200, resp.text
        assert "authorization" not in _only_upstream_request(transport).headers

    def test_custom_header_is_dropped_and_provider_bearer_kept(self):
        with _upstream_capturing_client() as (c, transport):
            resp = c.post(
                "/v1/chat/completions",
                headers={"X-Headroom-Proxy-Token": TOKEN, "Authorization": "Bearer sk-test"},
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
            )
        assert resp.status_code == 200, resp.text

        upstream = _only_upstream_request(transport)
        assert upstream.headers["authorization"] == "Bearer sk-test"
        assert "x-headroom-proxy-token" not in upstream.headers

    def test_custom_header_is_dropped_even_with_internal_strip_disabled(self, monkeypatch):
        """``HEADROOM_STRIP_INTERNAL_HEADERS=disabled`` must not leak the credential."""
        monkeypatch.setenv("HEADROOM_STRIP_INTERNAL_HEADERS", "disabled")
        with _upstream_capturing_client() as (c, transport):
            resp = _anthropic_request(
                c, {"x-api-key": "sk-ant-test", "X-Headroom-Proxy-Token": TOKEN}
            )
        assert resp.status_code == 200, resp.text
        assert "x-headroom-proxy-token" not in _only_upstream_request(transport).headers

    def test_loopback_caller_token_is_dropped(self):
        """Loopback skips the check, but its token is still not the provider's."""
        with _upstream_capturing_client(client=LOOPBACK, base_url="http://127.0.0.1") as (
            c,
            transport,
        ):
            resp = _anthropic_request(
                c, {"x-api-key": "sk-ant-test", "Authorization": f"Bearer {TOKEN}"}
            )
        assert resp.status_code == 200, resp.text
        assert "authorization" not in _only_upstream_request(transport).headers

    def test_catch_all_passthrough_drops_bearer_token(self):
        with _upstream_capturing_client() as (c, transport):
            resp = c.get(
                "/v1/models",
                headers={"x-api-key": "sk-ant-test", "Authorization": f"Bearer {TOKEN}"},
            )
        assert resp.status_code == 200, resp.text
        upstream = _only_upstream_request(transport)
        assert "authorization" not in upstream.headers
        assert upstream.headers["x-api-key"] == "sk-ant-test"


class TestScrubProxyTokenHeaders:
    """The scope-level scrub both gates share."""

    @staticmethod
    def _scope(*headers):
        return {"headers": [(k.encode("latin-1"), v.encode("latin-1")) for k, v in headers]}

    def test_drops_matching_bearer_keeps_other_headers(self):
        scope = self._scope(("authorization", f"Bearer {TOKEN}"), ("x-api-key", "sk-ant"))
        scrub_proxy_token_headers(scope, TOKEN.encode())
        assert scope["headers"] == [(b"x-api-key", b"sk-ant")]

    def test_keeps_non_matching_bearer(self):
        scope = self._scope(("authorization", "Bearer sk-provider"))
        scrub_proxy_token_headers(scope, TOKEN.encode())
        assert scope["headers"] == [(b"authorization", b"Bearer sk-provider")]

    def test_keeps_non_bearer_authorization(self):
        scope = self._scope(("authorization", f"Basic {TOKEN}"))
        scrub_proxy_token_headers(scope, TOKEN.encode())
        assert scope["headers"] == [(b"authorization", f"Basic {TOKEN}".encode())]

    def test_duplicate_authorization_drops_only_the_token(self):
        scope = self._scope(
            ("authorization", f"Bearer {TOKEN}"),
            ("authorization", "Bearer sk-provider"),
        )
        scrub_proxy_token_headers(scope, TOKEN.encode())
        assert scope["headers"] == [(b"authorization", b"Bearer sk-provider")]

    def test_custom_header_dropped_even_without_configured_token(self):
        scope = self._scope(("x-headroom-proxy-token", "anything"), ("x-api-key", "sk-ant"))
        scrub_proxy_token_headers(scope, b"")
        assert scope["headers"] == [(b"x-api-key", b"sk-ant")]

    def test_no_configured_token_leaves_authorization_alone(self):
        scope = self._scope(("authorization", f"Bearer {TOKEN}"))
        scrub_proxy_token_headers(scope, b"")
        assert scope["headers"] == [(b"authorization", f"Bearer {TOKEN}".encode())]

    def test_non_ascii_header_value_does_not_raise(self):
        scope = {"headers": [(b"authorization", "Bearer café".encode("latin-1"))]}
        scrub_proxy_token_headers(scope, TOKEN.encode())
        assert len(scope["headers"]) == 1


class _HeaderSpyApp:
    """Downstream ASGI app that records the headers it was handed."""

    def __init__(self) -> None:
        self.headers: list[tuple[bytes, bytes]] | None = None

    async def __call__(self, scope, receive, send) -> None:
        self.headers = list(scope["headers"])


class TestWebSocketScrubsProxyToken:
    @pytest.mark.parametrize("client", [NONLOOPBACK, LOOPBACK])
    async def test_bearer_token_removed_before_the_app(self, client):
        downstream = _HeaderSpyApp()
        mw = WebSocketAuthMiddleware(downstream, proxy_token=TOKEN)

        await _drive(
            mw,
            _ws_scope(
                client=client,
                headers=[("authorization", f"Bearer {TOKEN}"), ("x-api-key", "sk-ant")],
            ),
        )

        assert downstream.headers == [(b"x-api-key", b"sk-ant")]

    async def test_custom_header_removed_provider_bearer_kept(self):
        downstream = _HeaderSpyApp()
        mw = WebSocketAuthMiddleware(downstream, proxy_token=TOKEN)

        await _drive(
            mw,
            _ws_scope(
                headers=[
                    ("authorization", "Bearer sk-provider"),
                    ("x-headroom-proxy-token", TOKEN),
                ]
            ),
        )

        assert downstream.headers == [(b"authorization", b"Bearer sk-provider")]

    def test_responses_route_handler_never_sees_the_token(self, monkeypatch):
        app = _make_app(proxy_token=TOKEN)
        seen: list[dict[str, str]] = []

        async def _responses_spy(websocket):
            seen.append(dict(websocket.headers))
            await websocket.close(code=1000)

        monkeypatch.setattr(app.state.proxy, "handle_openai_responses_ws", _responses_spy)

        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            try:
                with c.websocket_connect(
                    "/v1/responses", headers={"Authorization": f"Bearer {TOKEN}"}
                ):
                    pass
            except Exception:  # noqa: BLE001 - spy closes the socket immediately
                pass

        assert len(seen) == 1
        assert "authorization" not in seen[0]


# ───────────────────────────── 3.1 security headers ───────────────────────


class TestSecurityHeaders:
    def test_headers_present_on_responses(self):
        app = _make_app()
        with TestClient(app, base_url="http://127.0.0.1", client=LOOPBACK) as c:
            h = c.get("/livez").headers
            assert h.get("X-Content-Type-Options") == "nosniff"
            assert h.get("X-Frame-Options") == "DENY"
            assert h.get("Referrer-Policy") == "no-referrer"
            assert "max-age=" in h.get("Strict-Transport-Security", "")

    def test_headers_present_on_401(self):
        app = _make_app(proxy_token="s3cr3t-token")
        with TestClient(app, base_url="http://testserver", client=NONLOOPBACK) as c:
            resp = c.get("/stats")
            assert resp.status_code == 401
            assert resp.headers.get("X-Content-Type-Options") == "nosniff"


# ───────────────────────────── 2.4 admin audit log ────────────────────────


class TestAdminAuditLog:
    def test_auditable_path_classification(self):
        assert is_auditable_path("/admin/runtime-env")
        assert is_auditable_path("/cache/clear")
        assert is_auditable_path("/stats/reset")
        assert not is_auditable_path("/v1/messages")
        assert not is_auditable_path("/livez")

    def test_cache_clear_emits_audit_event(self):
        # Capture the dedicated audit logger directly (the proxy's logging setup
        # configures propagation, so attach to the logger rather than rely on
        # caplog's root handler).
        messages: list[str] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                messages.append(record.getMessage())

        handler = _Capture()
        audit_logger = logging.getLogger("headroom.audit")
        audit_logger.setLevel(logging.INFO)
        audit_logger.addHandler(handler)
        try:
            app = _make_app()
            with TestClient(app, base_url="http://127.0.0.1", client=LOOPBACK) as c:
                assert c.post("/cache/clear").status_code == 200
        finally:
            audit_logger.removeHandler(handler)

        assert messages, "expected an audit record for /cache/clear"
        assert any("/cache/clear" in m for m in messages)
        assert any("headroom_admin_audit" in m for m in messages)
        assert any('"source_ip": "127.0.0.1"' in m for m in messages)


# ───────────────────────────── 2.2 air-gap switch ─────────────────────────


class TestOfflineSwitch:
    def test_is_offline_reads_env(self, monkeypatch):
        monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
        assert is_offline() is False
        monkeypatch.setenv("HEADROOM_OFFLINE", "1")
        assert is_offline() is True
        monkeypatch.setenv("HEADROOM_OFFLINE", "off")
        assert is_offline() is False

    def test_offline_disables_telemetry(self, monkeypatch):
        from headroom.telemetry.beacon import is_telemetry_enabled

        monkeypatch.setenv("HEADROOM_TELEMETRY", "on")
        monkeypatch.setenv("HEADROOM_OFFLINE", "1")
        assert is_telemetry_enabled() is False  # offline overrides the opt-in

    def test_offline_disables_update_check(self, monkeypatch):
        from headroom.update_check import is_update_check_enabled

        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
        monkeypatch.setenv("HEADROOM_OFFLINE", "1")
        assert is_update_check_enabled() is False

    def test_apply_offline_env_sets_hf_offline(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        monkeypatch.setenv("HEADROOM_OFFLINE", "1")
        apply_offline_env()
        import os

        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
