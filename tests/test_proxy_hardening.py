"""Tests for the Tier-2 pilot hardening features:

- 2.1 optional inbound auth token (HEADROOM_PROXY_TOKEN) on the data plane
- 3.1 response security headers
- 2.4 admin/state-mutating audit log
- 2.2 air-gap master switch (HEADROOM_OFFLINE)
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.cache.compression_store import reset_compression_store
from headroom.offline import apply_offline_env, is_offline
from headroom.proxy.audit import is_auditable_path
from headroom.proxy.server import ProxyConfig, WebSocketAuthMiddleware, create_app

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
