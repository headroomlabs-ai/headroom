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
from headroom.proxy.server import ProxyConfig, create_app

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
