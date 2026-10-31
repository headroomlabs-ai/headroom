"""Guard availability tracking, /metrics gauge and /health session_store (#4512).

Covers:
* ``Guard.state`` reports "unknown" before any command runs, "available" on
  success, and "unavailable" on backoff/failure.
* ``headroom_redis_checked{feature=...}`` gauge appears in /metrics output for
  every configured guard (1 if checked, 0 if unknown).
* ``headroom_redis_available{feature=...}`` gauge appears in /metrics output
  only for observed guards (1 if available, 0 during outage).
* ``/health`` reports ``session_store`` feature state strings ("unknown" |
  "available" | "unavailable") and stays degraded when unobserved or unavailable
  while top-level ``ready`` stays True.
* Unregistering/closing guards on store close/detach.
"""

from __future__ import annotations

import pytest

from headroom.cache import redis_client  # noqa: E402
from headroom.cache.redis_client import Guard, guards  # noqa: E402

# --- Guard.state & Guard.available --------------------------------------------


class TestGuardAvailable:
    def test_unknown_initially_before_any_call(self):
        g = Guard("test feature", "test")
        assert g.state == "unknown"
        assert g.available is False
        g.close()

    def test_available_after_successful_call(self):
        g = Guard("test feature", "test")
        result = g.call("read", lambda: "ok", "default")
        assert result == "ok"
        assert g.state == "available"
        assert g.available is True
        g.close()

    def test_unavailable_after_connection_error(self, monkeypatch):
        monkeypatch.setattr(redis_client, "BACKOFF_SECONDS", 60.0)
        g = Guard("test feature", "test")
        import redis

        def failing():
            raise redis.ConnectionError("connection refused")

        result = g.call("read", failing, "default")
        assert result == "default"
        assert g.state == "unavailable"
        assert g.available is False
        g.close()

    def test_available_again_after_successful_call(self, monkeypatch):
        monkeypatch.setattr(redis_client, "BACKOFF_SECONDS", 0.0)
        g = Guard("test feature", "test")
        import redis

        def failing():
            raise redis.RedisError("some error")

        g.call("read", failing, "default")
        assert g.state == "unavailable"
        assert g.available is False

        result = g.call("read", lambda: "ok", "default")
        assert result == "ok"
        assert g.state == "available"
        assert g.available is True
        g.close()

    def test_untested_shared_session_store_is_unknown(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_COMPRESS_SESSION_STORE", "redis://127.0.0.1:1")
        from headroom.cache.shared_sessions import open_shared_sessions

        store = open_shared_sessions()
        assert store is not None
        try:
            assert store._guard.state == "unknown"
            assert store._guard.available is False
            assert guards()["compress_session"].state == "unknown"
            assert guards()["compress_session"].available is False
        finally:
            store.close()

    def test_lifecycle_close_unregisters_guard(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_COMPRESS_SESSION_STORE", "redis://127.0.0.1:1")
        from headroom.cache.shared_sessions import open_shared_sessions

        store = open_shared_sessions()
        assert store is not None
        assert "compress_session" in guards()

        store.close()
        monkeypatch.delenv("HEADROOM_COMPRESS_SESSION_STORE", raising=False)
        assert open_shared_sessions() is None
        assert "compress_session" not in guards()


# --- /metrics gauge ----------------------------------------------------------


class TestMetricsGauge:
    @pytest.mark.asyncio
    async def test_gauge_checked_and_available_when_observed(self):
        g = Guard("test feature", "test_label")
        g.call("read", lambda: "ok", None)
        from headroom.proxy.prometheus_metrics import PrometheusMetrics

        metrics = PrometheusMetrics(stateless=True)
        text = await metrics.export()
        assert "# HELP headroom_redis_checked" in text
        assert "# TYPE headroom_redis_checked gauge" in text
        assert 'headroom_redis_checked{feature="test_label"} 1' in text
        assert "# HELP headroom_redis_available" in text
        assert "# TYPE headroom_redis_available gauge" in text
        assert 'headroom_redis_available{feature="test_label"} 1' in text
        g.close()

    @pytest.mark.asyncio
    async def test_gauge_unobserved_shows_checked_0_and_no_available(self):
        g = Guard("test feature", "unobserved_label")
        from headroom.proxy.prometheus_metrics import PrometheusMetrics

        metrics = PrometheusMetrics(stateless=True)
        text = await metrics.export()
        assert 'headroom_redis_checked{feature="unobserved_label"} 0' in text
        assert "headroom_redis_available" not in text
        g.close()

    @pytest.mark.asyncio
    async def test_gauge_zero_during_outage(self, monkeypatch):
        monkeypatch.setattr(redis_client, "BACKOFF_SECONDS", 60.0)
        import redis

        g = Guard("test feature", "outage_label")

        def failing():
            raise redis.ConnectionError("down")

        g.call("read", failing, None)

        from headroom.proxy.prometheus_metrics import PrometheusMetrics

        metrics = PrometheusMetrics(stateless=True)
        text = await metrics.export()
        assert 'headroom_redis_checked{feature="outage_label"} 1' in text
        assert 'headroom_redis_available{feature="outage_label"} 0' in text
        g.close()


# --- /health session_store ---------------------------------------------------


class TestHealthSessionStore:
    def test_session_store_degraded_does_not_flip_ready(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
        monkeypatch.setattr(redis_client, "BACKOFF_SECONDS", 60.0)
        import redis
        from fastapi.testclient import TestClient

        from headroom.proxy.models import ProxyConfig
        from headroom.proxy.server import create_app

        g = Guard("compress sessions", "compress_session")

        def failing():
            raise redis.ConnectionError("down")

        g.call("read", failing, None)
        assert g.state == "unavailable"

        app = create_app(
            ProxyConfig(
                optimize=False,
                cache_enabled=False,
                rate_limit_enabled=False,
            )
        )
        app.state.ready = True
        proxy = app.state.proxy
        proxy.http_client = object()

        client = TestClient(app)
        response = client.get("/readyz")
        assert response.status_code == 200
        payload = response.json()

        assert payload["ready"] is True
        assert payload["status"] == "healthy"

        ss = payload["checks"]["session_store"]
        assert ss["enabled"] is True
        assert ss["ready"] is False
        assert ss["status"] == "degraded"
        assert ss["optional"] is True
        assert ss["features"]["compress_session"] == "unavailable"
        g.close()

    def test_session_store_disabled_when_no_guards(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")

        from fastapi.testclient import TestClient

        from headroom.proxy.models import ProxyConfig
        from headroom.proxy.server import create_app

        app = create_app(
            ProxyConfig(
                optimize=False,
                cache_enabled=False,
                rate_limit_enabled=False,
            )
        )
        app.state.ready = True
        proxy = app.state.proxy
        proxy.http_client = object()

        client = TestClient(app)
        response = client.get("/readyz")
        assert response.status_code == 200
        payload = response.json()

        ss = payload["checks"]["session_store"]
        assert ss["enabled"] is False
        assert ss["status"] == "disabled"

    def test_session_store_healthy_when_available(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")

        from fastapi.testclient import TestClient

        from headroom.proxy.models import ProxyConfig
        from headroom.proxy.server import create_app

        g = Guard("CCR store", "ccr")
        g.call("read", lambda: "ok", None)
        assert g.state == "available"

        app = create_app(
            ProxyConfig(
                optimize=False,
                cache_enabled=False,
                rate_limit_enabled=False,
            )
        )
        app.state.ready = True
        proxy = app.state.proxy
        proxy.http_client = object()

        client = TestClient(app)
        response = client.get("/readyz")
        assert response.status_code == 200
        payload = response.json()

        ss = payload["checks"]["session_store"]
        assert ss["enabled"] is True
        assert ss["ready"] is True
        assert ss["status"] == "healthy"
        assert ss["features"]["ccr"] == "available"
        g.close()


# --- CCR store lifecycle across proxy shutdown -------------------------------


class TestCCRStoreSurvivesProxyShutdown:
    @pytest.mark.asyncio
    async def test_ccr_store_persists_across_proxy_shutdown(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
        from headroom.cache.compression_store import (
            get_compression_store,
            reset_compression_store,
        )
        from headroom.proxy.models import ProxyConfig
        from headroom.proxy.server import create_app

        reset_compression_store()
        try:
            app1 = create_app(
                ProxyConfig(
                    optimize=False,
                    cache_enabled=False,
                    rate_limit_enabled=False,
                )
            )
            app1.state.ready = True
            proxy1 = app1.state.proxy
            proxy1.http_client = None
            proxy1.http_client_h1 = None

            store = get_compression_store()
            original_hash = store.store(
                original='{"data": "hello"}',
                compressed='{"data": "[COMPRESSED:abc123]"}',
                original_tokens=10,
                compressed_tokens=2,
                tool_name="test_tool",
            )
            entry_before = store.retrieve(original_hash)
            assert entry_before is not None
            assert entry_before.original_content == '{"data": "hello"}'

            await proxy1.shutdown()

            store_after = get_compression_store()
            assert store_after is store
            entry_after = store_after.retrieve(original_hash)
            assert entry_after is not None
            assert entry_after.original_content == '{"data": "hello"}'
        finally:
            reset_compression_store()
