"""Tests for proxy scalability features.

These tests verify connection pooling, HTTP/2, and worker configuration.
"""

import asyncio
import json
import os
from typing import Any
from unittest.mock import patch

import httpx
import pytest


class TestConnectionPoolConfig:
    """Test connection pool configuration."""

    def test_httpx_limits_basic(self):
        """Test that httpx accepts our connection limits."""
        limits = httpx.Limits(
            max_connections=500,
            max_keepalive_connections=100,
        )
        assert limits.max_connections == 500
        assert limits.max_keepalive_connections == 100

    def test_httpx_limits_custom(self):
        """Test custom connection limits."""
        limits = httpx.Limits(
            max_connections=1000,
            max_keepalive_connections=200,
        )
        assert limits.max_connections == 1000
        assert limits.max_keepalive_connections == 200

    def test_httpx_timeout_config(self):
        """Test timeout configuration for proxy."""
        timeout = httpx.Timeout(
            connect=10.0,
            read=300.0,
            write=300.0,
            pool=10.0,
        )
        assert timeout.connect == 10.0
        assert timeout.read == 300.0
        assert timeout.write == 300.0
        assert timeout.pool == 10.0

    def test_async_client_with_limits(self):
        """Test AsyncClient accepts connection pool limits."""

        async def _run():
            limits = httpx.Limits(
                max_connections=500,
                max_keepalive_connections=100,
            )
            async with httpx.AsyncClient(
                limits=limits,
                timeout=httpx.Timeout(10.0),
            ) as client:
                assert client is not None
                assert limits.max_connections == 500
                assert limits.max_keepalive_connections == 100

        asyncio.run(_run())


class TestHTTP2Config:
    """Test HTTP/2 configuration."""

    def test_http2_requires_h2_package(self):
        """Test that http2=True requires h2 package."""
        import importlib.util

        h2_available = importlib.util.find_spec("h2") is not None

        if h2_available:
            client = httpx.Client(http2=True)
            assert client._base_url is not None
            client.close()
        else:
            with pytest.raises(ImportError):
                httpx.Client(http2=True)

    def test_async_client_http2(self):
        """Test AsyncClient with HTTP/2 enabled."""
        import importlib.util

        if not importlib.util.find_spec("h2"):
            pytest.skip("h2 package not installed")

        async def _run():
            async with httpx.AsyncClient(
                http2=True,
                limits=httpx.Limits(max_connections=100),
            ) as client:
                assert client is not None

        asyncio.run(_run())


class TestProxyConfigDataclass:
    """Test ProxyConfig dataclass with new fields."""

    def test_proxy_config_defaults(self):
        """Test default values for scalability settings."""
        from dataclasses import dataclass

        @dataclass
        class ProxyConfigTest:
            """Minimal proxy config for testing."""

            host: str = "127.0.0.1"
            port: int = 8787
            request_timeout_seconds: int = 300
            connect_timeout_seconds: int = 10
            max_connections: int = 500
            max_keepalive_connections: int = 100
            http2: bool = True

        config = ProxyConfigTest()
        assert config.max_connections == 500
        assert config.max_keepalive_connections == 100
        assert config.http2 is True

    def test_proxy_config_custom_values(self):
        """Test custom values for scalability settings."""
        from dataclasses import dataclass

        @dataclass
        class ProxyConfigTest:
            max_connections: int = 500
            max_keepalive_connections: int = 100
            http2: bool = True

        config = ProxyConfigTest(
            max_connections=1000,
            max_keepalive_connections=200,
            http2=False,
        )
        assert config.max_connections == 1000
        assert config.max_keepalive_connections == 200
        assert config.http2 is False


class TestConcurrencyPatterns:
    """Test async concurrency patterns used in proxy."""

    def test_semaphore_for_backpressure(self):
        """Test semaphore pattern for limiting concurrent requests."""

        async def _run():
            semaphore = asyncio.Semaphore(3)
            active = []
            completed = []

            async def task(task_id: int):
                async with semaphore:
                    active.append(task_id)
                    assert len(active) <= 3
                    await asyncio.sleep(0.01)
                    active.remove(task_id)
                    completed.append(task_id)

            tasks = [task(i) for i in range(10)]
            await asyncio.gather(*tasks)
            assert len(completed) == 10

        asyncio.run(_run())

    def test_connection_reuse_pattern(self):
        """Test that single client instance is reused (not recreated)."""

        async def _run():
            clients_created = []

            class MockProxyWithClient:
                def __init__(self):
                    self.http_client = None

                async def startup(self):
                    self.http_client = httpx.AsyncClient(
                        limits=httpx.Limits(max_connections=100),
                    )
                    clients_created.append(self.http_client)

                async def shutdown(self):
                    if self.http_client:
                        await self.http_client.aclose()

                async def make_request(self, url: str):
                    return self.http_client

            proxy = MockProxyWithClient()
            await proxy.startup()

            client1 = await proxy.make_request("http://example1.com")
            client2 = await proxy.make_request("http://example2.com")
            client3 = await proxy.make_request("http://example3.com")

            assert client1 is client2 is client3
            assert len(clients_created) == 1

            await proxy.shutdown()

        asyncio.run(_run())


class TestTimeoutOverrides:
    """Test per-request timeout overrides."""

    def test_request_level_timeout_override(self):
        """Test that timeout can be overridden per-request."""

        async def _run():
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(10.0),
            ):
                override_timeout = httpx.Timeout(120.0)
                assert override_timeout.read == 120.0
                assert override_timeout.connect == 120.0

        asyncio.run(_run())


class TestWorkerConfiguration:
    """Test worker process configuration."""

    def test_uvicorn_workers_parameter(self):
        """Test that uvicorn accepts workers parameter."""
        uvicorn = pytest.importorskip("uvicorn")

        config = uvicorn.Config(
            app="app:app",
            workers=4,
            limit_concurrency=1000,
        )
        assert config.workers == 4
        assert config.limit_concurrency == 1000

    def test_single_worker_default(self):
        """Test that default is single worker (None)."""
        uvicorn = pytest.importorskip("uvicorn")

        config = uvicorn.Config(app="app:app")
        assert config.workers is None or config.workers == 1

    def test_run_server_uses_import_string_for_multiple_workers(self, monkeypatch):
        from headroom.proxy.models import ProxyConfig
        from headroom.proxy.server import _MULTI_WORKER_CONFIG_ENV, run_server

        captured = {}
        config = ProxyConfig(
            host="0.0.0.0",
            port=8787,
            max_connections=200,
            http_proxy="http://proxy.local:8080",
        )

        def fake_run(app, **kwargs):
            captured["app"] = app
            captured["kwargs"] = kwargs

        monkeypatch.delenv(_MULTI_WORKER_CONFIG_ENV, raising=False)

        try:
            with patch("headroom.proxy.server.uvicorn.run", fake_run):
                run_server(config, workers=4, limit_concurrency=250)

            assert captured["app"] == "headroom.proxy.server:create_app_from_env"
            assert captured["kwargs"]["workers"] == 4
            assert captured["kwargs"]["limit_concurrency"] == 250
            assert captured["kwargs"]["factory"] is True
            payload = json.loads(os.environ[_MULTI_WORKER_CONFIG_ENV])
            assert payload["host"] == "0.0.0.0"
            assert payload["port"] == 8787
            assert payload["max_connections"] == 200
            assert payload["http_proxy"] == "http://proxy.local:8080"
        finally:
            # run_server sets this via raw os.environ. Pop it directly rather
            # than via monkeypatch.delenv: delenv records the current (JSON)
            # value and re-restores it on teardown, leaking the config into
            # later tests (e.g. _proxy_config_from_env then ignores HEADROOM_*).
            os.environ.pop(_MULTI_WORKER_CONFIG_ENV, None)

    def test_run_server_uses_selector_loop_on_windows(self, monkeypatch):
        import builtins

        import uvicorn
        from uvicorn.config import Config

        from headroom.proxy import server as server_mod
        from headroom.proxy.models import ProxyConfig

        captured = {}
        policy_calls: list[Any] = []
        real_hasattr = builtins.hasattr

        class _FakeSelectorPolicy:
            pass

        def fake_run(app, **kwargs):
            captured["app"] = app
            captured["kwargs"] = kwargs

        def fake_set_policy(policy):
            policy_calls.append(policy)

        def fake_hasattr(obj, name):
            if obj is Config and name == "get_loop_factory":
                return fake_hasattr.use_new_api
            return real_hasattr(obj, name)

        fake_hasattr.use_new_api = True

        monkeypatch.setattr(builtins, "hasattr", fake_hasattr)
        monkeypatch.setattr(server_mod.sys, "platform", "win32")
        monkeypatch.setattr(server_mod, "create_app", lambda config: "app")
        monkeypatch.setattr(server_mod.asyncio, "set_event_loop_policy", fake_set_policy)
        monkeypatch.setattr(
            server_mod.asyncio,
            "WindowsSelectorEventLoopPolicy",
            _FakeSelectorPolicy,
            raising=False,
        )

        with patch("headroom.proxy.server.uvicorn.run", fake_run):
            server_mod.run_server(ProxyConfig(), print_banner=False)

        assert captured["app"] == "app"
        assert captured["kwargs"]["loop"] == "asyncio:SelectorEventLoop"
        assert policy_calls == []

        fake_hasattr.use_new_api = False
        captured.clear()
        policy_calls.clear()

        with patch("headroom.proxy.server.uvicorn.run", fake_run):
            server_mod.run_server(ProxyConfig(), print_banner=False)

        assert "loop" not in captured["kwargs"]
        assert len(policy_calls) == 1
        assert isinstance(policy_calls[0], _FakeSelectorPolicy)
        _ = uvicorn  # keep import for parity with runtime module path

    def test_run_server_keeps_default_loop_off_windows(self, monkeypatch):
        from headroom.proxy import server as server_mod
        from headroom.proxy.models import ProxyConfig

        captured = {}

        def fake_run(app, **kwargs):
            captured["kwargs"] = kwargs

        monkeypatch.setattr(server_mod.sys, "platform", "linux")
        monkeypatch.setattr(server_mod, "create_app", lambda config: "app")

        with patch("headroom.proxy.server.uvicorn.run", fake_run):
            server_mod.run_server(ProxyConfig(), print_banner=False)

        assert "loop" not in captured["kwargs"]


class TestProviderHttpClientOptions:
    """Provider HTTPX options should keep proxy settings scoped to provider clients."""

    def test_default_http2_preserved_without_proxy(self):
        from headroom.proxy.models import ProxyConfig
        from headroom.proxy.server import _provider_httpx_client_options

        http2, kwargs = _provider_httpx_client_options(ProxyConfig(http2=True), verify=True)

        assert http2 is True
        assert "proxy" not in kwargs

    def test_http_proxy_sets_proxy_and_forces_http1(self):
        from headroom.proxy.models import ProxyConfig
        from headroom.proxy.server import _provider_httpx_client_options

        http2, kwargs = _provider_httpx_client_options(
            ProxyConfig(http2=True, http_proxy="http://proxy.local:8080"),
            verify=True,
        )

        assert http2 is False
        assert kwargs["proxy"] == "http://proxy.local:8080"
