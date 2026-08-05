"""Proxy configuration for disabling Kompress while keeping optimization on."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from headroom.proxy.server import ProxyConfig, create_app
from headroom.transforms import CompressionStrategy, ContentRouter


def _proxy_router(config: ProxyConfig):
    app = create_app(config)
    proxy = app.state.proxy
    return next(
        transform
        for transform in proxy.anthropic_pipeline.transforms
        if isinstance(transform, ContentRouter)
    )


def test_disable_kompress_config_keeps_optimization_but_disables_ml_fallback() -> None:
    router = _proxy_router(
        ProxyConfig(
            optimize=True,
            disable_kompress=True,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
        )
    )

    assert router.config.enable_kompress is False
    assert router.config.fallback_strategy == CompressionStrategy.KOMPRESS


def test_disable_kompress_defaults_to_existing_kompress_behavior() -> None:
    router = _proxy_router(
        ProxyConfig(
            optimize=True,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
        )
    )

    assert router.config.enable_kompress is True
    assert router.config.fallback_strategy == CompressionStrategy.KOMPRESS


def test_health_config_reports_disable_kompress_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    app = create_app(
        ProxyConfig(
            optimize=True,
            disable_kompress=True,
            disable_kompress_fallback=True,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
        )
    )
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        config = client.get("/health").json()["config"]

    assert config["disable_kompress"] is True
    assert config["disable_kompress_fallback"] is True


def test_disable_kompress_fallback_restores_passthrough() -> None:
    router = _proxy_router(
        ProxyConfig(
            optimize=True,
            disable_kompress=True,
            disable_kompress_fallback=True,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
        )
    )

    assert router.config.enable_kompress is False
    assert router.config.fallback_strategy == CompressionStrategy.PASSTHROUGH


def test_disable_kompress_fallback_without_disable_kompress_is_noop() -> None:
    router = _proxy_router(
        ProxyConfig(
            optimize=True,
            disable_kompress_fallback=True,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
        )
    )

    assert router.config.enable_kompress is True
    assert router.config.fallback_strategy == CompressionStrategy.KOMPRESS
