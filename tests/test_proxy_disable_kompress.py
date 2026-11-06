"""Proxy configuration for disabling Kompress while keeping optimization on."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from headroom.proxy.server import ProxyConfig, create_app
from headroom.transforms import CompressionStrategy, ContentRouter


@pytest.fixture(autouse=True)
def _stub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_REQUIRE_RUST_CORE", "false")
    monkeypatch.setenv("HEADROOM_HTTP2", "false")


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


def test_disable_kompress_prevents_kompress_instantiation_in_content_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    import headroom.transforms.kompress_compressor as kc
    from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

    mock_avail = MagicMock(return_value=True)
    mock_compressor = MagicMock()
    monkeypatch.setattr(kc, "is_kompress_available", mock_avail)
    monkeypatch.setattr(kc, "KompressCompressor", mock_compressor)

    router = ContentRouter(ContentRouterConfig(enable_kompress=False))
    result = router._get_kompress()
    assert result is None
    mock_avail.assert_not_called()
    mock_compressor.assert_not_called()


def test_disable_kompress_bypasses_remote_kompress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

    mock_remote = MagicMock()
    router = ContentRouter(ContentRouterConfig(enable_kompress=False))
    monkeypatch.setattr(router, "_get_remote_kompress", mock_remote)

    result = router._get_kompress()
    assert result is None
    mock_remote.assert_not_called()


def test_disable_kompress_reasoning_compaction_skips_kompress_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    import headroom.transforms.kompress_compressor as kc
    from headroom.proxy.server import ProxyConfig, create_app
    from headroom.transforms.compression_units import find_content_router

    mock_avail = MagicMock(return_value=True)
    mock_compressor = MagicMock()
    monkeypatch.setattr(kc, "is_kompress_available", mock_avail)
    monkeypatch.setattr(kc, "KompressCompressor", mock_compressor)

    config = ProxyConfig(
        optimize=True,
        disable_kompress=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
    )
    app = create_app(config)
    proxy = app.state.proxy
    router = find_content_router(proxy.openai_pipeline)
    assert router is not None
    assert router.config.enable_kompress is False
    assert router._get_kompress() is None
    mock_avail.assert_not_called()
    mock_compressor.assert_not_called()
