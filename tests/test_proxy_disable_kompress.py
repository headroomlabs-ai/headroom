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


def test_disabled_kompress_reasoning_request_never_queries_remote_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from unittest.mock import MagicMock

    import httpx
    from fastapi.testclient import TestClient

    from headroom.transforms.compression_units import find_content_router

    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    monkeypatch.setenv("HEADROOM_THINKING_COMPACT", "1")
    monkeypatch.setenv("HEADROOM_THINKING_COMPACT_KEEP_LAST", "0")
    monkeypatch.setenv("HEADROOM_KOMPRESS_ENDPOINT", "http://127.0.0.1:9")
    app = create_app(
        ProxyConfig(
            optimize=True,
            disable_kompress=True,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            image_optimize=False,
        )
    )
    forwarded = []

    def upstream(request: httpx.Request) -> httpx.Response:
        forwarded.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "model": "kimi-k2.7",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 1, "total_tokens": 101},
            },
        )

    reasoning = " ".join(f"reason{i}" for i in range(100))
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        proxy = app.state.proxy

        async def install_transport() -> None:
            await proxy.http_client.aclose()
            if proxy.http_client_h1 is not None:
                await proxy.http_client_h1.aclose()
            proxy.http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
            proxy.http_client_h1 = None

        client.portal.call(install_transport)
        router = find_content_router(proxy.openai_pipeline)
        assert router is not None
        remote_loader = MagicMock(return_value=None)
        monkeypatch.setattr(router, "_get_remote_kompress", remote_loader)
        response = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer local-fixture"},
            json={
                "model": "kimi-k2.7",
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer", "reasoning_content": reasoning},
                    {"role": "user", "content": "continue"},
                ],
            },
        )
        assert response.status_code == 200
        remote_loader.assert_not_called()
        assert len(forwarded) == 1
        assert forwarded[0]["messages"][1]["reasoning_content"] == reasoning
