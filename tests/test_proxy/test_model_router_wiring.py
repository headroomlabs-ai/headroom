"""Wiring tests for cost-aware model routing (issue #1706).

Covers env -> ProxyConfig, ProxyConfig -> live proxy, and the presence of the
routing block in the Anthropic request handler.
"""

from __future__ import annotations

import inspect
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.model_router import ModelRoute, ModelRouter, ModelRouterConfig
from headroom.proxy.server import ProxyConfig, _proxy_config_from_env, create_app

MESSAGES = "/v1/messages"


def _install_fake_client(proxy) -> MagicMock:
    """Replace proxy.http_client so forwarding never touches the network.

    The buffered ``/v1/messages`` path forwards via ``http_client.post(content=...)``;
    the other forward shapes are stubbed too so the mock is robust to path choice.
    """
    response = httpx.Response(
        200, json={"ok": True}, request=httpx.Request("POST", "http://upstream/v1/messages")
    )
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    client.request = AsyncMock(return_value=response)
    client.send = AsyncMock(return_value=response)
    client.build_request = MagicMock(
        return_value=httpx.Request("POST", "http://upstream/v1/messages", content=b"{}")
    )
    client.aclose = AsyncMock()
    proxy.http_client = client
    return client


def _forwarded_model(client: MagicMock) -> str:
    """Parse the outgoing model from the content forwarded upstream."""
    return _forwarded_body(client)["model"]


def _forwarded_body(client: MagicMock) -> dict:
    """Parse the JSON body forwarded upstream."""
    content = client.post.call_args.kwargs["content"]
    return json.loads(content)


def _router_config() -> ModelRouterConfig:
    return ModelRouterConfig(
        enabled=True,
        routes=(
            ModelRoute(
                to_model="claude-haiku-4-5",
                max_input_tokens=100_000,
                require_no_tools=True,
                name="low-risk",
            ),
        ),
    )


def test_proxy_config_from_env_reads_router(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_MODEL_ROUTER_ENABLED", "true")
    monkeypatch.setenv(
        "HEADROOM_MODEL_ROUTES",
        '[{"name":"small","max_input_tokens":4000,"require_no_tools":true,'
        '"to_model":"claude-haiku-4-5"}]',
    )
    config = _proxy_config_from_env()
    assert config.model_router is not None
    assert config.model_router.enabled
    assert config.model_router.routes[0].to_model == "claude-haiku-4-5"


def test_proxy_config_from_env_router_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("HEADROOM_MODEL_ROUTER_ENABLED", raising=False)
    monkeypatch.delenv("HEADROOM_MODEL_ROUTES", raising=False)
    config = _proxy_config_from_env()
    assert config.model_router is not None
    assert not config.model_router.enabled


def test_create_app_wires_model_router() -> None:
    config = ProxyConfig(
        optimize=False,
        image_optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        model_router=ModelRouterConfig(
            enabled=True,
            routes=(ModelRoute(to_model="cheap", max_input_tokens=10_000, name="small"),),
        ),
    )
    app = create_app(config)
    with TestClient(app) as client:
        router = client.app.state.proxy.model_router
        assert router.enabled
        decision = router.select(model="strong", input_tokens=500, has_tools=False)
        assert decision.changed and decision.routed_model == "cheap"


def test_create_app_router_disabled_when_unset() -> None:
    app = create_app(ProxyConfig(optimize=False, cost_tracking_enabled=False))
    with TestClient(app) as client:
        assert not client.app.state.proxy.model_router.enabled


def test_handler_delegates_to_maybe_route_model() -> None:
    src = inspect.getsource(AnthropicHandlerMixin.handle_anthropic_messages)
    assert "_maybe_route_model(" in src, "handler must apply model routing"


class _RouterHost(AnthropicHandlerMixin):
    """Minimal mixin host (like a handler test double) for routing-only tests."""


def test_maybe_route_model_fails_closed_without_router() -> None:
    # A host that never set model_router (test doubles, alternate mixin hosts that
    # do not run HeadroomProxy.__init__) must not crash when routing is off.
    host = _RouterHost()
    tracker = MagicMock()
    out = host._maybe_route_model(
        "claude-sonnet-4-6", [{"content": "hi"}], {"model": "claude-sonnet-4-6"}, tracker, False
    )
    assert out == "claude-sonnet-4-6"
    tracker.mark_mutated.assert_not_called()


def test_maybe_route_model_routes_when_enabled() -> None:
    host = _RouterHost()
    host.model_router = ModelRouter(
        ModelRouterConfig(
            enabled=True,
            routes=(
                ModelRoute(
                    to_model="claude-haiku-4-5", max_input_tokens=100_000, require_no_tools=True
                ),
            ),
        )
    )
    tracker = MagicMock()
    body = {"model": "claude-sonnet-4-6"}
    out = host._maybe_route_model("claude-sonnet-4-6", [{"content": "hi"}], body, tracker, False)
    assert out == "claude-haiku-4-5"
    assert body["model"] == "claude-haiku-4-5"
    tracker.mark_mutated.assert_called_once_with("model_router")


@pytest.mark.parametrize(
    ("routes", "expected_reason"),
    [
        ((ModelRoute(to_model="keep", from_models=("keep",), name="exempt"),), "exempt"),
        ((ModelRoute(to_model="cheap", from_models=("other",)),), "no rule matched"),
    ],
)
def test_maybe_route_model_logs_unchanged_decision(
    caplog: pytest.LogCaptureFixture,
    routes: tuple[ModelRoute, ...],
    expected_reason: str,
) -> None:
    host = _RouterHost()
    host.model_router = ModelRouter(ModelRouterConfig(enabled=True, routes=routes))

    with caplog.at_level(logging.INFO, logger="headroom.proxy"):
        out = host._maybe_route_model(
            "keep", [{"content": "hi"}], {"model": "keep"}, MagicMock(), False
        )

    assert out == "keep"
    decisions = [
        record.message for record in caplog.records if "model routing decision" in record.message
    ]
    assert len(decisions) == 1
    assert expected_reason in decisions[0]


def test_maybe_route_model_skips_on_bypass() -> None:
    host = _RouterHost()
    host.model_router = ModelRouter(
        ModelRouterConfig(enabled=True, routes=(ModelRoute(to_model="cheap"),))
    )
    tracker = MagicMock()
    out = host._maybe_route_model("keep", [{"content": "hi"}], {"model": "keep"}, tracker, True)
    assert out == "keep"
    tracker.mark_mutated.assert_not_called()


def _messages_config() -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        mode="token",
        model_router=_router_config(),
    )


def test_messages_request_gets_model_rewritten_when_enabled() -> None:
    app = create_app(_messages_config())
    with TestClient(app) as client:
        http = _install_fake_client(client.app.state.proxy)
        resp = client.post(
            MESSAGES,
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert resp.status_code == 200
    # A low-risk request routes to the cheaper model on the forwarded body.
    assert _forwarded_model(http) == "claude-haiku-4-5"


def test_bypass_request_is_never_model_rewritten() -> None:
    app = create_app(_messages_config())
    with TestClient(app) as client:
        http = _install_fake_client(client.app.state.proxy)
        resp = client.post(
            MESSAGES,
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"x-headroom-bypass": "true"},
        )
    assert resp.status_code == 200
    # Byte-faithful passthrough must keep the client's original model.
    assert _forwarded_model(http) == "claude-sonnet-4-6"


def test_vertex_raw_predict_model_is_not_rewritten_in_body() -> None:
    # When the model comes from the provider URL (Vertex rawPredict), the upstream
    # model is set by the path, so routing must not rewrite body["model"].
    app = create_app(_messages_config())
    with TestClient(app) as client:
        http = _install_fake_client(client.app.state.proxy)
        resp = client.post(
            "/v1/projects/p/locations/us-central1/publishers/anthropic/models/"
            "claude-sonnet-4-6:rawPredict",
            json={
                "anthropic_version": "vertex-2023-10-16",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert resp.status_code == 200
    assert "model" not in _forwarded_body(http)
