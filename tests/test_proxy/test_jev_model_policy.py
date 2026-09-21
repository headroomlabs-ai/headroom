"""Unit tests for optional Jev model routing policy (#3690)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest

from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.jev_model_policy import (
    JevModelPolicy,
    JevModelPolicyConfig,
    latest_user_text,
)
from headroom.proxy.model_router import ModelRoute, ModelRouter, ModelRouterConfig
from headroom.proxy.server import ProxyConfig, _proxy_config_from_env, create_app


def _ok_response(tier: str, confidence: float, difficulty: float = 1.0) -> httpx.Response:
    body = {
        "model": "jev-1.13.0",
        "answers": {
            "tier": {
                "type": "choice",
                "choice": tier,
                "confidence": confidence,
                "probabilities": {tier: confidence},
            },
            "difficulty": {
                "type": "score",
                "score": difficulty,
                "confidence": 0.9,
                "legend": {"0": "trivial", "1": "moderate", "2": "hard"},
                "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0},
            },
        },
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }
    return httpx.Response(
        200,
        json=body,
        request=httpx.Request("POST", "https://api.typesafe.ai/v1/systemone"),
    )


def _policy(
    *,
    post: MagicMock,
    confidence_min: float = 0.6,
) -> JevModelPolicy:
    return JevModelPolicy(
        JevModelPolicyConfig(
            enabled=True,
            api_key="test-key",
            tier_models={
                "economy": "cheap-model",
                "standard": "mid-model",
                "frontier": "strong-model",
            },
            confidence_min=confidence_min,
        ),
        post=post,
    )


def test_select_routes_to_economy_on_high_confidence() -> None:
    post = MagicMock(return_value=_ok_response("economy", 0.95, 0.1))
    decision = _policy(post=post).select(model="strong-model", user_text="Rename foo to bar")
    assert decision.changed
    assert decision.routed_model == "cheap-model"
    assert "jev matched tier=economy" in decision.reason
    post.assert_called_once()
    kwargs = post.call_args.kwargs
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["json"]["model"] == "jev-latest"
    assert "Rename foo to bar" in kwargs["json"]["state"]


def test_select_keeps_original_on_low_confidence() -> None:
    post = MagicMock(return_value=_ok_response("economy", 0.4))
    decision = _policy(post=post).select(model="strong-model", user_text="maybe rename?")
    assert decision.matched
    assert not decision.changed
    assert decision.routed_model == "strong-model"
    assert "keeping strong-model" in decision.reason


def test_select_fail_open_on_http_error() -> None:
    post = MagicMock(side_effect=httpx.TimeoutException("boom"))
    decision = _policy(post=post).select(model="strong-model", user_text="hard refactor")
    assert not decision.matched
    assert not decision.changed
    assert "jev error" in decision.reason


def test_select_disabled_without_config() -> None:
    decision = JevModelPolicy().select(model="m", user_text="hi")
    assert not decision.matched
    assert decision.reason == "jev router disabled"


def test_from_env_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_JEV_MODEL_ROUTER", raising=False)
    cfg = JevModelPolicyConfig.from_env()
    assert not cfg.enabled


def test_from_env_requires_key_and_tiers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_JEV_MODEL_ROUTER", "1")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv(
        "HEADROOM_JEV_TIER_MODELS",
        json.dumps({"economy": "a", "standard": "b", "frontier": "c"}),
    )
    assert not JevModelPolicyConfig.from_env().enabled

    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.delenv("HEADROOM_JEV_TIER_MODELS", raising=False)
    assert not JevModelPolicyConfig.from_env().enabled

    monkeypatch.setenv(
        "HEADROOM_JEV_TIER_MODELS",
        json.dumps({"economy": "a", "standard": "b", "frontier": "c"}),
    )
    cfg = JevModelPolicyConfig.from_env()
    assert cfg.enabled
    assert cfg.tier_models["economy"] == "a"


def test_proxy_config_from_env_wires_jev(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_JEV_MODEL_ROUTER", "true")
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setenv(
        "HEADROOM_JEV_TIER_MODELS",
        '{"economy":"e","standard":"s","frontier":"f"}',
    )
    config = _proxy_config_from_env()
    assert config.jev_model_policy is not None
    assert config.jev_model_policy.enabled


def test_create_app_wires_jev_policy() -> None:
    config = ProxyConfig(
        optimize=False,
        image_optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        jev_model_policy=JevModelPolicyConfig(
            enabled=True,
            api_key="k",
            tier_models={"economy": "e", "standard": "s", "frontier": "f"},
        ),
    )
    app = create_app(config)
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        assert client.app.state.proxy.jev_model_policy.enabled


def test_latest_user_text_prefers_newest_user() -> None:
    messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "text", "text": "new ask"}]},
    ]
    assert latest_user_text(messages) == "new ask"


class _Host(AnthropicHandlerMixin):
    pass


def test_maybe_route_model_jev_precedence_over_static() -> None:
    host = _Host()
    post = MagicMock(return_value=_ok_response("economy", 0.99))
    host.jev_model_policy = _policy(post=post)
    host.model_router = ModelRouter(
        ModelRouterConfig(
            enabled=True,
            routes=(ModelRoute(to_model="static-cheap", max_input_tokens=100_000, name="static"),),
        )
    )
    tracker = MagicMock()
    body: dict = {"model": "strong-model"}
    out = host._maybe_route_model(
        "strong-model",
        [{"role": "user", "content": "rename x"}],
        body,
        tracker,
        False,
    )
    assert out == "cheap-model"
    assert body["model"] == "cheap-model"
    tracker.mark_mutated.assert_called_once_with("jev_model_router")


def test_maybe_route_model_falls_through_to_static_on_jev_error() -> None:
    host = _Host()
    post = MagicMock(side_effect=httpx.ConnectError("nope"))
    host.jev_model_policy = _policy(post=post)
    host.model_router = ModelRouter(
        ModelRouterConfig(
            enabled=True,
            routes=(
                ModelRoute(
                    to_model="static-cheap",
                    max_input_tokens=100_000,
                    require_no_tools=True,
                    name="static",
                ),
            ),
        )
    )
    tracker = MagicMock()
    body: dict = {"model": "strong-model"}
    out = host._maybe_route_model(
        "strong-model",
        [{"role": "user", "content": "hi"}],
        body,
        tracker,
        False,
    )
    assert out == "static-cheap"
    tracker.mark_mutated.assert_called_once_with("model_router")
