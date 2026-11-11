"""Unit tests for hybrid Anthropic Cloud + local oMLX proxy routing (PR #4597)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from headroom.config import HeadroomConfig, resolve_omlx_target_api_url
from headroom.providers.anthropic import ANTHROPIC_CONTEXT_LIMITS, ANTHROPIC_PRICING
from headroom.proxy.handlers.anthropic import is_omlx_local_model, strip_omlx_model_prefix
from headroom.proxy.model_router import ModelRoute, ModelRouterConfig
from headroom.proxy.server import ProxyConfig, _proxy_config_from_env, create_app
from headroom.proxy.upstream_trust import is_trusted_upstream, trusted_upstream_hosts

MESSAGES = "/v1/messages"
OMLX_MESSAGES = "/omlx/v1/messages"
OMLX_URL = "http://127.0.0.1:8888"


def _install_fake_client(proxy) -> MagicMock:
    """Replace proxy.http_client so forwarding never touches the network."""
    response = httpx.Response(
        200,
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": "test",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 2},
        },
        request=httpx.Request("POST", "http://upstream/v1/messages"),
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


def _make_config(**overrides) -> ProxyConfig:
    base = {
        "optimize": False,
        "image_optimize": False,
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "cost_tracking_enabled": False,
        "ccr_inject_tool": False,
        "ccr_handle_responses": False,
        "ccr_context_tracking": False,
        "mode": "token",
    }
    base.update(overrides)
    return ProxyConfig(**base)


def _forwarded_url(http: MagicMock) -> str:
    return str(http.post.call_args.args[0])


def _forwarded_headers(http: MagicMock) -> dict[str, str]:
    raw = http.post.call_args.kwargs.get("headers") or {}
    return {k.lower(): v for k, v in raw.items()}


def _forwarded_body(http: MagicMock) -> dict:
    content = http.post.call_args.kwargs["content"]
    return json.loads(content)


@pytest.mark.parametrize(
    "claude_model",
    [
        "claude-haiku-4-5-20251001",
        "claude-haiku-5-5",
        "claude-fable-5",
        "claude-fable-5-1",
        "claude-sonnet-4-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        *sorted(set(ANTHROPIC_CONTEXT_LIMITS) | set(ANTHROPIC_PRICING)),
    ],
)
def test_claude_models_route_to_anthropic_cloud_even_with_omlx_configured(
    monkeypatch: pytest.MonkeyPatch,
    claude_model: str,
) -> None:
    monkeypatch.setenv("HEADROOM_OMLX_BASE_URL", OMLX_URL)
    monkeypatch.setenv("OMLX_TARGET_API_URL", OMLX_URL)
    assert not is_omlx_local_model(claude_model)

    app = create_app(_make_config(omlx_target_api_url=OMLX_URL))
    with TestClient(app) as client:
        http = _install_fake_client(client.app.state.proxy)
        resp = client.post(
            MESSAGES,
            json={
                "model": claude_model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"x-api-key": "sk-ant-cloud-secret"},
        )

    assert resp.status_code == 200
    assert _forwarded_url(http) == "https://api.anthropic.com/v1/messages"
    assert _forwarded_headers(http).get("x-api-key") == "sk-ant-cloud-secret"


@pytest.mark.parametrize(
    ("local_model", "expected_wire_model"),
    [
        ("omlx/qwen3-8b", "qwen3-8b"),
        ("local/gemma-4-9b", "gemma-4-9b"),
        ("gemma-4-9b", "gemma-4-9b"),
        ("qwen2.5-coder", "qwen2.5-coder"),
    ],
)
def test_explicit_local_models_route_to_omlx_target(
    monkeypatch: pytest.MonkeyPatch,
    local_model: str,
    expected_wire_model: str,
) -> None:
    monkeypatch.delenv("HEADROOM_OMLX_API_KEY", raising=False)
    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    assert is_omlx_local_model(local_model)
    assert strip_omlx_model_prefix(local_model) == expected_wire_model

    app = create_app(_make_config(omlx_target_api_url=OMLX_URL))
    with TestClient(app) as client:
        http = _install_fake_client(client.app.state.proxy)
        resp = client.post(
            MESSAGES,
            json={
                "model": local_model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert resp.status_code == 200
    assert _forwarded_url(http) == f"{OMLX_URL}/v1/messages"
    assert _forwarded_body(http)["model"] == expected_wire_model


def test_allowlist_env_routes_custom_model_to_omlx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_OMLX_MODELS", "custom-apple-silicon-7b, another-local")
    assert is_omlx_local_model("custom-apple-silicon-7b")
    assert not is_omlx_local_model("unlisted-cloud-model")


def test_maybe_route_model_runs_before_local_or_cloud_target_selection() -> None:
    # 1. Cloud -> local rewrite must route to omlx_target and strip omlx/ prefix.
    cloud_to_local_cfg = _make_config(
        omlx_target_api_url=OMLX_URL,
        model_router=ModelRouterConfig(
            enabled=True,
            routes=(
                ModelRoute(
                    to_model="omlx/qwen3-8b",
                    max_input_tokens=100_000,
                    require_no_tools=True,
                    name="offload-local",
                ),
            ),
        ),
    )
    app = create_app(cloud_to_local_cfg)
    with TestClient(app) as client:
        http = _install_fake_client(client.app.state.proxy)
        resp = client.post(
            MESSAGES,
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"x-api-key": "sk-ant-cloud-secret"},
        )
    assert resp.status_code == 200
    assert _forwarded_url(http) == f"{OMLX_URL}/v1/messages"
    assert _forwarded_body(http)["model"] == "qwen3-8b"
    assert "x-api-key" not in _forwarded_headers(http)

    # 2. Local -> cloud rewrite must route to Anthropic cloud and keep cloud credentials.
    local_to_cloud_cfg = _make_config(
        omlx_target_api_url=OMLX_URL,
        model_router=ModelRouterConfig(
            enabled=True,
            routes=(
                ModelRoute(
                    to_model="claude-haiku-4-5-20251001",
                    max_input_tokens=100_000,
                    require_no_tools=True,
                    name="escalate-cloud",
                ),
            ),
        ),
    )
    app2 = create_app(local_to_cloud_cfg)
    with TestClient(app2) as client:
        http2 = _install_fake_client(client.app.state.proxy)
        resp2 = client.post(
            MESSAGES,
            json={
                "model": "omlx/qwen3-8b",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"x-api-key": "sk-ant-cloud-secret"},
        )
    assert resp2.status_code == 200
    assert _forwarded_url(http2) == "https://api.anthropic.com/v1/messages"
    assert _forwarded_body(http2)["model"] == "claude-haiku-4-5-20251001"
    assert _forwarded_headers(http2).get("x-api-key") == "sk-ant-cloud-secret"


def test_cloud_credentials_stripped_or_replaced_when_forwarding_to_omlx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HEADROOM_OMLX_API_KEY", raising=False)
    monkeypatch.delenv("OMLX_API_KEY", raising=False)

    app = create_app(_make_config(omlx_target_api_url=OMLX_URL))
    with TestClient(app) as client:
        http = _install_fake_client(client.app.state.proxy)
        resp = client.post(
            MESSAGES,
            json={
                "model": "omlx/qwen3-8b",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={
                "x-api-key": "sk-ant-api03-secret",
                "Authorization": "Bearer sk-ant-oat01-secret",
            },
        )
    assert resp.status_code == 200
    forwarded_headers = _forwarded_headers(http)
    assert "x-api-key" not in forwarded_headers
    assert "authorization" not in forwarded_headers

    # When HEADROOM_OMLX_API_KEY is configured, replace with the local key.
    monkeypatch.setenv("HEADROOM_OMLX_API_KEY", "local-omlx-token")
    with TestClient(app) as client:
        http2 = _install_fake_client(client.app.state.proxy)
        resp2 = client.post(
            OMLX_MESSAGES,
            json={
                "model": "omlx/qwen3-8b",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={
                "x-api-key": "sk-ant-api03-secret",
                "Authorization": "Bearer sk-ant-oat01-secret",
            },
        )
    assert resp2.status_code == 200
    forwarded_headers2 = _forwarded_headers(http2)
    assert forwarded_headers2.get("x-api-key") == "local-omlx-token"
    assert forwarded_headers2.get("authorization") == "Bearer local-omlx-token"


def test_headroom_omlx_base_url_and_omlx_target_api_url_env_consistency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMLX_TARGET_API_URL", raising=False)
    monkeypatch.delenv("HEADROOM_OMLX_URL", raising=False)
    monkeypatch.setenv("HEADROOM_OMLX_BASE_URL", "http://127.0.0.1:8000/")

    assert resolve_omlx_target_api_url() == "http://127.0.0.1:8000"
    assert HeadroomConfig().omlx_target_api_url == "http://127.0.0.1:8000"
    assert ProxyConfig().omlx_target_api_url == "http://127.0.0.1:8000"
    assert _proxy_config_from_env().omlx_target_api_url == "http://127.0.0.1:8000"

    from headroom.cli.proxy import proxy as proxy_cli_cmd

    runner = CliRunner()
    help_res = runner.invoke(proxy_cli_cmd, ["--help"])
    assert help_res.exit_code == 0
    assert "--omlx-api-url" in help_res.output


def test_omlx_target_api_url_included_in_trusted_upstream_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMLX_TARGET_API_URL", raising=False)
    monkeypatch.delenv("HEADROOM_OMLX_BASE_URL", raising=False)
    monkeypatch.delenv("HEADROOM_OMLX_URL", raising=False)

    cfg = _make_config(
        omlx_target_api_url="http://omlx.internal:8888",
        anthropic_extra_headers={"X-Custom-Gateway-Auth": "gateway-secret"},
    )
    assert "omlx.internal" in trusted_upstream_hosts(cfg)
    assert is_trusted_upstream("http://omlx.internal:8888/v1/messages", cfg)

    app = create_app(cfg)
    with TestClient(app) as client:
        http = _install_fake_client(client.app.state.proxy)
        resp = client.post(
            MESSAGES,
            json={
                "model": "omlx/qwen3-8b",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert resp.status_code == 200
    assert _forwarded_headers(http).get("x-custom-gateway-auth") == "gateway-secret"


@pytest.mark.parametrize("endpoint", [MESSAGES, OMLX_MESSAGES])
@pytest.mark.parametrize(
    ("extra_header_name", "extra_header_value"),
    [
        ("x-api-key", "dummy-cloud-secret"),
        ("Authorization", "Bearer dummy-cloud-oauth"),
    ],
)
@pytest.mark.parametrize("local_omlx_key", [None, "local-omlx-token"])
def test_configured_anthropic_extra_headers_never_leak_or_override_on_omlx_routes(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    extra_header_name: str,
    extra_header_value: str,
    local_omlx_key: str | None,
) -> None:
    """Cover all 8 destination-aware credential isolation combinations (PR #4597 review)."""
    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    if local_omlx_key is None:
        monkeypatch.delenv("HEADROOM_OMLX_API_KEY", raising=False)
    else:
        monkeypatch.setenv("HEADROOM_OMLX_API_KEY", local_omlx_key)

    cfg = _make_config(
        omlx_target_api_url=OMLX_URL,
        anthropic_extra_headers={
            extra_header_name: extra_header_value,
            "X-Custom-Gateway-Auth": "gateway-secret",
        },
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        http = _install_fake_client(client.app.state.proxy)
        resp = client.post(
            endpoint,
            json={
                "model": "omlx/qwen3-8b",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={
                "x-api-key": "inbound-cloud-secret",
                "Authorization": "Bearer inbound-cloud-oauth",
            },
        )

    assert resp.status_code == 200
    assert _forwarded_url(http) == f"{OMLX_URL}/v1/messages"
    assert _forwarded_body(http)["model"] == "qwen3-8b"
    forwarded_headers = _forwarded_headers(http)
    assert forwarded_headers.get("x-custom-gateway-auth") == "gateway-secret"
    if local_omlx_key is None:
        assert "x-api-key" not in forwarded_headers
        assert "authorization" not in forwarded_headers
    else:
        assert forwarded_headers.get("x-api-key") == local_omlx_key
        assert forwarded_headers.get("authorization") == f"Bearer {local_omlx_key}"


@pytest.mark.parametrize(
    ("extra_header_name", "extra_header_value", "expected_lower_key"),
    [
        ("x-api-key", "configured-cloud-api-key", "x-api-key"),
        ("Authorization", "Bearer configured-cloud-oauth", "authorization"),
    ],
)
def test_cloud_requests_retain_configured_anthropic_extra_headers_with_omlx_enabled(
    monkeypatch: pytest.MonkeyPatch,
    extra_header_name: str,
    extra_header_value: str,
    expected_lower_key: str,
) -> None:
    monkeypatch.setenv("HEADROOM_OMLX_API_KEY", "local-omlx-token")
    cfg = _make_config(
        omlx_target_api_url=OMLX_URL,
        anthropic_extra_headers={extra_header_name: extra_header_value},
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        http = _install_fake_client(client.app.state.proxy)
        resp = client.post(
            MESSAGES,
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={extra_header_name: "inbound-overridden-secret"},
        )

    assert resp.status_code == 200
    assert _forwarded_url(http) == "https://api.anthropic.com/v1/messages"
    assert _forwarded_headers(http).get(expected_lower_key) == extra_header_value


def test_subscription_tracker_skips_omlx_routes_and_notifies_cloud_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import headroom.subscription.tracker as sub_tracker_mod

    fake_tracker = MagicMock()
    monkeypatch.setattr(sub_tracker_mod, "get_subscription_tracker", lambda: fake_tracker)
    monkeypatch.delenv("HEADROOM_OMLX_API_KEY", raising=False)
    monkeypatch.delenv("OMLX_API_KEY", raising=False)

    cfg = _make_config(
        omlx_target_api_url=OMLX_URL,
        anthropic_extra_headers={"Authorization": "Bearer configured-cloud-oauth"},
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        _install_fake_client(client.app.state.proxy)
        for endpoint in (MESSAGES, OMLX_MESSAGES):
            resp = client.post(
                endpoint,
                json={
                    "model": "omlx/qwen3-8b",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"Authorization": "Bearer inbound-cloud-oauth"},
            )
            assert resp.status_code == 200
        assert fake_tracker.notify_active.call_count == 0

        cloud_resp = client.post(
            MESSAGES,
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert cloud_resp.status_code == 200
        assert fake_tracker.notify_active.call_count == 1
        assert fake_tracker.notify_active.call_args.args[0] == "Bearer configured-cloud-oauth"
