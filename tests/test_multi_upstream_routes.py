"""Unit tests for HEADROOM_UPSTREAM_ROUTES multi-upstream routing.

Covers parse_upstream_routes() (env JSON -> UpstreamRoute tuple) and
ProxyProviderRuntime.resolve_upstream() (model-prefix match -> base URL
+ auth mode). With no routes configured, resolve_upstream() must fall
back to the legacy select_passthrough_base_url() + PassthroughAuth
resolution so behavior is byte-identical to a build without this
feature.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from headroom.providers.registry import (
    BearerAuth,
    PassthroughAuth,
    ProviderApiTargets,
    ProxyProviderRuntime,
    UpstreamRoute,
    UpstreamRoutesConfigError,
    parse_upstream_routes,
)


def _routes(*entries: dict[str, Any]) -> tuple[UpstreamRoute, ...]:
    return parse_upstream_routes({"HEADROOM_UPSTREAM_ROUTES": json.dumps(list(entries))})


def _runtime(
    *,
    routes: tuple[UpstreamRoute, ...] = (),
    anthropic: str | None = None,
    openai: str | None = None,
) -> ProxyProviderRuntime:
    targets = ProviderApiTargets(
        anthropic=anthropic or "https://api.anthropic.com",
        openai=openai or "https://api.openai.com",
    )
    return ProxyProviderRuntime(
        api_targets=targets,
        pipeline_providers={},  # type: ignore[arg-type]
        upstream_routes=routes,
    )


# --- parse_upstream_routes ---


def test_parse_empty_env_returns_empty() -> None:
    assert parse_upstream_routes({}) == ()
    assert parse_upstream_routes({"HEADROOM_UPSTREAM_ROUTES": ""}) == ()


def test_parse_invalid_json_fails_closed() -> None:
    with pytest.raises(UpstreamRoutesConfigError, match="invalid JSON"):
        parse_upstream_routes({"HEADROOM_UPSTREAM_ROUTES": "{not json"})


def test_parse_non_array_fails_closed() -> None:
    with pytest.raises(UpstreamRoutesConfigError, match="JSON array"):
        parse_upstream_routes({"HEADROOM_UPSTREAM_ROUTES": '{"a":1}'})


def test_parse_valid_routes() -> None:
    routes = _routes(
        {"model_prefix": "glm-", "upstream": "https://ollama.com", "auth": "env:OLLAMA_API_KEY"},
        {"model_prefix": "claude-", "protocol": "anthropic"},
        {"model_prefix": "*", "auth": "passthrough"},
    )
    assert len(routes) == 3
    assert routes[0].model_prefix == "glm-"
    assert routes[0].upstream == "https://ollama.com"
    assert isinstance(routes[0].auth, BearerAuth)
    assert routes[0].auth.env_var == "OLLAMA_API_KEY"
    assert routes[1].protocol == "anthropic"
    assert routes[1].upstream is None
    assert isinstance(routes[1].auth, PassthroughAuth)
    assert routes[2].model_prefix == "*"


def test_parse_strips_trailing_slash() -> None:
    routes = _routes({"model_prefix": "glm-", "upstream": "https://ollama.com/"})
    assert routes[0].upstream == "https://ollama.com"


def test_parse_empty_prefix_fails_closed() -> None:
    with pytest.raises(UpstreamRoutesConfigError, match="model_prefix"):
        _routes(
            {"model_prefix": "", "upstream": "https://x"},
            {"model_prefix": "glm-", "upstream": "https://ollama.com"},
        )


def test_parse_bad_protocol_fails_closed() -> None:
    with pytest.raises(UpstreamRoutesConfigError, match="protocol"):
        _routes({"model_prefix": "x-", "protocol": "bogus"})


def test_parse_bad_auth_fails_closed() -> None:
    with pytest.raises(UpstreamRoutesConfigError, match="auth"):
        _routes({"model_prefix": "x-", "auth": "weird"})


@pytest.mark.parametrize(
    "upstream",
    [
        "file:///tmp/socket",
        "https://user:secret@example.com",
        "example.com",
        123,
        "https://gateway.example/api?tenant=x",
        "https://gateway.example/api#fragment",
    ],
)
def test_parse_unsafe_or_invalid_upstream_fails_closed(upstream: object) -> None:
    with pytest.raises(UpstreamRoutesConfigError, match="upstream"):
        _routes({"model_prefix": "x-", "upstream": upstream})


@pytest.mark.parametrize("auth", ["env:", "env:BAD-NAME", 123])
def test_parse_invalid_bearer_env_fails_closed(auth: object) -> None:
    with pytest.raises(UpstreamRoutesConfigError, match="auth"):
        _routes({"model_prefix": "x-", "auth": auth})


# --- resolve_upstream ---


def test_resolve_empty_routes_is_legacy() -> None:
    rt = _runtime()
    res = rt.resolve_upstream(protocol="openai", model="gpt-5.5", headers={})
    assert res.base_url == "https://api.openai.com"
    assert isinstance(res.auth, PassthroughAuth)


def test_resolve_prefix_match_uses_route_upstream() -> None:
    rt = _runtime(
        routes=_routes(
            {
                "model_prefix": "glm-",
                "upstream": "https://ollama.com",
                "auth": "env:OLLAMA_API_KEY",
            },
        )
    )
    res = rt.resolve_upstream(protocol="openai", model="glm-5.2", headers={})
    assert res.base_url == "https://ollama.com"
    assert isinstance(res.auth, BearerAuth)
    assert res.auth.env_var == "OLLAMA_API_KEY"


def test_resolve_no_match_falls_back_to_star() -> None:
    rt = _runtime(
        routes=_routes(
            {"model_prefix": "glm-", "upstream": "https://ollama.com"},
            {"model_prefix": "*", "upstream": "https://api.openai.com"},
        )
    )
    res = rt.resolve_upstream(protocol="openai", model="gpt-5.5", headers={})
    assert res.base_url == "https://api.openai.com"


def test_resolve_no_match_no_star_falls_back_legacy() -> None:
    rt = _runtime(
        routes=_routes(
            {"model_prefix": "glm-", "upstream": "https://ollama.com"},
        )
    )
    res = rt.resolve_upstream(protocol="openai", model="gpt-5.5", headers={})
    assert res.base_url == "https://api.openai.com"


def test_resolve_missing_model_uses_star() -> None:
    rt = _runtime(
        routes=_routes(
            {"model_prefix": "glm-", "upstream": "https://ollama.com"},
            {"model_prefix": "*", "upstream": "https://api.openai.com"},
        )
    )
    res = rt.resolve_upstream(protocol="openai", model=None, headers={})
    assert res.base_url == "https://api.openai.com"


def test_resolve_deterministic_order_first_match_wins() -> None:
    rt = _runtime(
        routes=_routes(
            {"model_prefix": "glm-5", "upstream": "https://a.example"},
            {"model_prefix": "glm-", "upstream": "https://b.example"},
        )
    )
    res = rt.resolve_upstream(protocol="openai", model="glm-5.2", headers={})
    assert res.base_url == "https://a.example"


def test_resolve_route_without_upstream_uses_protocol_slot() -> None:
    rt = _runtime(
        routes=_routes({"model_prefix": "claude-", "protocol": "anthropic"}),
        anthropic="https://my-anthropic.example",
    )
    res = rt.resolve_upstream(protocol="anthropic", model="claude-opus-4-8", headers={})
    assert res.base_url == "https://my-anthropic.example"


@pytest.mark.parametrize("handler_protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("model_prefix", ["claude-", "*"])
def test_omitted_protocol_uses_documented_openai_default(
    handler_protocol: str, model_prefix: str
) -> None:
    rt = _runtime(routes=_routes({"model_prefix": model_prefix}))
    result = rt.resolve_upstream(protocol=handler_protocol, model="claude-sonnet", headers={})
    assert result.base_url == "https://api.openai.com"


@pytest.mark.parametrize("handler_protocol", ["openai", "anthropic"])
def test_explicit_protocol_selects_target_independently_of_handler(handler_protocol: str) -> None:
    rt = _runtime(routes=_routes({"model_prefix": "*", "protocol": "anthropic"}))
    result = rt.resolve_upstream(protocol=handler_protocol, model="claude-sonnet", headers={})
    assert result.base_url == "https://api.anthropic.com"


def test_operator_can_configure_http_gateway_transport() -> None:
    routes = _routes(
        {"model_prefix": "glm-", "upstream": "http://gateway.internal:8080", "auth": "env:KEY"}
    )
    result = _runtime(routes=routes).resolve_upstream(protocol="openai", model="glm-5", headers={})
    assert result.base_url == "http://gateway.internal:8080"
    assert isinstance(result.auth, BearerAuth)


def test_resolve_case_insensitive_model() -> None:
    rt = _runtime(
        routes=_routes(
            {"model_prefix": "glm-", "upstream": "https://ollama.com"},
        )
    )
    res = rt.resolve_upstream(protocol="openai", model="GLM-5.2", headers={})
    assert res.base_url == "https://ollama.com"


def test_resolve_anthropic_auth_header_routes_to_anthropic_target() -> None:
    """Legacy fallback: anthropic x-api-key header selects anthropic target."""
    rt = _runtime(routes=())
    res = rt.resolve_upstream(
        protocol="openai", model="gpt-5.5", headers={"x-api-key": "sk-ant-xxx"}
    )
    assert res.base_url == "https://api.anthropic.com"
    assert isinstance(res.auth, PassthroughAuth)


# --- Bearer token is read per-request from the environment ---


def test_bearer_env_var_exposed_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    routes = _routes(
        {"model_prefix": "glm-", "upstream": "https://ollama.com", "auth": "env:OLLAMA_API_KEY"},
    )
    rt = _runtime(routes=routes)
    monkeypatch.setenv("OLLAMA_API_KEY", "key-one")
    res = rt.resolve_upstream(protocol="openai", model="glm-5.2", headers={})
    assert isinstance(res.auth, BearerAuth)
    assert res.auth.env_var == "OLLAMA_API_KEY"
