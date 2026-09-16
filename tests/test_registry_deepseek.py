"""Tests for the deepseek upstream target in the provider registry."""

from __future__ import annotations

from headroom.providers.registry import (
    DEFAULT_DEEPSEEK_API_URL,
    ProviderApiOverrides,
    resolve_api_overrides,
    resolve_api_targets,
)
from headroom.proxy.models import ProxyConfig


def test_resolve_api_overrides_deepseek_target_api_url() -> None:
    overrides = resolve_api_overrides(
        anthropic_api_url=None,
        openai_api_url=None,
        gemini_api_url=None,
        cloudcode_api_url=None,
        deepseek_api_url=None,
        environ={"DEEPSEEK_TARGET_API_URL": "https://deepseek.internal"},
    )
    assert overrides.deepseek == "https://deepseek.internal"


def test_resolve_api_overrides_deepseek_explicit_beats_env() -> None:
    overrides = resolve_api_overrides(
        anthropic_api_url=None,
        openai_api_url=None,
        gemini_api_url=None,
        cloudcode_api_url=None,
        deepseek_api_url="https://explicit.internal",
        environ={"DEEPSEEK_TARGET_API_URL": "https://env.internal"},
    )
    assert overrides.deepseek == "https://explicit.internal"


def test_resolve_api_targets_deepseek_default() -> None:
    targets = resolve_api_targets(ProviderApiOverrides(deepseek=None))
    assert targets.deepseek == DEFAULT_DEEPSEEK_API_URL
    assert DEFAULT_DEEPSEEK_API_URL == "https://api.deepseek.com"


def test_resolve_api_targets_deepseek_strips_v1() -> None:
    targets = resolve_api_targets(ProviderApiOverrides(deepseek="http://127.0.0.1:4000/v1"))
    assert targets.deepseek == "http://127.0.0.1:4000"


def test_proxy_config_exposes_deepseek_api_url_in_overrides() -> None:
    config = ProxyConfig(deepseek_api_url="https://deepseek.internal")
    assert config.provider_api_overrides.deepseek == "https://deepseek.internal"


def test_proxy_config_deepseek_defaults_to_none() -> None:
    config = ProxyConfig()
    assert config.deepseek_api_url is None
    assert config.provider_api_overrides.deepseek is None
