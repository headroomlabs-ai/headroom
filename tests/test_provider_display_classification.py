"""Upstream-provider display classification for OpenAI-compatible endpoints.

Covers issue #1533: when ``--openai-api-url`` points at a non-OpenAI upstream
(OpenRouter, Groq, …), the dashboard should show that provider instead of
always "openai". The internal provider key stays ``openai`` so pricing and
request formatting are unaffected — only the display label changes.
"""

from __future__ import annotations

import pytest

from headroom.proxy.helpers import (
    classify_openai_upstream,
    resolve_display_provider,
)
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import _remap_provider_counts


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://api.openai.com/v1", "OpenAI"),
        ("https://openrouter.ai/api/v1", "OpenRouter"),
        ("https://api.groq.com/openai/v1", "Groq"),
        ("https://api.together.xyz/v1", "Together AI"),
        ("https://my-resource.openai.azure.com/", "Azure OpenAI"),
        ("https://api.deepseek.com/v1", "DeepSeek"),
    ],
)
def test_classify_known_hosts(url: str, expected: str) -> None:
    assert classify_openai_upstream(url) == expected


@pytest.mark.parametrize("url", [None, "", "not-a-url", "https://api.mycorp.internal/v1"])
def test_classify_unknown_or_missing_returns_none(url: str | None) -> None:
    assert classify_openai_upstream(url) is None


def test_resolve_detects_from_url() -> None:
    assert (
        resolve_display_provider("openai", openai_api_url="https://openrouter.ai/api/v1")
        == "OpenRouter"
    )


def test_resolve_explicit_name_wins_over_detection() -> None:
    assert (
        resolve_display_provider(
            "openai",
            openai_api_url="https://openrouter.ai/api/v1",
            provider_name="Internal Gateway",
        )
        == "Internal Gateway"
    )


def test_resolve_non_openai_provider_untouched() -> None:
    # Anthropic/Bedrock/etc. keep their own label regardless of openai_api_url.
    assert (
        resolve_display_provider("anthropic", openai_api_url="https://openrouter.ai/api/v1")
        == "anthropic"
    )


def test_resolve_plain_openai_unchanged() -> None:
    assert resolve_display_provider("openai") == "openai"
    # Unknown custom host with no override falls back to the raw label.
    assert (
        resolve_display_provider("openai", openai_api_url="https://api.mycorp.internal/v1")
        == "openai"
    )


def test_remap_provider_counts_relabels_only_openai() -> None:
    config = ProxyConfig(openai_api_url="https://openrouter.ai/api/v1")
    counts = {"openai": 7, "anthropic": 3}
    assert _remap_provider_counts(counts, config) == {"OpenRouter": 7, "anthropic": 3}


def test_remap_provider_counts_noop_without_custom_upstream() -> None:
    config = ProxyConfig()
    counts = {"openai": 5}
    assert _remap_provider_counts(counts, config) == {"openai": 5}
