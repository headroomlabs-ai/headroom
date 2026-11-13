from __future__ import annotations

from headroom.providers.litellm_models import (
    LITELLM_PROVIDER_PREFIXES,
    resolve_litellm_model_with_probe,
)


def test_litellm_provider_prefix_policy_is_declarative() -> None:
    assert [(rule.model_prefix, rule.provider_prefix) for rule in LITELLM_PROVIDER_PREFIXES] == [
        ("claude-", "anthropic/"),
        ("gpt-", "openai/"),
        ("o1-", "openai/"),
        ("o3-", "openai/"),
        ("o4-", "openai/"),
        ("gemini-", "google/"),
    ]


def test_resolve_litellm_model_uses_exact_match_before_provider_prefix() -> None:
    seen: list[str] = []

    def supports_model(model: str) -> bool:
        seen.append(model)
        return model == "gpt-4o"

    assert resolve_litellm_model_with_probe("gpt-4o", supports_model) == "gpt-4o"
    assert seen == ["gpt-4o"]


def test_resolve_litellm_model_uses_first_matching_provider_prefix() -> None:
    seen: list[str] = []

    def supports_model(model: str) -> bool:
        seen.append(model)
        return model == "anthropic/claude-sonnet-4-6"

    assert (
        resolve_litellm_model_with_probe("claude-sonnet-4-6", supports_model)
        == "anthropic/claude-sonnet-4-6"
    )
    assert seen == ["claude-sonnet-4-6", "anthropic/claude-sonnet-4-6"]


def test_resolve_litellm_model_returns_original_when_no_probe_matches() -> None:
    assert resolve_litellm_model_with_probe("mystery-model", lambda model: False) == "mystery-model"
