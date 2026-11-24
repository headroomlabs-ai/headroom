"""Provider-owned LiteLLM model resolution policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LiteLLMProviderPrefix:
    """Declarative provider prefix rule for bare model names."""

    model_prefix: str
    provider_prefix: str

    def candidate_for(self, model: str) -> str | None:
        if model.startswith(self.model_prefix):
            return f"{self.provider_prefix}{model}"
        return None


LITELLM_PROVIDER_PREFIXES: tuple[LiteLLMProviderPrefix, ...] = (
    LiteLLMProviderPrefix("claude-", "anthropic/"),
    LiteLLMProviderPrefix("gpt-", "openai/"),
    LiteLLMProviderPrefix("o1-", "openai/"),
    LiteLLMProviderPrefix("o3-", "openai/"),
    LiteLLMProviderPrefix("o4-", "openai/"),
    LiteLLMProviderPrefix("gemini-", "google/"),
)


def resolve_litellm_model_with_probe(
    model: str,
    supports_model: Callable[[str], bool],
) -> str:
    """Resolve a model name by probing LiteLLM-compatible provider prefixes."""
    if supports_model(model):
        return model

    for rule in LITELLM_PROVIDER_PREFIXES:
        candidate = rule.candidate_for(model)
        if candidate is None:
            continue
        if supports_model(candidate):
            return candidate
        break

    return model


__all__ = [
    "LITELLM_PROVIDER_PREFIXES",
    "LiteLLMProviderPrefix",
    "resolve_litellm_model_with_probe",
]
