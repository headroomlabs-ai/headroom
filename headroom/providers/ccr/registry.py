"""CCR provider adapter registry."""

from __future__ import annotations

from .anthropic import AnthropicCcrAdapter
from .base import GenericCcrAdapter, ProviderCcrAdapter
from .google import GoogleCcrAdapter
from .openai import OpenAICcrAdapter

ANTHROPIC_CCR_ADAPTER = AnthropicCcrAdapter()
OPENAI_CCR_ADAPTER = OpenAICcrAdapter()
GOOGLE_CCR_ADAPTER = GoogleCcrAdapter()

CCR_API_URLS = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
    "google": "https://generativelanguage.googleapis.com",
}

_ADAPTERS: dict[str, ProviderCcrAdapter] = {
    "anthropic": ANTHROPIC_CCR_ADAPTER,
    "openai": OPENAI_CCR_ADAPTER,
    "google": GOOGLE_CCR_ADAPTER,
    "gemini": GOOGLE_CCR_ADAPTER,
}
_GENERIC = GenericCcrAdapter()


def get_ccr_adapter(provider: str) -> ProviderCcrAdapter:
    """Resolve a CCR adapter for a provider name."""
    return _ADAPTERS.get(provider, _GENERIC)
