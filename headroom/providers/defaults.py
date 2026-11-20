"""Provider-owned default provider factories."""

from __future__ import annotations

from headroom.providers.base import Provider
from headroom.providers.openai import OpenAIProvider

DEFAULT_TOKEN_MODEL = "gpt-4o"


def create_default_provider() -> Provider:
    """Create Headroom's default provider for generic token counting."""
    return OpenAIProvider()
