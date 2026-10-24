"""LangChain provider integration facade."""

from __future__ import annotations

from typing import Any

from headroom.providers.base import Provider
from headroom.providers.langchain import (
    create_provider,
    default_model_name,
    detect_provider,
)

__all__ = ["detect_provider", "get_headroom_provider", "get_model_name_from_langchain"]


def get_headroom_provider(model: Any) -> Provider:
    """Get the provider-owned Headroom provider for a LangChain model."""
    return create_provider(model)


def get_model_name_from_langchain(model: Any) -> str:
    """Extract the model name string from a LangChain model."""
    return default_model_name(model)
