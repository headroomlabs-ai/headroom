"""Strands provider integration facade."""

from __future__ import annotations

from typing import Any

from headroom.providers.base import Provider
from headroom.providers.strands import (
    create_provider,
    default_model_name,
)


def get_headroom_provider(model: Any) -> Provider:
    """Get the provider-owned Headroom provider for a Strands model."""
    return create_provider(model)


def get_model_name_from_strands(model: Any) -> str:
    """Extract the model name/ID from a Strands model."""
    return default_model_name(model)
