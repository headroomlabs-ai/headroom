"""Agno provider integration facade."""

from __future__ import annotations

from typing import Any

from headroom.providers.agno import create_provider, default_model_name
from headroom.providers.base import Provider


def get_headroom_provider(agno_model: Any) -> Provider:
    """Get the provider-owned Headroom provider for an Agno model."""
    return create_provider(agno_model)


def get_model_name_from_agno(agno_model: Any) -> str:
    """Extract the model name/ID from an Agno model."""
    return default_model_name(agno_model)
