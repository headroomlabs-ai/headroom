"""Strands Agents integration for Headroom SDK.

This module provides seamless integration with Strands Agents,
enabling automatic context optimization for Strands agents.

Components:
1. HeadroomStrandsModel - Wraps any Strands model to apply Headroom transforms
2. HeadroomHookProvider - Hook provider for Strands agents
3. get_headroom_provider - Detects appropriate provider for a Strands model
4. get_model_name_from_strands - Extracts model name from a Strands model

Example:
    from strands import Agent
    from strands.models import BedrockModel
    from headroom.integrations.strands import HeadroomStrandsModel

    # Wrap any Strands model
    model = BedrockModel(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")
    optimized_model = HeadroomStrandsModel(model)

    # Use with agent
    agent = Agent(model=optimized_model)
    response = agent("Hello!")
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .bundle import HeadroomBundle
    from .hooks import HeadroomHookProvider
    from .model import HeadroomStrandsModel, OptimizationMetrics, optimize_messages
    from .providers import get_headroom_provider, get_model_name_from_strands


def strands_available() -> bool:
    """Check if strands-agents is installed and available.

    Returns:
        True if strands-agents package is available, False otherwise.
    """
    return importlib.util.find_spec("strands") is not None


# Lazy imports to avoid import errors when strands is not installed
def __getattr__(name: str) -> Any:
    """Lazy import of integration components."""
    if name == "HeadroomHookProvider":
        from .hooks import HeadroomHookProvider

        return HeadroomHookProvider
    elif name == "HeadroomStrandsModel":
        from .model import HeadroomStrandsModel

        return HeadroomStrandsModel
    elif name == "OptimizationMetrics":
        from .model import OptimizationMetrics

        return OptimizationMetrics
    elif name == "optimize_messages":
        from .model import optimize_messages

        return optimize_messages
    elif name == "get_headroom_provider":
        from .providers import get_headroom_provider

        return get_headroom_provider
    elif name == "get_model_name_from_strands":
        from .providers import get_model_name_from_strands

        return get_model_name_from_strands
    elif name == "HeadroomBundle":
        from .bundle import HeadroomBundle

        return HeadroomBundle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Availability check
    "strands_available",
    # Hook provider
    "HeadroomHookProvider",
    # Model wrapper
    "HeadroomStrandsModel",
    "OptimizationMetrics",
    "optimize_messages",
    # Provider detection
    "get_headroom_provider",
    "get_model_name_from_strands",
    # One-helper MCP + hook wiring (Headroom + Serena + RTK-equivalent)
    "HeadroomBundle",
]
