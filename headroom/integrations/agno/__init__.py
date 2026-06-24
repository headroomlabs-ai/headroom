"""Agno integration for Headroom SDK.

This module provides seamless integration with Agno (formerly Phidata),
enabling automatic context optimization for Agno agents.

Components:
1. HeadroomAgnoModel - Wraps any Agno model to apply Headroom transforms
2. create_headroom_hooks - Creates pre/post hooks for Agno agents
3. optimize_messages - Standalone function for manual optimization

Example:
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat
    from headroom.integrations.agno import HeadroomAgnoModel

    # Wrap any Agno model
    model = OpenAIChat(id="gpt-4o")
    optimized_model = HeadroomAgnoModel(model)

    # Use with agent
    agent = Agent(model=optimized_model)
    response = agent.run("Hello!")
"""

from .hooks import (
    HeadroomPostHook,
    HeadroomPreHook,
    HookMetrics,
    create_headroom_hooks,
)
from .model import (
    HeadroomAgnoModel,
    OptimizationMetrics,
    agno_available,
    optimize_messages,
)
from .providers import get_headroom_provider, get_model_name_from_agno

__all__ = [
    # Model wrapper
    "HeadroomAgnoModel",
    "OptimizationMetrics",
    "agno_available",
    "optimize_messages",
    # Hooks
    "create_headroom_hooks",
    "HeadroomPreHook",
    "HeadroomPostHook",
    "HookMetrics",
    # Provider detection
    "get_headroom_provider",
    "get_model_name_from_agno",
]
