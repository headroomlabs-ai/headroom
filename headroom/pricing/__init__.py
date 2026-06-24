"""Pricing module for LLM cost estimation.

This module provides pricing information and cost estimation utilities
for various LLM providers. Uses LiteLLM's community-maintained pricing
database for up-to-date costs across 100+ models.
"""

# Legacy imports for backwards compatibility
from .anthropic_prices import (
    ANTHROPIC_PRICES,
    get_anthropic_registry,
)
from .anthropic_prices import (
    LAST_UPDATED as ANTHROPIC_LAST_UPDATED,
)
from .litellm_pricing import (
    LiteLLMModelPricing,
    estimate_cost,
    get_litellm_model_cost,
    get_model_pricing,
    list_available_models,
)
from .openai_prices import (
    LAST_UPDATED as OPENAI_LAST_UPDATED,
)
from .openai_prices import (
    OPENAI_PRICES,
    get_openai_registry,
)
from .registry import CostEstimate, ModelPricing, PricingRegistry

__all__ = [
    # LiteLLM-based pricing (preferred)
    "LiteLLMModelPricing",
    "estimate_cost",
    "get_litellm_model_cost",
    "get_model_pricing",
    "list_available_models",
    # Core classes
    "CostEstimate",
    "ModelPricing",
    "PricingRegistry",
    # Legacy - OpenAI (deprecated, use LiteLLM instead)
    "OPENAI_LAST_UPDATED",
    "OPENAI_PRICES",
    "get_openai_registry",
    # Legacy - Anthropic (deprecated, use LiteLLM instead)
    "ANTHROPIC_LAST_UPDATED",
    "ANTHROPIC_PRICES",
    "get_anthropic_registry",
]
