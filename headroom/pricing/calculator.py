"""CostCalculator: Encapsulates the 4-tier cost evaluation hierarchy.

Evaluation Hierarchy:
1. Provider-supplied cost (if present on response/outcome metadata) -> use directly.
2. User-configured custom model pricing (if present in user PricingRegistry) -> calculate cost.
3. Default pricing database (LiteLLM / built-in database) -> calculate cost.
4. Fallback -> return None (preserve existing behavior when no cost or pricing is available).
"""

from __future__ import annotations

import logging

from headroom.pricing.registry import PricingRegistry

logger = logging.getLogger("headroom.pricing.calculator")


class CostCalculator:
    """Calculates request cost and savings based on provider data and model pricing."""

    def __init__(
        self,
        custom_registry: PricingRegistry | None = None,
        cost_fallback_enabled: bool = True,
    ):
        """Initialize CostCalculator.

        Args:
            custom_registry: Optional user-configured PricingRegistry containing custom model rates.
            cost_fallback_enabled: Whether to fall back to custom/LiteLLM estimation when provider cost is missing.
        """
        self.custom_registry = custom_registry
        self.cost_fallback_enabled = cost_fallback_enabled

    def set_custom_registry(self, registry: PricingRegistry | None) -> None:
        """Update or set the user-configured PricingRegistry."""
        self.custom_registry = registry

    def calculate_request_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        provider_cost_usd: float | None = None,
    ) -> float | None:
        """Calculate the total cost in USD for a single request.

        Args:
            model: Model name/identifier.
            input_tokens: Non-cached input tokens.
            output_tokens: Output tokens.
            cache_read_tokens: Input tokens served from prompt cache.
            cache_write_tokens: Input tokens written to prompt cache.
            provider_cost_usd: Cost supplied directly by provider response, if any.

        Returns:
            Cost in USD as a float, or None if pricing is unavailable.
        """
        # Tier 1: Upstream provider supplied exact cost
        if provider_cost_usd is not None and provider_cost_usd >= 0:
            return float(provider_cost_usd)

        if not self.cost_fallback_enabled or not model:
            return None

        # Tier 2: User-configured custom pricing registry
        if self.custom_registry is not None:
            pricing = self.custom_registry.get_price(model)
            if pricing is not None:
                try:
                    estimate = self.custom_registry.estimate_cost(
                        model=pricing.model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cached_input_tokens=cache_read_tokens,
                    )
                    return float(estimate.cost_usd)
                except Exception as e:
                    logger.debug(f"Custom pricing estimation failed for model {model}: {e}")

        # Tier 3: Default LiteLLM pricing database lookup
        try:
            from headroom.pricing.litellm_pricing import (
                LITELLM_AVAILABLE,
                resolve_litellm_model,
            )

            if LITELLM_AVAILABLE:
                import litellm

                resolved_model = resolve_litellm_model(model)
                input_cost, output_cost = litellm.cost_per_token(
                    model=resolved_model,
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    cache_read_input_tokens=cache_read_tokens,
                    cache_creation_input_tokens=cache_write_tokens,
                )
                total_cost = input_cost + output_cost
                if total_cost > 0:
                    return float(total_cost)
        except Exception as e:
            logger.debug(f"LiteLLM estimation failed for model {model}: {e}")

        # Tier 4: Fallback (no pricing available)
        return None
