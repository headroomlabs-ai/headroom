"""Anthropic model pricing information."""

from datetime import date

from .registry import ModelPricing, PricingRegistry

# Last verified date for pricing information
LAST_UPDATED = date(2025, 1, 6)

# Official pricing page
SOURCE_URL = "https://www.anthropic.com/pricing"

# All prices are in USD per 1 million tokens
ANTHROPIC_PRICES: dict[str, ModelPricing] = {
    "claude-3-5-sonnet-20241022": ModelPricing(
        model="claude-3-5-sonnet-20241022",
        provider="anthropic",
        input_per_1m=3.00,
        output_per_1m=15.00,
        cached_input_per_1m=0.30,
        batch_input_per_1m=1.50,
        batch_output_per_1m=7.50,
        context_window=200_000,
        notes="Most intelligent Claude model, best for complex tasks",
    ),
    "claude-3-5-sonnet-latest": ModelPricing(
        model="claude-3-5-sonnet-latest",
        provider="anthropic",
        input_per_1m=3.00,
        output_per_1m=15.00,
        cached_input_per_1m=0.30,
        batch_input_per_1m=1.50,
        batch_output_per_1m=7.50,
        context_window=200_000,
        notes="Alias for claude-3-5-sonnet-20241022",
    ),
    "claude-3-5-haiku-20241022": ModelPricing(
        model="claude-3-5-haiku-20241022",
        provider="anthropic",
        input_per_1m=0.80,
        output_per_1m=4.00,
        cached_input_per_1m=0.08,
        batch_input_per_1m=0.40,
        batch_output_per_1m=2.00,
        context_window=200_000,
        notes="Fast and cost-effective for simple tasks",
    ),
    "claude-3-opus-20240229": ModelPricing(
        model="claude-3-opus-20240229",
        provider="anthropic",
        input_per_1m=15.00,
        output_per_1m=75.00,
        cached_input_per_1m=1.50,
        batch_input_per_1m=7.50,
        batch_output_per_1m=37.50,
        context_window=200_000,
        notes="Previous generation powerful model for complex tasks",
    ),
    "claude-3-haiku-20240307": ModelPricing(
        model="claude-3-haiku-20240307",
        provider="anthropic",
        input_per_1m=0.25,
        output_per_1m=1.25,
        cached_input_per_1m=0.03,
        context_window=200_000,
        notes="Previous generation fastest and most compact model",
    ),
}


def get_anthropic_registry() -> PricingRegistry:
    """Create and return an Anthropic pricing registry.

    Returns:
        PricingRegistry configured with Anthropic model prices.
    """
    return PricingRegistry(
        last_updated=LAST_UPDATED,
        source_url=SOURCE_URL,
        prices=ANTHROPIC_PRICES.copy(),
    )
