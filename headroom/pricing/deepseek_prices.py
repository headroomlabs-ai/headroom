"""DeepSeek model pricing information."""

from datetime import date

from .registry import ModelPricing, PricingRegistry

# Last verified date for pricing information
LAST_UPDATED = date(2026, 6, 19)

# Official pricing page
SOURCE_URL = "https://api-docs.deepseek.com/quick_start/pricing"

# All prices are in USD per 1 million tokens
DEEPSEEK_PRICES: dict[str, ModelPricing] = {
    "deepseek-v4-flash": ModelPricing(
        model="deepseek-v4-flash",
        provider="deepseek",
        input_per_1m=0.14,
        output_per_1m=0.28,
        cached_input_per_1m=0.0028,
        context_window=1_000_000,
        notes="DeepSeek V4 Flash - 13B active params; non-thinking + thinking modes",
    ),
    "deepseek-v4-pro": ModelPricing(
        model="deepseek-v4-pro",
        provider="deepseek",
        input_per_1m=0.435,
        output_per_1m=0.87,
        cached_input_per_1m=0.003625,
        context_window=1_000_000,
        notes="DeepSeek V4 Pro - 49B active params",
    ),
}


def get_deepseek_registry() -> PricingRegistry:
    """Create and return a DeepSeek pricing registry.

    Returns:
        PricingRegistry configured with DeepSeek model prices.
    """
    return PricingRegistry(
        last_updated=LAST_UPDATED,
        source_url=SOURCE_URL,
        prices=DEEPSEEK_PRICES.copy(),
    )
