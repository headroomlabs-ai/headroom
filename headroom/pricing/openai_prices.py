"""OpenAI model pricing information."""

from datetime import date

from .registry import ModelPricing, PricingRegistry

# Last verified date for pricing information
LAST_UPDATED = date(2025, 1, 6)

# Official pricing page
SOURCE_URL = "https://openai.com/api/pricing/"

# All prices are in USD per 1 million tokens
OPENAI_PRICES: dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(
        model="gpt-4o",
        provider="openai",
        input_per_1m=2.50,
        output_per_1m=10.00,
        cached_input_per_1m=1.25,
        context_window=128_000,
        notes="Most capable GPT-4o model",
    ),
    "gpt-4o-mini": ModelPricing(
        model="gpt-4o-mini",
        provider="openai",
        input_per_1m=0.15,
        output_per_1m=0.60,
        cached_input_per_1m=0.075,
        context_window=128_000,
        notes="Affordable small model for fast, lightweight tasks",
    ),
    "o1": ModelPricing(
        model="o1",
        provider="openai",
        input_per_1m=15.00,
        output_per_1m=60.00,
        cached_input_per_1m=7.50,
        context_window=200_000,
        notes="Reasoning model for complex, multi-step tasks",
    ),
    "o1-mini": ModelPricing(
        model="o1-mini",
        provider="openai",
        input_per_1m=1.10,
        output_per_1m=4.40,
        cached_input_per_1m=0.55,
        context_window=128_000,
        notes="Smaller reasoning model, cost-effective for coding tasks",
    ),
    "o3-mini": ModelPricing(
        model="o3-mini",
        provider="openai",
        input_per_1m=1.10,
        output_per_1m=4.40,
        cached_input_per_1m=0.55,
        context_window=200_000,
        notes="Latest small reasoning model",
    ),
    "gpt-4-turbo": ModelPricing(
        model="gpt-4-turbo",
        provider="openai",
        input_per_1m=10.00,
        output_per_1m=30.00,
        cached_input_per_1m=5.00,
        context_window=128_000,
        notes="Previous generation GPT-4 Turbo model",
    ),
    "gpt-3.5-turbo": ModelPricing(
        model="gpt-3.5-turbo",
        provider="openai",
        input_per_1m=0.50,
        output_per_1m=1.50,
        cached_input_per_1m=0.25,
        context_window=16_385,
        notes="Fast, inexpensive model for simple tasks",
    ),
}


def get_openai_registry() -> PricingRegistry:
    """Create and return an OpenAI pricing registry.

    Returns:
        PricingRegistry configured with OpenAI model prices.
    """
    return PricingRegistry(
        last_updated=LAST_UPDATED,
        source_url=SOURCE_URL,
        prices=OPENAI_PRICES.copy(),
    )
