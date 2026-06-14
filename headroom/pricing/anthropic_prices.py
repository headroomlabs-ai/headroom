"""Anthropic model pricing information."""

from datetime import date

from .registry import ModelPricing, PricingRegistry

# Last verified date for pricing information
LAST_UPDATED = date(2026, 6, 14)

# Official pricing page
SOURCE_URL = "https://www.anthropic.com/pricing"


def _p(
    model: str,
    input_per_1m: float,
    output_per_1m: float,
    cached_input_per_1m: float,
    context_window: int,
    notes: str = "",
) -> ModelPricing:
    """Build a ModelPricing entry; batch prices default to 50% of base."""
    return ModelPricing(
        model=model,
        provider="anthropic",
        input_per_1m=input_per_1m,
        output_per_1m=output_per_1m,
        cached_input_per_1m=cached_input_per_1m,
        batch_input_per_1m=round(input_per_1m * 0.5, 4),
        batch_output_per_1m=round(output_per_1m * 0.5, 4),
        context_window=context_window,
        notes=notes,
    )


# All prices are in USD per 1 million tokens.
# Context windows mirror headroom/providers/anthropic.py ANTHROPIC_CONTEXT_LIMITS:
# Opus 4.6-4.8 and Sonnet 4.6 ship a 1M-context tier (the "[1m]" model id Claude
# Code sends); Opus <=4.5, Sonnet <=4.5 and Haiku stay at 200K. Keep bare AND
# "[1m]"-suffixed keys because PricingRegistry.get_price is an exact-match lookup.
_FRONTIER_1M = 1_000_000
_STD = 200_000
ANTHROPIC_PRICES: dict[str, ModelPricing] = {
    # Frontier (Fable / Mythos tier — $10 / $50, 1M context)
    "claude-fable-5": _p("claude-fable-5", 10.00, 50.00, 1.00, _FRONTIER_1M, "Frontier tier"),
    "claude-mythos-5": _p("claude-mythos-5", 10.00, 50.00, 1.00, _FRONTIER_1M, "Frontier tier"),
    "claude-mythos-preview": _p(
        "claude-mythos-preview", 10.00, 50.00, 1.00, _FRONTIER_1M, "Frontier preview"
    ),
    # Opus 4.6 / 4.7 / 4.8 (current Opus tier — $5 / $25, 1M context)
    "claude-opus-4-8": _p("claude-opus-4-8", 5.00, 25.00, 0.50, _FRONTIER_1M, "Current Opus"),
    "claude-opus-4-8[1m]": _p(
        "claude-opus-4-8[1m]", 5.00, 25.00, 0.50, _FRONTIER_1M, "Current Opus, 1M tier"
    ),
    "claude-opus-4-7": _p("claude-opus-4-7", 5.00, 25.00, 0.50, _FRONTIER_1M, "Opus 4.7"),
    "claude-opus-4-7[1m]": _p(
        "claude-opus-4-7[1m]", 5.00, 25.00, 0.50, _FRONTIER_1M, "Opus 4.7, 1M tier"
    ),
    "claude-opus-4-6": _p("claude-opus-4-6", 5.00, 25.00, 0.50, _FRONTIER_1M, "Opus 4.6"),
    "claude-opus-4-6[1m]": _p(
        "claude-opus-4-6[1m]", 5.00, 25.00, 0.50, _FRONTIER_1M, "Opus 4.6, 1M tier"
    ),
    # Opus 4.5 (current pricing, 200K context)
    "claude-opus-4-5": _p("claude-opus-4-5", 5.00, 25.00, 0.50, _STD, "Opus 4.5"),
    "claude-opus-4-5-20251101": _p("claude-opus-4-5-20251101", 5.00, 25.00, 0.50, _STD, "Opus 4.5"),
    # Opus 4.1 / 4.0 (legacy Opus tier — $15 / $75, 200K context)
    "claude-opus-4-1": _p("claude-opus-4-1", 15.00, 75.00, 1.50, _STD, "Legacy Opus"),
    "claude-opus-4-1-20250805": _p(
        "claude-opus-4-1-20250805", 15.00, 75.00, 1.50, _STD, "Legacy Opus"
    ),
    "claude-opus-4-0": _p("claude-opus-4-0", 15.00, 75.00, 1.50, _STD, "Legacy Opus"),
    "claude-opus-4-20250514": _p("claude-opus-4-20250514", 15.00, 75.00, 1.50, _STD, "Legacy Opus"),
    # Sonnet 4.6 (1M context) / 4.5 / 4 (200K) — $3 / $15
    "claude-sonnet-4-6": _p("claude-sonnet-4-6", 3.00, 15.00, 0.30, _FRONTIER_1M, "Sonnet 4.6"),
    "claude-sonnet-4-6[1m]": _p(
        "claude-sonnet-4-6[1m]", 3.00, 15.00, 0.30, _FRONTIER_1M, "Sonnet 4.6, 1M tier"
    ),
    "claude-sonnet-4-5": _p("claude-sonnet-4-5", 3.00, 15.00, 0.30, _STD, "Sonnet 4.5"),
    "claude-sonnet-4-5-20250929": _p(
        "claude-sonnet-4-5-20250929", 3.00, 15.00, 0.30, _STD, "Sonnet 4.5"
    ),
    "claude-sonnet-4-0": _p("claude-sonnet-4-0", 3.00, 15.00, 0.30, _STD, "Sonnet 4"),
    "claude-sonnet-4-20250514": _p("claude-sonnet-4-20250514", 3.00, 15.00, 0.30, _STD, "Sonnet 4"),
    # Haiku 4.5 ($1 / $5, 200K context)
    "claude-haiku-4-5": _p("claude-haiku-4-5", 1.00, 5.00, 0.10, _STD, "Haiku 4.5"),
    "claude-haiku-4-5-20251001": _p("claude-haiku-4-5-20251001", 1.00, 5.00, 0.10, _STD, "Haiku 4.5"),
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
