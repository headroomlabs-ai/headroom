from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date, timedelta

import pytest

import headroom.pricing as pricing
from headroom.pricing.anthropic_prices import ANTHROPIC_PRICES, get_anthropic_registry
from headroom.pricing.openai_prices import OPENAI_PRICES, get_openai_registry
from headroom.pricing.registry import ModelPricing, PricingRegistry


def test_pricing_public_exports_and_provider_registries() -> None:
    assert pricing.ModelPricing is ModelPricing
    assert pricing.PricingRegistry is PricingRegistry
    assert "get_openai_registry" in pricing.__all__
    assert "get_anthropic_registry" in pricing.__all__
    assert "estimate_cost" in pricing.__all__

    openai_registry = get_openai_registry()
    anthropic_registry = get_anthropic_registry()
    assert openai_registry.source_url == "https://openai.com/api/pricing/"
    assert anthropic_registry.source_url == "https://www.anthropic.com/pricing"
    assert openai_registry.prices["gpt-4o"] == OPENAI_PRICES["gpt-4o"]
    assert (
        anthropic_registry.prices["claude-3-5-sonnet-20241022"]
        == ANTHROPIC_PRICES["claude-3-5-sonnet-20241022"]
    )

    openai_registry.prices.pop("gpt-4o")
    anthropic_registry.prices.pop("claude-3-5-sonnet-20241022")
    assert "gpt-4o" in OPENAI_PRICES
    assert "claude-3-5-sonnet-20241022" in ANTHROPIC_PRICES


def test_model_pricing_is_frozen() -> None:
    model = ModelPricing(model="demo", provider="test", input_per_1m=1.5, output_per_1m=2.5)
    with pytest.raises(FrozenInstanceError):
        model.model = "other"  # type: ignore[misc]


def test_registry_staleness_and_warning() -> None:
    fresh = PricingRegistry(last_updated=date.today() - timedelta(days=30))
    assert fresh.is_stale() is False
    assert fresh.staleness_warning() is None

    stale = PricingRegistry(
        last_updated=date.today() - timedelta(days=31),
        source_url="https://example.test/pricing",
    )
    assert stale.is_stale() is True
    assert stale.staleness_warning() == (
        f"Pricing data is 31 days old (last updated: {stale.last_updated})."
        " Please verify at: https://example.test/pricing"
    )


def test_registry_estimate_cost_with_all_token_types() -> None:
    registry = PricingRegistry(
        last_updated=date.today() - timedelta(days=31),
        prices={
            "demo": ModelPricing(
                model="demo",
                provider="test",
                input_per_1m=2.0,
                output_per_1m=4.0,
                cached_input_per_1m=1.0,
                batch_input_per_1m=0.5,
                batch_output_per_1m=0.25,
            )
        },
    )

    estimate = registry.estimate_cost(
        "demo",
        input_tokens=1_000_000,
        output_tokens=500_000,
        cached_input_tokens=250_000,
        batch_input_tokens=200_000,
        batch_output_tokens=100_000,
    )

    assert estimate.cost_usd == pytest.approx(4.375)
    assert estimate.breakdown == {
        "input": {"tokens": 1_000_000, "rate_per_1m": 2.0, "cost_usd": 2.0},
        "output": {"tokens": 500_000, "rate_per_1m": 4.0, "cost_usd": 2.0},
        "cached_input": {"tokens": 250_000, "rate_per_1m": 1.0, "cost_usd": 0.25},
        "batch_input": {"tokens": 200_000, "rate_per_1m": 0.5, "cost_usd": 0.1},
        "batch_output": {"tokens": 100_000, "rate_per_1m": 0.25, "cost_usd": 0.025},
    }
    assert estimate.pricing_date == registry.last_updated
    assert estimate.is_stale is True
    assert estimate.warning == (
        f"Pricing data is 31 days old (last updated: {registry.last_updated})."
    )


def test_registry_estimate_cost_zero_usage_returns_empty_breakdown() -> None:
    registry = PricingRegistry(
        last_updated=date.today(),
        prices={
            "demo": ModelPricing(model="demo", provider="test", input_per_1m=1.0, output_per_1m=2.0)
        },
    )
    estimate = registry.estimate_cost("demo")
    assert estimate.cost_usd == 0.0
    assert estimate.breakdown == {}
    assert estimate.is_stale is False
    assert estimate.warning is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "Model 'missing' not found in registry"),
        ({"cached_input_tokens": 1}, "Model 'demo' does not have cached input pricing"),
        ({"batch_input_tokens": 1}, "Model 'demo' does not have batch input pricing"),
        ({"batch_output_tokens": 1}, "Model 'demo' does not have batch output pricing"),
    ],
)
def test_registry_estimate_cost_error_paths(kwargs: dict[str, int], message: str) -> None:
    registry = PricingRegistry(
        last_updated=date.today(),
        prices={
            "demo": ModelPricing(model="demo", provider="test", input_per_1m=1.0, output_per_1m=2.0)
        },
    )
    with pytest.raises(ValueError, match=message):
        registry.estimate_cost("missing" if not kwargs else "demo", **kwargs)
