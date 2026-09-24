"""Unit tests for custom model pricing, CostCalculator fallback hierarchy, and provider cost extraction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from headroom.pricing.calculator import CostCalculator
from headroom.pricing.litellm_pricing import LITELLM_AVAILABLE
from headroom.pricing.registry import ModelPricing, PricingRegistry
from headroom.proxy.cost import CostTracker
from headroom.proxy.helpers import extract_provider_cost


def test_provider_returns_cost_overrides_estimation() -> None:
    """Test Requirement: Provider returns cost -> exact provider cost is returned."""
    custom_pricing = {
        "gpt-4o": {
            "input_per_1m": 10.0,
            "output_per_1m": 30.0,
        }
    }
    registry = PricingRegistry.from_dict(custom_pricing)
    calc = CostCalculator(custom_registry=registry)

    # Provider supplied exact cost $0.0521
    cost = calc.calculate_request_cost(
        model="gpt-4o",
        input_tokens=100_000,
        output_tokens=50_000,
        provider_cost_usd=0.0521,
    )
    assert cost == 0.0521


def test_provider_no_cost_custom_pricing_exists() -> None:
    """Test Requirement: Provider returns no cost and custom pricing exists -> estimated using custom pricing."""
    custom_pricing = {
        "custom-internal-model": ModelPricing(
            model="custom-internal-model",
            input_per_1m=5.0,
            output_per_1m=15.0,
        )
    }
    registry = PricingRegistry.from_dict(custom_pricing)
    calc = CostCalculator(custom_registry=registry)

    # 100k input ($0.50) + 20k output ($0.30) = $0.80
    cost = calc.calculate_request_cost(
        model="custom-internal-model",
        input_tokens=100_000,
        output_tokens=20_000,
        provider_cost_usd=None,
    )
    assert cost == pytest.approx(0.80)


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not installed in environment")
def test_provider_no_cost_fallback_to_litellm() -> None:
    """Test Requirement: Provider returns no cost, no custom pricing, but model exists in default LiteLLM database."""
    calc = CostCalculator(custom_registry=None)

    # gpt-4o exists in LiteLLM DB
    cost = calc.calculate_request_cost(
        model="gpt-4o",
        input_tokens=1_000,
        output_tokens=1_000,
        provider_cost_usd=None,
    )
    assert cost is not None
    assert cost > 0


def test_litellm_tier3_fallback_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Tier 3 LiteLLM fallback using monkeypatched litellm module for deterministic coverage across Python versions."""
    import sys

    fake_litellm = SimpleNamespace(
        cost_per_token=lambda model, prompt_tokens, completion_tokens, **kwargs: (0.0025, 0.01)
    )
    monkeypatch.setattr("headroom.pricing.litellm_pricing.LITELLM_AVAILABLE", True)
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    calc = CostCalculator(custom_registry=None)
    cost = calc.calculate_request_cost(
        model="gpt-4o",
        input_tokens=1_000,
        output_tokens=1_000,
        provider_cost_usd=None,
    )
    assert cost == pytest.approx(0.0125)


def test_unknown_model_returns_none() -> None:
    """Test Requirement: Unknown model with no pricing -> returns None."""
    calc = CostCalculator(custom_registry=None)

    cost = calc.calculate_request_cost(
        model="totally-unknown-model-xyz-12345",
        input_tokens=1_000,
        output_tokens=1_000,
        provider_cost_usd=None,
    )
    assert cost is None


def test_cost_fallback_disabled() -> None:
    """Test requirement: cost_fallback_enabled=False suppresses estimation when provider cost is missing."""
    custom_pricing = {
        "gpt-4o": {
            "input_per_1m": 10.0,
            "output_per_1m": 30.0,
        }
    }
    registry = PricingRegistry.from_dict(custom_pricing)
    calc = CostCalculator(custom_registry=registry, cost_fallback_enabled=False)

    # Missing provider cost with fallback disabled -> returns None
    assert (
        calc.calculate_request_cost(
            model="gpt-4o",
            input_tokens=100_000,
            output_tokens=50_000,
            provider_cost_usd=None,
        )
        is None
    )

    # Explicit provider cost with fallback disabled -> returns provider cost
    assert (
        calc.calculate_request_cost(
            model="gpt-4o",
            input_tokens=100_000,
            output_tokens=50_000,
            provider_cost_usd=0.05,
        )
        == 0.05
    )


def test_partial_pricing_configuration() -> None:
    """Test Requirement: Partial pricing configuration (e.g., input/output set, cached input missing)."""
    custom_pricing = {
        "partial-model": {
            "model": "partial-model",
            "input_per_1m": 2.0,
            "output_per_1m": 4.0,
            # cached_input_per_1m is None
        }
    }
    registry = PricingRegistry.from_dict(custom_pricing)
    calc = CostCalculator(custom_registry=registry)

    # Without cached tokens: 50k input ($0.10) + 10k output ($0.04) = $0.14
    cost = calc.calculate_request_cost(
        model="partial-model",
        input_tokens=50_000,
        output_tokens=10_000,
    )
    assert cost == pytest.approx(0.14)


def test_existing_behavior_unchanged_when_unconfigured() -> None:
    """Test Requirement: Existing behavior remains unchanged when no custom pricing is configured."""
    tracker = CostTracker()
    assert tracker.calculator.custom_registry is None
    assert tracker.cost_fallback_enabled is True


def test_cost_tracker_respects_disabled_fallback_with_custom_pricing() -> None:
    """Proxy CostTracker must pass HEADROOM_COST_FALLBACK_ENABLED through."""
    tracker = CostTracker(
        custom_pricing={
            "custom-internal-model": {
                "input_per_1m": 5.0,
                "output_per_1m": 15.0,
            }
        },
        cost_fallback_enabled=False,
    )

    tracker.record_tokens(
        model="custom-internal-model",
        tokens_saved=0,
        tokens_sent=100_000,
        uncached_tokens=100_000,
        output_tokens=20_000,
    )

    assert tracker.get_period_cost() == 0.0
    assert (
        tracker.estimate_cost(
            model="custom-internal-model",
            input_tokens=100_000,
            output_tokens=20_000,
        )
        is None
    )


def test_extract_provider_cost_headers_and_payload() -> None:
    """Test helper for extracting provider-supplied cost from headers or JSON payload."""
    # 1. From standard header
    headers = {"X-Request-Cost-USD": "0.0123"}
    assert extract_provider_cost(headers=headers) == 0.0123

    # 2. From Portkey header
    headers_portkey = {"x-portkey-cost": "0.0456"}
    assert extract_provider_cost(headers=headers_portkey) == 0.0456

    # 3. From payload usage field
    payload = {"usage": {"cost": 0.0789}}
    assert extract_provider_cost(payload=payload) == 0.0789

    # 4. Invalid/negative values ignored
    headers_invalid = {"x-cost-usd": "not-a-number"}
    assert extract_provider_cost(headers=headers_invalid) is None

    headers_negative = {"x-cost-usd": "-1.5"}
    assert extract_provider_cost(headers=headers_negative) is None


def test_cost_tracker_with_provider_cost_and_custom_pricing() -> None:
    """Test CostTracker recording and stats calculation with custom model pricing."""
    custom_pricing = {
        "my-gateway/custom-claude": {
            "input_per_1m": 3.0,
            "output_per_1m": 15.0,
            "cached_input_per_1m": 0.30,
        }
    }

    tracker = CostTracker(custom_pricing=custom_pricing)

    # Record request 1: Provider supplied cost directly ($0.05)
    tracker.record_tokens(
        model="my-gateway/custom-claude",
        tokens_saved=50_000,
        tokens_sent=50_000,
        uncached_tokens=50_000,
        output_tokens=10_000,
        provider_cost_usd=0.05,
    )

    stats = tracker.stats()
    assert stats["total_tokens_saved"] == 50_000
    # 50,000 saved tokens at $3.00/1M = $0.15 saved
    assert stats["savings_usd"] == pytest.approx(0.15)


def test_provider_cost_without_usage_is_measured_budget_basis() -> None:
    """An exact upstream dollar cost must not be labeled as a token estimate."""
    tracker = CostTracker()

    tracker.record_tokens(
        model="gateway/custom-model",
        tokens_saved=0,
        tokens_sent=1_000,
        provider_cost_usd=0.25,
    )

    breakdown = tracker.period_cost_breakdown()
    assert breakdown["measured_usd"] == pytest.approx(0.25)
    assert breakdown["estimated_usd"] == 0.0
