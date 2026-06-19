"""Tests for DeepSeek model pricing and cost estimation."""

from headroom.pricing.deepseek_prices import (
    DEEPSEEK_PRICES,
    get_deepseek_registry,
)
from headroom.pricing.registry import PricingRegistry


class TestDeepSeekPricingModule:
    """Tests for the DeepSeek pricing data module."""

    def test_deepseek_pricing_contains_v4_models(self):
        assert "deepseek-v4-flash" in DEEPSEEK_PRICES
        assert "deepseek-v4-pro" in DEEPSEEK_PRICES
        assert len(DEEPSEEK_PRICES) == 2

    def test_deepseek_v4_flash_pricing(self):
        pricing = DEEPSEEK_PRICES["deepseek-v4-flash"]
        assert pricing.input_per_1m == 0.14
        assert pricing.output_per_1m == 0.28
        assert pricing.cached_input_per_1m == 0.0028
        assert pricing.context_window == 1_000_000
        assert pricing.provider == "deepseek"
        assert pricing.notes is not None

    def test_deepseek_v4_pro_pricing(self):
        pricing = DEEPSEEK_PRICES["deepseek-v4-pro"]
        assert pricing.input_per_1m == 0.435
        assert pricing.output_per_1m == 0.87
        assert pricing.cached_input_per_1m == 0.003625
        assert pricing.context_window == 1_000_000
        assert pricing.provider == "deepseek"
        assert pricing.notes is not None

    def test_get_deepseek_registry(self):
        registry = get_deepseek_registry()
        assert isinstance(registry, PricingRegistry)
        assert registry.get_price("deepseek-v4-flash") is not None
        assert registry.get_price("deepseek-v4-pro") is not None
        assert registry.get_price("nonexistent") is None

    def test_registry_staleness_and_source_url(self):
        registry = get_deepseek_registry()
        assert registry.source_url == "https://api-docs.deepseek.com/quick_start/pricing"
        assert not registry.is_stale()

    def test_deepseek_registry_estimate_cost(self):
        registry = get_deepseek_registry()
        cost = registry.estimate_cost("deepseek-v4-flash", input_tokens=1_000_000)
        assert cost.cost_usd == 0.14
        assert "input" in cost.breakdown
        assert cost.pricing_date is not None

    def test_deepseek_registry_estimate_cost_with_cached(self):
        registry = get_deepseek_registry()
        cost = registry.estimate_cost(
            "deepseek-v4-flash",
            input_tokens=1_000_000,
            cached_input_tokens=1_000_000,
        )
        assert cost.cost_usd == 0.14 + 0.0028
