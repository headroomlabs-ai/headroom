"""Tests for DeepSeek model pricing and cost estimation."""

import pytest

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


class TestDeepSeekLiteLLMInjection:
    """Tests for DeepSeek V4 pricing injection into litellm."""

    def test_deepseek_v4_models_in_litellm_model_cost(self):
        from headroom.pricing.litellm_pricing import LITELLM_AVAILABLE, litellm

        if not LITELLM_AVAILABLE:
            pytest.skip("litellm not available")
        assert "deepseek-v4-flash" in litellm.model_cost
        assert "deepseek-v4-pro" in litellm.model_cost

    def test_deepseek_v4_prefixed_models_in_litellm_model_cost(self):
        from headroom.pricing.litellm_pricing import LITELLM_AVAILABLE, litellm

        if not LITELLM_AVAILABLE:
            pytest.skip("litellm not available")
        assert "deepseek/deepseek-v4-flash" in litellm.model_cost
        assert "deepseek/deepseek-v4-pro" in litellm.model_cost

    def test_deepseek_v4_flash_litellm_pricing(self):
        from headroom.pricing.litellm_pricing import LITELLM_AVAILABLE, litellm

        if not LITELLM_AVAILABLE:
            pytest.skip("litellm not available")
        flash = litellm.model_cost["deepseek-v4-flash"]
        assert flash["input_cost_per_token"] == 0.14 / 1_000_000
        assert flash["output_cost_per_token"] == 0.28 / 1_000_000
        assert flash["cache_read_input_token_cost"] == 0.0028 / 1_000_000
        assert flash["litellm_provider"] == "deepseek"

    def test_deepseek_v4_pro_litellm_pricing(self):
        from headroom.pricing.litellm_pricing import LITELLM_AVAILABLE, litellm

        if not LITELLM_AVAILABLE:
            pytest.skip("litellm not available")
        pro = litellm.model_cost["deepseek-v4-pro"]
        assert pro["input_cost_per_token"] == 0.435 / 1_000_000
        assert pro["output_cost_per_token"] == 0.87 / 1_000_000
        assert pro["cache_read_input_token_cost"] == 0.003625 / 1_000_000
        assert pro["litellm_provider"] == "deepseek"

    def test_cost_per_token_resolves_deepseek_v4_flash(self):
        from headroom.pricing.litellm_pricing import (
            LITELLM_AVAILABLE,
            litellm,
            resolve_litellm_model,
        )

        if not LITELLM_AVAILABLE:
            pytest.skip("litellm not available")
        # resolve_litellm_model adds the deepseek/ prefix so that
        # litellm.cost_per_token can determine the provider via
        # get_llm_provider(). Bare model names without a provider prefix
        # would fail with BadRequestError.
        resolved = resolve_litellm_model("deepseek-v4-flash")
        input_cost, output_cost = litellm.cost_per_token(
            model=resolved,
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        )
        assert input_cost == pytest.approx(0.14, rel=0.01)
        assert output_cost == pytest.approx(0.28, rel=0.01)

    def test_resolve_litellm_model_prefixes_deepseek(self):
        from headroom.pricing.litellm_pricing import resolve_litellm_model

        resolved = resolve_litellm_model("deepseek-v4-flash")
        assert resolved == "deepseek/deepseek-v4-flash"

    def test_injection_does_not_overwrite_existing_upstream_entries(self):
        """If litellm upstream already has these, our injection is a no-op."""
        from headroom.pricing.litellm_pricing import LITELLM_AVAILABLE, litellm

        if not LITELLM_AVAILABLE:
            pytest.skip("litellm not available")
        # Force-inject with wrong value, then verify the injection guard
        litellm.model_cost["deepseek-v4-flash"] = {"input_cost_per_token": 999}
        # Reimport to trigger _inject_deepseek_pricing — but it should NOT overwrite
        import importlib

        import headroom.pricing.litellm_pricing as lp

        importlib.reload(lp)
        assert litellm.model_cost["deepseek-v4-flash"]["input_cost_per_token"] == 999
        # Reset to correct value
        litellm.model_cost["deepseek-v4-flash"] = {
            "input_cost_per_token": 0.14 / 1_000_000,
            "output_cost_per_token": 0.28 / 1_000_000,
            "cache_read_input_token_cost": 0.0028 / 1_000_000,
            "litellm_provider": "deepseek",
            "max_tokens": 384_000,
            "max_input_tokens": 1_000_000,
        }


class TestDeepSeekAnthropicProviderFallback:
    """Tests that Anthropic provider's _get_pricing handles DeepSeek models."""

    def test_deepseek_v4_flash_fallback(self):
        from headroom.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        pricing = provider._get_pricing("deepseek-v4-flash")
        assert pricing is not None
        assert pricing["input"] == 0.14
        assert pricing["output"] == 0.28
        assert pricing["cached_input"] == 0.0028

    def test_deepseek_v4_pro_fallback(self):
        from headroom.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        pricing = provider._get_pricing("deepseek-v4-pro")
        assert pricing is not None
        assert pricing["input"] == 0.435
        assert pricing["output"] == 0.87
        assert pricing["cached_input"] == 0.003625

    def test_deepseek_unknown_model_returns_none(self):
        from headroom.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        pricing = provider._get_pricing("deepseek-unknown-model")
        assert pricing is None

    def test_deepseek_partial_match_v4_flash_alias(self):
        from headroom.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        # Should match via partial match (flash in v4-flash)
        pricing = provider._get_pricing("deepseek-v4-flash-v1")
        assert pricing is not None

    def test_estimate_cost_deepseek_v4_flash(self):
        from headroom.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        cost = provider.estimate_cost(
            input_tokens=1_000_000,
            output_tokens=0,
            model="deepseek-v4-flash",
        )
        assert cost is not None
        assert cost == 0.14

    def test_estimate_cost_deepseek_v4_flash_with_cache(self):
        from headroom.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        cost = provider.estimate_cost(
            input_tokens=1_000_000,
            output_tokens=0,
            model="deepseek-v4-flash",
            cached_tokens=1_000_000,
        )
        assert cost is not None
        # All 1M input tokens are cached, so cost = cached_input only
        assert cost == 0.0028
