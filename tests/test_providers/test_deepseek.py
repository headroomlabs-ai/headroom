"""Tests for DeepSeek model pricing and cost estimation."""

from datetime import datetime, timezone

import pytest

from headroom.pricing.deepseek_prices import (
    DEEPSEEK_PRICES,
    get_deepseek_registry,
)
from headroom.pricing.registry import ModelPricing, PricingRegistry

# Monday 2026-08-17: 02:00 UTC = 10:00 Beijing (peak), 12:00 UTC = 20:00 Beijing (off-peak).
PEAK = datetime(2026, 8, 17, 2, 0, tzinfo=timezone.utc)
OFF_PEAK = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)


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

    def test_registry_source_url(self):
        registry = get_deepseek_registry()
        assert registry.source_url == "https://api-docs.deepseek.com/quick_start/pricing"
        # Deliberately no `assert not registry.is_stale()` here: is_stale()
        # compares the shipped LAST_UPDATED against date.today(), so asserting
        # freshness makes this test fail on wall-clock time alone once the
        # pricing date ages past STALENESS_THRESHOLD_DAYS (30) - which then
        # breaks CI on every unrelated PR in the repo. The staleness mechanism
        # is covered time-independently in tests/test_pricing.py, and the
        # sibling Anthropic/OpenAI registries make no freshness assertion.

    def test_deepseek_registry_estimate_cost(self):
        registry = get_deepseek_registry()
        cost = registry.estimate_cost("deepseek-flash", input_tokens=1_000_000, now=OFF_PEAK)
        assert cost.cost_usd == pytest.approx(0.15)
        assert "input" in cost.breakdown
        assert cost.pricing_date is not None

    def test_deepseek_registry_estimate_cost_with_cached(self):
        registry = get_deepseek_registry()
        cost = registry.estimate_cost(
            "deepseek-flash",
            input_tokens=1_000_000,
            cached_input_tokens=1_000_000,
            now=OFF_PEAK,
        )
        assert cost.cost_usd == pytest.approx(0.15 + 0.003)

    def test_registry_prices_the_peak_tier_at_a_peak_instant(self):
        registry = get_deepseek_registry()
        peak = registry.estimate_cost("deepseek-flash", input_tokens=1_000_000, now=PEAK)
        off = registry.estimate_cost("deepseek-flash", input_tokens=1_000_000, now=OFF_PEAK)
        assert off.cost_usd == pytest.approx(0.15)
        assert peak.cost_usd == pytest.approx(0.30)
        assert peak.breakdown["tier"] == "peak"
        assert off.breakdown["tier"] == "off_peak"
        assert peak.breakdown["input"]["rate_per_1m"] == pytest.approx(0.30)
        assert off.breakdown["input"]["rate_per_1m"] == pytest.approx(0.15)

    def test_registry_cached_bucket_uses_the_tier_cache_hit_rate(self):
        registry = get_deepseek_registry()
        cost = registry.estimate_cost(
            "deepseek-flash",
            cached_input_tokens=1_000_000,
            now=OFF_PEAK,
        )
        assert cost.cost_usd == pytest.approx(0.003)
        assert cost.breakdown["cached_input"]["rate_per_1m"] == pytest.approx(0.003)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"batch_input_tokens": 1}, "batch input pricing"),
            ({"batch_output_tokens": 1}, "batch output pricing"),
        ],
    )
    def test_registry_rejects_batch_tokens_for_deepseek(self, kwargs, match):
        registry = get_deepseek_registry()
        with pytest.raises(ValueError, match=match):
            registry.estimate_cost("deepseek-flash", now=OFF_PEAK, **kwargs)

    def test_tiered_and_flat_paths_carry_the_same_estimate_metadata(self):
        # A tagged id is out of tier scope, so it is priced by the flat body; the
        # two paths must not drift on how they build the CostEstimate.
        registry = get_deepseek_registry()
        registry.prices["deepseek/deepseek-v4-pro:free"] = DEEPSEEK_PRICES["deepseek-v4-pro"]

        tiered = registry.estimate_cost("deepseek-flash", input_tokens=1_000_000, now=OFF_PEAK)
        flat = registry.estimate_cost(
            "deepseek/deepseek-v4-pro:free", input_tokens=1_000_000, now=OFF_PEAK
        )
        assert (tiered.pricing_date, tiered.is_stale, tiered.warning) == (
            flat.pricing_date,
            flat.is_stale,
            flat.warning,
        )

    def test_registry_ignores_a_flat_row_for_a_tiered_id(self):
        # The tier table is authoritative for flash/pro, so a stale or custom flat
        # row for one of those ids must not be able to reprice it (the documented
        # contract on estimate_cost).
        registry = get_deepseek_registry()
        registry.prices["deepseek-flash"] = ModelPricing(
            model="deepseek-flash",
            provider="deepseek",
            input_per_1m=999.0,
            output_per_1m=999.0,
        )

        cost = registry.estimate_cost("deepseek-flash", input_tokens=1_000_000, now=OFF_PEAK)

        assert cost.cost_usd == pytest.approx(0.15)
        assert cost.breakdown["input"]["rate_per_1m"] == pytest.approx(0.15)
        assert cost.breakdown["tier"] == "off_peak"


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
        assert flash["input_cost_per_token"] > 0
        assert flash["output_cost_per_token"] > 0
        assert flash["litellm_provider"] == "deepseek"

    def test_deepseek_v4_pro_litellm_pricing(self):
        from headroom.pricing.litellm_pricing import LITELLM_AVAILABLE, litellm

        if not LITELLM_AVAILABLE:
            pytest.skip("litellm not available")
        pro = litellm.model_cost["deepseek-v4-pro"]
        assert pro["input_cost_per_token"] > 0
        assert pro["output_cost_per_token"] > 0
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
        active_pricing = litellm.model_cost["deepseek-v4-flash"]
        assert input_cost == pytest.approx(
            active_pricing["input_cost_per_token"] * 1_000_000,
        )
        assert output_cost == pytest.approx(
            active_pricing["output_cost_per_token"] * 1_000_000,
        )

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

    def test_deepseek_flash_fallback_rates_are_off_peak(self):
        from headroom.providers.anthropic import AnthropicProvider

        pricing = AnthropicProvider()._get_pricing("deepseek-flash")
        assert pricing is not None
        assert pricing["input"] == 0.15
        assert pricing["output"] == 0.60
        assert pricing["cached_input"] == 0.003

    def test_deepseek_v4_pro_fallback_rates_are_off_peak(self):
        from headroom.providers.anthropic import AnthropicProvider

        pricing = AnthropicProvider()._get_pricing("deepseek-v4-pro")
        assert pricing is not None
        assert pricing["input"] == 0.66
        assert pricing["output"] == 1.98
        assert pricing["cached_input"] == 0.022

    def test_retired_flash_id_shares_the_flash_rates(self):
        from headroom.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        legacy = provider._get_pricing("deepseek-v4-flash")
        current = provider._get_pricing("deepseek-flash")
        assert current is not None
        assert legacy == current
        assert provider._get_pricing("deepseek-v4-flash-vision-exp") == current

    def test_deepseek_unknown_model_returns_none(self):
        from headroom.providers.anthropic import AnthropicProvider

        assert AnthropicProvider()._get_pricing("deepseek-unknown-model") is None

    def test_deepseek_partial_match_v4_flash_alias(self):
        from headroom.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        # Should match via partial match (flash in v4-flash)
        pricing = provider._get_pricing("deepseek-v4-flash-v1")
        assert pricing is not None

    def test_estimate_cost_prices_the_off_peak_tier(self):
        from headroom.providers.anthropic import AnthropicProvider

        cost = AnthropicProvider().estimate_cost(
            input_tokens=1_000_000,
            output_tokens=0,
            model="deepseek-flash",
            now=OFF_PEAK,
        )
        assert cost == pytest.approx(0.15)

    def test_estimate_cost_prices_the_peak_tier(self):
        from headroom.providers.anthropic import AnthropicProvider

        cost = AnthropicProvider().estimate_cost(
            input_tokens=1_000_000,
            output_tokens=0,
            model="deepseek-flash",
            now=PEAK,
        )
        assert cost == pytest.approx(0.30)

    def test_estimate_cost_cached_tokens_use_the_cache_hit_rate(self):
        from headroom.providers.anthropic import AnthropicProvider

        cost = AnthropicProvider().estimate_cost(
            input_tokens=1_000_000,
            output_tokens=0,
            model="deepseek-flash",
            cached_tokens=1_000_000,
            now=OFF_PEAK,
        )
        assert cost == pytest.approx(0.003)


class TestDeepSeekTieredCost:
    """The per-request cost seam prices DeepSeek by the request's tier.

    The tier tests pass with litellm absent: the tier branch sits in front of
    the litellm availability check. The last two cover the litellm path itself.
    """

    def test_flash_off_peak_one_megatoken_each_way(self):
        from headroom.pricing.litellm_pricing import estimate_cost_from_tokens

        cost = estimate_cost_from_tokens(
            "deepseek-flash", input_tokens=1_000_000, output_tokens=1_000_000, now=OFF_PEAK
        )
        assert cost == pytest.approx(0.15 + 0.60)

    def test_peak_is_exactly_twice_off_peak(self):
        from headroom.pricing.litellm_pricing import estimate_cost_from_tokens

        peak = estimate_cost_from_tokens(
            "deepseek-flash", input_tokens=1_000_000, output_tokens=1_000_000, now=PEAK
        )
        off = estimate_cost_from_tokens(
            "deepseek-flash", input_tokens=1_000_000, output_tokens=1_000_000, now=OFF_PEAK
        )
        assert peak == pytest.approx(off * 2)

    def test_pro_off_peak_one_megatoken_each_way(self):
        from headroom.pricing.litellm_pricing import estimate_cost_from_tokens

        cost = estimate_cost_from_tokens(
            "deepseek-v4-pro", input_tokens=1_000_000, output_tokens=1_000_000, now=OFF_PEAK
        )
        assert cost == pytest.approx(0.66 + 1.98)

    def test_cached_tokens_bill_at_the_cache_hit_rate_exactly_once(self):
        from headroom.pricing.litellm_pricing import estimate_cost_from_tokens

        # input_tokens is the TOTAL prompt, cached included (litellm contract).
        cost = estimate_cost_from_tokens(
            "deepseek-flash",
            input_tokens=1_000_000,
            output_tokens=0,
            cached_tokens=1_000_000,
            now=OFF_PEAK,
        )
        assert cost == pytest.approx(0.003)

    def test_partially_cached_prompt_splits_the_two_input_rates(self):
        from headroom.pricing.litellm_pricing import estimate_cost_from_tokens

        cost = estimate_cost_from_tokens(
            "deepseek-flash",
            input_tokens=1_000_000,
            output_tokens=0,
            cached_tokens=400_000,
            now=OFF_PEAK,
        )
        assert cost == pytest.approx(0.6 * 0.15 + 0.4 * 0.003)

    @pytest.mark.parametrize(
        "model",
        ["deepseek-v4-flash", "deepseek-v4-flash-vision-exp", "deepseek/deepseek-v4-pro"],
    )
    def test_legacy_and_prefixed_ids_price_like_the_canonical_id(self, model):
        from headroom.pricing.litellm_pricing import estimate_cost_from_tokens

        canonical = "deepseek-v4-pro" if model.endswith("v4-pro") else "deepseek-flash"
        alias_cost = estimate_cost_from_tokens(
            model, input_tokens=1_000_000, output_tokens=0, now=OFF_PEAK
        )
        canonical_cost = estimate_cost_from_tokens(
            canonical, input_tokens=1_000_000, output_tokens=0, now=OFF_PEAK
        )
        # Both must be real numbers first: ``None == pytest.approx(None)`` is True,
        # so comparing unbound results would pass even if both fell out of scope.
        assert alias_cost is not None
        assert canonical_cost is not None
        assert alias_cost == pytest.approx(canonical_cost)

    def test_non_deepseek_models_still_take_the_litellm_path(self):
        from headroom.pricing.litellm_pricing import LITELLM_AVAILABLE, estimate_cost_from_tokens

        if not LITELLM_AVAILABLE:
            pytest.skip("litellm not available")
        cost = estimate_cost_from_tokens("gpt-4o", input_tokens=1_000_000, now=OFF_PEAK)
        assert cost is not None
        assert cost > 0.0

    def test_deepseek_models_outside_the_rate_card_take_the_litellm_path(self):
        from headroom.pricing.litellm_pricing import LITELLM_AVAILABLE, estimate_cost_from_tokens

        if not LITELLM_AVAILABLE:
            pytest.skip("litellm not available")
        # deepseek-chat is not on the flash/pro card, so the tier branch must not
        # claim it; litellm prices it (0.28/0.42 per 1M). The provider prefix is
        # required: litellm.cost_per_token refuses a bare "deepseek-chat".
        cost = estimate_cost_from_tokens(
            "deepseek/deepseek-chat", input_tokens=1_000_000, now=OFF_PEAK
        )
        assert cost == pytest.approx(0.28, rel=0.01)
