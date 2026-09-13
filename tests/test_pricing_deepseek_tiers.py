"""Tests for DeepSeek peak/off-peak tier resolution."""

from __future__ import annotations

import pytest

from headroom.pricing.deepseek_tiers import (
    PEAK_MULTIPLIER,
    bare_model,
    off_peak_rates,
)


class TestBareModel:
    def test_provider_prefix_is_stripped(self):
        assert bare_model("deepseek/deepseek-v4-pro") == "deepseek-v4-pro"

    def test_case_is_normalised(self):
        assert bare_model("DeepSeek-Flash") == "deepseek-flash"

    def test_bare_id_is_returned_unchanged(self):
        assert bare_model("deepseek-flash") == "deepseek-flash"


class TestOffPeakRates:
    def test_flash_matches_the_published_off_peak_usd_list(self):
        rates = off_peak_rates("deepseek-flash")
        assert rates is not None
        assert rates.cache_hit_per_1m == 0.003
        assert rates.input_per_1m == 0.15
        assert rates.output_per_1m == 0.60
        assert rates.tier == "off_peak"

    def test_pro_matches_the_published_off_peak_usd_list(self):
        rates = off_peak_rates("deepseek-v4-pro")
        assert rates is not None
        assert rates.cache_hit_per_1m == 0.022
        assert rates.input_per_1m == 0.66
        assert rates.output_per_1m == 1.98

    def test_deepseek_bills_no_cache_write_surcharge(self):
        rates = off_peak_rates("deepseek-flash")
        assert rates is not None
        assert rates.cache_write_per_1m == 0.0

    def test_peak_is_exactly_twice_off_peak(self):
        assert PEAK_MULTIPLIER == 2.0

    @pytest.mark.parametrize(
        "alias",
        ["deepseek-v4-flash", "deepseek-v4-flash-vision-exp"],
    )
    def test_retired_ids_resolve_to_the_flash_tier(self, alias):
        flash = off_peak_rates("deepseek-flash")
        legacy = off_peak_rates(alias)
        assert flash is not None
        assert legacy is not None
        assert (
            legacy.cache_hit_per_1m,
            legacy.input_per_1m,
            legacy.output_per_1m,
        ) == (
            flash.cache_hit_per_1m,
            flash.input_per_1m,
            flash.output_per_1m,
        )

    @pytest.mark.parametrize(
        "model",
        ["deepseek/deepseek-flash", "deepseek/deepseek-v4-pro"],
    )
    def test_provider_prefixed_ids_are_tiered(self, model):
        assert off_peak_rates(model) is not None

    @pytest.mark.parametrize(
        "model",
        ["deepseek-chat", "deepseek-reasoner", "deepseek-v3.2", "gpt-4o", ""],
    )
    def test_out_of_scope_models_have_no_tier(self, model):
        assert off_peak_rates(model) is None

    def test_rates_are_frozen(self):
        rates = off_peak_rates("deepseek-flash")
        assert rates is not None
        with pytest.raises(AttributeError):
            rates.input_per_1m = 1.0  # type: ignore[misc]
