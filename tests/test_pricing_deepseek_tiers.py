"""Tests for DeepSeek peak/off-peak tier resolution."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from headroom.pricing.deepseek_tiers import (
    PEAK_MULTIPLIER,
    WEEKEND_OFF_PEAK_FROM,
    bare_model,
    is_peak,
    off_peak_rates,
    rates_for,
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


def utc(day: int, hour: int, minute: int = 0) -> datetime:
    """UTC instant in August 2026; day 17 is a Monday, 22/29 Saturdays, 23 Sunday."""
    return datetime(2026, 8, day, hour, minute, tzinfo=timezone.utc)


class TestBeijingPeakWindows:
    @pytest.mark.parametrize("hour,minute", [(1, 0), (3, 59), (6, 0), (9, 59)])
    def test_weekday_peak_windows(self, hour, minute):
        # 01:00-04:00 and 06:00-10:00 UTC are 09:00-12:00 and 14:00-18:00 Beijing.
        assert is_peak(utc(17, hour, minute)) is True

    @pytest.mark.parametrize("hour,minute", [(4, 0), (10, 0), (0, 0), (12, 0), (23, 59)])
    def test_weekday_off_peak(self, hour, minute):
        assert is_peak(utc(17, hour, minute)) is False

    def test_pre_rule_saturday_keeps_the_weekday_windows(self):
        # 2026-08-22 02:00 UTC is Saturday 10:00 Beijing, before the rule takes
        # effect, so it is still peak.
        assert WEEKEND_OFF_PEAK_FROM > utc(22, 2)
        assert is_peak(utc(22, 2)) is True

    @pytest.mark.parametrize("day,hour", [(23, 2), (23, 7), (29, 2), (29, 7)])
    def test_post_rule_weekends_are_all_day_off_peak(self, day, hour):
        assert is_peak(utc(day, hour)) is False

    # Every peak case above uses 2026-08-17, which is *before*
    # WEEKEND_OFF_PEAK_FROM, so none of them pins the weekday arm of the weekend
    # gate. Without the cases below, a mutant that made the gate unconditional
    # ("no peak pricing ever again" from 2026-08-23) would pass the whole file.
    @pytest.mark.parametrize("hour", [2, 7])
    def test_post_rule_weekday_peak_windows_still_apply(self, hour):
        # 2026-08-24 is a Monday: 02:00 and 07:00 UTC are 10:00 and 15:00 Beijing.
        assert is_peak(utc(24, hour)) is True

    def test_post_rule_weekday_evening_is_off_peak(self):
        # 12:00 UTC on that same Monday is 20:00 Beijing, outside both windows.
        assert is_peak(utc(24, 12)) is False

    def test_the_effective_instant_itself_is_off_peak(self):
        assert WEEKEND_OFF_PEAK_FROM == utc(22, 16)
        assert is_peak(WEEKEND_OFF_PEAK_FROM) is False
        # The instant above is Beijing Sunday 00:00, outside both windows, so it
        # is off-peak with or without the weekend rule. This boundary is the
        # informative one: Saturday 09:00 Beijing is inside a window but only
        # just before the gate, so peak here means the gate did not apply early.
        assert WEEKEND_OFF_PEAK_FROM > utc(22, 1)
        assert is_peak(utc(22, 1)) is True

    def test_naive_instants_are_read_as_utc(self):
        assert is_peak(datetime(2026, 8, 17, 2, 0)) is True
        assert is_peak(datetime(2026, 8, 17, 12, 0)) is False


class TestRatesFor:
    def test_peak_instant_selects_the_peak_tier(self):
        off = off_peak_rates("deepseek-flash")
        rates = rates_for("deepseek-flash", utc(17, 2))
        assert off is not None
        assert rates is not None
        assert rates.tier == "peak"
        # Literal vendor peak rates, not just the product of the off-peak row and
        # PEAK_MULTIPLIER: re-deriving from the same constant cannot catch a wrong
        # multiplier or a field that should not have been multiplied.
        assert rates.cache_hit_per_1m == 0.006
        assert rates.input_per_1m == 0.30
        assert rates.output_per_1m == 1.20
        assert rates.cache_write_per_1m == 0.0
        assert rates.input_per_1m == off.input_per_1m * PEAK_MULTIPLIER

    def test_off_peak_instant_selects_the_off_peak_tier(self):
        off = off_peak_rates("deepseek-v4-pro")
        rates = rates_for("deepseek-v4-pro", utc(17, 12))
        assert off is not None
        assert rates is not None
        assert rates.tier == "off_peak"
        assert rates.input_per_1m == off.input_per_1m

    def test_post_rule_weekday_instant_selects_the_peak_tier(self):
        # The live regime: past WEEKEND_OFF_PEAK_FROM, a Monday mid-morning must
        # still bill peak, through the seam the cost paths actually call.
        rates = rates_for("deepseek-flash", utc(24, 2))
        assert rates is not None
        assert rates.tier == "peak"
        assert rates.input_per_1m == 0.30

    def test_default_instant_reads_the_wall_clock(self):
        before = datetime.now(timezone.utc)
        rates = rates_for("deepseek-flash")
        after = datetime.now(timezone.utc)
        assert rates is not None
        # The default read is its own clock read, so it can land on either side of
        # a window boundary. Bracket it and accept either bracket's tier.
        assert rates.tier in {
            "peak" if is_peak(before) else "off_peak",
            "peak" if is_peak(after) else "off_peak",
        }

    def test_naive_instant_selects_the_same_tier_as_the_aware_one(self):
        aware = rates_for("deepseek-flash", utc(17, 2))
        naive = rates_for("deepseek-flash", datetime(2026, 8, 17, 2))
        assert aware is not None
        assert naive is not None
        assert (naive.tier, naive.input_per_1m) == (aware.tier, aware.input_per_1m)

    @pytest.mark.parametrize(
        "model",
        [
            "deepseek-v4-flash",
            "deepseek-v4-flash-vision-exp",
            "deepseek/deepseek-v4-pro",
        ],
    )
    def test_aliases_and_prefixed_ids_are_tiered(self, model):
        # Compare against the bare id rather than asserting non-None: a prefix bug
        # that resolved the prefixed id to the wrong model's row would still be
        # non-None.
        bare = model.rsplit("/", 1)[-1]
        assert rates_for(model, utc(17, 2)) == rates_for(bare, utc(17, 2))

    @pytest.mark.parametrize("model", ["deepseek-chat", "gpt-4o", ""])
    def test_out_of_scope_models_return_none(self, model):
        assert rates_for(model, utc(17, 2)) is None
