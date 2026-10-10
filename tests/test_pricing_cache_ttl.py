"""Tests for prompt-cache TTL pricing structure."""

from __future__ import annotations

import pytest

from headroom.pricing import cache_ttl


def test_multipliers_match_anthropic_structure() -> None:
    assert cache_ttl.CACHE_READ_MULTIPLIER == 0.10
    assert cache_ttl.CACHE_WRITE_MULTIPLIERS == {"5m": 1.25, "1h": 2.00}
    assert cache_ttl.DEFAULT_CACHE_TTL == "5m"


def test_cache_write_multiplier() -> None:
    assert cache_ttl.cache_write_multiplier("5m") == 1.25
    assert cache_ttl.cache_write_multiplier("1h") == 2.00


def test_unknown_ttl_raises_rather_than_defaulting_cheap() -> None:
    """Silently returning the 5m rate would understate cost."""
    with pytest.raises(ValueError, match="unknown cache TTL"):
        cache_ttl.cache_write_multiplier("30m")


def test_rates_derive_from_base_input() -> None:
    rates = cache_ttl.cache_rates_per_1m(5.00)  # opus-class base input
    assert rates == {"read": 0.50, "write_5m": 6.25, "write_1h": 10.00}


def test_breakeven_share_is_39_5_percent() -> None:
    assert cache_ttl.ttl_breakeven_share() == pytest.approx(0.3947, abs=1e-4)


def test_breakeven_is_model_independent() -> None:
    """Every term scales with base input, so the threshold is a pure ratio."""
    for base in (1.00, 3.00, 5.00, 15.00):
        r = cache_ttl.cache_rates_per_1m(base)
        share = (r["write_1h"] - r["write_5m"]) / (r["write_1h"] - r["read"])
        assert share == pytest.approx(cache_ttl.ttl_breakeven_share())


class TestBreakevenDecision:
    """The threshold must actually predict which TTL is cheaper."""

    @staticmethod
    def _cost(total_writes: int, idle_gap_writes: int, base: float) -> tuple[float, float]:
        r = cache_ttl.cache_rates_per_1m(base)
        at_5m = total_writes * r["write_5m"]
        at_1h = (total_writes - idle_gap_writes) * r["write_1h"] + idle_gap_writes * r["read"]
        return at_5m / 1e6, at_1h / 1e6

    def test_above_threshold_1h_wins(self) -> None:
        at_5m, at_1h = self._cost(1_000_000, 500_000, 5.00)  # 50% > 39.5%
        assert at_1h < at_5m

    def test_below_threshold_5m_wins(self) -> None:
        at_5m, at_1h = self._cost(1_000_000, 300_000, 5.00)  # 30% < 39.5%
        assert at_5m < at_1h

    def test_at_threshold_costs_are_equal(self) -> None:
        share = cache_ttl.ttl_breakeven_share()
        at_5m, at_1h = self._cost(1_000_000, int(1_000_000 * share), 5.00)
        assert at_1h == pytest.approx(at_5m, rel=1e-5)


def test_write_premium_is_not_forgotten() -> None:
    """Regression guard for the 1.9x overstatement class of error.

    Figures are the real measured corpus: 306,631,892 cache-write tokens of
    which 177,636,344 followed a 5m-1h idle gap, at opus-class $5/1M input.
    Counting only the recovered rewrites reports ~$1,021; the honest net after
    the write premium on the remaining 128,995,548 writes is ~$538.
    """
    total_writes = 306_631_892
    idle_gap = 177_636_344
    r = cache_ttl.cache_rates_per_1m(5.00)

    naive = idle_gap * (r["write_5m"] - r["read"]) / 1e6
    at_5m = total_writes * r["write_5m"] / 1e6
    at_1h = ((total_writes - idle_gap) * r["write_1h"] + idle_gap * r["read"]) / 1e6
    net = at_5m - at_1h

    assert naive == pytest.approx(1021.41, abs=0.5)
    assert net == pytest.approx(537.68, abs=0.5)
    assert naive / net == pytest.approx(1.9, abs=0.05)
    # And this corpus is past the threshold, so the switch is correct here.
    assert idle_gap / total_writes > cache_ttl.ttl_breakeven_share()
