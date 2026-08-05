"""HeadroomContribution ratio methods."""

from __future__ import annotations

from headroom.subscription.models import HeadroomContribution


def test_efficiency_pct_never_exceeds_100_from_cache_reads():
    """Cache reads must not push efficiency above 100%.

    Cache-read tokens are a provider-side discount on tokens that were still
    forwarded, not tokens Headroom removed. Counting them in the numerator while
    the denominator excludes them let efficiency report impossible values like
    1000%.
    """
    c = HeadroomContribution(tokens_submitted=100, tokens_saved_cache_reads=1000)

    assert c.efficiency_pct() <= 100.0
    # Nothing was compressed or filtered, so the removal efficiency is 0.
    assert c.efficiency_pct() == 0.0


def test_efficiency_pct_is_compression_removal_ratio():
    """Efficiency is compression + CLI filtering over the pre-Headroom input."""
    c = HeadroomContribution(
        tokens_submitted=1000,
        tokens_saved_compression=400,
        tokens_saved_cache_reads=300,  # must not affect the ratio
    )

    # 400 removed out of (1000 forwarded + 400 removed) = 28.6%.
    assert c.efficiency_pct() == 28.6
    assert c.efficiency_pct() <= 100.0


def test_efficiency_pct_zero_when_no_input():
    assert HeadroomContribution().efficiency_pct() == 0.0
