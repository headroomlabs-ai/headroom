"""CostTracker.totals() must be stats() minus the work, not minus the accuracy.

It exists only so the per-request metrics path stops walking 31 days of cost
records to read two fields. If the two ever disagree, the savings history
silently drifts from /stats.
"""

from __future__ import annotations

import random

import pytest

from headroom.pricing.cache_ttl import CACHE_WRITE_MULTIPLIERS
from headroom.pricing.counterfactual import resolve_rates
from headroom.proxy.cost import CostTracker
from tests._pricing_models import anthropic_pricing_model


def _tracker(seed: int, n_models: int, n_requests: int) -> CostTracker:
    r = random.Random(seed)
    tracker = CostTracker()
    models = [
        "claude-sonnet-5",
        "claude-opus-4-1",
        "gpt-4o",
        "gpt-4o-mini",
        "some-unpriceable-model",
    ][:n_models]
    for _ in range(n_requests):
        model = r.choice(models)
        sent = r.randint(0, 20000)
        # Alternate between requests that carry an API cache breakdown and ones
        # that do not — totals() has a branch for each, and only the second
        # falls back to list price.
        with_cache = r.random() < 0.5
        tracker.record_tokens(
            model=model,
            tokens_saved=r.randint(0, 5000),
            tokens_sent=sent,
            cache_read_tokens=r.randint(0, sent) if with_cache else 0,
            cache_write_tokens=r.randint(0, 500) if with_cache else 0,
            uncached_tokens=r.randint(0, sent) if with_cache else 0,
            output_tokens=r.randint(0, 2000),
        )
    return tracker


@pytest.mark.parametrize(
    ("n_models", "n_requests"),
    [(0, 0), (1, 1), (1, 50), (3, 200), (5, 500)],
)
def test_totals_matches_stats(n_models: int, n_requests: int) -> None:
    tracker = _tracker(seed=n_models * 100 + n_requests, n_models=n_models, n_requests=n_requests)
    stats = tracker.stats()
    assert tracker.totals() == (
        stats["total_input_tokens"],
        stats["total_input_cost_usd"],
    )


def test_totals_matches_stats_on_a_fresh_tracker() -> None:
    tracker = CostTracker()
    stats = tracker.stats()
    assert tracker.totals() == (stats["total_input_tokens"], stats["total_input_cost_usd"])


def test_totals_does_not_walk_the_cost_records() -> None:
    """The point of the method: no period_cost_breakdown, at any ledger size."""
    tracker = _tracker(seed=7, n_models=3, n_requests=100)
    called = False
    real = tracker.period_cost_breakdown

    def spy(*a, **kw):
        nonlocal called
        called = True
        return real(*a, **kw)

    tracker.period_cost_breakdown = spy  # type: ignore[method-assign]
    tracker.totals()
    assert not called, "totals() still walks the cost records"
    tracker.stats()
    assert called, "stats() should still report budget_basis"


# --- Context tiers (#4960) -------------------------------------------------
#
# ``stats()``/``totals()`` recompute input cost from per-model aggregates, and
# those aggregates had no per-tier split for sent/cache tokens. Every token was
# therefore repriced at ``long_context=False``, so a request that crossed the
# model's 200k threshold was booked at the base rate in /stats and in the
# metrics path while the per-request budget path booked the long-context rate.
# The two paths disagreed on the same traffic.

_LONG_CONTEXT_TOKENS = 250_000


def _tier_model() -> str:
    """A currently-priced model that publishes an above-200k input rate."""
    return anthropic_pricing_model("input_cost_per_token_above_200k_tokens")


def _input_rates(model: str) -> tuple[float, float]:
    """``(base, long_context)`` uncached input rates per token."""
    import litellm

    from headroom.pricing.litellm_pricing import resolve_litellm_model

    info = litellm.model_cost.get(resolve_litellm_model(model), {})
    return info["input_cost_per_token"], info["input_cost_per_token_above_200k_tokens"]


def test_totals_agrees_with_the_budget_path_on_a_long_context_turn() -> None:
    """The same request must book the same dollars on both paths."""
    model = _tier_model()
    tracker = CostTracker()
    tracker.record_tokens(
        model,
        tokens_saved=0,
        tokens_sent=_LONG_CONTEXT_TOKENS,
        uncached_tokens=_LONG_CONTEXT_TOKENS,
    )

    _, input_cost = tracker.totals()

    assert input_cost == pytest.approx(tracker.get_period_cost(), rel=1e-6)


def test_a_long_context_turn_is_not_repriced_at_the_base_rate() -> None:
    model = _tier_model()
    base, long = _input_rates(model)
    assert long > base, "fixture model does not publish a long-context premium"

    tracker = CostTracker()
    tracker.record_tokens(
        model,
        tokens_saved=0,
        tokens_sent=_LONG_CONTEXT_TOKENS,
        uncached_tokens=_LONG_CONTEXT_TOKENS,
    )

    expected = _LONG_CONTEXT_TOKENS * long
    assert tracker.totals()[1] == pytest.approx(expected)
    assert tracker.stats()["total_input_cost_usd"] == pytest.approx(expected)
    assert tracker.stats()["total_input_cost_usd"] != pytest.approx(_LONG_CONTEXT_TOKENS * base)


def test_each_request_is_priced_at_the_tier_it_was_billed_at() -> None:
    """One model, two tiers: the split has to survive into the recompute."""
    model = _tier_model()
    base, long = _input_rates(model)

    tracker = CostTracker()
    tracker.record_tokens(model, tokens_saved=0, tokens_sent=100_000, uncached_tokens=100_000)
    tracker.record_tokens(
        model,
        tokens_saved=0,
        tokens_sent=_LONG_CONTEXT_TOKENS,
        uncached_tokens=_LONG_CONTEXT_TOKENS,
    )

    expected = 100_000 * base + _LONG_CONTEXT_TOKENS * long
    assert tracker.totals() == (350_000, pytest.approx(expected))
    stats = tracker.stats()
    assert stats["total_input_tokens"] == 350_000
    assert stats["total_input_cost_usd"] == pytest.approx(expected)


def test_the_list_price_fallback_is_tier_aware_too() -> None:
    """No provider cache breakdown -> tokens_sent at list, but at its own tier."""
    model = _tier_model()
    _base, long = _input_rates(model)

    tracker = CostTracker()
    tracker.record_tokens(model, tokens_saved=0, tokens_sent=_LONG_CONTEXT_TOKENS)

    expected = _LONG_CONTEXT_TOKENS * long
    assert tracker.totals()[1] == pytest.approx(expected)
    assert tracker.stats()["total_input_cost_usd"] == pytest.approx(expected)


# --- Cached input and mixed-provenance tiers ---------------------------------
#
# The tests above cover uncached traffic and the no-breakdown fallback. These
# pin the branches in between: cache reads, the 5m/1h write split across BOTH
# tiers of one model, and a tier that carries measured AND estimated turns.

_TIERED_CATALOG = {
    "tiered": {
        "input_cost_per_token": 1e-6,
        "input_cost_per_token_above_200k_tokens": 2e-6,
        "cache_read_input_token_cost": 1e-7,
        "cache_read_input_token_cost_above_200k_tokens": 2e-7,
        "cache_creation_input_token_cost": 1.25e-6,
        "cache_creation_input_token_cost_above_200k_tokens": 2.5e-6,
        "cache_creation_input_token_cost_above_1hr": 2e-6,
        "output_cost_per_token": 5e-6,
    }
}

_BASE_INPUT = 1e-6
_LONG_INPUT = 2e-6
_BASE_READ = 1e-7
_LONG_READ = 2e-7
_BASE_WRITE_5M = 1.25e-6
_LONG_WRITE_5M = 2.5e-6
_BASE_WRITE_1H = 2e-6
# No catalog publishes a 1h + long-context rate, so that tier derives it from
# its own base input price and the structural TTL multiplier.
_LONG_WRITE_1H = _LONG_INPUT * CACHE_WRITE_MULTIPLIERS["1h"]


@pytest.fixture
def tiered_catalog(monkeypatch: pytest.MonkeyPatch):
    """A deterministic catalog with a 200k tier and both write TTLs."""
    from types import SimpleNamespace

    resolve_rates.cache_clear()
    monkeypatch.setattr(
        "headroom.pricing.counterfactual._litellm",
        lambda: SimpleNamespace(model_cost=_TIERED_CATALOG),
    )
    monkeypatch.setattr(
        "headroom.pricing.litellm_pricing.resolve_litellm_model", lambda model: model
    )
    yield "tiered"
    resolve_rates.cache_clear()


def test_cached_long_context_turn_prices_reads_and_both_write_ttls(tiered_catalog: str) -> None:
    """A warm long-context turn, next to a base-tier one on the same model."""
    model = tiered_catalog
    tracker = CostTracker()
    tracker.record_tokens(
        model,
        tokens_saved=0,
        tokens_sent=150_000,
        cache_read_tokens=100_000,
        cache_write_tokens=40_000,
        cache_write_5m_tokens=40_000,
        uncached_tokens=10_000,
    )
    tracker.record_tokens(
        model,
        tokens_saved=0,
        tokens_sent=260_000,
        cache_read_tokens=200_000,
        cache_write_tokens=50_000,
        cache_write_5m_tokens=40_000,
        cache_write_1h_tokens=10_000,
        uncached_tokens=10_000,
    )

    base_turn = 100_000 * _BASE_READ + 40_000 * _BASE_WRITE_5M + 10_000 * _BASE_INPUT
    long_turn = (
        200_000 * _LONG_READ
        + 40_000 * _LONG_WRITE_5M
        + 10_000 * _LONG_WRITE_1H
        + 10_000 * _LONG_INPUT
    )
    expected = base_turn + long_turn

    stats = tracker.stats()
    assert tracker.totals() == (410_000, pytest.approx(expected))
    assert stats["total_input_cost_usd"] == pytest.approx(expected)
    # Each TTL and tier is actually applied, not collapsed onto one rate.
    assert long_turn > (200_000 * _BASE_READ + 50_000 * _BASE_WRITE_5M + 10_000 * _BASE_INPUT)


def test_estimated_spend_is_not_erased_by_a_later_measured_turn(tiered_catalog: str) -> None:
    """A tier can carry both provenances; dropping the estimate makes spend fall."""
    model = tiered_catalog
    tracker = CostTracker()
    # Measured short turn.
    tracker.record_tokens(model, tokens_saved=0, tokens_sent=100_000, uncached_tokens=100_000)
    after_measured_short = tracker.totals()[1]
    assert after_measured_short == pytest.approx(100_000 * _BASE_INPUT)

    # Estimated long turn: no usage breakdown, list-priced at its own tier.
    tracker.record_tokens(model, tokens_saved=0, tokens_sent=250_000)
    after_estimated_long = tracker.totals()[1]
    assert after_estimated_long == pytest.approx(after_measured_short + 250_000 * _LONG_INPUT)

    # Measured long turn. If the estimate were dropped here, spend would FALL
    # from $0.60 to $0.52; both turns were billed, so it must rise.
    tracker.record_tokens(model, tokens_saved=0, tokens_sent=210_000, uncached_tokens=210_000)
    after_measured_long = tracker.totals()[1]

    assert after_measured_long == pytest.approx(after_estimated_long + 210_000 * _LONG_INPUT)
    assert after_measured_long > after_estimated_long
    assert tracker.stats()["total_input_cost_usd"] == pytest.approx(after_measured_long)


def test_totals_and_stats_agree_on_passthrough_traffic(tiered_catalog: str) -> None:
    """passthrough: labels are not billable requests, on both findings."""
    model = tiered_catalog
    tracker = CostTracker()
    tracker.record_tokens("passthrough:count_tokens", tokens_saved=10, tokens_sent=90)
    tracker.record_tokens(model, tokens_saved=0, tokens_sent=1_000, uncached_tokens=1_000)

    stats = tracker.stats()
    assert tracker.totals() == (stats["total_input_tokens"], stats["total_input_cost_usd"])
    # The probe's tokens are not forwarded input and are excluded from both.
    assert tracker.totals()[0] == 1_000
