"""CostTracker.totals() must be stats() minus the work, not minus the accuracy.

It exists only so the per-request metrics path stops walking 31 days of cost
records to read two fields. If the two ever disagree, the savings history
silently drifts from /stats.
"""

from __future__ import annotations

import random

import pytest

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
