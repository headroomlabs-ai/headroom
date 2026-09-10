"""CostTracker.totals() must be stats() minus the work, not minus the accuracy.

It exists only so the per-request metrics path stops walking 31 days of cost
records to read two fields. If the two ever disagree, the savings history
silently drifts from /stats.
"""

from __future__ import annotations

import random

import pytest

from headroom.proxy.cost import CostTracker


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
