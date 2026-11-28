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


def test_totals_prices_requests_without_a_usage_breakdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requests that reported no API usage breakdown still count their tokens in
    ``total_input_tokens`` but must not contribute $0 to the recomputed input
    cost. Once any request for a model carried a breakdown, the remaining
    ``sent`` tokens (the no-breakdown requests) were priced at $0.
    """
    # $3/M list price = $0.000003/token uncached, read/write rates arbitrary.
    uncached_price = 3.0 / 1_000_000
    monkeypatch.setattr(
        CostTracker,
        "_get_cache_prices",
        lambda _self, model, long_context=False, for_billing=True: (
            uncached_price * 0.1,
            uncached_price * 1.25,
            uncached_price * 2.0,
            uncached_price,
        ),
    )

    tracker = CostTracker()
    model = "claude-sonnet-4-5"
    tracker.record_tokens(model, 0, 1000, uncached_tokens=1000)  # with breakdown
    tracker.record_tokens(model, 0, 1000)  # no breakdown

    assert tracker.totals() == (2000, round(2000 * uncached_price, 4))
    stats = tracker.stats()
    assert stats["total_input_tokens"] == 2000
    assert stats["total_input_cost_usd"] == round(2000 * uncached_price, 4)


@pytest.mark.parametrize(
    ("estimated_tokens", "cache_read_tokens", "fallback_tokens", "expected_cost"),
    [(50000, 40000, 0, 0.012), (1000, 4000, 1000, 0.0042)],
)
@pytest.mark.parametrize("fallback_first", [False, True])
def test_totals_use_per_request_fallback_with_mismatched_estimates(
    monkeypatch: pytest.MonkeyPatch,
    estimated_tokens: int,
    cache_read_tokens: int,
    fallback_tokens: int,
    expected_cost: float,
    fallback_first: bool,
) -> None:
    monkeypatch.setattr(
        CostTracker, "_get_cache_prices", lambda *args, **kwargs: (0.3e-6, 3.75e-6, 6e-6, 3e-6)
    )
    tracker = CostTracker()
    model = "claude-sonnet-4-5"
    if fallback_first:
        tracker.record_tokens(model, 0, fallback_tokens)
    tracker.record_tokens(model, 0, estimated_tokens, cache_read_tokens=cache_read_tokens)
    if not fallback_first:
        tracker.record_tokens(model, 0, fallback_tokens)

    assert tracker.totals() == (estimated_tokens + fallback_tokens, expected_cost)
    assert tracker.stats()["total_input_cost_usd"] == expected_cost


def test_runtime_reset_discards_no_usage_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        CostTracker, "_get_cache_prices", lambda *args, **kwargs: (0.3e-6, 3.75e-6, 6e-6, 3e-6)
    )
    tracker = CostTracker()
    model = "claude-sonnet-4-5"
    tracker.record_tokens(model, 0, 1000)
    tracker.reset_runtime()
    tracker.record_tokens(model, 0, 50000, cache_read_tokens=40000)

    assert tracker.totals() == (50000, 0.012)
    assert tracker.stats()["total_input_cost_usd"] == 0.012
