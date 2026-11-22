"""A late-repriced premium must land on the request that earned it.

``SavingsTracker.record_request`` is fed the cost tracker's cumulative total
*before* the current request is recorded (the metrics path runs first), so it
resolves the difference with ``max(previous + delta, passed_cumulative)``. When
the per-request delta is priced at base rates and the cumulative is tier-aware,
the cumulative wins on a long-context turn and the whole premium is booked
against the NEXT request's model and timestamp in the history.

These pin the two halves of the fix: the per-request input estimate prices the
model's long-context tier, and a two-request history attributes each step to its
own request.
"""

from __future__ import annotations

import types

import pytest

from headroom.pricing.counterfactual import resolve_rates
from headroom.proxy import savings_tracker as st
from headroom.proxy.cost import CostTracker
from headroom.proxy.savings_tracker import SavingsTracker, _estimate_input_cost_usd

_CATALOG = {
    "long-model": {
        "input_cost_per_token": 1e-6,
        "input_cost_per_token_above_200k_tokens": 2e-6,
        "output_cost_per_token": 1e-5,
    },
    "cheap-model": {
        "input_cost_per_token": 1e-5,
        "output_cost_per_token": 1e-5,
    },
}


@pytest.fixture
def tiered_pricing(monkeypatch: pytest.MonkeyPatch) -> dict:
    """The same catalog on the savings side and the cost-tracker side."""
    from types import SimpleNamespace

    fake = SimpleNamespace(model_cost=_CATALOG, cost_per_token=lambda **_kw: (0.0, 0.0))
    monkeypatch.setattr(st, "_get_litellm_module", lambda: fake)
    monkeypatch.setattr(st, "_resolve_litellm_model", lambda model: model)
    resolve_rates.cache_clear()
    monkeypatch.setattr("headroom.pricing.counterfactual._litellm", lambda: fake)
    monkeypatch.setattr(
        "headroom.pricing.litellm_pricing.resolve_litellm_model", lambda model: model
    )
    yield _CATALOG
    resolve_rates.cache_clear()


def test_per_request_estimate_prices_the_upper_tier(tiered_pricing: dict) -> None:
    """250k tokens above a 200k threshold bills at the long-context rate."""
    assert _estimate_input_cost_usd("long-model", 100_000) == pytest.approx(0.1)
    assert _estimate_input_cost_usd("long-model", 250_000) == pytest.approx(0.5)


def test_history_books_a_late_premium_on_the_request_that_earned_it(
    tiered_pricing: dict, tmp_path
) -> None:
    cost_tracker = CostTracker()
    tracker = SavingsTracker(path=tmp_path / "savings.json", save_flush_every=1)

    def record(model: str, uncached: int) -> None:
        # The metrics path hands over the cumulative BEFORE this request is
        # recorded; the tracker then records it. Same order as outcome.py.
        total_tokens, total_cost = cost_tracker.totals()
        tracker.record_request(
            model=model,
            input_tokens=uncached,
            tokens_saved=0,
            uncached_input_tokens=uncached,
            output_tokens=1_000,
            total_input_tokens=total_tokens,
            total_input_cost_usd=total_cost,
        )
        cost_tracker.record_tokens(
            model,
            tokens_saved=0,
            tokens_sent=uncached,
            uncached_tokens=uncached,
            output_tokens=1_000,
        )

    record("long-model", 250_000)  # $0.50, at the long-context rate
    record("cheap-model", 1_000)  # $0.01

    lifetime = tracker._state["lifetime"]
    assert lifetime["total_input_cost_usd"] == pytest.approx(cost_tracker.totals()[1])

    history = [entry for entry in tracker._state["history"] if entry["total_input_cost_usd"]]
    assert [entry["model"] for entry in history] == ["long-model", "cheap-model"]
    long_model_step, cheap_model_step = history

    # The long turn's own premium is in ITS checkpoint, not deferred into the
    # next one: the second step is the second request's $0.01, not $0.25.
    assert long_model_step["total_input_cost_usd"] == pytest.approx(0.5)
    assert (
        cheap_model_step["total_input_cost_usd"] - long_model_step["total_input_cost_usd"]
    ) == pytest.approx(0.01)


def test_row_that_advertises_a_tier_without_a_tier_price_keeps_the_base_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``None`` tier rate is an unpublished price, not a free one.

    The tier suffix is discovered from the row's own
    ``input_cost_per_token_above_<N>k_tokens`` field, but catalogs do publish
    rows that carry the field with a ``None`` value. Taking that at face value
    would price the request at $0 instead of the base rate it demonstrably
    bills at.
    """
    from types import SimpleNamespace

    catalog = {
        "half-priced-model": {
            "input_cost_per_token": 1e-6,
            "input_cost_per_token_above_200k_tokens": None,
        }
    }
    fake = SimpleNamespace(model_cost=catalog, cost_per_token=lambda **_kw: (0.0, 0.0))
    monkeypatch.setattr(st, "_get_litellm_module", lambda: fake)
    monkeypatch.setattr(st, "_resolve_litellm_model", lambda model: model)

    assert _estimate_input_cost_usd("half-priced-model", 250_000) == pytest.approx(0.25)


def test_tier_lookup_failure_does_not_break_request_recording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tier helper that raises must degrade to base rates, not drop the request.

    ``record_request`` runs on the response path for every request; a catalog
    shape this code does not expect can cost accuracy, never bookkeeping.
    """
    import headroom.pricing.counterfactual as counterfactual

    def _boom(_info):
        raise RuntimeError("unexpected catalog shape")

    monkeypatch.setattr(counterfactual, "long_context_tier", _boom)

    catalog = {
        "long-model": {
            "input_cost_per_token": 1e-6,
            "input_cost_per_token_above_200k_tokens": 2e-6,
        }
    }
    fake = types.SimpleNamespace(model_cost=catalog, cost_per_token=lambda **_kw: (0.0, 0.0))
    monkeypatch.setattr(st, "_get_litellm_module", lambda: fake)
    monkeypatch.setattr(st, "_resolve_litellm_model", lambda model: model)

    # Base rate, because the tier could not be read — and no exception escapes.
    assert _estimate_input_cost_usd("long-model", 250_000) == pytest.approx(0.25)
