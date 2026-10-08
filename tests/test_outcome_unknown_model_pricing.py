"""Outcome-path pricing integration: a model LiteLLM can't price.

#3825 review follow-up: ``CostTracker.record_tokens`` had a ``price_basis``
stamp but ``emit_request_outcome`` never supplied it, and ``estimate_cost``
returning None meant an unknown model booked ZERO ledger entries -- the
budget could never trip on it. The reviewer's probe: 200k input + 1k output
tokens against a $0.01 limit recorded zero entries, zero estimated-price
records, $0 total, and allowed under count, ignore, AND block.

These tests run the real ``emit_request_outcome -> CostTracker`` path with a
model LiteLLM can't price and assert the request now books a
PRICE_BASIS_ESTIMATED entry ($2.50/$10 per 1M provider fallback) and the
selected budget policy applies to it. ``estimated_pct`` is untouched: the
usage was provider-reported, so it stays 0.0 -- price provenance is the
independent second dimension, not a repurposed usage flag.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from headroom.proxy.budget_basis_policy import (
    BUDGET_BASIS_BLOCK,
    BUDGET_BASIS_COUNT,
    BUDGET_BASIS_IGNORE,
    PRICE_BASIS_ESTIMATED,
    PRICE_BASIS_MEASURED,
)
from headroom.proxy.outcome import RequestOutcome, emit_request_outcome

MODEL = "zz-made-up-model-3825"  # unknown to every price source

# 200k input + 1k output at the $2.50/$10 per 1M fallback.
EXPECTED_USD = 200_000 / 1_000_000 * 2.50 + 1_000 / 1_000_000 * 10.00


class _Metrics:
    async def record_request(self, **kwargs):
        pass


@pytest.mark.parametrize(
    "uncached,cache_read,inferred_write,expected",
    [(200_000, 0, False, 20.20), (100_000, 100_000, False, 15.20), (100_000, 100_000, True, 15.20)],
)
def test_known_model_override_is_booked_and_enforces_budget(
    monkeypatch, uncached, cache_read, inferred_write, expected
):
    from headroom.providers.openai import OpenAIProvider

    monkeypatch.setenv(
        "HEADROOM_MODEL_LIMITS", json.dumps({"openai": {"pricing": {"gpt-4o": [100, 200]}}})
    )
    tracker = _tracker(budget_limit_usd=10.0)
    handler = _Handler(tracker)
    handler.openai_provider = OpenAIProvider()
    assert handler.openai_provider.resolve_pricing_for_ledger("gpt-4o") == (
        (100.0, 200.0),
        PRICE_BASIS_MEASURED,
    )
    asyncio.run(
        emit_request_outcome(
            handler,
            RequestOutcome(
                request_id="override-3825",
                tokens_saved=0,
                attempted_input_tokens=200_000,
                provider="openai",
                model="gpt-4o",
                original_tokens=200_000,
                optimized_tokens=200_000,
                provider_input_tokens=200_000,
                uncached_input_tokens=uncached,
                cache_read_tokens=cache_read,
                cache_write_tokens=uncached if inferred_write else 0,
                cache_inferred=inferred_write,
                output_tokens=1_000,
            ),
        )
    )
    basis = _budget_basis(tracker)
    assert basis["total_usd"] == pytest.approx(expected)
    assert basis["price_estimated_records"] == 0
    assert basis["estimated_records"] == 0
    assert tracker.check_budget() == (False, 0.0)
    displayed = tracker.stats()
    assert displayed["total_input_cost_usd"] == pytest.approx(expected - 0.20)
    assert displayed["output_cost_usd"] == pytest.approx(0.20)
    assert displayed["total_cost_usd"] == pytest.approx(expected)
    assert tracker.totals()[1] == pytest.approx(expected - 0.20)


@pytest.mark.parametrize("model", ["gpt-5", "gemini-2.5-pro"])
def test_catalog_outcome_retains_native_cache_and_context_tiers(monkeypatch, model):
    from headroom.providers.openai import OpenAIProvider

    monkeypatch.setenv("HEADROOM_MODEL_LIMITS", "{}")
    tracker = _tracker(budget_limit_usd=100.0)
    handler = _Handler(tracker)
    handler.openai_provider = OpenAIProvider()
    expected = tracker.estimate_cost(model, 300_000, 1_000, cache_read_tokens=100_000)
    assert expected is not None
    asyncio.run(
        emit_request_outcome(
            handler,
            RequestOutcome(
                request_id="catalog-3825",
                tokens_saved=0,
                attempted_input_tokens=300_000,
                provider="openai",
                model=model,
                original_tokens=300_000,
                optimized_tokens=300_000,
                provider_input_tokens=300_000,
                uncached_input_tokens=200_000,
                cache_read_tokens=100_000,
                output_tokens=1_000,
            ),
        )
    )
    assert _budget_basis(tracker)["total_usd"] == pytest.approx(expected)
    assert _budget_basis(tracker)["price_estimated_records"] == 0
    assert tracker.stats()["total_cost_usd"] == pytest.approx(expected)


class _Handler:
    """Minimal funnel stub: real CostTracker, no provider object."""

    def __init__(self, cost_tracker):
        self.metrics = _Metrics()
        self.cost_tracker = cost_tracker
        self.logger = None


def test_display_retains_each_requests_selected_price():
    tracker = _tracker(budget_limit_usd=100.0)
    for pricing in [(100.0, 200.0), (50.0, 100.0)]:
        tracker.record_tokens(
            "gpt-4o",
            0,
            200_000,
            uncached_tokens=200_000,
            output_tokens=1_000,
            pricing=pricing,
            pricing_override=True,
        )
    displayed = tracker.stats()
    assert displayed["total_input_cost_usd"] == pytest.approx(30.0)
    assert displayed["output_cost_usd"] == pytest.approx(0.30)
    assert displayed["total_cost_usd"] == pytest.approx(30.30)
    assert displayed["budget_basis"]["total_usd"] == pytest.approx(30.30)
    assert tracker.totals() == (400_000, 30.0)


@pytest.mark.parametrize("write_5m,write_1h", [(0, 0), (20_000, 30_000), (0, 200_000)])
def test_unsplit_or_partial_cache_writes_remain_in_displayed_spend(write_5m, write_1h):
    tracker = _tracker()
    tracker.record_tokens(
        "gpt-4o",
        0,
        100_000,
        cache_write_tokens=100_000,
        cache_write_5m_tokens=write_5m,
        cache_write_1h_tokens=write_1h,
        pricing=(100.0, 200.0),
        pricing_override=True,
    )
    assert tracker.stats()["total_cost_usd"] == pytest.approx(10.0)
    assert tracker.stats()["budget_basis"]["total_usd"] == pytest.approx(10.0)
    assert tracker.totals() == (100_000, 10.0)


def test_runtime_reset_clears_selected_prices_and_spend():
    tracker = _tracker()
    tracker.record_tokens(
        "gpt-4o",
        0,
        100_000,
        uncached_tokens=100_000,
        output_tokens=1_000,
        pricing=(100.0, 200.0),
        pricing_override=True,
    )
    tracker.reset_runtime()
    assert tracker.stats()["total_cost_usd"] == 0.0
    assert tracker.totals() == (0, 0.0)
    assert tracker._fixed_pricing_by_model == {}
    tracker.record_tokens(
        "gpt-4o",
        0,
        100_000,
        uncached_tokens=100_000,
        output_tokens=1_000,
        pricing=(50.0, 100.0),
        pricing_override=True,
    )
    assert tracker.stats()["total_cost_usd"] == pytest.approx(5.10)
    assert tracker.stats()["budget_basis"]["total_usd"] == pytest.approx(5.10)
    assert tracker.totals() == (100_000, 5.0)


@pytest.fixture
def no_litellm(monkeypatch: pytest.MonkeyPatch):
    """Simulate LiteLLM being unable to price: estimate_cost returns None."""
    import headroom.proxy.cost as cost_mod

    cost_mod._warned_pricing_models.clear()
    monkeypatch.setattr(cost_mod, "_get_litellm_module", lambda: None)


def _tracker(**kwargs):
    from headroom.proxy.server import CostTracker

    return CostTracker(**kwargs)


def _outcome(**kwargs) -> RequestOutcome:
    return RequestOutcome(
        request_id="req-3825",
        provider="openai",
        model=MODEL,
        original_tokens=200_000,
        optimized_tokens=200_000,
        output_tokens=1_000,
        tokens_saved=0,
        attempted_input_tokens=200_000,
        # Provider-reported usage: the usage basis is measured; only the
        # PRICE is a guess.
        provider_input_tokens=200_000,
        uncached_input_tokens=200_000,
        **kwargs,
    )


def _emit(handler, outcome=None):
    asyncio.run(emit_request_outcome(handler, outcome or _outcome()))


def _budget_basis(ct):
    return ct.stats()["budget_basis"]


# ── The reviewer's probe ─────────────────────────────────────────────


def test_unknown_model_books_estimated_entry_and_denies_under_count(no_litellm):
    """200k in + 1k out vs a $0.01 limit: entry booked, policy=count denies."""
    ct = _tracker(budget_limit_usd=0.01, estimated_basis_policy=BUDGET_BASIS_COUNT)
    _emit(_Handler(ct))

    basis = _budget_basis(ct)
    assert basis["price_estimated_records"] == 1
    assert basis["price_estimated_usd"] == pytest.approx(EXPECTED_USD)
    assert basis["total_usd"] == pytest.approx(EXPECTED_USD)
    assert ct.stats()["total_cost_usd"] == pytest.approx(EXPECTED_USD)
    assert ct.totals()[1] == pytest.approx(0.50)
    # Usage was provider-reported: estimated_pct keeps its "input count was
    # estimated" meaning and is untouched by the price guess.
    assert basis["estimated_pct"] == 0.0
    assert basis["estimated_records"] == 0

    allowed, remaining = ct.check_budget()
    assert allowed is False
    assert remaining == 0.0


def test_unknown_model_policy_ignore_allows_but_still_books(no_litellm):
    """policy=ignore: the entry is booked and reported, but doesn't consume."""
    ct = _tracker(budget_limit_usd=0.01, estimated_basis_policy=BUDGET_BASIS_IGNORE)
    _emit(_Handler(ct))

    basis = _budget_basis(ct)
    assert basis["price_estimated_records"] == 1
    assert basis["price_estimated_usd"] == pytest.approx(EXPECTED_USD)

    allowed, remaining = ct.check_budget()
    assert allowed is True
    assert remaining == pytest.approx(0.01)


def test_unknown_model_policy_block_refuses_naming_price_guess(no_litellm):
    """policy=block: any price-guessed spend refuses, naming the price guess."""
    ct = _tracker(budget_limit_usd=100.0, estimated_basis_policy=BUDGET_BASIS_BLOCK)
    _emit(_Handler(ct))

    allowed, remaining = ct.check_budget()
    assert allowed is False
    assert remaining == 0.0
    assert "unknown-model default guess" in ct.budget_denial_detail()


# ── Provenance propagation ───────────────────────────────────────────


class _StubProvider:
    """Stands in for a provider's resolve_pricing_for_ledger."""

    def __init__(self, pricing, basis):
        self._pricing = pricing
        self._basis = basis

    def resolve_pricing_for_ledger(self, model):
        assert model == MODEL
        return self._pricing, self._basis


def test_provider_resolved_price_and_provenance_propagate(no_litellm):
    """The provider's per-lookup price AND provenance reach the ledger.

    An explicitly-configured price is a decision, not a guess: it books at
    the provider's number with measured provenance.
    """
    ct = _tracker(budget_limit_usd=100.0)
    handler = _Handler(ct)
    handler.openai_provider = _StubProvider((1.00, 2.00), PRICE_BASIS_MEASURED)
    _emit(handler)

    basis = _budget_basis(ct)
    assert basis["price_estimated_records"] == 0
    assert basis["price_estimated_usd"] == 0.0
    # 200k in + 1k out at the provider's $1.00/$2.00, not the $2.50/$10 default.
    assert basis["total_usd"] == pytest.approx(
        200_000 / 1_000_000 * 1.00 + 1_000 / 1_000_000 * 2.00
    )


def test_provider_estimated_flag_propagates(no_litellm):
    """A provider-flagged guess keeps its estimated provenance end to end."""
    ct = _tracker(budget_limit_usd=100.0)
    handler = _Handler(ct)
    handler.openai_provider = _StubProvider((2.50, 10.00), PRICE_BASIS_ESTIMATED)
    _emit(handler)

    basis = _budget_basis(ct)
    assert basis["price_estimated_records"] == 1
    assert basis["price_estimated_usd"] == pytest.approx(EXPECTED_USD)


def test_real_openai_provider_resolves_unknown_default_estimated(
    no_litellm, monkeypatch: pytest.MonkeyPatch
):
    """The real provider object on the funnel path: unknown model -> the
    GPT-4o-tier default with estimated provenance (the reviewer's scenario)."""
    import headroom.pricing.litellm_pricing as lp
    import headroom.providers.openai as openai_mod

    monkeypatch.setattr(lp, "LITELLM_AVAILABLE", False)
    monkeypatch.setattr(openai_mod, "_UNKNOWN_PRICING_MODELS", set())

    ct = _tracker(budget_limit_usd=0.01)
    handler = _Handler(ct)
    handler.openai_provider = openai_mod.OpenAIProvider()
    _emit(handler)

    basis = _budget_basis(ct)
    assert basis["price_estimated_records"] == 1
    assert basis["price_estimated_usd"] == pytest.approx(EXPECTED_USD)
    allowed, _ = ct.check_budget()
    assert allowed is False


def test_broken_provider_degrades_to_generic_fallback(no_litellm):
    """A raising provider probe must not break the funnel: the tracker's
    generic fallback books the entry as estimated."""
    ct = _tracker(budget_limit_usd=0.01)
    handler = _Handler(ct)

    class _Broken:
        def resolve_pricing_for_ledger(self, model):
            raise RuntimeError("provider hiccup")

    handler.openai_provider = _Broken()
    _emit(handler)

    basis = _budget_basis(ct)
    assert basis["price_estimated_records"] == 1
    assert basis["price_estimated_usd"] == pytest.approx(EXPECTED_USD)
