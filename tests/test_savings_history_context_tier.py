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
        # Publishes a write premium, so the resolver has a 5m/1h trade to derive:
        # 5m at the long tier is $2.50/M, and 1h derives to $4/M from that tier's
        # $2/M input. A row with no write fields bills writes at the input rate,
        # which would hide the TTL bug this file pins.
        "cache_creation_input_token_cost": 1.25e-6,
        "cache_creation_input_token_cost_above_200k_tokens": 2.5e-6,
        "output_cost_per_token": 1e-5,
    },
    "cheap-model": {
        "input_cost_per_token": 1e-5,
        "output_cost_per_token": 1e-5,
    },
}


def _billing_calculator(catalog: dict):
    """Stand-in for LiteLLM's calculator: an unpriced slice bills at list input.

    ``_estimate_input_cost_usd`` prices SPEND, so a catalog row that publishes no
    cache-read rate resolves it the way ``CostTracker`` does — through the model's
    billing calculator. Returning a flat 0 here would have made cache reads free
    and hidden exactly the divergence these tests exist to catch.
    """

    def _price(*, model, prompt_tokens, completion_tokens=0, **_kw):
        rate = (catalog.get(model) or {}).get("input_cost_per_token", 0.0)
        return (float(prompt_tokens) * float(rate or 0.0), 0.0)

    return _price


@pytest.fixture
def tiered_pricing(monkeypatch: pytest.MonkeyPatch) -> dict:
    """The same catalog on the savings side and the cost-tracker side."""
    from types import SimpleNamespace

    fake = SimpleNamespace(model_cost=_CATALOG, cost_per_token=_billing_calculator(_CATALOG))
    monkeypatch.setattr(st, "_get_litellm_module", lambda: fake)
    monkeypatch.setattr(st, "_resolve_litellm_model", lambda model: model)
    resolve_rates.cache_clear()
    monkeypatch.setattr("headroom.pricing.counterfactual._litellm", lambda: fake)
    monkeypatch.setattr(
        "headroom.pricing.litellm_pricing.resolve_litellm_model", lambda model: model
    )
    yield _CATALOG
    resolve_rates.cache_clear()


@pytest.fixture
def catalog_pricing(monkeypatch: pytest.MonkeyPatch):
    """Point the shared rate resolver at a catalog for the rest of the test.

    ``_estimate_input_cost_usd`` now prices through
    :func:`headroom.pricing.counterfactual.resolve_rates` — the same resolver the
    cost tracker uses — so a test has to swap the catalog there, not on this
    module's own litellm handle.
    """

    def _apply(catalog: dict) -> dict:
        fake = types.SimpleNamespace(
            model_cost=catalog, cost_per_token=_billing_calculator(catalog)
        )
        resolve_rates.cache_clear()
        monkeypatch.setattr("headroom.pricing.counterfactual._litellm", lambda: fake)
        monkeypatch.setattr(
            "headroom.pricing.litellm_pricing.resolve_litellm_model", lambda model: model
        )
        return catalog

    yield _apply
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
    catalog_pricing,
) -> None:
    """A ``None`` tier rate is an unpublished price, not a free one.

    The tier suffix is discovered from the row's own
    ``input_cost_per_token_above_<N>k_tokens`` field, but catalogs do publish
    rows that carry the field with a ``None`` value. Taking that at face value
    would price the request at $0 instead of the base rate it demonstrably
    bills at.
    """
    catalog_pricing(
        {
            "half-priced-model": {
                "input_cost_per_token": 1e-6,
                "input_cost_per_token_above_200k_tokens": None,
            }
        }
    )

    assert _estimate_input_cost_usd("half-priced-model", 250_000) == pytest.approx(0.25)


def test_tier_lookup_failure_does_not_break_request_recording(
    monkeypatch: pytest.MonkeyPatch, catalog_pricing
) -> None:
    """A tier helper that raises must degrade to base rates, not drop the request.

    ``record_request`` runs on the response path for every request; a catalog
    shape this code does not expect can cost accuracy, never bookkeeping.
    """
    import headroom.pricing.counterfactual as counterfactual

    def _boom(_info):
        raise RuntimeError("unexpected catalog shape")

    monkeypatch.setattr(counterfactual, "long_context_tier", _boom)
    catalog_pricing(
        {
            "long-model": {
                "input_cost_per_token": 1e-6,
                "input_cost_per_token_above_200k_tokens": 2e-6,
            }
        }
    )

    # Unpriceable once the resolver raises, so the blended fallback ($3/M):
    # accuracy is what an unexpected row shape can cost, never bookkeeping.
    assert _estimate_input_cost_usd("long-model", 250_000) == pytest.approx(0.75)


def test_rollup_read_cost_probe_stays_on_the_base_tier(tiered_pricing: dict, tmp_path) -> None:
    """The rollup's per-token read rate must be the BASE tier's.

    The rate is probed by pricing a cache read, and the token count IS the tier
    selector, so a 1M-token probe priced the upper tier for every model with a
    long-context row and reported it for buckets that never crossed the
    threshold. A 1-token probe cannot cross any model's threshold.
    """
    tracker = SavingsTracker(path=tmp_path / "savings.json", save_flush_every=1)
    history = [
        {
            "timestamp": "2026-03-27T09:00:00Z",
            "provider": "anthropic",
            "model": "long-model",
            "total_tokens_saved": 0,
            "cache_read_tokens": 0,
        },
        {
            "timestamp": "2026-03-27T09:30:00Z",
            "provider": "anthropic",
            "model": "long-model",
            "total_tokens_saved": 0,
            "cache_read_tokens": 1_000,
            "cache_savings_usd": 0.9,
        },
    ]

    bucket = tracker._build_rollup(history, "hour")[0]

    # long-model publishes no cache-read rate, so reads bill at the base input
    # rate — $1/M — not the long-context $2/M the 1M probe used to return.
    assert bucket["cache_read_cost_usd_delta"] == pytest.approx(0.001)


def test_unpublished_cache_rates_fall_back_to_the_base_rate(catalog_pricing) -> None:
    """A `None` cache rate must not throw the request onto the blended fallback.

    ``dict.get(key, fallback)`` returns a published ``None`` rather than the
    fallback, so multiplying it raised inside the try block and the whole
    request — input tokens included — was priced at $3/M.
    """
    catalog_pricing(
        {
            "half-priced-model": {
                "input_cost_per_token": 1e-5,
                "cache_read_input_token_cost": None,
                "cache_creation_input_token_cost": None,
            }
        }
    )

    assert _estimate_input_cost_usd("half-priced-model", 0, cache_read_tokens=1_000) == (
        pytest.approx(0.01)
    )
    assert _estimate_input_cost_usd("half-priced-model", 0, cache_write_tokens=1_000) == (
        pytest.approx(0.01)
    )


def test_one_hour_cache_writes_are_priced_at_their_own_ttl(tiered_pricing: dict, tmp_path) -> None:
    """A 1h write bills 2x base; the savings history has to carry that TTL.

    Long-context 5m writes are $2.50/M here and 1h writes $4/M, so a 250k 1h turn
    costs $1.00. Pricing it at the 5m rate reported $0.625, and because
    ``record_request`` reconciles the estimate against ``CostTracker.totals()``
    with ``max()``, the missing $0.375 landed on the next request's model and
    timestamp instead — which is what this pins.
    """
    cost_tracker = CostTracker()
    tracker = SavingsTracker(path=tmp_path / "savings.json", save_flush_every=1)

    def record(model: str, *, uncached: int = 0, write: int = 0, write_1h: int = 0) -> None:
        # Metrics first, cost second: the order outcome.py records in.
        total_tokens, total_cost = cost_tracker.totals()
        tracker.record_request(
            model=model,
            input_tokens=uncached + write,
            tokens_saved=0,
            cache_write_tokens=write,
            cache_write_1h_tokens=write_1h,
            uncached_input_tokens=uncached,
            output_tokens=1_000,
            total_input_tokens=total_tokens,
            total_input_cost_usd=total_cost,
        )
        cost_tracker.record_tokens(
            model,
            tokens_saved=0,
            tokens_sent=uncached + write,
            cache_write_tokens=write,
            cache_write_1h_tokens=write_1h,
            uncached_tokens=uncached,
            output_tokens=1_000,
        )

    record("long-model", write=250_000, write_1h=250_000)  # $1.00 at the 1h rate
    record("cheap-model", uncached=1_000)  # $0.01

    history = [entry for entry in tracker._state["history"] if entry["total_input_cost_usd"]]
    assert [entry["model"] for entry in history] == ["long-model", "cheap-model"]
    assert [entry["total_input_cost_usd"] for entry in history] == [
        pytest.approx(1.00),
        pytest.approx(1.01),
    ]


def test_savings_history_spend_matches_the_tracker_when_cache_reads_are_unpriced(
    catalog_pricing, tmp_path
) -> None:
    """Reported spend must use the BILLED cache-rate fallback, like the tracker.

    $1/M input, $2/M above 200k, no published cache-read price, and a calculator
    that bills reads at $1/M. Reading 250k tokens therefore costs $0.25, not the
    $0.50 the counterfactual fallback implied by pricing reads at the tier's
    input rate — and since ``record_request`` keeps the larger of its estimate
    and the tracker total, the excess $0.25 was booked against the next request
    and never released.
    """
    catalog = {
        "reader-model": {
            "input_cost_per_token": 1e-6,
            "input_cost_per_token_above_200k_tokens": 2e-6,
        },
        "expensive-model": {"input_cost_per_token": 1e-5},
    }
    catalog_pricing(catalog)

    cost_tracker = CostTracker()
    tracker = SavingsTracker(path=tmp_path / "savings.json", save_flush_every=1)

    def record(model: str, *, reads: int = 0, uncached: int = 0) -> None:
        total_tokens, total_cost = cost_tracker.totals()
        tracker.record_request(
            model=model,
            input_tokens=reads + uncached,
            tokens_saved=0,
            cache_read_tokens=reads,
            uncached_input_tokens=uncached,
            output_tokens=1_000,
            total_input_tokens=total_tokens,
            total_input_cost_usd=total_cost,
        )
        cost_tracker.record_tokens(
            model,
            tokens_saved=0,
            tokens_sent=reads + uncached,
            cache_read_tokens=reads,
            uncached_tokens=uncached,
            output_tokens=1_000,
        )

    record("reader-model", reads=250_000)  # $0.25 of reads at the billed rate
    record("expensive-model", uncached=1_000)  # $0.01

    history = [entry for entry in tracker._state["history"] if entry["total_input_cost_usd"]]
    assert [entry["total_input_cost_usd"] for entry in history] == [
        pytest.approx(0.25),
        pytest.approx(0.26),
    ]
    # Parity with the tracker, which is what the budget path reports.
    assert tracker._state["lifetime"]["total_input_cost_usd"] == pytest.approx(
        cost_tracker.totals()[1]
    )
    assert cost_tracker.totals()[1] == pytest.approx(0.26)
