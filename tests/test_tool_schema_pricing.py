"""Tool-schema deferral must be priced by the request's cache mix, not at list.

Reported from a real install: pricing deferral with
``_estimate_compression_savings_usd`` (full ``input_cost_per_token``) pushed the
blended $/token savings figure from $4.99/M to $32.88/M on traffic where
deferral ran 5.59x message compression -- a rate above the input list price of
every model in the mix, which no genuine input-token saving can reach.

Deferred schemas are byte-stable prompt-prefix content, so the counterfactual is
the prefix's own billing rhythm: cold-written once, read at the cache-read
discount on warm turns, re-written on TTL expiry. ``cost.py`` already models
exactly that for the cost card via ``_bucket_by_cache_mix``; this module reuses
that helper rather than keeping a second, differently-shaped estimate of the
same tokens.
"""

from __future__ import annotations

import types

import pytest

from headroom.proxy import cost
from headroom.proxy import savings_tracker as st
from headroom.proxy.savings_tracker import (
    DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN,
    _estimate_tool_schema_savings_usd,
    estimate_request_savings_usd,
)

LIST = 5.0 / 1_000_000
READ = 0.5 / 1_000_000
WRITE = 6.25 / 1_000_000
MODEL = "claude-opus-5"


def _fake_litellm(model_cost: dict) -> types.SimpleNamespace:
    # cost_per_token succeeding makes _resolve_litellm_model return the name as-is.
    return types.SimpleNamespace(
        model_cost=model_cost,
        cost_per_token=lambda **_kw: (0.0, 0.0),
    )


@pytest.fixture
def priced(monkeypatch):
    monkeypatch.setattr(
        st,
        "_get_litellm_module",
        lambda: _fake_litellm(
            {
                MODEL: {
                    "input_cost_per_token": LIST,
                    "cache_read_input_token_cost": READ,
                    "cache_creation_input_token_cost": WRITE,
                }
            }
        ),
    )


def test_a_fully_warm_request_prices_deferral_at_the_cache_read_rate(priced):
    got = _estimate_tool_schema_savings_usd(MODEL, 1_000_000, cache_read_tokens=100_000)
    assert got == pytest.approx(1_000_000 * READ)
    # The rate this replaces would have claimed the full input price -- 10x.
    assert got * 10 == pytest.approx(1_000_000 * LIST)


def test_a_cold_request_prices_deferral_at_the_cache_write_rate(priced):
    got = _estimate_tool_schema_savings_usd(MODEL, 1_000_000, cache_write_tokens=100_000)
    assert got == pytest.approx(1_000_000 * WRITE)


def test_a_mixed_request_splits_by_its_own_observed_mix(priced):
    # 50k read / 30k write / 20k uncached -> the deferred tokens are assumed to
    # have ridden the same prefix in the same proportions.
    got = _estimate_tool_schema_savings_usd(
        MODEL,
        100_000,
        cache_read_tokens=50_000,
        cache_write_tokens=30_000,
        uncached_tokens=20_000,
    )
    expected = 50_000 * READ + 30_000 * WRITE + 20_000 * LIST
    assert got == pytest.approx(expected)


def test_no_billed_breakdown_falls_back_to_list_price(priced):
    # Matches _bucket_by_cache_mix's documented behaviour, so the cost card and
    # this estimator cannot disagree about the same request.
    assert _estimate_tool_schema_savings_usd(MODEL, 1_000_000) == pytest.approx(1_000_000 * LIST)


def test_missing_cache_prices_price_exactly_as_the_cost_card_does(monkeypatch):
    """Both real pricing paths, on a catalog entry with no cache rates at all.

    A hardcoded provider multiplier here would have quoted $0.50/M for a fully
    read-cached request while ``CostTracker`` quoted $5/M for the same tokens of
    the same request -- and would have overstated savings for any provider that
    does not bill cache like Anthropic. Both sides now read the absent rate as
    the list price, via the one shared policy in ``cost._cache_input_rates``.
    """
    catalog = {"some-model": {"input_cost_per_token": LIST}}
    monkeypatch.setattr(st, "_get_litellm_module", lambda: _fake_litellm(catalog))
    monkeypatch.setattr(cost, "_get_litellm_module", lambda: _fake_litellm(catalog))
    monkeypatch.setattr("headroom.pricing.litellm_pricing.resolve_litellm_model", lambda m: m)
    st._resolve_litellm_model.cache_clear()

    card = cost.CostTracker()._get_cache_prices("some-model")
    assert card == (LIST, LIST, LIST)
    for mix, card_rate in zip(
        ("cache_read_tokens", "cache_write_tokens", "uncached_tokens"), card, strict=True
    ):
        got = _estimate_tool_schema_savings_usd("some-model", 1_000_000, **{mix: 1})
        assert got == pytest.approx(1_000_000 * card_rate)


def test_free_model_prices_as_zero(monkeypatch):
    monkeypatch.setattr(
        st,
        "_get_litellm_module",
        lambda: _fake_litellm(
            {
                "free-model": {
                    "input_cost_per_token": 0.0,
                    "cache_read_input_token_cost": 0.0,
                    "cache_creation_input_token_cost": 0.0,
                }
            }
        ),
    )
    assert _estimate_tool_schema_savings_usd("free-model", 1_000_000, cache_read_tokens=1) == 0.0


@pytest.mark.parametrize("litellm", [None, "empty"])
def test_unpriced_model_falls_back_to_the_default_rate(monkeypatch, litellm):
    monkeypatch.setattr(
        st,
        "_get_litellm_module",
        (lambda: None) if litellm is None else (lambda: _fake_litellm({})),
    )
    st._resolve_litellm_model.cache_clear()
    got = _estimate_tool_schema_savings_usd("unknown-model", 1_000_000, cache_read_tokens=1)
    assert got == pytest.approx(1_000_000 * DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN)


def test_zero_and_negative_token_counts_price_as_zero(priced):
    assert _estimate_tool_schema_savings_usd(MODEL, 0, cache_read_tokens=100) == 0.0
    assert _estimate_tool_schema_savings_usd(MODEL, -5, cache_read_tokens=100) == 0.0


def test_priced_buckets_separate_message_and_deferral_rates(priced):
    result = estimate_request_savings_usd(
        MODEL,
        compression_tokens_saved=1_000_000,
        tool_schema_tokens_saved=1_000_000,
        cache_read_tokens=100_000,
    )
    # Message compression removes tokens that would have billed at full input
    # rate; deferral removes prefix tokens billed at the cache-read rate.
    assert result["compression"] == pytest.approx(1_000_000 * LIST)
    assert result["tool_schema"] == pytest.approx(1_000_000 * READ)


@pytest.mark.asyncio
async def test_record_request_forwards_the_requests_cache_mix(monkeypatch, priced):
    """The estimator only helps if the real path hands it the mix.

    ``record_request`` already receives ``cache_write_tokens`` and
    ``uncached_input_tokens``; before this change it passed only
    ``cache_read_tokens`` on, so every deferred token priced as list no matter
    how warm the request actually was.
    """
    from headroom.proxy.prometheus_metrics import PrometheusMetrics

    seen: dict = {}
    real = st.estimate_request_savings_usd

    def _capture(model, **kwargs):
        seen.update(kwargs)
        return real(model, **kwargs)

    monkeypatch.setattr("headroom.proxy.prometheus_metrics.estimate_request_savings_usd", _capture)

    await PrometheusMetrics().record_request(
        provider="anthropic",
        model=MODEL,
        input_tokens=100_000,
        output_tokens=100,
        tokens_saved=0,
        latency_ms=1.0,
        cache_read_tokens=50_000,
        cache_write_tokens=30_000,
        uncached_input_tokens=20_000,
        tool_search_saved=100_000,
    )

    assert seen["cache_read_tokens"] == 50_000
    assert seen["cache_write_tokens"] == 30_000
    assert seen["uncached_tokens"] == 20_000
    # Priced by that mix, not at list: list would have been 100_000 * LIST.
    assert real(
        MODEL,
        tool_schema_tokens_saved=100_000,
        **{k: seen[k] for k in ("cache_read_tokens", "cache_write_tokens", "uncached_tokens")},
    )["tool_schema"] == pytest.approx(50_000 * READ + 30_000 * WRITE + 20_000 * LIST)
