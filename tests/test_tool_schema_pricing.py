"""Tool-schema deferral must be priced at the cache-read rate, not full input.

Reported from a real install: pricing deferral with
``_estimate_compression_savings_usd`` (full ``input_cost_per_token``) pushed the
blended $/token savings figure from $4.99/M to $32.88/M on traffic where
deferral ran 5.59x message compression -- a rate above the input list price of
every model in the mix, which no genuine input-token saving can reach.

The correct counterfactual for a deferred schema token is the cache-READ rate:
schemas are byte-stable prompt-prefix content, so had they stayed in the
prompt, caching providers would have billed them at the cache-read discount
(0.1x input on Anthropic) on every turn after the first. This module's own
PERF docstring already applies the same principle to prompt-cache reads
("valued at cache_read price ... prevents overstating dollar savings").
"""

from __future__ import annotations

import types

import pytest

from headroom.proxy import savings_tracker as st
from headroom.proxy.savings_tracker import (
    DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN,
    TOOL_SCHEMA_CACHE_READ_FRACTION,
    _estimate_tool_schema_savings_usd,
    estimate_request_savings_usd,
)


def _fake_litellm(model_cost: dict) -> types.SimpleNamespace:
    # cost_per_token succeeding makes _resolve_litellm_model return the name as-is.
    return types.SimpleNamespace(
        model_cost=model_cost,
        cost_per_token=lambda **_kw: (0.0, 0.0),
    )


def test_uses_the_models_cache_read_price_when_available(monkeypatch):
    monkeypatch.setattr(
        st,
        "_get_litellm_module",
        lambda: _fake_litellm(
            {
                "claude-opus-5": {
                    "input_cost_per_token": 5.0 / 1_000_000,
                    "cache_read_input_token_cost": 0.5 / 1_000_000,
                }
            }
        ),
    )
    got = _estimate_tool_schema_savings_usd("claude-opus-5", 1_000_000)
    assert got == pytest.approx(0.5)
    # The rate this replaces would have claimed the full input price -- 10x.
    assert got * 10 == pytest.approx(1_000_000 * (5.0 / 1_000_000))


def test_falls_back_to_a_fraction_of_input_rate_without_cache_read_price(monkeypatch):
    monkeypatch.setattr(
        st,
        "_get_litellm_module",
        lambda: _fake_litellm({"some-model": {"input_cost_per_token": 3.0 / 1_000_000}}),
    )
    got = _estimate_tool_schema_savings_usd("some-model", 1_000_000)
    assert got == pytest.approx(3.0 * TOOL_SCHEMA_CACHE_READ_FRACTION)


def test_free_model_prices_as_zero(monkeypatch):
    monkeypatch.setattr(
        st,
        "_get_litellm_module",
        lambda: _fake_litellm(
            {
                "free-model": {
                    "input_cost_per_token": 0.0,
                    "cache_read_input_token_cost": 0.0,
                }
            }
        ),
    )
    assert _estimate_tool_schema_savings_usd("free-model", 1_000_000) == 0.0


def test_unknown_model_falls_back_to_fraction_of_default_rate(monkeypatch):
    monkeypatch.setattr(st, "_get_litellm_module", lambda: _fake_litellm({}))
    got = _estimate_tool_schema_savings_usd("unknown-model", 1_000_000)
    expected = 1_000_000 * TOOL_SCHEMA_CACHE_READ_FRACTION * DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN
    assert got == pytest.approx(expected)


def test_unavailable_litellm_falls_back_to_fraction_of_default_rate(monkeypatch):
    monkeypatch.setattr(st, "_get_litellm_module", lambda: None)
    got = _estimate_tool_schema_savings_usd("any-model", 1_000_000)
    expected = 1_000_000 * TOOL_SCHEMA_CACHE_READ_FRACTION * DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN
    assert got == pytest.approx(expected)


def test_priced_buckets_separate_message_and_deferral_rates(monkeypatch):
    monkeypatch.setattr(
        st,
        "_get_litellm_module",
        lambda: _fake_litellm(
            {
                "claude-opus-5": {
                    "input_cost_per_token": 5.0 / 1_000_000,
                    "cache_read_input_token_cost": 0.5 / 1_000_000,
                }
            }
        ),
    )
    priced = estimate_request_savings_usd(
        "claude-opus-5",
        compression_tokens_saved=1_000_000,
        tool_schema_tokens_saved=1_000_000,
    )
    # Message compression removes tokens that would have billed at full input
    # rate; deferral removes tokens that would have billed at cache-read rate.
    assert priced["compression"] == pytest.approx(5.0)
    assert priced["tool_schema"] == pytest.approx(0.5)
