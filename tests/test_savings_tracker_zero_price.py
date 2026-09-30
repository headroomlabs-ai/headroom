"""Regression: free (0-priced) models must not be billed the fallback rate.

`_estimate_compression_savings_usd` / `_estimate_input_cost_usd` read
`input_cost_per_token` from litellm and used `if not input_cost_per_token: raise`,
which treats a legitimate `0.0` (a free / local / vendored-at-0 model) as "price
unavailable" and falls back to DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN ($3/M) —
fabricating savings/cost for a model that costs nothing. A missing key (unknown
model) must still fall back.
"""

from __future__ import annotations

import types

from headroom.proxy import savings_tracker as st
from headroom.proxy.savings_tracker import (
    DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN,
    DEFAULT_FALLBACK_OUTPUT_COST_PER_TOKEN,
    _estimate_compression_savings_usd,
    _estimate_input_cost_usd,
    _estimate_output_savings_usd,
)


def _fake_litellm(model_cost: dict) -> types.SimpleNamespace:
    # cost_per_token succeeding makes _resolve_litellm_model return the name as-is.
    return types.SimpleNamespace(
        model_cost=model_cost,
        cost_per_token=lambda **_kw: (0.0, 0.0),
    )


def test_compression_savings_zero_for_free_model(monkeypatch):
    monkeypatch.setattr(
        st,
        "_get_litellm_module",
        lambda: _fake_litellm({"free-model": {"input_cost_per_token": 0.0}}),
    )
    assert _estimate_compression_savings_usd("free-model", 1_000_000) == 0.0


def test_compression_savings_falls_back_for_unknown_model(monkeypatch):
    # Model absent from litellm → input_cost_per_token is None → fall back.
    monkeypatch.setattr(st, "_get_litellm_module", lambda: _fake_litellm({}))
    got = _estimate_compression_savings_usd("unknown-model", 1_000_000)
    assert got == 1_000_000 * DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN


def test_compression_savings_uses_real_price_for_paid_model(monkeypatch):
    price = 3.0 / 1_000_000
    monkeypatch.setattr(
        st,
        "_get_litellm_module",
        lambda: _fake_litellm({"paid-model": {"input_cost_per_token": price}}),
    )
    assert _estimate_compression_savings_usd("paid-model", 1_000_000) == 1_000_000 * price


def test_input_cost_zero_for_free_model(monkeypatch):
    monkeypatch.setattr(
        st,
        "_get_litellm_module",
        lambda: _fake_litellm({"free-model": {"input_cost_per_token": 0.0}}),
    )
    assert _estimate_input_cost_usd("free-model", 500_000) == 0.0


def test_output_savings_zero_for_free_model(monkeypatch):
    # output_cost_per_token == 0.0 (free model) must yield $0, not the fallback.
    monkeypatch.setattr(
        st,
        "_get_litellm_module",
        lambda: _fake_litellm({"free-model": {"output_cost_per_token": 0.0}}),
    )
    assert _estimate_output_savings_usd("free-model", 1_000_000) == 0.0


def test_output_savings_falls_back_for_unknown_model(monkeypatch):
    # Model absent from litellm → output_cost_per_token is None → fall back.
    monkeypatch.setattr(st, "_get_litellm_module", lambda: _fake_litellm({}))
    got = _estimate_output_savings_usd("unknown-model", 1_000_000)
    assert got == 1_000_000 * DEFAULT_FALLBACK_OUTPUT_COST_PER_TOKEN


def test_output_savings_uses_real_price_for_paid_model(monkeypatch):
    price = 15.0 / 1_000_000
    monkeypatch.setattr(
        st,
        "_get_litellm_module",
        lambda: _fake_litellm({"paid-model": {"output_cost_per_token": price}}),
    )
    assert _estimate_output_savings_usd("paid-model", 1_000_000) == 1_000_000 * price
