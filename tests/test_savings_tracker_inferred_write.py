"""Lifetime input cost must not double-charge inferred cache writes.

OpenAI-style traffic has no provider write counter, so Headroom derives
``cache_write_tokens`` from the uncached slice (``cache_inferred=True``) — the
write is the *same tokens* as ``uncached``, not an extra billed bucket.
``estimate_request_savings_usd`` already drops it via ``CacheMix.normalized()``;
``_estimate_input_cost_usd`` did not, so ``SavingsTracker.record_lifetime_request``
priced the same tokens twice and inflated the lifetime input cost
(``/stats-lifetime`` ``cost.input_usd``) for every inferred-write request.
"""

from __future__ import annotations

import types

from headroom.proxy import savings_tracker as st
from headroom.proxy.savings_tracker import (
    SavingsTracker,
    _estimate_input_cost_usd,
)


def _priced_litellm() -> types.SimpleNamespace:
    # gpt-4o at $2.50/M input; a cache write on a real (disjoint) provider
    # is priced at the same input rate so the two calls below differ only by
    # whether the write is inferred.
    return types.SimpleNamespace(
        model_cost={
            "gpt-4o": {"input_cost_per_token": 2.5e-06},
        },
        cost_per_token=lambda **_kw: (0.0, 0.0),
    )


def test_inferred_write_is_not_charged_on_top_of_uncached(monkeypatch):
    monkeypatch.setattr(st, "_get_litellm_module", _priced_litellm)
    # 1000 inferred write tokens ARE the 1000 uncached tokens: charge once.
    got = _estimate_input_cost_usd(
        "gpt-4o",
        1000,
        cache_write_tokens=1000,
        uncached_input_tokens=1000,
        cache_inferred=True,
    )
    assert got == 0.0025


def test_a_disjoint_write_is_still_priced_separately(monkeypatch):
    monkeypatch.setattr(st, "_get_litellm_module", _priced_litellm)
    # Without the inferred flag a write is a real, separately billed bucket.
    got = _estimate_input_cost_usd(
        "gpt-4o",
        1000,
        cache_write_tokens=1000,
        uncached_input_tokens=1000,
        cache_inferred=False,
    )
    assert got == 0.005


def test_default_stays_list_price_for_non_inferred_calls(monkeypatch):
    """Backward compatible: omitting the flag keeps the previous behavior."""
    monkeypatch.setattr(st, "_get_litellm_module", _priced_litellm)
    got = _estimate_input_cost_usd(
        "gpt-4o",
        1000,
        cache_write_tokens=1000,
        uncached_input_tokens=1000,
    )
    assert got == 0.005


def test_lifetime_input_cost_ignores_inferred_write(monkeypatch, tmp_path):
    monkeypatch.setattr(st, "_get_litellm_module", _priced_litellm)
    tracker = SavingsTracker(path=tmp_path / "savings.json", save_flush_every=1)

    tracker.record_lifetime_request(
        model="gpt-4o",
        provider="openai",
        stack="default",
        input_tokens=1000,
        cache_write_tokens=1000,
        uncached_input_tokens=1000,
        cache_inferred=True,
    )

    # 1000 tokens at $2.50/M, charged once — not $0.005.
    assert tracker.lifetime_response()["cost"]["input_usd"] == 0.0025
