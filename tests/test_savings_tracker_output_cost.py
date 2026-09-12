"""Lifetime output spend, so a bill-share rate has a denominator that fits it.

`total_input_cost_usd` has been tracked since the tracker existed; the tokens
the model EMITTED were priced nowhere durable. Any "what share of my bill did
Headroom remove" figure therefore had to divide savings that include output
shaping by an input-only denominator, which overstates the rate. These pin the
new `total_output_cost_usd` on the lifetime block and its per-bucket delta on
the rollup series, plus the legacy-checkpoint path.
"""

from __future__ import annotations

import types

from headroom.proxy import savings_tracker as st
from headroom.proxy.savings_tracker import SavingsTracker, _normalize_history_entry


def _priced_litellm() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model_cost={
            "test-model": {
                "input_cost_per_token": 1e-06,
                "output_cost_per_token": 5e-06,
            }
        },
        cost_per_token=lambda **_kw: (0.0, 0.0),
    )


def _tracker(tmp_path) -> SavingsTracker:
    return SavingsTracker(path=tmp_path / "savings.json", save_flush_every=1)


def test_emitted_output_tokens_are_priced_onto_lifetime(monkeypatch, tmp_path):
    monkeypatch.setattr(st, "_get_litellm_module", _priced_litellm)
    tracker = _tracker(tmp_path)

    tracker.record_request(
        model="test-model",
        input_tokens=1_000,
        tokens_saved=100,
        output_tokens=2_000,
    )

    lifetime = tracker.snapshot()["lifetime"]
    # 2,000 emitted tokens at $5/M.
    assert lifetime["total_output_cost_usd"] == 0.01
    # The input side is untouched by the new field.
    assert lifetime["total_input_cost_usd"] > 0


def test_output_cost_is_disjoint_from_output_savings(monkeypatch, tmp_path):
    """Emitted and not-emitted are separate counts at the same rate."""
    monkeypatch.setattr(st, "_get_litellm_module", _priced_litellm)
    tracker = _tracker(tmp_path)

    tracker.record_request(
        model="test-model",
        input_tokens=1_000,
        tokens_saved=0,
        output_tokens=2_000,
        output_tokens_saved=400,
    )

    lifetime = tracker.snapshot()["lifetime"]
    assert lifetime["total_output_cost_usd"] == 0.01  # 2,000 emitted
    assert lifetime["output_savings_usd"] == 0.002  # 400 not emitted


def test_bill_share_rate_is_computable_and_below_the_input_only_one(monkeypatch, tmp_path):
    """The reason the field exists: an input-only denominator overstates."""
    monkeypatch.setattr(st, "_get_litellm_module", _priced_litellm)
    tracker = _tracker(tmp_path)

    tracker.record_request(
        model="test-model",
        input_tokens=1_000,
        tokens_saved=100,
        output_tokens=2_000,
        output_tokens_saved=400,
    )

    lifetime = tracker.snapshot()["lifetime"]
    saved = lifetime["compression_savings_usd"] + lifetime["output_savings_usd"]
    input_only = saved / (saved + lifetime["total_input_cost_usd"])
    whole_bill = saved / (
        saved + lifetime["total_input_cost_usd"] + lifetime["total_output_cost_usd"]
    )
    assert whole_bill < input_only


def test_legacy_checkpoints_default_the_field_rather_than_dropping(tmp_path):
    normalized = _normalize_history_entry(
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "total_tokens_saved": 10,
            "compression_savings_usd": 0.5,
        }
    )
    assert normalized is not None
    assert normalized["total_output_cost_usd"] == 0.0


def test_rollup_series_carries_the_per_bucket_output_spend(monkeypatch, tmp_path):
    monkeypatch.setattr(st, "_get_litellm_module", _priced_litellm)
    tracker = _tracker(tmp_path)

    for _ in range(3):
        tracker.record_request(
            model="test-model",
            input_tokens=1_000,
            tokens_saved=100,
            output_tokens=2_000,
        )

    daily = tracker.history_response()["series"]["daily"]
    assert daily, "expected at least one bucket"
    bucket = daily[-1]
    # Three requests at $0.01 of emitted output each, as a delta and a total.
    assert bucket["total_output_cost_usd_delta"] == 0.03
    assert bucket["total_output_cost_usd"] == 0.03


def test_lifetime_output_spend_survives_a_restart(monkeypatch, tmp_path):
    """The field is a LIFETIME total, so a reload must not restart it at 0.

    The lifetime block accumulates in memory and is checkpointed into history;
    on load the block has to be recovered from both. Without that, request two
    lands on a zeroed counter and the rollup's ``max(delta, 0)`` clamp hides
    the regression by reporting a flat total instead of a negative one.
    """
    monkeypatch.setattr(st, "_get_litellm_module", _priced_litellm)
    path = tmp_path / "savings.json"

    def record(tracker: SavingsTracker) -> None:
        tracker.record_request(
            model="test-model",
            input_tokens=1_000,
            tokens_saved=100,
            output_tokens=2_000,
            output_tokens_saved=400,
        )

    first = SavingsTracker(path=path, save_flush_every=1)
    record(first)
    assert first.snapshot()["lifetime"]["total_output_cost_usd"] == 0.01

    reloaded = SavingsTracker(path=path, save_flush_every=1)
    restored = reloaded.snapshot()["lifetime"]
    assert restored["total_output_cost_usd"] == 0.01
    assert restored["output_savings_usd"] == 0.002
    assert restored["output_tokens_saved"] == 400

    record(reloaded)
    lifetime = reloaded.snapshot()["lifetime"]
    assert lifetime["total_output_cost_usd"] == 0.02
    assert lifetime["output_savings_usd"] == 0.004
    assert lifetime["output_tokens_saved"] == 800

    bucket = reloaded.history_response()["series"]["daily"][-1]
    assert bucket["total_output_cost_usd"] == 0.02
    assert bucket["total_output_cost_usd_delta"] == 0.02
    assert bucket["output_savings_usd_delta"] == 0.004
