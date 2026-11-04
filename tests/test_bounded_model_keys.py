"""A client-supplied ``model`` string must not grow per-model state without bound."""

from __future__ import annotations

import json

from headroom.proxy.cost import CostTracker
from headroom.proxy.persistent_metrics import (
    MAX_LABEL_LENGTH,
    MAX_TRACKED_MODELS,
)
from headroom.proxy.savings_tracker import SavingsTracker

_NAMES = MAX_TRACKED_MODELS + 50


def test_cost_tracker_per_model_maps_are_bounded():
    tracker = CostTracker()
    for i in range(_NAMES):
        tracker.record_tokens(f"m-{i}-" + "x" * 2000, tokens_saved=1, tokens_sent=10)

    assert len(tracker._requests_by_model) == MAX_TRACKED_MODELS + 1
    assert len(tracker._tokens_sent_by_model) == MAX_TRACKED_MODELS + 1
    assert tracker._requests_by_model["other"] == _NAMES - MAX_TRACKED_MODELS
    assert all(len(k) <= MAX_LABEL_LENGTH for k in tracker._requests_by_model)
    # Totals stay exact: over-cap requests are bucketed, not dropped.
    assert sum(tracker._tokens_sent_by_model.values()) == 10 * _NAMES


def test_estimated_basis_warning_set_is_bounded():
    import headroom.proxy.cost as cost_mod

    cost_mod._warned_estimated_basis_models.clear()
    tracker = CostTracker()
    for i in range(_NAMES):
        tracker.record_tokens(f"w-{i}", tokens_saved=0, tokens_sent=10)
    assert len(cost_mod._warned_estimated_basis_models) <= MAX_TRACKED_MODELS + 1
    cost_mod._warned_estimated_basis_models.clear()


def test_savings_tracker_by_model_is_bounded_and_persisted_small(tmp_path):
    path = tmp_path / "proxy_savings.json"
    tracker = SavingsTracker(path=str(path), save_flush_every=1)
    for i in range(_NAMES):
        tracker.record_request(model=f"s-{i}-" + "y" * 2000, input_tokens=100, tokens_saved=10)

    by_model = json.loads(path.read_text(encoding="utf-8"))["by_model"]
    assert len(by_model) == MAX_TRACKED_MODELS + 1
    assert all(len(k) <= MAX_LABEL_LENGTH for k in by_model)
    assert by_model["other"]["requests"] == _NAMES - MAX_TRACKED_MODELS
    assert sum(e["tokens_saved"] for e in by_model.values()) == 10 * _NAMES


def test_overflowing_passthrough_names_stay_out_of_processed_totals():
    tracker = CostTracker()
    for i in range(MAX_TRACKED_MODELS):
        tracker.record_tokens(f"passthrough:batch:m{i}", tokens_saved=1, tokens_sent=1)
    tracker.record_tokens("passthrough:batch:next", tokens_saved=10, tokens_sent=90)

    stats = tracker.stats()
    assert stats["total_tokens_saved"] == 0
    assert stats["total_input_tokens"] == 0
    assert "other" not in tracker._requests_by_model
    assert tracker._requests_by_model["passthrough:other"] == 1


def test_estimated_basis_warning_set_is_capped_across_resets():
    import headroom.proxy.cost as cost_mod

    cost_mod._warned_estimated_basis_models.clear()
    tracker = CostTracker()
    for i in range(3):
        for j in range(MAX_TRACKED_MODELS):
            tracker.record_tokens(f"r{i}-{j}", tokens_saved=0, tokens_sent=10)
        tracker.reset_runtime()
    assert len(cost_mod._warned_estimated_basis_models) <= MAX_TRACKED_MODELS
    cost_mod._warned_estimated_basis_models.clear()


def test_savings_history_checkpoints_clip_model_names(tmp_path):
    tracker = SavingsTracker(path=str(tmp_path / "proxy_savings.json"), save_flush_every=1)
    tracker.record_request(model="h-" + "z" * 2000, input_tokens=100, tokens_saved=10)

    history = json.loads((tmp_path / "proxy_savings.json").read_text(encoding="utf-8"))["history"]
    assert history
    assert all(len(point["model"]) <= MAX_LABEL_LENGTH for point in history)


def _saturate_with_unpriced_names(tracker: CostTracker) -> None:
    for i in range(MAX_TRACKED_MODELS):
        tracker.record_tokens(f"unpriced-{i}", tokens_saved=0, tokens_sent=0)


def test_priced_request_after_saturation_keeps_aggregate_dollars():
    control = CostTracker()
    control.record_tokens("gpt-4o", tokens_saved=0, tokens_sent=100_000, uncached_tokens=100_000)
    assert control.totals() == (100_000, 0.25)

    tracker = CostTracker()
    _saturate_with_unpriced_names(tracker)
    tracker.record_tokens("gpt-4o", tokens_saved=0, tokens_sent=100_000, uncached_tokens=100_000)

    assert tracker._requests_by_model["other"] == 1
    assert tracker.totals() == (100_000, 0.25)
    assert tracker.totals()[1] == tracker.period_cost_breakdown()["total_usd"]
    assert tracker.stats()["total_input_cost_usd"] == 0.25


def test_overflow_conserves_every_priced_stats_dollar_figure():
    kwargs = {
        "tokens_saved": 40_000,
        "tokens_sent": 60_000,
        "cache_read_tokens": 20_000,
        "cache_write_tokens": 10_000,
        "uncached_tokens": 30_000,
        "output_tokens": 5_000,
        "tool_schema_saved": 8_000,
    }
    control = CostTracker()
    control.record_tokens("gpt-4o", **kwargs)
    tracker = CostTracker()
    _saturate_with_unpriced_names(tracker)
    tracker.record_tokens("gpt-4o", **kwargs)

    want, got = control.stats(), tracker.stats()
    assert want["total_cost_usd"] > 0
    assert want["cache_aware_savings_usd"] > 0
    assert want["tool_savings_usd"] > 0
    for field in (
        "total_input_cost_usd",
        "output_cost_usd",
        "total_cost_usd",
        "savings_usd",
        "cache_aware_savings_usd",
        "tool_savings_usd",
    ):
        assert got[field] == want[field], field
    assert tracker.totals() == control.totals()


def test_overflow_keeps_cache_aware_savings_for_reviewer_probe():
    kwargs = {
        "tokens_saved": 40_000,
        "tokens_sent": 60_000,
        "cache_read_tokens": 20_000,
        "cache_write_tokens": 10_000,
        "uncached_tokens": 30_000,
        "output_tokens": 5_000,
    }
    control = CostTracker()
    control.record_tokens("gpt-4o", **kwargs)
    tracker = CostTracker()
    _saturate_with_unpriced_names(tracker)
    tracker.record_tokens("gpt-4o", **kwargs)
    assert control.stats()["cache_aware_savings_usd"] > 0
    assert tracker.stats()["cache_aware_savings_usd"] == control.stats()["cache_aware_savings_usd"]
    tracker.reset_runtime()
    assert tracker.stats()["cache_aware_savings_usd"] == 0.0
    assert tracker.stats()["tool_savings_usd"] == 0.0


def test_overflowing_passthrough_names_add_no_dollars():
    tracker = CostTracker()
    for i in range(MAX_TRACKED_MODELS):
        tracker.record_tokens(f"passthrough:batch:m{i}", tokens_saved=0, tokens_sent=1)
    tracker.record_tokens(
        "passthrough:gpt-4o", tokens_saved=0, tokens_sent=100_000, uncached_tokens=100_000
    )
    assert tracker.stats()["total_input_cost_usd"] == 0.0
