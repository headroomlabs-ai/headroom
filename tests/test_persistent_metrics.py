"""Tests for durable, aggregate-only proxy Lifetime metrics."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from headroom.proxy.persistent_metrics import PersistentMetricsState

FIXED_NOW = datetime(2026, 7, 14, 8, 30, tzinfo=timezone.utc)


def _new_state() -> PersistentMetricsState:
    return PersistentMetricsState(now=lambda: FIXED_NOW)


def test_snapshot_accumulates_request_token_cache_cost_and_waste_metrics() -> None:
    state = _new_state()

    state.record_request(
        provider="anthropic",
        stack="codex",
        model="claude-test",
        input_tokens=100,
        output_tokens=20,
        attempted_input_tokens=150,
        tokens_saved=50,
        cached=True,
        cache_read_tokens=80,
        cache_write_tokens=40,
        cache_write_5m_tokens=10,
        cache_write_1h_tokens=30,
        uncached_input_tokens=20,
        input_usd=0.4,
        compression_savings_usd=0.2,
        cache_savings_usd=0.1,
        waste_signals={"repetition": 7},
    )
    state.record_failed(provider="anthropic", model="claude-test")
    state.record_rate_limited(provider="anthropic", model="claude-test")
    state.record_cache_bust(tokens_lost=9)
    state.record_cache_miss(provider="anthropic", reason="prefix_change")

    snapshot = state.snapshot(persistence={"enabled": True, "healthy": True})

    assert snapshot["scope"] == "lifetime"
    assert snapshot["requests"] == {
        "total": 1,
        "cached": 1,
        "failed": 1,
        "rate_limited": 1,
        "by_provider": {"anthropic": 1},
        "by_stack": {"codex": 1},
    }
    assert snapshot["tokens"] == {
        "input": 100,
        "output": 20,
        "attempted_input": 150,
        "saved": 50,
        "token_savings_percent": pytest.approx(50 / 150 * 100),
    }
    assert snapshot["prefix_cache"]["requests"] == 1
    assert snapshot["prefix_cache"]["hit_requests"] == 1
    assert snapshot["prefix_cache"]["cache_read_tokens"] == 80
    assert snapshot["prefix_cache"]["cache_write_tokens"] == 40
    assert snapshot["prefix_cache"]["cache_hit_rate"] == 100.0
    assert snapshot["prefix_cache"]["ttl_1h_percent"] == 75.0
    assert snapshot["prefix_cache"]["ttl_5m_percent"] == 25.0
    assert snapshot["prefix_cache"]["bust_count"] == 1
    assert snapshot["prefix_cache"]["bust_tokens"] == 9
    assert snapshot["prefix_cache"]["misses_by_reason"] == {"prefix_change": 1}
    assert snapshot["cost"] == {
        "input_usd": 0.4,
        "compression_savings_usd": 0.2,
        "cache_savings_usd": 0.1,
    }
    assert snapshot["waste_signals"] == {"repetition": 7}
    assert snapshot["by_model"]["claude-test"]["input_tokens"] == 100


def test_snapshot_uses_null_for_ratios_without_a_denominator() -> None:
    snapshot = _new_state().snapshot(persistence={"enabled": True, "healthy": True})

    assert snapshot["tokens"]["token_savings_percent"] is None
    assert snapshot["prefix_cache"]["cache_hit_rate"] is None
    assert snapshot["prefix_cache"]["ttl_1h_percent"] is None
    assert snapshot["prefix_cache"]["ttl_5m_percent"] is None


def test_candidate_models_remain_available_until_the_two_hundred_and_first_model() -> None:
    state = _new_state()
    for index in range(200):
        state.record_request(
            provider="provider",
            stack="stack",
            model=f"model-{index:03}",
            input_tokens=index + 1,
        )

    persisted = state.to_dict()
    snapshot = state.snapshot(persistence={"enabled": True, "healthy": True})

    assert len(persisted["models"]["tracked"]) == 200
    assert "model-000" not in snapshot["by_model"]
    assert snapshot["by_model"]["other"]["input_tokens"] == sum(range(1, 101))


def test_two_hundred_and_first_model_permanently_compacts_non_top_candidates() -> None:
    state = _new_state()
    for index in range(201):
        state.record_request(
            provider="provider",
            stack="stack",
            model=f"model-{index:03}",
            input_tokens=index + 1,
        )

    persisted = state.to_dict()
    snapshot = state.snapshot(persistence={"enabled": True, "healthy": True})

    assert len(persisted["models"]["tracked"]) == 100
    assert set(snapshot["by_model"]) == {
        *(f"model-{index:03}" for index in range(101, 201)),
        "other",
    }
    assert snapshot["by_model"]["other"]["input_tokens"] == sum(range(1, 102))


def test_state_normalizes_invalid_values_and_unknown_dimension_labels() -> None:
    state = PersistentMetricsState(
        {
            "requests": {"total": "not-a-number"},
            "tokens": {"input": float("nan"), "output": -3},
            "models": {"tracked": {"unknown": {"input_tokens": "7"}}},
        },
        now=lambda: FIXED_NOW,
    )
    state.record_request(
        provider=" ",
        stack=None,
        model=" ",
        input_tokens=-1,
        output_tokens=float("inf"),
        waste_signals={"unrecognized": 9},
    )

    snapshot = state.snapshot(persistence={"enabled": True, "healthy": True})

    assert snapshot["tokens"]["input"] == 0
    assert snapshot["tokens"]["output"] == 0
    assert snapshot["requests"]["by_provider"] == {"other": 1}
    assert snapshot["requests"]["by_stack"] == {"other": 1}
    assert snapshot["by_model"]["other"]["input_tokens"] == 7
    assert snapshot["waste_signals"] == {"other": 9}
