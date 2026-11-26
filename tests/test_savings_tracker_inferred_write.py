from __future__ import annotations

import tempfile
import types
from pathlib import Path

import pytest

from headroom.proxy import savings_tracker as st
from headroom.proxy.savings_tracker import (
    DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN,
    SavingsTracker,
    _estimate_input_cost_usd,
)


def _fake_litellm_gpt4o() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model_cost={
            "gpt-4o": {
                "input_cost_per_token": 2.5e-06,
                "cache_read_input_token_cost": 1.25e-06,
                "cache_creation_input_token_cost": 2.5e-06,
            }
        },
        cost_per_token=lambda **_kw: (0.0, 0.0),
    )


def test_issue_4961_reproducer_with_inferred_flag(monkeypatch):
    monkeypatch.setattr(st, "_get_litellm_module", _fake_litellm_gpt4o)
    # Inferred write: the 1000 write tokens ARE the 1000 uncached tokens.
    cost = _estimate_input_cost_usd(
        "gpt-4o",
        1000,
        cache_write_tokens=1000,
        uncached_input_tokens=1000,
        cache_inferred=True,
    )
    # Expected: 1000 tokens at $2.50/M ($0.0025), not charged twice ($0.005).
    assert cost == pytest.approx(0.0025)


def test_disjoint_write_is_priced_separately(monkeypatch):
    monkeypatch.setattr(st, "_get_litellm_module", _fake_litellm_gpt4o)
    cost = _estimate_input_cost_usd(
        "gpt-4o",
        1000,
        cache_write_tokens=1000,
        uncached_input_tokens=1000,
        cache_inferred=False,
    )
    # Disjoint write: 1000 uncached + 1000 write = 0.0025 + 0.0025 = 0.005.
    assert cost == pytest.approx(0.005)


def test_default_cache_inferred_is_false(monkeypatch):
    monkeypatch.setattr(st, "_get_litellm_module", _fake_litellm_gpt4o)
    # Backward compatibility: omitting cache_inferred defaults to False.
    cost = _estimate_input_cost_usd(
        "gpt-4o",
        1000,
        cache_write_tokens=1000,
        uncached_input_tokens=1000,
    )
    assert cost == pytest.approx(0.005)


def test_inferred_write_with_partial_cache_read(monkeypatch):
    monkeypatch.setattr(st, "_get_litellm_module", _fake_litellm_gpt4o)
    # 400 cached read (1.25e-6) + 600 uncached (2.5e-6) + 600 inferred write (zeroed)
    cost = _estimate_input_cost_usd(
        "gpt-4o",
        1000,
        cache_read_tokens=400,
        cache_write_tokens=600,
        uncached_input_tokens=600,
        cache_inferred=True,
    )
    expected = 400 * 1.25e-06 + 600 * 2.5e-06  # 0.0005 + 0.0015 = 0.002
    assert cost == pytest.approx(expected)


def test_inferred_write_fallback_pricing(monkeypatch):
    monkeypatch.setattr(st, "_get_litellm_module", lambda: types.SimpleNamespace(model_cost={}))
    cost = _estimate_input_cost_usd(
        "unknown-model",
        1000,
        cache_write_tokens=1000,
        uncached_input_tokens=1000,
        cache_inferred=True,
    )
    # Fallback should price 1000 chargeable tokens, not 2000.
    expected = 1000 * DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN
    assert cost == pytest.approx(expected)


def test_inferred_write_without_uncached_breakdown_falls_back_to_total(monkeypatch):
    monkeypatch.setattr(st, "_get_litellm_module", _fake_litellm_gpt4o)
    cost = _estimate_input_cost_usd(
        "gpt-4o",
        1000,
        cache_write_tokens=1000,
        uncached_input_tokens=0,
        cache_inferred=True,
    )
    assert cost == pytest.approx(0.0025)


def test_record_lifetime_request_inferred_write(monkeypatch):
    monkeypatch.setattr(st, "_get_litellm_module", _fake_litellm_gpt4o)
    with tempfile.TemporaryDirectory() as tmp:
        tracker = SavingsTracker(path=str(Path(tmp) / "savings.json"))
        tracker.record_lifetime_request(
            model="gpt-4o",
            provider="openai",
            stack="chat",
            input_tokens=1000,
            cache_write_tokens=1000,
            uncached_input_tokens=1000,
            cache_inferred=True,
        )
        resp = tracker.lifetime_response()
        assert resp["cost"]["input_usd"] == pytest.approx(0.0025)


def test_record_request_inferred_write(monkeypatch):
    monkeypatch.setattr(st, "_get_litellm_module", _fake_litellm_gpt4o)
    with tempfile.TemporaryDirectory() as tmp:
        tracker = SavingsTracker(path=str(Path(tmp) / "savings.json"))
        tracker.record_request(
            model="gpt-4o",
            input_tokens=1000,
            tokens_saved=0,
            cache_write_tokens=1000,
            uncached_input_tokens=1000,
            cache_inferred=True,
        )
        assert tracker._state["lifetime"]["total_input_cost_usd"] == pytest.approx(0.0025)


@pytest.mark.anyio
async def test_prometheus_metrics_record_request_inferred_write(monkeypatch):
    from headroom.proxy.prometheus_metrics import PrometheusMetrics

    monkeypatch.setattr(st, "_get_litellm_module", _fake_litellm_gpt4o)
    with tempfile.TemporaryDirectory() as tmp:
        tracker = SavingsTracker(path=str(Path(tmp) / "savings.json"))
        metrics = PrometheusMetrics(savings_tracker=tracker, stateless=True)
        await metrics.record_request(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=50,
            tokens_saved=0,
            latency_ms=100.0,
            cache_write_tokens=1000,
            uncached_input_tokens=1000,
            cache_inferred=True,
        )
        assert tracker._state["lifetime"]["total_input_cost_usd"] == pytest.approx(0.0025)
        resp = tracker.lifetime_response()
        assert resp["cost"]["input_usd"] == pytest.approx(0.0025)


def test_request_outcome_updates_lifetime_cost_for_inferred_write(tmp_path, monkeypatch):
    import asyncio

    from headroom.proxy.outcome import RequestOutcome, emit_request_outcome
    from headroom.proxy.prometheus_metrics import PrometheusMetrics

    monkeypatch.setattr(st, "_get_litellm_module", _fake_litellm_gpt4o)
    tracker = SavingsTracker(path=str(tmp_path / "proxy_savings.json"))
    handler = types.SimpleNamespace(
        metrics=PrometheusMetrics(savings_tracker=tracker, stateless=True),
        cost_tracker=None,
        logger=None,
        pipeline_extensions=None,
    )

    initial_cost = tracker.lifetime_response()["cost"]["input_usd"]
    asyncio.run(
        emit_request_outcome(
            handler,
            RequestOutcome(
                request_id="issue-4961",
                provider="openai",
                model="gpt-4o",
                original_tokens=1000,
                optimized_tokens=1000,
                provider_input_tokens=1000,
                output_tokens=10,
                tokens_saved=0,
                attempted_input_tokens=1000,
                cache_write_tokens=1000,
                uncached_input_tokens=1000,
                cache_inferred=True,
            ),
        )
    )
    after_cost = tracker.lifetime_response()["cost"]["input_usd"]

    assert after_cost - initial_cost == pytest.approx(0.0025)
