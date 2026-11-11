"""Tests for per-request ``cost_usd`` in ``/stats`` recent_requests (issue #1079).

The proxy exposes a per-request billed-cost estimate so tooling (e.g. claude-hud)
can attribute cost to a Claude Code session by summing over turn_id / time window,
without an external ``ccusage`` subprocess.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient

from headroom.proxy.cost import CostTracker, request_cost_usd
from headroom.proxy.models import RequestLog
from headroom.proxy.outcome import RequestOutcome, emit_request_outcome
from headroom.proxy.prometheus_metrics import PrometheusMetrics
from headroom.proxy.request_logger import RequestLogger
from headroom.proxy.server import ProxyConfig, create_app


def _tracker(estimate):
    """A stand-in CostTracker exposing only ``estimate_cost``."""
    return SimpleNamespace(estimate_cost=estimate)


# ---------------------------------------------------------------------------
# Unit: request_cost_usd helper
# ---------------------------------------------------------------------------


def test_none_when_cost_tracker_missing() -> None:
    log = {
        "model": "claude-3-5-sonnet-20241022",
        "input_tokens_optimized": 600,
        "output_tokens": 50,
    }
    assert request_cost_usd(None, log) is None


def test_none_when_model_missing() -> None:
    tracker = _tracker(lambda **_: 0.01)
    assert request_cost_usd(tracker, {"input_tokens_optimized": 600, "output_tokens": 50}) is None


def test_none_when_input_tokens_missing() -> None:
    tracker = _tracker(lambda **_: 0.01)
    assert request_cost_usd(tracker, {"model": "claude-3-5-sonnet-20241022"}) is None


def test_passes_through_estimate_and_uses_optimized_tokens() -> None:
    captured: dict = {}

    def estimate(**kwargs):
        captured.update(kwargs)
        return 0.0135

    log = {
        "model": "claude-3-5-sonnet-20241022",
        "input_tokens_original": 1000,
        "input_tokens_optimized": 600,
        "output_tokens": 50,
    }
    assert request_cost_usd(_tracker(estimate), log) == 0.0135
    # Bills the post-compression input the provider actually saw, plus output.
    assert captured["input_tokens"] == 600
    assert captured["output_tokens"] == 50


def test_prefers_provider_billed_input_over_local_count() -> None:
    # Local tokenizer counted 600, the provider billed 800: price the 800 the
    # cost tracker records, not the local estimate.
    captured: dict = {}

    def estimate(**kwargs):
        captured.update(kwargs)
        return 0.02

    log = {
        "model": "claude-3-5-sonnet-20241022",
        "input_tokens_optimized": 600,
        "provider_input_tokens": 800,
        "output_tokens": 50,
    }
    assert request_cost_usd(_tracker(estimate), log) == 0.02
    assert captured["input_tokens"] == 800
    assert captured["output_tokens"] == 50


def test_falls_back_to_optimized_tokens_when_provider_count_unreported() -> None:
    # 0 means the provider reported no input usage, never "zero tokens billed".
    captured: dict = {}

    def estimate(**kwargs):
        captured.update(kwargs)
        return 0.01

    log = {
        "model": "m",
        "input_tokens_optimized": 600,
        "provider_input_tokens": 0,
        "output_tokens": 50,
    }
    assert request_cost_usd(_tracker(estimate), log) == 0.01
    assert captured["input_tokens"] == 600


def test_none_when_output_tokens_unknown() -> None:
    # Pricing a missing output count as zero would publish an input-only figure
    # that looks like a complete request cost.
    calls: list[dict] = []

    def estimate(**kwargs):
        calls.append(kwargs)
        return 0.002

    log = {"model": "m", "input_tokens_optimized": 100, "output_tokens": None}
    assert request_cost_usd(_tracker(estimate), log) is None
    assert (
        request_cost_usd(_tracker(estimate), {"model": "m", "input_tokens_optimized": 100}) is None
    )
    assert calls == []


def test_zero_output_tokens_is_a_real_count() -> None:
    captured: dict = {}

    def estimate(**kwargs):
        captured.update(kwargs)
        return 0.002

    log = {"model": "m", "input_tokens_optimized": 100, "output_tokens": 0}
    assert request_cost_usd(_tracker(estimate), log) == 0.002
    assert captured["output_tokens"] == 0


def test_none_when_estimate_returns_none() -> None:
    # Unknown model / LiteLLM unavailable → estimate_cost returns None.
    log = {"model": "totally-unknown-model", "input_tokens_optimized": 600, "output_tokens": 50}
    assert request_cost_usd(_tracker(lambda **_: None), log) is None


def test_none_and_isolated_when_estimate_raises() -> None:
    def boom(**_):
        raise RuntimeError("pricing exploded")

    log = {"model": "m", "input_tokens_optimized": 600, "output_tokens": 50}
    assert request_cost_usd(_tracker(boom), log) is None


# ---------------------------------------------------------------------------
# Chain: emit_request_outcome -> RequestLog -> request_cost_usd
# ---------------------------------------------------------------------------


def test_logged_request_prices_the_input_the_cost_tracker_billed(monkeypatch) -> None:
    cost_tracker = CostTracker()
    billed_inputs: list[int] = []

    def estimate(**kwargs):
        billed_inputs.append(kwargs["input_tokens"])
        return kwargs["input_tokens"] * 1e-6

    monkeypatch.setattr(cost_tracker, "estimate_cost", estimate)
    handler = SimpleNamespace(
        metrics=PrometheusMetrics(cost_tracker=cost_tracker, stateless=True),
        cost_tracker=cost_tracker,
        logger=RequestLogger(),
    )
    outcome = RequestOutcome(
        request_id="hr_test_mismatch",
        provider="openai",
        model="gpt-4o-mini",
        original_tokens=1000,
        optimized_tokens=600,
        provider_input_tokens=800,
        output_tokens=50,
        tokens_saved=400,
        attempted_input_tokens=1000,
    )

    asyncio.run(emit_request_outcome(handler, outcome))

    (log,) = handler.logger.get_recent(1)
    assert log["input_tokens_optimized"] == 600
    assert log["provider_input_tokens"] == 800
    assert request_cost_usd(cost_tracker, log) == 800 * 1e-6
    # The cost tracker and the per-request field price the same input count.
    assert billed_inputs == [800, 800]


# ---------------------------------------------------------------------------
# Integration: /stats exposes cost_usd in recent_requests
# ---------------------------------------------------------------------------


def _sample_log(request_id: str = "hr_test_000001", turn_id: str | None = None) -> RequestLog:
    return RequestLog(
        request_id=request_id,
        timestamp="2026-06-29T00:00:00Z",
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        input_tokens_original=1000,
        input_tokens_optimized=600,
        output_tokens=50,
        tokens_saved=400,
        savings_percent=40.0,
        optimization_latency_ms=1.0,
        total_latency_ms=2.0,
        tags={},
        cache_hit=False,
        transforms_applied=[],
        turn_id=turn_id,
    )


def _stats_config() -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        image_optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=True,
        log_requests=True,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
    )


def test_stats_recent_requests_includes_cost_usd(monkeypatch) -> None:
    app = create_app(_stats_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        assert proxy.cost_tracker is not None
        # Deterministic price, independent of the live LiteLLM DB.
        monkeypatch.setattr(proxy.cost_tracker, "estimate_cost", lambda **_: 0.0135)
        # recent_requests is served only to loopback callers; TestClient's peer
        # host is "testclient", so force the loopback branch (not under test).
        monkeypatch.setattr("headroom.proxy.server._request_is_loopback", lambda request: True)
        proxy.logger._logs.append(_sample_log())

        resp = client.get("/stats")
        assert resp.status_code == 200
        recent = resp.json()["recent_requests"]
        assert recent, "expected at least one recent request"
        entry = next(e for e in recent if e["request_id"] == "hr_test_000001")
        assert entry["cost_usd"] == 0.0135


def test_stats_recent_requests_carry_turn_id_beyond_raw_log_tail(monkeypatch) -> None:
    # recent_requests serves up to 25 priced rows while request_logs keeps only
    # the newest 10, so every priced row must carry its own turn_id for a
    # consumer to group cost by turn.
    app = create_app(_stats_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        monkeypatch.setattr(proxy.cost_tracker, "estimate_cost", lambda **_: 0.001)
        monkeypatch.setattr("headroom.proxy.server._request_is_loopback", lambda request: True)
        for i in range(20):
            proxy.logger._logs.append(_sample_log(f"hr_test_{i:06d}", turn_id=f"turn_{i // 4}"))

        payload = client.get("/stats").json()
        assert len(payload["request_logs"]) == 10
        recent = [e for e in payload["recent_requests"] if e["request_id"].startswith("hr_test_")]
        assert len(recent) == 20

        cost_by_turn: dict[str, float] = {}
        for entry in recent:
            assert entry["cost_usd"] == 0.001
            cost_by_turn[entry["turn_id"]] = (
                cost_by_turn.get(entry["turn_id"], 0.0) + entry["cost_usd"]
            )
        assert sorted(cost_by_turn) == [f"turn_{t}" for t in range(5)]
        assert all(abs(total - 0.004) < 1e-12 for total in cost_by_turn.values())
