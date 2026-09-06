"""Tests for per-request ``cost_usd`` in ``/stats`` recent_requests (issue #1079).

The proxy exposes a per-request billed-cost estimate so tooling (e.g. claude-hud)
can attribute cost to a Claude Code session by summing over turn_id / time window,
without an external ``ccusage`` subprocess.
"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from headroom.proxy.cost import request_cost_usd
from headroom.proxy.models import RequestLog
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


def test_handles_null_output_tokens() -> None:
    captured: dict = {}

    def estimate(**kwargs):
        captured.update(kwargs)
        return 0.002

    log = {"model": "m", "input_tokens_optimized": 100, "output_tokens": None}
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
# Integration: /stats exposes cost_usd in recent_requests
# ---------------------------------------------------------------------------


def _sample_log() -> RequestLog:
    return RequestLog(
        request_id="hr_test_000001",
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
    )


def test_stats_recent_requests_includes_cost_usd(monkeypatch) -> None:
    config = ProxyConfig(
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
    app = create_app(config)
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
