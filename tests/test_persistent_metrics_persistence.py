"""Tests schema v5 persistence around the pure Lifetime aggregate."""

from __future__ import annotations

import json

from headroom.proxy.savings_tracker import SavingsTracker


def test_savings_tracker_migrates_v4_lifetime_to_v5_metrics_and_preserves_legacy_state(tmp_path):
    path = tmp_path / "proxy_savings.json"
    legacy_state = {
        "schema_version": 4,
        "lifetime": {
            "requests": 7,
            "tokens_saved": 20,
            "compression_savings_usd": 0.5,
            "cache_read_tokens": 5,
            "cache_savings_usd": 0.2,
            "total_input_tokens": 80,
            "total_input_cost_usd": 1.5,
        },
        "display_session": {
            "requests": 2,
            "tokens_saved": 4,
            "compression_savings_usd": 0.1,
            "cache_read_tokens": 1,
            "cache_savings_usd": 0.01,
            "total_input_tokens": 10,
            "total_input_cost_usd": 0.2,
            "started_at": "2026-07-01T00:00:00Z",
            "last_activity_at": "2026-07-02T00:00:00Z",
        },
        "history": [
            {
                "timestamp": "2026-07-02T00:00:00Z",
                "total_tokens_saved": 20,
                "compression_savings_usd": 0.5,
                "total_input_tokens": 80,
                "total_input_cost_usd": 1.5,
            }
        ],
        "projects": {"keep-me": {"requests": 1}},
    }
    path.write_text(json.dumps(legacy_state), encoding="utf-8")

    tracker = SavingsTracker(path=str(path), save_flush_every=25)
    lifetime = tracker.lifetime_response()

    assert lifetime["schema_version"] == 5
    assert lifetime["requests"]["total"] == 7
    assert lifetime["tokens"]["input"] == 80
    assert lifetime["tokens"]["attempted_input"] == 100
    assert lifetime["tokens"]["saved"] == 20
    assert lifetime["prefix_cache"]["cache_read_tokens"] == 5
    assert lifetime["cost"] == {
        "input_usd": 1.5,
        "compression_savings_usd": 0.5,
        "cache_savings_usd": 0.2,
    }
    assert lifetime["by_model"]["other"]["input_tokens"] == 80

    tracker.flush()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 5
    assert saved["lifetime"] == legacy_state["lifetime"]
    assert saved["display_session"]["requests"] == 2
    assert saved["projects"]["keep-me"]["requests"] == 1
    assert saved["lifetime_metrics"]["models"]["other"]["input_tokens"] == 80
    assert isinstance(saved["lifetime_metrics"]["persistence"]["last_saved_at"], str)


def test_lifetime_response_reports_stateless_mode_without_writing(tmp_path):
    path = tmp_path / "proxy_savings.json"
    tracker = SavingsTracker(path=str(path), stateless=True, save_flush_every=1)

    tracker.record_lifetime_request(
        provider="openai", stack="codex", model="gpt-test", input_tokens=3
    )

    response = tracker.lifetime_response()
    assert response["persistence"] == {
        "enabled": False,
        "healthy": True,
        "error": "Lifetime metrics unavailable in stateless mode",
        "pending_records": 0,
        "last_saved_at": None,
    }
    assert path.exists() is False
