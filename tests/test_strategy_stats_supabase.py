"""Phase 3e.0: surface per-strategy compression counters.

The internal `compressions_by_strategy` and `tokens_saved_by_strategy`
counters (PR #302) were tracked in process but never exported, because
the Prometheus→Supabase pipeline treats each metric name as a column and
adding columns is operationally expensive.

These tests pin the alternative path:

1. The `/stats` endpoint exposes both dicts on the response root.
2. The telemetry beacon nests them under `pipeline_timing._strategies`,
   landing as JSONB inside the existing `pipeline_timing` Supabase
   column — zero schema change.
"""

from __future__ import annotations

import pytest


def test_stats_endpoint_exposes_per_strategy_counters() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
        )
    )

    proxy = app.state.proxy
    proxy.metrics.record_compression("smart_crusher", original_tokens=300, compressed_tokens=80)
    proxy.metrics.record_compression("smart_crusher", original_tokens=100, compressed_tokens=40)
    proxy.metrics.record_compression("diff", original_tokens=120, compressed_tokens=70)

    with TestClient(app) as client:
        response = client.get("/stats")

    assert response.status_code == 200
    body = response.json()
    assert body["compressions_by_strategy"] == {"smart_crusher": 2, "diff": 1}
    assert body["tokens_saved_by_strategy"] == {
        "smart_crusher": (300 - 80) + (100 - 40),
        "diff": 120 - 70,
    }


def test_beacon_nests_strategies_under_pipeline_timing() -> None:
    from headroom.telemetry.beacon import _build_pipeline_timing

    stats = {
        "pipeline_timing": {
            "smart_crusher": {"average_ms": 12.345, "max_ms": 50.0, "count": 10},
            "diff_compressor": {"average_ms": 5.1, "max_ms": 9.0, "count": 4},
        },
        "compressions_by_strategy": {"smart_crusher": 4321, "diff": 87},
        "tokens_saved_by_strategy": {"smart_crusher": 1_234_567, "diff": 23_456},
    }

    timing = _build_pipeline_timing(stats)

    assert timing["smart_crusher"] == 12.35
    assert timing["diff_compressor"] == 5.1
    assert timing["_strategies"] == {
        "compressions": {"smart_crusher": 4321, "diff": 87},
        "tokens_saved": {"smart_crusher": 1_234_567, "diff": 23_456},
    }


def test_beacon_omits_strategies_subkey_when_counters_empty() -> None:
    from headroom.telemetry.beacon import _build_pipeline_timing

    stats = {
        "pipeline_timing": {"router": {"average_ms": 1.2}},
        "compressions_by_strategy": {},
        "tokens_saved_by_strategy": {},
    }

    timing = _build_pipeline_timing(stats)

    assert timing == {"router": 1.2}
    assert "_strategies" not in timing
