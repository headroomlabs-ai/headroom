"""`/stats` payload surfaces the CCR retrieval drawback (net savings) additively.

Asserts the new compression/cost/feedback keys exist, that net == gross when no
retrieval has happened, and that an MCP-side retrieval debits net through the
proxy's aggregation — without renaming any existing key.
"""

from __future__ import annotations

import asyncio
import json
import time

from fastapi.testclient import TestClient

import headroom.ccr.mcp_server as mcp
from headroom.proxy.server import ProxyConfig, create_app


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "proxy_savings.json"))
    return TestClient(
        create_app(ProxyConfig(cache_enabled=False, rate_limit_enabled=False, log_requests=False))
    )


def _record_compression(proxy) -> None:
    asyncio.run(
        proxy.metrics.record_request(
            provider="anthropic",
            model="claude-opus-4-6",
            input_tokens=10_000,
            output_tokens=200,
            tokens_saved=5_000,
            latency_ms=1.0,
        )
    )


def test_stats_payload_has_ccr_net_keys_without_retrieval(tmp_path, monkeypatch):
    # Point the MCP shared-events log at an empty file so no retrieval is seen.
    monkeypatch.setattr(mcp, "SHARED_STATS_FILE", tmp_path / "empty.jsonl", raising=False)
    with _client(tmp_path, monkeypatch) as client:
        _record_compression(client.app.state.proxy)
        payload = client.get("/stats").json()

    comp = payload["compression"]
    assert "ccr_retrievals" in comp  # existing key untouched
    assert comp["tokens_retrieved"] == 0
    assert comp["retrievals_total"] == 0
    assert comp["net_tokens_saved"] == 5_000  # no retrieval -> net == gross
    assert comp["retrieval_cost_usd"] == 0.0

    summary = payload["summary"]
    assert summary["compression"]["net_tokens_saved"] == 5_000
    assert summary["cost"]["retrieval_cost_usd"] == 0.0
    assert "ccr_overhead" in summary["cost"]

    assert isinstance(payload["feedback_loop"]["high_retrieval_tools"], list)


def test_stats_payload_debits_net_from_mcp_retrieval(tmp_path, monkeypatch):
    shared = tmp_path / "session_stats.jsonl"
    shared.write_text(
        json.dumps(
            {
                "type": "retrieve",
                "hash": "abc123",
                "tokens": 3_000,
                "timestamp": time.time(),
                "pid": 999,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mcp, "SHARED_STATS_FILE", shared, raising=False)

    with _client(tmp_path, monkeypatch) as client:
        _record_compression(client.app.state.proxy)
        payload = client.get("/stats").json()

    comp = payload["compression"]
    assert comp["tokens_retrieved"] == 3_000
    assert comp["retrievals_total"] == 1
    # net = 5_000 gross - 3_000 retrieved - 1 * 50 overhead
    assert comp["net_tokens_saved"] == 1_950
    assert payload["summary"]["cost"]["retrieval_cost_usd"] > 0
