"""Provider-side cache economics land in the per-request JSONL (#2438).

`cache_hit` alone can't distinguish a call billed cache-*creation* (write)
from a real cache-*read* hit, which is exactly the telemetry gap the issue
reported (the proxy stamps `cache_hit: true` while the client is billed
cache-write tokens). The raw provider deltas already live on RequestOutcome;
this pins that they survive into the RequestLog feed.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.models import RequestLog  # noqa: E402
from headroom.proxy.outcome import RequestOutcome, emit_request_outcome  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def test_request_log_carries_provider_cache_deltas(tmp_path):
    log_file = tmp_path / "proxy.jsonl"
    config = ProxyConfig(
        cache_enabled=False,
        rate_limit_enabled=False,
        log_requests=True,
        log_file=str(log_file),
    )

    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy

        # A call billed cache-*creation* (write), zero reads: cache_hit would
        # be False here, but the write/uncached deltas must still be recorded
        # so the true economics are visible.
        outcome = RequestOutcome(
            request_id="req-cache",
            provider="anthropic",
            model="claude-sonnet-5",
            original_tokens=1000,
            optimized_tokens=1000,
            output_tokens=20,
            tokens_saved=0,
            attempted_input_tokens=1000,
            cache_read_tokens=0,
            cache_write_tokens=800,
            uncached_input_tokens=200,
        )
        asyncio.run(emit_request_outcome(proxy, outcome))

    lines = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
    entry = next(e for e in lines if e["request_id"] == "req-cache")
    assert entry["cache_read_tokens"] == 0
    assert entry["cache_write_tokens"] == 800
    assert entry["uncached_input_tokens"] == 200


def test_request_log_cache_delta_fields_default_zero():
    # Backward-compatible: the new fields are optional and default to 0.
    entry = RequestLog(
        request_id="r",
        timestamp="t",
        provider="anthropic",
        model="m",
        input_tokens_original=0,
        input_tokens_optimized=0,
        output_tokens=0,
        tokens_saved=0,
        savings_percent=0.0,
        optimization_latency_ms=0.0,
        total_latency_ms=None,
        tags={},
        cache_hit=False,
        transforms_applied=[],
    )
    assert entry.cache_read_tokens == 0
    assert entry.cache_write_tokens == 0
    assert entry.uncached_input_tokens == 0
