"""New-content-relative savings rate in /stats (tokens.new_input_savings_percent).

The whole-request ratios recount the full transcript on every turn, so long
cached sessions dilute toward 0% regardless of how well compression performs
on content that newly enters context. The new rate divides by provider-billed
non-cache-read input (uncached + cache-write) plus the tokens compression
removed before they could be billed.
"""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app


def _make_client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "proxy_savings.json"))
    config = ProxyConfig(
        cache_enabled=False,
        rate_limit_enabled=False,
        log_requests=False,
    )
    return TestClient(create_app(config))


def test_stats_reports_new_input_savings_rate(tmp_path, monkeypatch):
    with _make_client(tmp_path, monkeypatch) as client:
        proxy = client.app.state.proxy
        # A late turn of a long cached session: the local transcript recount
        # (input_tokens) dwarfs what the provider newly billed (uncached +
        # cache_write = 50k), so the whole-request ratio dilutes to ~0.5%
        # while the new-content rate reports the undiluted 9.09%.
        asyncio.run(
            proxy.metrics.record_request(
                provider="anthropic",
                model="claude-opus-4-6",
                input_tokens=1_000_000,
                output_tokens=200,
                tokens_saved=5_000,
                latency_ms=10.0,
                cache_read_tokens=900_000,
                cache_write_tokens=45_000,
                uncached_input_tokens=5_000,
            )
        )

        stats = client.get("/stats")
        assert stats.status_code == 200
        tokens = stats.json()["tokens"]

    assert tokens["new_input_tokens"] == 50_000
    # 5_000 saved / (50_000 billed-new + 5_000 saved) = 9.09%
    assert tokens["new_input_savings_percent"] == 9.09
    # The transcript-diluted ratio stays as-is — the new rate sits alongside,
    # it does not replace existing fields.
    assert tokens["proxy_savings_percent"] == 0.5


def test_stats_new_input_rate_is_zero_without_cache_usage_data(tmp_path, monkeypatch):
    with _make_client(tmp_path, monkeypatch) as client:
        proxy = client.app.state.proxy
        # Savings recorded but no cache usage observed (provider without
        # cache metrics): the rate must report 0, not savings/savings=100%.
        asyncio.run(
            proxy.metrics.record_request(
                provider="bedrock",
                model="claude-opus-4-6",
                input_tokens=10_000,
                output_tokens=200,
                tokens_saved=2_000,
                latency_ms=10.0,
            )
        )

        tokens = client.get("/stats").json()["tokens"]

    assert tokens["new_input_tokens"] == 0
    assert tokens["new_input_savings_percent"] == 0
