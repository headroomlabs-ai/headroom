"""Tool-schema deferral must not read as a near-total saving in /stats.

Deferral withholds the same schemas on every turn, so its token count grows
with the session while the tokens it removed would, after the first turn, have
been prefix-cache reads. Two things therefore have to hold:

* the token percentage divides by every input token the provider billed, cached
  prefix included — a gateway turn that reports only its uncached remainder as
  input made this read ~97% on real Claude Code traffic;
* the same saving is published in money as well, priced at cache rates, because
  the token share and the cost share are not the same number.
"""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app


def _make_client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "proxy_savings.json"))
    config = ProxyConfig(cache_enabled=False, rate_limit_enabled=False, log_requests=False)
    return TestClient(create_app(config))


def _record_warm_agent_turns(proxy, turns: int = 20) -> None:
    """One cold turn then warm ones, each withholding the same tool schemas."""
    for turn in range(turns):
        cache_read = 0 if turn == 0 else 40_000
        cache_write = 40_000 if turn == 0 else 500
        asyncio.run(
            proxy.metrics.record_request(
                provider="anthropic",
                model="claude-sonnet-4-5",
                # The whole billed prompt, as every handler now reports it.
                input_tokens=cache_read + cache_write + 20,
                output_tokens=150,
                tokens_saved=0,
                latency_ms=10.0,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                uncached_input_tokens=20,
                tool_search_saved=10_000,
            )
        )


def test_token_percentage_counts_the_cached_prefix_in_the_denominator(tmp_path, monkeypatch):
    with _make_client(tmp_path, monkeypatch) as client:
        proxy = client.app.state.proxy
        _record_warm_agent_turns(proxy)
        tokens = client.get("/stats").json()["tokens"]

    # 200k withheld against ~800k billed: a fifth, not the ~97% that came out
    # when the denominator held only the uncached remainder of each turn.
    assert tokens["saved"] == 200_000
    assert tokens["input"] > 700_000
    assert 15.0 < tokens["savings_percent"] < 30.0
    assert tokens["savings_percent"] == tokens["all_layers_savings_percent"]


def test_stats_publishes_the_cost_weighted_saving_next_to_the_token_saving(tmp_path, monkeypatch):
    with _make_client(tmp_path, monkeypatch) as client:
        proxy = client.app.state.proxy
        _record_warm_agent_turns(proxy)
        payload = client.get("/stats").json()

    tokens = payload["tokens"]
    assert "cost_weighted_savings_percent" in tokens
    # Removed tokens are mostly cache reads, so the money share is the smaller
    # of the two. Equal values would mean the cost side priced them at list.
    assert tokens["cost_weighted_savings_percent"] < tokens["savings_percent"]


def test_tool_schema_layer_carries_its_dollar_value(tmp_path, monkeypatch):
    with _make_client(tmp_path, monkeypatch) as client:
        proxy = client.app.state.proxy
        _record_warm_agent_turns(proxy)
        layer = client.get("/stats").json()["savings"]["by_layer"]["tool_search"]

    assert layer["tokens"] >= 0  # window-scoped; the dollars are lifetime
    assert "usd" in layer
    assert layer["usd"] >= 0.0
