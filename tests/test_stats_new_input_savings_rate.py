"""New-content-relative savings rate in /stats (tokens.new_input_savings_percent).

The whole-request ratios recount the full transcript on every turn, so long
cached sessions dilute toward 0% regardless of how well compression performs
on content that newly enters context. The new rate divides by provider-billed
non-cache-read input (uncached + cache-write) plus the tokens compression
removed before they could be billed.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from headroom import savings_ledger
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


def test_stats_new_input_rate_pairs_savings_with_qualified_requests(tmp_path, monkeypatch):
    """The numerator must come from the same requests as the denominator. A
    request with no cache breakdown (Bedrock, an MCP tool) never enters
    new_input_tokens, so its savings must not lend themselves to that ratio:
    one qualified 50 percent request plus one unqualified 10,000-token saving
    used to read as 99 percent."""
    with _make_client(tmp_path, monkeypatch) as client:
        proxy = client.app.state.proxy
        asyncio.run(
            proxy.metrics.record_request(
                provider="anthropic",
                model="claude-opus-4-6",
                input_tokens=1_100,
                output_tokens=10,
                tokens_saved=100,
                latency_ms=10.0,
                cache_read_tokens=1_000,
                uncached_input_tokens=100,
            )
        )
        assert client.get("/stats").json()["tokens"]["new_input_savings_percent"] == 50.0
        asyncio.run(
            proxy.metrics.record_request(
                provider="bedrock",
                model="claude-opus-4-6",
                input_tokens=20_000,
                output_tokens=10,
                tokens_saved=10_000,
                latency_ms=10.0,
            )
        )
        tokens = client.get("/stats").json()["tokens"]
    assert tokens["new_input_tokens"] == 100
    assert tokens["new_input_savings_percent"] == 50.0
    # The whole-wire figures still count the unqualified request.
    assert tokens["saved"] >= 10_100


def test_stats_and_ledger_share_one_new_input_cohort(tmp_path, monkeypatch):
    """The dashboard rate and `headroom savings` must be the same measurement.

    /stats accumulated its new-input pair under a cache-ACTIVITY gate while the
    ledger writes under a newly-BILLED-input one, so the two admitted different
    requests. An uncached-only turn (real new input, no cache read or write)
    reached the ledger and nothing else, and a cache-read-only turn lent its
    savings to the dashboard numerator with no denominator to match. The
    reviewer's case: 100 saved / 100 new / 1,000 cache-read followed by
    0 saved / 10,000 new left the dashboard at 50% and the ledger near 1%.
    """
    ledger_path = tmp_path / "savings_events.jsonl"
    monkeypatch.setattr(savings_ledger, "_resolve_path", lambda path=None: ledger_path)

    with _make_client(tmp_path, monkeypatch) as client:
        proxy = client.app.state.proxy

        async def record(**kwargs) -> None:
            await proxy.metrics.record_request(
                provider="anthropic",
                model="claude-opus-4-6",
                output_tokens=10,
                latency_ms=10.0,
                **kwargs,
            )

        # Cache-read plus a little new input: in both cohorts.
        asyncio.run(
            record(
                input_tokens=1_100,
                tokens_saved=100,
                cache_read_tokens=1_000,
                uncached_input_tokens=100,
            )
        )
        # Uncached-only: real new input, no cache activity at all. Used to be
        # a ledger-only observation.
        asyncio.run(record(input_tokens=10_000, tokens_saved=0, uncached_input_tokens=10_000))
        # Cache-read-only: no new input, so it belongs to neither side despite
        # saving tokens. Used to inflate the dashboard numerator alone.
        asyncio.run(record(input_tokens=2_000, tokens_saved=500, cache_read_tokens=2_000))

        tokens = client.get("/stats").json()["tokens"]

    lifetime = savings_ledger.aggregate_savings(path=ledger_path).lifetime

    assert tokens["new_input_tokens"] == 10_100
    assert lifetime["new_input_tokens"] == 10_100
    # 100 saved / (10,100 new + 100 saved) — not the old 50%.
    assert tokens["new_input_savings_percent"] == pytest.approx(0.98, abs=0.05)
    assert lifetime["new_input_savings_percent"] == pytest.approx(
        tokens["new_input_savings_percent"], abs=0.05
    )
    # The cache-read-only turn's savings still count on the whole-wire figure.
    assert tokens["saved"] == 600


def test_inferred_openai_write_counts_as_new_input_once(tmp_path, monkeypatch):
    """OpenAI paths with no reported write pass ``cache_inferred=True`` with the
    write equal to the uncached slice (input - cached): one set of tokens under
    two names. Adding both doubled new input, so a true 50% rate read 33%, and
    the provider row showed (and priced a premium on) a write nobody billed."""
    ledger_path = tmp_path / "savings_events.jsonl"
    monkeypatch.setattr(savings_ledger, "_resolve_path", lambda path=None: ledger_path)

    with _make_client(tmp_path, monkeypatch) as client:
        proxy = client.app.state.proxy
        asyncio.run(
            proxy.metrics.record_request(
                provider="openai",
                model="gpt-5.4",
                input_tokens=2_000,
                output_tokens=10,
                tokens_saved=1_000,
                latency_ms=10.0,
                cache_read_tokens=1_000,
                cache_write_tokens=1_000,
                uncached_input_tokens=1_000,
                cache_inferred=True,
            )
        )
        stats = client.get("/stats").json()

    lifetime = savings_ledger.aggregate_savings(path=ledger_path).lifetime
    tokens = stats["tokens"]
    totals = stats["prefix_cache"]["totals"]
    openai_row = stats["prefix_cache"]["by_provider"]["openai"]

    assert tokens["new_input_tokens"] == 1_000
    assert lifetime["new_input_tokens"] == 1_000
    # 1,000 saved / (1,000 new + 1,000 saved), not 1,000 / 3,000.
    assert tokens["new_input_savings_percent"] == 50.0
    assert lifetime["new_input_savings_percent"] == 50.0
    assert openai_row["cache_write_tokens"] == 0
    assert openai_row["hit_rate"] == 50.0
    new_input = totals["cache_write_tokens"] + totals["uncached_input_tokens"]
    assert new_input == totals["new_input_tokens"]


def test_inferred_openai_write_carries_no_write_premium(tmp_path, monkeypatch):
    """The OpenAI row prices its write premium from the cost tracker's model.
    Where the catalog gives that model a write rate above input (gpt-5.6
    publishes 1.25x), an inferred write counted as a write was charged the
    premium on tokens OpenAI billed as plain input."""
    import litellm

    from headroom.pricing.counterfactual import resolve_rates
    from headroom.proxy.outcome import RequestOutcome

    ledger_path = tmp_path / "savings_events.jsonl"
    monkeypatch.setattr(savings_ledger, "_resolve_path", lambda path=None: ledger_path)
    # A real catalog id with known rates, so the live catalog cannot move them.
    model = "gpt-4o"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {
            "input_cost_per_token": 4e-6,
            "output_cost_per_token": 3.2e-5,
            "cache_read_input_token_cost": 4e-7,
            "cache_creation_input_token_cost": 5e-6,
            "litellm_provider": "openai",
            "mode": "chat",
        },
    )
    resolve_rates.cache_clear()
    try:
        with _make_client(tmp_path, monkeypatch) as client:
            proxy = client.app.state.proxy
            # The Chat Completions handler's outcome: no reported write, so the
            # write is inferred as input - cached. Feeds metrics AND the cost
            # tracker, which supplies the price build_prefix_cache_stats uses.
            asyncio.run(
                proxy._record_request_outcome(
                    RequestOutcome(
                        request_id="req_inferred_write",
                        provider="openai",
                        model=model,
                        original_tokens=2_000,
                        optimized_tokens=2_000,
                        provider_input_tokens=2_000,
                        output_tokens=10,
                        tokens_saved=0,
                        attempted_input_tokens=2_000,
                        cache_read_tokens=1_000,
                        cache_write_tokens=1_000,
                        uncached_input_tokens=1_000,
                        cache_inferred=True,
                    )
                )
            )
            openai_row = client.get("/stats").json()["prefix_cache"]["by_provider"]["openai"]
    finally:
        resolve_rates.cache_clear()

    # The catalog write rate is in force, so a write would be charged 25%.
    assert openai_row["cache_pricing_source"] == "catalog"
    assert openai_row["write_premium"] == "25%"
    # 1,000 reads x $4/M x 90% read discount.
    assert openai_row["savings_usd"] == 0.0036
    # Counted as a write: 1,000 x $4/M x 25% = $0.001 premium, net $0.0026.
    assert openai_row["write_premium_usd"] == 0.0
    assert openai_row["net_savings_usd"] == 0.0036
    assert openai_row["cache_write_tokens"] == 0
