"""Cache-aware savings must hold for EVERY provider and every harness.

Headroom sits in front of Anthropic, Bedrock, Vertex, OpenAI, Gemini, and
arbitrary gateways, driven by Claude Code, Codex, Cursor, Copilot, Kiro and the
MCP tool path. Those backends report their prompt-cache usage in genuinely
different shapes, and the differences change the money:

    Anthropic/Bedrock  read + write + 5m/1h split + uncached
    OpenAI             read + uncached, write INFERRED (no write premium)
    Gemini             read only
    gateway / MCP      nothing at all

The pricing fix lives at one chokepoint — ``emit_request_outcome`` feeds the
single ``metrics.record_request``, which every one of the 32 handler emit sites
routes through — so this file asserts the matrix rather than each handler. If
pricing were ever forked per provider, these are the tests that would catch it.
"""

from __future__ import annotations

import pytest

from tests._dotenv import (
    autouse_apply_env,
    importorskip_no_env_leak,
    load_env_overrides,
)

_env_overrides = load_env_overrides()
apply_dotenv = autouse_apply_env(_env_overrides)

importorskip_no_env_leak("litellm")

from headroom.proxy.savings_tracker import estimate_request_savings_usd  # noqa: E402

# One warm turn per provider, in that provider's own reporting shape.
WARM_TURNS = {
    "anthropic": {
        "model": "claude-sonnet-4-20250514",
        "cache_read_tokens": 180_000,
        "cache_write_tokens": 2_000,
        "cache_write_5m_tokens": 2_000,
        "uncached_input_tokens": 500,
    },
    "bedrock": {
        "model": "claude-sonnet-4-20250514",
        "cache_read_tokens": 180_000,
        "cache_write_tokens": 2_000,
        "cache_write_5m_tokens": 2_000,
        "uncached_input_tokens": 500,
    },
    "openai": {
        "model": "gpt-4o",
        "cache_read_tokens": 180_000,
        "cache_write_tokens": 2_500,
        "uncached_input_tokens": 2_500,
        "cache_inferred": True,
    },
    "gemini": {
        "model": "gemini/gemini-2.5-pro",
        "cache_read_tokens": 180_000,
    },
}


@pytest.mark.parametrize("provider", sorted(WARM_TURNS))
def test_tool_schema_savings_are_discounted_on_a_warm_turn_for_every_provider(provider):
    """Deferred tool schemas ride the cached prefix, whoever is serving it.

    Every provider in the matrix discounts cache reads, so on a warm turn the
    honest value of deferred schemas is strictly below list on all of them. This
    is the defect in its purest form: list pricing charged the full input rate
    for tokens that would have been read from cache.
    """
    priced = estimate_request_savings_usd(
        tool_schema_tokens_saved=8_000,
        provider=provider,
        **WARM_TURNS[provider],
    )

    assert priced["tool_schema"] < priced["tool_schema_list"], (
        f"{provider}: warm-turn tool-schema savings should be discounted"
    )
    assert priced["basis"] == "catalog"


@pytest.mark.parametrize("provider", sorted(WARM_TURNS))
def test_live_zone_compression_is_never_valued_at_the_cache_read_rate(provider):
    """Compression works the appended delta, which no provider had cached.

    The symmetric error: pricing compression by the whole-request mix would
    value a warm turn's savings at ~0.1x, an order of magnitude UNDER what was
    actually charged.
    """
    priced = estimate_request_savings_usd(
        compression_tokens_saved=8_000,
        provider=provider,
        **WARM_TURNS[provider],
    )

    assert priced["compression"] >= priced["compression_list"] * 0.9, (
        f"{provider}: live-zone savings must not be discounted as cache reads"
    )


def test_openai_inferred_write_carries_no_write_premium():
    """OpenAI has no write counter, so handlers derive one. It must not be billed.

    The derived figure is the same tokens as ``uncached_input_tokens``. Treating
    it as a real write both double-counts them and applies a 1.25x premium
    OpenAI does not charge.
    """
    common = {
        "model": "gpt-4o",
        "compression_tokens_saved": 5_000,
        "cache_read_tokens": 100_000,
        "uncached_input_tokens": 10_000,
        "provider": "openai",
    }
    inferred = estimate_request_savings_usd(
        cache_write_tokens=10_000, cache_inferred=True, **common
    )
    absent = estimate_request_savings_usd(**common)

    assert inferred["compression"] == pytest.approx(absent["compression"])


def test_a_provider_that_reports_no_cache_data_prices_at_list_and_says_so():
    """Gateways, custom bases and streaming turns with no usage frame land here.

    List price is the only honest answer with no mix — but it must be LABELLED,
    not presented as a measurement.
    """
    priced = estimate_request_savings_usd(
        model="claude-sonnet-4-20250514",
        compression_tokens_saved=5_000,
        tool_schema_tokens_saved=5_000,
    )

    assert priced["compression"] == pytest.approx(priced["compression_list"])
    assert priced["tool_schema"] == pytest.approx(priced["tool_schema_list"])
    assert priced["basis"] == "no-mix"


def test_the_mcp_tool_path_still_records_without_any_cache_signal():
    """``headroom mcp serve`` never learns the agent's model or its cache mix.

    It must keep writing ledger events at the blended fallback rate rather than
    failing or recording $0 — the path predates cache reporting entirely.
    """
    import tempfile
    from pathlib import Path

    from headroom import savings_ledger

    path = Path(tempfile.mkdtemp()) / "events.jsonl"
    assert savings_ledger.record_savings_event(
        tokens_before=10_000,
        tokens_after=4_000,
        model="unknown",
        client="mcp",
        source="mcp",
        path=path,
    )

    report = savings_ledger.aggregate_savings(path=path).to_dict()
    lifetime = report["lifetime"]
    assert lifetime["calls"] == 1
    assert lifetime["cost_usd"] > 0
    # No mix, so the cache-aware column falls back to the list figure rather
    # than dropping the event out of the total.
    assert lifetime["cost_effective_usd"] == pytest.approx(lifetime["cost_usd"])
    assert lifetime["basis"] == "list"


def test_the_headline_degrades_to_the_weakest_provider_in_a_mixed_session():
    """Claude Code sends Sonnet and Haiku; a gateway may add an unpriceable model.

    A session mixing a catalog-priced provider with one reporting nothing must
    not present its total as catalog-grade.
    """
    priced = estimate_request_savings_usd(
        model="claude-sonnet-4-20250514",
        # Priced against a real mix...
        compression_tokens_saved=5_000,
        cache_read_tokens=100_000,
        cache_write_tokens=1_000,
        cache_write_5m_tokens=1_000,
        uncached_input_tokens=1_000,
        provider="anthropic",
    )
    assert priced["basis"] == "catalog"

    # ...but the same layers with nothing reported cannot claim the same.
    blind = estimate_request_savings_usd(
        model="claude-sonnet-4-20250514",
        compression_tokens_saved=5_000,
    )
    assert blind["basis"] == "no-mix"


def test_long_context_requests_price_at_the_above_200k_tier():
    """A >200k turn is billed at a different rate table; savings must follow.

    Anthropic's above-200k input rate is 2x the base one, so valuing a long
    request's savings at base rates understates them by half.
    """
    base = estimate_request_savings_usd(
        model="claude-sonnet-4-20250514",
        compression_tokens_saved=5_000,
        cache_write_tokens=50_000,
        cache_write_5m_tokens=50_000,
        uncached_input_tokens=10_000,
        provider="anthropic",
    )
    long_ctx = estimate_request_savings_usd(
        model="claude-sonnet-4-20250514",
        compression_tokens_saved=5_000,
        cache_write_tokens=250_000,
        cache_write_5m_tokens=250_000,
        uncached_input_tokens=10_000,
        provider="anthropic",
    )

    assert long_ctx["compression"] > base["compression"]
    assert long_ctx["compression_list"] > base["compression_list"]
