"""The Cost Saved card must report what Headroom itself saved.

Before this, the card summed compression + tool deferral + the provider's
prefix-cache discount into one "Cost Saved" number, so a session that removed
1.4M tokens showed ~$25 saved — 85% of which was the provider's cache discount,
paid with or without Headroom. These tests pin the split, the cache-aware
valuation of removed tokens, and the inclusion of completion spend.
"""

from __future__ import annotations

from types import SimpleNamespace

from tests._dotenv import (
    autouse_apply_env,
    importorskip_no_env_leak,
    load_env_overrides,
)

_env_overrides = load_env_overrides()
apply_dotenv = autouse_apply_env(_env_overrides)

importorskip_no_env_leak("litellm")

MODEL = "claude-sonnet-4-20250514"


def _prices(model: str = MODEL) -> tuple[float, float, float]:
    import litellm

    from headroom.pricing.litellm_pricing import resolve_litellm_model

    info = litellm.model_cost.get(resolve_litellm_model(model), {})
    uncached = info["input_cost_per_token"]
    return (
        info.get("cache_read_input_token_cost", uncached),
        info.get("cache_creation_input_token_cost", uncached),
        uncached,
    )


def test_compressed_tokens_priced_at_the_rate_they_would_have_been_billed():
    """A cold turn's removed tokens would have been billed as cache writes."""
    from headroom.proxy.server import CostTracker

    ct = CostTracker()
    ct.record_tokens(
        MODEL,
        tokens_saved=100_000,
        tokens_sent=50_000,
        cache_read_tokens=0,
        cache_write_tokens=90_000,
        uncached_tokens=10_000,
    )
    stats = ct.stats()

    _cache_read, cache_write, uncached = _prices()
    write_share = 90_000 / 100_000
    expected = 100_000 * (write_share * cache_write + (1 - write_share) * uncached)

    assert abs(stats["cache_aware_savings_usd"] - expected) < 1e-6
    # Writes cost MORE than list, so flat list pricing understates a cold turn.
    assert stats["cache_aware_savings_usd"] > stats["savings_usd"]
    assert abs(stats["savings_usd"] - 100_000 * uncached) < 1e-6


def test_warm_prefix_does_not_drag_the_live_delta_down_to_the_read_rate():
    """Compression works the live zone; the frozen prefix is not its rate.

    Handlers keep the cached prefix byte-identical for prefix-cache safety and
    compress only the appended delta, so removed tokens could never have been
    billed as cache reads. Splitting them across the WHOLE request's mix valued
    a warm turn at ~a tenth of what the provider would have charged.
    """
    from headroom.proxy.server import CostTracker

    ct = CostTracker()
    ct.record_tokens(
        MODEL,
        tokens_saved=10_000,
        tokens_sent=200_000,
        cache_read_tokens=190_000,
        cache_write_tokens=5_000,
        uncached_tokens=5_000,
    )
    stats = ct.stats()

    cache_read, cache_write, uncached = _prices()
    expected = 10_000 * (0.5 * cache_write + 0.5 * uncached)
    # stats() rounds dollars to 4dp.
    assert abs(stats["cache_aware_savings_usd"] - expected) < 1e-4

    # The whole-request mix is 95% cache reads; pricing the delta that way would
    # have valued it near the read rate.
    whole_request_mix = 10_000 * (0.95 * cache_read + 0.025 * cache_write + 0.025 * uncached)
    assert stats["cache_aware_savings_usd"] > 5 * whole_request_mix


def test_fully_uncached_request_values_removed_tokens_at_list():
    """With no cache in play there is nothing to discount."""
    from headroom.proxy.server import CostTracker

    ct = CostTracker()
    ct.record_tokens(
        MODEL,
        tokens_saved=40_000,
        tokens_sent=20_000,
        uncached_tokens=20_000,
    )
    stats = ct.stats()

    assert abs(stats["cache_aware_savings_usd"] - stats["savings_usd"]) < 1e-6


def test_completion_spend_is_reported_alongside_input_spend():
    """`total_cost_usd` is the bill; `cost_with_headroom_usd` stays input-only."""
    import litellm

    from headroom.pricing.litellm_pricing import resolve_litellm_model
    from headroom.proxy.server import CostTracker

    ct = CostTracker()
    ct.record_tokens(
        MODEL,
        tokens_saved=0,
        tokens_sent=100_000,
        uncached_tokens=100_000,
        output_tokens=20_000,
    )
    stats = ct.stats()

    info = litellm.model_cost.get(resolve_litellm_model(MODEL), {})
    expected_output = 20_000 * info["output_cost_per_token"]

    assert abs(stats["output_cost_usd"] - expected_output) < 1e-6
    assert stats["cost_with_headroom_usd"] > 0
    expected_total = stats["cost_with_headroom_usd"] + stats["output_cost_usd"]
    assert abs(stats["total_cost_usd"] - expected_total) < 1e-6


def test_long_context_turn_is_priced_at_the_above_200k_rates():
    """Past 200k the catalog charges a second, higher tier for input and output."""
    import litellm

    from headroom.pricing.litellm_pricing import resolve_litellm_model
    from headroom.proxy.server import CostTracker

    info = litellm.model_cost.get(resolve_litellm_model(MODEL), {})
    long_input = info["input_cost_per_token_above_200k_tokens"]
    long_output = info["output_cost_per_token_above_200k_tokens"]
    assert long_input > info["input_cost_per_token"]
    assert long_output > info["output_cost_per_token"]

    ct = CostTracker()
    ct.record_tokens(
        MODEL,
        tokens_saved=10_000,
        tokens_sent=300_000,
        cache_read_tokens=0,
        cache_write_tokens=0,
        uncached_tokens=300_000,
        output_tokens=5_000,
    )
    stats = ct.stats()

    assert abs(stats["output_cost_usd"] - 5_000 * long_output) < 1e-6
    assert abs(stats["cache_aware_savings_usd"] - 10_000 * long_input) < 1e-6


def _summary(cache_net_usd: float, cost_stats: dict) -> dict:
    from headroom.proxy.cost import build_session_summary

    proxy = SimpleNamespace(
        config=SimpleNamespace(mode="token"),
        logger=SimpleNamespace(_logs=[]),
        cost_tracker=SimpleNamespace(stats=lambda: cost_stats),
    )
    metrics = SimpleNamespace(requests_by_model={}, tokens_saved_total=0)
    prefix_cache_stats = {"totals": {"net_savings_usd": cache_net_usd}}
    return build_session_summary(proxy, metrics, prefix_cache_stats, total_tokens_before=0)


def test_provider_cache_discount_is_reported_beside_the_headline_not_inside_it():
    payload = _summary(
        23.62,
        {
            "cost_with_headroom_usd": 7.88,
            "output_cost_usd": 3.60,
            "total_cost_usd": 11.48,
            "savings_usd": 4.20,
            "cache_aware_savings_usd": 1.05,
            "tool_savings_usd": 0.25,
        },
    )
    cost = payload["cost"]

    assert cost["total_saved_usd"] == 1.30  # compression (cache-aware) + tool deferral
    assert cost["provider_cache_discount_usd"] == 23.62
    assert cost["breakdown"]["compression_savings_usd"] == 1.05
    assert cost["breakdown"]["compression_savings_list_usd"] == 4.2
    # Spend is the whole bill, and the baseline is spend + what Headroom saved.
    assert cost["with_headroom_usd"] == 11.48
    assert cost["with_headroom_input_usd"] == 7.88
    assert cost["with_headroom_output_usd"] == 3.6
    assert cost["without_headroom_usd"] == 12.78


def test_summary_falls_back_to_list_pricing_when_cache_aware_is_absent():
    """An older tracker payload must still produce a coherent card."""
    payload = _summary(1.0, {"cost_with_headroom_usd": 2.0, "savings_usd": 0.5})
    cost = payload["cost"]

    assert cost["total_saved_usd"] == 0.5
    assert cost["with_headroom_usd"] == 2.0


def test_prefix_cache_savings_use_the_model_catalog_rates():
    """Cache economics come from LiteLLM per model, not hardcoded ratios."""
    from headroom.proxy.cost import build_prefix_cache_stats
    from headroom.proxy.prometheus_metrics import PrometheusMetrics
    from headroom.proxy.server import CostTracker

    ct = CostTracker()
    ct.record_tokens(MODEL, tokens_saved=0, tokens_sent=1_000_000, uncached_tokens=1_000_000)

    metrics = PrometheusMetrics()
    metrics.cache_by_provider["anthropic"] = {
        "requests": 10,
        "hit_requests": 8,
        "cache_read_tokens": 1_000_000,
        "cache_write_tokens": 0,
        "cache_write_5m_tokens": 0,
        "cache_write_1h_tokens": 0,
        "cache_write_5m_requests": 0,
        "cache_write_1h_requests": 0,
        "uncached_input_tokens": 100_000,
        "bust_count": 0,
        "bust_write_tokens": 0,
    }

    stats = build_prefix_cache_stats(metrics, ct)
    provider = stats["by_provider"]["anthropic"]

    cache_read, _cache_write, uncached = _prices()
    expected = 1_000_000 * (uncached - cache_read)

    assert provider["cache_pricing_source"] == "catalog"
    assert abs(provider["savings_usd"] - expected) < 1e-6


def test_dashboard_card_shows_the_cache_discount_separately():
    from headroom.dashboard import get_dashboard_html

    html = get_dashboard_html()

    assert "provider cache discount" in html
    assert "cost?.provider_cache_discount_usd" in html
