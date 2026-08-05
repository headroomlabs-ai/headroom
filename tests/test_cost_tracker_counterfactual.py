"""Tests for CostTracker savings calculation.

Savings are computed at model list price: saved_tokens * input_cost_per_token.
This is simple, monotonic, and transparent.
"""

from __future__ import annotations

from tests._dotenv import (
    autouse_apply_env,
    importorskip_no_env_leak,
    load_env_overrides,
)

_env_overrides = load_env_overrides()
apply_dotenv = autouse_apply_env(_env_overrides)

importorskip_no_env_leak("litellm")


def test_savings_at_list_price():
    """savings_usd = tokens_saved * model list input price."""
    from headroom.proxy.server import CostTracker

    ct = CostTracker()
    model = "claude-sonnet-4-20250514"

    ct.record_tokens(
        model,
        tokens_saved=100_000,
        tokens_sent=50_000,
        cache_read_tokens=900_000,
        cache_write_tokens=0,
        uncached_tokens=50_000,
    )
    stats = ct.stats()

    # Savings should be 100k tokens * list input price (NOT affected by cache mix)
    import litellm

    from headroom.pricing.litellm_pricing import resolve_litellm_model

    resolved = resolve_litellm_model(model)
    info = litellm.model_cost.get(resolved, {})
    list_price = info.get("input_cost_per_token", 0)

    expected = 100_000 * list_price
    assert stats["total_tokens_saved"] == 100_000
    assert abs(stats["savings_usd"] - expected) < 0.001


def test_savings_monotonic():
    """Adding more saved tokens always increases savings_usd."""
    from headroom.proxy.server import CostTracker

    ct = CostTracker()
    model = "claude-sonnet-4-20250514"

    ct.record_tokens(model, tokens_saved=10_000, tokens_sent=5_000)
    stats1 = ct.stats()

    ct.record_tokens(model, tokens_saved=10_000, tokens_sent=5_000)
    stats2 = ct.stats()

    assert stats2["savings_usd"] >= stats1["savings_usd"]
    assert stats2["total_tokens_saved"] == 20_000


def test_savings_zero_when_no_tokens_saved():
    """No tokens saved → savings_usd is 0."""
    from headroom.proxy.server import CostTracker

    ct = CostTracker()
    model = "claude-sonnet-4-20250514"

    ct.record_tokens(model, tokens_saved=0, tokens_sent=5_000)
    stats = ct.stats()

    assert stats["savings_usd"] == 0
    assert stats["total_tokens_saved"] == 0


def test_negative_token_savings_are_clamped_to_zero():
    """Estimator artifacts must not reduce cumulative savings below reality."""
    from headroom.proxy.server import CostTracker

    ct = CostTracker()

    ct.record_tokens("openai-compatible", tokens_saved=-500, tokens_sent=5_000)
    stats = ct.stats()

    assert stats["total_tokens_saved"] == 0
    assert stats["per_model"]["openai-compatible"]["tokens_saved"] == 0


def test_multi_model_savings():
    """Savings across multiple models use each model's own list price."""
    from headroom.proxy.server import CostTracker

    ct = CostTracker()

    ct.record_tokens("claude-sonnet-4-20250514", tokens_saved=50_000, tokens_sent=10_000)
    ct.record_tokens("claude-haiku-4-5-20251001", tokens_saved=50_000, tokens_sent=10_000)

    stats = ct.stats()

    # Haiku is cheaper than Sonnet, so same tokens saved → different $
    assert stats["total_tokens_saved"] == 100_000
    assert stats["savings_usd"] > 0

    # Verify per-model breakdown exists
    assert len(stats["per_model"]) == 2


def test_no_cost_without_headroom_field():
    """cost_without_headroom_usd should NOT be in stats (removed to avoid confusion)."""
    from headroom.proxy.server import CostTracker

    ct = CostTracker()
    ct.record_tokens("claude-sonnet-4-20250514", tokens_saved=10_000, tokens_sent=5_000)
    stats = ct.stats()

    assert "cost_without_headroom_usd" not in stats


def test_budget_enforced_after_recording_costs():
    """record_tokens must populate cost history so check_budget enforces the limit.

    Regression test: _costs was never written, so check_budget always
    returned (True, budget_limit) and budgets were silently unenforced.
    """
    from headroom.proxy.server import CostTracker

    ct = CostTracker(budget_limit_usd=0.0001, budget_period="daily")
    allowed, remaining = ct.check_budget()
    assert allowed  # nothing spent yet

    # ~$1.50+ of Sonnet input at list price — far over the budget
    ct.record_tokens(
        "claude-sonnet-4-20250514",
        tokens_saved=0,
        tokens_sent=500_000,
        uncached_tokens=500_000,
        output_tokens=10_000,
    )

    assert ct.get_period_cost() > 0
    allowed, remaining = ct.check_budget()
    assert not allowed
    assert remaining == 0


def test_budget_input_cost_counted_without_usage_breakdown():
    """When the call site has no API usage breakdown (cache/uncached all 0),
    tokens_sent must be used as the input count — input cost must not be
    silently dropped from the budget."""
    from headroom.proxy.server import CostTracker

    ct = CostTracker(budget_limit_usd=100.0)
    ct.record_tokens(
        "claude-sonnet-4-20250514",
        tokens_saved=0,
        tokens_sent=500_000,
    )

    # 500k input tokens at Sonnet list price is ~$1.50 — must be > output-only
    assert ct.get_period_cost() > 0.5
