"""Budget records must say whether their input count was measured or estimated.

#2713: when a provider response carries no input-token breakdown,
``record_tokens`` substitutes Headroom's own ``tokens_sent`` for the input
count. The fallback is right — dropping input cost would under-enforce far
worse — but the resulting record used to be indistinguishable from a
provider-measured one, so ``check_budget`` (a hard control) could refuse or
allow on the strength of a guess with nothing saying so.

These tests pin the marking, the deduped warning, the separable ledger, and
the three operator policies.
"""

from __future__ import annotations

import logging

import pytest

from tests._dotenv import (
    autouse_apply_env,
    importorskip_no_env_leak,
    load_env_overrides,
)

_env_overrides = load_env_overrides()
apply_dotenv = autouse_apply_env(_env_overrides)

importorskip_no_env_leak("litellm")

MODEL = "claude-sonnet-4-20250514"


@pytest.fixture(autouse=True)
def _reset_warning_dedup():
    """The per-model warn-once set is module-global; keep tests order-independent."""
    import headroom.proxy.cost as cost_mod

    cost_mod._warned_estimated_basis_models.clear()
    yield
    cost_mod._warned_estimated_basis_models.clear()


def _tracker(**kwargs):
    from headroom.proxy.server import CostTracker

    return CostTracker(**kwargs)


# ── Basis marking ────────────────────────────────────────────────────


def test_missing_usage_breakdown_books_estimated_basis():
    """No breakdown → the record is marked estimated and reported as such."""
    ct = _tracker(budget_limit_usd=100.0)
    ct.record_tokens(MODEL, tokens_saved=0, tokens_sent=50_000, output_tokens=1_000)

    basis = ct.stats()["budget_basis"]
    assert basis["estimated_usd"] > 0
    assert basis["measured_usd"] == 0
    assert basis["estimated_records"] == 1
    assert basis["estimated_pct"] == 100.0


def test_provider_breakdown_books_measured_basis():
    """A reported breakdown → measured; nothing lands in the estimated bucket."""
    ct = _tracker(budget_limit_usd=100.0)
    ct.record_tokens(
        MODEL,
        tokens_saved=0,
        tokens_sent=50_000,
        uncached_tokens=30_000,
        output_tokens=1_000,
    )

    basis = ct.stats()["budget_basis"]
    assert basis["measured_usd"] > 0
    assert basis["estimated_usd"] == 0
    assert basis["estimated_records"] == 0
    assert basis["estimated_pct"] == 0.0


def test_cache_read_only_response_counts_as_measured():
    """A fully cache-read turn reports usage, so it is not an estimate."""
    ct = _tracker(budget_limit_usd=100.0)
    ct.record_tokens(
        MODEL,
        tokens_saved=0,
        tokens_sent=50_000,
        cache_read_tokens=40_000,
        output_tokens=1_000,
    )

    assert ct.stats()["budget_basis"]["estimated_usd"] == 0


def test_mixed_records_stay_separable_and_sum_to_total():
    """Default policy is unchanged: the budget still sees every booked dollar."""
    ct = _tracker(budget_limit_usd=100.0)
    ct.record_tokens(MODEL, tokens_saved=0, tokens_sent=50_000, output_tokens=1_000)
    ct.record_tokens(
        MODEL,
        tokens_saved=0,
        tokens_sent=50_000,
        uncached_tokens=30_000,
        output_tokens=1_000,
    )

    basis = ct.stats()["budget_basis"]
    assert basis["records"] == 2
    assert basis["estimated_records"] == 1
    assert basis["measured_usd"] > 0
    assert basis["estimated_usd"] > 0
    assert basis["total_usd"] == pytest.approx(basis["measured_usd"] + basis["estimated_usd"])
    # Regression guard: `count` (the default) enforces against total spend
    # exactly as it did before this change.
    assert ct.get_period_cost() == pytest.approx(basis["total_usd"])


def test_get_period_cost_can_filter_by_basis():
    ct = _tracker(budget_limit_usd=100.0)
    ct.record_tokens(MODEL, tokens_saved=0, tokens_sent=50_000, output_tokens=1_000)
    ct.record_tokens(
        MODEL, tokens_saved=0, tokens_sent=50_000, uncached_tokens=30_000, output_tokens=1_000
    )

    measured = ct.get_period_cost("measured")
    estimated = ct.get_period_cost("estimated")
    assert measured > 0
    assert estimated > 0
    assert ct.get_period_cost() == pytest.approx(measured + estimated)


# ── Warning ──────────────────────────────────────────────────────────


def test_estimated_basis_warns_once_per_model(caplog):
    """A route that never reports usage must not flood proxy.log (cf. #2504)."""
    ct = _tracker(budget_limit_usd=100.0)

    with caplog.at_level(logging.WARNING, logger="headroom.proxy"):
        for _ in range(5):
            ct.record_tokens(MODEL, tokens_saved=0, tokens_sent=10_000, output_tokens=100)

    hits = [r for r in caplog.records if "budget basis estimated" in r.getMessage()]
    assert len(hits) == 1
    assert MODEL in hits[0].getMessage()


def test_distinct_models_each_warn_once(caplog):
    ct = _tracker(budget_limit_usd=100.0)
    other = "claude-haiku-4-5-20251001"

    with caplog.at_level(logging.WARNING, logger="headroom.proxy"):
        for model in (MODEL, MODEL, other, other):
            ct.record_tokens(model, tokens_saved=0, tokens_sent=10_000, output_tokens=100)

    msgs = [r.getMessage() for r in caplog.records if "budget basis estimated" in r.getMessage()]
    assert sum(MODEL in m for m in msgs) == 1
    assert sum(other in m for m in msgs) == 1


def test_measured_records_do_not_warn(caplog):
    ct = _tracker(budget_limit_usd=100.0)

    with caplog.at_level(logging.WARNING, logger="headroom.proxy"):
        ct.record_tokens(
            MODEL, tokens_saved=0, tokens_sent=10_000, uncached_tokens=9_000, output_tokens=100
        )

    assert not [r for r in caplog.records if "budget basis estimated" in r.getMessage()]


# ── Enforcement policies ─────────────────────────────────────────────


def test_count_policy_lets_estimated_spend_exhaust_the_budget():
    """Default: an estimate consumes the budget, as it always has."""
    ct = _tracker(budget_limit_usd=0.0001, estimated_basis_policy="count")
    ct.record_tokens(MODEL, tokens_saved=0, tokens_sent=500_000, output_tokens=10_000)

    allowed, remaining = ct.check_budget()
    assert not allowed
    assert remaining == 0


def test_ignore_policy_keeps_estimated_spend_out_of_enforcement():
    """`ignore`: the record is still booked and reported, but doesn't enforce."""
    ct = _tracker(budget_limit_usd=0.0001, estimated_basis_policy="ignore")
    ct.record_tokens(MODEL, tokens_saved=0, tokens_sent=500_000, output_tokens=10_000)

    allowed, _remaining = ct.check_budget()
    assert allowed
    # Still visible in the ledger — ignored for enforcement, not dropped.
    assert ct.stats()["budget_basis"]["estimated_usd"] > 0


def test_ignore_policy_still_enforces_measured_spend():
    ct = _tracker(budget_limit_usd=0.0001, estimated_basis_policy="ignore")
    ct.record_tokens(
        MODEL,
        tokens_saved=0,
        tokens_sent=500_000,
        uncached_tokens=500_000,
        output_tokens=10_000,
    )

    allowed, _remaining = ct.check_budget()
    assert not allowed


def test_block_policy_refuses_once_any_estimated_spend_exists():
    """`block`: fail closed rather than enforce a hard limit on a guess."""
    ct = _tracker(budget_limit_usd=1_000_000.0, estimated_basis_policy="block")
    allowed, _remaining = ct.check_budget()
    assert allowed  # nothing booked yet

    ct.record_tokens(MODEL, tokens_saved=0, tokens_sent=1_000, output_tokens=10)

    allowed, remaining = ct.check_budget()
    assert not allowed
    assert remaining == 0.0


def test_block_policy_is_inert_without_a_budget_limit():
    """No limit configured means no hard control to protect."""
    ct = _tracker(budget_limit_usd=None, estimated_basis_policy="block")
    ct.record_tokens(MODEL, tokens_saved=0, tokens_sent=1_000, output_tokens=10)

    allowed, remaining = ct.check_budget()
    assert allowed
    assert remaining == float("inf")


def test_block_policy_allows_purely_measured_traffic():
    ct = _tracker(budget_limit_usd=1_000_000.0, estimated_basis_policy="block")
    ct.record_tokens(
        MODEL, tokens_saved=0, tokens_sent=1_000, uncached_tokens=900, output_tokens=10
    )

    allowed, _remaining = ct.check_budget()
    assert allowed


def test_invalid_policy_falls_back_to_count():
    ct = _tracker(budget_limit_usd=0.0001, estimated_basis_policy="nonsense")
    assert ct.estimated_basis_policy == "count"

    ct.record_tokens(MODEL, tokens_saved=0, tokens_sent=500_000, output_tokens=10_000)
    allowed, _remaining = ct.check_budget()
    assert not allowed


# ── Denial message ───────────────────────────────────────────────────


def test_denial_detail_names_the_estimated_share():
    ct = _tracker(budget_limit_usd=0.0001)
    ct.record_tokens(MODEL, tokens_saved=0, tokens_sent=500_000, output_tokens=10_000)

    detail = ct.budget_denial_detail()
    assert "Budget exceeded for daily period" in detail
    assert "Headroom token estimates" in detail


def test_denial_detail_unchanged_for_purely_measured_spend():
    ct = _tracker(budget_limit_usd=0.0001)
    ct.record_tokens(
        MODEL,
        tokens_saved=0,
        tokens_sent=500_000,
        uncached_tokens=500_000,
        output_tokens=10_000,
    )

    assert ct.budget_denial_detail() == "Budget exceeded for daily period"


def test_block_denial_is_distinguishable_from_overspend():
    ct = _tracker(budget_limit_usd=1_000_000.0, estimated_basis_policy="block")
    ct.record_tokens(MODEL, tokens_saved=0, tokens_sent=1_000, output_tokens=10)

    detail = ct.budget_denial_detail()
    assert "Budget enforcement blocked" in detail
    assert "HEADROOM_BUDGET_ESTIMATED_BASIS=block" in detail


# ── Policy resolver ──────────────────────────────────────────────────


def test_resolver_precedence_and_fallback():
    from headroom.proxy.budget_basis_policy import resolve_estimated_basis_policy

    env = {"HEADROOM_BUDGET_ESTIMATED_BASIS": "ignore"}
    assert resolve_estimated_basis_policy("block", env) == "block"  # explicit wins
    assert resolve_estimated_basis_policy(None, env) == "ignore"  # env next
    assert resolve_estimated_basis_policy(None, {}) == "count"  # default
    assert resolve_estimated_basis_policy(None, None) == "count"
    assert resolve_estimated_basis_policy("BLOCK", None) == "block"  # normalized
    assert resolve_estimated_basis_policy("nope", None) == "count"  # fallback


def test_stats_reports_the_active_policy():
    ct = _tracker(budget_limit_usd=10.0, estimated_basis_policy="block")
    assert ct.stats()["budget_estimated_basis"] == "block"
    assert ct.stats()["budget_basis"]["policy"] == "block"
