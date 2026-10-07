"""Price provenance is a ledger dimension independent of usage provenance.

#3732: when no price source knows a model, ``OpenAIProvider`` falls back to the
GPT-4o-tier default -- a guess, not a lookup. Per the maintainer's direction on
#3825, that guess must NOT reuse ``COST_BASIS_ESTIMATED``: a record can carry
provider-reported usage priced at a guessed default, and consumers need the two
dimensions separable. ``estimated_pct`` keeps meaning "the input count was
estimated"; price-guessed spend is reported under ``price_estimated_*``.

These tests pin the marking, the separable reporting, and that the existing
estimated-basis policy treats a guessed-price record as non-authoritative on
all three policies. The provenance source itself (per-lookup, not the sticky
warning-dedup set) is pinned in test_pricing_from_litellm.py.
"""

from __future__ import annotations

import pytest

from headroom.proxy.budget_basis_policy import (
    BUDGET_BASIS_BLOCK,
    BUDGET_BASIS_IGNORE,
    PRICE_BASIS_ESTIMATED,
    PRICE_BASIS_MEASURED,
)
from tests._dotenv import (
    autouse_apply_env,
    importorskip_no_env_leak,
    load_env_overrides,
)
from tests._pricing_models import anthropic_pricing_model

_env_overrides = load_env_overrides()
apply_dotenv = autouse_apply_env(_env_overrides)

importorskip_no_env_leak("litellm")

MODEL = anthropic_pricing_model()


def _tracker(**kwargs):
    from headroom.proxy.server import CostTracker

    return CostTracker(**kwargs)


def _book_reported_usage_guessed_price(ct, tokens: int = 200_000):
    """Book one record: provider-reported usage, guessed price.

    ``uncached_tokens`` set means the usage basis is measured; the price basis
    is stamped estimated, as the provider lookup would report for an unknown
    model (#3732). The ledger stamps what it is told -- the stamp's source is
    pinned separately in test_pricing_from_litellm.py.
    """
    ct.record_tokens(
        MODEL,
        tokens_saved=0,
        tokens_sent=tokens,
        uncached_tokens=tokens,
        output_tokens=1_000,
        price_basis=PRICE_BASIS_ESTIMATED,
    )


# ── Separable reporting ──────────────────────────────────────────────


def test_guessed_price_reported_separately_from_usage_estimates():
    """Price-guessed spend lands in price_estimated_*, not the usage buckets."""
    ct = _tracker(budget_limit_usd=100.0)
    _book_reported_usage_guessed_price(ct)

    basis = ct.stats()["budget_basis"]
    assert basis["price_estimated_usd"] > 0
    assert basis["price_estimated_records"] == 1
    # Usage was provider-reported, so the same dollars also sit in the
    # usage-measured bucket: the two dimensions are independent views of one
    # record, not disjoint buckets.
    assert basis["measured_usd"] > 0
    assert basis["price_estimated_usd"] == pytest.approx(basis["measured_usd"])
    assert basis["estimated_usd"] == 0
    assert basis["estimated_records"] == 0
    assert basis["estimated_pct"] == 0.0
    assert basis["total_usd"] == pytest.approx(basis["measured_usd"])


def test_price_basis_defaults_to_measured():
    """Records booked without a stamp behave exactly as before."""
    ct = _tracker(budget_limit_usd=100.0)
    ct.record_tokens(
        MODEL, tokens_saved=0, tokens_sent=200_000, uncached_tokens=200_000, output_tokens=1_000
    )

    basis = ct.stats()["budget_basis"]
    assert basis["price_estimated_usd"] == 0
    assert basis["price_estimated_records"] == 0
    assert basis["non_authoritative_usd"] == 0


def test_record_estimated_on_both_dimensions_counts_once():
    """A record guessed on both dimensions is one non-authoritative record."""
    ct = _tracker(budget_limit_usd=100.0)
    # No usage breakdown -> usage-estimated; price stamped estimated too.
    ct.record_tokens(
        MODEL,
        tokens_saved=0,
        tokens_sent=200_000,
        output_tokens=1_000,
        price_basis=PRICE_BASIS_ESTIMATED,
    )

    basis = ct.stats()["budget_basis"]
    assert basis["estimated_usd"] > 0
    assert basis["price_estimated_usd"] > 0
    assert basis["non_authoritative_usd"] == pytest.approx(basis["total_usd"])


# ── Policy behavior ──────────────────────────────────────────────────


def test_ignore_policy_excludes_guessed_price_from_budget():
    """policy=ignore: a guessed-price record does not consume the budget."""
    ct = _tracker(budget_limit_usd=0.01, estimated_basis_policy=BUDGET_BASIS_IGNORE)
    _book_reported_usage_guessed_price(ct)

    total = ct.stats()["budget_basis"]["total_usd"]
    assert total > 0.01  # would exceed the budget if it counted
    allowed, remaining = ct.check_budget()
    assert allowed is True
    assert remaining == pytest.approx(0.01)


def test_block_policy_refuses_on_guessed_price():
    """policy=block: any guessed-price spend refuses, naming the price guess."""
    ct = _tracker(budget_limit_usd=100.0, estimated_basis_policy=BUDGET_BASIS_BLOCK)
    _book_reported_usage_guessed_price(ct)

    allowed, remaining = ct.check_budget()
    assert allowed is False
    assert remaining == 0.0
    detail = ct.budget_denial_detail()
    assert "unknown-model default guess" in detail


def test_count_policy_enforces_guessed_price_like_before():
    """policy=count (default): guessed-price spend enforces exactly as before."""
    ct = _tracker(budget_limit_usd=0.01)  # default policy is count
    _book_reported_usage_guessed_price(ct)

    allowed, _ = ct.check_budget()
    assert allowed is False
    detail = ct.budget_denial_detail()
    assert "unknown-model default guess" in detail


def test_unrecognized_price_basis_degrades_to_measured():
    """A bad stamp must not silently change enforcement."""
    ct = _tracker(budget_limit_usd=100.0)
    ct.record_tokens(
        MODEL,
        tokens_saved=0,
        tokens_sent=200_000,
        uncached_tokens=200_000,
        output_tokens=1_000,
        price_basis="bogus",
    )

    basis = ct.stats()["budget_basis"]
    assert basis["price_estimated_usd"] == 0
    assert basis["price_estimated_records"] == 0


def test_price_basis_constants_are_distinct_from_usage_basis():
    """The two dimensions can never collapse into one bucket."""
    assert PRICE_BASIS_ESTIMATED != "estimated"
    assert PRICE_BASIS_MEASURED == "measured"
