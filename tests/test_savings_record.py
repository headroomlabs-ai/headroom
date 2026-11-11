"""The per-request savings record: counterfactual cost minus actual cost."""

from __future__ import annotations

import dataclasses

import pytest

from headroom.pricing import counterfactual
from headroom.pricing.counterfactual import CacheMix, CacheRates
from headroom.proxy.savings import (
    ItemCounts,
    ItemMeasure,
    compute_savings_record,
    measure_items,
)

# $3/M input, $0.30/M cache read, $3.75/M 5m write, $6/M 1h write; the
# long-context tier doubles everything.
IN, READ, WRITE, WRITE_1H = 3e-6, 3e-7, 3.75e-6, 6e-6
BASE = CacheRates(read=READ, write_5m=WRITE, write_1h=WRITE_1H, uncached=IN)
LONG = CacheRates(read=2 * READ, write_5m=2 * WRITE, write_1h=2 * WRITE_1H, uncached=2 * IN)


@pytest.fixture(autouse=True)
def _fixed_catalog(monkeypatch):
    """Pin the catalog so the arithmetic is tested whatever litellm ships."""

    def rates(model, *, long_context=False, provider=None, for_billing=False):
        return None if model == "unpriced" else (LONG if long_context else BASE)

    monkeypatch.setattr(counterfactual, "resolve_rates", rates)
    monkeypatch.setattr(
        counterfactual,
        "catalog_min_cacheable_tokens",
        lambda model: 4096 if model == "haiku" else 1024,
    )


def _formula(m: ItemMeasure, u: CacheMix) -> float:
    """The closed form the record must agree with."""
    billed = u.read + u.write_5m + u.uncached
    k = billed / m.sent_tokens
    warm = min(1.0, (u.read / billed) / (m.sent_shared_prefix / m.sent_tokens))
    w = u.write_5m / (u.uncached + u.write_5m)
    p_miss = w * WRITE + (1 - w) * IN
    return (
        k * m.removed_old * (warm * READ + (1 - warm) * p_miss)
        + k * m.removed_new * p_miss
        - k * m.bust * warm * (p_miss - READ)
    )


WARM = CacheMix(uncached=2_000, read=90_000, write_5m=8_000)
MEASURE = ItemMeasure(
    sent_tokens=95_000, removed_old=50_000, removed_new=10_000, sent_shared_prefix=88_000, bust=0
)


def test_worked_example():
    record = compute_savings_record(MEASURE, WARM, model="sonnet", provider="anthropic")
    assert record.k == pytest.approx(1.0526, abs=1e-4)
    assert record.warm == pytest.approx(0.97159, abs=1e-5)
    assert record.write_share == pytest.approx(0.8)
    assert record.saved_usd == pytest.approx(0.058618, abs=1e-6)
    assert record.saved_usd == pytest.approx(_formula(MEASURE, WARM), abs=1e-9)
    assert record.saved_usd == pytest.approx(
        record.counterfactual_input_cost_usd - record.actual_input_cost_usd
    )
    assert record.saved_tokens == pytest.approx(100_000 / 95_000 * 60_000)
    assert not record.estimated


def test_a_cache_bust_headroom_caused_is_charged_against_it():
    busted = dataclasses.replace(MEASURE, bust=5_000)
    record = compute_savings_record(busted, WARM, model="sonnet", provider="anthropic")
    assert record.saved_usd == pytest.approx(_formula(busted, WARM), abs=1e-9)
    assert record.saved_usd == pytest.approx(0.058618 - 0.016875, abs=1e-6)


def test_a_bust_never_claims_more_reads_than_the_request_missed():
    """Item-level P_S undercounts a token-level hit, so hit > sigma clips warm
    to 1; the bust charge must still not drive the counterfactual negative."""
    measure = ItemMeasure(
        sent_tokens=10_000, removed_old=0, removed_new=0, sent_shared_prefix=2_000, bust=8_000
    )
    usage = CacheMix(uncached=500, read=9_000, write_5m=500)
    record = compute_savings_record(measure, usage, model="sonnet", provider="anthropic")
    assert record.counterfactual_input_cost_usd >= 0
    # At most the 1,000 missed tokens could have been reads instead.
    assert record.saved_usd == pytest.approx(-(500 * (IN - READ) + 500 * (WRITE - READ)))


def test_an_old_item_headroom_grew_inherits_no_reads_below_the_cache_minimum():
    """JerrettDavis's #4236 case: a 1,000-token item sent as 10,000 tokens on
    successive requests, prompt cached. The 1,000-token counterfactual is
    under the 1,024-token minimum, so none of it could have been a read."""
    measure = ItemMeasure(
        sent_tokens=10_000, removed_old=-9_000, removed_new=0, sent_shared_prefix=10_000, bust=0
    )
    usage = CacheMix(read=9_000, uncached=1_000)
    for provider in ("openai", "anthropic"):
        record = compute_savings_record(measure, usage, model="gpt", provider=provider)
        assert record.actual_input_cost_usd == pytest.approx(0.0057)
        assert record.counterfactual_input_cost_usd == pytest.approx(1_000 * IN)  # $0.003
        assert record.saved_usd == pytest.approx(-0.0027)


def test_summed_iteration_usage_does_not_calibrate_k():
    """Usage summed over server-side iterations describes more prompts than
    the one sent body, so it must not scale the removal."""
    summed = CacheMix(read=270_000, uncached=6_000, write_5m=24_000)
    record = compute_savings_record(
        MEASURE, summed, model="sonnet", provider="anthropic", estimated=True
    )
    assert record.k == 1.0
    assert record.estimated
    assert record.saved_tokens == 60_000


def test_a_cold_cache_prices_every_removed_token_as_a_miss():
    cold = CacheMix(uncached=10_000, read=0, write_5m=90_000)
    record = compute_savings_record(MEASURE, cold, model="sonnet", provider="anthropic")
    assert record.warm == 0.0
    p_miss = 0.9 * WRITE + 0.1 * IN
    assert record.saved_usd == pytest.approx(100_000 / 95_000 * 60_000 * p_miss)


def test_one_hour_writes_price_at_their_own_rate():
    usage = CacheMix(uncached=2_000, read=90_000, write_1h=8_000)
    record = compute_savings_record(MEASURE, usage, model="sonnet", provider="anthropic")
    k, warm = 100_000 / 95_000, record.warm
    p_miss = 0.8 * WRITE_1H + 0.2 * IN
    expected = k * 50_000 * (warm * READ + (1 - warm) * p_miss) + k * 10_000 * p_miss
    assert record.saved_usd == pytest.approx(expected)


def test_an_inferred_openai_write_is_the_uncached_slice_not_a_write():
    usage = CacheMix(read=90_000, write_total=10_000, uncached=10_000, inferred_write=True)
    record = compute_savings_record(MEASURE, usage, model="gpt", provider="openai")
    assert record.k == pytest.approx(100_000 / 95_000)
    assert record.write_share == 0.0


def test_openai_reads_come_in_128_token_blocks():
    measure = ItemMeasure(
        sent_tokens=10_000, removed_old=1_000, removed_new=0, sent_shared_prefix=10_000, bust=0
    )
    usage = CacheMix(read=9_984, uncached=16)  # 78 blocks
    record = compute_savings_record(measure, usage, model="gpt", provider="openai")
    # 1,000 * k removed reads would end mid-block: only whole blocks count.
    total_read = 9_984 + 1_000 * record.k
    reads = (total_read // 128) * 128 - 9_984
    assert record.saved_usd == pytest.approx(reads * READ + (1_000 * record.k - reads) * IN)


def test_a_counterfactual_prefix_below_the_models_cache_minimum_is_never_warm():
    # A 3,000-token old prefix: cacheable on Sonnet (1,024), not on Haiku (4,096).
    measure = ItemMeasure(
        sent_tokens=10_000, removed_old=1_000, removed_new=0, sent_shared_prefix=2_000, bust=0
    )
    usage = CacheMix(uncached=5_000, read=5_000)
    assert compute_savings_record(measure, usage, model="sonnet").warm > 0
    assert compute_savings_record(measure, usage, model="haiku").warm == 0.0


def test_the_counterfactual_is_priced_at_its_own_context_tier():
    measure = ItemMeasure(
        sent_tokens=150_000, removed_old=0, removed_new=100_000, sent_shared_prefix=0, bust=0
    )
    record = compute_savings_record(measure, CacheMix(uncached=150_000), model="sonnet")
    assert record.actual_input_cost_usd == pytest.approx(150_000 * IN)
    assert record.counterfactual_input_cost_usd == pytest.approx(250_000 * 2 * IN)


def test_missing_usage_prices_at_list_per_tier_and_says_so():
    measure = dataclasses.replace(MEASURE, sent_tokens=150_000, removed_new=50_000)
    record = compute_savings_record(measure, None, model="sonnet")
    assert record.estimated
    assert record.actual_input_cost_usd == pytest.approx(150_000 * IN)
    assert record.counterfactual_input_cost_usd == pytest.approx(250_000 * 2 * IN)


def test_an_unpriced_model_reports_tokens_but_no_dollars():
    record = compute_savings_record(MEASURE, WARM, model="unpriced")
    assert not record.priced
    assert record.saved_usd == 0.0
    assert record.removed_tokens == 60_000


def test_items_split_into_old_and_new_against_the_previous_request():
    def item(name: str, original: int, sent: int, sent_name: str | None = None) -> ItemCounts:
        return ItemCounts(name, original, sent_name or name + "'", sent)

    first = [item("tools", 1_000, 200), item("sys", 500, 500, "sys"), item("u1", 300, 300, "u1")]
    second = [
        item("tools", 1_000, 200),
        item("sys", 500, 500, "sys"),
        # Headroom compressed this item differently this time: its sent copy
        # changed, so the provider missed it although the client resent it.
        item("u1", 300, 250, "u1-recompressed"),
        item("t1", 4_000, 1_000),
        # An item Headroom grew counts against the removal.
        item("t2", 100, 160),
    ]
    assert measure_items(first, None) == ItemMeasure(
        sent_tokens=1_000, removed_old=0, removed_new=800, sent_shared_prefix=0, bust=0
    )
    assert measure_items(second, first) == ItemMeasure(
        sent_tokens=2_110,
        removed_old=800 + 50,
        removed_new=3_000 - 60,
        sent_shared_prefix=700,
        bust=250,
        grown_new=60,
    )


def test_billed_buckets_price_at_billing_rates(monkeypatch):
    """A catalog row without a cache-read rate bills reads via litellm's
    calculator (``for_billing``); only the changed tokens use the estimate."""
    billing = dataclasses.replace(BASE, read=5e-7)

    def rates(model, *, long_context=False, provider=None, for_billing=False):
        return billing if for_billing else BASE

    monkeypatch.setattr(counterfactual, "resolve_rates", rates)
    record = compute_savings_record(MEASURE, WARM, model="sonnet", provider="anthropic")
    assert record.actual_input_cost_usd == pytest.approx(2_000 * IN + 90_000 * 5e-7 + 8_000 * WRITE)
    assert record.saved_usd == pytest.approx(0.058618, abs=1e-6)


def _fully_cached_original_of(original: int, *, model: str, provider: str):
    """A 10,000-token sent body read wholly from cache, whose original was
    ``original`` tokens: the counterfactual is that original alone."""
    measure = ItemMeasure(
        sent_tokens=10_000,
        removed_old=original - 10_000,
        removed_new=0,
        sent_shared_prefix=10_000,
        bust=0,
    )
    return compute_savings_record(measure, CacheMix(read=10_000), model=model, provider=provider)


@pytest.mark.parametrize(
    ("original", "reads"),
    [(1_023, 0), (1_024, 1_024), (1_151, 1_024), (1_152, 1_152), (2_000, 1_920)],
)
def test_an_openai_counterfactual_reads_whole_blocks_above_its_own_minimum(original, reads):
    record = _fully_cached_original_of(original, model="gpt", provider="openai")
    expected = reads * READ + (original - reads) * IN
    assert record.counterfactual_input_cost_usd == pytest.approx(expected)


@pytest.mark.parametrize(("original", "reads"), [(4_095, 0), (4_096, 4_096), (4_097, 4_097)])
def test_an_anthropic_counterfactual_obeys_its_models_cache_minimum(original, reads):
    record = _fully_cached_original_of(original, model="haiku", provider="anthropic")
    expected = reads * READ + (original - reads) * IN
    assert record.counterfactual_input_cost_usd == pytest.approx(expected)


def test_a_counterfactual_too_small_to_cache_pays_no_write_premium():
    """Below its minimum the counterfactual can be neither read nor written."""
    measure = ItemMeasure(
        sent_tokens=10_000, removed_old=-9_000, removed_new=0, sent_shared_prefix=10_000, bust=0
    )
    usage = CacheMix(read=9_000, write_5m=1_000)
    record = compute_savings_record(measure, usage, model="sonnet", provider="anthropic")
    assert record.counterfactual_input_cost_usd == pytest.approx(1_000 * IN)


def test_a_grown_old_item_takes_its_reads_out_even_with_new_items():
    """The new items keep the counterfactual above the minimum overall; its
    1,000-token old prefix is still too small to have been read."""
    measure = ItemMeasure(
        sent_tokens=30_000,
        removed_old=-9_000,
        removed_new=20_000,
        sent_shared_prefix=10_000,
        bust=0,
    )
    usage = CacheMix(read=9_000, uncached=21_000)
    record = compute_savings_record(measure, usage, model="gpt", provider="openai")
    assert record.counterfactual_input_cost_usd == pytest.approx(41_000 * IN)


def test_a_catalog_minimum_the_provider_disproved_is_not_applied():
    """Haiku's catalog says 4,096, but the provider read 3,000: keeping the
    counterfactual's 1,500 reads is the conservative price."""
    measure = ItemMeasure(
        sent_tokens=10_000, removed_old=-5_000, removed_new=0, sent_shared_prefix=10_000, bust=0
    )
    usage = CacheMix(read=3_000, uncached=7_000)
    record = compute_savings_record(measure, usage, model="haiku", provider="anthropic")
    assert record.counterfactual_input_cost_usd == pytest.approx(1_500 * READ + 3_500 * IN)


def test_nothing_removed_saves_nothing_whatever_the_read_alignment():
    measure = ItemMeasure(
        sent_tokens=10_000, removed_old=0, removed_new=0, sent_shared_prefix=2_000, bust=0
    )
    for provider, usage in (
        ("openai", CacheMix(read=9_000, uncached=1_000)),
        ("anthropic", CacheMix(read=900, write_5m=9_100)),
    ):
        record = compute_savings_record(measure, usage, model="haiku", provider=provider)
        assert record.saved_usd == pytest.approx(0.0, abs=1e-12)


def test_a_shrunk_bucket_is_never_priced_below_zero(monkeypatch):
    """Reads a grown item gave up price at the billing rate they were billed
    at, not at the estimate rate, so the counterfactual cannot go negative."""
    billing = dataclasses.replace(BASE, read=1.5e-6)

    def rates(model, *, long_context=False, provider=None, for_billing=False):
        return billing if for_billing else BASE

    monkeypatch.setattr(counterfactual, "resolve_rates", rates)
    measure = ItemMeasure(
        sent_tokens=10_000, removed_old=-9_000, removed_new=0, sent_shared_prefix=10_000, bust=0
    )
    record = compute_savings_record(
        measure, CacheMix(read=9_000, uncached=1_000), model="gpt", provider="openai"
    )
    assert record.counterfactual_input_cost_usd == pytest.approx(1_000 * IN)


def test_a_disproved_minimum_still_writes_nothing_smaller_than_was_cached():
    """Haiku's catalog minimum is disproved by a 2,000-token read; a 500-token
    counterfactual is below anything shown cacheable, so it writes nothing."""
    measure = ItemMeasure(
        sent_tokens=3_000, removed_old=-2_500, removed_new=0, sent_shared_prefix=3_000, bust=0
    )
    usage = CacheMix(read=2_000, write_5m=1_000)
    record = compute_savings_record(measure, usage, model="haiku", provider="anthropic")
    reads = 2_000 + -2_500 * (2 / 3)
    assert record.counterfactual_input_cost_usd == pytest.approx(reads * READ + (500 - reads) * IN)


def test_a_lost_baseline_prices_the_new_removal_at_the_read_floor():
    """With the conversation's baseline evicted, the new items may be old ones
    the provider had cached. Pricing them as new, at the miss rate, booked
    $0.2236 here; the read rate, the floor of any input token, books $0.0152.
    The 1,000 tokens Headroom added to a new item still cost the miss rate."""
    lost = ItemMeasure(
        sent_tokens=95_000,
        removed_old=0,
        removed_new=60_000 - 1_000,
        sent_shared_prefix=0,
        bust=0,
        aligned=False,
        baseline_lost=True,
        grown_new=1_000,
    )
    k, p_miss = 100_000 / 95_000, 0.8 * WRITE + 0.2 * IN
    as_new = compute_savings_record(
        dataclasses.replace(lost, baseline_lost=False), WARM, model="sonnet", provider="anthropic"
    )
    assert as_new.saved_usd == pytest.approx(k * 59_000 * p_miss)  # $0.2236

    record = compute_savings_record(lost, WARM, model="sonnet", provider="anthropic")
    assert record.saved_usd == pytest.approx(k * 60_000 * READ - k * 1_000 * p_miss)  # $0.0152
    assert record.saved_usd == pytest.approx(0.015158, abs=1e-6)
    assert record.estimated
    # Without usage too: list price for the rest, the read rate for the removal.
    unbilled = compute_savings_record(lost, None, model="sonnet", provider="anthropic")
    assert unbilled.saved_usd == pytest.approx(60_000 * READ - 1_000 * IN)
