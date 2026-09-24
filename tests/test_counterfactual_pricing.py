"""Cache-aware pricing of the tokens Headroom kept off the wire.

The defect these cover: savings were priced at ``tokens * input_cost_per_token``
— flat list price — which is correct only for the first request of a cache
window. On traffic that is ~88% cache reads (measured on a real 238M-token
corpus) that overstates the saving by ~2.7x overall and by ~10x for tool-schema
deferral specifically, because tool definitions sit at the very front of the
prompt and are the most cacheable tokens in it.

The tests are organised around the three things that have to be true:
region correctness, provider agnosticism, and honest labelling.
"""

from __future__ import annotations

import pytest

from tests._dotenv import (
    autouse_apply_env,
    importorskip_no_env_leak,
    load_env_overrides,
)
from tests._pricing_models import anthropic_pricing_model

_env_overrides = load_env_overrides()
apply_dotenv = autouse_apply_env(_env_overrides)

importorskip_no_env_leak("litellm")

from headroom.pricing.counterfactual import (  # noqa: E402
    BASIS_CATALOG,
    BASIS_CATALOG_TTL_RATIO,
    BASIS_LIST,
    BASIS_NO_MIX,
    BASIS_UNPRICED,
    CacheMix,
    Region,
    price_savings,
    resolve_rates,
    split_tokens,
    weakest_basis,
)

SONNET = anthropic_pricing_model()


def _warm_anthropic() -> CacheMix:
    """A steady-state agent turn: big cached prefix, small appended delta."""
    return CacheMix.from_usage(
        cache_read_tokens=180_000,
        cache_write_tokens=2_000,
        cache_write_5m_tokens=2_000,
        uncached_input_tokens=500,
    )


def _cold_anthropic() -> CacheMix:
    """First turn of a window: everything is a cache write."""
    return CacheMix.from_usage(
        cache_write_tokens=50_000,
        cache_write_5m_tokens=50_000,
    )


# ── Region correctness: the heart of the fix ──────────────────────────────


def test_prefix_savings_on_a_warm_turn_price_at_the_cache_read_rate():
    """Tool schemas sit ahead of every breakpoint, so a warm turn's are reads.

    This is the 10x case. List pricing valued these at the full input rate; they
    would actually have been billed at Anthropic's 0.10x read rate.
    """
    priced = price_savings(8_000, model=SONNET, mix=_warm_anthropic(), region=Region.PREFIX)

    assert priced.ratio == pytest.approx(0.10, abs=1e-6)
    assert priced.usd < priced.usd_list


def test_prefix_split_is_read_first_not_pro_rata():
    """Proportional bucketing hands prefix tokens a write premium they never paid.

    On a 90/10 read/write request a pro-rata split prices them at
    0.9*0.1 + 0.1*1.25 = 0.215x instead of 0.10x — still ~2x high. Filling the
    read bucket first is what makes the region argument mean anything.
    """
    mix = CacheMix.from_usage(
        cache_read_tokens=90_000,
        cache_write_tokens=10_000,
        cache_write_5m_tokens=10_000,
    )
    split = split_tokens(5_000, mix, Region.PREFIX)

    assert split.read == 5_000
    assert split.write_5m == 0
    assert split.write_1h == 0
    assert split.uncached == 0


def test_live_zone_savings_exclude_the_read_bucket_entirely():
    """Compression works the appended delta, which was never cached.

    Pricing it by the whole-request mix would value a warm turn's compression at
    ~0.1x — an order of magnitude UNDER what the provider would have charged.
    """
    split = split_tokens(5_000, _warm_anthropic(), Region.LIVE_ZONE)

    assert split.read == 0
    assert split.total == pytest.approx(5_000)


def test_live_zone_savings_on_a_cold_turn_exceed_list_price():
    """A removed cache-write token saves 1.25x list, not 1.0x.

    Deliberately NOT clamped to list: that is real money the provider would
    have charged, and clamping it would under-report Headroom in exactly the
    situation where compression is worth most.
    """
    priced = price_savings(8_000, model=SONNET, mix=_cold_anthropic(), region=Region.LIVE_ZONE)

    assert priced.ratio == pytest.approx(1.25, abs=1e-6)
    assert priced.usd > priced.usd_list


def test_region_accepts_the_bare_string_form():
    """``proxy/cost.py`` passes a string to keep litellm off the startup path.

    ``Region`` is a str enum, so "prefix" compares EQUAL to Region.PREFIX but is
    not IDENTICAL to it — an `is` test would route it down the live-zone branch
    and price a cached prefix as if it had never been cached.
    """
    mix = _warm_anthropic()

    assert split_tokens(5_000, mix, "prefix") == split_tokens(5_000, mix, Region.PREFIX)
    assert split_tokens(5_000, mix, "live_zone") == split_tokens(5_000, mix, Region.LIVE_ZONE)


# ── TTL: the 1h write rate that was never priced ──────────────────────────


def test_one_hour_writes_price_above_five_minute_writes():
    """A 1h write bills at 2.00x base against a 5m write's 1.25x.

    The proxy counted and displayed the 5m/1h split all along but priced every
    write at the 5m rate.
    """
    rates = resolve_rates(SONNET)

    assert rates.write_1h > rates.write_5m > rates.uncached > rates.read
    assert rates.write_1h / rates.uncached == pytest.approx(2.00, abs=1e-6)
    assert rates.write_5m / rates.uncached == pytest.approx(1.25, abs=1e-6)


def test_one_hour_ttl_savings_are_valued_at_the_one_hour_rate():
    mix = CacheMix.from_usage(cache_write_tokens=50_000, cache_write_1h_tokens=50_000)
    priced = price_savings(8_000, model=SONNET, mix=mix, region=Region.PREFIX)

    assert priced.ratio == pytest.approx(2.00, abs=1e-6)


def test_untagged_writes_default_to_the_five_minute_ttl():
    """A provider reporting a write total but no split gets the DEFAULT TTL.

    Attributing the remainder to 1h would inflate the counterfactual by 60% of
    the write premium on traffic that never asked for the extended TTL.
    """
    mix = CacheMix.from_usage(cache_write_tokens=10_000).normalized()

    assert mix.write_5m == 10_000
    assert mix.write_1h == 0


def test_long_context_derives_the_one_hour_rate_and_says_so():
    """No catalog publishes a combined 1h + above-200k rate, so it derives."""
    rates = resolve_rates(SONNET, long_context=True)

    assert rates.basis == BASIS_CATALOG_TTL_RATIO
    assert rates.write_1h / rates.uncached == pytest.approx(2.00, abs=1e-6)
    # And it is the EXPENSIVE tier, not the base one.
    assert rates.uncached > resolve_rates(SONNET).uncached


# ── Provider agnosticism: every harness, every backend ────────────────────


def test_openai_discounts_reads_and_charges_no_write_premium():
    """OpenAI: 0.50x reads, writes at list. There is no 5m/1h trade to price."""
    rates = resolve_rates("gpt-4o")

    assert rates.read / rates.uncached == pytest.approx(0.50, abs=1e-6)
    assert rates.write_5m == rates.uncached
    assert rates.write_1h == rates.uncached


def test_openai_inferred_writes_are_dropped_not_charged():
    """An inferred write is the same tokens as ``uncached`` and bills as input.

    Counting it as a write would both double-count those tokens and apply a
    premium OpenAI does not charge.
    """
    mix = CacheMix.from_usage(
        cache_read_tokens=90_000,
        cache_write_tokens=10_000,
        uncached_input_tokens=10_000,
        cache_inferred=True,
    ).normalized()

    assert mix.write_5m == 0
    assert mix.write_1h == 0
    assert mix.read == 90_000
    assert mix.uncached == 10_000


def test_gemini_reports_reads_only_and_still_prices():
    """Gemini populates the read bucket alone; that is enough for PREFIX."""
    mix = CacheMix.from_usage(cache_read_tokens=100_000)
    priced = price_savings(5_000, model="gemini/gemini-2.5-pro", mix=mix, region=Region.PREFIX)

    assert priced.ratio == pytest.approx(0.10, abs=1e-6)
    assert priced.basis == BASIS_CATALOG


# ── Honest labelling: never a precise-looking guess ───────────────────────


def test_no_cache_breakdown_prices_at_list_and_labels_it():
    """The MCP tool path and non-reporting gateways land here.

    List price is the honest answer when there is no mix; what must not happen
    is it being reported as though it were measured.
    """
    priced = price_savings(8_000, model=SONNET, mix=None, region=Region.PREFIX)

    assert priced.basis == BASIS_NO_MIX
    assert priced.usd == priced.usd_list


def test_unknown_model_without_a_fallback_is_unpriced_not_guessed():
    priced = price_savings(8_000, model="not-a-real-model-xyz", mix=_warm_anthropic())

    assert priced.basis == BASIS_UNPRICED
    assert priced.usd == 0.0


def test_unknown_model_with_a_fallback_uses_the_blended_rate():
    priced = price_savings(
        8_000,
        model="not-a-real-model-xyz",
        fallback_rate_per_token=3.0 / 1_000_000,
    )

    assert priced.basis == BASIS_LIST
    assert priced.usd == pytest.approx(0.024)


def test_a_total_reports_its_weakest_contributing_basis():
    """One unpriced request must not launder a total into looking catalog-grade."""
    assert weakest_basis(BASIS_CATALOG, BASIS_NO_MIX) == BASIS_NO_MIX
    assert weakest_basis(BASIS_CATALOG, BASIS_CATALOG) == BASIS_CATALOG
    assert weakest_basis(BASIS_LIST, BASIS_UNPRICED) == BASIS_UNPRICED
    assert weakest_basis(None, None) == BASIS_UNPRICED


def test_a_free_model_saves_nothing_rather_than_falling_back():
    """A real 0.0 price is not a missing price.

    Treating it as missing bills the blended fallback and invents savings for a
    model that costs nothing.
    """
    import litellm

    from headroom.pricing.litellm_pricing import resolve_litellm_model

    free_model = "free-local-model-for-test"
    litellm.model_cost[resolve_litellm_model(free_model)] = {"input_cost_per_token": 0.0}
    resolve_rates.cache_clear()
    try:
        priced = price_savings(
            10_000,
            model=free_model,
            mix=_warm_anthropic(),
            fallback_rate_per_token=3.0 / 1_000_000,
        )
        assert priced.usd == 0.0
        assert priced.usd_list == 0.0
    finally:
        litellm.model_cost.pop(resolve_litellm_model(free_model), None)
        resolve_rates.cache_clear()


# ── Invariants ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("region", [Region.PREFIX, Region.LIVE_ZONE])
def test_a_split_always_conserves_its_tokens(region):
    """Shares must sum to the input exactly, or a long session's total drifts."""
    mixes = [
        _warm_anthropic(),
        _cold_anthropic(),
        CacheMix(),
        CacheMix.from_usage(cache_read_tokens=1, uncached_input_tokens=999_999),
        CacheMix.from_usage(
            cache_read_tokens=7,
            cache_write_tokens=13,
            cache_write_5m_tokens=5,
            cache_write_1h_tokens=3,
            uncached_input_tokens=11,
        ),
    ]
    for mix in mixes:
        for tokens in (1, 999, 40_000, 1_000_000):
            split = split_tokens(tokens, mix, region)
            assert split.total == pytest.approx(tokens)
            assert min(split.read, split.write_5m, split.write_1h, split.uncached) >= 0


def test_a_write_split_exceeding_its_total_cannot_go_negative():
    """A malformed usage frame must not manufacture negative token shares."""
    mix = CacheMix(write_5m=10, write_1h=10, write_total=5).normalized()

    assert mix.write_5m >= 0
    assert mix.write_1h >= 0


def test_zero_and_negative_token_counts_price_at_zero():
    for tokens in (0, -1, -10_000):
        priced = price_savings(tokens, model=SONNET, mix=_warm_anthropic())
        assert priced.usd == 0.0
        assert priced.usd_list == 0.0
