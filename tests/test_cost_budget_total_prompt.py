"""``record_tokens`` must hand litellm the TOTAL prompt, not the uncached slice.

`litellm.cost_per_token` charges::

    (prompt_tokens - cache_read - cache_creation) * input_rate
  + cache_read     * read_rate
  + cache_creation * write_rate

so ``prompt_tokens`` is the whole prompt and litellm removes the cached parts
itself. Passing only the uncached slice drove the input term NEGATIVE as soon as
anything was cached; ``estimate_cost`` returns None on a non-positive total, no
``CostEntry`` was appended, and ``check_budget()`` therefore saw $0. Every
cache-warm request — the normal case in an agent session — booked zero spend, so
``--budget`` could never trip. Measured against real litellm pricing on a 100k
prompt with 80k cached: gpt-5 -$0.065, gpt-4o-mini -$0.003,
claude-sonnet-4-5 -$0.156.

These assert on the arguments handed to ``estimate_cost`` rather than on dollar
values, so they pin the contract that broke without depending on litellm's
pricing tables (or on litellm being installed).
"""

from __future__ import annotations

from headroom.proxy.cost import COST_BASIS_MEASURED, CostTracker


def _tracker_capturing_cost(**kwargs) -> tuple[CostTracker, dict]:
    """A tracker whose ``estimate_cost`` records its kwargs and returns a real cost."""
    tracker = CostTracker(**kwargs)
    seen: dict = {}

    def fake_estimate_cost(**kw):
        seen.update(kw)
        return 0.5  # non-None, so a CostEntry is appended

    tracker.estimate_cost = fake_estimate_cost  # type: ignore[method-assign]
    return tracker, seen


def test_disjoint_buckets_send_the_summed_total_as_prompt_tokens() -> None:
    """Anthropic reports uncached / read / creation as three disjoint buckets."""
    tracker, seen = _tracker_capturing_cost()

    tracker.record_tokens(
        "claude-sonnet-4-5",
        tokens_saved=0,
        tokens_sent=1_000,
        cache_read_tokens=48_000,
        cache_write_tokens=1_500,
        uncached_tokens=900,
    )

    assert seen["input_tokens"] == 900 + 48_000 + 1_500
    # The write premium still applies — those tokens really were written.
    assert seen["cache_write_tokens"] == 1_500
    assert seen["cache_read_tokens"] == 48_000


def test_inferred_write_is_excluded_from_the_total_and_the_premium() -> None:
    """OpenAI exposes no write counter, so the write value IS the uncached tokens.

    Counting it again would double the prompt, and charging it at a write premium
    would invent a cost OpenAI does not have.
    """
    tracker, seen = _tracker_capturing_cost()

    tracker.record_tokens(
        "gpt-5",
        tokens_saved=0,
        tokens_sent=1_000,
        cache_read_tokens=80_000,
        cache_write_tokens=20_000,  # inferred: identical to uncached_tokens
        uncached_tokens=20_000,
        cache_inferred=True,
    )

    assert seen["input_tokens"] == 20_000 + 80_000, "inferred write must not be added"
    assert seen["cache_write_tokens"] == 0, "inferred write must not be charged a premium"


def test_a_cache_warm_request_actually_books_spend() -> None:
    """The regression itself: the budget must see this request."""
    tracker, _ = _tracker_capturing_cost(budget_limit_usd=100.0)

    tracker.record_tokens(
        "claude-sonnet-4-5",
        tokens_saved=0,
        tokens_sent=1_000,
        cache_read_tokens=48_000,
        cache_write_tokens=1_500,
        uncached_tokens=900,
    )

    assert len(tracker._costs) == 1, "cache-warm request booked no spend — budget is blind"
    assert tracker._costs[0].basis == COST_BASIS_MEASURED


def test_no_usage_breakdown_still_falls_back_to_tokens_sent() -> None:
    """Pre-existing estimated-basis fallback must be untouched by this change."""
    tracker, seen = _tracker_capturing_cost()

    tracker.record_tokens("gpt-4o", tokens_saved=0, tokens_sent=4_242)

    assert seen["input_tokens"] == 4_242
    assert len(tracker._costs) == 1
    assert tracker._costs[0].basis != COST_BASIS_MEASURED


def test_cache_inferred_defaults_false_so_reporting_providers_are_unchanged() -> None:
    """Callers that never pass the flag keep the disjoint-bucket arithmetic."""
    tracker, seen = _tracker_capturing_cost()

    tracker.record_tokens(
        "claude-sonnet-4-5",
        tokens_saved=0,
        tokens_sent=1_000,
        cache_read_tokens=10,
        cache_write_tokens=20,
        uncached_tokens=30,
    )

    assert seen["input_tokens"] == 60
    assert seen["cache_write_tokens"] == 20
