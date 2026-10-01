"""Tests for pure token-bucket rate-limit policy helpers."""

from __future__ import annotations

from headroom.proxy.rate_limit_policy import (
    consume_from_bucket,
    refilled_tokens,
    stale_bucket_keys,
)


def test_refilled_tokens_caps_at_bucket_rate() -> None:
    assert (
        refilled_tokens(
            current_tokens=9,
            last_update=0,
            now=120,
            rate_per_minute=10,
        )
        == 10
    )


def test_refilled_tokens_ignores_negative_elapsed_time() -> None:
    assert (
        refilled_tokens(
            current_tokens=3,
            last_update=10,
            now=5,
            rate_per_minute=60,
        )
        == 3
    )


def test_consume_from_bucket_allows_and_debits_available_tokens() -> None:
    allowed, remaining, wait_seconds = consume_from_bucket(
        available_tokens=5,
        requested_tokens=2,
        rate_per_minute=60,
    )

    assert allowed is True
    assert remaining == 3
    assert wait_seconds == 0


def test_consume_from_bucket_denies_and_reports_wait_time() -> None:
    allowed, remaining, wait_seconds = consume_from_bucket(
        available_tokens=0.5,
        requested_tokens=1,
        rate_per_minute=60,
    )

    assert allowed is False
    assert remaining == 0.5
    assert wait_seconds == 0.5


def test_consume_from_bucket_admits_request_larger_than_bucket_once_full() -> None:
    # 0.39.0 enforced TPM with a 100k default; a 232k-token coding-agent turn
    # was refused forever with a Retry-After that never came true.
    allowed, remaining, wait_seconds = consume_from_bucket(
        available_tokens=100_000,
        requested_tokens=232_000,
        rate_per_minute=100_000,
    )

    assert allowed is True
    assert remaining == -132_000
    assert wait_seconds == 0


def test_consume_from_bucket_oversized_request_waits_only_for_a_full_bucket() -> None:
    allowed, remaining, wait_seconds = consume_from_bucket(
        available_tokens=40_000,
        requested_tokens=232_000,
        rate_per_minute=100_000,
    )

    assert allowed is False
    assert remaining == 40_000
    assert wait_seconds == 36


def test_consume_from_bucket_debt_delays_the_next_request() -> None:
    # Charged in full, so the average rate holds: after the 232k request the
    # next small one waits out the 132k debt plus its own tokens.
    allowed, remaining, wait_seconds = consume_from_bucket(
        available_tokens=-132_000,
        requested_tokens=1_000,
        rate_per_minute=100_000,
    )

    assert allowed is False
    assert remaining == -132_000
    assert wait_seconds == 79.8


def test_stale_bucket_keys_returns_only_old_buckets() -> None:
    assert stale_bucket_keys(
        {"fresh": 950, "edge": 400, "stale": 399},
        now=1000,
        stale_after_seconds=600,
    ) == ["stale"]
