"""Regression tests for the async token-bucket rate limiter."""

from __future__ import annotations

from collections import OrderedDict

import pytest

from headroom.proxy.models import RateLimitState
from headroom.proxy.rate_limiter import MAX_RATE_LIMITER_BUCKETS, TokenBucketRateLimiter


class _NoIterationDict(dict[str, RateLimitState]):
    """Bucket table that fails if a hot-path operation scans every state."""

    def __iter__(self):  # type: ignore[override]
        raise AssertionError("rate-limit hot path scanned all bucket states")

    def items(self):  # type: ignore[override]
        raise AssertionError("rate-limit hot path scanned all bucket states")

    def keys(self):  # type: ignore[override]
        raise AssertionError("rate-limit hot path scanned all bucket states")

    def values(self):  # type: ignore[override]
        raise AssertionError("rate-limit hot path scanned all bucket states")


class _NoIterationOrderedDict(OrderedDict[str, None]):
    """OrderedDict that fails if a hot-path operation scans all identities."""

    def __iter__(self):  # type: ignore[override]
        raise AssertionError("rate-limit hot path scanned all bucket identities")

    def items(self):  # type: ignore[override]
        raise AssertionError("rate-limit hot path scanned all bucket identities")

    def keys(self):  # type: ignore[override]
        raise AssertionError("rate-limit hot path scanned all bucket identities")

    def values(self):  # type: ignore[override]
        raise AssertionError("rate-limit hot path scanned all bucket identities")


@pytest.mark.asyncio
async def test_existing_bucket_check_does_not_scan_all_identities() -> None:
    limiter = TokenBucketRateLimiter(requests_per_minute=1_000_000)
    for index in range(MAX_RATE_LIMITER_BUCKETS):
        await limiter.check_request(f"client-{index}")

    limiter._request_buckets = _NoIterationDict(limiter._request_buckets)
    limiter._bucket_lru = _NoIterationOrderedDict(limiter._bucket_lru)

    allowed, wait_seconds = await limiter.check_request("client-0")

    assert allowed is True
    assert wait_seconds == 0


@pytest.mark.asyncio
async def test_bucket_state_is_hard_capped_and_lru_evicted() -> None:
    limiter = TokenBucketRateLimiter(
        requests_per_minute=1_000_000,
        tokens_per_minute=1_000_000,
    )
    for index in range(MAX_RATE_LIMITER_BUCKETS):
        key = f"client-{index}"
        await limiter.check_request(key)
        await limiter.check_tokens(key, 1)

    # Keep client-0 recent so client-1 is now the least-recently-used identity.
    await limiter.check_request("client-0")
    await limiter.check_request("overflow-client")

    assert len(limiter._bucket_lru) == MAX_RATE_LIMITER_BUCKETS
    assert len(limiter._request_buckets) == MAX_RATE_LIMITER_BUCKETS
    assert len(limiter._token_buckets) == MAX_RATE_LIMITER_BUCKETS - 1
    assert "client-0" in limiter._request_buckets
    assert "client-1" not in limiter._request_buckets
    assert "client-1" not in limiter._token_buckets
    assert "overflow-client" in limiter._request_buckets


@pytest.mark.asyncio
async def test_token_only_checks_share_the_same_hard_cap() -> None:
    limiter = TokenBucketRateLimiter(tokens_per_minute=1_000_000)

    for index in range(MAX_RATE_LIMITER_BUCKETS + 50):
        allowed, wait_seconds = await limiter.check_tokens(f"client-{index}", 1)
        assert allowed is True
        assert wait_seconds == 0

    assert len(limiter._bucket_lru) == MAX_RATE_LIMITER_BUCKETS
    assert len(limiter._token_buckets) == MAX_RATE_LIMITER_BUCKETS
    assert not limiter._request_buckets
    assert (await limiter.stats())["active_keys"] == MAX_RATE_LIMITER_BUCKETS


@pytest.mark.asyncio
async def test_request_and_token_limits_still_debit_independently() -> None:
    limiter = TokenBucketRateLimiter(requests_per_minute=1, tokens_per_minute=5)

    assert (await limiter.check_request("client"))[0] is True
    assert (await limiter.check_request("client"))[0] is False
    assert (await limiter.check_tokens("client", 5))[0] is True
    assert (await limiter.check_tokens("client", 1))[0] is False
