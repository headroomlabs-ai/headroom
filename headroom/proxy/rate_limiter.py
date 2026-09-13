"""Token bucket rate limiter for the Headroom proxy.

Rate limits requests and token usage per API key or IP address.

Extracted from server.py for maintainability.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict

from headroom.proxy.models import RateLimitState
from headroom.proxy.rate_limit_policy import consume_from_bucket, refilled_tokens

# Maximum rate limiter buckets (prevents DoS via spoofed API keys)
MAX_RATE_LIMITER_BUCKETS = 1000


class TokenBucketRateLimiter:
    """Token bucket rate limiter for requests and tokens."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 100000,
    ):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute

        # Per-key buckets (key = API key or IP). A shared LRU keeps request and
        # token state on the same bounded lifecycle without scanning all keys.
        self._request_buckets: dict[str, RateLimitState] = {}
        self._token_buckets: dict[str, RateLimitState] = {}
        self._bucket_lru: OrderedDict[str, None] = OrderedDict()
        self._lock = asyncio.Lock()

    def _touch_bucket(self, key: str) -> None:
        """Mark a bucket active, evicting the least-recently-used key at capacity."""
        if key in self._bucket_lru:
            self._bucket_lru.move_to_end(key)
            return

        if len(self._bucket_lru) >= MAX_RATE_LIMITER_BUCKETS:
            evicted_key, _ = self._bucket_lru.popitem(last=False)
            self._request_buckets.pop(evicted_key, None)
            self._token_buckets.pop(evicted_key, None)
        self._bucket_lru[key] = None

    def _request_bucket(self, key: str) -> RateLimitState:
        state = self._request_buckets.get(key)
        if state is None:
            state = RateLimitState(tokens=self.requests_per_minute, last_update=time.time())
            self._request_buckets[key] = state
        return state

    def _token_bucket(self, key: str) -> RateLimitState:
        state = self._token_buckets.get(key)
        if state is None:
            state = RateLimitState(tokens=self.tokens_per_minute, last_update=time.time())
            self._token_buckets[key] = state
        return state

    def _refill(self, state: RateLimitState, rate_per_minute: float) -> float:
        """Refill bucket based on elapsed time."""
        now = time.time()
        state.tokens = refilled_tokens(
            current_tokens=state.tokens,
            last_update=state.last_update,
            now=now,
            rate_per_minute=rate_per_minute,
        )
        state.last_update = now
        return state.tokens

    async def check_request(self, key: str = "default") -> tuple[bool, float]:
        """Check if request is allowed. Returns (allowed, wait_seconds)."""
        async with self._lock:
            self._touch_bucket(key)
            state = self._request_bucket(key)
            available = self._refill(state, self.requests_per_minute)

            allowed, state.tokens, wait_seconds = consume_from_bucket(
                available_tokens=available,
                requested_tokens=1,
                rate_per_minute=self.requests_per_minute,
            )
            return allowed, wait_seconds

    async def check_tokens(self, key: str, token_count: int) -> tuple[bool, float]:
        """Check if token usage is allowed."""
        async with self._lock:
            self._touch_bucket(key)
            state = self._token_bucket(key)
            available = self._refill(state, self.tokens_per_minute)

            allowed, state.tokens, wait_seconds = consume_from_bucket(
                available_tokens=available,
                requested_tokens=token_count,
                rate_per_minute=self.tokens_per_minute,
            )
            return allowed, wait_seconds

    async def stats(self) -> dict:
        """Get rate limiter statistics."""
        async with self._lock:
            return {
                "requests_per_minute": self.requests_per_minute,
                "tokens_per_minute": self.tokens_per_minute,
                "active_keys": len(self._bucket_lru),
            }
