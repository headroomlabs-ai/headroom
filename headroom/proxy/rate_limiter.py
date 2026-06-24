"""Token bucket rate limiter for the Headroom proxy.

Rate limits requests and token usage per API key or IP address.

Extracted from server.py for maintainability.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict

from headroom.proxy.models import RateLimitState

logger = logging.getLogger("headroom.proxy")

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

        # Per-key buckets (key = API key or IP)
        self._request_buckets: dict[str, RateLimitState] = defaultdict(
            lambda: RateLimitState(tokens=requests_per_minute, last_update=time.time())
        )
        self._token_buckets: dict[str, RateLimitState] = defaultdict(
            lambda: RateLimitState(tokens=tokens_per_minute, last_update=time.time())
        )
        self._lock = asyncio.Lock()

    async def _cleanup_stale_buckets(self) -> None:
        """Remove buckets that haven't been used in the last 10 minutes."""
        now = time.time()
        stale_threshold = now - 600  # 10 minutes
        stale_keys = [
            k for k, v in self._request_buckets.items() if v.last_update < stale_threshold
        ]
        for k in stale_keys:
            del self._request_buckets[k]
            self._token_buckets.pop(k, None)
        if stale_keys:
            logger.debug(f"Cleaned up {len(stale_keys)} stale rate limiter buckets")

    def _refill(self, state: RateLimitState, rate_per_minute: float) -> float:
        """Refill bucket based on elapsed time."""
        now = time.time()
        elapsed = now - state.last_update
        refill = elapsed * (rate_per_minute / 60.0)
        state.tokens = min(rate_per_minute, state.tokens + refill)
        state.last_update = now
        return state.tokens

    async def check_request(self, key: str = "default") -> tuple[bool, float]:
        """Check if request is allowed. Returns (allowed, wait_seconds)."""
        async with self._lock:
            # Prevent unbounded bucket growth from spoofed keys
            if len(self._request_buckets) > MAX_RATE_LIMITER_BUCKETS:
                await self._cleanup_stale_buckets()
            state = self._request_buckets[key]
            available = self._refill(state, self.requests_per_minute)

            if available >= 1:
                state.tokens -= 1
                return True, 0

            wait_seconds = (1 - available) * (60.0 / self.requests_per_minute)
            return False, wait_seconds

    async def check_tokens(self, key: str, token_count: int) -> tuple[bool, float]:
        """Check if token usage is allowed."""
        async with self._lock:
            state = self._token_buckets[key]
            available = self._refill(state, self.tokens_per_minute)

            if available >= token_count:
                state.tokens -= token_count
                return True, 0

            wait_seconds = (token_count - available) * (60.0 / self.tokens_per_minute)
            return False, wait_seconds

    async def stats(self) -> dict:
        """Get rate limiter statistics."""
        async with self._lock:
            return {
                "requests_per_minute": self.requests_per_minute,
                "tokens_per_minute": self.tokens_per_minute,
                "active_keys": len(self._request_buckets),
            }
