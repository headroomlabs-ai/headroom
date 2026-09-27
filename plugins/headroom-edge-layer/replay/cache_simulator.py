"""Cache-prefix simulator for the offline replay harness.

Approximates the provider's prompt cache the way the test protocol's Phase 1
describes: "the tokens shared with the previous forwarded request (longest
common prefix, if the time gap is inside the TTL) are cache reads; the rest
are cache writes." Operates on the *actual* bodies the fake upstream
received — i.e., after whatever the proxy/edge layer did to them — not the
original session bodies, since what matters for cache economics is what
went out on the wire.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from headroom_edge_layer.cache_safety import estimate_tokens

DEFAULT_TTL_SECONDS = 300  # Anthropic's short-lived cache tier


def longest_common_message_prefix(a: list[Any], b: list[Any]) -> int:
    count = 0
    for x, y in zip(a, b):
        if x == y:
            count += 1
        else:
            break
    return count


@dataclass
class CacheSimResult:
    cache_read_tokens: int
    cache_write_tokens: int

    @property
    def cache_read_share(self) -> float:
        total = self.cache_read_tokens + self.cache_write_tokens
        return self.cache_read_tokens / total if total else 0.0


def simulate_turn_cache(
    previous_messages: list[dict[str, Any]] | None,
    current_messages: list[dict[str, Any]],
    *,
    previous_timestamp: float | None,
    current_timestamp: float,
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
) -> CacheSimResult:
    if previous_messages is None or previous_timestamp is None:
        return CacheSimResult(cache_read_tokens=0, cache_write_tokens=_tokens_of(current_messages))
    if (current_timestamp - previous_timestamp) > ttl_seconds:
        return CacheSimResult(cache_read_tokens=0, cache_write_tokens=_tokens_of(current_messages))

    prefix_len = longest_common_message_prefix(previous_messages, current_messages)
    shared = current_messages[:prefix_len]
    rest = current_messages[prefix_len:]
    return CacheSimResult(
        cache_read_tokens=_tokens_of(shared) if shared else 0,
        cache_write_tokens=_tokens_of(rest),
    )


def _tokens_of(messages: list[Any]) -> int:
    if not messages:
        return 0
    return estimate_tokens(json.dumps(messages, sort_keys=True))


def simulate_session_cache(
    request_bodies: list[dict[str, Any]],
    timestamps: list[float],
    *,
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
) -> list[CacheSimResult]:
    """One `CacheSimResult` per request body, in call order."""
    results: list[CacheSimResult] = []
    previous_messages: list[dict[str, Any]] | None = None
    previous_timestamp: float | None = None
    for body, timestamp in zip(request_bodies, timestamps):
        messages = body.get("messages", [])
        result = simulate_turn_cache(
            previous_messages,
            messages,
            previous_timestamp=previous_timestamp,
            current_timestamp=timestamp,
            ttl_seconds=ttl_seconds,
        )
        results.append(result)
        previous_messages = messages
        previous_timestamp = timestamp
    return results
