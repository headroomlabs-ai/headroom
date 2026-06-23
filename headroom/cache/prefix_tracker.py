"""Prefix Cache Tracker — session-scoped state for cache-aware compression.

Tracks provider prefix cache state between turns so the transform pipeline
can freeze already-cached messages and only compress new content.

Problem: Clients like Claude Code already manage prefix caching (up to 4
cache_control breakpoints, growing-prefix strategy). If Headroom compresses
or modifies messages in the cached prefix, it invalidates the cache —
replacing a 90% read discount (Anthropic) or 50% (OpenAI) with a 25%
write penalty.

Solution: After each API response, record how many tokens the provider
cached. On the next turn, freeze that many messages so the transform
pipeline skips them entirely.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Provider cache economics for cost comparisons
_PROVIDER_READ_DISCOUNT = {
    "anthropic": 0.9,  # 90% discount on reads
    "openai": 0.5,  # 50% discount on reads
    "gemini": 0.9,
    "bedrock": 0.9,
}

_PROVIDER_WRITE_PENALTY = {
    "anthropic": 0.25,  # 25% surcharge on writes
    "openai": 0.0,  # No write penalty
    "gemini": 0.0,
    "bedrock": 0.25,
}


@dataclass
class PrefixFreezeConfig:
    """Configuration for cache-aware prefix freezing."""

    enabled: bool = True
    min_cached_tokens: int = 1024  # Min cached tokens to activate freeze
    session_ttl_seconds: int = 600  # Session tracker cleanup TTL
    force_compress_threshold: float = 0.5  # Bust cache if compression saves > this fraction


@dataclass
class FreezeStats:
    """Statistics from prefix freezing for metrics/dashboard."""

    busts_avoided: int = 0
    tokens_preserved: int = 0
    compression_foregone_tokens: int = 0
    net_benefit_tokens: int = 0  # tokens_preserved - compression_foregone
    frozen_message_count: int = 0
    turn_number: int = 0


class PrefixCacheTracker:
    """Tracks provider prefix cache state across turns in a session.

    Usage:
        tracker = PrefixCacheTracker("anthropic")

        # Before compression (turn 2+):
        frozen = tracker.get_frozen_message_count()
        result = pipeline.apply(messages, model, frozen_message_count=frozen)

        # After API response:
        tracker.update_from_response(
            cache_read_tokens=usage["cache_read_input_tokens"],
            cache_write_tokens=usage["cache_creation_input_tokens"],
            messages=optimized_messages,
            tokenizer=tokenizer,
        )
    """

    def __init__(self, provider: str, config: PrefixFreezeConfig | None = None):
        self.provider = provider
        self.config = config or PrefixFreezeConfig()
        self._cached_token_count: int = 0
        self._cached_message_count: int = 0
        self._turn_number: int = 0
        self._last_activity: float = time.time()
        self._last_original_messages: list[dict[str, Any]] = []
        self._last_forwarded_messages: list[dict[str, Any]] = []

        # Session-scoped ReadMaturationManager (Mechanism B), created
        # lazily by the handler when read maturation is enabled. Rides
        # here so it shares the session's affinity and TTL cleanup.
        self.read_maturation_manager: Any = None

        # Stats
        self._busts_avoided: int = 0
        self._tokens_preserved: int = 0
        self._compression_foregone_tokens: int = 0

    def get_frozen_message_count(self) -> int:
        """How many leading messages to skip compression on the next turn.

        Returns 0 on turn 0 (cold start) or if caching is disabled/below threshold.
        """
        if not self.config.enabled:
            return 0
        if self._turn_number == 0:
            return 0
        if self._cached_token_count < self.config.min_cached_tokens:
            return 0
        return self._cached_message_count

    def update_from_response(
        self,
        cache_read_tokens: int,
        cache_write_tokens: int,
        messages: list[dict[str, Any]],
        message_token_counts: list[int] | None = None,
        original_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Update tracker with cache metrics from the API response.

        Called after every API call. Computes how many messages to freeze
        on the next turn based on the cache_read_tokens reported.

        Args:
            cache_read_tokens: Tokens read from cache (cache hit portion).
            cache_write_tokens: Tokens written to cache (new cache entries).
            messages: The messages that were sent to the API.
            message_token_counts: Pre-computed token counts per message.
                If None, estimates from content length.
        """
        self._last_activity = time.time()
        self._turn_number += 1
        self._last_original_messages = copy.deepcopy(original_messages or messages)
        self._last_forwarded_messages = copy.deepcopy(messages)

        # Compute total cached tokens (read + write = what's in cache now)
        total_cached = cache_read_tokens + cache_write_tokens

        if total_cached == 0:
            self._cached_token_count = 0
            self._cached_message_count = 0
            return

        # Estimate per-message token counts if not provided
        if message_token_counts is None:
            message_token_counts = self._estimate_message_tokens(messages)

        # Walk messages from the start, accumulating tokens until we exceed
        # the cached amount. All messages within the cached prefix are frozen.
        accumulated = 0
        frozen_count = 0
        for i, tok_count in enumerate(message_token_counts):
            accumulated += tok_count
            if accumulated <= total_cached:
                frozen_count = i + 1
            else:
                break

        self._cached_token_count = total_cached
        self._cached_message_count = frozen_count

        logger.debug(
            "PrefixCacheTracker[%s]: turn=%d, cached=%d tokens, "
            "frozen=%d/%d messages (read=%d, write=%d)",
            self.provider,
            self._turn_number,
            total_cached,
            frozen_count,
            len(messages),
            cache_read_tokens,
            cache_write_tokens,
        )

    def get_last_original_messages(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._last_original_messages)

    def get_last_forwarded_messages(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._last_forwarded_messages)

    def record_bust_avoided(self, tokens_preserved: int, compression_foregone: int) -> None:
        """Record when we chose to preserve cache over compressing."""
        self._busts_avoided += 1
        self._tokens_preserved += tokens_preserved
        self._compression_foregone_tokens += compression_foregone

    def should_force_compress(
        self,
        message_index: int,
        message_tokens: int,
        estimated_compressed_tokens: int,
    ) -> bool:
        """Check if compression savings outweigh cache preservation.

        Returns True if we should bust the cache and compress anyway.
        This happens when compression would save a large fraction of tokens
        AND the savings exceed the cache read discount.
        """
        if message_index >= self._cached_message_count:
            return True  # Not in frozen prefix, always compress

        if message_tokens == 0:
            return False

        savings_fraction = (message_tokens - estimated_compressed_tokens) / message_tokens

        # Would compression savings exceed the cache read discount?
        read_discount = _PROVIDER_READ_DISCOUNT.get(self.provider, 0.5)
        return savings_fraction > read_discount

    @property
    def is_expired(self) -> bool:
        """Check if this tracker has been idle beyond TTL."""
        return (time.time() - self._last_activity) > self.config.session_ttl_seconds

    def seconds_since_activity(self) -> float:
        """Wall-clock seconds since this tracker last saw activity.

        #856 P3b feeds this to the net-cost gate as an idle signal: as it
        approaches the provider's prompt-cache TTL (~300s for Anthropic),
        P_alive decays toward 0 and deep edits near cache lapse become free.
        Distinct from :attr:`is_expired`, which uses the much longer
        session-tracker *cleanup* TTL (``session_ttl_seconds``), not the cache
        TTL.

        Wiring caveat: ``SessionTrackerStore.get_or_create`` refreshes
        ``_last_activity`` on access, so a caller that wants the idle gap
        since the *previous turn's response* must read this before fetching
        the tracker for the current request (or the store must capture it at
        fetch time). ``update_from_response`` is the per-turn activity stamp.
        """
        return max(0.0, time.time() - self._last_activity)

    @property
    def stats(self) -> FreezeStats:
        """Return stats for dashboard/metrics."""
        return FreezeStats(
            busts_avoided=self._busts_avoided,
            tokens_preserved=self._tokens_preserved,
            compression_foregone_tokens=self._compression_foregone_tokens,
            net_benefit_tokens=self._tokens_preserved - self._compression_foregone_tokens,
            frozen_message_count=self._cached_message_count,
            turn_number=self._turn_number,
        )

    @staticmethod
    def _estimate_message_tokens(messages: list[dict[str, Any]]) -> list[int]:
        """Rough token count per message (chars / 3.5).

        Counts text, tool_result content, and tool_use input fields
        for accurate Anthropic-format estimation.
        """
        counts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                chars = len(content)
            elif isinstance(content, list):
                chars = 0
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type", "")
                    if block_type == "text":
                        chars += len(block.get("text", ""))
                    elif block_type == "tool_result":
                        inner = block.get("content", "")
                        if isinstance(inner, str):
                            chars += len(inner)
                        elif isinstance(inner, list):
                            chars += sum(
                                len(b.get("text", "")) for b in inner if isinstance(b, dict)
                            )
                    elif block_type == "tool_use":
                        inp = block.get("input")
                        if isinstance(inp, str):
                            chars += len(inp)
                        elif isinstance(inp, dict):
                            chars += len(json.dumps(inp, separators=(",", ":")))
                    else:
                        text = block.get("text", "")
                        if text:
                            chars += len(text)
            else:
                chars = 0
            # Add overhead for role, block structure, etc.
            chars += 20
            counts.append(max(1, int(chars / 3.5)))
        return counts


class SessionTrackerStore:
    """Manages PrefixCacheTracker instances across sessions.

    Keyed by session ID (from x-headroom-session-id header or computed hash).
    Automatically cleans up expired sessions.
    """

    def __init__(self, default_config: PrefixFreezeConfig | None = None):
        self._trackers: dict[str, PrefixCacheTracker] = {}
        self._default_config = default_config or PrefixFreezeConfig()
        self._last_cleanup: float = time.time()
        self._cleanup_interval: float = 60.0  # Cleanup every 60s

    def get_or_create(self, session_id: str, provider: str) -> PrefixCacheTracker:
        """Get existing tracker or create a new one for this session."""
        self._maybe_cleanup()

        if session_id in self._trackers:
            tracker = self._trackers[session_id]
            tracker._last_activity = time.time()
            return tracker

        tracker = PrefixCacheTracker(provider, self._default_config)
        self._trackers[session_id] = tracker
        return tracker

    def compute_session_id(
        self,
        request: Any,
        model: str,
        messages: list[dict[str, Any]],
    ) -> str:
        """Compute a session ID from the request.

        Priority:
        1. x-headroom-session-id header (explicit)
        2. Hash of (model + system prompt) — stable per conversation
        """
        # Check for explicit session header
        if hasattr(request, "headers"):
            session_header = request.headers.get("x-headroom-session-id")
            if session_header:
                return str(session_header)

        # Fall back to hashing model + system prompt
        system_content = ""
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_content = content[:500]  # First 500 chars is enough
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            system_content = block.get("text", "")[:500]
                            break
                break

        key = f"{model}:{system_content}"
        return hashlib.md5(key.encode()).hexdigest()[:16]  # nosec B324

    def _maybe_cleanup(self) -> None:
        """Remove expired trackers periodically."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        expired = [sid for sid, tracker in self._trackers.items() if tracker.is_expired]
        for sid in expired:
            del self._trackers[sid]

        if expired:
            logger.debug("SessionTrackerStore: cleaned up %d expired sessions", len(expired))

        self._last_cleanup = now

    @property
    def active_sessions(self) -> int:
        """Number of active session trackers."""
        return len(self._trackers)
