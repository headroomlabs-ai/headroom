"""SharedContext — compressed inter-agent context sharing.

When agents hand off to each other, context gets replayed in full.
SharedContext compresses what moves between agents, using Headroom's
existing CCR (Compress-Cache-Retrieve) architecture.

Usage:

    from headroom import SharedContext

    ctx = SharedContext()

    # Agent A stores large output
    ctx.put("research", big_research_output)

    # Agent B gets compressed version (~80% smaller)
    summary = ctx.get("research")

    # Agent B needs full details on something specific
    full = ctx.get("research", full=True)

Works with any agent framework. The compression pipeline is the same
one used by the proxy and MCP server.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ContextEntry:
    """A stored context entry with original and compressed versions."""

    key: str
    original: str
    compressed: str
    original_tokens: int
    compressed_tokens: int
    agent: str | None
    timestamp: float
    transforms: list[str] = field(default_factory=list)

    @property
    def savings_percent(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return round((1 - self.compressed_tokens / self.original_tokens) * 100, 1)


@dataclass
class SharedContextStats:
    """Aggregated stats for the shared context."""

    entries: int
    total_original_tokens: int
    total_compressed_tokens: int
    total_tokens_saved: int
    savings_percent: float


class SharedContext:
    """Compressed shared context for multi-agent workflows.

    Agents put content in, other agents get compressed versions out.
    Originals are stored for on-demand full retrieval.

    Args:
        model: Model name for token counting (default: claude-sonnet-4-5).
        ttl: Time-to-live in seconds (default: 3600 = 1 hour).
        max_entries: Maximum stored entries (default: 100).
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        ttl: int = 3600,
        max_entries: int = 100,
    ) -> None:
        self._model = model
        self._ttl = ttl
        self._max_entries = max_entries
        self._entries: dict[str, ContextEntry] = {}
        self._lock = threading.Lock()

    def put(
        self,
        key: str,
        content: str,
        *,
        agent: str | None = None,
    ) -> ContextEntry:
        """Store content under a key, compressing automatically.

        Args:
            key: Name for this context (e.g., "research_findings").
            content: The content to store and compress.
            agent: Optional agent identifier for tracking.

        Returns:
            ContextEntry with compression stats.
        """
        from headroom.compress import compress

        messages = [{"role": "tool", "content": content}]
        result = compress(messages, model=self._model)

        compressed = result.messages[0].get("content", content)
        if not isinstance(compressed, str):
            import json

            compressed = json.dumps(compressed)

        entry = ContextEntry(
            key=key,
            original=content,
            compressed=compressed,
            original_tokens=result.tokens_before,
            compressed_tokens=result.tokens_after,
            agent=agent,
            timestamp=time.time(),
            transforms=result.transforms_applied,
        )

        with self._lock:
            self._evict_if_needed()
            self._entries[key] = entry

        logger.debug(
            "SharedContext.put(%s): %d → %d tokens (%.1f%% saved)",
            key,
            entry.original_tokens,
            entry.compressed_tokens,
            entry.savings_percent,
        )

        return entry

    def get(
        self,
        key: str,
        *,
        full: bool = False,
    ) -> str | None:
        """Get content by key.

        Args:
            key: The key to retrieve.
            full: If True, return the original uncompressed content.
                  If False (default), return the compressed version.

        Returns:
            Content string, or None if key not found or expired.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if time.time() - entry.timestamp > self._ttl:
                del self._entries[key]
                return None

        return entry.original if full else entry.compressed

    def get_entry(self, key: str) -> ContextEntry | None:
        """Get the full ContextEntry with metadata."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if time.time() - entry.timestamp > self._ttl:
                del self._entries[key]
                return None
            return entry

    def keys(self) -> list[str]:
        """List all non-expired keys."""
        now = time.time()
        with self._lock:
            return [k for k, e in self._entries.items() if now - e.timestamp <= self._ttl]

    def stats(self) -> SharedContextStats:
        """Get aggregated stats."""
        now = time.time()
        with self._lock:
            active = [e for e in self._entries.values() if now - e.timestamp <= self._ttl]
        total_orig = sum(e.original_tokens for e in active)
        total_comp = sum(e.compressed_tokens for e in active)
        total_saved = total_orig - total_comp
        pct = round(total_saved / total_orig * 100, 1) if total_orig > 0 else 0.0
        return SharedContextStats(
            entries=len(active),
            total_original_tokens=total_orig,
            total_compressed_tokens=total_comp,
            total_tokens_saved=total_saved,
            savings_percent=pct,
        )

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._entries.clear()

    def _evict_if_needed(self) -> None:
        """Evict expired and oldest entries if at capacity. Lock must be held."""
        now = time.time()
        expired = [k for k, e in self._entries.items() if now - e.timestamp > self._ttl]
        for k in expired:
            del self._entries[k]

        while len(self._entries) >= self._max_entries:
            oldest_key = min(self._entries, key=lambda k: self._entries[k].timestamp)
            del self._entries[oldest_key]
