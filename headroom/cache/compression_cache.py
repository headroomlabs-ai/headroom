"""Content-addressed compression cache with LRU eviction.

Used in "token headroom mode" to avoid re-compressing messages across turns.
Maps original content hashes to their compressed versions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    """Internal cache entry storing compressed text and metadata."""

    compressed: str
    tokens_saved: int


def _is_tool_result_message(msg: dict) -> bool:
    """Check if a message is a tool result in Anthropic or OpenAI format."""
    # OpenAI format: role="tool"
    if msg.get("role") == "tool":
        return True
    # Anthropic format: role="user" with content list containing tool_result blocks
    content = msg.get("content")
    if isinstance(content, list):
        return any(
            isinstance(block, dict) and block.get("type") == "tool_result" for block in content
        )
    return False


def _extract_tool_result_content(msg: dict) -> str | None:
    """Extract text content from a tool result message (both formats)."""
    # OpenAI format
    if msg.get("role") == "tool":
        content = msg.get("content")
        return content if isinstance(content, str) else None
    # Anthropic format
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                inner = block.get("content")
                if isinstance(inner, str):
                    return inner
    return None


def _swap_tool_result_content(msg: dict, new_content: str) -> dict:
    """Deep copy msg and replace tool result content with new_content."""
    new_msg = copy.deepcopy(msg)
    # OpenAI format
    if new_msg.get("role") == "tool":
        new_msg["content"] = new_content
        return new_msg
    # Anthropic format
    content = new_msg.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                block["content"] = new_content
                break
    return new_msg


class CompressionCache:
    """Content-addressed cache mapping content hashes to compressed versions.

    Uses an OrderedDict for O(1) LRU eviction. Entries are evicted oldest-first
    when the cache exceeds max_entries.
    """

    def __init__(self, max_entries: int = 10000) -> None:
        self.max_entries = max_entries
        # Reentrant lock guarding all mutable state below. Required because
        # the proxy is async and dispatches multiple concurrent requests per
        # session (Claude Code background tools, parallel agents, etc.) into
        # `asyncio.to_thread` workers — without this, two concurrent
        # requests for the same `session_id` race on `_cache`,
        # `_stable_hashes`, `_first_seen`, and `_total_tokens_saved`. The
        # observable failures are (a) lost-update on `_total_tokens_saved`,
        # (b) `OrderedDict mutated during iteration` in `apply_cached`, and
        # (c) lost stable-hash records that drop the next-turn cache lookup.
        # `RLock` (not `Lock`) so future code can call locked methods from
        # inside another locked method without self-deadlock.
        self._lock = threading.RLock()
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        # `_stable_hashes` is CONTENT-KEYED, not positional. It records "we
        # have seen this content before and it is known not to compress
        # further." It MUST NOT be used as a positional cache-safety gate
        # — Anthropic's prefix cache is positional (bytes 0..K cached,
        # anything past K fresh), and content-equality with an old message
        # does not imply Anthropic has cached the new byte position. Issue
        # #327 was caused by such a misuse in the Anthropic token-mode
        # walker; the walker has been removed. Legitimate uses today:
        # `compute_frozen_count` (bounded above by the `min` clamp at
        # `proxy/handlers/anthropic.py`) and `update_from_result`'s
        # "unchanged content" tracking.
        self._stable_hashes: set[str] = set()
        self._first_seen: dict[str, float] = {}
        self._hits: int = 0
        self._misses: int = 0
        self._total_tokens_saved: int = 0

    def get_compressed(self, hash: str) -> str | None:
        """Retrieve compressed content by hash, refreshing LRU position on hit."""
        with self._lock:
            entry = self._cache.get(hash)
            if entry is None:
                self._misses += 1
                return None
            self._hits += 1
            self._cache.move_to_end(hash)
            return entry.compressed

    def store_compressed(self, hash: str, compressed: str, tokens_saved: int) -> None:
        """Store a compressed version keyed by content hash.

        If the hash already exists, the entry is overwritten and moved to the
        end (most recently used). When the cache exceeds max_entries, the oldest
        entry is evicted.
        """
        with self._lock:
            if hash in self._cache:
                old_entry = self._cache[hash]
                self._total_tokens_saved -= old_entry.tokens_saved
                del self._cache[hash]

            self._cache[hash] = _CacheEntry(compressed=compressed, tokens_saved=tokens_saved)
            self._total_tokens_saved += tokens_saved

            while len(self._cache) > self.max_entries:
                _, evicted = self._cache.popitem(last=False)
                self._total_tokens_saved -= evicted.tokens_saved

    def mark_stable(self, content_hash: str) -> None:
        """Mark a content hash as stable (unchanged, not compressed).

        Used for tool_results that the content router excluded or skipped.
        These messages appear verbatim every turn, so they are prefix-stable
        even though no compressed version exists in the cache.
        """
        with self._lock:
            self._stable_hashes.add(content_hash)

    def mark_stable_from_messages(self, messages: list[dict], up_to: int) -> None:
        """Mark all tool_result hashes in messages[:up_to] as stable."""
        with self._lock:
            for msg in messages[:up_to]:
                if _is_tool_result_message(msg):
                    content = _extract_tool_result_content(msg)
                    if content is not None:
                        self._stable_hashes.add(self.content_hash(content))

    def should_defer_compression(
        self,
        content_hash: str,
        ttl_seconds: float = 300.0,
        batch_window: float = 30.0,
    ) -> bool:
        """Whether to defer compressing this content to avoid mid-TTL busts.

        Returns True if we have evidence this content has been re-sent
        within the cache TTL window — recompressing it now would bust an
        existing prefix-cache entry without TTL-amortizing the bust over
        future turns. Returns False otherwise:

        - **First sight** of the content. Compress now: there is no
          prefix-cache entry to preserve yet (this byte range was not in
          a prior request), so compression carries no bust cost. Issue
          #327: a previous version returned True here, which marked the
          freshest tool_result on every turn as "stable" and effectively
          disabled compression for typical Claude Code workloads where
          each tool_result is unique-per-turn.
        - **Near the TTL boundary**: compress now and amortize the bust
          across future turns (batched recompression).
        """
        with self._lock:
            now = time.time()
            first_seen = self._first_seen.get(content_hash)
            if first_seen is None:
                self._first_seen[content_hash] = now
                return False  # First time — compress now (no cache entry to preserve)
            age = now - first_seen
            if age >= ttl_seconds - batch_window:
                return False  # Near TTL boundary — compress now (batch window)
            return True  # Seen recently within TTL — defer to preserve cache

    def get_stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "stable_hashes": len(self._stable_hashes),
                "hits": self._hits,
                "misses": self._misses,
                "tokens_saved": self._total_tokens_saved,
            }

    @staticmethod
    def content_hash(content: str | list) -> str:
        """Compute a truncated SHA-256 hash for string or list content.

        For list content (Anthropic-format messages with type/text/content fields),
        the list is JSON-serialized with sorted keys for deterministic hashing.
        """
        if isinstance(content, list):
            raw = json.dumps(content, sort_keys=True, ensure_ascii=False)
        else:
            raw = content
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]  # nosec B324

    def compute_frozen_count(self, messages: list[dict]) -> int:
        """Count consecutive stable messages from the start.

        A message is stable if it is a plain user/assistant/system message,
        an assistant message with tool_use blocks, or a tool_result whose
        content hash is already in the cache. The first unstable tool_result
        (cache miss) stops the count.
        """
        with self._lock:
            count = 0
            for msg in messages:
                if _is_tool_result_message(msg):
                    content = _extract_tool_result_content(msg)
                    if content is not None:
                        h = self.content_hash(content)
                        if h not in self._cache and h not in self._stable_hashes:
                            break
                    else:
                        # tool_result with non-string content; treat as unstable
                        break
                # Regular user/assistant/system messages and assistant+tool_use
                # are always stable — fall through.
                count += 1
            return count

    def apply_cached(self, messages: list[dict]) -> list[dict]:
        """Return a new list with cached compressions swapped into tool results.

        Never mutates the input list or any message dict within it.
        Output always has the same length as input.
        """
        # `get_compressed` re-acquires the lock (RLock); single contiguous
        # critical section so concurrent `store_compressed` cannot mutate
        # `_cache` mid-iteration.
        with self._lock:
            result: list[dict] = []
            for msg in messages:
                if _is_tool_result_message(msg):
                    content = _extract_tool_result_content(msg)
                    if content is not None:
                        h = self.content_hash(content)
                        compressed = self.get_compressed(h)
                        if compressed is not None:
                            result.append(_swap_tool_result_content(msg, compressed))
                            continue
                result.append(msg)
            return result

    def update_from_result(self, originals: list[dict], compressed: list[dict]) -> None:
        """Cache new compressions by comparing original and compressed messages.

        Index-aligned: for each position, if both are tool results and the
        content differs, store the mapping original_hash -> compressed_content.
        """
        if len(originals) != len(compressed):
            logger.warning(
                "update_from_result: length mismatch (originals=%d, compressed=%d), skipping",
                len(originals),
                len(compressed),
            )
            return

        # Single critical section: `store_compressed` re-acquires the lock
        # (RLock) safely.
        with self._lock:
            for orig, comp in zip(originals, compressed):
                orig_content = _extract_tool_result_content(orig)
                comp_content = _extract_tool_result_content(comp)
                if orig_content is None or comp_content is None:
                    continue
                if orig_content == comp_content:
                    # Content unchanged — mark as stable for frozen count walk
                    self._stable_hashes.add(self.content_hash(orig_content))
                    continue
                h = self.content_hash(orig_content)
                tokens_saved = len(orig_content) // 4 - len(comp_content) // 4
                self.store_compressed(h, comp_content, tokens_saved=max(tokens_saved, 0))
