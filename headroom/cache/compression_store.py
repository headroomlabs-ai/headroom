"""Compression Store for CCR (Compress-Cache-Retrieve) architecture.

This module implements reversible compression: when SmartCrusher compresses
tool outputs, the original data is cached here for on-demand retrieval.

Key insight from research: REVERSIBLE compression beats irreversible compression.
If the LLM needs data that was compressed away, it can retrieve it instantly.

Features:
- Thread-safe in-memory storage with TTL expiration
- BM25-based search within cached content
- Retrieval event tracking for feedback loop
- Automatic eviction when capacity is reached

Usage:
    store = get_compression_store()

    # Store compressed content
    hash_key = store.store(
        original=original_json,
        compressed=compressed_json,
        original_tokens=1000,
        compressed_tokens=100,
        tool_name="search_api",
    )

    # Retrieve later
    entry = store.retrieve(hash_key)

    # Or search within
    results = store.search(hash_key, "user query")
"""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
import os
import re
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from ..relevance.bm25 import BM25Scorer

if TYPE_CHECKING:
    from ..memory.tracker import ComponentStats
    from .backends import CompressionStoreBackend

logger = logging.getLogger(__name__)

DEFAULT_CCR_TTL_SECONDS = 1800  # session-scale; override via HEADROOM_CCR_TTL_SECONDS
CCR_TTL_SECONDS_ENV = "HEADROOM_CCR_TTL_SECONDS"

_RETRIEVAL_LOG_PREVIEW_CHARS = 4096
_SECRET_KEY_VALUE_RE = re.compile(
    r"(?i)\b([A-Z0-9_-]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH)[A-Z0-9_-]*)"
    r"(\s*[:=]\s*)([\"']?)([^\"'\s,}]+)"
)
_AUTH_VALUE_RE = re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{12,}")
_API_KEY_VALUE_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")


def _get_env_default_ttl_seconds() -> int:
    raw_value = os.environ.get(CCR_TTL_SECONDS_ENV)
    if raw_value is None or not raw_value.strip():
        return DEFAULT_CCR_TTL_SECONDS

    try:
        ttl_seconds = int(raw_value)
    except ValueError:
        logger.warning(
            "%s must be a positive integer number of seconds, got %r; using %s",
            CCR_TTL_SECONDS_ENV,
            raw_value,
            DEFAULT_CCR_TTL_SECONDS,
        )
        return DEFAULT_CCR_TTL_SECONDS

    if ttl_seconds <= 0:
        logger.warning(
            "%s must be greater than 0, got %s; using %s",
            CCR_TTL_SECONDS_ENV,
            ttl_seconds,
            DEFAULT_CCR_TTL_SECONDS,
        )
        return DEFAULT_CCR_TTL_SECONDS

    return ttl_seconds


def format_retrieval_miss_detail(status: dict[str, Any]) -> str:
    """Return an operator-facing miss reason for CCR retrieval failures."""
    default_ttl = status.get("default_ttl_seconds", DEFAULT_CCR_TTL_SECONDS)
    ttl_seconds = status.get("ttl_seconds", default_ttl)

    if status.get("status") == "expired":
        age_seconds = status.get("age_seconds")
        if isinstance(age_seconds, (int, float)):
            return f"Entry expired (CCR TTL: {ttl_seconds} seconds; age: {age_seconds:.0f} seconds)"
        return f"Entry expired (CCR TTL: {ttl_seconds} seconds)"

    return f"Entry not found (CCR TTL: {default_ttl} seconds)"


def _redact_retrieval_log_payload(payload: str) -> str:
    redacted = _SECRET_KEY_VALUE_RE.sub(r"\1\2\3[REDACTED]", payload)
    redacted = _AUTH_VALUE_RE.sub(r"\1 [REDACTED]", redacted)
    return _API_KEY_VALUE_RE.sub("sk-[REDACTED]", redacted)


def _payload_for_retrieval_log(payload: str) -> dict[str, Any]:
    redacted = _redact_retrieval_log_payload(payload)
    preview = redacted[:_RETRIEVAL_LOG_PREVIEW_CHARS]
    truncated = len(redacted) > len(preview)
    return {
        "payload_chars": len(payload),
        "payload_preview_chars": len(preview),
        "payload_truncated": truncated,
        "payload_preview": preview,
    }


# Single source of truth for the retrieval-miss message. Actionable by
# design: the model still has the marker in context (Read markers carry
# the file path), so tell it how to recover instead of just reporting
# the miss.
CCR_MISS_MESSAGE = (
    "Entry not found or expired. To recover: if the compression marker "
    "references a file Read, re-read that file (the path is in the "
    "marker; disk is the source of truth). If it was command output, "
    "re-run the command. Entries expire after the store TTL "
    "(default 30 minutes; configurable via HEADROOM_CCR_TTL_SECONDS)."
)


@dataclass
class CompressionEntry:
    """A cached compression entry with metadata for retrieval and feedback."""

    hash: str
    original_content: str
    compressed_content: str
    original_tokens: int
    compressed_tokens: int
    original_item_count: int
    compressed_item_count: int
    tool_name: str | None
    tool_call_id: str | None
    query_context: str | None
    created_at: float
    ttl: int = DEFAULT_CCR_TTL_SECONDS

    # TOIN integration: Store the tool signature hash for retrieval correlation
    # This MUST match the hash used by SmartCrusher when recording compression
    tool_signature_hash: str | None = None
    compression_strategy: str | None = None  # Strategy used for compression

    # Feedback tracking
    retrieval_count: int = 0
    search_queries: list[str] = field(default_factory=list)
    last_accessed: float | None = None

    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        return time.time() - self.created_at > self.ttl

    def record_access(self, query: str | None = None) -> None:
        """Record an access to this entry for feedback tracking."""
        self.retrieval_count += 1
        self.last_accessed = time.time()
        if query and query not in self.search_queries:
            self.search_queries.append(query)
            # Keep only last 10 queries
            if len(self.search_queries) > 10:
                self.search_queries = self.search_queries[-10:]


@dataclass
class RetrievalEvent:
    """Event logged when content is retrieved from cache."""

    hash: str
    query: str | None
    items_retrieved: int
    total_items: int
    tool_name: str | None
    timestamp: float
    retrieval_type: str  # "full" or "search"
    tool_signature_hash: str | None = None  # For TOIN correlation


class CompressionStore:
    """Thread-safe store for compressed content with retrieval support.

    This is the core of the CCR architecture. When SmartCrusher compresses
    an array, the original content is stored here. If the LLM needs more
    data, it can retrieve from this cache instantly.

    Design principles:
    - Zero external dependencies (pure Python)
    - Thread-safe for concurrent access
    - TTL-based expiration (default 300 seconds, env-configurable)
    - LRU-style eviction when capacity is reached
    - Built-in BM25 search for filtering
    """

    def __init__(
        self,
        max_entries: int = 1000,
        default_ttl: int = DEFAULT_CCR_TTL_SECONDS,
        enable_feedback: bool = True,
        backend: CompressionStoreBackend | None = None,
    ):
        """Initialize the compression store.

        Args:
            max_entries: Maximum number of entries to store.
            default_ttl: Default TTL in seconds (default 30 minutes — session scale).
            enable_feedback: Whether to track retrieval events.
            backend: Storage backend to use. Defaults to InMemoryBackend
                     when constructed directly; `get_compression_store()`
                     defaults to SQLiteBackend for restart/multi-worker
                     safety. Custom backends can be passed for
                     persistence (MongoDB, Redis).
        """
        # Import here to avoid circular imports
        from .backends import InMemoryBackend

        self._backend: CompressionStoreBackend = backend or InMemoryBackend()
        self._lock = threading.Lock()
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._enable_feedback = enable_feedback

        # Feedback tracking
        self._retrieval_events: list[RetrievalEvent] = []
        self._max_events = 1000  # Keep last 1000 events
        self._pending_feedback_events: list[RetrievalEvent] = []

        # MEDIUM FIX #16: Use a min-heap for O(log n) eviction instead of O(n)
        # Heap entries are (created_at, hash_key) tuples
        self._eviction_heap: list[tuple[float, str]] = []
        # CRITICAL FIX: Track stale entries count to know when heap cleanup is needed
        self._stale_heap_entries = 0
        # Threshold for triggering heap rebuild (when 50% are stale)
        self._heap_rebuild_threshold = 0.5

        # BM25 scorer for search
        self._scorer = BM25Scorer()

    @property
    def default_ttl_seconds(self) -> int:
        """Default TTL applied to new entries when callers do not override it."""
        return self._default_ttl

    def store(
        self,
        original: str,
        compressed: str,
        *,
        original_tokens: int = 0,
        compressed_tokens: int = 0,
        original_item_count: int = 0,
        compressed_item_count: int = 0,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        query_context: str | None = None,
        tool_signature_hash: str | None = None,
        compression_strategy: str | None = None,
        ttl: int | None = None,
        explicit_hash: str | None = None,
    ) -> str:
        """Store compressed content and return hash for retrieval.

        Args:
            original: Original JSON content before compression.
            compressed: Compressed JSON content.
            original_tokens: Token count of original content.
            compressed_tokens: Token count of compressed content.
            original_item_count: Number of items in original array.
            compressed_item_count: Number of items after compression.
            tool_name: Name of the tool that produced this output.
            tool_call_id: ID of the tool call.
            query_context: User query context for relevance matching.
            tool_signature_hash: Hash from ToolSignature for TOIN correlation.
            compression_strategy: Strategy used for compression.
            ttl: Custom TTL in seconds (uses default if not specified).
            explicit_hash: Use this exact hex hash as the storage key
                instead of computing SHA-256(original)[:24]. Required when
                the marker that points at this entry was emitted by a
                producer with its own hash function (e.g. SmartCrusher's
                Rust row-drop path uses SHA-256[:12]). If not a hex
                string, raises ``ValueError``. The marker hash and the
                store key MUST match — otherwise ``/v1/retrieve/{hash}``
                returns 404 even though the data is present.

        Returns:
            Hash key for retrieving this content.
        """
        # Generate hash from original content. Default: SHA-256[:24] of the
        # original. When the caller provides `explicit_hash`, use it
        # verbatim — required when the hash that ends up in the prompt
        # marker is produced by another component (e.g. the Rust
        # SmartCrusher row-drop path emits SHA-256[:12], which the
        # Python store has to mirror so /v1/retrieve resolves it).
        # 24 chars (96 bits) was chosen for collision resistance under the
        # birthday bound: 50% collision probability at ~280 trillion entries
        # (2^48), versus ~4 billion (2^32) for the previous 16-char default.
        if explicit_hash is not None:
            # Validate as hex. Bail loudly per `feedback_no_silent_fallbacks`
            # — silently falling back to the default hash when the caller
            # asked for a specific key would defeat the marker/store
            # consistency we're trying to preserve.
            if not explicit_hash or not all(c in "0123456789abcdefABCDEF" for c in explicit_hash):
                raise ValueError(
                    f"explicit_hash must be a non-empty hex string, got {explicit_hash!r}"
                )
            hash_key = explicit_hash.lower()
        else:
            # SHA-256 truncated to 24 hex chars (96 bits) — same collision
            # space as the MD5[:24] this replaced. Switched from MD5 in
            # PR #395 to silence CodeQL's `py/weak-sensitive-data-hashing`
            # rule (the `usedforsecurity=False` parameter and the `lgtm`
            # comment marker both failed to suppress it). The cache is
            # in-memory, so changing the hash function on upgrade has no
            # persistence-side effect — the same content always hashes
            # deterministically under whichever function is in use.
            hash_key = hashlib.sha256(original.encode()).hexdigest()[:24]

        entry = CompressionEntry(
            hash=hash_key,
            original_content=original,
            compressed_content=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            original_item_count=original_item_count,
            compressed_item_count=compressed_item_count,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            query_context=query_context,
            created_at=time.time(),
            ttl=ttl if ttl is not None else self._default_ttl,
            tool_signature_hash=tool_signature_hash,
            compression_strategy=compression_strategy,
        )

        # Process pending feedback BEFORE acquiring lock for eviction.
        # This ensures feedback from entries about to be evicted is captured.
        if self._enable_feedback:
            self.process_pending_feedback()

        with self._lock:
            self._evict_if_needed()

            # CRITICAL FIX: Hash collision detection
            # If hash already exists with DIFFERENT content, log a warning.
            # This indicates either a hash collision or duplicate store calls.
            existing = self._backend.get(hash_key)
            if existing is not None:
                if existing.original_content != original:
                    # True hash collision - different content, same hash
                    # This is extremely rare with SHA256[:24] but should be logged
                    logger.warning(
                        "Hash collision detected: hash=%s tool=%s (existing_len=%d, new_len=%d)",
                        hash_key,
                        tool_name,
                        len(existing.original_content),
                        len(original),
                    )
                else:
                    # Same content being stored again - this is fine, just update
                    logger.debug(
                        "Duplicate store for hash=%s, updating entry",
                        hash_key,
                    )
                # Mark old heap entry as stale since we're replacing
                self._stale_heap_entries += 1

            self._backend.set(hash_key, entry)
            # MEDIUM FIX #16: Add to eviction heap for O(log n) eviction
            heapq.heappush(self._eviction_heap, (entry.created_at, hash_key))

        return hash_key

    def retrieve(
        self,
        hash_key: str,
        query: str | None = None,
    ) -> CompressionEntry | None:
        """Retrieve original content by hash.

        Args:
            hash_key: Hash key returned by store().
            query: Optional query for feedback tracking.

        Returns:
            CompressionEntry if found and not expired, None otherwise.
        """
        with self._lock:
            entry = self._backend.get(hash_key)

            if entry is None:
                return None

            if entry.is_expired():
                self._backend.delete(hash_key)
                # CRITICAL FIX: Track stale heap entry
                self._stale_heap_entries += 1
                return None

            # Track access for feedback
            entry.record_access(query)
            # Update the backend with the modified entry
            self._backend.set(hash_key, entry)

            # Log retrieval event
            if self._enable_feedback:
                self._log_retrieval(
                    hash_key=hash_key,
                    query=query,
                    items_retrieved=entry.original_item_count,
                    total_items=entry.original_item_count,
                    tool_name=entry.tool_name,
                    retrieval_type="full",
                    tool_signature_hash=entry.tool_signature_hash,
                )
            self._log_retrieval_payload(
                hash_key=hash_key,
                query=query,
                retrieval_type="full",
                payload=entry.original_content,
                items_retrieved=entry.original_item_count,
                total_items=entry.original_item_count,
                entry=entry,
            )

            # CRITICAL: Make a deep copy to return
            # (entry could be modified/evicted after lock release)
            # The entry contains mutable fields (search_queries list) that must be copied
            result_entry = replace(entry, search_queries=list(entry.search_queries))

        # Process feedback immediately to ensure TOIN learns in real-time
        if self._enable_feedback:
            self.process_pending_feedback()

        return result_entry

    def get_metadata(
        self,
        hash_key: str,
    ) -> dict[str, Any] | None:
        """Get metadata about a stored entry without retrieving full content.

        Useful for context tracking to know what was compressed without
        fetching the entire original content.

        Args:
            hash_key: Hash key returned by store().

        Returns:
            Dict with metadata if found and not expired, None otherwise.
        """
        with self._lock:
            entry = self._backend.get(hash_key)

            if entry is None:
                return None

            if entry.is_expired():
                self._backend.delete(hash_key)
                self._stale_heap_entries += 1
                return None

            return {
                "hash": entry.hash,
                "tool_name": entry.tool_name,
                "original_item_count": entry.original_item_count,
                "compressed_item_count": entry.compressed_item_count,
                "query_context": entry.query_context,
                "compressed_content": entry.compressed_content,
                "created_at": entry.created_at,
                "ttl": entry.ttl,
            }

    def search(
        self,
        hash_key: str,
        query: str,
        max_results: int = 20,
        score_threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Search within cached content using BM25.

        Args:
            hash_key: Hash key of cached content.
            query: Search query.
            max_results: Maximum number of results to return.
            score_threshold: Minimum BM25 score to include.

        Returns:
            List of matching items from original content.
        """
        # Get entry without logging (we'll log the search separately)
        entry = self._get_entry_for_search(hash_key, query)
        if entry is None:
            return []

        items = self._search_items_from_original(entry.original_content)

        if not items:
            return []

        # Score each item using BM25
        item_strs = [json.dumps(item, default=str) for item in items]
        scores = self._scorer.score_batch(item_strs, query)

        # Filter and sort by score
        scored_items = [
            (items[i], scores[i].score)
            for i in range(len(items))
            if scores[i].score >= score_threshold
        ]
        scored_items.sort(key=lambda x: x[1], reverse=True)

        results = [item for item, _ in scored_items[:max_results]]

        # Log retrieval event
        if self._enable_feedback:
            with self._lock:
                self._log_retrieval(
                    hash_key=hash_key,
                    query=query,
                    items_retrieved=len(results),
                    total_items=len(items),
                    tool_name=entry.tool_name,
                    retrieval_type="search",
                    tool_signature_hash=entry.tool_signature_hash,
                )
            # Process feedback immediately to ensure TOIN learns in real-time
            self.process_pending_feedback()
        self._log_retrieval_payload(
            hash_key=hash_key,
            query=query,
            retrieval_type="search",
            payload=json.dumps(results, ensure_ascii=False),
            items_retrieved=len(results),
            total_items=len(items),
            entry=entry,
        )

        return results

    def _log_retrieval_payload(
        self,
        *,
        hash_key: str,
        query: str | None,
        retrieval_type: str,
        payload: str,
        items_retrieved: int,
        total_items: int,
        entry: CompressionEntry,
    ) -> None:
        event = {
            "event": "headroom_retrieve",
            "hash": hash_key,
            "retrieval_type": retrieval_type,
            "query": query,
            "items_retrieved": items_retrieved,
            "total_items": total_items,
            "tool_name": entry.tool_name,
            "tool_call_id": entry.tool_call_id,
            "compression_strategy": entry.compression_strategy,
            "tool_signature_hash": entry.tool_signature_hash,
            "original_tokens": entry.original_tokens,
            "compressed_tokens": entry.compressed_tokens,
            "original_item_count": entry.original_item_count,
            "compressed_item_count": entry.compressed_item_count,
            **_payload_for_retrieval_log(payload),
        }
        logger.info(
            "event=headroom_retrieve %s",
            json.dumps(event, ensure_ascii=False, separators=(",", ":")),
        )

    def _search_items_from_original(self, original_content: str) -> list[Any]:
        """Normalize cached originals into searchable items.

        CCR producers store different shapes:
        - SmartCrusher/search-style paths usually store JSON arrays.
        - Kompress stores the original plain text.
        - Some callers store JSON objects or scalar JSON values.

        Search should work for all of them. Preserve the legacy JSON-array
        result shape, but fall back to structured text chunks for everything
        else so `headroom_retrieve(hash, query=...)` can find plain-text
        originals.
        """

        try:
            parsed = json.loads(original_content)
        except json.JSONDecodeError:
            return self._plain_text_search_items(original_content)

        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return self._json_object_search_items(parsed)
        if isinstance(parsed, str):
            return self._plain_text_search_items(parsed)
        if parsed is None:
            return []
        return [{"type": "json_scalar", "value": parsed}]

    def _json_object_search_items(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        """Return searchable leaf records for a JSON object."""

        items: list[dict[str, Any]] = []

        def walk(node: Any, path: str) -> None:
            if isinstance(node, dict):
                for key, child in node.items():
                    child_path = f"{path}.{key}" if path else str(key)
                    walk(child, child_path)
                return
            if isinstance(node, list):
                for idx, child in enumerate(node):
                    walk(child, f"{path}[{idx}]")
                return
            if node is None:
                return
            items.append({"type": "json_leaf", "path": path, "value": node})

        walk(value, "")
        if items:
            return items
        return [{"type": "json_object", "value": value}]

    def _plain_text_search_items(self, text: str) -> list[dict[str, Any]]:
        """Chunk arbitrary text into searchable records.

        Line-aware chunks work well for logs/source. Word-window chunks handle
        Kompress originals, which are often long single-line text blobs.
        """

        if not text or not text.strip():
            return []

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized.split("\n")
        if len(lines) > 1:
            return self._line_text_search_items(lines)

        words = normalized.split()
        if not words:
            return []
        max_words = 350
        overlap_words = 50
        if len(words) <= max_words:
            return [
                {
                    "type": "text",
                    "text": normalized,
                    "chunk_index": 0,
                    "word_start": 1,
                    "word_end": len(words),
                }
            ]

        items: list[dict[str, Any]] = []
        start = 0
        chunk_index = 0
        step = max_words - overlap_words
        while start < len(words):
            end = min(len(words), start + max_words)
            items.append(
                {
                    "type": "text",
                    "text": " ".join(words[start:end]),
                    "chunk_index": chunk_index,
                    "word_start": start + 1,
                    "word_end": end,
                }
            )
            if end == len(words):
                break
            start += step
            chunk_index += 1
        return items

    @staticmethod
    def _line_text_search_items(lines: list[str]) -> list[dict[str, Any]]:
        max_chars = 2000
        items: list[dict[str, Any]] = []
        current: list[str] = []
        line_start = 1
        char_count = 0

        for idx, line in enumerate(lines, start=1):
            line_len = len(line) + 1
            if current and char_count + line_len > max_chars:
                items.append(
                    {
                        "type": "text",
                        "text": "\n".join(current),
                        "chunk_index": len(items),
                        "line_start": line_start,
                        "line_end": idx - 1,
                    }
                )
                current = []
                line_start = idx
                char_count = 0
            current.append(line)
            char_count += line_len

        if current:
            items.append(
                {
                    "type": "text",
                    "text": "\n".join(current),
                    "chunk_index": len(items),
                    "line_start": line_start,
                    "line_end": len(lines),
                }
            )
        return items

    def _get_entry_for_search(
        self,
        hash_key: str,
        query: str | None = None,
    ) -> CompressionEntry | None:
        """Get entry without logging retrieval (used by search to avoid double-logging).

        CRITICAL FIX #4: Returns a copy of the entry to prevent race conditions.
        The caller may use the entry after we release the lock, and another thread
        could modify or evict the original entry.

        Args:
            hash_key: Hash key returned by store().
            query: Optional query for access tracking.

        Returns:
            CompressionEntry copy if found and not expired, None otherwise.
        """
        with self._lock:
            entry = self._backend.get(hash_key)

            if entry is None:
                return None

            if entry.is_expired():
                self._backend.delete(hash_key)
                # CRITICAL FIX: Track stale heap entry
                self._stale_heap_entries += 1
                return None

            # Track access but don't log retrieval event (search will log separately)
            entry.record_access(query)
            # Update the backend with the modified entry
            self._backend.set(hash_key, entry)

            # CRITICAL FIX #4: Return a copy to prevent race conditions
            # The entry contains mutable fields (search_queries list) that could be
            # modified by other threads after we release the lock
            return replace(entry, search_queries=list(entry.search_queries))

    def exists(self, hash_key: str, clean_expired: bool = False) -> bool:
        """Check if a hash key exists and is not expired.

        Args:
            hash_key: The hash key to check.
            clean_expired: If True, delete the entry if expired.
                          LOW FIX #20: Default False to make this a pure check.

        Returns:
            True if the entry exists and is not expired.
        """
        with self._lock:
            entry = self._backend.get(hash_key)
            if entry is None:
                return False
            if entry.is_expired():
                # LOW FIX #20: Only delete if explicitly requested
                # This makes exists() a pure check by default
                if clean_expired:
                    self._backend.delete(hash_key)
                    # CRITICAL FIX: Track stale heap entry
                    self._stale_heap_entries += 1
                return False
            return True

    def get_entry_status(
        self,
        hash_key: str,
        *,
        clean_expired: bool = False,
    ) -> dict[str, Any]:
        """Return availability and TTL metadata for a stored entry."""
        now = time.time()
        with self._lock:
            entry = self._backend.get(hash_key)
            if entry is None:
                return {
                    "hash": hash_key,
                    "status": "missing",
                    "default_ttl_seconds": self._default_ttl,
                }

            age_seconds = now - entry.created_at
            expires_at = entry.created_at + entry.ttl
            expired = age_seconds > entry.ttl
            status = {
                "hash": hash_key,
                "status": "expired" if expired else "available",
                "ttl_seconds": entry.ttl,
                "default_ttl_seconds": self._default_ttl,
                "created_at": entry.created_at,
                "expires_at": expires_at,
                "age_seconds": age_seconds,
            }

            if expired and clean_expired:
                self._backend.delete(hash_key)
                self._stale_heap_entries += 1

            return status

    def get_stats(self) -> dict[str, Any]:
        """Get store statistics for monitoring."""
        with self._lock:
            # Clean expired entries
            self._clean_expired()

            # Get all entries for statistics
            entries = [entry for _, entry in self._backend.items()]
            total_original_tokens = sum(e.original_tokens for e in entries)
            total_compressed_tokens = sum(e.compressed_tokens for e in entries)
            total_retrievals = sum(e.retrieval_count for e in entries)

            # Include backend stats
            backend_stats = self._backend.get_stats()

            return {
                "entry_count": self._backend.count(),
                "max_entries": self._max_entries,
                "default_ttl_seconds": self._default_ttl,
                "total_original_tokens": total_original_tokens,
                "total_compressed_tokens": total_compressed_tokens,
                "total_retrievals": total_retrievals,
                "event_count": len(self._retrieval_events),
                "backend": backend_stats,
            }

    def get_memory_stats(self) -> ComponentStats:
        """Get memory statistics for the MemoryTracker.

        Returns:
            ComponentStats with current memory usage.
        """
        from ..memory.tracker import ComponentStats

        with self._lock:
            # Get backend stats which include bytes_used
            backend_stats = self._backend.get_stats()
            bytes_used = backend_stats.get("bytes_used", 0)

            # Add retrieval events memory
            import sys

            bytes_used += sys.getsizeof(self._retrieval_events)
            for event in self._retrieval_events:
                bytes_used += sys.getsizeof(event)

            # Add eviction heap memory
            bytes_used += sys.getsizeof(self._eviction_heap)

            return ComponentStats(
                name="compression_store",
                entry_count=self._backend.count(),
                size_bytes=bytes_used,
                budget_bytes=None,  # No budget set yet
                hits=sum(1 for _, e in self._backend.items() if e.retrieval_count > 0),
                misses=0,  # CompressionStore doesn't track misses directly
                evictions=0,  # Would need to track this separately
            )

    def get_retrieval_events(
        self,
        limit: int = 100,
        tool_name: str | None = None,
    ) -> list[RetrievalEvent]:
        """Get recent retrieval events for feedback analysis.

        Args:
            limit: Maximum number of events to return.
            tool_name: Filter by tool name if specified.

        Returns:
            List of recent retrieval events (copies to prevent mutation).
        """
        with self._lock:
            # MEDIUM FIX #17: Take a slice copy immediately to avoid race conditions
            # if another thread modifies _retrieval_events after we release the lock
            events_copy = list(self._retrieval_events)

        # Filter and slice outside lock (safe since we have a copy)
        if tool_name:
            events_copy = [e for e in events_copy if e.tool_name == tool_name]

        return list(reversed(events_copy[-limit:]))

    def clear(self) -> None:
        """Clear all entries. Mainly for testing."""
        with self._lock:
            self._backend.clear()
            self._retrieval_events.clear()
            self._pending_feedback_events.clear()
            self._eviction_heap.clear()  # MEDIUM FIX #16: Clear heap too
            self._stale_heap_entries = 0  # CRITICAL FIX: Reset stale counter

    def _evict_if_needed(self) -> None:
        """Evict old entries if at capacity. Must be called with lock held.

        MEDIUM FIX #16: Use heap for O(log n) eviction instead of O(n) scan.
        CRITICAL FIX: Track and clean stale heap entries to prevent memory leak.
        """
        # First, remove expired entries
        self._clean_expired()

        # CRITICAL FIX: Rebuild heap if too many stale entries
        # This prevents unbounded heap growth when entries are deleted/replaced
        heap_size = len(self._eviction_heap)
        if heap_size > 0:
            stale_ratio = self._stale_heap_entries / heap_size
            if stale_ratio >= self._heap_rebuild_threshold:
                self._rebuild_heap()

        # If still at capacity, remove oldest entries using heap
        while self._backend.count() >= self._max_entries and self._eviction_heap:
            # Pop oldest from heap (O(log n))
            created_at, hash_key = heapq.heappop(self._eviction_heap)

            # Check if entry still exists and matches timestamp
            # (entry might have been deleted or replaced)
            entry = self._backend.get(hash_key)
            if entry is not None and entry.created_at == created_at:
                # HIGH FIX: Track eviction as "successful compression" if never retrieved
                # This prevents state divergence between store and feedback loop
                if self._enable_feedback and entry.retrieval_count == 0:
                    # Entry was never retrieved = compression was successful
                    # Notify feedback system so it knows this strategy worked
                    self._record_eviction_success(entry)
                self._backend.delete(hash_key)
            else:
                # CRITICAL FIX: This was a stale entry, decrement counter
                # (we already popped it, so the stale entry is now gone)
                if self._stale_heap_entries > 0:
                    self._stale_heap_entries -= 1

    def _clean_expired(self) -> None:
        """Remove expired entries. Must be called with lock held.

        CRITICAL FIX: Track stale heap entries when deleting to prevent memory leak.
        """
        expired_keys = [key for key, entry in self._backend.items() if entry.is_expired()]
        for key in expired_keys:
            self._backend.delete(key)
            # CRITICAL FIX: Increment stale counter - the heap still has an entry
            # for this key that will be stale when we try to evict
            self._stale_heap_entries += 1

    def _rebuild_heap(self) -> None:
        """Rebuild heap from current store entries. Must be called with lock held.

        CRITICAL FIX: This removes stale heap entries that accumulate when entries
        are deleted or replaced. Without this, the heap grows unboundedly.
        """
        # Build new heap from current store entries only
        self._eviction_heap = [
            (entry.created_at, hash_key) for hash_key, entry in self._backend.items()
        ]
        heapq.heapify(self._eviction_heap)
        # Reset stale counter - heap is now clean
        self._stale_heap_entries = 0
        logger.debug(
            "Rebuilt eviction heap: %d entries",
            len(self._eviction_heap),
        )

    def _record_eviction_success(self, entry: CompressionEntry) -> None:
        """Record successful compression when an entry is evicted without retrieval.

        HIGH FIX: State divergence on eviction
        When an entry is evicted and was NEVER retrieved, this indicates the
        compression was fully successful - the LLM never needed the original data.
        We notify the feedback system so it can learn from this success.

        Must be called with lock held (entry data access).
        Actual feedback notification happens outside lock.

        Args:
            entry: The entry being evicted.
        """
        # Capture entry data while we have the lock
        tool_name = entry.tool_name
        sig_hash = entry.tool_signature_hash
        strategy = entry.compression_strategy

        # We can't call feedback while holding the lock (would cause deadlock)
        # Instead, queue this for deferred processing
        if sig_hash is not None and strategy is not None:
            # Create a synthetic "success" event that we'll process later
            # Use a special retrieval type to indicate this was an eviction success
            success_event = RetrievalEvent(
                hash=entry.hash,
                query=None,
                items_retrieved=0,  # No retrieval happened
                total_items=entry.original_item_count,
                tool_name=tool_name,
                timestamp=time.time(),
                retrieval_type="eviction_success",  # Special marker
                tool_signature_hash=sig_hash,
            )
            self._pending_feedback_events.append(success_event)
            logger.debug(
                "Recorded eviction success: hash=%s strategy=%s",
                entry.hash[:8],
                strategy,
            )

    def _log_retrieval(
        self,
        hash_key: str,
        query: str | None,
        items_retrieved: int,
        total_items: int,
        tool_name: str | None,
        retrieval_type: str,
        tool_signature_hash: str | None = None,
    ) -> None:
        """Log a retrieval event. Must be called with lock held."""
        event = RetrievalEvent(
            hash=hash_key,
            query=query,
            items_retrieved=items_retrieved,
            total_items=total_items,
            tool_name=tool_name,
            timestamp=time.time(),
            retrieval_type=retrieval_type,
            tool_signature_hash=tool_signature_hash,
        )

        self._retrieval_events.append(event)

        # Keep only recent events
        if len(self._retrieval_events) > self._max_events:
            self._retrieval_events = self._retrieval_events[-self._max_events :]

        # Queue event for feedback processing (will be processed after lock release)
        # This is safe because process_pending_feedback() uses the lock to atomically
        # swap out the pending list before processing
        self._pending_feedback_events.append(event)

    def process_pending_feedback(self) -> None:
        """Process pending feedback events.

        Forwards events to:
        1. CompressionFeedback - for learning compression hints
        2. TelemetryCollector - for the data flywheel
        3. TOIN - for cross-user intelligence network

        This is called automatically on each retrieval to ensure the
        feedback loop operates in real-time.
        """
        from ..telemetry import get_telemetry_collector
        from ..telemetry.toin import get_toin
        from .compression_feedback import get_compression_feedback

        # Get pending events and related entry data atomically
        with self._lock:
            events = self._pending_feedback_events
            self._pending_feedback_events = []

            # Gather entry data while holding lock to avoid race conditions
            # Tuple: (event, tool_name, sig_hash, strategy, compressed_content)
            event_data: list[
                tuple[RetrievalEvent, str | None, str | None, str | None, str | None]
            ] = []
            for event in events:
                entry = self._backend.get(event.hash)
                if entry:
                    # Use the ACTUAL tool_signature_hash stored during compression
                    # This MUST match the hash used by SmartCrusher
                    event_data.append(
                        (
                            event,
                            entry.tool_name,
                            entry.tool_signature_hash,  # The correct hash!
                            entry.compression_strategy,
                            entry.compressed_content,  # For TOIN field-level learning
                        )
                    )
                else:
                    event_data.append((event, None, None, None, None))

        # Process outside lock
        if event_data:
            feedback = get_compression_feedback()
            telemetry = get_telemetry_collector()
            toin = get_toin()

            for event, _tool_name, sig_hash, strategy, compressed_content in event_data:
                # Notify feedback system (pass strategy for success rate tracking)
                feedback.record_retrieval(event, strategy=strategy)

                # Extract query fields if present
                query_fields = None
                if event.query:
                    # Extract field:value patterns
                    query_fields = re.findall(r"(\w+)[=:]", event.query)

                # Notify telemetry for data flywheel
                try:
                    if sig_hash is not None:
                        telemetry.record_retrieval(
                            tool_signature_hash=sig_hash,
                            retrieval_type=event.retrieval_type,
                            query_fields=query_fields,
                        )
                except Exception:
                    # Telemetry should never break the feedback loop
                    logger.debug("Telemetry record_retrieval failed", exc_info=True)

                # Parse compressed content to extract items for TOIN field-level learning
                retrieved_items: list[dict[str, Any]] | None = None
                if compressed_content:
                    try:
                        parsed = json.loads(compressed_content)
                        # Handle both direct arrays and wrapped arrays
                        if isinstance(parsed, list):
                            # Filter to dicts only (field learning needs dict items)
                            retrieved_items = [item for item in parsed if isinstance(item, dict)]
                        elif isinstance(parsed, dict):
                            # Check for common wrapper patterns: {"items": [...], "results": [...]}
                            for key in ("items", "results", "data", "records"):
                                if key in parsed and isinstance(parsed[key], list):
                                    retrieved_items = [
                                        item for item in parsed[key] if isinstance(item, dict)
                                    ]
                                    break
                    except (json.JSONDecodeError, TypeError):
                        # Invalid JSON - skip field learning for this retrieval
                        pass

                # Notify TOIN for cross-user learning
                try:
                    if sig_hash is not None:
                        toin.record_retrieval(
                            tool_signature_hash=sig_hash,
                            retrieval_type=event.retrieval_type,
                            query=event.query,
                            query_fields=query_fields,
                            strategy=strategy,  # Pass strategy for success rate tracking
                            retrieved_items=retrieved_items,  # For field-level learning
                        )
                except Exception:
                    # TOIN should never break the feedback loop
                    logger.debug("TOIN record_retrieval failed", exc_info=True)


# Request-scoped store (for multi-tenant SaaS: one store per request/tenant)
_request_ccr_store: ContextVar[CompressionStore | None] = ContextVar(
    "headroom_request_ccr_store", default=None
)

# Global store instance (lazy initialization)
_compression_store: CompressionStore | None = None
_store_lock = threading.Lock()


def set_request_compression_store(store: CompressionStore | None) -> None:
    """Set the compression store for the current request context.

    Used by middleware (e.g. SaaS) to provide a tenant-scoped store.
    When set, get_compression_store() returns this store instead of the global one.

    Args:
        store: CompressionStore to use for this request, or None to clear.
    """
    _request_ccr_store.set(store)


def clear_request_compression_store() -> None:
    """Clear the request-scoped compression store."""
    _request_ccr_store.set(None)


def _create_default_ccr_backend() -> CompressionStoreBackend | None:
    """Create a CCR backend from env (e.g. HEADROOM_CCR_BACKEND=redis).

    Default (env unset or "sqlite"): SQLiteBackend at
    ~/.headroom/ccr_store.db — restart-safe and shared across worker
    processes, which the session-scale 30-minute TTL assumes.
    "memory" opts back into the in-process dict. Other values load
    adapters via setuptools entry point 'headroom.ccr_backend'.
    Returns None to use InMemoryBackend.
    """
    backend_type = (os.environ.get("HEADROOM_CCR_BACKEND") or "").strip().lower()
    if backend_type == "memory":
        return None
    if not backend_type or backend_type == "sqlite":
        try:
            from .backends.sqlite import SQLiteBackend

            return SQLiteBackend()
        except Exception as e:
            logger.warning(
                "Failed to initialize SQLite CCR backend (%s); "
                "falling back to in-memory store. Retrieval will not "
                "survive proxy restarts.",
                e,
            )
            return None
    try:
        from importlib.metadata import entry_points

        all_eps = entry_points(group="headroom.ccr_backend")
        ep = next((e for e in all_eps if e.name == backend_type), None)
        if ep is None:
            logger.warning(
                "HEADROOM_CCR_BACKEND=%s but no entry point headroom.ccr_backend[%s]",
                backend_type,
                backend_type,
            )
            return None
        fn = ep.load()
        kwargs = {
            "url": os.environ.get("HEADROOM_REDIS_URL", ""),
            "tenant_prefix": os.environ.get("HEADROOM_CCR_TENANT_PREFIX", ""),
        }
        backend: CompressionStoreBackend = fn(**kwargs)
        return backend
    except Exception as e:
        logger.warning("Failed to load CCR backend %s: %s", backend_type, e)
        return None


def get_compression_store(
    max_entries: int = 1000,
    default_ttl: int | None = None,
    backend: CompressionStoreBackend | None = None,
) -> CompressionStore:
    """Get the compression store instance.

    If a request-scoped store was set (e.g. by SaaS middleware), returns it.
    Otherwise uses lazy-initialized global singleton. Backend can be supplied
    explicitly or created from env (HEADROOM_CCR_BACKEND) when building the global.

    Args:
        max_entries: Maximum entries (only used on first call for global store).
        default_ttl: Default TTL (only used on first call for global store).
            When omitted, HEADROOM_CCR_TTL_SECONDS overrides the 1800-second default.
        backend: Custom storage backend (only used on first call for global store).
                 Defaults to InMemoryBackend if not provided; env backend used if backend is None.

    Returns:
        Request-scoped CompressionStore if set, else global CompressionStore instance.
    """
    request_store = _request_ccr_store.get()
    if request_store is not None:
        return request_store

    global _compression_store
    if _compression_store is None:
        with _store_lock:
            if _compression_store is None:
                if backend is None:
                    backend = _create_default_ccr_backend()
                effective_default_ttl = (
                    default_ttl if default_ttl is not None else _get_env_default_ttl_seconds()
                )
                _compression_store = CompressionStore(
                    max_entries=max_entries,
                    default_ttl=effective_default_ttl,
                    backend=backend,
                )
    return _compression_store


def reset_compression_store() -> None:
    """Reset the global compression store. Mainly for testing."""
    global _compression_store

    with _store_lock:
        if _compression_store is not None:
            _compression_store.clear()
        _compression_store = None
