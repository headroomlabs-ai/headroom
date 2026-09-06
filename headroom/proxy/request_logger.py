"""Request logger for the Headroom proxy.

Logs requests to an in-memory deque and optionally to a JSONL file.

Extracted from server.py for maintainability.

Phase G PR-G3 (P4-45): base64-encoded image payloads in the
``request_messages`` / ``response_content`` are redacted before
write to keep request logs small. Multi-MB base64 strings would
otherwise saturate the JSONL log and the in-memory deque.

Remediation (M2, M5): the redactor now ONLY fires inside known
image-bearing JSON paths or against strings that carry an explicit
``data:image/...;base64,`` URL prefix. The earlier "density
heuristic" over-fired on encrypted blobs, signed tokens, minified
JSON, and tool outputs. The replacement placeholder now reports
the UTF-8 byte length under a ``bytes=`` label (was character
length; for the ASCII base64 alphabet the two happen to coincide
but the label is now accurate for any future Unicode payload).
"""

from __future__ import annotations

import json
import logging
import sys
from collections import deque
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..memory.tracker import ComponentStats

from headroom.proxy import request_log_redaction_policy
from headroom.proxy.models import RequestLog

IMAGE_BASE64_REDACT_THRESHOLD_BYTES = (
    request_log_redaction_policy.IMAGE_BASE64_REDACT_THRESHOLD_BYTES
)
IMAGE_BASE64_REPLACEMENT_TEMPLATE = request_log_redaction_policy.IMAGE_BASE64_REPLACEMENT_TEMPLATE
IMAGE_BEARING_FIELD_NAMES = request_log_redaction_policy.IMAGE_BEARING_FIELD_NAMES
_is_base64_image_payload = request_log_redaction_policy.is_base64_image_payload

logger = logging.getLogger(__name__)

# Constants for log redaction counter export (Prometheus). The
# Python proxy's ``/metrics`` exporter surfaces
# ``proxy_image_generation_call_log_redacted_total`` from this
# module-level counter. C3 remediation: the Rust proxy previously
# held a dead counter; that's been removed in favour of this
# Python-side counter, which is the natural owner.
_redactions_total: int = 0
_redactions_lock = Lock()


def redactions_total() -> int:
    """Return the running count of base64 redactions performed.

    Exposed for unit tests, the legacy Python ``/stats`` endpoint,
    and the Prometheus exporter
    (``proxy_image_generation_call_log_redacted_total``).
    """
    with _redactions_lock:
        return _redactions_total


def redact_image_base64(payload: Any) -> Any:
    """Public entry point for base64-image redaction.

    Walks ``payload`` (a dict, list, or string) and replaces any
    over-threshold base64 string with a size-only placeholder.
    Idempotent — applying twice yields the same structure.
    """
    global _redactions_total

    result = request_log_redaction_policy.redact_image_base64_value(payload)
    if result.redactions:
        with _redactions_lock:
            _redactions_total += result.redactions
    return result.value


class RequestLogger:
    """Log requests to JSONL file.

    Uses a deque with max ``max_entries`` entries (default 500) to cap
    in-memory footprint. Full bodies in the deque are truncated to
    ``MAX_BODY_BYTES`` (default 2 KB) with a ``[truncated]`` suffix;
    the on-disk JSONL file always receives the full body.

    Gracefully degrades to in-memory-only if the log file cannot be written
    (read-only filesystem, permissions error, etc.).
    """

    # Default cap: 500 entries.  At ~2 KB per stored body this is ~1 MB
    # of retained text — a 20x reduction from the previous 10,000-entry
    # default that could accumulate ~100 MB on a busy proxy.
    DEFAULT_MAX_ENTRIES: int = 500

    # Body strings stored in the deque are truncated at this byte length.
    # The on-disk log file always retains the full body.
    MAX_BODY_BYTES: int = 2048

    def __init__(
        self,
        log_file: str | None = None,
        log_full_messages: bool = False,
        max_entries: int | None = None,
    ) -> None:
        """Initialize the request logger.

        Args:
            log_file: Path to the JSONL log file. Pass ``None`` to disable
                on-disk logging (in-memory only).
            log_full_messages: When True, ``request_messages`` and
                ``response_content`` are included in log entries.
            max_entries: Maximum number of entries retained in the in-memory
                deque. Defaults to ``DEFAULT_MAX_ENTRIES`` (500).
        """
        self.log_file = Path(log_file) if log_file else None
        self.log_full_messages = log_full_messages
        self._max_entries: int = (
            max_entries if max_entries is not None else self.DEFAULT_MAX_ENTRIES
        )
        # Use deque with maxlen for automatic FIFO eviction
        self._logs: deque[RequestLog] = deque(maxlen=self._max_entries)

        if self.log_file:
            try:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.warning(
                    "Cannot create log directory %s: %s — logging to memory only",
                    self.log_file.parent,
                    e,
                )
                self.log_file = None

    @staticmethod
    def _truncate_body(value: str, max_bytes: int) -> str:
        """Truncate a body string to ``max_bytes`` UTF-8 bytes.

        Appends a ``[truncated]`` marker when the value exceeds the limit.
        Returns the value unchanged when it is already within bounds.

        Args:
            value: The string to truncate.
            max_bytes: Maximum allowed UTF-8 byte length.

        Returns:
            The (possibly truncated) string.
        """
        encoded = value.encode("utf-8")
        if len(encoded) <= max_bytes:
            return value
        return encoded[:max_bytes].decode("utf-8", errors="replace") + " [truncated]"

    def log(self, entry: RequestLog) -> None:
        """Log a request. Oldest entries are automatically removed when limit reached.

        Phase G PR-G3 (P4-45): base64-encoded image payloads in
        ``request_messages`` / ``compressed_messages`` / ``response_content``
        are redacted before write. Redaction also applies to the in-memory
        deque so the ``/stats/recent_requests`` endpoint never serves a
        multi-MB image either.

        Bodies stored in the deque are additionally truncated to
        ``MAX_BODY_BYTES`` (2 KB) to cap per-entry memory. The on-disk
        JSONL file always receives the full (but still image-redacted)
        body.
        """
        # Redact image payloads first (before truncation, so we measure
        # the redacted size, not the original multi-MB payload).
        if entry.request_messages is not None:
            entry.request_messages = redact_image_base64(entry.request_messages)
        if entry.compressed_messages is not None:
            entry.compressed_messages = redact_image_base64(entry.compressed_messages)
        if entry.response_content is not None:
            entry.response_content = redact_image_base64(entry.response_content)

        # Write the full (image-redacted) body to disk before truncation.
        if self.log_file:
            try:
                with open(self.log_file, "a") as f:
                    log_dict = asdict(entry)
                    if not self.log_full_messages:
                        log_dict.pop("request_messages", None)
                        log_dict.pop("compressed_messages", None)
                        log_dict.pop("response_content", None)
                    f.write(json.dumps(log_dict) + "\n")
            except OSError:
                pass  # Graceful degradation: memory-only logging continues

        # Truncate bodies in the deque copy to keep per-entry memory bounded.
        # We store a shallow copy of the entry with truncated strings rather
        # than mutating the caller's object (callers may re-use the entry).
        if entry.response_content is not None and isinstance(entry.response_content, str):
            truncated_content = self._truncate_body(entry.response_content, self.MAX_BODY_BYTES)
            if truncated_content is not entry.response_content:
                from dataclasses import replace as _dc_replace

                entry = _dc_replace(entry, response_content=truncated_content)

        self._logs.append(entry)

    def get_recent(self, n: int = 100) -> list[dict]:
        """Get recent log entries (without request/compressed messages and response_content)."""
        # Convert deque to list for slicing (deque doesn't support slicing)
        entries = list(self._logs)[-n:]
        return [
            {
                k: v
                for k, v in asdict(e).items()
                if k not in ("request_messages", "compressed_messages", "response_content")
            }
            for e in entries
        ]

    def get_recent_with_messages(self, n: int = 20) -> list[dict]:
        """Get recent log entries including full request/response messages."""
        entries = list(self._logs)[-n:]
        return [asdict(e) for e in entries]

    def stats(self) -> dict:
        """Get logging statistics."""
        return {
            "total_logged": len(self._logs),
            "log_file": str(self.log_file) if self.log_file else None,
        }

    def get_memory_stats(self) -> ComponentStats:
        """Get memory statistics for the MemoryTracker.

        Returns:
            ComponentStats with current memory usage.
        """
        from ..memory.tracker import ComponentStats

        # Calculate size
        size_bytes = sys.getsizeof(self._logs)

        for log_entry in self._logs:
            size_bytes += sys.getsizeof(log_entry)
            # Add string fields
            if log_entry.request_id:
                size_bytes += len(log_entry.request_id)
            if log_entry.provider:
                size_bytes += len(log_entry.provider)
            if log_entry.model:
                size_bytes += len(log_entry.model)
            if log_entry.error:
                size_bytes += len(log_entry.error)
            # Messages and response can be large
            if log_entry.request_messages:
                size_bytes += sys.getsizeof(log_entry.request_messages)
            if log_entry.compressed_messages:
                size_bytes += sys.getsizeof(log_entry.compressed_messages)
            if log_entry.response_content:
                size_bytes += len(log_entry.response_content)

        return ComponentStats(
            name="request_logger",
            entry_count=len(self._logs),
            size_bytes=size_bytes,
            budget_bytes=None,
            hits=0,
            misses=0,
            evictions=0,
        )
