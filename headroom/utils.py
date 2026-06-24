"""Shared utilities for Headroom SDK."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

# Marker format for Headroom modifications
MARKER_PREFIX = "<headroom:"
MARKER_SUFFIX = ">"


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


def compute_hash(data: str | bytes) -> str:
    """Compute SHA256 hash, returning hex string."""
    if isinstance(data, str):
        data = data.encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(data).hexdigest()


def compute_short_hash(data: str | bytes, length: int = 16) -> str:
    """Compute truncated SHA256 hash."""
    return compute_hash(data)[:length]


def fast_hash(data: str | bytes, length: int = 16) -> str:
    """Fast non-cryptographic content hash for caches and dedup.

    Uses MD5 (2-3x faster than SHA256).  Not used for security — only for
    content-addressable lookups in compression caches, prefix tracking, etc.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.md5(data).hexdigest()[:length]  # nosec B324


def extract_user_query(messages: list[dict[str, Any]]) -> str:
    """Extract the most recent user question from messages.

    Used to pass context through the compression pipeline so transforms like
    SmartCrusher can score items by relevance to the user's actual question,
    not just by statistical properties (position, anomaly, boundary).
    """
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = str(block.get("text", "")).strip()
                        if text:
                            return text
    return ""


def compute_messages_hash(messages: list[dict[str, Any]]) -> str:
    """Compute hash of messages list for deduplication."""
    # Serialize deterministically
    serialized = json.dumps(messages, sort_keys=True, separators=(",", ":"))
    return compute_short_hash(serialized)


def compute_prefix_hash(messages: list[dict[str, Any]], prefix_count: int | None = None) -> str:
    """
    Compute hash of message prefix for cache alignment.

    Args:
        messages: List of messages.
        prefix_count: Number of messages to include (default: all system messages + 1).

    Returns:
        Hash of the prefix content.
    """
    if not messages:
        return compute_short_hash("")

    if prefix_count is None:
        # Default: system messages + first non-system
        prefix_count = 1
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                prefix_count = i + 2
            else:
                break

    prefix_messages = messages[:prefix_count]
    serialized = json.dumps(prefix_messages, sort_keys=True, separators=(",", ":"))
    return compute_short_hash(serialized)


def format_timestamp(dt: datetime | None = None) -> str:
    """Format datetime as ISO8601 string."""
    if dt is None:
        dt = datetime.now(timezone.utc).replace(tzinfo=None)
    return dt.isoformat() + "Z"


def parse_timestamp(ts: str) -> datetime:
    """Parse ISO8601 timestamp string."""
    # Handle both with and without Z suffix
    ts = ts.rstrip("Z")
    return datetime.fromisoformat(ts)


def create_marker(marker_type: str, **kwargs: Any) -> str:
    """
    Create a Headroom marker string.

    Args:
        marker_type: Type of marker (e.g., "tool_digest", "dropped_context").
        **kwargs: Attributes to include in the marker.

    Returns:
        Formatted marker string.
    """
    attrs = " ".join(f'{k}="{v}"' for k, v in kwargs.items())
    if attrs:
        return f"{MARKER_PREFIX}{marker_type} {attrs}{MARKER_SUFFIX}"
    return f"{MARKER_PREFIX}{marker_type}{MARKER_SUFFIX}"


def create_tool_digest_marker(original_hash: str) -> str:
    """Create marker for crushed tool output."""
    return create_marker("tool_digest", sha256=original_hash)


def create_dropped_context_marker(reason: str, count: int | None = None) -> str:
    """Create marker for dropped context."""
    if count is not None:
        return create_marker("dropped_context", reason=reason, count=str(count))
    return create_marker("dropped_context", reason=reason)


def create_truncated_marker(original_length: int, truncated_to: int) -> str:
    """Create marker for truncated content."""
    return create_marker(
        "truncated",
        original=str(original_length),
        truncated_to=str(truncated_to),
    )


def extract_markers(text: str) -> list[dict[str, Any]]:
    """
    Extract Headroom markers from text.

    Returns:
        List of dicts with marker_type and attributes.
    """
    pattern = re.compile(r"<headroom:(\w+)([^>]*)>")
    markers = []

    for match in pattern.finditer(text):
        marker_type = match.group(1)
        attrs_str = match.group(2).strip()

        # Parse attributes
        attrs: dict[str, str] = {}
        if attrs_str:
            attr_pattern = re.compile(r'(\w+)="([^"]*)"')
            for attr_match in attr_pattern.finditer(attrs_str):
                attrs[attr_match.group(1)] = attr_match.group(2)

        markers.append({"type": marker_type, "attributes": attrs})

    return markers


def safe_json_loads(text: str) -> tuple[Any | None, bool]:
    """
    Safely parse JSON, returning (result, success).

    Args:
        text: JSON string to parse.

    Returns:
        Tuple of (parsed_result or None, success_bool).
    """
    try:
        return json.loads(text), True
    except (json.JSONDecodeError, ValueError):
        return None, False


def safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    """
    Safely serialize to JSON with defaults.

    Args:
        obj: Object to serialize.
        **kwargs: Additional json.dumps arguments.

    Returns:
        JSON string.
    """
    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("separators", (",", ":"))  # Compact by default
    return json.dumps(obj, **kwargs)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
    cached_tokens: int = 0,
    provider: Any = None,
) -> float | None:
    """
    Estimate API cost in USD using provider.

    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        model: Model name.
        cached_tokens: Number of cached input tokens.
        provider: Provider instance for cost estimation.

    Returns:
        Estimated cost in USD, or None if not available.
    """
    if provider is None:
        return None
    result = provider.estimate_cost(input_tokens, output_tokens, model, cached_tokens)
    return float(result) if result is not None else None


def format_cost(cost: float) -> str:
    """Format cost as human-readable string."""
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def deep_copy_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create a deep copy of messages list.

    Uses copy.deepcopy instead of json roundtrip (2-5x faster, avoids
    serialisation overhead on large conversation histories).
    """
    return copy.deepcopy(messages)
