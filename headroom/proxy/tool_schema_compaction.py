"""Shared tool-schema compaction for Headroom proxy handlers.

Strips JSON Schema annotation keys ($schema, title, examples, etc.)
and normalises description whitespace to reduce the token cost of
tool definitions without changing their semantics.

Both the OpenAI and Anthropic handlers call the same compaction
logic from this module.

**Layer 2 — Tool Description Compaction**

Truncates tool and parameter ``description`` strings to a configurable
maximum length, preserving the first complete sentence so that the model
can still select the right tool.  Opt-in via
``HEADROOM_TOOL_DESC_MAX_CHARS`` (default ``0`` = disabled).

**Layer 3 — Semantic Parameter Description Removal**

When a parameter name is self-explanatory (e.g. ``query``, ``owner``,
``repo``), the ``description`` field adds little value — the model can
infer the meaning from the name alone.  Opt-in via
``HEADROOM_TOOL_DESC_STRIP_SEMANTIC=1`` (default disabled).

**Caching**

Compaction results are keyed by the JSON digest of the tools array and
the compaction config.  Within a session where tools don't change, the
cached result is reused — avoiding redundant recursive walks over 141+
tool schemas on every API call.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
from typing import Any

# Keys that are JSON Schema annotations, not constraints.
# Removing them does not change the set of valid inputs.
TOOL_SCHEMA_DROP_KEYS: frozenset[str] = frozenset(
    {
        "$id",
        "$schema",
        "$comment",
        "deprecated",
        "examples",
        "example",
        "markdownDescription",
        "readOnly",
        "title",
        "writeOnly",
    }
)

# Parameter names that are self-explanatory.  When ``description``
# matches the name (case-insensitive prefix), it can be stripped.
_SEMANTIC_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "query",
        "search",
        "filter",
        "sort",
        "order",
        "limit",
        "offset",
        "page",
        "per_page",
        "perpage",
        "cursor",
        "after",
        "before",
        "owner",
        "repo",
        "repository",
        "org",
        "organization",
        "user",
        "username",
        "email",
        "name",
        "title",
        "description",
        "id",
        "number",
        "count",
        "url",
        "path",
        "file",
        "filename",
        "branch",
        "tag",
        "sha",
        "commit",
        "ref",
        "key",
        "token",
        "type",
        "format",
        "state",
        "status",
        "action",
        "method",
        "body",
        "content",
        "message",
        "text",
        "comment",
        "note",
        "start",
        "end",
        "from",
        "to",
        "direction",
        "ascending",
        "dry_run",
        "verbose",
        "force",
        "recursive",
        "include",
        "exclude",
        "pattern",
        "regex",
        "since",
        "until",
    }
)

# ---------------------------------------------------------------------------
# Env-var helpers
# ---------------------------------------------------------------------------

_TOOL_DESC_MAX_CHARS: int | None = None
_STRIP_SEMANTIC: bool | None = None


def tool_desc_max_chars() -> int:
    """Return the configured max description length (cached per-process).

    ``HEADROOM_TOOL_DESC_MAX_CHARS=0`` (default) disables truncation.
    """
    global _TOOL_DESC_MAX_CHARS
    if _TOOL_DESC_MAX_CHARS is None:
        try:
            _TOOL_DESC_MAX_CHARS = int(os.environ.get("HEADROOM_TOOL_DESC_MAX_CHARS", "0"))
        except ValueError:
            _TOOL_DESC_MAX_CHARS = 0
    return _TOOL_DESC_MAX_CHARS


def strip_semantic_params() -> bool:
    """Return whether Layer 3 (semantic param removal) is enabled."""
    global _STRIP_SEMANTIC
    if _STRIP_SEMANTIC is None:
        _STRIP_SEMANTIC = os.environ.get("HEADROOM_TOOL_DESC_STRIP_SEMANTIC", "0") == "1"
    return _STRIP_SEMANTIC


# ---------------------------------------------------------------------------
# Compaction cache
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_compaction_cache: dict[str, tuple[dict[str, Any], int, int]] = {}
_CACHE_MAX_ENTRIES = 8


def _cache_key(tools: list[Any], *config_vals: Any) -> str:
    """Deterministic cache key from tools content + config values."""
    h = hashlib.sha256()
    h.update(json.dumps(tools, sort_keys=True, default=str, separators=(",", ":")).encode())
    for v in config_vals:
        h.update(str(v).encode())
    return h.hexdigest()[:16]


def _cache_get(key: str) -> tuple[dict[str, Any], int, int] | None:
    with _cache_lock:
        return _compaction_cache.get(key)


def _cache_put(key: str, compacted: dict[str, Any], before: int, after: int) -> None:
    with _cache_lock:
        if len(_compaction_cache) >= _CACHE_MAX_ENTRIES:
            # Evict oldest entry (first key).
            _compaction_cache.pop(next(iter(_compaction_cache)))
        _compaction_cache[key] = (compacted, before, after)


def invalidate_cache() -> None:
    """Clear the compaction cache (e.g. on config change)."""
    with _cache_lock:
        _compaction_cache.clear()


# ---------------------------------------------------------------------------
# Layer 1: annotation-key compaction
# ---------------------------------------------------------------------------


def _json_byte_len(value: Any) -> int:
    """Byte length of compact JSON serialisation (for size comparisons)."""
    return len(json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":")))


def compact_tool_schema_value(
    value: Any,
    _parent_key: str | None = None,
) -> Any:
    """Recursively compact a tool-schema structure.

    - Drops annotation keys (``TOOL_SCHEMA_DROP_KEYS``) unless they appear
      as property *names* inside a ``properties`` object (e.g. a field
      literally named ``"title"`` must survive).
    - Normalises ``description`` strings by collapsing whitespace.
    """
    if isinstance(value, list):
        return [compact_tool_schema_value(item, _parent_key) for item in value]

    if not isinstance(value, dict):
        return value

    compacted: dict[str, Any] = {}
    for key, child in value.items():
        # Don't drop keys that are property *names* inside a JSON Schema
        # `properties` object — only drop them when they are schema annotations.
        if _parent_key != "properties" and key in TOOL_SCHEMA_DROP_KEYS:
            continue

        if key == "description" and isinstance(child, str):
            compacted[key] = " ".join(child.split())
            continue

        compacted[key] = compact_tool_schema_value(child, key)

    return compacted


def compact_tools(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], bool, int, int]:
    """Compact the ``tools`` array in *payload*.

    Returns ``(updated_payload, modified, before_bytes, after_bytes)``.
    If compaction did not reduce size, the original payload is returned
    unchanged and *modified* is ``False``.

    Results are cached by tools digest — repeated calls with the same
    tools array return the cached compacted version immediately.
    """
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return payload, False, 0, 0

    key = _cache_key(tools, "L1")
    cached = _cache_get(key)
    if cached is not None:
        compacted_tools, before, after = cached
        if after >= before:
            return payload, False, before, after
        updated = copy.deepcopy(payload)
        updated["tools"] = compacted_tools
        return updated, True, before, after

    compacted_tools = compact_tool_schema_value(tools)
    before = _json_byte_len(tools)
    after = _json_byte_len(compacted_tools)
    _cache_put(key, compacted_tools, before, after)

    if after >= before:
        return payload, False, before, after

    updated = copy.deepcopy(payload)
    updated["tools"] = compacted_tools
    return updated, True, before, after


# ---------------------------------------------------------------------------
# Layer 2: description truncation
# ---------------------------------------------------------------------------

_FIRST_SENTENCE_RE = re.compile(r"^(.*?[.!?])(?:\s|$)", re.DOTALL)


def _truncate_description(desc: str, max_chars: int) -> str:
    """Truncate *desc* to *max_chars*, preserving the first complete sentence.

    Strategy:
    - *max_chars* ≤ 0: return *desc* unchanged (feature disabled).
    - Short descriptions (≤ *max_chars*) pass through unchanged.
    - Normalise whitespace before any truncation.
    - If the first sentence fits in *max_chars*, keep it and optionally
      append the second sentence when the combined length ≤ 1.5× *max_chars*.
    - If the first sentence alone exceeds *max_chars*, hard-truncate
      and append ``…``.
    """
    if max_chars <= 0:
        return desc

    # Normalise whitespace first (mirrors Layer 1 behaviour).
    desc = " ".join(desc.split())

    if len(desc) <= max_chars:
        return desc

    m = _FIRST_SENTENCE_RE.match(desc)
    if m and len(m.group(1)) <= max_chars:
        first = m.group(1)
        rest = desc[len(first) :].strip()
        if rest:
            m2 = _FIRST_SENTENCE_RE.match(rest)
            if m2 and len(first) + 1 + len(m2.group(1)) <= int(max_chars * 1.5):
                return f"{first} {m2.group(1)}"
        return first

    # First sentence too long → hard truncation.
    return desc[:max_chars].rstrip() + "…"


def _is_semantic_param_name(name: str) -> bool:
    """Check if a parameter name is self-explanatory."""
    return name.lower().replace("-", "_") in _SEMANTIC_PARAM_NAMES


def _truncate_descriptions_in_schema(
    value: Any,
    max_chars: int,
    strip_semantic: bool = False,
    _parent_key: str | None = None,
    _grandparent_key: str | None = None,
) -> Any:
    """Recursively truncate ``description`` fields in a tool-schema structure.

    When *strip_semantic* is True and the field lives inside
    ``properties.<name>.description`` where *name* is self-explanatory,
    the description is dropped entirely instead of truncated.
    """
    if isinstance(value, list):
        return [_truncate_descriptions_in_schema(item, max_chars, strip_semantic) for item in value]

    if not isinstance(value, dict):
        return value

    compacted: dict[str, Any] = {}
    for key, child in value.items():
        if key == "description" and isinstance(child, str):
            # Layer 3: strip descriptions on self-explanatory params.
            if (
                strip_semantic
                and _grandparent_key == "properties"
                and _parent_key
                and _is_semantic_param_name(_parent_key)
            ):
                # ponytail: drop description on semantic params — the name alone is enough.
                continue
            compacted[key] = _truncate_description(child, max_chars)
        else:
            compacted[key] = _truncate_descriptions_in_schema(
                child,
                max_chars,
                strip_semantic,
                _parent_key=key,
                _grandparent_key=_parent_key,
            )

    return compacted


def compact_tool_descriptions(
    payload: dict[str, Any],
    max_chars: int = 0,
) -> tuple[dict[str, Any], bool, int, int]:
    """Truncate tool descriptions in *payload* to *max_chars*.

    Returns ``(updated_payload, modified, before_bytes, after_bytes)``.
    If *max_chars* is 0 (default) or compaction doesn't reduce size,
    the original payload is returned unchanged.

    When ``HEADROOM_TOOL_DESC_STRIP_SEMANTIC=1``, descriptions on
    self-explanatory parameters (e.g. ``query``, ``owner``) are
    removed entirely instead of truncated.

    Results are cached by tools digest + config.
    """
    if max_chars <= 0:
        return payload, False, 0, 0

    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return payload, False, 0, 0

    strip_sem = strip_semantic_params()
    key = _cache_key(tools, "L2", max_chars, strip_sem)
    cached = _cache_get(key)
    if cached is not None:
        compacted_tools, before, after = cached
        if after >= before:
            return payload, False, before, after
        updated = copy.deepcopy(payload)
        updated["tools"] = compacted_tools
        return updated, True, before, after

    compacted_tools = _truncate_descriptions_in_schema(tools, max_chars, strip_sem)
    before = _json_byte_len(tools)
    after = _json_byte_len(compacted_tools)
    _cache_put(key, compacted_tools, before, after)

    if after >= before:
        return payload, False, before, after

    updated = copy.deepcopy(payload)
    updated["tools"] = compacted_tools
    return updated, True, before, after
