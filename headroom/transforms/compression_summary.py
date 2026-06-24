"""Compression summary generator — describes what was dropped.

When content is compressed, the LLM needs to know what it's missing.
Instead of just "[480 items omitted]", we generate a categorical summary:
"[480 items omitted: 150 log entries (3 with errors), 200 test results (12 failures)]"

This helps the LLM decide whether to call headroom_retrieve and what to search for.

Used by:
- SmartCrusher: categorizes dropped JSON items by field values
- CodeCompressor: lists removed function/class names (from AST, language-agnostic)
"""

from __future__ import annotations

import re
from collections import Counter


def summarize_dropped_items(
    all_items: list[dict],
    kept_items: list[dict],
    kept_indices: set[int] | None = None,
    max_categories: int = 5,
    max_notable: int = 3,
) -> str:
    """Generate a categorical summary of items that were dropped.

    Args:
        all_items: The original full list of items.
        kept_items: The items that were kept after compression (used for count).
        kept_indices: Indices of kept items (preferred over identity comparison).
        max_categories: Maximum number of categories to show.
        max_notable: Maximum number of notable items to call out.

    Returns:
        Summary string or empty string if no useful summary can be generated.
    """
    if not all_items or len(kept_items) >= len(all_items):
        return ""

    # Determine which items were dropped
    if kept_indices is not None:
        dropped = [item for i, item in enumerate(all_items) if i not in kept_indices]
    else:
        # Fallback: index-based comparison using JSON equality
        kept_json = {_item_key(item) for item in kept_items}
        dropped = [item for item in all_items if _item_key(item) not in kept_json]

    if not dropped:
        return ""

    # Strategy 1: Categorize by type/status/kind fields
    categories = _categorize_by_fields(dropped)

    # Strategy 2: Find notable items (errors, failures, warnings)
    notable = _find_notable_items(dropped, max_notable)

    # Build summary
    parts = []

    if categories:
        cat_strs = []
        for field_val, count in categories.most_common(max_categories):
            cat_strs.append(f"{count} {field_val}")
        parts.append(", ".join(cat_strs))

    if notable:
        parts.append(f"notable: {'; '.join(notable)}")

    if not parts:
        # Fallback: just describe the data shape
        keys = _common_keys(dropped)
        if keys:
            parts.append(f"fields: {', '.join(keys[:5])}")

    return "; ".join(parts)


def summarize_compressed_code(
    function_bodies: list[tuple[str, str, int]],
    compressed_bodies_count: int,
) -> str:
    """Generate a summary of compressed code sections from AST data.

    Language-agnostic: works with any language tree-sitter supports because
    it reads function signatures directly from the CodeCompressor's AST output.

    Args:
        function_bodies: List of (signature, body, line) from CodeStructure.
        compressed_bodies_count: Number of bodies that were compressed.

    Returns:
        Summary string like "5 bodies compressed: authenticate(), validate_token(), ..."
        or empty string.
    """
    if not function_bodies or compressed_bodies_count == 0:
        return ""

    # Extract short names from signatures
    names = []
    for sig, _body, _line in function_bodies:
        name = _extract_name_from_signature(sig)
        if name:
            names.append(name)

    if not names:
        return f"{compressed_bodies_count} function bodies compressed"

    # Show up to 6 names
    shown = names[:6]
    result = f"{compressed_bodies_count} bodies compressed: {', '.join(shown)}"
    if len(names) > 6:
        result += f" (+{len(names) - 6} more)"
    return result


# ---- Internal helpers ----

# Fields that commonly indicate item category/type
_CATEGORY_FIELDS = (
    "type",
    "status",
    "kind",
    "category",
    "level",
    "severity",
    "state",
    "phase",
    "action",
    "event_type",
    "log_level",
    "result",
    "outcome",
)

# Values that indicate something notable/important
_NOTABLE_PATTERNS = re.compile(
    r"error|fail|critical|warning|exception|crash|timeout|denied|rejected|invalid",
    re.IGNORECASE,
)

# Values that look like URLs or paths (not useful as categories)
_URL_PATTERN = re.compile(r"^https?://|^/[a-z]", re.IGNORECASE)


def _item_key(item: dict) -> str:
    """Create a hashable key for an item (for dropped detection without id())."""
    # Use first few field values as a fingerprint
    parts = []
    for k, v in list(item.items())[:4]:
        parts.append(f"{k}={v}")
    return "|".join(parts)


def _categorize_by_fields(items: list[dict]) -> Counter:
    """Categorize items by their type/status/kind field values."""
    categories: Counter = Counter()

    for item in items:
        categorized = False
        for field in _CATEGORY_FIELDS:
            val = item.get(field)
            if val and isinstance(val, str) and len(val) < 50:
                clean_val = val.replace("\n", " ").replace("\r", "").strip()
                if clean_val:
                    categories[clean_val] += 1
                    categorized = True
                break
        if not categorized:
            # Try to infer from the item's first short string field
            for key, val in item.items():
                if (
                    isinstance(val, str)
                    and 2 < len(val) < 30
                    and key not in ("id", "name", "path", "url", "href", "email")
                    and not _URL_PATTERN.match(val)
                ):
                    clean_val = val.replace("\n", " ").replace("\r", "").strip()
                    categories[f"{key}={clean_val}"] += 1
                    break

    return categories


def _find_notable_items(items: list[dict], max_notable: int) -> list[str]:
    """Find items that contain error/failure/warning indicators."""
    notable = []
    for item in items:
        item_str = str(item)[:500]
        matches = _NOTABLE_PATTERNS.findall(item_str)
        if matches:
            name = item.get("name", item.get("id", item.get("path", "")))
            if name:
                clean_name = str(name).replace("\n", " ").strip()[:50]
                notable.append(f"{clean_name} ({matches[0]})")
            else:
                notable.append(matches[0])
            if len(notable) >= max_notable:
                break
    return notable


def _common_keys(items: list[dict]) -> list[str]:
    """Get the most common keys across items."""
    key_counts: Counter = Counter()
    for item in items[:50]:
        for key in item.keys():
            key_counts[key] += 1
    return [k for k, _ in key_counts.most_common(8)]


def _extract_name_from_signature(sig: str) -> str:
    """Extract the function/method name from a signature string.

    Works for any language because it looks for common patterns:
    - Python: "def authenticate(", "async def fetch("
    - JavaScript: "function authenticate(", "async function fetch("
    - Go: "func (s *Server) HandleRequest("
    - Rust: "fn authenticate("
    - Java/C++: "public void authenticate("
    """
    # Try common function definition patterns
    match = re.search(r"(?:def|func|fn|function)\s+(?:\([^)]*\)\s*)?(\w+)", sig)
    if match:
        return match.group(1) + "()"

    # Try method patterns: "public static void methodName("
    match = re.search(r"(?:public|private|protected|static|async|export)\s+.*?(\w+)\s*\(", sig)
    if match:
        return match.group(1) + "()"

    # Try class patterns
    match = re.search(r"class\s+(\w+)", sig)
    if match:
        return match.group(1)

    # Fallback: last word before (
    match = re.search(r"(\w+)\s*\(", sig)
    if match:
        return match.group(1) + "()"

    return ""
