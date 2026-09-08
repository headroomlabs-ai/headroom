"""Tool-schema savings attribution policy for proxy stats.

Headroom saves input tokens in two accounting shapes, and the split is not a
style choice — it follows from what each transform can observe:

* **Compaction** (``*:tool_schema_compaction``, ``*:tool_desc_compaction``)
  rewrites the tool array in place, so both endpoints are countable. Handlers
  fold the delta into ``original_tokens``/``optimized_tokens``, which keeps
  ``tok_before - tok_after == tok_saved`` coherent in the PERF line. It is
  therefore ALREADY inside ``tokens_saved`` and must never be added again.
* **Deferral / hook shrink** (tool search, turn hooks) removes schemas that
  ``count_messages`` never saw, so it cannot move ``original_tokens``. It is
  recorded in per-request tags and is ADDITIVE to ``tokens_saved``.

The one rule a caller needs: the headline is
``tokens_saved + tool_schema_saved_from_tags(tags)``. Use
:func:`headline_tokens_saved` rather than open-coding it — three surfaces had
drifted inline copies of that sum, and two harnesses were silently dropping
their compaction savings entirely because the convention was never written down.

Adding a new tool-schema-shrinking feature? If it moves the tool array, fold it
in the handler like the compaction sites do. If it defers schemas, add its tag
name to :data:`TOOL_SCHEMA_SAVINGS_TAGS` and every surface picks it up.
"""

from __future__ import annotations

TOOL_SCHEMA_SAVINGS_TAGS: tuple[str, ...] = (
    "tool_search_deferred_tokens",
    "turn_hook_tools_saved_tokens",
)


def tool_schema_saved_from_tags(tags: object) -> int:
    """Return tool-definition tokens Headroom kept out of context for one request.

    The summed tags are set only on paths where Headroom performed the deferral,
    so clients that already had tool search enabled contribute zero here.

    These tags are additive to ``tokens_saved`` — see the module docstring.
    """
    if not isinstance(tags, dict):
        return 0

    total = 0
    for key in TOOL_SCHEMA_SAVINGS_TAGS:
        try:
            total += int(tags.get(key, 0) or 0)
        except (TypeError, ValueError):
            continue
    return total


def headline_tokens_saved(tokens_saved: object, tags: object) -> int:
    """Return the single "Tokens saved" figure for one request.

    This is the only correct total to show a user: message compression plus the
    tool-definition tokens that never entered the context. Every reporting
    surface (PERF line, ``headroom perf``, ``/api/stats``, dashboard, session
    summary) must route through here so they cannot disagree.

    Clamped at zero: handlers already revert any inflation before forwarding, so
    a negative is a token-count artifact that never reached the model.
    """
    base = 0
    if isinstance(tokens_saved, (int, float, str)) and not isinstance(tokens_saved, bool):
        try:
            base = int(tokens_saved)
        except (TypeError, ValueError):
            base = 0
    return max(0, base + tool_schema_saved_from_tags(tags))
