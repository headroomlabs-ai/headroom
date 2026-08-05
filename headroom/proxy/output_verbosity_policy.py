"""Pure output verbosity steering policy."""

from __future__ import annotations

# Sentinel prefix marks the steering block so application is idempotent and
# the block is recognizable in logs/diffs.
STEERING_SENTINEL = "<headroom_output_shaping>"
STEERING_SUFFIX = "</headroom_output_shaping>"

# Levels are cumulative: each includes everything above it. Text must stay
# byte-stable across releases for prefix-cache friendliness; edits to these
# strings are cache-busting changes.
VERBOSITY_LEVELS = {
    1: (
        "Skip preamble and postamble. Do not announce what you are about to "
        "do or recap what you just did; start with the substance."
    ),
    2: (
        "Skip preamble and postamble; start with the substance. Never restate "
        "code, file contents, diffs, or tool output that already appear in "
        "this conversation — reference them by path and line instead. After a "
        "tool call succeeds, continue without narrating the result."
    ),
    3: (
        "Skip preamble and postamble. Never restate code, file contents, "
        "diffs, or tool output already in this conversation — reference by "
        "path and line. Give conclusions only; omit rationale unless the user "
        "asks why. Prefer the smallest edit over rewriting whole files. Keep "
        "prose to the minimum needed to be unambiguous."
    ),
    4: (
        "Minimum tokens. Fragments fine. No preamble, no postamble, no "
        "restating context, no rationale. Answer, smallest-possible edits, "
        "nothing else."
    ),
}


def steering_text(level: int) -> str | None:
    """The full steering block for a verbosity level, or ``None`` for level 0."""
    text = VERBOSITY_LEVELS.get(level)
    if text is None:
        return None
    return f"{STEERING_SENTINEL}\n{text}\n{STEERING_SUFFIX}"


def replace_or_append_steering_block(existing: str, block: str) -> tuple[str, bool]:
    """Replace an existing steering block in text, or append one at the tail."""
    start = existing.find(STEERING_SENTINEL)
    if start >= 0:
        end = existing.find(STEERING_SUFFIX, start)
        end = len(existing) if end < 0 else end + len(STEERING_SUFFIX)
        prefix = existing[:start].rstrip()
        suffix = existing[end:].lstrip("\n")
        parts = [part for part in (prefix, block, suffix) if part]
        updated = "\n\n".join(parts)
        return updated, updated != existing

    updated = f"{existing.rstrip()}\n\n{block}" if existing.strip() else block
    return updated, updated != existing
