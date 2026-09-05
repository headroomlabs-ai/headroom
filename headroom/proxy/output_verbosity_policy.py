"""Pure output verbosity steering policy."""

from __future__ import annotations

# Sentinel prefix marks the steering block so application is idempotent and
# the block is recognizable in logs/diffs.
STEERING_SENTINEL = "<headroom_output_shaping>"
STEERING_SUFFIX = "</headroom_output_shaping>"

# Levels are cumulative: each includes everything above it. Text must stay
# byte-stable across releases for prefix-cache friendliness; edits to these
# strings are cache-busting changes.
#
# The "cite the exact file path and line" clause is an OBLIGATION, not a
# suggestion, because the loose form ("reference by path and line") was
# measured to cost turns.
#
# Measured on SWE-bench tasks run to completion, with a blind judge playing the
# engineer who must apply the fix and asking a follow-up whenever blocked. Of
# the four blocker categories, only one moved between L0 and L3 -- and notably
# it was NOT the one predicted. "Asked why" scored 0 at both levels, so
# "omit rationale unless the user asks why" does not generate follow-ups.
# What moved was "under-specified location", 0 -> 3 across 9 paired Claude
# tasks: told to stop restating code, models wrote "change the check in the
# mixin" and the reader had to ask which file.
#
# Tightening this one clause fixed it -- 3 -> 0 on a re-run, with turn
# inflation going +23.1% -> 0.0% and L3 needing no more rounds than L0. The
# lesson generalises: suppressing content is safe, suppressing the POINTER to
# that content is not, because the reader buys it back at the price of a whole
# turn. Sample is small (n=9, Claude only); the mechanism is better evidenced
# than the magnitude.
#
# L3 and L4 carry two rules L1/L2 do not, because only they instruct the model
# to drop content rather than ceremony:
#
#   * A completeness floor. "Omit rationale" and "fragments fine" reward any
#     cut that shortens the answer, including cuts that change it -- the
#     dangerous case is a dropped negation, since losing "not" from "does not
#     retry" inverts the meaning while making it shorter. The floor names what
#     brevity may not touch: content the task needs to be correct.
#   * A clarity exception. Without it these levels are tersest precisely where
#     terseness is most costly: destructive actions, security warnings, and
#     multi-step sequences.
#
# The floor is phrased as a prohibition on dropping content, NOT as the earlier
# "reproduce code, error strings, numbers ... exactly". That wording read as a
# licence to emit code verbatim: measured against the same SWE-bench turn it
# left gpt-5.2 unchanged but inflated gpt-5.1 by 58% (1631 -> 2573 output
# tokens), cutting L3's saving on that model from 51% to 23%. Constrain what
# may be lost; do not instruct the model to reproduce anything.
#
# L1/L2 only forbid preamble and restating context, so they carry neither --
# every token in this block is paid on every request of every conversation.
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
        "diffs, or tool output already in this conversation — cite the exact "
        "file path and line or symbol instead, always; a reference that omits "
        "the location is not a reference. Give conclusions only; omit "
        "rationale unless the user asks why. Prefer the smallest edit over "
        "rewriting whole files. Keep prose to the minimum needed to be "
        "unambiguous. Never drop anything the turn or task needs to be "
        "correct, including negations (not, never, "
        "no, only, except) — shorten how you say it, not what you say. Use "
        "full prose for destructive or irreversible actions, security "
        "warnings, and any multi-step sequence where brevity would create "
        "ambiguity."
    ),
    4: (
        "Minimum tokens. Fragments fine. No preamble, no postamble, no "
        "restating context, no rationale. Answer, smallest-possible edits, "
        "nothing else. Never drop anything the turn or task needs to be "
        "correct, including negations (not, never, no, only, except). Use "
        "full prose for destructive or irreversible actions, security "
        "warnings, and any multi-step sequence where brevity would create "
        "ambiguity."
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
