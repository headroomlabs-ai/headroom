"""Pure output verbosity steering policy."""

from __future__ import annotations

# Sentinel prefix marks the steering block so application is idempotent and
# the block is recognizable in logs/diffs.
STEERING_SENTINEL = "<headroom_output_shaping>"
STEERING_SUFFIX = "</headroom_output_shaping>"

# Levels are cumulative in intent; L4 is literally L3 plus two clauses. Text must stay
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
# L3 and L4 (which inherits them through ``_L3``) carry two rules L1/L2 do
# not, because only they instruct the model to drop content rather than
# ceremony:
#
#   * A completeness floor. "Omit rationale" (and the old L4's "fragments
#     fine") reward any cut that shortens the answer, including cuts that
#     change it -- the
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
#
# L4 is L3 plus a fixed final-report shape and a batching instruction. It
# replaced an earlier "Minimum tokens. Fragments fine. ... no rationale ...
# nothing else" text. Measured with Claude Code 2.1.325 on Opus 5.5 against
# vanilla Claude Code, 49 SWE-bench Verified tasks x 2 attempts, paired:
#
#   level      output   cost    solved (vs vanilla)  judge: reply needs a follow-up
#   old L4     -21.2%   -16.5%  44.5 vs 45 of 49     20% vs 3% vanilla (dropped caveats)
#   L3         -11.7%    -9.5%  44.5 vs 45 of 49     14% vs 11%
#   L4 (this)  -20.1%   -15.5%  43 vs 44 of 48       28% vs 9% (dropped root cause)
#
# The L4 row has 48 tasks because one task's second attempt was still running
# when it was frozen. The judge rates each reply against its vanilla twin, so
# each row's vanilla rate comes from its own run: compare within a row only.
#
# The trade-off is deliberate: this text keeps the old L4's saving and its
# replies say what changed, how it was verified and what is risky, but the
# report shape has no slot for the root cause, so "why" is the first thing
# lost. The batching sentence did not measurably change batching (0.90 tool
# calls per API call vs 0.92 vanilla); it stays because this is the text
# that was measured. A change to it needs a re-run, not an edit.
_L3 = (
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
)

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
    3: _L3,
    4: (
        _L3 + " Final message: at most three short lines: (1) what changed, "
        "with exact file and symbol; (2) how it was verified and the result; "
        "(3) any behaviour change, risk, or limitation, or say there is none. "
        "Batch independent reads and searches: issue them as parallel tool "
        "calls in one turn, or chain them into one shell command."
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
