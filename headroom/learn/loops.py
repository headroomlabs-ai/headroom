"""Loop detection for Headroom Learn — find repeated tool-call patterns.

A *loop* is the single highest-value pattern for `headroom learn` to catch,
because its token waste scales with the number of repetitions rather than
being a one-time cost. Two loop shapes matter:

1. **Error loops** — the same call fails, the agent retries, it fails again
   (e.g. a wrong path read N times). Every repetition is pure waste.

2. **RTK re-fetch loops** — RTK (Realtime Token Kompress) rewrites a shell
   command to truncate its output (``grep foo`` → ``grep foo | head -50``).
   When the truncation drops what the agent needed, the agent re-runs a
   *variant* of the same command to fetch more (``head -100``, a new offset,
   a narrower pattern). Each call succeeds (``is_error=False``) but returns
   insufficient output, so the loop is invisible to failure-only analysis.
   See ``docs/rtk-architecture.md`` for why RTK truncates commands.

This module collapses such variants to a canonical signature, counts the
repetitions, and measures the wasted tokens so the analyzer can (a) surface
loops to the LLM and (b) weight loop-derived recommendations above one-offs.
The analyzer historically ranked recommendations purely by an LLM-guessed
``estimated_tokens_saved`` with a flat confidence — loops had no special
weight at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import Recommendation, SessionData, ToolCall

# Minimum repetitions of one signature before it counts as a loop. Three is the
# smallest count that distinguishes a loop ("again, and again") from a one-off
# retry ("that failed once, try once more") — matching the analyzer's existing
# "2+ occurrences or explicit user direction" evidence bar but one stricter so
# a single retry is not mislabeled a loop.
DEFAULT_MIN_OCCURRENCES = 3

# Rough bytes-per-token used to convert measured output sizes into a token
# estimate. The analyzer's digest builder uses the same 4:1 approximation.
_BYTES_PER_TOKEN = 4

# Pagination / output-limiting fragments that vary between RTK re-fetch
# attempts but do NOT change which command is being run. Stripping these is
# what collapses ``grep foo | head -50`` and ``grep foo | head -100`` to one
# signature. Order-independent: applied as a global substitution.
_PAGINATION_PATTERNS = [
    r"\|\s*head\s+-n?\s*\d+",  # | head -50, | head -n 50
    r"\|\s*tail\s+-n?\s*\d+",  # | tail -50
    r"-n\s*\d+",  # -n 50 (git log -n 50, grep -n is rare but harmless here)
    r"--max-count[= ]\d+",  # grep --max-count=50
    r"--lines[= ]\d+",
    r"\bhead\s+-\d+",  # head -50
    r"\b(limit|offset)[= ]\d+",  # LIMIT 50 / offset=100 (sql-ish)
    r"\bLIMIT\s+\d+",
    r"\bOFFSET\s+\d+",
]
_PAGINATION_RE = re.compile("|".join(_PAGINATION_PATTERNS), re.IGNORECASE)

# Collapse any remaining bare integers so e.g. line numbers / byte offsets in
# otherwise identical commands do not split a loop into singletons.
_INT_RE = re.compile(r"\b\d+\b")
_WS_RE = re.compile(r"\s+")


@dataclass
class LoopPattern:
    """A repeated tool-call pattern detected within a session.

    ``wasted_tokens`` is a *measured* lower bound (from real output sizes),
    not an LLM guess — for an N-occurrence loop it counts the N-1 redundant
    repetitions, since the first call is legitimate work.
    """

    tool: str
    signature: str  # Canonical, variant-collapsed signature
    sample_input: str  # A human-readable example of the looped call
    count: int
    is_error_loop: bool
    wasted_tokens: int
    msg_indices: list[int] = field(default_factory=list)

    @property
    def kind(self) -> str:
        return "error-loop" if self.is_error_loop else "rtk-refetch-loop"


def _canonical_signature(tc: ToolCall) -> str:
    """Collapse a tool call to a signature stable across re-fetch variants.

    For shell commands this strips pagination/limit fragments and bare
    integers so RTK truncation variants of the same command map together.
    For other tools the input summary is normalized on whitespace only.
    """
    raw = tc.input_summary.strip()
    if tc.name.lower() in ("bash", "shell"):
        raw = _PAGINATION_RE.sub(" ", raw)
        raw = _INT_RE.sub("N", raw)
    raw = _WS_RE.sub(" ", raw).strip().lower()
    return f"{tc.name.lower()}::{raw}"


def _tokens(tc: ToolCall) -> int:
    """Token estimate for a single call's output."""
    nbytes = tc.output_bytes or len(tc.output)
    return nbytes // _BYTES_PER_TOKEN


def detect_loops(
    sessions: list[SessionData],
    *,
    min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
) -> list[LoopPattern]:
    """Detect repeated tool-call patterns across sessions.

    Calls are grouped by canonical signature *within each session* (a loop is
    a within-conversation phenomenon; the same command in two unrelated
    sessions is not a loop). Groups meeting ``min_occurrences`` become
    ``LoopPattern`` results, sorted by measured wasted tokens descending.
    """
    groups: dict[str, list[ToolCall]] = {}
    for session in sessions:
        per_session: dict[str, list[ToolCall]] = {}
        for tc in session.tool_calls:
            per_session.setdefault(_canonical_signature(tc), []).append(tc)
        # Merge each session's qualifying groups into the global view keyed by
        # signature so cross-session recurrence of the SAME loop accumulates.
        for sig, calls in per_session.items():
            if len(calls) >= min_occurrences:
                groups.setdefault(sig, []).extend(calls)

    loops: list[LoopPattern] = []
    for sig, calls in groups.items():
        count = len(calls)
        is_error_loop = sum(1 for c in calls if c.is_error) >= (count / 2)
        if is_error_loop:
            # Every repetition of a failing call is waste — including the first,
            # since with upfront knowledge it would never have run.
            wasted = sum(_tokens(c) for c in calls)
        else:
            # Re-fetch loop: the first call is legitimate; the N-1 follow-ups
            # are the redundant re-fetches RTK truncation provoked.
            per_call = sorted((_tokens(c) for c in calls), reverse=True)
            wasted = sum(per_call[1:])
        loops.append(
            LoopPattern(
                tool=calls[0].name,
                signature=sig,
                sample_input=calls[0].input_summary[:120],
                count=count,
                is_error_loop=is_error_loop,
                wasted_tokens=wasted,
                msg_indices=sorted(c.msg_index for c in calls),
            )
        )

    loops.sort(key=lambda lp: lp.wasted_tokens, reverse=True)
    return loops


def format_loops_for_digest(loops: list[LoopPattern]) -> str:
    """Render detected loops as a high-priority digest section for the LLM.

    Returns "" when there are no loops so the digest is unchanged in the
    common case.
    """
    if not loops:
        return ""
    lines = [
        "=== Detected Loops (HIGHEST PRIORITY) ===",
        (
            "These tool-call patterns REPEATED within a session — the most "
            "expensive kind of waste, since cost scales with repetition. A rule "
            "that prevents a loop is worth far more than one that prevents a "
            "one-off error. Emit a guardrail for EACH loop below and set its "
            "estimated_tokens_saved to at least the measured wasted tokens shown."
        ),
        "",
    ]
    for lp in loops:
        lines.append(
            f'- [{lp.kind}] {lp.tool}: "{lp.sample_input}" '
            f"repeated {lp.count}x, ~{lp.wasted_tokens:,} tokens wasted "
            f"(messages {lp.msg_indices})"
        )
    lines.append("")
    return "\n".join(lines)


def _signature_tokens(signature: str) -> set[str]:
    """Word tokens from a canonical signature, for fuzzy rule matching."""
    body = signature.split("::", 1)[-1]
    return {t for t in re.split(r"[^a-z0-9]+", body) if len(t) > 2}


def apply_loop_weighting(recommendations: list[Recommendation], loops: list[LoopPattern]) -> None:
    """Boost recommendations that address a detected loop, in place.

    The analyzer ranks recommendations by ``estimated_tokens_saved`` (an LLM
    guess). For a recommendation whose text overlaps a detected loop's
    signature, we raise that figure to at least the loop's *measured* wasted
    tokens and tag it as loop-derived. Because measured loop waste aggregates
    many repetitions, this reliably lifts loop guardrails above one-off rules
    without trusting the LLM to have weighted them correctly.
    """
    if not loops:
        return
    for rec in recommendations:
        haystack = f"{rec.section} {rec.content}".lower()
        best: LoopPattern | None = None
        for lp in loops:
            sig_tokens = _signature_tokens(lp.signature)
            if not sig_tokens:
                continue
            overlap = sum(1 for t in sig_tokens if t in haystack)
            # Require a majority of the signature's salient tokens to appear so
            # we don't over-credit a generic rule.
            if overlap >= max(1, (len(sig_tokens) + 1) // 2):
                if best is None or lp.wasted_tokens > best.wasted_tokens:
                    best = lp
        if best is not None:
            rec.estimated_tokens_saved = max(rec.estimated_tokens_saved, best.wasted_tokens)
            rec.is_loop_guardrail = True
            rec.loop_occurrences = best.count
