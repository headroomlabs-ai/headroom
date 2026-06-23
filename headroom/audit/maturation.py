"""Simulate read maturation (Mechanism B) against local transcripts.

Answers, from real traffic, the questions that size Mechanism B's risk
and tune its `quiesce_turns` policy:

- How often is the same file re-read at all (and how often as a partial
  range)? Partial re-reads happening *despite* full content in context
  are evidence the model's natural recovery is already "go back to disk".
- What share of maturation-eligible reads is never touched again
  (pure savings, zero recovery events)?
- For files that are touched again, how long until the next touch
  (the quiesce-window coverage table)?
- Activity-based at-risk edits: edits landing on a file that had been
  quiet longer than N turns — the moments a matured read would force a
  re-read under the activity policy.

Findings on the development corpus (2026-06-10, 81 sessions): 35.5% of
reads are re-reads (95% of those partial); 60.7% of big reads are never
touched again; next-touch p50 is 4 turns — hence quiesce_turns=5.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

MATURE_FLOOR = 2048  # ReadMaturationConfig.min_size_bytes
QUIESCE_CANDIDATES = [1, 2, 3, 5, 10, 25]
_MUTATING = ("Edit", "Write", "MultiEdit", "NotebookEdit")


@dataclass
class MaturationSimReport:
    """Aggregated simulation results."""

    read_calls: int = 0
    rereads_any: int = 0
    rereads_partial: int = 0
    big_reads: int = 0
    big_read_bytes: int = 0
    never_touched_again: int = 0
    next_touch_p50: int = 0
    next_touch_p90: int = 0
    next_touch_p95: int = 0
    # quiesce N -> % of touched-again reads whose next touch is within N
    next_touch_within: dict[int, float] = field(default_factory=dict)
    edits_with_prior_read: int = 0
    edits_without_prior_read: int = 0
    # quiesce N -> edits whose file was quiet > N turns when edited
    # (the matured-read moments under the activity policy)
    at_risk_edits: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _block_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def simulate_maturation(root: Path) -> MaturationSimReport:
    """Run the maturation simulation over ``root/**/*.jsonl``."""
    r = MaturationSimReport()
    next_touch_gaps: list[int] = []
    prev_touch_gaps: list[int] = []  # edit -> previous touch of same file
    at_risk = dict.fromkeys(QUIESCE_CANDIDATES, 0)

    for path in sorted(root.glob("**/*.jsonl")):
        tool_meta: dict[str, tuple[str, dict]] = {}
        timeline: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
        session_reads: list[tuple[str, int, int]] = []
        seen_files: set[str] = set()
        a_idx = 0
        try:
            with path.open(errors="replace") as f:
                for raw in f:
                    try:
                        line = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    msg = line.get("message") or {}
                    role, content = msg.get("role"), msg.get("content")
                    if role == "assistant" and isinstance(content, list):
                        a_idx += 1
                        for b in content:
                            if isinstance(b, dict) and b.get("type") == "tool_use":
                                name = b.get("name", "")
                                inp = b.get("input") or {}
                                tool_meta[b.get("id", "")] = (name, inp)
                                fp = inp.get("file_path") or inp.get("path") or ""
                                if name in _MUTATING and fp:
                                    timeline[fp].append((a_idx, "edit", 0))
                    if role == "user" and isinstance(content, list):
                        for b in content:
                            if not (isinstance(b, dict) and b.get("type") == "tool_result"):
                                continue
                            name, inp = tool_meta.get(b.get("tool_use_id", ""), ("", {}))
                            if name != "Read":
                                continue
                            fp = inp.get("file_path") or inp.get("path") or ""
                            if not fp:
                                continue
                            text = _block_text(b.get("content"))
                            size = len(text.encode("utf-8", errors="replace"))
                            partial = inp.get("offset") is not None or inp.get("limit") is not None
                            r.read_calls += 1
                            if fp in seen_files:
                                r.rereads_any += 1
                                if partial:
                                    r.rereads_partial += 1
                            seen_files.add(fp)
                            timeline[fp].append((a_idx, "read", size))
                            session_reads.append((fp, a_idx, size))
        except OSError:
            continue

        for ops in timeline.values():
            ops.sort(key=lambda t: t[0])
            for turn, kind, _size in ops:
                if kind != "edit":
                    continue
                prev = [t for t, _, _ in ops if t < turn]
                had_read = any(k == "read" and t <= turn for t, k, _ in ops)
                if not had_read:
                    r.edits_without_prior_read += 1
                    continue
                r.edits_with_prior_read += 1
                if prev:
                    gap = turn - max(prev)
                    prev_touch_gaps.append(gap)
                    for n in QUIESCE_CANDIDATES:
                        if gap > n:
                            at_risk[n] += 1

        for fp, rturn, size in session_reads:
            if size < MATURE_FLOOR:
                continue
            r.big_reads += 1
            r.big_read_bytes += size
            later = [t for t, _, _ in timeline[fp] if t > rturn]
            if later:
                next_touch_gaps.append(min(later) - rturn)
            else:
                r.never_touched_again += 1

    def pct(xs: list[int], p: float) -> int:
        return sorted(xs)[int(len(xs) * p)] if xs else 0

    r.next_touch_p50 = pct(next_touch_gaps, 0.5)
    r.next_touch_p90 = pct(next_touch_gaps, 0.9)
    r.next_touch_p95 = pct(next_touch_gaps, 0.95)
    if next_touch_gaps:
        r.next_touch_within = {
            n: round(100 * sum(1 for g in next_touch_gaps if g <= n) / len(next_touch_gaps), 1)
            for n in QUIESCE_CANDIDATES
        }
    r.at_risk_edits = at_risk
    return r


def render_sim_text(r: MaturationSimReport) -> str:
    """Human-readable simulation summary."""
    out: list[str] = []
    out.append("── maturation simulation (Mechanism B) ──")
    out.append(
        f"  re-reads: {r.rereads_any}/{r.read_calls} reads target an already-read file "
        f"({100 * r.rereads_any / max(r.read_calls, 1):.1f}%); "
        f"{r.rereads_partial} of those are partial ranges"
    )
    out.append(
        f"  big reads (≥{MATURE_FLOOR}B): {r.big_reads} "
        f"({r.big_read_bytes / 1e6:.1f}MB); never touched again: "
        f"{r.never_touched_again} ({100 * r.never_touched_again / max(r.big_reads, 1):.1f}%) "
        f"← pure savings"
    )
    out.append(
        f"  next-touch gap for the rest (turns): p50={r.next_touch_p50} "
        f"p90={r.next_touch_p90} p95={r.next_touch_p95}"
    )
    if r.next_touch_within:
        for n, share in r.next_touch_within.items():
            out.append(f"    next touch within {n:>2} turn(s): {share:.1f}%")
    total_edits = r.edits_with_prior_read + r.edits_without_prior_read
    out.append(
        f"  edits: {r.edits_with_prior_read} with a prior read of the file, "
        f"{r.edits_without_prior_read} without"
    )
    out.append("  activity-based at-risk edits (file quiet > N turns when edited):")
    for n, count in r.at_risk_edits.items():
        out.append(
            f"    quiesce {n:>2}: {count:>5} edits ({100 * count / max(total_edits, 1):.1f}%)"
        )
    return "\n".join(out)
