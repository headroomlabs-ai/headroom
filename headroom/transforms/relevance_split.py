"""Prompt-conditioned relevance split for KEEP/DROP compression decisions.

Segments tool output into coherent records, scores each against the request's
*information need* (user prompt + the triggering tool call's args) using the
existing :class:`~headroom.relevance.RelevanceScorer` (BM25 / bge-small
embeddings / hybrid), and partitions the content into ordered KEEP/DROP runs.

The split is **mode-agnostic**: this module decides *what* is worth keeping
verbatim vs. what is a low-value tail; the caller applies the disposition. In
lossless (no-CCR) mode the KEEP runs stay byte-verbatim and the DROP tail is
Kompressed marker-free; in CCR mode the same DROP tail can be dropped with a
retrieval marker. Nothing here emits markers or calls a compressor.

Segmentation is boundary-aware, not line-based: blank lines delimit records,
indented continuation lines stay attached to their parent (so stack traces and
pretty-printed blobs are scored as one unit), and dense blank-free streams
(grep, tight logs) are packed into small fixed windows. The partition is
lossless -- ``"".join(segment(content)) == content`` -- so KEEP runs
reconstruct the original bytes exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from headroom.relevance import RelevanceScorer

__all__ = ["adaptive_threshold", "build_relevance_query", "plan_relevance_split", "segment"]


def build_relevance_query(user_query: str, tool_name: str = "", tool_args: str = "") -> str:
    """Compose the information-need query for relevance scoring.

    The user's prompt is the high-level intent; the triggering tool call's args
    (a grep pattern, a read path, a search query) are the *precise*, per-output
    ask and usually the sharpest signal. Both are included so the lexical (BM25)
    half locks onto exact tokens (e.g. the grep pattern) while the semantic half
    tracks the intent.
    """
    parts: list[str] = []
    q = (user_query or "").strip()
    if q:
        parts.append(q)
    call = " ".join(p for p in ((tool_name or "").strip(), (tool_args or "").strip()) if p)
    if call:
        parts.append(call)
    return "\n".join(parts)


def segment(content: str, *, window: int = 8, max_chars: int = 1200) -> list[str]:
    """Partition ``content`` into coherent records.

    Lossless partition: ``"".join(segment(content)) == content``. Blank lines
    delimit records; oversized or dense blank-free blocks are packed into
    windows of at most ``window`` lines / ``max_chars`` chars, with indented
    continuation lines held to their window so multi-line units aren't cut.
    """
    lines = content.splitlines(keepends=True)
    if len(lines) <= 1:
        return [content] if content else []

    # Pass 1: blank-line-delimited blocks (paragraphs / record gaps).
    blocks: list[list[str]] = []
    cur: list[str] = []
    for ln in lines:
        cur.append(ln)
        if ln.strip() == "":
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)

    # Pass 2: pack/window each block. Dense blank-free streams (grep, tight
    # logs) become fixed windows; indented continuation lines stay attached to
    # their window so stack traces / pretty JSON aren't split mid-unit.
    segments: list[str] = []
    for block in blocks:
        if len(block) <= window and sum(len(x) for x in block) <= max_chars:
            segments.append("".join(block))
            continue
        i = 0
        n = len(block)
        while i < n:
            j = min(i + window, n)
            while j < n and block[j][:1] in (" ", "\t"):
                j += 1  # don't cut off an indented continuation run
            segments.append("".join(block[i:j]))
            i = j
    return segments


def _otsu_threshold(values: list[float]) -> float:
    """Otsu's method: the cut between two classes that maximizes between-class
    variance. Parameter-free — the candidate cuts are the data's own values, so
    there's no bin size or magic constant. Returns the midpoint of the winning
    adjacent pair.
    """
    xs = sorted(values)
    n = len(xs)
    total = sum(xs)
    w0 = 0.0
    sum0 = 0.0
    best_t = xs[0]
    best_var = -1.0
    for i in range(n - 1):
        w0 += 1
        sum0 += xs[i]
        w1 = n - w0
        m0 = sum0 / w0
        m1 = (total - sum0) / w1
        between = w0 * w1 * (m0 - m1) ** 2
        if between > best_var:
            best_var = between
            best_t = (xs[i] + xs[i + 1]) / 2.0
    return best_t


def adaptive_threshold(values: list[float], floor: float) -> float:
    """Data-driven KEEP/DROP cut for one output's relevance scores.

    The operative cut is the natural relevant/irrelevant break (Otsu) *for this
    output and query* — so it moves with the score distribution instead of a
    fixed constant. It's floored by ``floor`` so records below the absolute
    minimum relevance are never kept verbatim (an all-irrelevant output drops
    entirely). When every record scores the same there's no break to find, so
    the floor decides (all-in or all-out).
    """
    if len({round(v, 9) for v in values}) < 2:
        return floor
    return max(_otsu_threshold(values), floor)


def plan_relevance_split(
    content: str,
    query: str,
    scorer: RelevanceScorer,
    *,
    threshold: float,
    adaptive: bool = True,
    window: int = 8,
    max_chars: int = 1200,
    max_records: int | None = None,
) -> list[tuple[bool, str]]:
    """Split ``content`` into ordered ``(keep, text)`` runs by relevance to ``query``.

    A record is KEEP when its relevance score clears the cut. With
    ``adaptive`` (default), the cut is the natural relevant/irrelevant break in
    *this* output's score distribution (Otsu), floored by ``threshold`` — so it
    adapts to the content instead of being a fixed constant. With
    ``adaptive=False`` the cut is ``threshold`` exactly. Either way *which*
    records clear it is entirely prompt-driven, so the KEEP fraction ranges
    from 0% to 100% with the content, not a fixed quota. Consecutive
    same-disposition records are merged into runs (order preserved) so the
    caller applies one disposition per run. Returns a single KEEP run -- i.e.
    no split -- when the query is empty, the content is a single record, or it
    segments into more than ``max_records`` records (a latency guard).
    """
    if not query.strip():
        return [(True, content)]
    segs = segment(content, window=window, max_chars=max_chars)
    if len(segs) < 2 or (max_records and len(segs) > max_records):
        return [(True, content)]

    scores = scorer.score_batch(segs, query)
    cut = adaptive_threshold([s.score for s in scores], threshold) if adaptive else threshold
    runs: list[tuple[bool, str]] = []
    for seg, sc in zip(segs, scores):
        keep = sc.score >= cut
        if runs and runs[-1][0] == keep:
            runs[-1] = (keep, runs[-1][1] + seg)
        else:
            runs.append((keep, seg))
    return runs
