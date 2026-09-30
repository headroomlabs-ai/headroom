"""Cross-turn (whole-conversation) verbatim de-duplication.

Bash coding agents re-display the same file bytes many times across turns
(``cat foo.py`` -> ``sed -n 75,100p foo.py`` -> ``git diff`` -> ``cat foo.py``
again). Every per-block compressor is blind to this: the redundancy is *across*
blocks. This transform replaces a contiguous span in a later tool output that
already appeared verbatim in an earlier tool output with a compact in-context
pointer to the original.

Two hard invariants, both required for production use:

1. CACHE-SAFETY via *prefix-monotonicity*. Blocks are processed in order and a
   block is only ever matched against content from *strictly earlier* blocks.
   Therefore the rewritten output of blocks ``0..k`` is byte-identical whether
   or not block ``k+1`` exists — appending a turn never mutates an earlier turn,
   so the upstream prompt-cache prefix stays byte-stable. References are
   ABSOLUTE (an earlier block's ordinal), never relative, so a frozen pointer's
   text never changes. :func:`is_prefix_monotonic` asserts this.

2. ACCURACY via *no information leaves the window*. Only spans that are present
   VERBATIM in an earlier block's already-emitted output are back-referenced
   (the "verbatim corpus"), and the earliest occurrence is never rewritten
   (keep-earliest), so the original the pointer names is always physically in
   context. Only large, non-trivial contiguous spans are folded.

Pure stdlib, deterministic, never raises (returns input unchanged on any error).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["DedupBlock", "dedup_blocks", "is_prefix_monotonic"]

# A run must be at least this many lines AND this many chars to be worth a
# pointer. Small dups are left alone (fragmenting context is not worth it) —
# and a larger floor keeps the pointer comfortably shorter than the span it
# replaces, so a fold is always a net byte win.
DEFAULT_MIN_LINES = 3
DEFAULT_MIN_CHARS = 40
# Cap anchor candidates examined per line so a hot line (e.g. ``    return``)
# can't blow up matching. Deterministic: candidates are kept in first-seen order.
MAX_ANCHOR_CANDIDATES = 16

# An UNPADDED leading line-number prefix: ``123:...`` or ``123<TAB>...`` at the
# very start of the line (grep -n / sed -n / ripgrep --heading data rows). We do
# NOT match a padded/right-aligned prefix (``   123<TAB>`` from ``cat -n``): the
# line must start with the digit, so ``number + delta`` re-numbers byte-exactly
# without touching alignment padding, keeping renumbered folds strictly lossless.
#
# ``[1-9]\d*`` (not ``\d+``) is load-bearing: a LEADING-ZERO run (``08:00:01`` in
# a timestamped log, ``007:...``) is not an unpadded grep/sed line number, and
# recovery is ``str(int(number) + delta)``, which drops the zero pad — ``int("08")
# + 1`` renders ``"9"``, not ``"09"``. Matching those would fold them under a
# uniform delta and the round-trip would NOT be byte-exact, breaking the strictly
# lossless promise above. Excluding them keeps such lines non-numbered, so they
# fold only on an EXACT match (delta 0) — the "prefer false negatives" posture.
# Real grep -n / sed -n / rg -n numbers never carry a leading zero, so the
# intended renumber-fold feature is unaffected.
_LINENO_RE = re.compile(r"^([1-9]\d*)(:|\t)(.*)$", re.DOTALL)


def _num_and_key(line: str) -> tuple[int | None, str, str]:
    """Split a leading unpadded line-number.

    Returns ``(number, match_key, content)``:
      * ``number`` — the leading line number (``int``) or ``None`` if absent.
      * ``match_key`` — the line with the leading number stripped (separator
        kept), so two occurrences of the same content at DIFFERENT line numbers
        share a key (e.g. ``92:foo`` and ``94:foo`` -> ``:foo``). Non-numbered
        lines key on themselves, so exact matching is unchanged for them.
      * ``content`` — the text after the separator, for the triviality test.

    This is what makes cross-turn dedup survive an edit that renumbers a file:
    a re-read whose line numbers all shifted by a constant still folds, and the
    shift is carried in the pointer so the original bytes stay recoverable.
    """
    m = _LINENO_RE.match(line)
    if m is None:
        return None, line, line
    return int(m.group(1)), m.group(2) + m.group(3), m.group(3)


@dataclass
class DedupBlock:
    """One tool-output block. ``turn`` is a STABLE absolute ordinal used in the
    pointer text (must not change as the conversation grows). ``protected`` marks
    blocks that must not be rewritten (e.g. carry a cache_control breakpoint) —
    they are still indexed as reference targets."""

    text: str
    turn: int
    protected: bool = False


def _is_trivial(line: str) -> bool:
    """A line too common/short to safely anchor a match on its own."""
    s = line.strip()
    if len(s) < 4:
        return True
    return s in {
        "return",
        "pass",
        "else:",
        "try:",
        "except:",
        "finally:",
        "break",
        "continue",
        "});",
        "})",
        "],",
        "),",
        '"""',
        "'''",
        "...",
    }


def _pointer(span: list[str], ref_turn: int, delta: int = 0) -> str:
    """A one-line, obviously-a-reference marker naming the in-context original.

    Includes a first-line anchor so the model can locate the block it already
    saw. Marker-free of any ``hash=`` retrieval token: recovery is in-context
    (the original is physically present earlier in the same request).

    ``delta`` is the uniform line-number offset of THIS span relative to the
    referenced original (0 for a byte-identical fold). When non-zero the span is
    the same content re-read after an edit shifted its line numbers; the offset
    is stated so the original the pointer names + ``delta`` reconstructs this
    span's exact numbered bytes (recovery stays in-window)."""
    # Compact form (~35c vs the old ~100c): the ~100c pointer made sub-7-line
    # folds net-negative, so the abundant short (2-4 line) re-read repeats were
    # left uncompressed. Trimming the pointer + MIN_LINES=3 lets those pay off
    # (~4.3% -> ~6% lossless on Opus). Still no hash= token (in-context recovery)
    # and keeps a short first-line anchor so the model can locate the original.
    anchor = next((_num_and_key(ln)[2].strip() for ln in span if ln.strip()), "")
    if len(anchor) > 20:
        anchor = anchor[:17] + "..."
    if delta:
        return f"[↑{len(span)}L same as msg {ref_turn} {delta:+d}L: {anchor!r}]"
    return f"[↑{len(span)}L same as msg {ref_turn}: {anchor!r}]"


def _index_lines(
    lines: list[str | None],
    block_pos: int,
    anchor_index: dict[str, list[tuple[int, int]]],
) -> None:
    """Record each non-trivial line's (block_pos, line_idx) as a future anchor.

    Keeps first-seen order and caps the candidate list per line. Only VERBATIM
    (surviving) lines should be passed here — never the lines of a span that was
    replaced by a pointer. Indexed by the line-number-stripped ``match_key`` so a
    later renumbered re-read of the same content finds this anchor."""
    for li, ln in enumerate(lines):
        if ln is None:
            continue
        _, key, content = _num_and_key(ln)
        if _is_trivial(content):
            continue
        bucket = anchor_index.setdefault(key, [])
        if len(bucket) < MAX_ANCHOR_CANDIDATES:
            bucket.append((block_pos, li))


def _longest_match(
    cur: list[str],
    start: int,
    anchor_index: dict[str, list[tuple[int, int]]],
    corpus: list[list[str | None]],
) -> tuple[int, int, int, int] | None:
    """Longest contiguous run in ``cur`` starting at ``start`` whose content
    appears in a single earlier block — allowing a UNIFORM line-number shift.

    Returns ``(length, block_pos, ref_line_idx, delta)`` or ``None``. ``delta``
    is the constant line-number offset of the run vs the referenced original
    (0 for a byte-identical fold). ``corpus[block_pos]`` holds that block's
    VERBATIM lines (``None`` where a span was already folded, which breaks
    contiguity).

    A run extends while each pair shares a ``match_key`` (content modulo the
    leading line number) AND every numbered pair has the SAME numeric offset —
    so an edit that shifts a file's numbers by a constant still folds, but a
    non-uniform change (edit *inside* the span) ends the run at the divergence.
    Non-numbered lines must match exactly (they carry no offset)."""
    _, anchor_key, _ = _num_and_key(cur[start])
    candidates = anchor_index.get(anchor_key)
    if not candidates:
        return None
    best_len = 0
    best_bp = best_li = -1
    best_delta = 0
    for bp, li in candidates:
        block_lines = corpus[bp]
        k = 0
        delta: int | None = None
        while start + k < len(cur) and li + k < len(block_lines):
            ca = cur[start + k]
            cb = block_lines[li + k]
            if cb is None:  # end of a folded block in the corpus — run ends here
                break
            na, ka, _ = _num_and_key(ca)
            nb, kb, _ = _num_and_key(cb)
            if ka != kb:
                break  # different content -> run ends
            if na is not None and nb is not None:
                d = na - nb
                if delta is None:
                    delta = d
                elif delta != d:
                    break  # non-uniform shift (edit inside span) -> run ends
            elif ca != cb:
                break  # non-numbered line must match exactly
            k += 1
        # Deterministic tie-break: longer wins; on ties keep the earliest
        # (smallest block_pos, then line) already held in best_*.
        if k > best_len:
            best_len, best_bp, best_li, best_delta = k, bp, li, (delta or 0)
    if best_len == 0:
        return None
    return best_len, best_bp, best_li, best_delta


def dedup_blocks(
    blocks: list[DedupBlock],
    *,
    min_lines: int = DEFAULT_MIN_LINES,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> tuple[list[DedupBlock], dict]:
    """Rewrite later verbatim spans to in-context pointers. Prefix-monotonic
    (cache-safe) and information-preserving (accuracy-safe). Returns
    (new_blocks, stats). Never raises."""
    stats = {"spans_folded": 0, "lines_removed": 0, "chars_removed": 0, "blocks": len(blocks)}
    try:
        # corpus[i] = verbatim lines of block i's OUTPUT (None where folded).
        corpus: list[list[str | None]] = []
        anchor_index: dict[str, list[tuple[int, int]]] = {}
        out_blocks: list[DedupBlock] = []

        for blk in blocks:
            lines = blk.text.split("\n")

            if blk.protected:
                # Never rewrite; still a valid verbatim reference target.
                verbatim: list[str | None] = list(lines)
                _index_lines(verbatim, len(corpus), anchor_index)
                corpus.append(verbatim)
                out_blocks.append(blk)
                continue

            out: list[str] = []
            verbatim = []
            i = 0
            n = len(lines)
            while i < n:
                m = _longest_match(lines, i, anchor_index, corpus)
                if m is not None and m[0] >= min_lines:
                    span = lines[i : i + m[0]]
                    span_text = "\n".join(span)
                    if len(span_text) >= min_chars:
                        ref_turn = blocks[m[1]].turn
                        ptr = _pointer(span, ref_turn, m[3])
                        out.append(ptr)
                        # Folded span is NOT verbatim in this block's output:
                        # mark None so it can't seed a later contiguous match,
                        # and don't index it (keep-earliest).
                        verbatim.extend([None] * m[0])
                        stats["spans_folded"] += 1
                        stats["lines_removed"] += m[0]
                        stats["chars_removed"] += len(span_text) - len(ptr)
                        i += m[0]
                        continue
                out.append(lines[i])
                verbatim.append(lines[i])
                i += 1

            # Index only the surviving verbatim lines of THIS block (first-seen).
            # None entries (folded spans) are kept in place so positions stay
            # aligned with ``corpus``; _index_lines skips them.
            _index_lines(verbatim, len(corpus), anchor_index)
            corpus.append(verbatim)
            out_blocks.append(DedupBlock(text="\n".join(out), turn=blk.turn, protected=False))

        return out_blocks, stats
    except Exception:  # never break the proxy
        return blocks, {"spans_folded": 0, "lines_removed": 0, "chars_removed": 0, "error": True}


def is_prefix_monotonic(
    blocks: list[DedupBlock],
    *,
    min_lines: int = DEFAULT_MIN_LINES,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> bool:
    """CACHE-SAFETY invariant: for every k, dedup(blocks[:k]) equals dedup(full)
    truncated to its first k blocks. i.e. appending a later turn never changes an
    earlier turn's rewritten bytes, so the prompt-cache prefix stays stable."""
    full, _ = dedup_blocks(blocks, min_lines=min_lines, min_chars=min_chars)
    full_text = [b.text for b in full]
    for k in range(1, len(blocks) + 1):
        partial, _ = dedup_blocks(blocks[:k], min_lines=min_lines, min_chars=min_chars)
        if [b.text for b in partial] != full_text[:k]:
            return False
    return True
