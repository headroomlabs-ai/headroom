"""Structural (recursive) JSON routing for the ContentRouter.

Today the router is *linear*: it splits a block into textual sections and picks
one strategy per section. It never looks *inside* a structure, so JSON embedded
in a larger payload (a ``gh api`` dump, an MCP result, a ``curl | jq`` tail) is
invisible to the JSON compressors — even though, in practice, that embedded shape
is the overwhelming majority of JSON the agent ever sees.

This module adds the missing structural step: find balanced JSON spans at any
offset in a block and route each one through the router's *existing* dispatch,
splicing the result back in place with the surrounding bytes kept exact.

Why this is CCR-safe by construction
-------------------------------------
Each span is handed to the router's own ``_apply_strategy_to_content`` — the same
code path a whole-block JSON already takes — so SmartCrusher / CodeCompressor
register their ``<<ccr:HASH…>>`` retrieval markers exactly as they do today. CCR
is hash-keyed and therefore location-agnostic: a marker resolves whether it sits
at the top of a block or nested inside one. This module never touches the CCR
store; it only relocates where the dispatch is invoked.

Safety invariants (no thresholds — outcome-gated only):
  * A span that already contains a ``<<ccr:`` marker is passed through untouched
    (never re-compressed / re-hashed).
  * Traversal is deterministic (left-to-right, no clocks/rng) so prompt bytes and
    CCR hashes are stable across turns → prefix cache and store both stay stable.
  * A rewrite is kept only if it is strictly smaller in tokens; otherwise the
    original bytes are returned unchanged. No min-size, no max-depth.
"""

from __future__ import annotations

import json
from collections.abc import Callable

_OPEN = "[{"
_CLOSE = "]}"
_PAIR = {"}": "{", "]": "["}

#: A dispatch callback: given a JSON span's text, return the compressed text
#: (which may carry CCR markers) or ``None`` to leave it unchanged.
Dispatch = Callable[[str], "str | None"]


def _match_span(text: str, start: int) -> int | None:
    """Index just past the balanced JSON container opening at ``start`` (honoring
    string/escape rules), or ``None`` if it never balances."""
    stack: list[str] = []
    in_str = esc = False
    for j in range(start, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in _OPEN:
            stack.append(ch)
        elif ch in _CLOSE:
            if not stack or stack[-1] != _PAIR[ch]:
                return None
            stack.pop()
            if not stack:
                return j + 1
    return None


#: Backstop for ``_scan_spans``: characters it may walk, per character of input,
#: before it stops looking for further spans. The memoized walk below is linear
#: on real payloads (source, logs, JSON, prose), so this only bounds inputs built
#: to defeat the memo. Callers that must not miss a span check ``complete``.
_SCAN_BUDGET_PER_CHAR = 4
_SCAN_BUDGET_FLOOR = 4096


def _scan_from(text: str, start: int, known: dict[int, int | None]) -> tuple[int | None, int]:
    """``_match_span(text, start)``, recording every nested container's verdict.

    A walk from ``start`` also decides the result of ``_match_span`` for every
    ``[``/``{`` it pushes outside a string: from that position the walk's string
    state is the same (not in a string), its stack is the suffix above that
    bracket, so it pops, hits a mismatch or runs off the end exactly when this
    walk does. Those verdicts go into ``known`` so the caller never walks the same
    bytes again for them — which is what made unmatched ``{`` in source and logs
    quadratic. Returns ``(end or None, characters walked)``.
    """
    stack: list[int] = []
    in_str = esc = False
    n = len(text)
    for j in range(start, n):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in _OPEN:
            stack.append(j)
        elif ch in _CLOSE:
            if not stack or text[stack[-1]] != _PAIR[ch]:
                for p in stack:
                    known[p] = None
                return None, j - start + 1
            known[stack.pop()] = j + 1
            if not stack:
                return j + 1, j - start + 1
    for p in stack:
        known[p] = None
    return None, n - start


def _scan_spans(text: str, *, include_nested: bool = False) -> tuple[list[tuple[int, int]], bool]:
    """``(spans, complete)``: the top-level balanced spans ``_spans`` returns, and
    whether the walk covered the whole text within its budget.

    Same left-to-right semantics as calling ``_match_span`` at every opening
    bracket, but each bracket's verdict is computed at most once (see
    ``_scan_from``), so a block full of unmatched ``{`` costs one pass instead of
    one pass per brace.
    """
    out: list[tuple[int, int]] = []
    known: dict[int, int | None] = {}
    n = len(text)
    budget = _SCAN_BUDGET_PER_CHAR * n + _SCAN_BUDGET_FLOOR
    spent = 0
    i = 0
    while i < n:
        if text[i] in _OPEN:
            if i in known:
                end = known[i]
            elif spent >= budget:
                return out, False
            else:
                end, walked = _scan_from(text, i, known)
                spent += walked
            if end is not None:
                out.append((i, end))
                i = end
                continue
        i += 1
    if include_nested:
        # The walk already records every nested bracket's verdict. Reuse it
        # in source order without scanning any suffix again.
        out = []
        for start in range(n):
            end = known.get(start)
            if end is not None:
                out.append((start, end))
    return out, True


def _spans(text: str) -> list[tuple[int, int]]:
    """Deterministic list of ``(start, end)`` for top-level balanced JSON spans.
    Nested spans are not returned separately — the dispatch handles depth."""
    return _scan_spans(text)[0]


def json_document_spans(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` of every top-level balanced span of ``text`` that parses
    as a JSON object or array, left to right. Scalars are not documents; a
    whole-document ``text`` yields one span covering it (modulo whitespace)."""
    return scan_json_documents(text)[0]


def scan_json_documents(text: str) -> tuple[list[tuple[int, int]], bool]:
    """``json_document_spans`` plus whether the scan covered all of ``text``.

    ``complete`` is False when the bracket walk or cumulative JSON validation
    exhausts its work budget, or the decoder reaches its recursion limit.
    A caller that must not miss a document treats any incomplete result as
    "may contain one" and declines lossy compression.
    """
    spans, complete = _scan_spans(text, include_nested=True)
    out: list[tuple[int, int]] = []
    covered_until = 0
    parsed_chars = 0
    parse_budget = _SCAN_BUDGET_PER_CHAR * len(text) + _SCAN_BUDGET_FLOOR
    for a, b in spans:
        if a < covered_until:
            continue
        # Invalid wrappers may contain valid JSON. Checking each nested
        # candidate must not turn deeply nested malformed input quadratic.
        if parsed_chars + b - a > parse_budget:
            return out, False
        parsed_chars += b - a
        try:
            parsed = json.loads(text[a:b])
        except RecursionError:
            return out, False
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict | list):
            out.append((a, b))
            covered_until = b
    return out, complete


def carries_record_array(span: str) -> bool:
    """True if ``span`` parses and contains an array of objects somewhere — the
    shape the JSON compressors act on, and the one whose record delimiters a
    prose model can delete while leaving valid JSON behind (#3673). Cheap
    structural check, no size threshold."""
    try:
        v = json.loads(span)
    except (ValueError, TypeError):
        return False

    found = False

    def walk(x: object) -> None:
        nonlocal found
        if found:
            return
        if isinstance(x, list):
            if len(x) >= 2 and sum(isinstance(e, dict) for e in x) >= 0.8 * len(x):
                found = True
                return
            for e in x:
                walk(e)
        elif isinstance(x, dict):
            for e in x.values():
                walk(e)

    walk(v)
    return found


def route_embedded_json(
    content: str,
    dispatch: Dispatch,
    *,
    tok: Callable[[str], int] | None = None,
) -> str | None:
    """Route every embedded JSON span in ``content`` through ``dispatch`` and
    splice the results back in place. Returns the rewritten block, or ``None``
    when nothing safe/smaller applied.

    ``content`` that is itself a single JSON value is intentionally skipped — the
    caller already routes pure-JSON blocks; this exists for the *embedded* case.
    """
    tok = tok or (lambda s: max(1, len(s) // 4))
    spans = _spans(content)
    if not spans:
        return None
    # Whole-block JSON is the caller's job, not ours.
    if len(spans) == 1 and spans[0] == (0, len(content.strip())):
        return None

    repls: list[tuple[int, int, str]] = []
    for a, b in spans:
        chunk = content[a:b]
        if "<<ccr:" in chunk:  # R1: already compressed — never re-route
            continue
        if not carries_record_array(chunk):
            continue
        out = dispatch(chunk)
        if out is None or out == chunk:
            continue
        if tok(out) < tok(chunk):  # benefit gate (outcome, not a threshold)
            repls.append((a, b, out))

    if not repls:
        return None
    parts: list[str] = []
    last = 0
    for a, b, out in repls:
        parts.append(content[last:a])
        parts.append(out)
        last = b
    parts.append(content[last:])
    new = "".join(parts)
    return new if tok(new) < tok(content) else None
