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


def _spans(text: str) -> list[tuple[int, int]]:
    """Deterministic list of ``(start, end)`` for top-level balanced JSON spans.
    Nested spans are not returned separately — the dispatch handles depth."""
    out: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] in _OPEN:
            end = _match_span(text, i)
            if end is not None:
                out.append((i, end))
                i = end
                continue
        i += 1
    return out


def _has_routable_json(span: str) -> bool:
    """True if ``span`` parses and contains an array of objects somewhere — the
    shape the JSON compressors actually act on. Cheap structural check, no size
    threshold."""
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
        if not _has_routable_json(chunk):
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
