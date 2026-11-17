"""Pure mixed-content parsing helpers for the content router."""

from __future__ import annotations

import json
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from .content_detector import ContentType, _is_search_result_line
from .lossless_compaction import _TIMESTAMP_ROW_RE


@dataclass
class ContentSection:
    """A typed section of content."""

    content: str
    content_type: ContentType
    language: str | None = None
    start_line: int = 0
    end_line: int = 0
    is_code_fence: bool = False
    # Never merged into a neighbor by the post-pass coalescer. Set on
    # tag-protection placeholder lines (merging would drag prose into their
    # compression exemption) and on bracket-balanced-but-invalid-JSON blocks
    # (kept standalone so a short prose banner meets the compressors' size
    # floors on its own instead of riding a larger merged section into a
    # lossy pass).
    atomic: bool = False


# Group 1 is the whole info string: any text without a backtick, so
# ``c++``, ``c#``, ``objective-c``, ``js {1,3}`` and `` c++`` (space first) all
# open a fence. ``\w*`` missed such openers and then took the closing fence
# for one. A single quantifier keeps a long line linear; the language is the
# info string's first word (see ``_fence_language``).
_CODE_FENCE_PATTERN = re.compile(r"^```([^\n`]*)$", re.MULTILINE)
# ``[^\S\n]*``, not ``\s*``: under MULTILINE, ``^\s*`` rescans every
# following blank line from each line start (quadratic in a blank run).
_JSON_BLOCK_START = re.compile(r"^[^\S\n]*[\[{]", re.MULTILINE)
_SEARCH_RESULT_PATTERN = re.compile(r"^\S+:\d+:", re.MULTILINE)
_PROSE_PATTERN = re.compile(r"[A-Z][a-z]+\s+\w+\s+\w+")


def _fence_language(info_string: str) -> str:
    """First word of a fence info string, or ``"unknown"`` for a bare fence."""
    words = info_string.split()
    return words[0] if words else "unknown"


def _is_search_line(line: str) -> bool:
    """True for grep match lines and ``-A``/``-B``/``-C`` context lines.

    Unions the legacy splitter pattern (kept so every previously carved line
    still carves) with the detector's predicate, which additionally claims
    both context-line shapes (``path-NN-content`` and ``path:NN-content``).
    Sharing it keeps the splitter carving the same lines the detector claims
    (#3580); otherwise code in context lines would strand in PLAIN_TEXT
    sections and reach the prose compressor.

    Timestamped log rows (``2026-09-23T10:05:07Z ...``) match the legacy
    pattern but are never grep output; the detector rejects them (#3736), so
    the splitter must too or it hands them to the lossy search compressor.
    """
    if _TIMESTAMP_ROW_RE.match(line):
        return False
    return bool(_SEARCH_RESULT_PATTERN.match(line)) or _is_search_result_line(line)


def is_mixed_content(content: str) -> bool:
    """Detect if content contains multiple distinct content types."""
    return sum(mixed_content_indicators(content).values()) >= 2


def mixed_content_indicators(content: str) -> dict[str, bool]:
    """Return the individual signals used to classify mixed content."""
    return {
        "has_code_fences": bool(_CODE_FENCE_PATTERN.search(content)),
        "has_json_blocks": bool(_JSON_BLOCK_START.search(content)),
        "has_embedded_json_with_text": _has_valid_json_block_with_text(content),
        "has_prose": len(_PROSE_PATTERN.findall(content)) > 5,
        "has_search_results": bool(_SEARCH_RESULT_PATTERN.search(content))
        or any(_is_search_result_line(line) for line in content.split("\n")),
    }


def _any_nonblank(lines: list[str], start: int, stop: int) -> bool:
    """True when some line in [start, stop) has non-whitespace.

    Equivalent to ``bool("\n".join(lines[start:stop]).strip())`` — a join of
    lines is blank exactly when every line is blank — but it short-circuits
    instead of building a copy of the whole body for each candidate.
    """
    return any(lines[i].strip() for i in range(start, stop))


def _has_valid_json_block_with_text(content: str) -> bool:
    """Return true when prose or log text wraps a valid JSON block."""
    lines = content.split("\n")
    blocks = _JsonBlockScanner(lines)

    for index, line in enumerate(lines):
        if not _is_json_candidate(line):
            continue

        json_content, end_index = blocks.extract(index)
        if json_content is None:
            continue

        try:
            json.loads(json_content)
        except (TypeError, ValueError):
            continue

        if _any_nonblank(lines, 0, index) or _any_nonblank(lines, end_index + 1, len(lines)):
            return True

    return False


def split_into_sections(content: str, *, isolate: tuple[str, ...] = ()) -> list[ContentSection]:
    """Parse mixed content into typed sections.

    ``isolate`` lists substrings (the router's tag-protection placeholders)
    whose lines must each become their OWN section: the router exempts any
    section carrying a placeholder from compression, so a placeholder that
    shares a section with ordinary prose would drag that prose into verbatim
    passthrough. Historically placeholders self-isolated by accident — a
    ``{{HEADROOM_TAG_N}}`` line bracket-balances, so the pre-validation
    splitter typed it JSON_ARRAY; now that JSON typing is validated, the
    isolation must be explicit.
    """
    sections: list[ContentSection] = []
    lines = content.split("\n")

    def _carries_isolate(text: str) -> bool:
        return any(marker in text for marker in isolate)

    blocks = _JsonBlockScanner(lines)

    i = 0
    while i < len(lines):
        line = lines[i]

        if isolate and _carries_isolate(line):
            sections.append(
                ContentSection(
                    content=line,
                    content_type=ContentType.PLAIN_TEXT,
                    start_line=i,
                    end_line=i,
                    atomic=True,
                )
            )
            i += 1
            continue

        if match := _CODE_FENCE_PATTERN.match(line):
            language = _fence_language(match.group(1))
            code_lines = []
            start_line = i
            i += 1

            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1

            sections.append(
                ContentSection(
                    content="\n".join(code_lines),
                    content_type=ContentType.SOURCE_CODE,
                    language=language,
                    start_line=start_line,
                    end_line=i,
                    is_code_fence=True,
                )
            )
            i += 1
            continue

        if _is_json_candidate(line):
            json_content, end_i = blocks.extract(i)
            if json_content is not None:
                # Bracket balance alone is not JSON: prose like a harness
                # sanitizer banner ("[harness: ... you.]") balances on one
                # line and used to be typed JSON_ARRAY here, sending it into
                # the structured compressors (and, via their fallback chain,
                # into lossy text compression). Validate before typing — the
                # mixed-content GATE (_has_valid_json_block_with_text) has
                # always validated; the splitter must agree with it.
                try:
                    json.loads(json_content)
                    valid_json = True
                except (TypeError, ValueError):
                    valid_json = False
                # Either way the block keeps its own section with the same
                # line span the JSON_ARRAY typing always gave it. For the
                # invalid case that standalone-ness is load-bearing: a short
                # prose banner must meet the text compressors' size floors
                # on its own, not merged into surrounding prose whose
                # combined size clears them (atomic=True keeps the
                # coalescer's hands off).
                sections.append(
                    ContentSection(
                        content=json_content,
                        content_type=(
                            ContentType.JSON_ARRAY if valid_json else ContentType.PLAIN_TEXT
                        ),
                        start_line=i,
                        end_line=end_i,
                        atomic=not valid_json,
                    )
                )
                i = end_i + 1
                continue

        if _is_search_line(line):
            search_lines = []
            start_line = i
            while i < len(lines) and _is_search_line(lines[i]):
                search_lines.append(lines[i])
                i += 1
            sections.append(
                ContentSection(
                    content="\n".join(search_lines),
                    content_type=ContentType.SEARCH_RESULTS,
                    start_line=start_line,
                    end_line=i - 1,
                )
            )
            continue

        text_lines = [line]
        start_line = i
        i += 1

        while i < len(lines):
            next_line = lines[i]
            if (
                _CODE_FENCE_PATTERN.match(next_line)
                or next_line.strip().startswith(("[", "{"))
                or _is_search_line(next_line)
                or (isolate and _carries_isolate(next_line))
            ):
                break
            text_lines.append(next_line)
            i += 1

        text_content = "\n".join(text_lines)
        if text_content.strip():
            sections.append(
                ContentSection(
                    content=text_content,
                    content_type=ContentType.PLAIN_TEXT,
                    start_line=start_line,
                    end_line=i - 1,
                )
            )

    return _coalesce_adjacent_plain_text(sections)


def _coalesce_adjacent_plain_text(sections: list[ContentSection]) -> list[ContentSection]:
    """Merge line-contiguous PLAIN_TEXT neighbors back into one section.

    The text accumulator stops at every ``[``/``{``/search-shaped line so the
    main loop can retry it as a candidate; when a candidate never balances it
    becomes the start of a NEW text section. Left split, each fragment would
    be rejoined by the router's ``"\\n\\n"`` reassembly, turning the prose's
    original single newlines into doubles. Merging contiguous fragments with
    ``"\\n"`` keeps the original bytes of uncompressed prose.

    ``atomic`` sections (placeholder lines, balanced-but-invalid JSON blocks)
    are never merged, in either direction — their standalone-ness carries
    meaning (compression exemption, per-block size floors).
    """
    merged: list[ContentSection] = []
    for section in sections:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.content_type is ContentType.PLAIN_TEXT
            and section.content_type is ContentType.PLAIN_TEXT
            and not prev.is_code_fence
            and not section.is_code_fence
            and not prev.atomic
            and not section.atomic
            and section.start_line == prev.end_line + 1
        ):
            prev.content = f"{prev.content}\n{section.content}"
            prev.end_line = section.end_line
            continue
        merged.append(section)
    return merged


def _scan_line(line: str, in_string: bool, escaped: bool) -> tuple[int, int, bool, bool]:
    """Bracket/brace deltas for one line, given the parser state entering it.

    Split out so the per-line result can be memoised across scans: what a line
    does to the counters is a pure function of the line and the two entry-state
    flags, nothing else.
    """
    bracket = 0
    brace = 0
    for ch in line:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            if in_string:
                escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            bracket += 1
        elif ch == "]":
            bracket -= 1
        elif ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
    return bracket, brace, in_string, escaped


def _is_json_candidate(line: str) -> bool:
    return line.strip().startswith(("[", "{"))


class _JsonBlockScanner:
    """Answers :func:`_extract_json_block` for every candidate of one pass.

    Each candidate walks forward until its running bracket and brace counts
    are both <= 0. Walked one by one, N candidates that balance late or never
    cost N^2/2 line steps. The scanner walks them directly until they have
    walked as many lines as the content has, then answers all the rest at
    once, with exactly the same results:

    * A walk is a chain of (line, string state) nodes, and each node has one
      successor, so the walks of all candidates form a forest whose roots are
      past the last line. Every node is scanned once, however many walks
      share it (walks that start in different string states and later fall
      into the same state merge).
    * With ``D(v)`` the counts summed from node ``v`` to the end of its
      chain, a walk from ``u`` balances at ``w`` exactly when
      ``D(next(w)) >= D(u)`` in both counts. So a candidate ends just before
      its nearest ancestor ``z`` with ``D(z) >= D(u)``.
    * One depth-first pass from the root keeps, over the ancestors of the
      current node, a tree keyed by bracket sum whose nodes hold stacks of
      (brace sum, depth); stacks are updated by overwrite-and-undo so leaving
      a node restores them. Each query and each push costs O(log^2 n).
    """

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self._direct_steps_left = len(lines)
        # Candidate line -> last line of its block (None: never balances).
        self._ends: dict[int, int | None] | None = None

    def extract(self, start: int) -> tuple[str | None, int]:
        if self._direct_steps_left > 0:
            json_content, end = _extract_json_block(self._lines, start)
            walked = (end if json_content is not None else len(self._lines) - 1) - start + 1
            self._direct_steps_left -= walked
            return json_content, end
        if self._ends is None:
            self._ends = self._solve()
        end_line = self._ends.get(start)
        if end_line is None:
            return None, start
        return "\n".join(self._lines[start : end_line + 1]), end_line

    def _solve(self) -> dict[int, int | None]:
        lines = self._lines
        n = len(lines)
        candidates = [i for i, line in enumerate(lines) if _is_json_candidate(line)]

        # Build the forest: node ids in creation order, one per (line, state).
        node_of: dict[tuple[int, bool, bool], int] = {}
        node_line: list[int] = []
        d_bracket: list[int] = []
        d_brace: list[int] = []
        succ: list[int] = []  # -1: past the last line (the root)
        start_node: dict[int, int] = {}
        for s in candidates:
            key = (s, False, False)
            prev = -1
            while key[0] < n:
                found = node_of.get(key)
                if found is not None:
                    if prev >= 0:
                        succ[prev] = found
                    break
                b, c, in_string, escaped = _scan_line(lines[key[0]], key[1], key[2])
                v = len(node_line)
                node_of[key] = v
                node_line.append(key[0])
                d_bracket.append(b)
                d_brace.append(c)
                succ.append(-1)
                if prev >= 0:
                    succ[prev] = v
                prev = v
                key = (key[0] + 1, in_string, escaped)
            start_node[s] = node_of[(s, False, False)]

        # Suffix sums D, children lists (successors have larger lines).
        count = len(node_line)
        sum_b = [0] * count
        sum_c = [0] * count
        children: list[list[int]] = [[] for _ in range(count)]
        roots: list[int] = []
        for v in sorted(range(count), key=node_line.__getitem__, reverse=True):
            nxt = succ[v]
            if nxt < 0:
                sum_b[v], sum_c[v] = d_bracket[v], d_brace[v]
                roots.append(v)
            else:
                sum_b[v], sum_c[v] = d_bracket[v] + sum_b[nxt], d_brace[v] + sum_c[nxt]
                children[nxt].append(v)
        is_query = {start_node[s] for s in candidates}

        # Tree over bracket sums; each tree node stacks ancestors as
        # (-brace sum, depth) with -brace increasing toward the top.
        coords = sorted({0, *sum_b})
        size = 1
        while size < len(coords):
            size *= 2
        neg_c: list[list[int]] = [[] for _ in range(2 * size)]
        depth_of: list[list[int]] = [[] for _ in range(2 * size)]
        used = [0] * (2 * size)

        def push(bracket: int, brace: int, depth: int) -> list[tuple[int, int, int, int, int]]:
            undo = []
            node = bisect_left(coords, bracket) + size
            while node:
                stack_c, stack_d, top = neg_c[node], depth_of[node], used[node]
                # Entries with brace <= this one are shallower: drop them.
                pos = bisect_left(stack_c, -brace, 0, top)
                if pos < len(stack_c):
                    undo.append((node, top, pos, stack_c[pos], stack_d[pos]))
                    stack_c[pos], stack_d[pos] = -brace, depth
                else:
                    undo.append((node, top, pos, 0, -1))
                    stack_c.append(-brace)
                    stack_d.append(depth)
                used[node] = pos + 1
                node //= 2
            return undo

        def pop(undo: list[tuple[int, int, int, int, int]]) -> None:
            for node, top, pos, old_c, old_d in reversed(undo):
                if old_d < 0:
                    neg_c[node].pop()
                    depth_of[node].pop()
                else:
                    neg_c[node][pos], depth_of[node][pos] = old_c, old_d
                used[node] = top

        def deepest(bracket: int, brace: int) -> int:
            # Deepest ancestor with bracket sum >= bracket and brace sum >= brace.
            best = -1
            lo, hi = bisect_left(coords, bracket) + size, 2 * size
            while lo < hi:
                for node in ((lo,) if lo & 1 else ()) + ((hi - 1,) if hi & 1 else ()):
                    idx = bisect_right(neg_c[node], -brace, 0, used[node]) - 1
                    if idx >= 0 and depth_of[node][idx] > best:
                        best = depth_of[node][idx]
                lo = (lo + 1) // 2
                hi //= 2
            return best

        ends: dict[int, int | None] = {}
        path: list[int] = [-1]  # path[d]: ancestor at depth d; depth 0 is the root
        root_undo = push(0, 0, 0)
        # Iterative DFS: (node, undo record or None when entering).
        work: list[tuple[int, list[tuple[int, int, int, int, int]] | None]] = [
            (v, None) for v in roots
        ]
        while work:
            v, undo = work.pop()
            if undo is not None:
                pop(undo)
                path.pop()
                continue
            depth = len(path)
            if v in is_query:
                z = deepest(sum_b[v], sum_c[v])
                if z < 0:
                    ends[node_line[v]] = None
                else:
                    w = path[z + 1] if z + 1 < depth else v
                    ends[node_line[v]] = node_line[w]
            path.append(v)
            work.append((v, push(sum_b[v], sum_c[v], depth)))
            work.extend((child, None) for child in children[v])
        pop(root_undo)
        return ends


def _extract_json_block(
    lines: list[str],
    start: int,
    *,
    cache: dict[tuple[int, bool, bool], tuple[int, int, bool, bool]] | None = None,
) -> tuple[str | None, int]:
    """Extract a complete JSON object or array block from line-oriented content.

    ``cache`` memoises the per-line scan across repeated calls over the SAME
    ``lines``. Callers that try every ``{``-leading line share one dict; without
    it each candidate that never balances re-scans character-by-character to the
    end of the content, which is quadratic.

    Callers pass ``None`` until a scan has actually run to the end without
    balancing, and only build the dict from then on. That matters: on content
    that balances on the first try — pretty-printed JSON, the common case — the
    memo has nothing to reuse and its per-line dict traffic made that shape ~2x
    SLOWER. A failed scan is the signal that later candidates will re-walk the
    same tail, and it is the only point at which the memo pays. MEASURED before the cache: 4643ms for
    1200 lines of JS-style object logs and 3737ms for truncated JSONL, growing
    exactly 4x per doubling. Ordinary shapes — pretty-printed JSON, valid JSONL,
    source, stack traces, prose — were ~1-6ms and never hit it, which is why this
    stayed invisible.

    Keyed on the entry state as well as the line, so a cached entry is only
    reused where the parser is in the same string/escape state. Same deltas,
    same result: this is a memo, not a heuristic.
    """
    bracket_count = 0
    brace_count = 0
    in_string = False
    escaped = False

    for i in range(start, len(lines)):
        key = (i, in_string, escaped)
        step = cache.get(key) if cache is not None else None
        if step is None:
            step = _scan_line(lines[i], in_string, escaped)
            if cache is not None:
                cache[key] = step
        d_bracket, d_brace, in_string, escaped = step
        bracket_count += d_bracket
        brace_count += d_brace

        if bracket_count <= 0 and brace_count <= 0:
            return "\n".join(lines[start : i + 1]), i

    return None, start
