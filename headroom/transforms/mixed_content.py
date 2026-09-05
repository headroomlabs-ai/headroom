"""Pure mixed-content parsing helpers for the content router."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .content_detector import ContentType


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


_CODE_FENCE_PATTERN = re.compile(r"^```(\w*)\s*$", re.MULTILINE)
_JSON_BLOCK_START = re.compile(r"^\s*[\[{]", re.MULTILINE)
_SEARCH_RESULT_PATTERN = re.compile(r"^\S+:\d+:", re.MULTILINE)
_PROSE_PATTERN = re.compile(r"[A-Z][a-z]+\s+\w+\s+\w+")


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
        "has_search_results": bool(_SEARCH_RESULT_PATTERN.search(content)),
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
    # Built only after a scan has run to the end without balancing — see
    # _extract_json_block. Content that balances promptly never allocates it and
    # so pays nothing for it.
    scan_cache: dict[tuple[int, bool, bool], tuple[int, int, bool, bool]] | None = None

    for index, line in enumerate(lines):
        if not line.strip().startswith(("[", "{")):
            continue

        json_content, end_index = _extract_json_block(lines, index, cache=scan_cache)
        if json_content is None:
            if scan_cache is None:
                scan_cache = {}
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

    scan_cache: dict[tuple[int, bool, bool], tuple[int, int, bool, bool]] | None = None

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
            language = match.group(1) or "unknown"
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

        if line.strip().startswith(("[", "{")):
            json_content, end_i = _extract_json_block(lines, i, cache=scan_cache)
            if json_content is None and scan_cache is None:
                # First scan that ran to the end without balancing: from here on
                # every later candidate would re-walk the same tail.
                scan_cache = {}
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

        if _SEARCH_RESULT_PATTERN.match(line):
            search_lines = []
            start_line = i
            while i < len(lines) and _SEARCH_RESULT_PATTERN.match(lines[i]):
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
                or _SEARCH_RESULT_PATTERN.match(next_line)
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
