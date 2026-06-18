"""Content type detection for multi-format compression.

This module detects the type of tool output content to route it to the
appropriate compressor. SmartCrusher handles JSON arrays, but coding tasks
produce many other formats that need specialized handling.

Supported content types:
- JSON_ARRAY: Structured JSON data (existing SmartCrusher)
- SOURCE_CODE: Python, JavaScript, TypeScript, Go, etc.
- SEARCH_RESULTS: grep/ripgrep output (file:line:content)
- BUILD_OUTPUT: Compiler, test, lint logs
- GIT_DIFF: Unified diff format
- PLAIN_TEXT: Generic text (fallback)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum


class ContentType(Enum):
    """Types of content that can be compressed."""

    JSON_ARRAY = "json_array"  # Existing SmartCrusher handles this
    SOURCE_CODE = "source_code"  # Python, JS, TS, Go, Rust, etc.
    SEARCH_RESULTS = "search"  # grep/ripgrep output
    BUILD_OUTPUT = "build"  # Compiler, test, lint logs
    GIT_DIFF = "diff"  # Unified diff format
    HTML = "html"  # Web pages (needs content extraction, not compression)
    TABULAR = "tabular"  # CSV/TSV, markdown tables, fixed-width tables
    PLAIN_TEXT = "text"  # Fallback


@dataclass
class DetectionResult:
    """Result of content type detection."""

    content_type: ContentType
    confidence: float  # 0.0 to 1.0
    metadata: dict  # Type-specific metadata (e.g., language for code)


# Patterns for detection
_SEARCH_RESULT_PATTERN = re.compile(
    r"^[^\s:]+:\d+:"  # file:line: format (grep -n style)
)

# A markdown table separator row, e.g. "| --- | :--: |" or "---|---".
# Every cell must be dashes with optional alignment colons.
_MD_SEP_CELL = re.compile(r"^:?-{2,}:?$")

# Bug-fix (2026-04-25): extended to recognize merge-commit headers
# (`diff --combined <path>`, `diff --cc <path>`) and combined-diff hunk
# headers (`@@@`+ ranges). Previously only `git diff` shape was detected,
# so merge-commit diffs from `git log -p` got misrouted away from
# DiffCompressor entirely.
_DIFF_HEADER_PATTERN = re.compile(
    r"^("
    r"diff --git"
    r"|diff --combined "
    r"|diff --cc "
    r"|--- a/"
    r"|@@\s+-\d+,\d+\s+\+\d+,\d+\s+@@"
    r"|@@@+\s+-\d+(?:,\d+)?\s+(?:-\d+(?:,\d+)?\s+)+\+\d+(?:,\d+)?\s+@@@+"
    r")"
)

_DIFF_CHANGE_PATTERN = re.compile(r"^[+-][^+-]")

# Code patterns by language
_CODE_PATTERNS = {
    "python": [
        re.compile(r"^\s*(def|class|import|from|async def)\s+\w+"),
        re.compile(r"^\s*@\w+"),  # decorators
        re.compile(r'^\s*"""'),  # docstrings
        re.compile(r"^\s*if __name__\s*=="),
    ],
    "javascript": [
        re.compile(r"^\s*(function|const|let|var|class|import|export)\s+"),
        re.compile(r"^\s*(async\s+function|=>\s*\{)"),
        re.compile(r"^\s*module\.exports"),
    ],
    "typescript": [
        re.compile(r"^\s*(interface|type|enum|namespace)\s+\w+"),
        re.compile(r":\s*(string|number|boolean|any|void)\b"),
    ],
    "go": [
        re.compile(r"^\s*(func|type|package|import)\s+"),
        re.compile(r"^\s*func\s+\([^)]+\)\s+\w+"),  # method
    ],
    "rust": [
        re.compile(r"^\s*(fn|struct|enum|impl|mod|use|pub)\s+"),
        re.compile(r"^\s*#\["),  # attributes
    ],
    "java": [
        re.compile(r"^\s*(public|private|protected)\s+(class|interface|enum)"),
        re.compile(r"^\s*@\w+"),  # annotations
        re.compile(r"^\s*package\s+[\w.]+;"),
    ],
}

# Log/build output patterns
_LOG_PATTERNS = [
    re.compile(r"\b(ERROR|FAIL|FAILED|FATAL|CRITICAL)\b", re.IGNORECASE),
    re.compile(r"\b(WARN|WARNING)\b", re.IGNORECASE),
    re.compile(r"\b(INFO|DEBUG|TRACE)\b", re.IGNORECASE),
    re.compile(r"^\s*\d{4}-\d{2}-\d{2}"),  # timestamp
    re.compile(r"^\s*\[\d{2}:\d{2}:\d{2}\]"),  # time format
    re.compile(r"^={3,}|^-{3,}"),  # separators
    re.compile(r"^\s*PASSED|^\s*FAILED|^\s*SKIPPED"),  # test results
    re.compile(r"^npm ERR!|^yarn error|^cargo error"),  # build tools
    re.compile(r"Traceback \(most recent call last\)"),  # Python traceback
    re.compile(r"^\w*(Error|Exception):"),  # Python exception final line
    re.compile(r"^\s*at\s+[\w.$]+\("),  # JS/Java stack trace
]


def detect_content_type(content: str) -> DetectionResult:
    """Detect the type of content for appropriate compression.

    Args:
        content: The content to analyze.

    Returns:
        DetectionResult with type, confidence, and metadata.

    Examples:
        >>> result = detect_content_type('[{"id": 1}, {"id": 2}]')
        >>> result.content_type
        ContentType.JSON_ARRAY

        >>> result = detect_content_type('src/main.py:42:def process():')
        >>> result.content_type
        ContentType.SEARCH_RESULTS
    """
    if not content or not content.strip():
        return DetectionResult(ContentType.PLAIN_TEXT, 0.0, {})

    # 1. Try JSON first (highest priority for SmartCrusher compatibility)
    json_result = _try_detect_json(content)
    if json_result:
        return json_result

    # 2. Check for diff (very distinctive patterns)
    diff_result = _try_detect_diff(content)
    if diff_result and diff_result.confidence >= 0.7:
        return diff_result

    # 3. Check for HTML (very distinctive, needs extraction not compression)
    html_result = _try_detect_html(content)
    if html_result and html_result.confidence >= 0.7:
        return html_result

    # 4. Check for search results (file:line: format)
    search_result = _try_detect_search(content)
    if search_result and search_result.confidence >= 0.6:
        return search_result

    # 5. Check for build/log output
    log_result = _try_detect_log(content)
    if log_result and log_result.confidence >= 0.5:
        return log_result

    # 6. Check for tabular data (CSV/TSV, markdown tables). Runs after
    #    search/log so colon-delimited search output and freeform logs claim
    #    their content first; tabular requires a consistent multi-column
    #    delimiter or a markdown header+separator pair.
    tabular_result = _try_detect_tabular(content)
    if tabular_result and tabular_result.confidence >= 0.6:
        return tabular_result

    # 7. Check for source code
    code_result = _try_detect_code(content)
    if code_result and code_result.confidence >= 0.5:
        return code_result

    # 8. Fallback to plain text
    return DetectionResult(ContentType.PLAIN_TEXT, 0.5, {})


def _try_detect_json(content: str) -> DetectionResult | None:
    """Try to detect JSON array content."""
    content = content.strip()

    # Quick check: must start with [ for array
    if not content.startswith("["):
        return None

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            # Check if it's a list of dicts (SmartCrusher compatible)
            if parsed and all(isinstance(item, dict) for item in parsed):
                return DetectionResult(
                    ContentType.JSON_ARRAY,
                    1.0,
                    {"item_count": len(parsed), "is_dict_array": True},
                )
            # It's a list but not of dicts
            return DetectionResult(
                ContentType.JSON_ARRAY,
                0.8,
                {"item_count": len(parsed), "is_dict_array": False},
            )
    except json.JSONDecodeError:
        pass

    return None


def _try_detect_diff(content: str) -> DetectionResult | None:
    """Try to detect git diff format.

    Bug-fix (2026-04-25): widened the scan window from 50 to 500 lines.
    `git log -p` and `git format-patch` outputs commonly have multi-line
    commit messages or email headers ahead of the actual diff; with the
    50-line cap, those long preambles pushed the `diff --git` header out
    of the detection window, and the input was misrouted to a
    plain-text/code compressor instead of DiffCompressor. 500 lines
    covers commit messages of ~500 lines (rare; if longer, you've got
    bigger problems).
    """
    lines = content.split("\n")[:500]

    header_matches = 0
    change_matches = 0

    for line in lines:
        if _DIFF_HEADER_PATTERN.match(line):
            header_matches += 1
        if _DIFF_CHANGE_PATTERN.match(line):
            change_matches += 1

    if header_matches == 0:
        return None

    # High confidence if we see diff headers
    confidence = min(1.0, 0.5 + (header_matches * 0.2) + (change_matches * 0.05))

    return DetectionResult(
        ContentType.GIT_DIFF,
        confidence,
        {"header_matches": header_matches, "change_lines": change_matches},
    )


# HTML detection patterns
_HTML_DOCTYPE_PATTERN = re.compile(r"^\s*<!doctype\s+html", re.IGNORECASE)
_HTML_TAG_PATTERN = re.compile(r"<html[\s>]", re.IGNORECASE)
_HTML_HEAD_PATTERN = re.compile(r"<head[\s>]", re.IGNORECASE)
_HTML_BODY_PATTERN = re.compile(r"<body[\s>]", re.IGNORECASE)
_HTML_STRUCTURAL_TAGS = re.compile(
    r"<(div|span|script|style|link|meta|nav|header|footer|aside|article|section|main)[\s>]",
    re.IGNORECASE,
)


def _try_detect_html(content: str) -> DetectionResult | None:
    """Try to detect HTML content.

    HTML needs content extraction (removing scripts, styles, nav, etc.),
    not token-level compression like Kompress.
    """
    # Check first 3000 chars for HTML indicators
    sample = content[:3000]

    # Check for DOCTYPE (very strong signal)
    has_doctype = bool(_HTML_DOCTYPE_PATTERN.search(sample))

    # Check for <html> tag
    has_html_tag = bool(_HTML_TAG_PATTERN.search(sample))

    # Check for <head> or <body>
    has_head = bool(_HTML_HEAD_PATTERN.search(sample))
    has_body = bool(_HTML_BODY_PATTERN.search(sample))

    # Count structural HTML tags
    structural_matches = len(_HTML_STRUCTURAL_TAGS.findall(sample))

    # Quick rejection: not HTML if no indicators
    if not has_doctype and not has_html_tag and structural_matches < 3:
        return None

    # Calculate confidence
    confidence = 0.0

    if has_doctype:
        confidence += 0.5
    if has_html_tag:
        confidence += 0.3
    if has_head:
        confidence += 0.1
    if has_body:
        confidence += 0.1

    # Structural tags contribute to confidence
    confidence += min(0.3, structural_matches * 0.03)

    # Cap at 1.0
    confidence = min(1.0, confidence)

    if confidence < 0.5:
        return None

    return DetectionResult(
        ContentType.HTML,
        confidence,
        {
            "has_doctype": has_doctype,
            "has_html_tag": has_html_tag,
            "structural_tags": structural_matches,
        },
    )


def _try_detect_search(content: str) -> DetectionResult | None:
    """Try to detect grep/ripgrep search results."""
    lines = content.split("\n")[:100]  # Check first 100 lines
    if not lines:
        return None

    matching_lines = 0
    for line in lines:
        if line.strip() and _SEARCH_RESULT_PATTERN.match(line):
            matching_lines += 1

    if matching_lines == 0:
        return None

    # Calculate confidence based on proportion of matching lines
    non_empty_lines = sum(1 for line in lines if line.strip())
    if non_empty_lines == 0:
        return None

    ratio = matching_lines / non_empty_lines

    # Need at least 30% of lines to match the pattern
    if ratio < 0.3:
        return None

    confidence = min(1.0, 0.4 + (ratio * 0.6))

    return DetectionResult(
        ContentType.SEARCH_RESULTS,
        confidence,
        {"matching_lines": matching_lines, "total_lines": non_empty_lines},
    )


def _try_detect_log(content: str) -> DetectionResult | None:
    """Try to detect build/log output."""
    lines = content.split("\n")[:200]  # Check first 200 lines
    if not lines:
        return None

    pattern_matches = 0
    error_matches = 0

    for line in lines:
        for i, pattern in enumerate(_LOG_PATTERNS):
            if pattern.search(line):
                pattern_matches += 1
                if i < 2:  # ERROR or WARN patterns
                    error_matches += 1
                break  # One pattern per line is enough

    if pattern_matches == 0:
        return None

    non_empty_lines = sum(1 for line in lines if line.strip())
    if non_empty_lines == 0:
        return None

    ratio = pattern_matches / non_empty_lines

    # Need at least 10% of lines to match log patterns
    if ratio < 0.1:
        return None

    confidence = min(1.0, 0.3 + (ratio * 0.5) + (error_matches * 0.05))

    return DetectionResult(
        ContentType.BUILD_OUTPUT,
        confidence,
        {
            "pattern_matches": pattern_matches,
            "error_matches": error_matches,
            "total_lines": non_empty_lines,
        },
    )


def _md_cell_count(row: str) -> int:
    """Count cells in a markdown table row, ignoring the outer pipes."""
    return len(row.strip().strip("|").split("|"))


def _is_md_separator(row: str) -> bool:
    """True if `row` is a markdown table separator (e.g. ``| --- | :--: |``)."""
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    cells = [c for c in cells if c != ""]
    if len(cells) < 2:
        return False
    return all(_MD_SEP_CELL.match(c) for c in cells)


def _try_detect_markdown_table(lines: list[str]) -> DetectionResult | None:
    """Detect a markdown table: a piped header row followed by a separator."""
    for i in range(len(lines) - 1):
        header, sep = lines[i], lines[i + 1]
        if "|" in header and _is_md_separator(sep):
            cols = _md_cell_count(header)
            if cols >= 2:
                return DetectionResult(
                    ContentType.TABULAR,
                    0.95,
                    {"format": "markdown", "columns": cols},
                )
    return None


def _try_detect_delimited(lines: list[str]) -> DetectionResult | None:
    """Detect CSV/TSV by a delimiter with a consistent per-line column count.

    A stable column count is what separates real tabular data from prose that
    merely contains commas, and from ``file:line:content`` search output (which
    has a variable number of colons). Tabs are a stronger signal than commas
    (they rarely occur in prose), so they need less consistency.
    """
    from collections import Counter

    sample = lines[:20]
    if len(sample) < 3:
        return None

    best: DetectionResult | None = None
    for delim, min_consistency in ((",", 0.85), ("\t", 0.7), (";", 0.85), ("|", 0.85)):
        counts = [row.count(delim) for row in sample]
        if counts[0] == 0:  # header row must contain the delimiter
            continue
        common_count, freq = Counter(counts).most_common(1)[0]
        if common_count == 0:
            continue
        consistency = freq / len(sample)
        ncols = common_count + 1
        if ncols < 2 or consistency < min_consistency:
            continue
        # Prose guard: prose that merely contains commas ("Hello, friend.")
        # reads like sentences. Real table rows are short field tuples.
        if _looks_like_prose(sample, delim):
            continue
        confidence = min(0.95, 0.5 + consistency * 0.3 + min(ncols, 5) * 0.03)
        if best is None or confidence > best.confidence:
            best = DetectionResult(
                ContentType.TABULAR,
                confidence,
                {"format": "csv", "delimiter": delim, "columns": ncols},
            )
    return best


def _looks_like_prose(sample: list[str], delim: str) -> bool:
    """Heuristic: distinguish comma-bearing prose from real CSV rows.

    Prose reads like sentences (ends with ``.!?``) and has wordy cells; CSV
    rows are short field tuples. Either signal rejects the candidate.
    """
    enders = sum(1 for r in sample if r.rstrip().endswith((".", "!", "?")))
    if enders / len(sample) >= 0.5:
        return True
    cells = [c.strip() for r in sample for c in r.split(delim)]
    avg_words = sum(len(c.split()) for c in cells) / len(cells)
    return avg_words > 3


def _try_detect_tabular(content: str) -> DetectionResult | None:
    """Detect tabular text: markdown tables first, then delimited CSV/TSV."""
    lines = [ln for ln in content.split("\n") if ln.strip()][:50]
    if len(lines) < 3:
        return None

    md_result = _try_detect_markdown_table(lines)
    if md_result:
        return md_result

    return _try_detect_delimited(lines)


def _try_detect_code(content: str) -> DetectionResult | None:
    """Try to detect source code and identify language."""
    lines = content.split("\n")[:100]  # Check first 100 lines
    if not lines:
        return None

    language_scores: dict[str, int] = {}

    for line in lines:
        for lang, patterns in _CODE_PATTERNS.items():
            for pattern in patterns:
                if pattern.match(line):
                    language_scores[lang] = language_scores.get(lang, 0) + 1
                    break  # One pattern per language per line

    if not language_scores:
        return None

    # Find best matching language
    best_lang = max(language_scores, key=lambda k: language_scores[k])
    best_score = language_scores[best_lang]

    # Need at least 3 pattern matches to be confident
    if best_score < 3:
        return None

    non_empty_lines = sum(1 for line in lines if line.strip())
    ratio = best_score / max(non_empty_lines, 1)

    confidence = min(1.0, 0.4 + (ratio * 0.4) + (best_score * 0.02))

    return DetectionResult(
        ContentType.SOURCE_CODE,
        confidence,
        {"language": best_lang, "pattern_matches": best_score},
    )


def is_json_array_of_dicts(content: str) -> bool:
    """Quick check if content is a JSON array of dictionaries.

    This is the format SmartCrusher can handle natively.

    Args:
        content: The content to check.

    Returns:
        True if content is a JSON array where all items are dicts.
    """
    result = detect_content_type(content)
    return result.content_type == ContentType.JSON_ARRAY and result.metadata.get(
        "is_dict_array", False
    )
