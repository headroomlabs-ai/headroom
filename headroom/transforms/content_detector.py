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
- STRUCTURED_CONFIG: YAML/TOML/INI config files
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
    STRUCTURED_CONFIG = "structured_config"  # YAML/TOML/INI config files
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
    "csharp": [
        re.compile(r"^\s*using\s+[\w.]+\s*;"),  # using directive (not C++ `using namespace x;`)
        re.compile(r"^\s*namespace\s+[\w.]+"),
        re.compile(
            r"^\s*(public|private|protected|internal|sealed|static|abstract|partial)\s+"
            r"(class|struct|record|interface|enum)\b"
        ),
        re.compile(r"^.*\b(get|set|init);"),  # auto-property accessors
    ],
    "php": [
        re.compile(r"<\?php\b"),
        re.compile(r"^\s*namespace\s+[\w\\]+\s*;"),
        re.compile(r"^\s*use\s+[\w\\]+(\s+as\s+\w+)?\s*;"),
        re.compile(r"^\s*(public|private|protected|static|abstract|final)?\s*function\s+\w+\s*\("),
        re.compile(r"\$this->"),
    ],
}

# Structured-config (YAML/TOML/INI) patterns. TOML and INI share the
# `[section]` header shape; the stdlib parsers disambiguate. YAML is
# heuristic-only (PyYAML is not a dependency): key/list/document-marker
# line share plus structure signals, with prose and front-matter guards.
_CONFIG_SECTION_RE = re.compile(r"^\s*\[\[?[\w.\-\"' ]+\]\]?\s*$")
_TOML_ASSIGN_RE = re.compile(r"""^\s*(?:[\w.\-]+|"[^"]+"|'[^']+')\s*=\s*\S""")
_INI_ASSIGN_RE = re.compile(r"^\s*[\w.\-@ ]+?\s*[=:]\s*")
_YAML_KEY_RE = re.compile(r"""^\s*(?:-\s+)?(?:[\w.\-/]+|"[^"]+"|'[^']+')\s*:(?:\s|$)""")
_YAML_LIST_RE = re.compile(r"^\s*-\s+\S")
_YAML_DOC_RE = re.compile(r"^---\s*$|^\.\.\.\s*$")
_CONFIG_COMMENT_RE = re.compile(r"^\s*[#;]")

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
    re.compile(r"^\s*at\s+[\w.$/]+\("),  # JS/Java stack trace (JPMS module frames incl.)
    re.compile(r"^\s*at async \S"),  # Node async stack frame (no paren form)
    re.compile(r"^(panic|fatal error): "),  # Go panic opener
    re.compile(r"^goroutine \d+ \["),  # Go goroutine dump header
    re.compile(r"^\t\S+\.go:\d+ \+0x"),  # Go frame file line
    re.compile(r"^thread '[^']*' panicked at"),  # Rust panic
    re.compile(r"^stack backtrace:"),  # Rust backtrace header
    re.compile(r"^\s+\d+: \S"),  # Rust numbered backtrace frame
    re.compile(r"^\s+at \S+:\d+:\d+$"),  # Rust/JS bare path frame sub-line
    re.compile(r"^Unhandled exception\."),  # .NET unhandled exception
    re.compile(r"^\s*at .+\) in .+:line \d+"),  # .NET frame with PDB info
    re.compile(r"^Caused by: "),  # Java exception chain head
    re.compile(r"^\s*\.\.\. \d+ more$"),  # Java elided-frames summary
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

    # 7. Check for structured config (YAML/TOML/INI). Runs after tabular so
    #    delimited data keeps its claim, and before code so config files with
    #    code-ish lines route to the structure-aware config compressor.
    config_result = _try_detect_structured_config(content)
    if config_result and config_result.confidence >= 0.6:
        return config_result

    # 8. Check for source code
    code_result = _try_detect_code(content)
    if code_result and code_result.confidence >= 0.5:
        return code_result

    # 9. Fallback to plain text
    return DetectionResult(ContentType.PLAIN_TEXT, 0.5, {})


_JSON_DECODER = json.JSONDecoder()
# The decoded JSON value must be at least this fraction of the content for a
# WRAPPED payload to still count as JSON: a small structural wrapper (a harness
# observation shell, an ``Exit code:`` prefix) around a JSON body passes, but a
# prose/code blob that merely contains a JSON fragment does not. Fraction-based
# so it is size-correct — a large JSON with a proportionally small wrapper passes,
# a short mostly-prose string does not. (Pure JSON never reaches this check.)
_JSON_MIN_BULK_FRACTION = 0.6


def _decode_concatenated_json(content: str) -> list | None:
    """Decode a run of whitespace-separated top-level JSON values.

    Web search tools (SerpAPI, Tavily, custom backends) commonly emit
    back-to-back JSON objects separated only by whitespace rather than a real
    array: ``{"title": ...} {"title": ...} {"title": ...}``. Returns the list
    of decoded values, or None if the text isn't a clean run of JSON values
    separated only by whitespace.
    """
    decoder = json.JSONDecoder()
    idx, length = 0, len(content)
    items: list = []
    while idx < length:
        while idx < length and content[idx].isspace():
            idx += 1
        if idx >= length:
            break
        try:
            value, idx = decoder.raw_decode(content, idx)
        except ValueError:
            return None
        items.append(value)
    return items or None


def normalize_concatenated_json(content: str) -> str | None:
    """Convert whitespace-separated JSON objects into a canonical JSON array.

    SmartCrusher only compresses JSON arrays, so this rewrites the
    space-separated web_search shape (``{...} {...} {...}``) into
    ``[{...}, {...}, {...}]``. Returns None unless the content is two or more
    whitespace-separated JSON objects.
    """
    stripped = content.strip()
    if not stripped.startswith("{"):
        return None
    items = _decode_concatenated_json(stripped)
    if items and len(items) >= 2 and all(isinstance(item, dict) for item in items):
        return json.dumps(items)
    return None


def _try_detect_json(content: str) -> DetectionResult | None:
    """Detect JSON by PARSING, not by surface patterns.

    JSON is whatever parses as JSON — objects, arrays, and any nesting are all
    equally JSON, so a leading-``[`` check misses every ``{…}`` config/data file.
    Tool output is often a JSON value wrapped in a little surrounding text (a
    harness observation shell, an ``Exit code:`` prefix); we decode one JSON value
    out of the payload and accept it when it is the bulk of the content, which
    tolerates ANY wrapper without hard-coding a harness's tags. The whitespace-
    separated web_search shape (``{...} {...}``, #1741) is detected too and
    normalized to a real array before crushing (see normalize_concatenated_json).
    """
    stripped = content.strip()
    if not stripped:
        return None

    try:
        value = json.loads(stripped)
    except RecursionError:
        # Deeply nested JSON (e.g. ``[[[[...]]]]`` with 10k+ levels) can
        # exceed Python's recursion limit. Treat as non-JSON so the router
        # falls through to a safe strategy instead of crashing.
        return None
    except ValueError:
        # Not pure JSON. First: a run of whitespace-separated top-level JSON
        # objects (web_search output, #1741) -> JSON_ARRAY.
        if stripped.startswith("{"):
            try:
                items = _decode_concatenated_json(stripped)
            except RecursionError:
                return None
            if items and len(items) >= 2 and all(isinstance(item, dict) for item in items):
                return DetectionResult(
                    ContentType.JSON_ARRAY,
                    1.0,
                    {"item_count": len(items), "is_dict_array": True, "concatenated": True},
                )

        # Otherwise decode one JSON value out of a small wrapped payload.
        start = min((i for i in (stripped.find("{"), stripped.find("[")) if i >= 0), default=-1)
        if start < 0:
            return None
        try:
            value, end = _JSON_DECODER.raw_decode(stripped, start)
        except (ValueError, RecursionError):
            return None
        # Accept only when the decoded JSON is the BULK of the content (see
        # _JSON_MIN_BULK_FRACTION) — a small structural wrapper around a JSON body,
        # not a prose/code blob that merely contains a JSON fragment.
        if (end - start) < len(stripped) * _JSON_MIN_BULK_FRACTION:
            return None

    # A bare scalar (42, "s", true) is not structured data worth routing as JSON.
    if not isinstance(value, dict | list):
        return None

    if isinstance(value, list):
        is_dict_array = bool(value) and all(isinstance(item, dict) for item in value)
        return DetectionResult(
            ContentType.JSON_ARRAY,
            1.0 if is_dict_array else 0.8,
            {"item_count": len(value), "is_dict_array": is_dict_array},
        )
    return DetectionResult(
        ContentType.JSON_ARRAY,
        0.9,
        {"is_dict_array": False, "is_object": True},
    )


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


def _is_search_result_line(line: str) -> bool:
    """True when a line looks like ``path:line:content`` grep output.

    The bare ``^[^\\s:]+:\\d+:`` shape also matches ISO-8601 timestamps
    (``…T09:57:59…``) and XML-ish wrappers harnesses prepend to user turns
    (Copilot CLI's ``<current_datetime>…`` line), which misroutes prose to
    the SearchCompressor — and that compressor keeps only matching lines,
    deleting the rest. So the pre-colon segment must additionally look like
    a file path: no angle brackets and no ``=`` (rules out markup tags and
    ``key=value:12:`` log lines).
    """
    if not _SEARCH_RESULT_PATTERN.match(line):
        return False
    prefix = line.split(":", 1)[0]
    return "<" not in prefix and ">" not in prefix and "=" not in prefix


def _try_detect_search(content: str) -> DetectionResult | None:
    """Try to detect grep/ripgrep search results."""
    lines = content.split("\n")[:100]  # Check first 100 lines
    if not lines:
        return None

    matching_lines = 0
    for line in lines:
        if line.strip() and _is_search_result_line(line):
            matching_lines += 1

    # Absolute floor: a single coincidental `word:digits:` line (a timestamp,
    # a URL, a time literal inside prose) must not classify a whole payload as
    # search results — the SearchCompressor drops every non-matching line, so
    # a false positive is data loss. A genuine one-line grep result loses
    # nothing by staying uncompressed: all of its lines match, so the
    # compressor would have kept it verbatim anyway.
    if matching_lines < 2:
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


def _try_parse_toml(content: str) -> bool:
    """True if `content` parses as TOML (stdlib tomllib, or the tomli backport)."""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            return False
    try:
        tomllib.loads(content)
        return True
    except Exception:
        return False


def _parse_config_flavor(content: str) -> str | None:
    """Disambiguate `[section]`-shaped config: TOML first, then INI.

    Both flavors share the section-header line shape; only the stdlib parsers
    can tell them apart reliably. Returns "toml", "ini", or None when neither
    parser accepts the content (then it is not claimed as config at all).
    """
    if len(content) > 1_000_000:
        return None
    if _try_parse_toml(content):
        return "toml"
    import configparser

    parser = configparser.ConfigParser(interpolation=None, strict=False)
    try:
        parser.read_string(content)
    except Exception:
        return None
    return "ini" if parser.sections() else None


def _try_detect_structured_config(content: str) -> DetectionResult | None:
    """Try to detect structured config content (YAML, TOML, INI).

    TOML/INI claims are parser-confirmed (stdlib), so they carry high
    confidence. YAML has no stdlib parser, so its claim is heuristic:
    key/list/document-marker line share plus a structure signal, guarded
    against prose and markdown front-matter.
    """
    head = content.lstrip()[:1]
    if not head or head in "{<":
        # JSON objects and markup are never config; JSON arrays and real
        # TOML/INI `[section]` headers disambiguate below.
        return None

    lines = content.split("\n")[:200]
    non_empty = [ln for ln in lines if ln.strip()]
    if len(non_empty) < 3:
        return None
    # Comment lines are neutral: excluded from the line-share ratio so
    # comment-heavy configs and #-heading markdown don't skew it either way.
    body = [ln for ln in non_empty if not _CONFIG_COMMENT_RE.match(ln)]
    if len(body) < 3:
        return None

    # TOML / INI: require a section header plus assignment-dominant body,
    # then let the stdlib parsers confirm and disambiguate.
    sections = sum(1 for ln in body if _CONFIG_SECTION_RE.match(ln))
    if sections >= 1:
        assigns = sum(1 for ln in body if _TOML_ASSIGN_RE.match(ln) or _INI_ASSIGN_RE.match(ln))
        if assigns >= 2 and (sections + assigns) / len(body) >= 0.6:
            flavor = _parse_config_flavor(content)
            if flavor is not None:
                share = (sections + assigns) / len(body)
                return DetectionResult(
                    ContentType.STRUCTURED_CONFIG,
                    min(0.95, 0.7 + share * 0.25),
                    {"flavor": flavor, "sections": sections, "assignments": assigns},
                )

    # Markdown front-matter guard: a `---` fence closed within 60 lines and
    # followed by non-YAML content is a markdown document, not standalone YAML.
    if lines and lines[0].strip() == "---":
        for idx in range(1, min(len(lines), 60)):
            if lines[idx].strip() in ("---", "..."):
                tail = [ln for ln in lines[idx + 1 :] if ln.strip()]
                tail_yaml = sum(
                    1 for ln in tail if _YAML_KEY_RE.match(ln) or _YAML_LIST_RE.match(ln)
                )
                if tail and tail_yaml / len(tail) < 0.3:
                    return None
                break

    # YAML heuristic.
    yaml_keys = sum(1 for ln in body if _YAML_KEY_RE.match(ln))
    yaml_lists = sum(1 for ln in body if _YAML_LIST_RE.match(ln) and not _YAML_KEY_RE.match(ln))
    doc_marks = sum(1 for ln in body if _YAML_DOC_RE.match(ln.strip()))
    if yaml_keys < 3:
        return None
    share = (yaml_keys + yaml_lists + doc_marks) / len(body)
    if share < 0.6:
        return None
    # Prose guards: config lines are short field-ish tuples, prose reads like
    # sentences (mirrors _looks_like_prose for delimited data).
    enders = sum(1 for ln in body if ln.rstrip().endswith((".", "!", "?")))
    if enders / len(body) >= 0.5:
        return None
    avg_words = sum(len(ln.split()) for ln in body) / len(body)
    if avg_words > 8:
        return None
    # Structure signal: nested indentation, a document marker, or a real list.
    indents = {
        len(ln) - len(ln.lstrip(" "))
        for ln in body
        if _YAML_KEY_RE.match(ln) or _YAML_LIST_RE.match(ln)
    }
    if len(indents) < 2 and doc_marks == 0 and yaml_lists < 3:
        return None

    return DetectionResult(
        ContentType.STRUCTURED_CONFIG,
        min(0.9, 0.55 + share * 0.35),
        {"flavor": "yaml", "keys": yaml_keys, "list_items": yaml_lists},
    )


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
