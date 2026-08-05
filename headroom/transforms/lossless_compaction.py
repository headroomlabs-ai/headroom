"""Format-native, reversible lossless compaction for no-CCR proxy mode.

Every helper here is pure stdlib and keeps its output *looking like its own
type* — grep stays grep, logs stay logs, diffs stay diffs. No retrieval
marker (``<<ccr:…>>`` / ``Retrieve …``) is ever emitted, so the proxy needs
no MCP retrieve round-trip to stay recoverable.

The reversible transforms ship with exact inverses and are self-checked at
runtime by :func:`compact_lossless`: if a round-trip does not reproduce the
original (modulo intentionally-dropped non-semantic bits such as ANSI color)
or the result is not actually smaller, the original content is returned
unchanged. Nothing here raises.
"""

from __future__ import annotations

import re

__all__ = [
    "strip_ansi",
    "collapse_runs",
    "expand_runs",
    "is_run_collapsed",
    "fold_repeated_blocks",
    "unfold_repeated_blocks",
    "search_heading",
    "search_unheading",
    "search_dir_heading",
    "search_dir_unheading",
    "diff_strip_index",
    "compact_lossless",
]

# ANSI CSI SGR (color/style) escape sequences: ESC [ ... m. Color is
# non-semantic, so stripping it is a safe (one-way) lossless-of-meaning op.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# syslog-style run-collapse marker. The count is captured for exact inversion.
_RUN_MARKER_RE = re.compile(r"^\.\.\. \(repeated (\d+) times\)$")

# multi-line block back-reference marker. Length and distance (both in lines,
# in ORIGINAL coordinates) are captured for exact inversion: everything before
# a marker expands to the exact original prefix, so `distance` lines back in
# the expanded output is the block's first occurrence.
_BLOCK_MARKER_RE = re.compile(r"^\.\.\. \(repeats (\d+) lines from (\d+) lines back\)$")

# fold_repeated_blocks search bounds: minimum/maximum block length worth a
# marker, candidate anchors per line, and an input size cap so the scan stays
# negligible on huge payloads.
_FOLD_MIN_BLOCK = 3
_FOLD_MAX_BLOCK = 64
_FOLD_MAX_CANDIDATES = 8
_FOLD_MAX_LINES = 20_000

# grep/ripgrep default row shape: ``path:line:content``. ``line`` is digits;
# ``path`` must not itself look like ``line:content`` (i.e. not start with a
# bare number) so we don't mis-split a heading-form ``line:content`` row.
_GREP_ROW_RE = re.compile(r"^(?P<path>[^\n:]+):(?P<line>\d+):(?P<content>.*)$")
# heading-form data row (``line:content``) produced by search_heading.
_HEADING_ROW_RE = re.compile(r"^(?P<line>\d+):(?P<content>.*)$")

# unified-diff ``index <sha>..<sha> <mode>`` line. The diff still applies
# without it (git only uses it for rename/blob bookkeeping).
_DIFF_INDEX_RE = re.compile(r"^index [0-9a-fA-F]+\.\.[0-9a-fA-F]+( [0-7]+)?$")


def strip_ansi(text: str) -> str:
    """Remove ANSI CSI/SGR (color) escape sequences. Color is non-semantic."""
    return _ANSI_RE.sub("", text)


def _split_keep_trailing(text: str) -> tuple[list[str], bool]:
    """Split into lines, remembering whether a trailing newline was present.

    Returns (lines, had_trailing_newline). This lets the run helpers rejoin
    byte-exactly instead of always appending or always dropping a newline.
    """
    if text == "":
        return [], False
    had_trailing = text.endswith("\n")
    body = text[:-1] if had_trailing else text
    return body.split("\n"), had_trailing


def _join(lines: list[str], had_trailing: bool) -> str:
    out = "\n".join(lines)
    if had_trailing:
        out += "\n"
    return out


def collapse_runs(text: str) -> str:
    """Collapse runs of >=2 identical consecutive lines (syslog convention).

    A run of N (N>=2) identical lines becomes the line once followed by
    ``... (repeated N times)``. Exact inverse: :func:`expand_runs`.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        j = i
        while j + 1 < n and lines[j + 1] == lines[i]:
            j += 1
        run_len = j - i + 1
        if run_len >= 2:
            out.append(lines[i])
            out.append(f"... (repeated {run_len} times)")
        else:
            out.append(lines[i])
        i = j + 1
    return _join(out, had_trailing)


def expand_runs(text: str) -> str:
    """Exact inverse of :func:`collapse_runs`."""
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if i + 1 < n:
            m = _RUN_MARKER_RE.match(lines[i + 1])
            if m:
                count = int(m.group(1))
                out.extend([line] * count)
                i += 2
                continue
        out.append(line)
        i += 1
    return _join(out, had_trailing)


def is_run_collapsed(text: str) -> bool:
    """True if any run-collapse marker line is present."""
    for line in text.split("\n"):
        if _RUN_MARKER_RE.match(line):
            return True
    return False


def fold_repeated_blocks(text: str) -> str:
    """Collapse multi-line blocks that repeat earlier content into back-refs.

    The block-level generalization of :func:`collapse_runs`: a run of K
    consecutive lines (K >= 3) that exactly reproduces K lines seen D lines
    earlier becomes ``... (repeats K lines from D lines back)``. The repeats
    need not be adjacent, which is what config payloads actually look like —
    k8s container stanzas repeat with only the ``name:`` line differing, so
    their identical tails fold even though no two whole stanzas are
    consecutive. Coordinates are in original lines: the fold is only taken
    when the block does not overlap its anchor (K <= D), so on expansion the
    referenced region is always already reconstructed.
    Exact inverse: :func:`unfold_repeated_blocks`.
    """
    lines, had_trailing = _split_keep_trailing(text)
    n = len(lines)
    if n < _FOLD_MIN_BLOCK * 2 or n > _FOLD_MAX_LINES:
        return text
    positions: dict[str, list[int]] = {}
    out: list[str] = []
    i = 0
    while i < n:
        best_len = 0
        best_dist = 0
        for q in reversed(positions.get(lines[i], ())):
            max_len = min(_FOLD_MAX_BLOCK, n - i, i - q)
            length = 0
            while length < max_len and lines[q + length] == lines[i + length]:
                length += 1
            if length > best_len:
                best_len = length
                best_dist = i - q
        if best_len >= _FOLD_MIN_BLOCK:
            marker = f"... (repeats {best_len} lines from {best_dist} lines back)"
            block_chars = sum(len(lines[i + k]) + 1 for k in range(best_len))
            if block_chars > len(marker) + 1:
                out.append(marker)
                for k in range(best_len):
                    _remember(positions, lines[i + k], i + k)
                i += best_len
                continue
        _remember(positions, lines[i], i)
        out.append(lines[i])
        i += 1
    return _join(out, had_trailing)


def _remember(positions: dict[str, list[int]], line: str, index: int) -> None:
    """Track recent original positions of `line`, bounded per distinct line."""
    bucket = positions.setdefault(line, [])
    bucket.append(index)
    if len(bucket) > _FOLD_MAX_CANDIDATES:
        del bucket[0]


def unfold_repeated_blocks(text: str) -> str:
    """Exact inverse of :func:`fold_repeated_blocks`."""
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    for line in lines:
        m = _BLOCK_MARKER_RE.match(line)
        if m:
            length, dist = int(m.group(1)), int(m.group(2))
            start = len(out) - dist
            if start >= 0 and length <= dist:
                out.extend(out[start : start + length])
                continue
        out.append(line)
    return _join(out, had_trailing)


def search_heading(text: str) -> str:
    """Convert grep ``path:line:content`` rows into ripgrep --heading form.

    Consecutive rows sharing a path collapse to the path once on its own line
    (a *header* line), then ``line:content`` rows beneath it. Lines that don't
    match the ``path:line:content`` shape are passed through untouched. No
    blank separators are inserted (they would be ambiguous with passthrough
    content), keeping the transform exactly reversible via
    :func:`search_unheading`.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    current_path: str | None = None
    for line in lines:
        m = _GREP_ROW_RE.match(line)
        if m:
            path = m.group("path")
            if path != current_path:
                out.append(path)
                current_path = path
            out.append(f"{m.group('line')}:{m.group('content')}")
        else:
            # Any non-grep-row line ends the current file grouping.
            out.append(line)
            current_path = None
    return _join(out, had_trailing)


def search_unheading(text: str) -> str:
    """Exact inverse of :func:`search_heading`.

    A *header* line is any line that is not itself a ``line:content`` data row
    and is immediately followed by at least one ``line:content`` data row; it
    is consumed (not re-emitted) and its text becomes the ``path`` prefix for
    the data rows that follow, until a non-data line appears.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    current_path: str | None = None
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        data = _HEADING_ROW_RE.match(line)
        if current_path is not None and data:
            out.append(f"{current_path}:{data.group('line')}:{data.group('content')}")
            i += 1
            continue
        # Not a data row under an active header. Decide if THIS line is a new
        # header: it must not be a data row itself and must be followed by a
        # data row. If so, consume it as the path prefix (do not emit).
        if not data and i + 1 < n and _HEADING_ROW_RE.match(lines[i + 1]):
            current_path = line
            i += 1
            continue
        # Plain passthrough line (or a stray data row with no header): emit it
        # verbatim and clear any active grouping.
        current_path = None
        out.append(line)
        i += 1
    return _join(out, had_trailing)


# A dir-heading data row: ``<base>:<line>:<content>`` where base has no '/'.
_DIR_DATA_RE = re.compile(r"^(?P<base>[^/\n:]+):(?P<line>\d+):(?P<content>.*)$")


def search_dir_heading(text: str) -> str:
    """Fold grep ``path:line:content`` rows by DIRECTORY.

    Consecutive rows whose path shares a parent directory collapse to that
    directory once (a header ending in ``/``), then ``base:line:content`` rows
    beneath it. Complements :func:`search_heading` (which factors a repeated
    *file*): this factors a repeated *directory* across distinct files — the
    common ``grep -rn`` case where each file has a single match, so file-heading
    saves nothing but the shared directory repeats on every row. Rows whose path
    has no ``/`` pass through untouched. Exactly reversed by
    :func:`search_dir_unheading`; ``compact_lossless`` verifies the round-trip.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    current_dir: str | None = None
    for line in lines:
        m = _GREP_ROW_RE.match(line)
        if m and "/" in m.group("path"):
            path = m.group("path")
            cut = path.rindex("/") + 1
            dir_part, base = path[:cut], path[cut:]
            if dir_part != current_dir:
                out.append(dir_part)
                current_dir = dir_part
            out.append(f"{base}:{m.group('line')}:{m.group('content')}")
        else:
            out.append(line)
            current_dir = None
    return _join(out, had_trailing)


def search_dir_unheading(text: str) -> str:
    """Exact inverse of :func:`search_dir_heading`.

    A *header* is a line ending in ``/`` immediately followed by a
    ``base:line:content`` data row; it is consumed and re-prefixed onto each
    following data row until a non-data line appears.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    current_dir: str | None = None
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        data = _DIR_DATA_RE.match(line)
        if current_dir is not None and data:
            out.append(f"{current_dir}{line}")
            i += 1
            continue
        if line.endswith("/") and i + 1 < n and _DIR_DATA_RE.match(lines[i + 1]):
            current_dir = line
            i += 1
            continue
        current_dir = None
        out.append(line)
        i += 1
    return _join(out, had_trailing)


def diff_strip_index(text: str) -> str:
    """Drop ``index <sha>..<sha>`` lines from a unified diff (still applies)."""
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out = [line for line in lines if not _DIFF_INDEX_RE.match(line)]
    return _join(out, had_trailing)


# A whole-line file path: optional ``./``/``../`` root, >=1 directory segment,
# then a basename. No whitespace or ':' (so grep ``path:line:content`` rows —
# handled by search_heading — are excluded). Directory-only lines (trailing '/')
# don't match (empty basename), which keeps the fold unambiguous.
_PATH_ROW_RE = re.compile(r"^(?P<dir>(?:\.{0,2}/)?(?:[^/\s:]+/)+)(?P<base>[^/\s:]+)$")


def path_heading(text: str) -> str:
    """Fold a *pure* file-path listing (``find`` / ``ls -1`` / ``rg -l`` output)
    into ripgrep-heading form: each parent directory printed once on its own
    line (ending in ``/``), then the bare basenames beneath it.

    Reversibility is not assumed here — ``compact_lossless`` verifies the exact
    round-trip via :func:`path_unheading` and discards the fold on any mismatch
    (e.g. a stray no-slash line mistaken for a basename), so mixed content is
    always safe. Requires >=2 path rows or there is nothing to group.
    Complements ``search_heading``, which only handles the ``path:line:content``
    grep shape, not plain path lists.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if sum(1 for ln in lines if _PATH_ROW_RE.match(ln)) < 2:
        return text
    out: list[str] = []
    current: str | None = None
    for line in lines:
        m = _PATH_ROW_RE.match(line)
        if m:
            d = m.group("dir")
            if d != current:
                out.append(d)
                current = d
            out.append(m.group("base"))
        else:  # blank line inside/around the listing
            out.append(line)
            current = None
    return _join(out, had_trailing)


def path_unheading(text: str) -> str:
    """Exact inverse of :func:`path_heading`.

    A *header* is a line ending in ``/`` immediately followed by a basename row
    (a non-empty line with no ``/``); it is consumed and re-prefixed onto each
    following basename row until a blank line or another header.
    """
    lines, had_trailing = _split_keep_trailing(text)
    if not lines:
        return text
    out: list[str] = []
    current: str | None = None
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        is_base = line != "" and "/" not in line
        if current is not None and is_base:
            out.append(current + line)
            i += 1
            continue
        if line.endswith("/") and i + 1 < n and lines[i + 1] != "" and "/" not in lines[i + 1]:
            current = line
            i += 1
            continue
        current = None
        out.append(line)
        i += 1
    return _join(out, had_trailing)


def _smaller(candidate: str, original: str) -> bool:
    return len(candidate) < len(original)


def compact_lossless(content: str, kind: str) -> str:
    """Dispatch format-native lossless compaction by ``kind``.

    ``kind`` in {'log', 'search', 'diff', 'text', 'config'}. For reversible kinds the
    round-trip is verified internally (modulo the intentionally-dropped
    non-semantic bits, e.g. ANSI color for logs); if verification fails or the
    result is not smaller, the original content is returned unchanged. Never
    raises; unknown kinds pass through.
    """
    if not content:
        return content
    try:
        if kind == "log":
            # ANSI is non-semantic and dropped one-way; run-collapse must be
            # exactly reversible against the de-ANSI'd baseline.
            baseline = strip_ansi(content)
            candidate = collapse_runs(baseline)
            if expand_runs(candidate) != baseline:
                return content
            return candidate if _smaller(candidate, content) else content

        if kind == "search":
            # Two independent folds; keep the smaller that round-trips exactly.
            # search_heading factors a repeated FILE (many matches in one file);
            # search_dir_heading factors a repeated DIRECTORY (one match each
            # across many files in a dir — the grep -rn case the file fold misses).
            best = content
            for candidate, inverse in (
                (search_heading(content), search_unheading),
                (search_dir_heading(content), search_dir_unheading),
            ):
                if inverse(candidate) == content and _smaller(candidate, best):
                    best = candidate
            return best

        if kind == "paths":
            # Pure path listings (find/ls -1/rg -l): fold repeated parent dirs.
            candidate = path_heading(content)
            if path_unheading(candidate) != content:
                return content
            return candidate if _smaller(candidate, content) else content

        if kind == "diff":
            # Purely subtractive of non-semantic bookkeeping lines; the
            # remaining hunks still apply. No exact inverse needed.
            candidate = diff_strip_index(content)
            return candidate if _smaller(candidate, content) else content

        if kind == "text":
            # Collapse blank-line runs; reversible against itself.
            candidate = collapse_runs(content)
            if expand_runs(candidate) != content:
                return content
            return candidate if _smaller(candidate, content) else content

        if kind == "config":
            # Structured config (YAML/TOML/INI): single-line runs first, then
            # repeated multi-line stanzas. Inverse applies in reverse order.
            candidate = fold_repeated_blocks(collapse_runs(content))
            if expand_runs(unfold_repeated_blocks(candidate)) != content:
                return content
            return candidate if _smaller(candidate, content) else content
    except Exception:
        return content
    return content
