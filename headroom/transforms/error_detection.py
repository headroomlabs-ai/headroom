"""Centralized error/importance detection — thin Python shim over Rust.

Phase 3e.1 ported the keyword data + scoring logic to
``crates/headroom-core/src/signals/`` (see the trait architecture in
``signals/README.md``). This module is now a compatibility surface that:

1. Pulls the keyword tables out of Rust via
   ``headroom._core.keyword_registry_snapshot()`` so the Python side
   never re-declares them and cannot drift from the Rust source of
   truth.
2. Re-exports the legacy ``frozenset`` and compiled-regex names
   (``ERROR_KEYWORDS``, ``ERROR_PATTERN``, ``PRIORITY_PATTERNS_TEXT``,
   …) so the existing callers in ``search_compressor``,
   ``diff_compressor``, and ``intelligent_context`` keep working
   without same-PR refactors.
3. Delegates ``content_has_error_indicators`` to the Rust
   aho-corasick automaton.

Caller migration to the trait API happens in the per-compressor port
PRs that follow (Phase 3e.2 onward); this shim is the bridge until
those land.

# Bug fixes baked in

The Rust implementation fixes two bugs the Python originals carried:

* ``ERROR_KEYWORDS`` listed ``timeout``/``abort``/``denied``/
  ``rejected`` but ``ERROR_PATTERN`` regex omitted them. The
  recompiled pattern below now includes all four — lines like
  ``"FATAL: timeout connecting upstream"`` now flag as errors via
  the regex too.
* ``token`` was dropped from ``SECURITY_KEYWORDS`` (it false-positived
  on every reference to LLM tokens — input_tokens, tokens_saved, …).
"""

from __future__ import annotations

import re
from typing import cast

from headroom._core import (
    content_has_error_indicators as _rust_content_has_error_indicators,
)
from headroom._core import (
    keyword_registry_snapshot as _rust_keyword_registry_snapshot,
)
from headroom._core import (
    score_line as _rust_score_line,
)


def score_line(line: str, context: str = "text") -> tuple[str | None, float, float]:
    """Score `line` against the default Rust keyword detector.

    Returns ``(category | None, priority, confidence)``. ``category`` is
    one of ``error|warning|importance|security|markdown`` or ``None`` if
    nothing matched.

    Raises :class:`ValueError` for unknown context names. The Rust
    binding returns ``None`` for unknown contexts to dodge a
    pyo3-0.22 + clippy false positive on ``PyResult``-returning
    ``#[pyfunction]``s; this shim translates that into the explicit
    Python error every caller would expect.
    """
    result = _rust_score_line(line, context)
    if result is None:
        raise ValueError(f"unknown importance context: {context}")
    return cast("tuple[str | None, float, float]", result)


_REGISTRY: dict[str, list[str]] = _rust_keyword_registry_snapshot()


def _alternation(words: list[str]) -> str:
    """Compile a `\b(w1|w2|…)\b` regex source from the Rust-supplied list.

    The keywords are static (compiled once on import) so we don't need
    `re.escape` for the current set, but using it keeps the shim
    correct if a future Rust update adds a regex meta-character.
    """
    escaped = [re.escape(w) for w in words]
    return r"\b(" + "|".join(escaped) + r")\b"


# ─── Canonical keyword sets (pulled from Rust at import time) ───────────────

ERROR_KEYWORDS: frozenset[str] = frozenset(_REGISTRY["error"])

# Importance keywords historically included the error set — preserve that
# union so consumers iterating the set get the same membership as before.
IMPORTANCE_KEYWORDS: frozenset[str] = frozenset(
    list(_REGISTRY["error"]) + list(_REGISTRY["importance"]) + list(_REGISTRY["warning"])
)

SECURITY_KEYWORDS: frozenset[str] = frozenset(_REGISTRY["security"])

ERROR_INDICATOR_KEYWORDS: tuple[str, ...] = tuple(_REGISTRY["error_indicators"])


# ─── Compiled patterns ──────────────────────────────────────────────────────

ERROR_PATTERN: re.Pattern[str] = re.compile(_alternation(_REGISTRY["error"]), re.IGNORECASE)
WARNING_PATTERN: re.Pattern[str] = re.compile(_alternation(_REGISTRY["warning"]), re.IGNORECASE)
IMPORTANCE_PATTERN: re.Pattern[str] = re.compile(
    _alternation(_REGISTRY["importance"]), re.IGNORECASE
)
SECURITY_PATTERN: re.Pattern[str] = re.compile(_alternation(_REGISTRY["security"]), re.IGNORECASE)


# ─── Per-context priority pattern lists ─────────────────────────────────────

PRIORITY_PATTERNS_SEARCH: list[re.Pattern[str]] = [
    ERROR_PATTERN,
    WARNING_PATTERN,
    IMPORTANCE_PATTERN,
]

PRIORITY_PATTERNS_DIFF: list[re.Pattern[str]] = [
    ERROR_PATTERN,
    IMPORTANCE_PATTERN,
    SECURITY_PATTERN,
]

# Markdown structural prefixes: matched on whole lines, anchored with `^`.
# Pulled from Rust so the prefix table can't drift either.
PRIORITY_PATTERNS_TEXT: list[re.Pattern[str]] = [
    ERROR_PATTERN,
    IMPORTANCE_PATTERN,
    *(re.compile("^" + re.escape(prefix)) for prefix in _REGISTRY["markdown_prefixes"]),
]


# ─── Triage helper ──────────────────────────────────────────────────────────


def content_has_error_indicators(text: str) -> bool:
    """Fast keyword check — does `text` contain any error indicator?

    Substring match (no word boundary). Distinct from the strict line
    scoring in :mod:`headroom._core.score_line` because the triage
    callsite (e.g. message-signature classification) cares about
    Python tracebacks and similar substrings more than connection
    states.
    """
    return bool(_rust_content_has_error_indicators(text))


# Success-summary phrases from common build/test/lint tools that legitimately
# pair two indicator keywords (`error` + `fail`) while reporting a PASS, e.g.
# tsc's "Found 0 errors", jest's "0 failing" / "0 failures" / "0 failed",
# eslint's "0 problems (0 errors, 0 warnings)", or label:value summaries like
# "Failures: 0", "failed: 0", "Errors=0". Stripped before the keyword scan
# below so a clean JS/TS toolchain run doesn't get permanently protected from
# compression for the rest of a long coding session (issue #1696).
#
# `fail(?:ed|ing|ures?)?` covers fail/failed/failing/failure/failures — the
# keyword scan below matches the "fail" substring inside all of them, so the
# scrubber must strip all of them too, not just the forms literally named
# "failing"/"failure(s)".
_ZERO_RESULT_PATTERN = re.compile(
    # "0 errors" / "no failed" — count-first forms.
    r"\b(?:0|no)\s+(?:errors?|fail(?:ed|ing|ures?)?)\b"
    # "Errors: 0" / "failed=0" — label:value / label=value forms used by
    # other CI/test tools' summary lines.
    r"|\b(?:errors?|fail(?:ed|ing|ures?)?)\s*[:=]\s*0\b",
    re.IGNORECASE,
)


def content_has_strong_error_indicators(text: str) -> bool:
    """Stricter triage for compression-protection gates.

    :func:`content_has_error_indicators` substring-matches a single
    keyword, which false-positives on benign outputs that merely
    mention errors — grep hits, ``"errors": []`` JSON fields,
    ``error_handler.py`` filenames, ``except Exception`` in file
    reads. Protection gates exempt content from compression entirely,
    so a lax match there silently costs savings on the hot path.

    Require at least two DISTINCT indicator keywords: genuine failure
    output nearly always pairs the failure kind with a second
    indicator (``Traceback`` + ``ValueError``, ``fatal`` +
    ``crash``), while passing mentions rarely do. Misses here are
    safe — downstream compressors (LogCompressor) still preserve
    error lines.

    Before scanning, ``_ZERO_RESULT_PATTERN`` strips zero-result
    summary phrases (``"0 errors"``, ``"failed: 0"``, ...) so a
    passing build/test/lint run doesn't trip the two-keyword
    threshold just because its PASS summary happens to mention both
    "error" and "fail" at count zero (see issue #1696 — this was
    firing on nearly every request in a long JS/TS coding session
    and defeating compression almost entirely).
    """
    lowered = _ZERO_RESULT_PATTERN.sub(" ", text.lower())
    hits = 0
    for keyword in ERROR_INDICATOR_KEYWORDS:
        if keyword in lowered:
            hits += 1
            if hits >= 2:
                return True
    return False


__all__ = [
    "ERROR_KEYWORDS",
    "IMPORTANCE_KEYWORDS",
    "SECURITY_KEYWORDS",
    "ERROR_INDICATOR_KEYWORDS",
    "ERROR_PATTERN",
    "WARNING_PATTERN",
    "IMPORTANCE_PATTERN",
    "SECURITY_PATTERN",
    "PRIORITY_PATTERNS_SEARCH",
    "PRIORITY_PATTERNS_DIFF",
    "PRIORITY_PATTERNS_TEXT",
    "content_has_error_indicators",
    "content_has_strong_error_indicators",
    "score_line",
]
