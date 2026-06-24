"""Phase 3e.1: parity contract for the Rust-backed error_detection shim.

The Python regex registry was retired in favor of recompiling regex from
the keyword tables exposed by the Rust `headroom._core.signals` module.
These tests pin:

* The shim re-exports the legacy frozenset and Pattern names callers
  rely on (search_compressor, intelligent_context).
* Two bug fixes from the Rust port land in the Python regex too:
    1. ERROR_PATTERN now matches abort/timeout/denied/rejected (was a
       drift between ERROR_KEYWORDS and the compiled regex).
    2. SECURITY_KEYWORDS no longer includes "token" (false-positives on
       LLM-token references in our own product).
* `content_has_error_indicators` matches the Python triage semantics
  (substring, no word boundary) for the canonical indicator set.
* The signals trait is reachable from Python via three thin functions
  (`score_line`, `content_has_error_indicators`, `keyword_registry_snapshot`).
"""

from __future__ import annotations

import pytest


def test_legacy_re_exports_present():
    from headroom.transforms import error_detection as ed

    # frozenset names the existing callers import directly
    assert isinstance(ed.ERROR_KEYWORDS, frozenset)
    assert isinstance(ed.IMPORTANCE_KEYWORDS, frozenset)
    assert isinstance(ed.SECURITY_KEYWORDS, frozenset)
    assert isinstance(ed.ERROR_INDICATOR_KEYWORDS, tuple)

    # Pattern objects must still be re.Pattern so callers can do .search()
    import re

    assert isinstance(ed.ERROR_PATTERN, re.Pattern)
    assert isinstance(ed.WARNING_PATTERN, re.Pattern)
    assert isinstance(ed.IMPORTANCE_PATTERN, re.Pattern)
    assert isinstance(ed.SECURITY_PATTERN, re.Pattern)

    # Per-context priority lists used by search/diff/text compressors
    assert all(isinstance(p, re.Pattern) for p in ed.PRIORITY_PATTERNS_SEARCH)
    assert all(isinstance(p, re.Pattern) for p in ed.PRIORITY_PATTERNS_DIFF)
    assert all(isinstance(p, re.Pattern) for p in ed.PRIORITY_PATTERNS_TEXT)


def test_bug_fix_error_regex_now_matches_canonical_keyword_set():
    """fixed_in_3e1: ERROR_PATTERN regex used to omit timeout/abort/denied/rejected
    even though ERROR_KEYWORDS canonically included them."""
    from headroom.transforms.error_detection import ERROR_KEYWORDS, ERROR_PATTERN

    for keyword in ("timeout", "abort", "denied", "rejected"):
        assert keyword in ERROR_KEYWORDS, f"{keyword} must stay in ERROR_KEYWORDS"
        assert ERROR_PATTERN.search(f"FATAL: {keyword} occurred"), (
            f"ERROR_PATTERN must now flag {keyword!r}"
        )


def test_bug_fix_security_keywords_dropped_token():
    """fixed_in_3e1: 'token' false-positived on input_tokens/tokens_saved/etc.
    in an LLM-proxy product. Dropped from the security set so the security
    pattern stops misclassifying our own metric output."""
    from headroom.transforms.error_detection import SECURITY_KEYWORDS, SECURITY_PATTERN

    assert "token" not in SECURITY_KEYWORDS
    assert "auth" in SECURITY_KEYWORDS  # the real security signal
    assert SECURITY_PATTERN.search("missing auth header") is not None
    assert SECURITY_PATTERN.search("input_tokens=512 output_tokens=128") is None


def test_content_has_error_indicators_lax_substring_semantics():
    from headroom.transforms.error_detection import content_has_error_indicators

    # Python ERROR_INDICATOR_KEYWORDS includes "traceback" — must still fire
    assert content_has_error_indicators("Traceback (most recent call last):")
    # Substring (no word-boundary) match preserved — "errored" matches "error"
    assert content_has_error_indicators("the request errored out")
    # Genuine non-match
    assert not content_has_error_indicators("everything is fine")


def test_rust_signals_bridge_score_line_diff_context():
    """The Phase 3g pipeline will consume the trait API directly. Today
    we cover the bridge surface so a future change can't silently break
    it."""
    from headroom.transforms.error_detection import score_line

    cat, priority, confidence = score_line("FATAL: timeout connecting", "diff")
    assert cat == "error"
    assert priority > 0.9
    assert confidence > 0.5

    cat_neutral, _, conf_neutral = score_line("the quick brown fox", "text")
    assert cat_neutral is None
    assert conf_neutral == 0.0


def test_rust_signals_bridge_unknown_context_raises():
    from headroom.transforms.error_detection import score_line

    with pytest.raises(ValueError, match="unknown importance context"):
        score_line("anything", "not_a_real_context")


def test_raw_rust_score_line_returns_none_on_unknown_context():
    """The raw `headroom._core.score_line` returns `None` for unknown
    contexts (the Python shim is responsible for translating to
    ValueError). Pin this contract so a future change can't shift the
    error-handling boundary unobserved."""
    from headroom._core import score_line as _raw

    assert _raw("anything", "not_a_real_context") is None
    result = _raw("ERROR: test", "diff")
    assert result is not None and result[0] == "error"


def test_keyword_registry_snapshot_has_dropped_token_and_added_indicators():
    from headroom._core import keyword_registry_snapshot

    snapshot = keyword_registry_snapshot()
    assert "token" not in snapshot["security"]
    assert "auth" in snapshot["security"]
    assert "timeout" in snapshot["error"]
    assert "traceback" in snapshot["error_indicators"]
    # Markdown prefixes must include at least the canonical four
    assert "# " in snapshot["markdown_prefixes"]
    assert "> " in snapshot["markdown_prefixes"]


def test_python_regex_recompiled_from_rust_keyword_tables():
    """The Python shim recompiles regex from keyword data Rust hands it.
    This guards against drift: if Rust and Python keyword sets ever
    diverge, this fails the suite."""
    from headroom._core import keyword_registry_snapshot
    from headroom.transforms.error_detection import (
        ERROR_KEYWORDS,
        IMPORTANCE_KEYWORDS,
        SECURITY_KEYWORDS,
    )

    rust = keyword_registry_snapshot()
    assert ERROR_KEYWORDS == frozenset(rust["error"])
    assert SECURITY_KEYWORDS == frozenset(rust["security"])
    # IMPORTANCE_KEYWORDS is a union of error + importance + warning sets
    expected_importance = (
        frozenset(rust["error"]) | frozenset(rust["importance"]) | frozenset(rust["warning"])
    )
    assert IMPORTANCE_KEYWORDS == expected_importance
