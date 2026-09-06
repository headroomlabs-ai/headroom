"""Regression tests for the error/importance detection triage helpers."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from headroom.transforms.error_detection import content_has_strong_error_indicators

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_without_rust_core(body: str) -> subprocess.CompletedProcess[str]:
    """Run `body` in a child interpreter where `headroom._core` won't import.

    Has to be a child process: by the time this test runs, the rest of the
    suite has already imported both `headroom._core` and the modules under
    test, so poisoning `sys.modules` in-process would prove nothing.

    Setting ``sys.modules["headroom._core"] = None`` is Python's own
    "known-unimportable" marker, so the `from headroom._core import ...`
    statements raise `ImportError` exactly as they do when Windows Smart
    App Control blocks the compiled `_core.pyd` (issue #2918).
    """
    script = "import sys\nsys.modules['headroom._core'] = None\n" + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=300,
    )


def test_module_imports_without_the_rust_extension() -> None:
    """`error_detection` must import even when `headroom._core` cannot load.

    `content_router` imports this module at *its* module level, so a hard
    top-level `from headroom._core import ...` here was the single
    unguarded native edge on the proxy's whole startup path — it made
    `import headroom.proxy.server` fail outright, which made the proxy's
    documented `HEADROOM_REQUIRE_RUST_CORE=false` degraded mode
    unreachable (issue #2918).
    """
    result = _run_without_rust_core(
        """
        import headroom.transforms.error_detection as ed

        assert callable(ed.content_has_strong_error_indicators)
        print("OK")
        """
    )
    assert result.returncode == 0, (
        f"importing error_detection without the Rust core failed:\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_rust_backed_constants_still_fail_loudly_when_used() -> None:
    """Deferring the import must not turn into a silent Python fallback.

    There is no Python copy of the keyword tables, so touching one of the
    Rust-derived constants without the extension has to raise `ImportError`
    rather than hand back an empty or made-up set.
    """
    result = _run_without_rust_core(
        """
        import headroom.transforms.error_detection as ed

        for name in ("ERROR_PATTERN", "ERROR_KEYWORDS", "PRIORITY_PATTERNS_TEXT"):
            try:
                getattr(ed, name)
            except ImportError:
                continue
            raise AssertionError(f"{name} resolved without headroom._core")
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_compiled_patterns_are_shared_not_rebuilt_per_access() -> None:
    """The per-context lists must reuse the shared compiled patterns.

    The eager version built each pattern exactly once and put the same
    object in every list; the lazy version has to keep that property so
    repeated access stays free.
    """
    from headroom.transforms import error_detection as ed

    assert ed.PRIORITY_PATTERNS_SEARCH[0] is ed.ERROR_PATTERN
    assert ed.PRIORITY_PATTERNS_DIFF[0] is ed.ERROR_PATTERN
    assert ed.PRIORITY_PATTERNS_TEXT[1] is ed.IMPORTANCE_PATTERN
    assert ed.PRIORITY_PATTERNS_SEARCH is ed.PRIORITY_PATTERNS_SEARCH


def test_real_error_output_is_detected() -> None:
    text = "Traceback (most recent call last):\n  ...\nValueError: fatal error during load"
    assert content_has_strong_error_indicators(text)


def test_single_keyword_mention_is_not_flagged() -> None:
    # Only one distinct indicator keyword ("error") — should not trip the
    # two-keyword threshold.
    text = 'Wrote error_handler.py with an "errors": [] field.'
    assert not content_has_strong_error_indicators(text)


def test_tsc_passing_summary_is_not_flagged() -> None:
    # Regression for issue #1696: a clean `tsc` run mentions both "error"
    # and (via "0 failures" in a paired test run) "fail" while reporting
    # success. Previously this tripped the two-keyword heuristic and got
    # the message permanently protected from compression.
    text = "Found 0 errors. Watching for file changes.\nTests: 0 failures, 42 passed"
    assert not content_has_strong_error_indicators(text)


def test_eslint_passing_summary_is_not_flagged() -> None:
    text = "0 problems (0 errors, 0 warnings)\nno failing tests"
    assert not content_has_strong_error_indicators(text)


def test_zero_result_phrase_does_not_mask_a_real_second_error() -> None:
    # "0 errors" is stripped, but a genuine second distinct indicator
    # elsewhere in the same blob must still trigger protection.
    text = "0 errors from linter, but the build crashed with a fatal signal"
    assert content_has_strong_error_indicators(text)


def test_zero_failed_form_is_not_flagged() -> None:
    # Reviewer regression (PR #1740): "0 failed" wasn't covered by the
    # original pattern (only "failing"/"failure(s)"), so "failed" still
    # contributed a "fail" keyword hit alongside "0 errors" and tripped
    # the false positive this fix targets.
    text = "Found 0 errors\nTests: 0 failed, 42 passed"
    assert not content_has_strong_error_indicators(text)


def test_label_value_summary_formats_are_not_flagged() -> None:
    # Broader CI summary formats (not just "N word" / "word N"): label:value
    # and label=value pairs, in either error/fail order.
    for text in (
        "Failures: 0, Errors: 0",
        "failed: 0, errors: 0",
        "Errors=0 Failures=0",
    ):
        assert not content_has_strong_error_indicators(text), text
