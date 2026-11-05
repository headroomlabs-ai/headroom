"""CMTrace (SCCM/Intune) deployment logs must detect as ``BUILD_OUTPUT``.

``_LOG_PATTERNS`` had no CMTrace signal. The format puts its timestamp in an
attribute *after* the message, so every anchored date/time/separator pattern
misses, and a record need not contain ERROR/WARN/INFO -- so ``_try_detect_log``
returned ``None`` at its ``pattern_matches == 0`` guard and the content fell
through to ``PLAIN_TEXT``.

A CMTrace record also satisfies the bare ``_SEARCH_RESULT_PATTERN``
(``^[^\\s:]+:\\d+:``), because ``]LOG]!><time="HH:MM:`` follows the message with
no intervening whitespace. Python is already protected from that by
``_is_search_result_line``'s path-shape guard, which rejects angle brackets --
the test below pins that behaviour so the protection cannot be lost. The Rust
port lacked the equivalent guard; this change adds it there.
"""

from __future__ import annotations

from headroom.transforms.content_detector import (
    ContentType,
    _is_search_result_line,
    detect_content_type,
)

_ATTRS = (
    'time="10:00:{sec:02d}.000-0" date="01-15-2026" '
    'component="ExampleCorp_SampleApp_1.0.0_x64_B1" context="SYSTEM" '
    'type="1" thread="4242" file="install.ps1"'
)

_MESSAGES = (
    "Starting deployment of SampleApp 1.0.0",
    "Detection rule evaluated, not installed",
    "Downloading package from content source",
    "Extracting files to target directory",
    "Registering application components",
    "Starting service SampleAppSvc",
    "Writing uninstall registry entries",
    "Deployment completed with exit code 0",
)


def _record(message: str, sec: int) -> str:
    """Build one CMTrace record.

    Args:
        message: Text inside the ``<![LOG[...]LOG]!>`` wrapper.
        sec: Seconds value for the record's ``time`` attribute.

    Returns:
        A single CMTrace record, with no trailing newline.
    """
    return f"<![LOG[{message}]LOG]!><{_ATTRS.format(sec=sec)}>"


def _cmtrace_records() -> list[str]:
    """Alternate ``===`` divider records with message records.

    Dividers are what made this content look like grep output, so they are the
    load-bearing half of the fixture, not decoration.

    Returns:
        Records in emission order, dividers first.
    """
    records: list[str] = []
    for i, message in enumerate(_MESSAGES):
        records.append(_record("=" * 75, i * 2))
        records.append(_record(message, i * 2 + 1))
    return records


def test_multiline_cmtrace_detected_as_build_output() -> None:
    """Newline-separated CMTrace detects as a log, not as grep output."""
    content = "\n".join(_cmtrace_records())

    result = detect_content_type(content)

    assert result.content_type is ContentType.BUILD_OUTPUT
    assert result.confidence >= 0.5


def test_single_line_cmtrace_detected_as_build_output() -> None:
    """Records run together with no separators -- how Intune actually writes them.

    Real deployment logs are emitted as one continuous stream: a 133 KB sample
    taken from a production estate held 571 records and 2 line feeds. This form
    never reached ``_try_detect_search`` (it exits at ``matching_lines < 2``),
    so it failed purely on defect 1 and fell through to ``PLAIN_TEXT``.
    """
    content = "".join(_cmtrace_records())

    result = detect_content_type(content)

    assert result.content_type is ContentType.BUILD_OUTPUT


def test_cmtrace_divider_is_not_treated_as_a_search_result_line() -> None:
    """Pin the path-shape guard that keeps CMTrace off the search path.

    The bare ``_SEARCH_RESULT_PATTERN`` *does* match a CMTrace divider; what
    rejects it is ``_prefix_looks_like_path``'s angle-bracket rule, one level
    up. Search is dispatched before logs, so losing that guard would shadow the
    ``_LOG_PATTERNS`` fix and route these logs to the search compressor -- which
    keeps only matching lines and drops the rest, making it data loss rather
    than a mis-labelling.
    """
    divider = _record("=" * 75, 0)

    assert _is_search_result_line(divider) is False


def test_grep_output_still_detected_as_search_results() -> None:
    """Guard against over-tightening: ordinary ``grep -n`` output is unchanged."""
    content = "\n".join(
        (
            "src/main.py:42:def process():",
            "src/util.py:13:    return None",
            "lib/x.py:7:class X:",
            "tests/test_a.py:3:    assert True",
        )
    )

    result = detect_content_type(content)

    assert result.content_type is ContentType.SEARCH_RESULTS


def test_extensionless_grep_paths_still_detected() -> None:
    """Extensionless filenames have no dot to lean on, so they are the tight case."""
    content = "\n".join(
        (
            "Makefile:12:\tpytest -q",
            "Dockerfile:3:RUN apt-get update",
            "Jenkinsfile:88:    sh 'make test'",
        )
    )

    result = detect_content_type(content)

    assert result.content_type is ContentType.SEARCH_RESULTS
