from __future__ import annotations

from types import SimpleNamespace

import pytest

from headroom.transforms.log_compressor import (
    LogCompressionResult,
    LogCompressor,
    LogCompressorConfig,
    LogFormat,
    LogLevel,
    LogLine,
)


def test_detect_parse_and_score_log_lines() -> None:
    compressor = LogCompressor(LogCompressorConfig(stack_trace_max_lines=2))
    pytest_lines = [
        "============================= test session starts =============================",
        "collected 2 items",
        "ERROR critical failure",
        "Traceback (most recent call last)",
        '  File "app.py", line 10',
        "",
        "2 failed, 1 warning",
    ]
    assert compressor._detect_format(pytest_lines) is LogFormat.PYTEST
    assert compressor._detect_format(["npm ERR! missing script"]) is LogFormat.NPM
    assert compressor._detect_format(["Compiling app", "warning: check this"]) is LogFormat.CARGO
    assert (
        compressor._detect_format(["PASS src/app.test.js", "Test Suites: 1 failed"])
        is LogFormat.JEST
    )
    assert compressor._detect_format(["make: *** fail", "gcc -o app app.c"]) is LogFormat.MAKE
    assert compressor._detect_format(["unclassified line"]) is LogFormat.GENERIC

    parsed = compressor._parse_lines(pytest_lines)
    assert parsed[0].is_summary is True
    assert parsed[2].level is LogLevel.ERROR
    assert parsed[3].is_stack_trace is True
    assert parsed[4].is_stack_trace is True
    assert parsed[6].is_summary is True
    assert compressor._score_line(LogLine(1, "warn", level=LogLevel.WARN)) == 0.5
    assert (
        compressor._score_line(
            LogLine(2, "error summary", level=LogLevel.ERROR, is_stack_trace=True, is_summary=True)
        )
        == 1.0
    )


def test_pytest_short_summary_helper_recognizes_only_section_entries() -> None:
    compressor = LogCompressor()
    lines = [
        "FAILED outside.py::test_before",
        "=== short test summary info ===",
        "FAILED tests/test_a.py::test_short",
        "ERROR tests/test_b.py::test_long - RuntimeError: boom",
        "=== 2 failed in 0.1s ===",
        "FAILED outside.py::test_after",
        "=== short test summary info ===",
        "FAILED tests/test_c.py::test_eof - AssertionError",
    ]

    parsed = compressor._parse_lines(lines)

    assert parsed[0].is_summary is False
    assert parsed[2].is_summary is True
    assert parsed[3].is_summary is True
    assert parsed[5].is_summary is False
    assert parsed[7].is_summary is True


def test_pytest_short_summary_helper_accepts_crlf_complete_separators() -> None:
    compressor = LogCompressor()
    lines = [
        "=== diagnostic: short test summary info unavailable\r",
        "FAILED outside.py::test_before\r",
        "=== short test summary info ===\r",
        "FAILED tests/test_a.py::test_short\r",
        "ERROR tests/test_b.py::test_long - RuntimeError: boom\r",
        "=== 2 failed in 0.1s ===\r",
        "FAILED outside.py::test_after\r",
        "=== short test summary info ===\r",
        "FAILED tests/test_c.py::test_eof - AssertionError\r",
    ]

    parsed = compressor._parse_lines(lines)

    assert parsed[1].is_summary is False
    assert parsed[3].is_summary is True
    assert parsed[4].is_summary is True
    assert parsed[6].is_summary is False
    assert parsed[8].is_summary is True
    assert parsed[3].content == lines[3]


def _malformed_short_summary_diagnostic_log() -> list[str]:
    lines = [f"INFO setup output {i}" for i in range(50)]
    lines.append("=== diagnostic: short test summary info unavailable")
    lines.extend(f"FAILED outside.py::test_{i}" for i in range(10))
    lines.extend(f"ERROR outside.py::test_{i}" for i in range(10, 20))
    return lines


@pytest.mark.parametrize("keep_summary_lines", [False, True])
def test_malformed_short_summary_diagnostic_does_not_affect_legacy_helpers(
    monkeypatch: pytest.MonkeyPatch, keep_summary_lines: bool
) -> None:
    compressor = LogCompressor(
        LogCompressorConfig(
            max_errors=2,
            error_context_lines=0,
            keep_summary_lines=keep_summary_lines,
            max_total_lines=1_000,
            enable_ccr=False,
        )
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "headroom.transforms.adaptive_sizer",
        SimpleNamespace(compute_optimal_k=lambda items, **kwargs: 1_000),
    )
    parsed = compressor._parse_lines(_malformed_short_summary_diagnostic_log())

    assert len(parsed) == 71
    assert all(not line.is_summary for line in parsed[51:])
    selected = compressor._select_lines(parsed)
    selected_lookalikes = [
        line.line_number for line in selected if line.level in (LogLevel.ERROR, LogLevel.FAIL)
    ]
    assert selected_lookalikes == [51, 60, 61, 70]

    output, _ = compressor._format_output(selected, parsed)
    assert "; omitted: " not in output
    if not keep_summary_lines:
        assert output == (
            "FAILED outside.py::test_0\n"
            "FAILED outside.py::test_9\n"
            "ERROR outside.py::test_10\n"
            "ERROR outside.py::test_19\n"
            "[67 lines omitted: 10 ERROR, 10 FAIL, 51 INFO]"
        )


def test_select_dedupe_add_context_and_format_output(monkeypatch: pytest.MonkeyPatch) -> None:
    compressor = LogCompressor(
        LogCompressorConfig(
            max_errors=2,
            max_warnings=1,
            error_context_lines=1,
            max_stack_traces=1,
            stack_trace_max_lines=2,
        )
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "headroom.transforms.adaptive_sizer",
        SimpleNamespace(compute_optimal_k=lambda items, **kwargs: 6),
    )
    log_lines = [
        LogLine(0, "info line", level=LogLevel.INFO, score=0.1),
        LogLine(1, "ERROR first", level=LogLevel.ERROR, score=1.0),
        LogLine(2, "context after first", level=LogLevel.UNKNOWN, score=0.1),
        LogLine(3, "WARNING /tmp/a/123 issue", level=LogLevel.WARN, score=0.5),
        LogLine(4, "WARNING /tmp/b/999 issue", level=LogLevel.WARN, score=0.5),
        LogLine(5, "FAIL final", level=LogLevel.FAIL, score=1.0),
        LogLine(6, "Traceback (most recent call last)", is_stack_trace=True, score=0.4),
        LogLine(7, '  File "app.py", line 2', is_stack_trace=True, score=0.4),
        LogLine(8, "1 failed, 1 warning", is_summary=True, score=0.5),
    ]
    selected = compressor._select_lines(log_lines)
    assert [line.line_number for line in selected] == [1, 3, 4, 5, 6, 8]

    assert compressor._select_with_first_last(log_lines[:2], max_count=5) == log_lines[:2]
    many_errors = [
        LogLine(10, "first", level=LogLevel.ERROR, score=0.1),
        LogLine(11, "mid", level=LogLevel.ERROR, score=0.9),
        LogLine(12, "last", level=LogLevel.ERROR, score=0.2),
    ]
    trimmed = compressor._select_with_first_last(many_errors, max_count=2)
    assert trimmed == [many_errors[0], many_errors[2]]
    # fixed_in_3e5: conservative dedupe preserves message prefix (everything
    # before the first `:` or `=`), so warnings without a colon keep their
    # full content as the dedupe key. The two lines below have different
    # paths/numbers and no `:`, so they DON'T collapse anymore — Python's
    # pre-3e5 aggressive normalization treated them as duplicates, masking
    # distinct error categories.
    distinct = compressor._dedupe_similar(log_lines[3:5])
    assert len(distinct) == 2
    # Same dedupe IS triggered when the prefix matches (lines have a colon).
    similar = compressor._dedupe_similar(
        [
            LogLine(20, "warning: file /tmp/a/123 issue", level=LogLevel.WARN),
            LogLine(21, "warning: file /tmp/b/999 issue", level=LogLevel.WARN),
        ]
    )
    assert len(similar) == 1

    output, stats = compressor._format_output(selected, log_lines)
    assert stats == {
        "errors": 1,
        "fails": 1,
        "warnings": 2,
        "info": 1,
        "total": 9,
        "selected": 6,
    }
    assert output.endswith("[3 lines omitted: 1 ERROR, 1 FAIL, 2 WARN, 1 INFO]")


@pytest.mark.parametrize(
    ("keep_summary_lines", "expected"),
    # True: the totals line (7) is reserved, and entries win the remaining ties.
    [(True, [4, 5, 7]), (False, [0, 1, 4])],
)
def test_short_summary_global_cap_precedence_matches_rust(
    monkeypatch: pytest.MonkeyPatch, keep_summary_lines: bool, expected: list[int]
) -> None:
    compressor = LogCompressor(
        LogCompressorConfig(
            max_errors=10,
            error_context_lines=0,
            keep_summary_lines=keep_summary_lines,
            max_total_lines=3,
        )
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "headroom.transforms.adaptive_sizer",
        SimpleNamespace(compute_optimal_k=lambda items, **kwargs: 3),
    )
    parsed = compressor._parse_lines(
        [
            "FAILED traceback.py::test_early_one\r",
            "FAILED traceback.py::test_early_two\r",
            "ordinary output\r",
            "=== short test summary info ===\r",
            "FAILED tests/test_summary.py::test_one\r",
            "ERROR tests/test_summary.py::test_two\r",
            "FAILED tests/test_summary.py::test_three\r",
            "=== 3 failed in 0.1s ===\r",
        ]
    )

    assert [line.line_number for line in compressor._select_lines(parsed)] == expected


def test_legacy_cap_reserves_totals_and_first_error_line(monkeypatch: pytest.MonkeyPatch) -> None:
    compressor = LogCompressor(
        LogCompressorConfig(max_errors=10, error_context_lines=0, max_total_lines=4)
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "headroom.transforms.adaptive_sizer",
        SimpleNamespace(compute_optimal_k=lambda items, **kwargs: 4),
    )
    parsed = compressor._parse_lines(
        [
            "E       AssertionError: first",
            "E       AssertionError: second",
            "=== short test summary info ===",
            "FAILED t.py::a",
            "FAILED t.py::b",
            "FAILED t.py::c",
            "FAILED t.py::d",
            "=== 4 failed in 0.1s ===",
        ]
    )

    # The first E line (0) and the totals line (7) are reserved; entries fill the rest.
    assert [line.line_number for line in compressor._select_lines(parsed)] == [0, 3, 4, 7]


@pytest.mark.parametrize(
    ("count", "suffix"),
    [
        (
            3,
            "; omitted: tests/test_ids.py::test_0, tests/test_ids.py::test_1, "
            "tests/test_ids.py::test_2",
        ),
        (
            5,
            "; omitted: tests/test_ids.py::test_0, tests/test_ids.py::test_1, "
            "tests/test_ids.py::test_2, tests/test_ids.py::test_3, tests/test_ids.py::test_4",
        ),
        (
            7,
            "; omitted: tests/test_ids.py::test_0, tests/test_ids.py::test_1, "
            "tests/test_ids.py::test_2, tests/test_ids.py::test_3, tests/test_ids.py::test_4, "
            "+2 more",
        ),
    ],
)
def test_short_summary_omission_naming_is_bounded(count: int, suffix: str) -> None:
    compressor = LogCompressor(LogCompressorConfig(keep_summary_lines=False))
    contents = ["=== short test summary info ===\r"] + [
        f"FAILED tests/test_ids.py::test_{i}\r" for i in range(count)
    ]
    contents.append("=== failures complete ===\r")
    all_lines = compressor._parse_lines(contents)
    selected = [all_lines[0], all_lines[-1]]

    output, _ = compressor._format_output(selected, all_lines)

    assert output.endswith(f"[{count} lines omitted: {count} FAIL, 1 INFO{suffix}]")


def test_short_summary_omission_naming_uses_ids_and_preserves_duplicates() -> None:
    compressor = LogCompressor(LogCompressorConfig(keep_summary_lines=False))
    all_lines = compressor._parse_lines(
        [
            "=== short test summary info ===",
            "ERROR tests/test_ids.py::test_error - RuntimeError: boom",
            "FAILED tests/test_ids.py::test_repeat",
            "FAILED tests/test_ids.py::test_repeat - AssertionError",
            "FAILED tests/test_ids.py::test_kept",
            "=== failures complete ===",
        ]
    )

    output, stats = compressor._format_output([all_lines[0], all_lines[4], all_lines[5]], all_lines)

    assert output.endswith(
        "[3 lines omitted: 1 ERROR, 3 FAIL, 1 INFO; omitted: tests/test_ids.py::test_error, "
        "tests/test_ids.py::test_repeat, tests/test_ids.py::test_repeat]"
    )
    assert stats["errors"] == 1
    assert stats["fails"] == 3


def test_log_compressor_compress_and_ccr_paths() -> None:
    """Phase 3e.5: `compress()` is now a single Rust call, so this test
    exercises end-to-end behavior instead of monkeypatching internal
    helpers (which the old orchestration relied on)."""
    compressor = LogCompressor(LogCompressorConfig(enable_ccr=True, min_lines_for_ccr=3))
    short = compressor.compress("a\nb")
    # Below min_lines_for_ccr (3 lines from "a\nb" = 2 lines) → verbatim
    assert short.format_detected is LogFormat.GENERIC
    assert short.compression_ratio == 1.0

    # Real npm log to exercise format detection + CCR end-to-end. Build
    # a long enough corpus so compute_optimal_k drops below the
    # min_compression_ratio_for_ccr=0.5 threshold.
    npm_lines = ["npm WARN deprecated x"] * 30 + ["npm ERR! something broke"] * 5
    npm_content = "\n".join(npm_lines)
    result = compressor.compress(npm_content)
    assert result.format_detected is LogFormat.NPM
    assert result.original_line_count == 35
    assert result.compressed_line_count < result.original_line_count

    # Short input below min_lines_for_ccr returns verbatim with ratio 1.0
    # (no compression attempted).
    too_short = compressor.compress("x\ny")
    assert too_short.compression_ratio == 1.0
    assert too_short.cache_key is None


def _pytest_reproduction(failure_count: int, line_ending: str = "\n") -> str:
    lines = [f"pytest setup output {i}" for i in range(20)]
    for i in range(20):
        lines.extend(
            [
                f"FAILED traceback detail {i}",
                f"E       AssertionError: failure detail {i}",
            ]
        )
    lines.append("=========================== short test summary info ===========================")
    lines.extend(
        f"FAILED tests/test_generated.py::test_case{i} - AssertionError: failure {i}"
        for i in range(failure_count)
    )
    lines.append(
        f"========================= {failure_count} failed in 1.00s ========================="
    )
    return line_ending.join(lines)


def _retained_or_named(output: str, node_id: str) -> bool:
    for line in output.splitlines():
        if line.startswith(("FAILED ", "ERROR ")):
            entry_id = line.split(" ", 1)[1].split(" - ", 1)[0]
            if entry_id == node_id:
                return True
        if line.startswith("[") and "; omitted: " in line:
            named = line.split("; omitted: ", 1)[1].removesuffix("]").split(", ")
            if node_id in named:
                return True
    return False


@pytest.mark.parametrize(("failure_count", "first_regression"), [(16, 13), (20, 13)])
def test_real_compress_preserves_or_names_issue_3814_middle_failures(
    failure_count: int, first_regression: int
) -> None:
    content = _pytest_reproduction(failure_count)
    assert len(content.splitlines()) >= 50
    compressor = LogCompressor(LogCompressorConfig(enable_ccr=False))

    result = compressor.compress(content)

    assert result.compressed_line_count < result.original_line_count
    assert result.compression_ratio < 1.0
    last_regression = 13 if failure_count == 16 else 17
    for i in range(first_regression, last_regression + 1):
        node_id = f"tests/test_generated.py::test_case{i}"
        assert _retained_or_named(result.compressed, node_id), node_id


def _pytest_issue_log(failure_count: int) -> str:
    """Build the issue #3814 reproduction with a distinct E message per failure."""
    lines = [
        "============================= test session starts =============================",
        "collected 400 items",
        "",
    ]
    lines += [
        f"tests/test_module_{i:02d}.py ..........................  [ {i:2d}%]" for i in range(40)
    ]
    lines.append("=================================== FAILURES ==================================")
    for i in range(1, failure_count + 1):
        lines += [
            f"____________________ test_case{i:03d} ____________________",
            "",
            ">       assert result == expected",
            f"E       AssertionError: mismatch in test_case{i:03d}",
            "",
            "tests/t.py:42: AssertionError",
        ]
    lines.append("=========================== short test summary info ===========================")
    lines += [
        f"FAILED tests/t.py::test_case{i:03d} - AssertionError: mismatch"
        for i in range(1, failure_count + 1)
    ]
    lines.append(f"=============== {failure_count} failed, 380 passed in 41.02s ==============")
    return "\n".join(lines)


@pytest.mark.parametrize("failure_count", [60, 200])
def test_real_compress_binding_cap_keeps_totals_and_first_error_line(failure_count: int) -> None:
    out = LogCompressor(LogCompressorConfig(enable_ccr=False)).compress(
        _pytest_issue_log(failure_count)
    )
    compressed = out.compressed

    assert f"{failure_count} failed, 380 passed" in compressed
    assert "E       AssertionError: mismatch in test_case001" in compressed
    # Every entry is still kept or named: kept + listed + K == total.
    kept = sum(1 for line in compressed.splitlines() if line.startswith("FAILED tests/t.py::"))
    named = compressed.splitlines()[-1].split("; omitted: ", 1)[1].removesuffix("]").split(", ")
    overflow = int(named[-1][1:].removesuffix(" more")) if named[-1].startswith("+") else 0
    listed = len(named) - (1 if overflow else 0)
    assert kept + listed + overflow == failure_count


def test_real_compress_keeps_all_crlf_short_summary_entries_with_non_binding_cap() -> None:
    content = _pytest_reproduction(20, "\r\n")
    assert len(content.splitlines()) >= 50
    compressor = LogCompressor(
        LogCompressorConfig(
            max_errors=2,
            error_context_lines=0,
            max_total_lines=1_000,
            enable_ccr=False,
        )
    )

    result = compressor.compress(content, bias=1_000)

    assert result.compressed_line_count < result.original_line_count
    retained_ids = {
        line.split(" ", 1)[1].split(" - ", 1)[0]
        for line in result.compressed.splitlines()
        if line.startswith(("FAILED ", "ERROR "))
    }
    assert {f"tests/test_generated.py::test_case{i}" for i in range(20)}.issubset(retained_ids)


def test_real_compress_ignores_malformed_short_summary_diagnostic() -> None:
    content = "\n".join(_malformed_short_summary_diagnostic_log())
    compressor = LogCompressor(
        LogCompressorConfig(
            max_errors=2,
            error_context_lines=0,
            keep_summary_lines=False,
            max_total_lines=1_000,
            enable_ccr=False,
        )
    )

    result = compressor.compress(content, bias=1_000)

    assert result.compressed_line_count < result.original_line_count
    assert result.compressed == (
        "FAILED outside.py::test_0\n"
        "FAILED outside.py::test_9\n"
        "ERROR outside.py::test_10\n"
        "ERROR outside.py::test_19\n"
        "[67 lines omitted: 10 ERROR, 10 FAIL, 51 INFO]"
    )


def test_store_in_ccr_and_result_properties(monkeypatch: pytest.MonkeyPatch) -> None:
    compressor = LogCompressor()
    monkeypatch.setitem(
        __import__("sys").modules,
        "headroom.cache.compression_store",
        SimpleNamespace(
            get_compression_store=lambda: SimpleNamespace(
                store=lambda original, compressed, original_item_count=0: "stored-log"
            )
        ),
    )
    assert compressor._store_in_ccr("orig", "comp", 10) == "stored-log"

    def broken_store():
        raise RuntimeError("boom")

    monkeypatch.setitem(
        __import__("sys").modules,
        "headroom.cache.compression_store",
        SimpleNamespace(get_compression_store=broken_store),
    )
    assert compressor._store_in_ccr("orig", "comp", 10) is None

    result = LogCompressionResult(
        compressed="small",
        original="this is a substantially longer log body",
        original_line_count=20,
        compressed_line_count=5,
        format_detected=LogFormat.GENERIC,
        compression_ratio=0.25,
    )
    assert result.tokens_saved_estimate > 0
    assert result.lines_omitted == 15
