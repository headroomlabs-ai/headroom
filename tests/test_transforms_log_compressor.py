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
