"""Comprehensive tests for log_compressor.py.

Tests cover:
1. Detection of different log formats (pytest, npm, cargo, make, jest, generic)
2. Line extraction and deduplication
3. Compression ratios
4. Edge cases
"""

from headroom.transforms.log_compressor import (
    LogCompressionResult,
    LogCompressor,
    LogCompressorConfig,
    LogFormat,
    LogLevel,
    LogLine,
)


class TestLogFormatDetection:
    """Tests for detecting different log formats."""

    def test_detect_pytest_format(self):
        """Pytest output is detected correctly."""
        content = """============================= test session starts ==============================
platform darwin -- Python 3.11.0
collected 15 items

tests/test_foo.py::test_basic PASSED [  6%]
tests/test_foo.py::test_edge FAILED [ 13%]

=================================== FAILURES ===================================
tests/test_foo.py::test_edge - AssertionError

=========================== short test summary info ============================
FAILED tests/test_foo.py::test_edge
========================= 1 failed, 14 passed =========================
"""
        compressor = LogCompressor()
        lines = content.split("\n")
        detected = compressor._detect_format(lines)
        assert detected == LogFormat.PYTEST

    def test_detect_npm_format(self):
        """npm output is detected correctly."""
        content = """npm WARN deprecated package@1.0.0: This package is deprecated
npm WARN deprecated another@2.0.0: Obsolete
npm ERR! code ERESOLVE
npm ERR! ERESOLVE unable to resolve dependency tree
npm info using npm@9.0.0
> added 150 packages in 5s
"""
        compressor = LogCompressor()
        lines = content.split("\n")
        detected = compressor._detect_format(lines)
        assert detected == LogFormat.NPM

    def test_detect_cargo_format(self):
        """Cargo/rustc output is detected correctly."""
        content = """   Compiling myproject v0.1.0 (/path/to/project)
warning: unused variable: `x`
 --> src/main.rs:5:9
  |
5 |     let x = 5;
  |         ^ help: if this is intentional, prefix it with an underscore: `_x`
  |
  = note: `#[warn(unused_variables)]` on by default

error[E0382]: borrow of moved value: `s`
    Finished dev [unoptimized + debuginfo] target(s) in 0.50s
     Running `target/debug/myproject`
"""
        compressor = LogCompressor()
        lines = content.split("\n")
        detected = compressor._detect_format(lines)
        assert detected == LogFormat.CARGO

    def test_detect_make_format(self):
        """make/gcc output is detected correctly."""
        content = """make[1]: Entering directory '/path/to/project'
gcc -c -o main.o main.c
gcc -c -o utils.o utils.c
make[1]: *** [Makefile:10: utils.o] Error 1
make: *** [Makefile:5: all] Error 2
g++ -Wall -o program main.cpp utils.cpp
"""
        compressor = LogCompressor()
        lines = content.split("\n")
        detected = compressor._detect_format(lines)
        assert detected == LogFormat.MAKE

    def test_detect_jest_format(self):
        """Jest output is detected correctly."""
        content = """PASS src/components/Button.test.js
FAIL src/utils/helpers.test.ts
  Test Suites: 1 failed, 1 passed, 2 total
  Tests:       2 failed, 10 passed, 12 total
"""
        compressor = LogCompressor()
        lines = content.split("\n")
        detected = compressor._detect_format(lines)
        assert detected == LogFormat.JEST

    def test_detect_generic_format(self):
        """Generic log format is detected for unrecognized output."""
        content = """INFO Starting application
DEBUG Initializing components
WARNING Low memory
ERROR Connection timeout
CRITICAL System failure
"""
        compressor = LogCompressor()
        lines = content.split("\n")
        detected = compressor._detect_format(lines)
        assert detected == LogFormat.GENERIC

    def test_detect_empty_returns_generic(self):
        """Empty or minimal input returns GENERIC."""
        compressor = LogCompressor()
        assert compressor._detect_format([]) == LogFormat.GENERIC
        assert compressor._detect_format(["random line"]) == LogFormat.GENERIC


class TestLogLevelDetection:
    """Tests for log level detection in lines."""

    def test_detect_error_levels(self):
        """ERROR, FATAL, CRITICAL are detected."""
        compressor = LogCompressor()

        error_lines = [
            "ERROR: something went wrong",
            "error: file not found",
            "Error: Invalid input",
            "FATAL: system crash",
            "fatal error occurred",
            "CRITICAL: database down",
        ]

        for line in error_lines:
            log_lines = compressor._parse_lines([line])
            assert log_lines[0].level == LogLevel.ERROR, f"Failed for: {line}"

    def test_detect_fail_levels(self):
        """FAIL, FAILED are detected."""
        compressor = LogCompressor()

        fail_lines = [
            "FAIL tests/test_foo.py",
            "FAILED to connect",
            "Test failed",
        ]

        for line in fail_lines:
            log_lines = compressor._parse_lines([line])
            assert log_lines[0].level == LogLevel.FAIL, f"Failed for: {line}"

    def test_detect_warn_levels(self):
        """WARN, WARNING are detected."""
        compressor = LogCompressor()

        warn_lines = [
            "WARN: deprecated function",
            "WARNING: low disk space",
            "warning: unused variable",
        ]

        for line in warn_lines:
            log_lines = compressor._parse_lines([line])
            assert log_lines[0].level == LogLevel.WARN, f"Failed for: {line}"

    def test_detect_info_debug_trace(self):
        """INFO, DEBUG, TRACE are detected."""
        compressor = LogCompressor()

        test_cases = [
            ("INFO: starting process", LogLevel.INFO),
            ("info starting", LogLevel.INFO),
            ("DEBUG: variable x = 5", LogLevel.DEBUG),
            ("debug mode enabled", LogLevel.DEBUG),
            ("TRACE: entering function", LogLevel.TRACE),
        ]

        for line, expected_level in test_cases:
            log_lines = compressor._parse_lines([line])
            assert log_lines[0].level == expected_level, f"Failed for: {line}"

    def test_unknown_level_default(self):
        """Lines without level markers default to UNKNOWN."""
        compressor = LogCompressor()
        log_lines = compressor._parse_lines(["Just some regular text"])
        assert log_lines[0].level == LogLevel.UNKNOWN


class TestStackTraceDetection:
    """Tests for stack trace detection."""

    def test_detect_python_traceback(self):
        """Python traceback is detected."""
        content = """Traceback (most recent call last):
  File "main.py", line 42, in process
    result = compute(data)
  File "utils.py", line 15, in compute
    return data / 0
ZeroDivisionError: division by zero
"""
        compressor = LogCompressor()
        log_lines = compressor._parse_lines(content.split("\n"))

        # First several lines should be marked as stack trace
        stack_trace_count = sum(1 for line in log_lines if line.is_stack_trace)
        assert stack_trace_count > 0

    def test_detect_javascript_stack_trace(self):
        """JavaScript stack trace is detected."""
        content = """Error: Connection failed
    at Connection.connect (src/db.js:42:15)
    at async main (src/index.js:10:5)
"""
        compressor = LogCompressor()
        log_lines = compressor._parse_lines(content.split("\n"))

        stack_trace_count = sum(1 for line in log_lines if line.is_stack_trace)
        assert stack_trace_count > 0

    def test_detect_rust_error_location(self):
        """Rust error location is detected."""
        content = """error[E0382]: borrow of moved value: `s`
 --> src/main.rs:5:13
  |
3 |     let s = String::from("hello");
  |         - move occurs
"""
        compressor = LogCompressor()
        log_lines = compressor._parse_lines(content.split("\n"))

        stack_trace_count = sum(1 for line in log_lines if line.is_stack_trace)
        assert stack_trace_count > 0


class TestLineDeduplication:
    """Tests for warning/line deduplication."""

    def test_dedupe_identical_warnings(self):
        """Identical warnings are deduplicated."""
        compressor = LogCompressor()

        lines = [
            LogLine(line_number=1, content="WARNING: unused variable 'x'", level=LogLevel.WARN),
            LogLine(line_number=2, content="WARNING: unused variable 'x'", level=LogLevel.WARN),
            LogLine(line_number=3, content="WARNING: unused variable 'x'", level=LogLevel.WARN),
        ]

        deduped = compressor._dedupe_similar(lines)
        assert len(deduped) == 1

    def test_dedupe_similar_with_numbers(self):
        """Similar warnings with different numbers are deduplicated."""
        compressor = LogCompressor()

        lines = [
            LogLine(line_number=1, content="WARNING: error at line 10", level=LogLevel.WARN),
            LogLine(line_number=2, content="WARNING: error at line 20", level=LogLevel.WARN),
            LogLine(line_number=3, content="WARNING: error at line 30", level=LogLevel.WARN),
        ]

        deduped = compressor._dedupe_similar(lines)
        # Numbers normalized to "N", so all three are treated as identical pattern
        assert len(deduped) == 1

    def test_dedupe_similar_with_paths(self):
        """Similar warnings with different paths are deduplicated.

        Note: The path regex /[\\w/]+/ requires paths to end with '/'.
        Paths like '/path/to/' will be normalized, but '/path/to/file' won't
        be fully normalized because 'file' doesn't end with '/'.
        """
        compressor = LogCompressor()

        # Paths ending with / are normalized
        lines = [
            LogLine(line_number=1, content="WARNING: in /path/to/ error", level=LogLevel.WARN),
            LogLine(line_number=2, content="WARNING: in /other/dir/ error", level=LogLevel.WARN),
            LogLine(line_number=3, content="WARNING: in /another/path/ error", level=LogLevel.WARN),
        ]

        deduped = compressor._dedupe_similar(lines)
        # Paths normalized to /PATH/, so all three are treated as identical pattern
        assert len(deduped) == 1

    def test_keeps_different_warnings(self):
        """Different warnings are preserved."""
        compressor = LogCompressor()

        lines = [
            LogLine(line_number=1, content="WARNING: unused variable", level=LogLevel.WARN),
            LogLine(line_number=2, content="WARNING: deprecated function", level=LogLevel.WARN),
            LogLine(line_number=3, content="WARNING: missing docstring", level=LogLevel.WARN),
        ]

        deduped = compressor._dedupe_similar(lines)
        assert len(deduped) == 3


class TestLineScoring:
    """Tests for line importance scoring."""

    def test_error_lines_score_highest(self):
        """ERROR and FAIL lines get highest scores."""
        compressor = LogCompressor()

        error_line = LogLine(line_number=1, content="ERROR: critical", level=LogLevel.ERROR)
        fail_line = LogLine(line_number=2, content="FAILED test", level=LogLevel.FAIL)
        info_line = LogLine(line_number=3, content="INFO: normal", level=LogLevel.INFO)

        error_score = compressor._score_line(error_line)
        fail_score = compressor._score_line(fail_line)
        info_score = compressor._score_line(info_line)

        assert error_score > info_score
        assert fail_score > info_score

    def test_stack_trace_boost(self):
        """Stack trace lines get boosted score."""
        compressor = LogCompressor()

        regular = LogLine(line_number=1, content="some line", level=LogLevel.UNKNOWN)
        stack_trace = LogLine(
            line_number=2, content="  File 'x.py'", level=LogLevel.UNKNOWN, is_stack_trace=True
        )

        assert compressor._score_line(stack_trace) > compressor._score_line(regular)

    def test_summary_line_boost(self):
        """Summary lines get boosted score."""
        compressor = LogCompressor()

        regular = LogLine(line_number=1, content="some line", level=LogLevel.UNKNOWN)
        summary = LogLine(
            line_number=2, content="10 passed, 2 failed", level=LogLevel.UNKNOWN, is_summary=True
        )

        assert compressor._score_line(summary) > compressor._score_line(regular)


class TestCompressionBehavior:
    """Tests for overall compression behavior."""

    def test_small_log_passthrough(self):
        """Logs smaller than threshold pass through unchanged."""
        content = "INFO: Starting\nINFO: Done"

        compressor = LogCompressor(config=LogCompressorConfig(min_lines_for_ccr=100))
        result = compressor.compress(content)

        assert result.compression_ratio == 1.0
        assert result.compressed == content
        assert result.original_line_count == 2

    def test_large_log_compressed(self):
        """Large logs are compressed."""
        lines = [f"INFO: Processing item {i}" for i in range(200)]
        lines.append("ERROR: Failed at item 100")
        content = "\n".join(lines)

        compressor = LogCompressor(
            config=LogCompressorConfig(
                min_lines_for_ccr=50,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        assert result.compression_ratio < 1.0
        assert result.compressed_line_count < result.original_line_count
        # Error is preserved
        assert "ERROR: Failed" in result.compressed

    def test_keeps_first_and_last_errors(self):
        """First and last errors are preserved."""
        lines = [f"INFO: item {i}" for i in range(100)]
        lines[10] = "ERROR: first error"
        lines[50] = "ERROR: middle error"
        lines[90] = "ERROR: last error"
        content = "\n".join(lines)

        compressor = LogCompressor(
            config=LogCompressorConfig(
                min_lines_for_ccr=50,
                keep_first_error=True,
                keep_last_error=True,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        assert "first error" in result.compressed
        assert "last error" in result.compressed

    def test_summary_lines_preserved(self):
        """Summary lines are always preserved."""
        content = """INFO: test 1
INFO: test 2
========================================
TOTAL: 10 tests passed
Build succeeded in 5.2s
"""
        compressor = LogCompressor(
            config=LogCompressorConfig(
                min_lines_for_ccr=2,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        assert "========" in result.compressed
        assert "TOTAL:" in result.compressed or "Build succeeded" in result.compressed

    def test_context_lines_added(self):
        """Context lines around errors are included."""
        lines = [f"INFO: item {i}" for i in range(100)]
        lines[50] = "ERROR: critical failure"
        content = "\n".join(lines)

        compressor = LogCompressor(
            config=LogCompressorConfig(
                min_lines_for_ccr=50,
                error_context_lines=2,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        # Should have context around the error
        assert "item 48" in result.compressed or "item 49" in result.compressed
        assert "item 51" in result.compressed or "item 52" in result.compressed


class TestCompressionRatios:
    """Tests for compression ratio calculations."""

    def test_compression_ratio_calculation(self):
        """Compression ratio is calculated correctly."""
        content = "a" * 1000  # 1000 chars
        compressed = "b" * 100  # 100 chars

        # Direct calculation: len(compressed) / len(content)
        expected_ratio = 100 / 1000  # 0.1

        # Result ratio is based on character counts
        result = LogCompressionResult(
            compressed=compressed,
            original=content,
            original_line_count=100,
            compressed_line_count=10,
            format_detected=LogFormat.GENERIC,
            compression_ratio=len(compressed) / len(content),
        )

        assert result.compression_ratio == expected_ratio

    def test_tokens_saved_estimate(self):
        """Token savings estimation works correctly."""
        content = "a" * 400  # ~100 tokens
        compressed = "b" * 40  # ~10 tokens

        result = LogCompressionResult(
            compressed=compressed,
            original=content,
            original_line_count=10,
            compressed_line_count=1,
            format_detected=LogFormat.GENERIC,
            compression_ratio=0.1,
        )

        # (400 - 40) / 4 = 90 tokens saved
        assert result.tokens_saved_estimate == 90

    def test_lines_omitted_property(self):
        """Lines omitted property works correctly."""
        result = LogCompressionResult(
            compressed="test",
            original="test\noriginal",
            original_line_count=100,
            compressed_line_count=10,
            format_detected=LogFormat.GENERIC,
            compression_ratio=0.1,
        )

        assert result.lines_omitted == 90


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_input(self):
        """Empty input is handled gracefully."""
        compressor = LogCompressor()
        result = compressor.compress("")

        assert result.compressed == ""
        assert result.original_line_count == 1  # Empty string splits to one empty line
        assert result.compression_ratio == 1.0

    def test_single_line_input(self):
        """Single line input passes through."""
        compressor = LogCompressor()
        result = compressor.compress("Single line of text")

        assert result.compressed == "Single line of text"
        assert result.compression_ratio == 1.0

    def test_all_errors_no_info(self):
        """Log with only errors is handled."""
        lines = [f"ERROR: failure {i}" for i in range(100)]
        content = "\n".join(lines)

        compressor = LogCompressor(
            config=LogCompressorConfig(
                min_lines_for_ccr=50,
                max_errors=5,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        # Should limit to max_errors
        assert result.compressed_line_count <= compressor.config.max_total_lines

    def test_unicode_content(self):
        """Unicode characters are handled correctly."""
        content = """INFO: Processing 日本語
ERROR: Failed with émoji 🚀
WARN: Über important
"""
        compressor = LogCompressor()
        result = compressor.compress(content)

        # Should not crash and preserve unicode
        assert (
            "日本語" in result.compressed
            or "émoji" in result.compressed
            or "Über" in result.compressed
        )

    def test_very_long_lines(self):
        """Very long lines don't cause issues."""
        long_line = "ERROR: " + "x" * 10000
        lines = [f"INFO: line {i}" for i in range(100)]
        lines[50] = long_line
        content = "\n".join(lines)

        compressor = LogCompressor(
            config=LogCompressorConfig(
                min_lines_for_ccr=50,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        # Should complete without error
        assert len(result.compressed) > 0

    def test_mixed_line_endings(self):
        """Mixed line endings are handled."""
        content = "INFO: line 1\r\nERROR: line 2\rINFO: line 3\n"

        compressor = LogCompressor()
        # Should not crash
        result = compressor.compress(content)
        assert result.compressed is not None

    def test_binary_like_content(self):
        """Content with binary-like patterns doesn't crash."""
        content = "INFO: data\x00\x01\x02ERROR: test"

        compressor = LogCompressor()
        result = compressor.compress(content)
        assert result.compressed is not None


class TestConfigOptions:
    """Tests for configuration options."""

    def test_max_errors_config(self):
        """max_errors configuration limits error selection."""
        lines = [f"ERROR: error {i}" for i in range(50)]
        content = "\n".join(lines)

        compressor = LogCompressor(
            config=LogCompressorConfig(
                min_lines_for_ccr=10,
                max_errors=3,
                max_total_lines=50,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        # Count error lines in output (excluding summary line)
        error_count = sum(1 for line in result.compressed.split("\n") if "ERROR:" in line)
        assert error_count <= 3 + compressor.config.error_context_lines * 2

    def test_max_warnings_config(self):
        """max_warnings configuration limits warning selection."""
        lines = [f"WARN: warning {i}" for i in range(50)]
        content = "\n".join(lines)

        compressor = LogCompressor(
            config=LogCompressorConfig(
                min_lines_for_ccr=10,
                max_warnings=2,
                dedupe_warnings=False,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        # Warnings should be limited
        warn_count = sum(1 for line in result.compressed.split("\n") if "WARN:" in line)
        assert warn_count <= 2 + compressor.config.error_context_lines * 2

    def test_max_total_lines_config(self):
        """max_total_lines configuration limits output."""
        lines = [f"ERROR: error {i}" for i in range(200)]
        content = "\n".join(lines)

        compressor = LogCompressor(
            config=LogCompressorConfig(
                min_lines_for_ccr=50,
                max_total_lines=20,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        # Output lines should be limited (plus summary line)
        output_lines = [line for line in result.compressed.split("\n") if line.strip()]
        assert len(output_lines) <= 21  # max_total_lines + 1 summary

    def test_dedupe_warnings_disabled(self):
        """dedupe_warnings=False preserves duplicate warnings."""
        lines = [
            "WARN: same warning",
            "WARN: same warning",
            "WARN: same warning",
        ]
        content = "\n".join(lines)

        compressor = LogCompressor(
            config=LogCompressorConfig(
                min_lines_for_ccr=1,
                dedupe_warnings=False,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        # All warnings preserved when dedupe disabled
        warn_count = sum(1 for line in result.compressed.split("\n") if "WARN:" in line)
        assert warn_count == 3


class TestLogLineDataclass:
    """Tests for LogLine dataclass behavior."""

    def test_equality_by_line_number(self):
        """LogLine equality is based on line_number."""
        line1 = LogLine(line_number=10, content="foo")
        line2 = LogLine(line_number=10, content="bar")
        line3 = LogLine(line_number=20, content="foo")

        assert line1 == line2
        assert line1 != line3

    def test_hash_by_line_number(self):
        """LogLine hash is based on line_number."""
        line1 = LogLine(line_number=10, content="foo")
        line2 = LogLine(line_number=10, content="bar")

        assert hash(line1) == hash(line2)

        # Can be used in sets
        line_set = {line1, line2}
        assert len(line_set) == 1

    def test_default_values(self):
        """LogLine default values are correct."""
        line = LogLine(line_number=1, content="test")

        assert line.level == LogLevel.UNKNOWN
        assert line.is_stack_trace is False
        assert line.is_summary is False
        assert line.score == 0.0


class TestOutputFormatting:
    """Tests for output formatting and stats."""

    def test_format_output_includes_stats(self):
        """Format output includes category stats."""
        lines = [
            "ERROR: error 1",
            "ERROR: error 2",
            "WARN: warning 1",
            "INFO: info 1",
            "INFO: info 2",
            "INFO: info 3",
        ] * 20  # Make it large enough to trigger compression
        content = "\n".join(lines)

        compressor = LogCompressor(
            config=LogCompressorConfig(
                min_lines_for_ccr=50,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        # Stats should be populated
        assert "errors" in result.stats
        assert "warnings" in result.stats
        assert "info" in result.stats
        assert result.stats["errors"] > 0
        assert result.stats["warnings"] > 0

    def test_format_output_summary_line(self):
        """Formatted output includes summary of omitted lines."""
        lines = [f"INFO: message {i}" for i in range(200)]
        lines.append("ERROR: critical")
        content = "\n".join(lines)

        compressor = LogCompressor(
            config=LogCompressorConfig(
                min_lines_for_ccr=50,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        # Should have omission summary
        assert "lines omitted" in result.compressed
