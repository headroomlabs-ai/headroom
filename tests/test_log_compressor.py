"""Comprehensive tests for log_compressor.py.

Tests cover:
1. Detection of different log formats (pytest, npm, cargo, make, jest, generic)
2. Line extraction and deduplication
3. Compression ratios
4. Edge cases
"""

from headroom._core import detect_log_format
from headroom.transforms.log_compressor import (
    LogCompressionResult,
    LogCompressor,
    LogCompressorConfig,
    LogFormat,
    LogLevel,
    LogLine,
)


def _detect(lines: list[str]) -> LogFormat:
    """The Rust format detector that compress() uses."""
    return LogFormat(detect_log_format(list(lines)))


_FILLER = [f"step {i} ok" for i in range(100)]


def _compress_amid_filler(lines: list[str], **config) -> LogCompressionResult:
    """Compress ``lines`` between 100 filler lines on each side, so selection
    actually has to choose."""
    content = "\n".join([*_FILLER, *lines, *_FILLER])
    return LogCompressor(LogCompressorConfig(enable_ccr=False, **config)).compress(content)


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
        lines = content.split("\n")
        detected = _detect(lines)
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
        lines = content.split("\n")
        detected = _detect(lines)
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
        lines = content.split("\n")
        detected = _detect(lines)
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
        lines = content.split("\n")
        detected = _detect(lines)
        assert detected == LogFormat.MAKE

    def test_detect_jest_format(self):
        """Jest output is detected correctly."""
        content = """PASS src/components/Button.test.js
FAIL src/utils/helpers.test.ts
  Test Suites: 1 failed, 1 passed, 2 total
  Tests:       2 failed, 10 passed, 12 total
"""
        lines = content.split("\n")
        detected = _detect(lines)
        assert detected == LogFormat.JEST

    def test_detect_generic_format(self):
        """Generic log format is detected for unrecognized output."""
        content = """INFO Starting application
DEBUG Initializing components
WARNING Low memory
ERROR Connection timeout
CRITICAL System failure
"""
        lines = content.split("\n")
        detected = _detect(lines)
        assert detected == LogFormat.GENERIC

    def test_detect_empty_returns_generic(self):
        """Empty or minimal input returns GENERIC."""
        assert _detect([]) == LogFormat.GENERIC
        assert _detect(["random line"]) == LogFormat.GENERIC


class TestLogLevelDetection:
    """Level counts reported by compress(). DEBUG/TRACE/UNKNOWN and the full
    keyword table are unit-tested in Rust (level_classifier_covers_every_level)."""

    def test_level_counts(self):
        lines = [
            "ERROR: something went wrong",
            "error: file not found",
            "Error: Invalid input",
            "FATAL: system crash",
            "fatal error occurred",
            "CRITICAL: database down",
            "FAIL tests/test_foo.py",
            "FAILED to connect",
            "Test failed",
            "WARN: deprecated function",
            "WARNING: low disk space",
            "warning: unused variable",
            "INFO: starting process",
            "info starting",
            "DEBUG: variable x = 5",
            "TRACE: entering function",
        ]
        stats = _compress_amid_filler(lines).stats
        assert stats["errors"] == 6
        assert stats["fails"] == 3
        assert stats["warnings"] == 3
        assert stats["info"] == 2


class TestStackTraceDetection:
    """Stack-trace lines survive compression. Python, Go, Rust-panic, .NET and
    Java traces are unit-tested in Rust; these pin the JS and rustc shapes."""

    def test_javascript_stack_trace_kept(self):
        trace = [
            "Error: Connection failed",
            "    at Connection.connect (src/db.js:42:15)",
            "    at async main (src/index.js:10:5)",
        ]
        # No context lines, so only stack-trace detection can keep the frames.
        compressed = _compress_amid_filler(trace, error_context_lines=0).compressed
        for line in trace:
            assert line in compressed

    def test_rust_error_location_kept(self):
        trace = [
            "error[E0382]: borrow of moved value: `s`",
            " --> src/main.rs:5:13",
        ]
        # No context lines, so only stack-trace detection can keep the frames.
        compressed = _compress_amid_filler(trace, error_context_lines=0).compressed
        for line in trace:
            assert line in compressed


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
