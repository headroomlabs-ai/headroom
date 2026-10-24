from __future__ import annotations

import pytest

from headroom.transforms.log_compressor import (
    LogCompressionResult,
    LogCompressor,
    LogCompressorConfig,
    LogFormat,
)


def _malformed_short_summary_diagnostic_log() -> list[str]:
    lines = [f"INFO setup output {i}" for i in range(50)]
    lines.append("=== diagnostic: short test summary info unavailable")
    lines.extend(f"FAILED outside.py::test_{i}" for i in range(10))
    lines.extend(f"ERROR outside.py::test_{i}" for i in range(10, 20))
    return lines


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


def test_result_properties() -> None:
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
