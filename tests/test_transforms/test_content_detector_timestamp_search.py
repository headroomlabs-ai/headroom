"""Regression for #3736: ISO-8601 timestamped logs must not route as search.

The colon branch of ``_is_search_result_line`` accepts ``^[^\\s:]+:\\d+:``
whenever the prefix passes ``_prefix_looks_like_path``. A bare ISO date-hour
prefix (``2026-09-23T10``) passes that guard, so every row of a timestamped
log counts as a grep match (ratio 1.0). Since #3419 the lossless fold skips
timestamp rows, so the payload falls through to the lossy SearchCompressor,
which keeps 5 of 2,000 lines and prints the minute back as an integer
(``10:0:00Z`` — a time that does not exist in the input).
"""

from __future__ import annotations

from headroom.transforms.content_detector import (
    ContentType,
    _is_search_result_line,
    _try_detect_search,
    detect_content_type,
)


def _timestamped_log(n: int = 200) -> str:
    return "\n".join(
        f"2026-09-23T10:{i // 60:02d}:{i % 60:02d}Z INFO worker[{i % 8}] "
        f"processed batch {i} in {(i * 13) % 900}ms"
        for i in range(n)
    )


def test_timestamp_row_is_not_a_search_result_line() -> None:
    line = "2026-09-23T10:00:00Z INFO worker[0] processed batch 0 in 0ms"
    assert not _is_search_result_line(line)


def test_timestamped_log_does_not_route_to_search() -> None:
    payload = _timestamped_log()
    assert _try_detect_search(payload) is None
    assert detect_content_type(payload).content_type is not ContentType.SEARCH_RESULTS


def test_space_separated_timestamp_log_does_not_route_to_search() -> None:
    payload = "\n".join(
        f"2026-09-23 10:{i // 60:02d}:{i % 60:02d} [INFO] worker {i} processed batch {i}"
        for i in range(50)
    )
    assert _try_detect_search(payload) is None
    assert detect_content_type(payload).content_type is not ContentType.SEARCH_RESULTS


def test_genuine_grep_output_still_routes_to_search() -> None:
    payload = "\n".join(
        [
            "src/app.py:10:def main():",
            "src/app.py:20:print('hello')",
            "README.md:5:usage docs",
        ]
    )
    result = _try_detect_search(payload)
    assert result is not None
    assert result.content_type is ContentType.SEARCH_RESULTS


def test_grep_on_timestamped_file_names_still_routes_to_search() -> None:
    """A path prefix that embeds a date is still a path, not a timestamp row."""
    payload = "\n".join(
        [
            "logs/2026-09-23.log:42:ERROR something failed",
            "logs/2026-09-23.log:43:INFO recovered",
            "app/2026-09-24.log:7:WARN retry",
        ]
    )
    assert _is_search_result_line("logs/2026-09-23.log:42:ERROR something failed")
    result = _try_detect_search(payload)
    assert result is not None
    assert result.content_type is ContentType.SEARCH_RESULTS


def test_extensionless_paths_still_match() -> None:
    assert _is_search_result_line("Makefile:12:all:")
    assert _is_search_result_line("Dockerfile:3:FROM scratch")


def test_ansi_tracing_subscriber_log_routes_as_log_not_search() -> None:
    """ANSI-colored tracing-subscriber logs must detect as BUILD_OUTPUT, not SEARCH_RESULTS."""
    mixed = "\n".join(
        f"\x1b[2m2026-11-17T22:44:{i % 60:02d}.520030Z\x1b[0m "
        + (
            "\x1b[31mERROR\x1b[0m \x1b[2mtlog\x1b[0m\x1b[2m:\x1b[0m upstream timeout"
            if i % 10 == 0
            else "\x1b[32m INFO\x1b[0m \x1b[2mtlog\x1b[0m\x1b[2m:\x1b[0m request handled"
        )
        for i in range(60)
    )
    error_only = "\n".join(
        f"\x1b[2m2026-11-17T22:45:{i % 60:02d}.806331Z\x1b[0m "
        f"\x1b[31mERROR\x1b[0m \x1b[2mtlog\x1b[0m\x1b[2m:\x1b[0m upstream timeout req_id={i}"
        for i in range(40)
    )

    for payload in (mixed, error_only):
        assert _try_detect_search(payload) is None
        result = detect_content_type(payload)
        assert result.content_type is ContentType.BUILD_OUTPUT
