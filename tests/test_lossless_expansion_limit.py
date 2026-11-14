"""The lossless round-trip checks must not expand past the input's own size.

``expand_runs`` and ``unfold_repeated_blocks`` read repeat counts from marker
lines, and tool output may already contain such lines. ``compact_lossless``
only needs to know whether the inverse reproduces the input, so an expansion
longer than the input is a failed round-trip and is abandoned before it is
allocated. Memory is measured with tracemalloc, so this runs on any OS.
"""

from __future__ import annotations

import tracemalloc
from collections.abc import Callable

import pytest

from headroom.exceptions import CompressionError
from headroom.transforms.lossless_compaction import (
    compact_lossless,
    expand_runs,
    unfold_repeated_blocks,
)

MB = 1024 * 1024


def _peak_of(call: Callable[[], str]) -> tuple[str, int]:
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        result = call()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


@pytest.mark.parametrize("kind", ["log", "text", "config"])
def test_run_marker_in_input_is_not_expanded(kind: str) -> None:
    text = "A" * 60 + "\n... (repeated 2000000 times)\ntail\n"
    result, peak = _peak_of(lambda: compact_lossless(text, kind))
    assert result == text
    assert peak < 8 * MB, f"peak {peak / MB:.1f}MB: the marker count was expanded"


def test_chained_block_markers_in_config_are_not_expanded() -> None:
    lines = ["a: 1", "b: 2", "c: 3"]
    size = 3
    for _ in range(20):  # 3 * 2**20 lines if unfolded
        lines.append(f"... (repeats {size} lines from {size} lines back)")
        size *= 2
    text = "\n".join(lines) + "\n"
    result, peak = _peak_of(lambda: compact_lossless(text, "config"))
    assert result == text
    assert peak < 8 * MB, f"peak {peak / MB:.1f}MB: the block markers were unfolded"


def test_real_repeats_still_fold() -> None:
    log = "same line\n" * 50 + "end\n"
    folded = compact_lossless(log, "log")
    assert "(repeated 50 times)" in folded
    assert len(folded) < len(log)

    config = "name: x\nvalue: 1\n" * 30
    assert len(compact_lossless(config, "config")) < len(config)


@pytest.mark.parametrize(
    "text",
    [
        "a\nb",  # no markers
        "a\n... (repeated 2 times)",  # valid run marker
        "a\n... (repeats 5 lines from 1 lines back)",  # invalid block marker, kept as a line
    ],
)
def test_inverses_hold_max_chars_for_every_line(text: str) -> None:
    for inverse in (expand_runs, unfold_repeated_blocks):
        with pytest.raises(CompressionError):
            inverse(text, max_chars=1)
        assert inverse(text, max_chars=100) == inverse(text)


# A long line expanded by a marker, offset by many markers that shrink the
# line count back: the line count stays under the input's, the size does not.
_WIDE_LINE_RUN = (
    "X" * (1024 * 1024) + "\n... (repeated 1000 times)\n" + "a\n... (repeated 1 times)\n" * 1000
)
_WIDE_LINE_BLOCK = "X" * (1024 * 1024) + "\n" + "... (repeats 1 lines from 1 lines back)\n" * 1000


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        (_WIDE_LINE_RUN, "text"),
        (_WIDE_LINE_RUN, "log"),
        (_WIDE_LINE_RUN, "config"),
        (_WIDE_LINE_BLOCK, "config"),
    ],
    ids=["run-text", "run-log", "run-config", "block-config"],
)
def test_wide_line_expansion_is_refused_before_allocating(text: str, kind: str) -> None:
    result, peak = _peak_of(lambda: compact_lossless(text, kind))
    assert result == text
    assert peak < 16 * MB, f"peak {peak / MB:.1f}MB: the wide line was expanded"
