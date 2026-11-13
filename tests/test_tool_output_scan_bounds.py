"""Scans over tool output must stay linear on long unclosed shapes.

The timing bounds are loose (the old code took minutes at these sizes); the
oracle comparison checks the banner scan still finds exactly what it did.
"""

from __future__ import annotations

import random
import re
import time

import pytest

from headroom.proxy.interceptors.astgrep import (
    _detect_truncation,
    _is_plausible_truncation_range,
)
from headroom.transforms.mixed_content import (
    ContentType,
    _extract_json_block,
    _has_valid_json_block_with_text,
    _JsonBlockScanner,
    split_into_sections,
)

_PREVIOUS_BANNER = re.compile(
    r"\[\s*truncated\s*:\s*partial\s+view\b"
    r"[^\[\]]*?"
    r"showing\s+lines?\s+(?P<start_line>\d+)\s*[-–]\s*(?P<end_line>\d+)"
    r"\s+of\s+(?P<total_lines>\d+)\s+total"
    r"[^\[\]]*\]",
    re.IGNORECASE,
)


def _previous_detect(source: str) -> tuple[int, int] | None:
    count = len(source.splitlines())
    for m in _PREVIOUS_BANNER.finditer(source):
        s, e, t = (int(m.group(g)) for g in ("start_line", "end_line", "total_lines"))
        if _is_plausible_truncation_range(s, e, t, count):
            return e, t
    return None


def test_truncation_banner_still_recognised() -> None:
    src = (
        "x\n" * 50 + "[Truncated: PARTIAL view -- a.py: showing lines 1-40 of 90 total (x). Read.]"
    )
    assert _detect_truncation(src) == (40, 90)


def test_truncation_banner_matches_previous_regex() -> None:
    rng = random.Random(1234)
    parts = [
        "[truncated: partial view", "[ Truncated :  PARTIAL   view", "]", "[", "\n",
        " showing lines 1-10 of 50 total ", "showing line 3–4 of 9 total", "showing lines 5-5 of 5 total",
        "x", " ", "path.py:", "showing lines", "total", "12", "(more)",
    ]  # fmt: skip
    for _ in range(3000):
        src = "".join(rng.choice(parts) for _ in range(rng.randint(1, 25)))
        assert _detect_truncation(src) == _previous_detect(src), repr(src)


def test_truncation_scan_is_linear_on_unclosed_banner() -> None:
    src = "[truncated: partial view " + "showing lines 1-10 of 50 total " * 12000
    start = time.perf_counter()
    assert _detect_truncation(src) is None
    assert time.perf_counter() - start < 5.0


def test_json_block_scans_are_bounded_on_unclosed_braces() -> None:
    content = "\n".join(["{"] * 20000)
    start = time.perf_counter()
    assert _has_valid_json_block_with_text(content) is False
    sections = split_into_sections(content)
    assert time.perf_counter() - start < 10.0
    assert "\n".join(s.content for s in sections).count("{") == 20000


def test_long_json_block_inside_text_still_found() -> None:
    body = ",\n".join(f'  {{"k": {i}}}' for i in range(5000))
    content = "log line\n[\n" + body + "\n]\ntrailing"
    assert _has_valid_json_block_with_text(content) is True
    types = [s.content_type for s in split_into_sections(content)]
    assert ContentType.JSON_ARRAY in types


@pytest.mark.parametrize("unclosed", [150, 20000])
def test_json_object_after_many_unclosed_braces_keeps_its_own_section(unclosed: int) -> None:
    obj = '{"critical": "disk full"}'
    content = "\n".join(["{"] * unclosed + ["some prose"] * 6 + [obj, "more prose"])
    assert _has_valid_json_block_with_text(content) is True
    json_sections = [
        s for s in split_into_sections(content) if s.content_type is ContentType.JSON_ARRAY
    ]
    assert [s.content for s in json_sections] == [obj]


@pytest.mark.parametrize("repeats", [1, 130])
def test_json_object_after_crossed_count_prefix_keeps_its_own_section(repeats: int) -> None:
    # Each count reaches zero on its own but never on the same line, so no
    # early rejection applies; the object after it must still be found.
    prefix = (["{"] * 150 + ["[" + "}" * 150]) * repeats
    obj = '{"critical": "disk full"}'
    content = "\n".join(prefix + ["some prose"] * 6 + [obj, "more prose"])
    start = time.perf_counter()
    assert _has_valid_json_block_with_text(content) is True
    json_sections = [
        s for s in split_into_sections(content) if s.content_type is ContentType.JSON_ARRAY
    ]
    assert time.perf_counter() - start < 10.0
    assert [s.content for s in json_sections] == [obj]


def test_block_scanner_matches_walking_each_candidate() -> None:
    rng = random.Random(7)
    parts = ["{", "}", "[", "]", '"', "\\\\", "a", " ", ",", "{}", "[]", '"x"', "}}", "]]", "{["]
    for _ in range(3000):
        lines = [
            "".join(rng.choice(parts) for _ in range(rng.randint(0, 5)))
            for _ in range(rng.randint(1, 25))
        ]
        scanner = _JsonBlockScanner(lines)
        scanner._direct_steps_left = 0  # answer every candidate from the sweep
        for i, line in enumerate(lines):
            if line.strip().startswith(("[", "{")):
                assert scanner.extract(i) == _extract_json_block(lines, i), (lines, i)
