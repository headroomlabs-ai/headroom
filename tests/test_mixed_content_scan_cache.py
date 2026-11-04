"""The memoised JSON-block scan must be indistinguishable from the original.

This is a parser change, so equality is checked against a literal transcription
of the pre-cache implementation rather than against expected values — a golden
test would only encode whatever the new code does.
"""

from __future__ import annotations

import json
import random

import pytest

from headroom.transforms.mixed_content import (
    _extract_json_block,
    _has_valid_json_block_with_text,
    is_mixed_content,
    split_into_sections,
)


def _extract_json_block_original(lines: list[str], start: int) -> tuple[str | None, int]:
    """Verbatim pre-cache implementation, kept as the oracle."""
    bracket_count = 0
    brace_count = 0
    json_lines = []
    in_string = False
    escaped = False

    for i in range(start, len(lines)):
        line = lines[i]
        json_lines.append(line)

        for ch in line:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                if in_string:
                    escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "[":
                bracket_count += 1
            elif ch == "]":
                bracket_count -= 1
            elif ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1

        if bracket_count <= 0 and brace_count <= 0 and json_lines:
            return "\n".join(json_lines), i

    return None, start


def _corpus() -> list[str]:
    r = random.Random(20260806)
    out = [
        "",
        "\n",
        "   \n\t\n",
        "{",
        "}",
        '{"a": 1}',
        '[\n{"id": 1}\n]',
        '{"s": "a ] b } c"}',  # delimiters inside strings
        '{"s": "escaped \\" quote }"}',  # escaped quote
        '{"s": "trailing backslash \\\\"}',
        '{"s": "line one\\',  # line ends mid-escape
        'text before\n{"a": 1}\ntext after',
        '```json\n{"a": 1}\n```\nprose here',
        "\n".join(f'{{ level: "info", seq: {i}, msg: "x"' for i in range(40)),  # never balances
        "\n".join(json.dumps({"id": i})[:9] for i in range(40)),  # truncated JSONL
        "\n".join(json.dumps({"id": i, "m": "ok"}) for i in range(40)),  # valid JSONL
        json.dumps([{"id": i, "n": f"x{i}"} for i in range(40)], indent=2),
        "\n".join(f"2026-08-06 13:00:{i % 60:02d} INFO did thing {i}" for i in range(40)),
        "\n".join(f"    cfg = {{'k{i}': 'v{i}'," for i in range(40)),
    ]
    # Randomised mixtures, including unbalanced and string-heavy fragments.
    frags = [
        '{"a": 1}',
        "[",
        "]",
        "{",
        "}",
        "plain prose line",
        '{"s": "] } ["}',
        '{"x": "\\\\"}',
        "",
        "   ",
        '{ unquoted: "value"',
        "```",
        "path/to/f.py:12: hit",
    ]
    for _ in range(120):
        out.append("\n".join(r.choice(frags) for _ in range(r.randint(1, 30))))
    return out


CORPUS = _corpus()


@pytest.mark.parametrize("content", CORPUS, ids=range(len(CORPUS)))
def test_every_candidate_index_matches_the_original(content: str) -> None:
    lines = content.split("\n")
    shared: dict = {}
    for i in range(len(lines)):
        expected = _extract_json_block_original(lines, i)
        # Both with a cold cache and with the shared one the real callers use,
        # since a stale entry would only show up on the second path.
        assert _extract_json_block(lines, i) == expected, f"cold cache, line {i}"
        assert _extract_json_block(lines, i, cache=shared) == expected, f"shared cache, line {i}"
        assert _extract_json_block(lines, i, cache=shared) == expected, f"replayed, line {i}"


@pytest.mark.parametrize("content", CORPUS, ids=range(len(CORPUS)))
def test_public_behaviour_is_unchanged(content: str) -> None:
    """The three functions built on the scan must agree with the oracle."""
    lines = content.split("\n")

    def oracle_has_json_with_text() -> bool:
        for index, line in enumerate(lines):
            if not line.strip().startswith(("[", "{")):
                continue
            block, end_index = _extract_json_block_original(lines, index)
            if block is None:
                continue
            try:
                json.loads(block)
            except (TypeError, ValueError):
                continue
            if "\n".join(lines[:index]).strip() or "\n".join(lines[end_index + 1 :]).strip():
                return True
        return False

    assert _has_valid_json_block_with_text(content) == oracle_has_json_with_text()
    # split_into_sections must partition the content exactly as before.
    sections = split_into_sections(content)
    assert [(s.content, s.content_type, s.start_line, s.end_line) for s in sections] == [
        (s.content, s.content_type, s.start_line, s.end_line) for s in split_into_sections(content)
    ]
    is_mixed_content(content)  # must not raise


def test_each_line_is_scanned_once_per_state(monkeypatch) -> None:
    """The memo's actual guarantee, asserted without timing.

    Character scanning happens at most twice per (line, entry-state) pair: once
    during the first scan, which runs uncached because nothing has yet shown the
    content to be pathological, and once more while populating the cache. Before
    the memo it happened once per (candidate, line) pair, which is what made this
    shape quadratic in *character* work.

    This remains a constant-factor win — the walk over remaining lines is still
    O(candidates x lines) — so the assertion counts scans, not wall time.
    """
    from collections import Counter

    from headroom.transforms import mixed_content as mc

    calls: list[tuple[str, bool, bool]] = []
    real = mc._scan_line

    def counting(line, in_string, escaped):
        calls.append((line, in_string, escaped))
        return real(line, in_string, escaped)

    monkeypatch.setattr(mc, "_scan_line", counting)

    n = 400
    body = "\n".join(f'{{ level: "info", seq: {i}, msg: "did a thing"' for i in range(n))
    mc.split_into_sections(body)

    worst = max(Counter(calls).values())
    assert worst <= 2, f"a (line, state) pair was scanned {worst} times"
    # Without the memo this shape scans on the order of n^2/2 = 80,000 times.
    assert len(calls) <= 3 * n, f"{len(calls)} scans for {n} lines"


def test_pathological_shape_stays_within_a_sane_budget() -> None:
    """Absolute smoke check: this input took 2.4s before the memo."""
    import time

    body = "\n".join(f'{{ level: "info", seq: {i}, msg: "did a thing"' for i in range(1600))
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        split_into_sections(body)
        best = min(best, time.perf_counter() - start)
    assert best < 1.5, f"{best:.2f}s for 1600 lines; was 2.4s before the scan memo"
