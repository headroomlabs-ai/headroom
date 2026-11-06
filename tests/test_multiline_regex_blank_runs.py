"""Line-anchored detection regexes must stay linear on runs of blank lines.

Under ``re.MULTILINE``, a pattern that opens with ``^\\s*`` lets every line
start scan forward through all the blank lines after it, so a run of n blank
(or whitespace-only) lines costs O(n^2). ``re`` holds the GIL while it runs,
so one such tool result stalled every concurrent proxy request: 10,000
whitespace-only lines took 3.3s in ``is_mixed_content`` and 2,000 blank lines
took 5.3s in the ``detect_language`` prefilter.

The patterns now use ``[^\\S\\n]*`` (whitespace except newline). Equivalence is
checked against the pre-fix ``\\s*`` patterns rather than against expected
values, so the tests pin "same matches, linear time" and nothing more.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable

import pytest

from headroom.transforms.code_compressor import _LANGUAGE_PREFILTER, detect_language
from headroom.transforms.mixed_content import _JSON_BLOCK_START, is_mixed_content

_LINE_ANCHORED_PATTERNS = [_JSON_BLOCK_START] + [
    pattern
    for patterns in _LANGUAGE_PREFILTER.values()
    for pattern in patterns
    if pattern.flags & re.MULTILINE and pattern.pattern.startswith("^")
]


_PHP_MODIFIERS = "(public|private|protected|static|abstract|final)"
_PHP_FUNCTION_PREFIX = rf"^[^\S\n]*(?:{_PHP_MODIFIERS}[^\S\n]*)?function"


def _original(pattern: re.Pattern[str]) -> re.Pattern[str]:
    """The pre-fix pattern: every ``[^\\S\\n]*`` was ``\\s*``.

    The PHP ``function`` pattern was also regrouped (the modifier now carries
    its own trailing gap), so its pre-fix form is spelled out.
    """
    if pattern.pattern.startswith(_PHP_FUNCTION_PREFIX):
        return re.compile(rf"^\s*{_PHP_MODIFIERS}?\s*function\s+\w+\s*\(", pattern.flags)
    return re.compile(pattern.pattern.replace(r"[^\S\n]*", r"\s*"), pattern.flags)


def _fastest_run(fn: Callable[[], object], budget: float) -> float:
    """Best of up to three runs, stopping once a run is under budget.

    Retries only absorb scheduler noise near the budget, so a run that is
    already 5x over it ends the loop instead of repeating a slow regex.
    """
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
        if best < budget or best > 5 * budget:
            break
    return best


def test_is_mixed_content_whitespace_run_stays_within_budget() -> None:
    """10,000 whitespace-only lines took 3.3s before the fix."""
    content = "x\n" + "   \n" * 10_000 + "y"
    elapsed = _fastest_run(lambda: is_mixed_content(content), budget=0.5)
    assert elapsed < 0.5, f"{elapsed:.2f}s for 10,000 whitespace-only lines; was 3.3s"


def test_detect_language_blank_run_stays_within_budget() -> None:
    """2,000 blank lines inside the 5,000-char prefilter sample took 5.3s."""
    code = "def f():\n    return 1\n" + "\n" * 2_000 + "x = 1\n"
    elapsed = _fastest_run(lambda: detect_language(code), budget=0.5)
    assert elapsed < 0.5, f"{elapsed:.2f}s for 2,000 blank lines; was 5.3s"


@pytest.mark.parametrize("pattern", _LINE_ANCHORED_PATTERNS, ids=lambda p: p.pattern)
def test_each_line_anchored_pattern_is_linear_on_blank_runs(pattern: re.Pattern[str]) -> None:
    """Covers any ``\\s*`` reachable from a line start, not only a leading one.

    The PHP ``(modifiers)?\\s*function`` pattern stayed slow until its middle
    ``\\s*`` was narrowed as well, since an absent modifier leaves it directly
    after the anchor. With two ``\\s*`` in a row it was cubic, not quadratic.
    """
    blank = "\n" * 5_000
    elapsed = _fastest_run(lambda: pattern.findall(blank), budget=0.01)
    assert elapsed < 0.01, f"{elapsed * 1000:.1f}ms on a 5,000-line blank run"


@pytest.mark.parametrize("pattern", _LINE_ANCHORED_PATTERNS, ids=lambda p: p.pattern)
def test_each_line_anchored_pattern_is_linear_on_one_long_indent(
    pattern: re.Pattern[str],
) -> None:
    """Two adjacent gaps can split one run of spaces every possible way.

    Narrowing both PHP gaps to ``[^\\S\\n]*`` still left them adjacent when the
    modifier is absent, so a single 5,000-space line cost 90-190ms (20-100ms
    with the original ``\\s*``). The modifier now carries its own trailing gap.
    """
    indent = " " * 5_000 + "x"
    elapsed = _fastest_run(lambda: pattern.findall(indent), budget=0.01)
    assert elapsed < 0.01, f"{elapsed * 1000:.1f}ms on a 5,000-space line"


_FUZZ_TOKENS = [
    "\n",
    "\n\n",
    "   ",
    "\t",
    "\r\n",
    "\f",
    "x",
    "{",
    "[",
    '"""',
    "@dec",
    "def f(x):",
    "class A:",
    "import os",
    "from a import b",
    "if __name__ ==",
    "export const x = 1",
    "async function",
    "module.exports",
    "interface I",
    "fn main()",
    "pub struct S",
    "#[derive]",
    "func (r T) M()",
    "package a.b;",
    "public class C",
    "#include <x>",
    "int main(",
    "typedef ",
    "sub f",
    "my $x",
    "using System;",
    "namespace A.B",
    "use A\\B;",
    "public function f(",
    "function g(",
    "publicfunction h(",
    "static ",
    "final",
    "abstract\n",
    "#!/bin/bash",
    "if",
    "done",
    "export X=1",
]


def test_match_counts_are_unchanged() -> None:
    """``detect_language`` scores by match count; it must not move."""
    assert any(p.pattern.startswith(_PHP_FUNCTION_PREFIX) for p in _LINE_ANCHORED_PATTERNS)
    rng = random.Random(0)
    for _ in range(3_000):
        text = "".join(rng.choice(_FUZZ_TOKENS) for _ in range(rng.randint(1, 14)))
        for pattern in _LINE_ANCHORED_PATTERNS:
            assert len(pattern.findall(text)) == len(_original(pattern).findall(text)), (
                pattern.pattern,
                text,
            )
