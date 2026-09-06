"""Regression: a bare ``` code fence (no language) must not be rewritten to
```unknown.

Bug: split_into_sections set language = match.group(1) or "unknown", and the
reassembly guard `if section.is_code_fence and section.language:` then emitted
the literal `unknown` tag, so compressed output differed from the input even
when the fenced block was passed through unchanged.

Requires the native `headroom._core` extension (build with `maturin develop`).
"""

import pytest

pytest.importorskip("headroom._core", reason="requires built Rust extension (maturin develop)")

from headroom.transforms.content_router import (  # noqa: E402
    ContentRouter,
    split_into_sections,
)


def test_bare_fence_parses_with_no_language():
    secs = split_into_sections("intro\n```\ncode\n```\noutro")
    fences = [s for s in secs if getattr(s, "is_code_fence", False)]
    assert fences, "no code fence section detected"
    assert fences[0].language is None  # was "unknown" before the fix


def test_language_fence_preserved():
    secs = split_into_sections("```python\nx = 1\n```")
    fences = [s for s in secs if getattr(s, "is_code_fence", False)]
    assert fences and fences[0].language == "python"


def test_compressed_output_has_no_unknown_tag():
    out = ContentRouter().compress("```\nprint(1)\nprint(2)\n```")
    text = str(getattr(out, "compressed", out))
    assert "```unknown" not in text
    assert "```" in text  # fence markers still preserved


if __name__ == "__main__":
    test_bare_fence_parses_with_no_language()
    test_language_fence_preserved()
    test_compressed_output_has_no_unknown_tag()
    print("PASS")
