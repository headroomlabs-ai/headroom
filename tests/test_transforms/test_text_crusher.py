"""Phase 2 (#1171): TextCrusher fast extractive compressor.

Validates the core contract: extractive (no invented words), deterministic,
actually compresses, suppresses near-duplicates, and preferentially keeps
query-relevant segments. End-to-end answer-quality vs kompress is validated
separately via headroom/evals before defaulting it on.
"""

from __future__ import annotations

from headroom.transforms.text_crusher import TextCrusher, TextCrusherConfig


def _doc(n: int = 40) -> str:
    return " ".join(
        f"Sentence number {i} describes a distinct topic {i} in some detail." for i in range(n)
    )


def test_extractive_invents_no_new_words():
    content = _doc()
    r = TextCrusher().compress(content, target_ratio=0.5)
    orig_words = set(content.split())
    assert set(r.compressed.split()) <= orig_words


def test_deterministic():
    content = _doc()
    a = TextCrusher().compress(content, target_ratio=0.4).compressed
    b = TextCrusher().compress(content, target_ratio=0.4).compressed
    assert a == b


def test_actually_compresses_large_text():
    r = TextCrusher().compress(_doc(60), target_ratio=0.3)
    assert r.compressed_tokens < r.original_tokens
    assert r.compression_ratio < 0.6


def test_passthrough_when_too_few_segments():
    content = "one thing. two thing. three thing."  # < min_segments_for_crush (6)
    r = TextCrusher().compress(content)
    assert r.compressed == content
    assert r.compression_ratio == 1.0


def test_near_duplicates_suppressed():
    dup = "The quick brown fox jumps over the very lazy dog today."
    uniques = [f"A unique fact about item {i} stated plainly here." for i in range(8)]
    content = "\n".join([dup] * 10 + uniques)
    r = TextCrusher(TextCrusherConfig(near_dup_threshold=0.8)).compress(content, target_ratio=0.9)
    # The duplicated sentence must not be kept 10 times.
    assert r.compressed.count("quick brown fox") <= 2


def test_relevance_keeps_query_relevant_segment():
    filler = [f"Filler line number {i} with generic words and padding here." for i in range(30)]
    needle = "The authentication token expires after thirty minutes of inactivity."
    content = "\n".join(filler[:15] + [needle] + filler[15:])
    r = TextCrusher().compress(
        content,
        context="how long until the authentication token expires",
        target_ratio=0.2,
    )
    assert "authentication token expires" in r.compressed
