"""A shorter model family must not shadow a longer one.

``_MODEL_ENCODINGS`` and ``_CONTEXT_LIMITS`` are matched by prefix. Iterating
them in plain dict order meant the first *inserted* prefix won, not the most
specific one, so ``gpt-4.1`` matched the ``gpt-4`` entry:

* context limit 8192 instead of ~1M -- a 128x under-estimate, which makes the
  proxy think a 1M-context model is nearly full and compress accordingly;
* encoding ``cl100k_base`` instead of ``o200k_base``, which over-counts CJK
  text by ~33%.

``gpt-4-32k-0613`` had the same problem (8192 instead of 32768).

``get_context_limit`` consults LiteLLM before this table, so the limit half only
surfaces where LiteLLM is missing or does not know the model -- notably any
install on Python >= 3.14, where the ``litellm`` dependency is excluded by its
``python_version < '3.14'`` marker. The encoding half has no such fallback and
was always wrong.
"""

from __future__ import annotations

import pytest

from headroom.providers.openai import (
    OpenAIProvider,
    _get_encoding_name_for_model,
)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # The shadowing cases.
        ("gpt-4.1", 1_047_576),
        ("gpt-4.1-mini", 1_047_576),
        ("gpt-4.1-nano", 1_047_576),
        ("gpt-4.1-2025-04-14", 1_047_576),
        ("gpt-4-32k-0613", 32768),
        # Newer families that fell through to the unknown-model default.
        ("gpt-5", 272_000),
        ("gpt-5-mini", 272_000),
        ("o4-mini", 200_000),
        # Must not regress.
        ("gpt-4", 8192),
        ("gpt-4-turbo", 128_000),
        ("gpt-4o", 128_000),
        ("o3", 200_000),
        ("gpt-3.5-turbo", 16385),
    ],
)
def test_context_limit_prefers_the_most_specific_prefix(model: str, expected: int) -> None:
    assert OpenAIProvider()._get_context_limit_manual(model) == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-4.1", "o200k_base"),
        ("gpt-4.1-mini", "o200k_base"),
        ("gpt-4.1-2025-04-14", "o200k_base"),
        ("gpt-5", "o200k_base"),
        ("gpt-5-mini", "o200k_base"),
        ("o4-mini", "o200k_base"),
        # Must not regress: these genuinely are cl100k_base.
        ("gpt-4", "cl100k_base"),
        ("gpt-4-turbo", "cl100k_base"),
        ("gpt-3.5-turbo", "cl100k_base"),
        ("gpt-4o", "o200k_base"),
    ],
)
def test_encoding_prefers_the_most_specific_prefix(model: str, expected: str) -> None:
    assert _get_encoding_name_for_model(model) == expected


def test_cjk_is_not_over_counted_for_gpt_41() -> None:
    """The concrete cost of picking cl100k_base for a gpt-4.1 request."""
    tiktoken = pytest.importorskip("tiktoken")
    text = "这是一个测试文档，用于验证分词器的差异。" * 30

    chosen = _get_encoding_name_for_model("gpt-4.1")
    assert len(tiktoken.get_encoding(chosen).encode(text)) == len(
        tiktoken.get_encoding("o200k_base").encode(text)
    )
