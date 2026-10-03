"""``get_encoding_for_model`` must not depend on casing, and must know gpt-5.

Two defects, both reachable through the normal ``get_tokenizer()`` path:

1. **gpt-5 had no prefix entry**, so it fell through to ``DEFAULT_ENCODING``
   (``cl100k_base``) instead of ``o200k_base``. On CJK text cl100k emits ~33%
   more tokens than o200k, so every gpt-5 count was inflated.

2. **Resolution was case-sensitive.** ``TokenizerRegistry.get`` lowercases only
   its *cache key*, then builds the counter from the caller's original string
   (``_create_tokenizer(model, ...)``). An uppercase deployment name -- routine
   on Azure, where the deployment name is user-chosen -- reached the resolver
   verbatim, matched nothing, and took the default encoding.

   The cache made (2) genuinely nasty: because the key is lowercased but
   construction is not, the encoding a model ends up with depended on the
   casing of whichever request warmed the cache first, and could differ across
   restarts. The tests below call ``clear_cache()`` so the uppercase spelling is
   resolved cold, which is the failing order.
"""

from __future__ import annotations

import pytest

from headroom.tokenizers import get_tokenizer
from headroom.tokenizers.registry import TokenizerRegistry
from headroom.tokenizers.tiktoken_counter import get_encoding_for_model

CJK = "这是一个测试文档，用于验证分词器的差异。" * 30


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # gpt-5 family: the missing entry.
        ("gpt-5", "o200k_base"),
        ("gpt-5-mini", "o200k_base"),
        ("gpt-5-nano", "o200k_base"),
        ("gpt-5-2025-08-07", "o200k_base"),
        # Casing must not change the answer.
        ("GPT-4o", "o200k_base"),
        ("GPT-4.1", "o200k_base"),
        ("Gpt-4O-Mini", "o200k_base"),
        ("GPT-5", "o200k_base"),
        ("O4-Mini", "o200k_base"),
        ("GPT-4", "cl100k_base"),
        ("GPT-4-Turbo", "cl100k_base"),
        # Must not regress.
        ("gpt-4o", "o200k_base"),
        ("gpt-4.1", "o200k_base"),
        ("gpt-4", "cl100k_base"),
        ("gpt-4-turbo", "cl100k_base"),
        ("gpt-3.5-turbo", "cl100k_base"),
        ("o4-mini", "o200k_base"),
    ],
)
def test_encoding_resolution(model: str, expected: str) -> None:
    assert get_encoding_for_model(model) == expected


@pytest.mark.parametrize("model", ["gpt-5", "GPT-4o", "GPT-4.1"])
def test_cold_cache_uppercase_still_gets_the_right_encoding(model: str) -> None:
    """End-to-end through the registry, with the uppercase spelling resolved first.

    Without clear_cache() a preceding lowercase lookup would populate the shared
    (lowercased) cache key and mask the defect entirely.
    """
    tiktoken = pytest.importorskip("tiktoken")
    o200k = len(tiktoken.get_encoding("o200k_base").encode(CJK))

    TokenizerRegistry.clear_cache()
    assert get_tokenizer(model).count_text(CJK) == o200k


def test_casing_is_not_load_order_dependent() -> None:
    """The same model must count identically whichever spelling arrives first."""
    TokenizerRegistry.clear_cache()
    upper_first = get_tokenizer("GPT-4o").count_text(CJK)

    TokenizerRegistry.clear_cache()
    lower_first = get_tokenizer("gpt-4o").count_text(CJK)

    assert upper_first == lower_first
