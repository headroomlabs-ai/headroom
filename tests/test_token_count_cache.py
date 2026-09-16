"""The token-count memo must be invisible: same integers, or it is a bug.

These counts feed context_pressure -> min_ratio -> which blocks get compressed,
so "the cache returned a different number" is a compression regression, not a
cache miss. Every test here is an equality test for that reason.
"""

from __future__ import annotations

import json

import pytest

from headroom.providers.anthropic import AnthropicProvider
from headroom.tokenizers.base import TokenCountCache
from headroom.tokenizers.estimator import EstimatingTokenCounter
from headroom.tokenizers.tiktoken_counter import TiktokenCounter

BODIES = [
    "word " * 500,
    json.dumps([{"id": i, "name": f"item-{i}", "ok": i % 2 == 0} for i in range(300)]),
    "def f(x):\n    return x + 1\n" * 200,
    "2026-08-06 13:00:00 INFO worker did a thing\n" * 400,
    "日本語のテキストをここに置きます。" * 200,
    "<|endoftext|> literal special token marker " * 100,  # forces the ValueError path
    "x" * 300,
]


def _counters():
    return [
        ("anthropic", AnthropicProvider().get_token_counter("claude-sonnet-5")),
        ("tiktoken", TiktokenCounter(model="gpt-4o")),
        ("estimator-auto", EstimatingTokenCounter()),
        ("estimator-fixed", EstimatingTokenCounter(chars_per_token=3.5)),
    ]


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("body", BODIES)
def test_cached_count_equals_uncached(body: str) -> None:
    for name, counter in _counters():
        counter._count_cache.clear()
        first = counter.count_text(body)  # miss, populates
        second = counter.count_text(body)  # hit
        counter._count_cache.clear()
        third = counter.count_text(body)  # miss again
        assert first == second == third, f"{name}: {first} != {second} != {third}"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_empty_and_tiny_text_still_correct() -> None:
    for _name, counter in _counters():
        assert counter.count_text("") == 0
        assert counter.count_text("hi") == counter.count_text("hi")


def test_cache_clears_when_full_rather_than_growing() -> None:
    cache = TokenCountCache(min_chars=1, max_entries=4, max_chars=10**9)
    for i in range(10):
        cache.put(f"text-number-{i}", i)
    assert len(cache._counts) <= 4


def test_cache_respects_the_character_budget() -> None:
    cache = TokenCountCache(min_chars=1, max_entries=10**6, max_chars=1000)
    for i in range(50):
        cache.put("x" * 100 + str(i), i)
    assert cache._chars <= 1000 + 200  # one entry may straddle the cap


def test_small_strings_are_not_cached() -> None:
    """They encode in microseconds; caching them would evict the entries that matter."""
    cache = TokenCountCache(min_chars=256)
    cache.put("short", 1)
    assert cache.get("short") is None


def test_distinct_texts_do_not_collide() -> None:
    cache = TokenCountCache(min_chars=1)
    cache.put("alpha", 1)
    cache.put("beta", 2)
    assert (cache.get("alpha"), cache.get("beta"), cache.get("gamma")) == (1, 2, None)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_counters_do_not_share_a_cache_across_encodings() -> None:
    """cl100k and o200k are both live in one process; a shared memo would mix them."""
    a = TiktokenCounter(encoding="cl100k_base")
    b = TiktokenCounter(encoding="o200k_base")
    body = "tokenization differs between these two encodings. " * 100
    assert a.count_text(body) == a.count_text(body)
    assert b.count_text(body) == b.count_text(body)
    assert a._count_cache is not b._count_cache


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_concurrent_counting_is_consistent() -> None:
    """The pipeline runs on a thread pool and shares one counter."""
    from concurrent.futures import ThreadPoolExecutor

    counter = AnthropicProvider().get_token_counter("claude-sonnet-5")
    bodies = [f"{b}\n{i}" for i, b in enumerate(BODIES * 3)]
    expected = {b: counter.count_text(b) for b in bodies}
    counter._count_cache.clear()
    with ThreadPoolExecutor(max_workers=8) as pool:
        got = list(pool.map(counter.count_text, bodies))
    assert got == [expected[b] for b in bodies]
