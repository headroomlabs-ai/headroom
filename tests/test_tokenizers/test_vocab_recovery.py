"""A tiktoken vocabulary that cannot load yet must not pin estimation forever.

Under ``HEADROOM_OFFLINE`` a cold ``TIKTOKEN_CACHE_DIR`` makes the bounded
loader refuse instead of downloading, and online a slow download outlives the
bounded wait. Both recover: an operator pre-seeds the cache, a download lands.
Every place that caches a token counter therefore has to hand out its
character estimate without caching it, so the next lookup reaches the loader
again and gets the exact counter once the vocabulary is there.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from headroom.offline import guard_egress
from headroom.tokenizers import tiktoken_counter as tc
from headroom.tokenizers.estimator import PendingVocabEstimator
from headroom.tokenizers.registry import TokenizerRegistry
from headroom.tokenizers.tiktoken_counter import TiktokenCounter


class _Vocab:
    """tiktoken.get_encoding with a cache that starts cold.

    Before :meth:`seed`, a load is refused exactly as the offline guard refuses
    tiktoken's download; after it, every encoding loads and counts one token
    per whitespace-separated word.
    """

    def __init__(self) -> None:
        self.seeded = False
        self.loads: list[str] = []

    def seed(self) -> None:
        self.seeded = True

    def get_encoding(self, name: str) -> object:
        self.loads.append(name)
        if not self.seeded:
            guard_egress(f"tiktoken {name} vocabulary download", "https://vocab.invalid/x")
        return SimpleNamespace(name=name, encode=lambda text, **_kw: text.split())


@pytest.fixture
def vocab(monkeypatch: pytest.MonkeyPatch) -> _Vocab:
    tiktoken = pytest.importorskip("tiktoken")
    from headroom.providers import openai as openai_provider

    fake = _Vocab()
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")
    monkeypatch.setattr(tiktoken, "get_encoding", fake.get_encoding)
    tc._get_encoding.cache_clear()
    openai_provider._get_encoding.cache_clear()
    TokenizerRegistry.clear_cache()
    yield fake
    tc._get_encoding.cache_clear()
    openai_provider._get_encoding.cache_clear()
    TokenizerRegistry.clear_cache()


@pytest.mark.parametrize("model", ["gpt-4o", "claude-haiku-4-5"])
def test_registry_picks_up_a_vocab_seeded_after_a_refusal(vocab: _Vocab, model: str) -> None:
    cold = TokenizerRegistry.get(model)
    assert isinstance(cold, PendingVocabEstimator)

    vocab.seed()
    warm = TokenizerRegistry.get(model)
    assert isinstance(warm, TiktokenCounter)
    assert warm.count_text("one two three") == 3
    assert TokenizerRegistry.get(model) is warm  # the exact counter is cached


def test_openai_provider_picks_up_a_vocab_seeded_after_a_refusal(vocab: _Vocab) -> None:
    from headroom.providers.openai import OpenAIProvider, OpenAITokenCounter

    provider = OpenAIProvider()
    assert isinstance(provider.get_token_counter("gpt-4o"), PendingVocabEstimator)

    vocab.seed()
    warm = provider.get_token_counter("gpt-4o")
    assert isinstance(warm, OpenAITokenCounter)
    assert warm.count_text("one two three") == 3
    assert provider.get_token_counter("gpt-4o") is warm


def test_anthropic_provider_picks_up_a_vocab_seeded_after_a_refusal(vocab: _Vocab) -> None:
    from headroom.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(warn=False)
    cold = provider.get_token_counter("claude-haiku-4-5")
    assert cold.vocab_pending

    vocab.seed()
    warm = provider.get_token_counter("claude-haiku-4-5")
    assert not warm.vocab_pending
    assert warm.count_text("one two three") == 3
    assert provider.get_token_counter("claude-haiku-4-5") is warm
