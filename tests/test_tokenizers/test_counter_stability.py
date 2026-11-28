"""One counter per model for the life of the process, even when the vocab arrives.

A request counts tokens through several lookups of the same model's counter:
the SDK counts ``tokens_before`` with its own lookup and the transform pipeline
counts ``tokens_after`` with another. If a vocabulary that could not load
(offline with a cold cache) became available between the two, a lookup that
handed out a fresh counter would compare a character estimate with BPE tokens:
308 before and 8 after for identical messages in review. The registry and the
providers therefore keep the counter they resolved first.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from headroom.offline import guard_egress
from headroom.tokenizers import tiktoken_counter as tc
from headroom.tokenizers.registry import TokenizerRegistry


class _Vocab:
    """tiktoken.get_encoding whose cache starts cold and can be seeded."""

    def __init__(self) -> None:
        self.seeded = False

    def seed(self) -> None:
        self.seeded = True

    def get_encoding(self, name: str) -> object:
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
    monkeypatch.setattr(tc, "_slots", {})
    tc._get_encoding.cache_clear()
    openai_provider._get_encoding.cache_clear()
    TokenizerRegistry.clear_cache()
    yield fake
    tc._get_encoding.cache_clear()
    openai_provider._get_encoding.cache_clear()
    TokenizerRegistry.clear_cache()


MESSAGES = [{"role": "user", "content": "alpha beta gamma delta " * 20}]


@pytest.mark.parametrize("model", ["gpt-4o", "claude-haiku-4-5"])
def test_registry_counter_survives_a_vocab_arriving(vocab: _Vocab, model: str) -> None:
    first = TokenizerRegistry.get(model)
    before = first.count_messages(MESSAGES)
    vocab.seed()
    assert TokenizerRegistry.get(model) is first
    assert TokenizerRegistry.get(model).count_messages(MESSAGES) == before


def test_openai_provider_counter_survives_a_vocab_arriving(vocab: _Vocab) -> None:
    from headroom.providers.openai import OpenAIProvider

    provider = OpenAIProvider()
    first = provider.get_token_counter("gpt-4o")
    before = first.count_messages(MESSAGES)
    vocab.seed()
    assert provider.get_token_counter("gpt-4o") is first
    assert provider.get_token_counter("gpt-4o").count_messages(MESSAGES) == before


def test_anthropic_provider_counter_survives_a_vocab_arriving(vocab: _Vocab) -> None:
    from headroom.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(warn=False)
    first = provider.get_token_counter("claude-haiku-4-5")
    before = first.count_messages(MESSAGES)
    vocab.seed()
    assert provider.get_token_counter("claude-haiku-4-5") is first
    assert provider.get_token_counter("claude-haiku-4-5").count_messages(MESSAGES) == before


def test_sdk_request_totals_use_one_unit_when_the_vocab_arrives(
    vocab: _Vocab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reviewer's SDK repro: the pipeline passes the messages through
    (open circuit breaker), and the vocab lands after the SDK counted
    tokens_before but before the pipeline counts. Identical messages must
    report identical totals."""
    from headroom.providers.openai import OpenAIProvider
    from headroom.transforms.pipeline import TransformPipeline

    provider = OpenAIProvider()
    pipeline = TransformPipeline(provider=provider)
    monkeypatch.setattr(pipeline, "_breaker_is_open", lambda: True)

    sdk_counter = provider.get_token_counter("gpt-4o")
    tokens_before = sdk_counter.count_messages(MESSAGES)
    vocab.seed()
    result = pipeline.apply(MESSAGES, "gpt-4o", model_limit=128_000)

    assert result.transforms_applied == ["pipeline:circuit_open"]
    assert result.tokens_after == tokens_before
