"""Every count that is compared within one request comes from ONE counter (#5037).

``tokens_saved`` is ``tokens_before - tokens_after``. When the two operands are
counted by two independently resolved counters, the subtraction is a
difference of two rulers: it reports savings (or inflation) on a request that
nothing touched.

On ``main`` every provider caches one counter per model, so two lookups usually
return the same object. Nothing in the ``Provider`` contract promises that,
though: a provider that hands out a fallback counter while a vocabulary cannot
load, and the real one once it can (#4990), returns two different rulers to two
lookups in one request. These tests use exactly such a provider, so they pin
the contract (resolve once, pass it through) rather than an accident of
caching.
"""

from __future__ import annotations

import time
from typing import Any

from headroom.providers.openai import OpenAIProvider
from headroom.tokenizer import Tokenizer
from headroom.tokenizers import EstimatingTokenCounter
from headroom.transforms.pipeline import TransformPipeline

MODEL = "gpt-4o"
MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize the release notes for version 2.4. " * 12},
]


class _FlippingProvider(OpenAIProvider):
    """Hands out a different counter on every lookup, like an uncached fallback.

    Odd lookups count ~4 chars per token, even lookups ~2 chars per token, so
    any pairing of two lookups shows up as a large, deterministic gap.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lookups = 0

    def get_token_counter(self, model: str) -> Any:
        self.lookups += 1
        return EstimatingTokenCounter(chars_per_token=4.0 if self.lookups % 2 else 2.0)


def _open_breaker(pipeline: TransformPipeline) -> None:
    pipeline._breaker_threshold = 1
    pipeline._breaker_open_until = time.monotonic() + 600


# ---------------------------------------------------------------------------
# TransformPipeline.apply(token_counter=...)
# ---------------------------------------------------------------------------


def test_pipeline_counts_with_the_counter_the_caller_passes() -> None:
    pipeline = TransformPipeline(transforms=[], provider=_FlippingProvider())
    mine = EstimatingTokenCounter(chars_per_token=3.0)

    result = pipeline.apply(MESSAGES, MODEL, token_counter=mine, model_limit=128_000)

    assert result.tokens_before == mine.count_messages(MESSAGES)
    assert result.tokens_after == mine.count_messages(result.messages)


def test_pipeline_accepts_a_tokenizer_wrapper_too() -> None:
    pipeline = TransformPipeline(transforms=[], provider=_FlippingProvider())
    mine = Tokenizer(EstimatingTokenCounter(chars_per_token=3.0), MODEL)

    result = pipeline.apply(MESSAGES, MODEL, token_counter=mine, model_limit=128_000)

    assert result.tokens_before == mine.count_messages(MESSAGES)


def test_open_breaker_passthrough_uses_the_callers_counter() -> None:
    pipeline = TransformPipeline(transforms=[], provider=_FlippingProvider())
    _open_breaker(pipeline)
    mine = EstimatingTokenCounter(chars_per_token=3.0)

    result = pipeline.apply(MESSAGES, MODEL, token_counter=mine, model_limit=128_000)

    assert result.transforms_applied == ["pipeline:circuit_open"]
    assert result.tokens_before == result.tokens_after == mine.count_messages(MESSAGES)


def test_simulate_forwards_the_callers_counter() -> None:
    pipeline = TransformPipeline(transforms=[], provider=_FlippingProvider())
    mine = EstimatingTokenCounter(chars_per_token=3.0)

    result = pipeline.simulate(MESSAGES, MODEL, token_counter=mine, model_limit=128_000)

    assert result.tokens_before == mine.count_messages(MESSAGES)


def test_token_counter_is_not_forwarded_to_transforms() -> None:
    """The argument is the pipeline's, not a transform kwarg."""
    seen: dict[str, Any] = {}

    class _Spy:
        name = "spy"

        def should_apply(self, messages: Any, tokenizer: Any, **kwargs: Any) -> bool:
            seen.update(kwargs)
            seen["tokenizer"] = tokenizer
            return False

    pipeline = TransformPipeline(transforms=[_Spy()], provider=_FlippingProvider())  # type: ignore[list-item]
    mine = EstimatingTokenCounter(chars_per_token=3.0)

    pipeline.apply(MESSAGES, MODEL, token_counter=mine, model_limit=128_000)

    assert "token_counter" not in seen
    assert seen["tokenizer"].count_messages(MESSAGES) == mine.count_messages(MESSAGES)
