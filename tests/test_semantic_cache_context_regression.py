"""Regression: semantic cache must not serve a response across different
conversation contexts when only the final user message is similar.

Bug: SemanticCache.get() matched purely on the final-query embedding and
ignored prior conversation, so convo B got convo A's cached answer when both
ended in the same message (e.g. "continue", "fix it").
"""

from headroom.cache.base import (
    BaseCacheOptimizer,
    CacheResult,
    CacheStrategy,
    OptimizationContext,
)
from headroom.cache.semantic import SemanticCacheLayer


class _NoopOptimizer(BaseCacheOptimizer):
    @property
    def name(self) -> str:
        return "noop"

    @property
    def provider(self) -> str:
        return "noop"

    @property
    def strategy(self) -> CacheStrategy:
        return list(CacheStrategy)[0]

    def optimize(self, messages, context, config=None):
        return CacheResult(messages=messages)


def _embed(text: str) -> list[float]:
    # Deterministic embedding: identical text -> identical vector, so the
    # final-query similarity is exactly 1.0. This isolates the context check.
    return [float(b) for b in text.encode()[:8]] or [0.0]


def _layer():
    return SemanticCacheLayer(
        _NoopOptimizer(),
        similarity_threshold=0.95,
        embedding_fn=_embed,
    )


def test_no_cross_context_hit():
    layer = _layer()

    convo_a = [
        {"role": "user", "content": "Refactor the auth module to use JWT."},
        {"role": "assistant", "content": "Done, JWT wired in."},
        {"role": "user", "content": "continue"},
    ]
    convo_b = [
        {"role": "user", "content": "Delete all production logs."},
        {"role": "assistant", "content": "Logs deleted."},
        {"role": "user", "content": "continue"},
    ]

    layer.store_response(convo_a, response="ANSWER_A")

    # Same final user turn ("continue"), different prior context.
    result = layer.process(convo_b, OptimizationContext())
    assert not result.semantic_cache_hit, "served convo A's answer to convo B"


def test_same_context_still_hits():
    layer = _layer()

    convo = [
        {"role": "user", "content": "Summarize the report."},
        {"role": "assistant", "content": "Here is the summary."},
        {"role": "user", "content": "continue"},
    ]
    layer.store_response(convo, response="ANSWER")

    result = layer.process(convo, OptimizationContext())
    assert result.semantic_cache_hit
    assert result.cached_response == "ANSWER"


if __name__ == "__main__":
    test_no_cross_context_hit()
    test_same_context_still_hits()
    print("PASS")
