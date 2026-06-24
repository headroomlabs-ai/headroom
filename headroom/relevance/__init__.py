"""Relevance scoring module for Headroom SDK.

This module provides a unified interface for computing item relevance against
query contexts. All scorers implement the RelevanceScorer protocol:

    relevance(item, context) -> RelevanceScore

Available scorers:

1. HybridScorer (DEFAULT - recommended)
   - Combines BM25 + embeddings for best accuracy
   - Adaptive alpha: more BM25 for UUIDs, more semantic for natural language
   - Falls back gracefully to BM25 if sentence-transformers not installed
   - Install for full support: pip install headroom[relevance]

2. BM25Scorer (zero dependencies)
   - Fast keyword matching
   - Good for exact UUIDs, IDs, specific terms
   - May miss semantic matches ("errors" won't match "failed")

3. EmbeddingScorer (requires sentence-transformers)
   - Pure semantic similarity
   - Best for natural language queries
   - Install: pip install headroom[relevance]

WHY HYBRID IS DEFAULT:
- Missing important items during compression is catastrophic
- BM25 alone gives low scores for single-term matches (e.g., "Alice" = 0.07)
- Semantic matching catches "errors" -> "failed", "issues", etc.
- 5-10ms latency is acceptable vs. losing critical data

Example usage:
    from headroom.relevance import HybridScorer, create_scorer

    # Default: Hybrid (recommended)
    scorer = create_scorer()  # or HybridScorer()

    # Zero-dependency fallback
    scorer = create_scorer("bm25")

    # Score items
    items = [
        '{"id": "123", "name": "Alice"}',
        '{"id": "456", "name": "Bob"}',
    ]
    scores = scorer.score_batch(items, "find user 123")
    # scores[0].score > scores[1].score
"""

from typing import Any

from .base import RelevanceScore, RelevanceScorer
from .bm25 import BM25Scorer
from .embedding import EmbeddingScorer, embedding_available
from .hybrid import HybridScorer

__all__ = [
    # Base types
    "RelevanceScore",
    "RelevanceScorer",
    # Scorers
    "BM25Scorer",
    "EmbeddingScorer",
    "HybridScorer",
    # Utilities
    "embedding_available",
    # Factory function
    "create_scorer",
]


def create_scorer(
    tier: str = "hybrid",
    **kwargs: Any,
) -> RelevanceScorer:
    """Factory function to create a relevance scorer.

    Args:
        tier: Scorer tier to create:
            - "hybrid": Hybrid BM25 + embedding (DEFAULT, recommended)
            - "bm25": BM25 keyword scorer (zero deps, fast)
            - "embedding": Embedding scorer (requires sentence-transformers)
        **kwargs: Additional arguments passed to scorer constructor.

    Returns:
        RelevanceScorer instance.

    Raises:
        ValueError: If tier is unknown.
        RuntimeError: If tier requires unavailable dependencies.

    Note:
        HybridScorer gracefully falls back to BM25 if sentence-transformers
        is not installed, so it's safe to use as the default.

    Example:
        # Create default hybrid scorer (recommended)
        scorer = create_scorer()

        # Create BM25 scorer for zero-dependency environments
        scorer = create_scorer("bm25")

        # Create hybrid scorer with custom alpha
        scorer = create_scorer("hybrid", alpha=0.6, adaptive=True)
    """
    tier = tier.lower()

    if tier == "bm25":
        return BM25Scorer(**kwargs)

    elif tier == "embedding":
        if not EmbeddingScorer.is_available():
            raise RuntimeError(
                "EmbeddingScorer requires sentence-transformers. "
                "Install with: pip install headroom[relevance]"
            )
        return EmbeddingScorer(**kwargs)

    elif tier == "hybrid":
        return HybridScorer(**kwargs)

    else:
        valid_tiers = ["bm25", "embedding", "hybrid"]
        raise ValueError(f"Unknown scorer tier: {tier}. Valid tiers: {valid_tiers}")
