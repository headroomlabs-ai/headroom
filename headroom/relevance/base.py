"""Base protocol for relevance scoring in Headroom SDK.

This module defines the RelevanceScorer protocol - a unified interface for
computing item relevance against a query context. All transforms that make
keep/drop decisions can use this abstraction.

The pattern: relevance(item, context) -> float [0.0, 1.0]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RelevanceScore:
    """Relevance score with explainability.

    Attributes:
        score: Relevance score from 0.0 (irrelevant) to 1.0 (highly relevant).
        reason: Human-readable explanation of the score.
        matched_terms: List of terms that contributed to the match (for debugging).
    """

    score: float
    reason: str = ""
    matched_terms: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Clamp score to valid range."""
        self.score = max(0.0, min(1.0, self.score))


class RelevanceScorer(ABC):
    """Abstract base class for relevance scoring.

    All relevance scorers must implement:
    - score(): Score a single item against context
    - score_batch(): Score multiple items efficiently

    Example usage:
        scorer = BM25Scorer()
        items = ['{"id": "123", "name": "Alice"}', '{"id": "456", "name": "Bob"}']
        context = "find user 123"
        scores = scorer.score_batch(items, context)
        # scores[0].score > scores[1].score (item 0 matches "123")
    """

    @abstractmethod
    def score(self, item: str, context: str) -> RelevanceScore:
        """Score a single item's relevance to the context.

        Args:
            item: String representation of the item (typically JSON).
            context: Query context (user message, tool call args, etc.).

        Returns:
            RelevanceScore with score [0.0, 1.0] and explanation.
        """
        pass

    @abstractmethod
    def score_batch(self, items: list[str], context: str) -> list[RelevanceScore]:
        """Score multiple items efficiently.

        Default implementation calls score() for each item.
        Subclasses should override for batch-optimized implementations.

        Args:
            items: List of string representations of items.
            context: Query context to score against.

        Returns:
            List of RelevanceScore objects, one per item.
        """
        pass

    @classmethod
    def is_available(cls) -> bool:
        """Check if this scorer is available (dependencies installed).

        Override in subclasses that have optional dependencies.

        Returns:
            True if the scorer can be instantiated.
        """
        return True


def default_batch_score(
    scorer: RelevanceScorer, items: list[str], context: str
) -> list[RelevanceScore]:
    """Default batch implementation that calls score() per item.

    Use this as a fallback for scorers that don't have optimized batching.

    Args:
        scorer: The scorer instance.
        items: List of items to score.
        context: Query context.

    Returns:
        List of scores.
    """
    return [scorer.score(item, context) for item in items]
