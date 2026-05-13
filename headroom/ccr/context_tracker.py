"""Multi-turn context tracking for CCR (Compress-Cache-Retrieve).

This module tracks compressed content across conversation turns and
provides intelligent context expansion based on query relevance.

Key features:
1. Track all compression hashes across the conversation
2. Analyze new queries to detect if they need expanded context
3. Proactively expand relevant compressed content before LLM responds
4. Prevent "context amnesia" where earlier compressed data is forgotten

Example:
    Turn 1: Search returns 100 files → compressed to 10 (hash=abc123)
    Turn 5: User asks "What about auth middleware?"

    Without tracking: LLM doesn't know auth_middleware.py exists
    With tracking: Tracker detects "auth middleware" might be in abc123,
                   proactively expands it, LLM gets the full context
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from ..cache.compression_store import get_compression_store

logger = logging.getLogger(__name__)


@dataclass
class CompressedContext:
    """Represents a piece of compressed context from the conversation."""

    hash_key: str
    turn_number: int
    timestamp: float
    tool_name: str | None
    original_item_count: int
    compressed_item_count: int
    query_context: str  # The query/context when compression happened
    sample_content: str  # Preview of what was compressed (for relevance matching)


@dataclass
class ExpansionRecommendation:
    """Recommendation to expand compressed context."""

    hash_key: str
    reason: str
    relevance_score: float
    expand_full: bool = True  # True = expand all, False = search only
    search_query: str | None = None


@dataclass
class ContextTrackerConfig:
    """Configuration for context tracking."""

    # Whether tracking is enabled
    enabled: bool = True

    # Maximum contexts to track (LRU eviction)
    max_tracked_contexts: int = 100

    # Relevance threshold for recommending expansion (0-1)
    relevance_threshold: float = 0.3

    # Maximum age for contexts (seconds) - older contexts less likely to expand
    max_context_age_seconds: float = 300.0  # 5 minutes

    # Whether to proactively expand based on query analysis
    proactive_expansion: bool = True

    # Maximum items to proactively expand per turn
    max_proactive_expansions: int = 2


class ContextTracker:
    """Tracks compressed contexts across conversation turns.

    This tracker maintains awareness of what has been compressed
    and can recommend expansions when new queries might need that data.

    Usage:
        tracker = ContextTracker()

        # Track compression events
        tracker.track_compression(
            hash_key="abc123",
            turn_number=1,
            tool_name="Bash",
            original_count=100,
            compressed_count=10,
            query_context="find all python files",
            sample_content='["src/main.py", "src/auth.py", ...]',
        )

        # On new user message, check for expansion needs
        recommendations = tracker.analyze_query(
            query="What about the authentication code?",
            current_turn=5,
        )

        # recommendations might suggest expanding abc123 because
        # "authentication" matches "auth.py" in the sample content
    """

    def __init__(self, config: ContextTrackerConfig | None = None):
        self.config = config or ContextTrackerConfig()
        self._contexts: dict[str, CompressedContext] = {}
        self._turn_order: list[str] = []  # For LRU
        self._current_turn: int = 0

    def track_compression(
        self,
        hash_key: str,
        turn_number: int,
        tool_name: str | None,
        original_count: int,
        compressed_count: int,
        query_context: str = "",
        sample_content: str = "",
    ) -> None:
        """Track a compression event.

        Args:
            hash_key: The CCR hash for this compression.
            turn_number: The conversation turn number.
            tool_name: Name of the tool whose output was compressed.
            original_count: Original item count.
            compressed_count: Compressed item count.
            query_context: The user query when compression happened.
            sample_content: Sample of the content for relevance matching.
        """
        if not self.config.enabled:
            return

        context = CompressedContext(
            hash_key=hash_key,
            turn_number=turn_number,
            timestamp=time.time(),
            tool_name=tool_name,
            original_item_count=original_count,
            compressed_item_count=compressed_count,
            query_context=query_context,
            sample_content=sample_content[:2000],  # Limit sample size
        )

        # Add or update context
        if hash_key in self._contexts:
            self._turn_order.remove(hash_key)
        self._contexts[hash_key] = context
        self._turn_order.append(hash_key)

        # LRU eviction
        while len(self._contexts) > self.config.max_tracked_contexts:
            oldest = self._turn_order.pop(0)
            del self._contexts[oldest]

        self._current_turn = max(self._current_turn, turn_number)

        logger.debug(
            f"CCR Tracker: Tracked compression {hash_key} "
            f"({original_count} -> {compressed_count} items)"
        )

    def analyze_query(
        self,
        query: str,
        current_turn: int | None = None,
    ) -> list[ExpansionRecommendation]:
        """Analyze a query to find relevant compressed contexts.

        Args:
            query: The user's query/message.
            current_turn: Current turn number (for age calculation).

        Returns:
            List of expansion recommendations, sorted by relevance.
        """
        if not self.config.enabled or not self.config.proactive_expansion:
            return []

        if current_turn is not None:
            self._current_turn = current_turn

        recommendations: list[ExpansionRecommendation] = []
        now = time.time()

        for hash_key, context in self._contexts.items():
            # Check age
            age = now - context.timestamp
            if age > self.config.max_context_age_seconds:
                continue

            # Calculate relevance
            relevance = self._calculate_relevance(query, context)

            # Age discount: older contexts get lower scores
            age_factor = 1.0 - (age / self.config.max_context_age_seconds) * 0.5
            relevance *= age_factor

            if relevance >= self.config.relevance_threshold:
                # Determine if full expansion or search
                expand_full, search_query = self._determine_expansion_type(
                    query, context, relevance
                )

                recommendations.append(
                    ExpansionRecommendation(
                        hash_key=hash_key,
                        reason=self._generate_reason(query, context, relevance),
                        relevance_score=relevance,
                        expand_full=expand_full,
                        search_query=search_query,
                    )
                )

        # Sort by relevance, limit count
        recommendations.sort(key=lambda r: r.relevance_score, reverse=True)
        return recommendations[: self.config.max_proactive_expansions]

    def _calculate_relevance(
        self,
        query: str,
        context: CompressedContext,
    ) -> float:
        """Calculate relevance score between query and compressed context.

        Uses simple but effective heuristics:
        1. Keyword overlap with sample content
        2. Keyword overlap with original query context
        3. Tool name relevance
        """
        query_lower = query.lower()
        query_words = set(self._extract_keywords(query_lower))

        if not query_words:
            return 0.0

        score = 0.0

        # Check sample content overlap
        sample_lower = context.sample_content.lower()
        sample_words = set(self._extract_keywords(sample_lower))

        if sample_words:
            overlap = query_words & sample_words
            score += len(overlap) / len(query_words) * 0.5

            # Bonus for exact substring matches
            for word in query_words:
                if len(word) >= 4 and word in sample_lower:
                    score += 0.2

        # Check original query context overlap
        if context.query_context:
            context_lower = context.query_context.lower()
            context_words = set(self._extract_keywords(context_lower))

            if context_words:
                overlap = query_words & context_words
                score += len(overlap) / len(query_words) * 0.3

        # Tool name relevance
        if context.tool_name:
            tool_lower = context.tool_name.lower()
            # File operations more likely to need expansion
            if any(w in tool_lower for w in ["find", "glob", "search", "grep", "ls"]):
                if any(w in query_lower for w in ["file", "where", "find", "show", "list"]):
                    score += 0.1

        return min(score, 1.0)

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from text."""
        # Remove common punctuation, split into words
        words = re.findall(r"\b[a-z][a-z0-9_.-]*[a-z0-9]\b|\b[a-z]{2,}\b", text)

        # Filter stop words and very short words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "and",
            "but",
            "if",
            "or",
            "because",
            "until",
            "while",
            "this",
            "that",
            "these",
            "those",
            "what",
            "which",
            "who",
            "whom",
            "it",
            "its",
            "me",
            "my",
            "i",
            "you",
        }

        return [w for w in words if w not in stop_words and len(w) >= 2]

    def _determine_expansion_type(
        self,
        query: str,
        context: CompressedContext,
        relevance: float,
    ) -> tuple[bool, str | None]:
        """Determine whether to do full expansion or search.

        Returns:
            Tuple of (expand_full, search_query)
        """
        # High relevance + small original count = full expansion
        if relevance > 0.6 or context.original_item_count <= 50:
            return True, None

        # Extract specific search terms from query
        keywords = self._extract_keywords(query.lower())

        # Filter to most specific keywords (longer, less common)
        specific_keywords = [
            k
            for k in keywords
            if len(k) >= 4 and k not in {"file", "code", "show", "find", "list", "what"}
        ]

        if specific_keywords:
            # Use top keywords as search query
            search_query = " ".join(specific_keywords[:3])
            return False, search_query

        # Default to full expansion if we can't form a good search
        return True, None

    def _generate_reason(
        self,
        query: str,
        context: CompressedContext,
        relevance: float,
    ) -> str:
        """Generate human-readable reason for expansion recommendation."""
        parts = []

        if context.tool_name:
            parts.append(f"from {context.tool_name}")

        parts.append(
            f"{context.original_item_count} items compressed in turn {context.turn_number}"
        )

        if relevance > 0.5:
            parts.append("high relevance to current query")
        else:
            parts.append("possible relevance to current query")

        return ", ".join(parts)

    def execute_expansions(
        self,
        recommendations: list[ExpansionRecommendation],
    ) -> list[dict[str, Any]]:
        """Execute expansion recommendations and return the expanded content.

        Args:
            recommendations: List of expansion recommendations.

        Returns:
            List of expanded content dicts with hash, content, and metadata.
        """
        store = get_compression_store()
        results = []

        for rec in recommendations:
            try:
                if rec.expand_full:
                    entry = store.retrieve(rec.hash_key)
                    if entry:
                        results.append(
                            {
                                "hash": rec.hash_key,
                                "type": "full",
                                "content": entry.original_content,
                                "item_count": entry.original_item_count,
                                "reason": rec.reason,
                            }
                        )
                        logger.info(
                            f"CCR Tracker: Proactively expanded {rec.hash_key} "
                            f"({entry.original_item_count} items)"
                        )
                else:
                    search_results = store.search(rec.hash_key, rec.search_query or "")
                    if search_results:
                        results.append(
                            {
                                "hash": rec.hash_key,
                                "type": "search",
                                "query": rec.search_query,
                                "content": search_results,
                                "item_count": len(search_results),
                                "reason": rec.reason,
                            }
                        )
                        logger.info(
                            f"CCR Tracker: Proactive search in {rec.hash_key} "
                            f"for '{rec.search_query}' ({len(search_results)} results)"
                        )
            except Exception as e:
                logger.warning(f"CCR Tracker: Failed to expand {rec.hash_key}: {e}")

        return results

    def format_expansions_for_context(
        self,
        expansions: list[dict[str, Any]],
    ) -> str:
        """Format expansions as additional context for the LLM.

        Args:
            expansions: Results from execute_expansions.

        Returns:
            Formatted string to add to context.
        """
        if not expansions:
            return ""

        count = len(expansions)
        noun = "item" if count == 1 else "items"
        parts = [
            f"[Proactive Context Expansion - scope: this session only, "
            f"{count} compressed {noun} expanded]"
        ]

        for exp in expansions:
            if exp["type"] == "full":
                parts.append(f"\n--- Expanded from earlier ({exp['reason']}) ---")
                parts.append(exp["content"])
            else:
                parts.append(f"\n--- Search results for '{exp['query']}' ({exp['reason']}) ---")
                if isinstance(exp["content"], list):
                    parts.append(json.dumps(exp["content"], indent=2))
                else:
                    parts.append(str(exp["content"]))

        parts.append("\n[End Proactive Expansion]")

        return "\n".join(parts)

    def get_tracked_hashes(self) -> list[str]:
        """Get list of currently tracked hashes."""
        return list(self._contexts.keys())

    def get_stats(self) -> dict[str, Any]:
        """Get tracker statistics."""
        return {
            "tracked_contexts": len(self._contexts),
            "current_turn": self._current_turn,
            "config": {
                "enabled": self.config.enabled,
                "max_contexts": self.config.max_tracked_contexts,
                "relevance_threshold": self.config.relevance_threshold,
                "proactive_expansion": self.config.proactive_expansion,
            },
            "contexts": [
                {
                    "hash": ctx.hash_key,
                    "turn": ctx.turn_number,
                    "tool": ctx.tool_name,
                    "items": f"{ctx.compressed_item_count}/{ctx.original_item_count}",
                }
                for ctx in self._contexts.values()
            ],
        }

    def clear(self) -> None:
        """Clear all tracked contexts."""
        self._contexts.clear()
        self._turn_order.clear()
        self._current_turn = 0


# Global instance (per-session)
_context_tracker: ContextTracker | None = None


def get_context_tracker() -> ContextTracker:
    """Get the global context tracker."""
    global _context_tracker
    if _context_tracker is None:
        _context_tracker = ContextTracker()
    return _context_tracker


def reset_context_tracker() -> None:
    """Reset the global context tracker."""
    global _context_tracker
    if _context_tracker is not None:
        _context_tracker.clear()
    _context_tracker = None
