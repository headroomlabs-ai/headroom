"""Relevance scorer benchmarks for Headroom SDK.

This module contains performance benchmarks for relevance scorers:
- BM25Scorer: Zero-dependency keyword matching
- HybridScorer: BM25 + embedding fusion (with graceful fallback)

Performance Targets:
    BM25Scorer:
        - Single item: < 0.1ms
        - Batch 100: < 1ms
        - Batch 1000: < 10ms

    HybridScorer (BM25 fallback):
        - Single item: < 0.2ms
        - Batch 100: < 2ms

    HybridScorer (with embeddings):
        - Single item: < 5ms
        - Batch 100: < 50ms

Run with:
    pytest benchmarks/bench_relevance.py --benchmark-only -v
"""

from __future__ import annotations

import json

import pytest


def _check_embedding_available() -> bool:
    """Check if sentence-transformers is available for embedding tests."""
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


class TestBM25Benchmarks:
    """Benchmarks for BM25 keyword relevance scorer.

    BM25Scorer performs:
    - Text tokenization (regex-based)
    - IDF computation
    - BM25 score calculation
    - Long-token bonus (UUIDs, IDs)

    Expected performance:
    - O(n*m) where n=tokens in item, m=tokens in query
    - Single item: < 0.1ms
    - Batch operations are linear with items
    """

    @pytest.fixture
    def scorer(self):
        """Create BM25 scorer instance."""
        from headroom.relevance.bm25 import BM25Scorer

        return BM25Scorer()

    def test_single_item(
        self,
        benchmark,
        scorer,
        json_items_100,
        query_context_uuid,
    ):
        """Benchmark scoring a single item.

        Target: < 0.1ms
        Tests basic scoring overhead.
        """
        item = json_items_100[0]
        result = benchmark(scorer.score, item, query_context_uuid)

        assert result.score >= 0.0
        assert result.score <= 1.0

    def test_batch_100(
        self,
        benchmark,
        scorer,
        json_items_100,
        query_context_uuid,
    ):
        """Benchmark scoring 100 items in batch.

        Target: < 1ms
        Tests typical batch size for SmartCrusher.
        """
        results = benchmark(scorer.score_batch, json_items_100, query_context_uuid)

        assert len(results) == 100
        assert all(0.0 <= r.score <= 1.0 for r in results)

    def test_batch_1000(
        self,
        benchmark,
        scorer,
        json_items_1000,
        query_context_uuid,
    ):
        """Benchmark scoring 1000 items in batch.

        Target: < 10ms
        Tests larger batch for stress testing.
        """
        results = benchmark(scorer.score_batch, json_items_1000, query_context_uuid)

        assert len(results) == 1000

    def test_uuid_matching(
        self,
        benchmark,
        scorer,
        json_items_100,
        query_context_uuid,
    ):
        """Benchmark UUID pattern matching.

        Target: < 1ms
        Tests regex efficiency for UUID detection.
        """
        # Query contains UUID - tests that BM25 can handle long token patterns
        results = benchmark(scorer.score_batch, json_items_100, query_context_uuid)

        # Verify scoring completes - specific matches depend on generated data
        assert len(results) == 100
        assert all(r.score >= 0.0 for r in results)

    def test_semantic_query(
        self,
        benchmark,
        scorer,
        json_items_100,
        query_context_semantic,
    ):
        """Benchmark semantic query (BM25 limitations).

        Target: < 1ms
        Tests keyword matching on semantic queries.
        """
        # BM25 will only match literal terms
        results = benchmark(scorer.score_batch, json_items_100, query_context_semantic)

        assert len(results) == 100

    def test_empty_context(
        self,
        benchmark,
        scorer,
        json_items_100,
    ):
        """Benchmark with empty query context.

        Target: < 0.5ms
        Tests early-exit optimization.
        """
        results = benchmark(scorer.score_batch, json_items_100, "")

        # All scores should be 0 with no context
        assert all(r.score == 0.0 for r in results)

    def test_long_items(
        self,
        benchmark,
        scorer,
        log_entries_1000,
        query_context_semantic,
    ):
        """Benchmark scoring longer items (log entries).

        Target: < 15ms
        Tests performance with larger text per item.
        """
        json_items = [json.dumps(entry) for entry in log_entries_1000]
        results = benchmark(scorer.score_batch, json_items, query_context_semantic)

        assert len(results) == 1000


class TestHybridBenchmarks:
    """Benchmarks for Hybrid BM25+Embedding scorer.

    HybridScorer performs:
    - BM25 scoring (always)
    - Embedding scoring (if available)
    - Adaptive alpha computation
    - Score fusion

    Without embeddings (fallback mode):
    - Single item: < 0.2ms
    - Batch 100: < 2ms

    With embeddings (full mode):
    - Single item: < 5ms (model inference)
    - Batch 100: < 50ms (batched inference)
    """

    @pytest.fixture
    def scorer_fallback(self):
        """Create hybrid scorer without embeddings (BM25 fallback)."""
        from headroom.relevance.bm25 import BM25Scorer
        from headroom.relevance.hybrid import HybridScorer

        # Force BM25-only mode by not providing embedding scorer
        scorer = HybridScorer(
            alpha=0.5,
            adaptive=True,
            bm25_scorer=BM25Scorer(),
            embedding_scorer=None,
        )
        # Ensure we're in fallback mode
        scorer._embedding_available = False
        return scorer

    @pytest.fixture
    def scorer_full(self):
        """Create hybrid scorer with embeddings (if available)."""
        from headroom.relevance.hybrid import HybridScorer

        scorer = HybridScorer(alpha=0.5, adaptive=True)
        return scorer

    def test_single_item_fallback(
        self,
        benchmark,
        scorer_fallback,
        json_items_100,
        query_context_uuid,
    ):
        """Benchmark single item scoring (BM25 fallback).

        Target: < 0.2ms
        Tests fallback mode overhead.
        """
        item = json_items_100[0]
        result = benchmark(scorer_fallback.score, item, query_context_uuid)

        assert "BM25 only" in result.reason

    def test_batch_100_fallback(
        self,
        benchmark,
        scorer_fallback,
        json_items_100,
        query_context_uuid,
    ):
        """Benchmark batch scoring (BM25 fallback).

        Target: < 2ms
        Tests fallback batch performance.
        """
        results = benchmark(scorer_fallback.score_batch, json_items_100, query_context_uuid)

        assert len(results) == 100

    def test_adaptive_alpha_uuid(
        self,
        benchmark,
        scorer_fallback,
        json_items_100,
        query_context_uuid,
    ):
        """Benchmark adaptive alpha with UUID query.

        Target: < 2ms
        Tests alpha computation overhead.
        """
        results = benchmark(scorer_fallback.score_batch, json_items_100, query_context_uuid)

        # UUID query should favor BM25 (but we're in fallback mode)
        assert len(results) == 100

    def test_adaptive_alpha_semantic(
        self,
        benchmark,
        scorer_fallback,
        json_items_100,
        query_context_semantic,
    ):
        """Benchmark adaptive alpha with semantic query.

        Target: < 2ms
        Tests alpha computation for semantic queries.
        """
        results = benchmark(scorer_fallback.score_batch, json_items_100, query_context_semantic)

        assert len(results) == 100

    @pytest.mark.skipif(
        not _check_embedding_available(),
        reason="sentence-transformers not installed",
    )
    def test_single_item_full(
        self,
        benchmark,
        scorer_full,
        json_items_100,
        query_context_uuid,
    ):
        """Benchmark single item with embeddings.

        Target: < 5ms
        Tests full hybrid mode (requires sentence-transformers).
        """
        if not scorer_full.has_embedding_support():
            pytest.skip("Embeddings not available")

        item = json_items_100[0]
        result = benchmark(scorer_full.score, item, query_context_uuid)

        # Should show hybrid scoring
        assert "Hybrid" in result.reason

    @pytest.mark.skipif(
        not _check_embedding_available(),
        reason="sentence-transformers not installed",
    )
    def test_batch_100_full(
        self,
        benchmark,
        scorer_full,
        json_items_100,
        query_context_uuid,
    ):
        """Benchmark batch scoring with embeddings.

        Target: < 50ms
        Tests batched embedding inference.
        """
        if not scorer_full.has_embedding_support():
            pytest.skip("Embeddings not available")

        results = benchmark(scorer_full.score_batch, json_items_100, query_context_uuid)

        assert len(results) == 100


class TestScorerFactoryBenchmarks:
    """Benchmarks for scorer factory and initialization."""

    def test_create_bm25_scorer(self, benchmark):
        """Benchmark BM25 scorer creation.

        Target: < 0.1ms
        Tests initialization overhead.
        """
        from headroom.relevance import create_scorer

        scorer = benchmark(create_scorer, tier="bm25")

        assert scorer is not None

    def test_create_hybrid_scorer(self, benchmark):
        """Benchmark hybrid scorer creation.

        Target: < 1ms (without embedding model load)
        Tests initialization with fallback.
        """
        from headroom.relevance import create_scorer

        scorer = benchmark(create_scorer, tier="hybrid")

        assert scorer is not None


class TestRelevanceInSmartCrusher:
    """Benchmarks for relevance scoring within SmartCrusher context.

    Tests the realistic scenario where SmartCrusher uses relevance
    scoring to determine which items to preserve during compression.
    """

    @pytest.fixture
    def crusher_with_bm25(self, smart_crusher_config):
        """SmartCrusher with BM25 relevance scorer."""
        from headroom.config import RelevanceScorerConfig
        from headroom.transforms.smart_crusher import SmartCrusher

        return SmartCrusher(
            config=smart_crusher_config,
            relevance_config=RelevanceScorerConfig(tier="bm25"),
        )

    @pytest.fixture
    def crusher_with_hybrid(self, smart_crusher_config):
        """SmartCrusher with hybrid relevance scorer."""
        from headroom.config import RelevanceScorerConfig
        from headroom.transforms.smart_crusher import SmartCrusher

        return SmartCrusher(
            config=smart_crusher_config,
            relevance_config=RelevanceScorerConfig(tier="hybrid"),
        )

    def test_crush_with_bm25_relevance(
        self,
        benchmark,
        crusher_with_bm25,
        mock_tokenizer,
        items_100,
    ):
        """Benchmark crushing with BM25 relevance scoring.

        Target: < 3ms
        Tests BM25 integration overhead.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Find user 550e8400-e29b-41d4-a716-446655440000"},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": json.dumps(items_100),
            },
        ]

        result = benchmark(crusher_with_bm25.apply, messages, mock_tokenizer)

        assert result.tokens_after < result.tokens_before

    def test_crush_with_hybrid_relevance(
        self,
        benchmark,
        crusher_with_hybrid,
        mock_tokenizer,
        items_100,
    ):
        """Benchmark crushing with hybrid relevance scoring.

        Target: < 60ms (with embeddings) or < 3ms (fallback)
        Tests hybrid integration.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Show me failed requests and errors"},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": json.dumps(items_100),
            },
        ]

        result = benchmark(crusher_with_hybrid.apply, messages, mock_tokenizer)

        assert result.tokens_after < result.tokens_before

    def test_crush_large_with_relevance(
        self,
        benchmark,
        crusher_with_bm25,
        mock_tokenizer,
        items_1000,
    ):
        """Benchmark crushing 1000 items with relevance.

        Target: < 15ms
        Tests scalability of relevance scoring.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Search for Alice and find any errors"},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": json.dumps(items_1000),
            },
        ]

        result = benchmark(crusher_with_bm25.apply, messages, mock_tokenizer)

        assert result.tokens_after < result.tokens_before


def _check_embedding_available() -> bool:
    """Check if embedding scorer is available."""
    try:
        from headroom.relevance.embedding import EmbeddingScorer

        return EmbeddingScorer.is_available()
    except ImportError:
        return False
