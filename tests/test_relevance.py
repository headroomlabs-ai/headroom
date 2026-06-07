"""Tests for the relevance scoring module.

Tests all scorer tiers:
- BM25Scorer (zero dependencies)
- EmbeddingScorer (requires sentence-transformers)
- HybridScorer (combines BM25 + embeddings)
"""

import pytest

from headroom.relevance import (
    BM25Scorer,
    EmbeddingScorer,
    HybridScorer,
    RelevanceScore,
    create_scorer,
    embedding_available,
)


class TestRelevanceScore:
    """Tests for RelevanceScore dataclass."""

    def test_score_clamping_high(self):
        """Scores above 1.0 are clamped."""
        score = RelevanceScore(score=1.5, reason="test")
        assert score.score == 1.0

    def test_score_clamping_low(self):
        """Scores below 0.0 are clamped."""
        score = RelevanceScore(score=-0.5, reason="test")
        assert score.score == 0.0

    def test_score_valid_range(self):
        """Scores in valid range are preserved."""
        score = RelevanceScore(score=0.75, reason="test")
        assert score.score == 0.75


class TestBM25Scorer:
    """Tests for BM25 keyword relevance scorer."""

    def test_exact_uuid_match(self):
        """BM25 finds exact UUID matches."""
        scorer = BM25Scorer()
        items = [
            '{"id": "550e8400-e29b-41d4-a716-446655440000", "name": "Alice"}',
            '{"id": "123e4567-e89b-12d3-a456-426614174000", "name": "Bob"}',
        ]
        context = "find record 550e8400-e29b-41d4-a716-446655440000"

        scores = scorer.score_batch(items, context)
        assert scores[0].score > scores[1].score
        assert "550e8400-e29b-41d4-a716-446655440000" in scores[0].matched_terms

    def test_numeric_id_match(self):
        """BM25 matches numeric IDs."""
        scorer = BM25Scorer()
        items = [
            '{"user_id": 12345, "name": "Alice"}',
            '{"user_id": 67890, "name": "Bob"}',
        ]
        context = "find user 12345"

        scores = scorer.score_batch(items, context)
        assert scores[0].score > scores[1].score

    def test_keyword_match(self):
        """BM25 matches keywords."""
        scorer = BM25Scorer()
        items = [
            '{"status": "error", "message": "Connection refused"}',
            '{"status": "success", "data": [1, 2, 3]}',
        ]
        context = "show me errors"

        scores = scorer.score_batch(items, context)
        # "error" should match "errors" via common stem
        assert scores[0].score >= scores[1].score

    def test_empty_context(self):
        """Empty context returns zero scores."""
        scorer = BM25Scorer()
        items = ['{"id": "123"}', '{"id": "456"}']
        context = ""

        scores = scorer.score_batch(items, context)
        assert all(s.score == 0.0 for s in scores)

    def test_empty_items(self):
        """Empty items list returns empty scores."""
        scorer = BM25Scorer()
        scores = scorer.score_batch([], "some context")
        assert scores == []

    def test_single_item(self):
        """Single item scoring works."""
        scorer = BM25Scorer()
        item = '{"name": "Alice", "role": "admin"}'
        context = "find Alice"

        score = scorer.score(item, context)
        assert score.score > 0
        assert "alice" in [t.lower() for t in score.matched_terms]

    def test_is_available(self):
        """BM25Scorer is always available."""
        assert BM25Scorer.is_available()

    def test_compute_idf_follows_standard_formula(self):
        """IDF rewards rare terms and decays toward zero for common terms."""
        scorer = BM25Scorer()

        # Absent term contributes nothing.
        assert scorer._compute_idf("x", doc_count=10, doc_freq=0) == 0.0

        # A term in 1/10 docs is more discriminative than one in 9/10 docs.
        rare = scorer._compute_idf("x", doc_count=10, doc_freq=1)
        common = scorer._compute_idf("x", doc_count=10, doc_freq=9)
        assert rare > common > 0

    def test_batch_idf_downweights_common_terms(self):
        """A discriminative term outranks one shared across the whole corpus.

        ``shared`` appears in every item, so its corpus IDF approaches zero,
        while ``zeta`` appears in a single item and stays discriminative. An
        item matched only on the rare term must therefore outrank an item
        matched only on the ubiquitous term.
        """
        scorer = BM25Scorer()
        items = [
            "shared zeta",  # matches both query terms, one of them rare
            "shared alpha",  # matches only the ubiquitous term
            "shared beta",
            "shared gamma",
        ]
        context = "shared zeta"

        scores = scorer.score_batch(items, context)
        assert scores[0].score > scores[1].score
        assert scores[0].score > scores[2].score

    def test_batch_idf_does_not_change_matched_terms(self):
        """Corpus IDF affects ranking only, not which terms are reported."""
        scorer = BM25Scorer()
        items = ["alpha", "alpha beta"]
        scores = scorer.score_batch(items, "alpha beta")
        assert scores[0].matched_terms == ["alpha"]
        assert sorted(scores[1].matched_terms) == ["alpha", "beta"]


class TestEmbeddingScorer:
    """Tests for embedding-based semantic scorer."""

    @pytest.fixture
    def scorer(self):
        """Create embedding scorer if available."""
        if not EmbeddingScorer.is_available():
            pytest.skip("sentence-transformers not installed")
        return EmbeddingScorer()

    def test_semantic_match(self, scorer):
        """Embeddings find semantic matches."""
        items = [
            '{"status": "failed", "error": "connection refused"}',
            '{"status": "success", "data": [1, 2, 3]}',
        ]
        context = "show me the errors"

        scores = scorer.score_batch(items, context)
        # "failed"/"error" semantically relates to "errors"
        assert scores[0].score > scores[1].score

    def test_paraphrase_match(self, scorer):
        """Embeddings match paraphrases."""
        items = [
            '{"message": "The server crashed with a fatal error"}',
            '{"message": "The weather today is sunny and warm"}',
        ]
        context = "system failure and errors"

        scores = scorer.score_batch(items, context)
        # "server crashed with fatal error" is much closer to "system failure and errors"
        # than "weather is sunny" - this should be a clear semantic difference
        assert scores[0].score > scores[1].score

    def test_batch_efficiency(self, scorer):
        """Batch scoring is efficient."""
        items = [f'{{"id": {i}}}' for i in range(100)]
        context = "find item"

        # Should not raise and should complete quickly
        scores = scorer.score_batch(items, context)
        assert len(scores) == 100


class TestHybridScorer:
    """Tests for hybrid BM25 + embedding scorer."""

    def test_always_available(self):
        """HybridScorer is always available (falls back to BM25)."""
        assert HybridScorer.is_available()

    def test_uuid_query_favors_bm25(self):
        """UUID queries increase BM25 weight."""
        scorer = HybridScorer(adaptive=True)

        # UUID query
        alpha_uuid = scorer._compute_alpha("find 550e8400-e29b-41d4-a716-446655440000")
        # Semantic query
        alpha_semantic = scorer._compute_alpha("show me recent errors")

        assert alpha_uuid > alpha_semantic
        assert alpha_uuid >= 0.8  # High BM25 weight for UUIDs

    def test_numeric_id_query_increases_alpha(self):
        """Numeric ID queries increase BM25 weight."""
        scorer = HybridScorer(adaptive=True)

        alpha_with_ids = scorer._compute_alpha("find users 12345 and 67890")
        alpha_no_ids = scorer._compute_alpha("show all users")

        assert alpha_with_ids > alpha_no_ids

    def test_fixed_alpha_mode(self):
        """Fixed alpha mode uses constant weight."""
        scorer = HybridScorer(alpha=0.7, adaptive=False)

        alpha1 = scorer._compute_alpha("find 550e8400-e29b-41d4-a716-446655440000")
        alpha2 = scorer._compute_alpha("show me errors")

        assert alpha1 == alpha2 == 0.7

    def test_fallback_to_bm25(self):
        """Without embeddings, returns BM25 only."""
        # Create scorer without embeddings
        scorer = HybridScorer()
        scorer._embedding_available = False
        scorer.embedding = None

        items = ['{"id": "123"}', '{"id": "456"}']
        context = "find 123"

        scores = scorer.score_batch(items, context)
        assert all("BM25 only" in s.reason for s in scores)

    def test_hybrid_scoring(self):
        """Hybrid scoring combines BM25 and embeddings when available."""
        scorer = HybridScorer(adaptive=False, alpha=0.5)

        if not scorer.has_embedding_support():
            pytest.skip("sentence-transformers not installed")

        items = ['{"id": "123", "name": "Alice"}']
        context = "find user 123"

        scores = scorer.score_batch(items, context)
        assert "Hybrid" in scores[0].reason
        assert "BM25=" in scores[0].reason


class TestCreateScorer:
    """Tests for the create_scorer factory function."""

    def test_create_bm25(self):
        """Create BM25 scorer."""
        scorer = create_scorer("bm25")
        assert isinstance(scorer, BM25Scorer)

    def test_create_bm25_case_insensitive(self):
        """Tier names are case insensitive."""
        scorer = create_scorer("BM25")
        assert isinstance(scorer, BM25Scorer)

    def test_create_hybrid(self):
        """Create hybrid scorer."""
        scorer = create_scorer("hybrid")
        assert isinstance(scorer, HybridScorer)

    def test_create_embedding_requires_deps(self):
        """Embedding scorer requires sentence-transformers."""
        if EmbeddingScorer.is_available():
            scorer = create_scorer("embedding")
            assert isinstance(scorer, EmbeddingScorer)
        else:
            with pytest.raises(RuntimeError, match="sentence-transformers"):
                create_scorer("embedding")

    def test_invalid_tier(self):
        """Invalid tier raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scorer tier"):
            create_scorer("invalid")

    def test_pass_kwargs(self):
        """Kwargs are passed to scorer constructor."""
        scorer = create_scorer("bm25", k1=2.0, b=0.5)
        assert scorer.k1 == 2.0
        assert scorer.b == 0.5


class TestEmbeddingAvailable:
    """Tests for embedding_available helper."""

    def test_returns_bool(self):
        """Returns boolean."""
        result = embedding_available()
        assert isinstance(result, bool)

    def test_matches_class_method(self):
        """Matches EmbeddingScorer.is_available()."""
        assert embedding_available() == EmbeddingScorer.is_available()


class TestSmartCrusherIntegration:
    """Integration tests for SmartCrusher with RelevanceScorer."""

    def test_context_extraction(self):
        """Context is extracted from messages."""
        from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

        crusher = SmartCrusher(config=SmartCrusherConfig())

        messages = [
            {"role": "user", "content": "Find user with ID 12345"},
            {"role": "assistant", "content": "I'll search for that user."},
        ]

        context = crusher._extract_context_from_messages(messages)
        assert "12345" in context
        assert "Find user" in context

    def test_context_from_anthropic_style_messages(self):
        """Context extracted from Anthropic-style messages."""
        from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

        crusher = SmartCrusher(config=SmartCrusherConfig())

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Search for Alice's records"},
                ],
            },
        ]

        context = crusher._extract_context_from_messages(messages)
        assert "Alice" in context

    def test_context_from_tool_calls(self):
        """Context extracted from tool call arguments."""
        from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

        crusher = SmartCrusher(config=SmartCrusherConfig())

        messages = [
            {"role": "user", "content": "Get user info"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_user",
                            "arguments": '{"user_id": "550e8400-e29b-41d4"}',
                        },
                    },
                ],
            },
        ]

        context = crusher._extract_context_from_messages(messages)
        assert "550e8400-e29b-41d4" in context
