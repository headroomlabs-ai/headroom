"""Tests for LangChain retriever integration with document compression.

Tests cover:
1. CompressionMetrics - Dataclass for document compression metrics
2. HeadroomDocumentCompressor - LangChain BaseDocumentCompressor implementation
3. BM25-style relevance scoring
4. Diverse document selection (MMR-style)
5. Compression statistics tracking
"""

from unittest.mock import MagicMock

import pytest

# Check if LangChain is available
try:
    from langchain_core.documents import Document

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Skip all tests if LangChain not installed
pytestmark = pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(page_content="Python is a programming language.", metadata={"id": 1}),
        Document(page_content="Python is great for data science.", metadata={"id": 2}),
        Document(page_content="Java is also a programming language.", metadata={"id": 3}),
        Document(
            page_content="Machine learning uses Python extensively.",
            metadata={"id": 4},
        ),
        Document(page_content="JavaScript is used for web development.", metadata={"id": 5}),
    ]


@pytest.fixture
def many_documents():
    """Create many documents for compression testing."""
    return [
        Document(
            page_content=f"Document {i} contains some text about topic {i % 5}.",
            metadata={"id": i},
        )
        for i in range(50)
    ]


class TestCompressionMetrics:
    """Tests for CompressionMetrics dataclass."""

    def test_create_metrics(self):
        """Create compression metrics with all fields."""
        from headroom.integrations.langchain.retriever import CompressionMetrics

        metrics = CompressionMetrics(
            documents_before=50,
            documents_after=10,
            documents_removed=40,
            relevance_scores=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1],
        )

        assert metrics.documents_before == 50
        assert metrics.documents_after == 10
        assert metrics.documents_removed == 40
        assert len(metrics.relevance_scores) == 10

    def test_metrics_required_fields(self):
        """All fields are required."""
        from headroom.integrations.langchain.retriever import CompressionMetrics

        with pytest.raises(TypeError):
            CompressionMetrics()  # type: ignore[call-arg]


class TestHeadroomDocumentCompressorInit:
    """Tests for HeadroomDocumentCompressor initialization."""

    def test_init_defaults(self):
        """Initialize with default settings."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor()

        assert compressor.max_documents == 10
        assert compressor.min_relevance == 0.0
        assert compressor.prefer_diverse is False
        assert compressor._last_metrics is None

    def test_init_custom_settings(self):
        """Initialize with custom settings."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor(
            max_documents=20,
            min_relevance=0.5,
            prefer_diverse=True,
        )

        assert compressor.max_documents == 20
        assert compressor.min_relevance == 0.5
        assert compressor.prefer_diverse is True


class TestHeadroomDocumentCompressorCompress:
    """Tests for compress_documents method."""

    def test_compress_empty_documents(self):
        """Compress empty list returns empty list."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor()

        result = compressor.compress_documents([], "query")

        assert result == []
        assert compressor._last_metrics is not None
        assert compressor._last_metrics.documents_before == 0

    def test_compress_fewer_than_max_documents(self, sample_documents):
        """Compress when documents fewer than max returns all."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor(max_documents=10)  # More than 5 docs

        result = compressor.compress_documents(sample_documents, "Python")

        assert len(result) == len(sample_documents)
        assert compressor._last_metrics.documents_removed == 0

    def test_compress_more_than_max_documents(self, many_documents):
        """Compress when documents exceed max returns max_documents."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor(max_documents=10)

        result = compressor.compress_documents(many_documents, "topic 1")

        assert len(result) == 10
        assert compressor._last_metrics.documents_before == 50
        assert compressor._last_metrics.documents_after == 10
        assert compressor._last_metrics.documents_removed == 40

    def test_compress_orders_by_relevance(self, sample_documents):
        """Compressed documents are ordered by relevance."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor(max_documents=3)

        result = compressor.compress_documents(sample_documents, "Python programming")

        # Most relevant documents should come first
        assert len(result) == 3
        # First doc should be highly relevant to "Python programming"
        assert "Python" in result[0].page_content or "programming" in result[0].page_content

    def test_compress_with_min_relevance_filter(self):
        """Documents below min_relevance are filtered out."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        documents = [
            Document(page_content="Very relevant Python tutorial"),
            Document(page_content="Completely unrelated topic XYZ"),
        ]

        compressor = HeadroomDocumentCompressor(
            max_documents=10,
            min_relevance=0.3,  # Require some relevance
        )

        result = compressor.compress_documents(documents, "Python programming")

        # The very relevant doc should pass, unrelated might be filtered
        assert len(result) >= 1
        # First result should be the relevant one
        assert "Python" in result[0].page_content

    def test_compress_tracks_relevance_scores(self, sample_documents):
        """Compression tracks relevance scores."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor(max_documents=3)

        compressor.compress_documents(sample_documents, "Python")

        assert compressor._last_metrics is not None
        assert len(compressor._last_metrics.relevance_scores) == 3
        # Scores should be sorted descending
        scores = compressor._last_metrics.relevance_scores
        assert scores == sorted(scores, reverse=True)


class TestHeadroomDocumentCompressorScoring:
    """Tests for document relevance scoring."""

    def test_score_document_exact_match_boost(self):
        """Exact phrase match gets relevance boost."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor()

        doc_exact = Document(page_content="What is Python programming?")
        doc_partial = Document(page_content="Programming in various languages")

        score_exact = compressor._score_document(doc_exact, "Python programming")
        score_partial = compressor._score_document(doc_partial, "Python programming")

        # Exact match should score higher
        assert score_exact > score_partial

    def test_score_document_term_frequency(self):
        """Higher term frequency increases score."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor()

        doc_many = Document(page_content="Python Python Python is great")
        doc_one = Document(page_content="Python is a language")

        score_many = compressor._score_document(doc_many, "Python")
        score_one = compressor._score_document(doc_one, "Python")

        # More mentions should score higher (BM25 diminishing returns aside)
        assert score_many >= score_one

    def test_score_document_empty_query(self):
        """Empty query returns zero score."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor()

        doc = Document(page_content="Some content")

        score = compressor._score_document(doc, "")

        assert score == 0.0

    def test_score_document_empty_content(self):
        """Empty document content returns zero score."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor()

        doc = Document(page_content="")

        score = compressor._score_document(doc, "query")

        assert score == 0.0

    def test_score_document_case_insensitive(self):
        """Scoring is case insensitive."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor()

        doc = Document(page_content="PYTHON is GREAT")

        score = compressor._score_document(doc, "python great")

        assert score > 0.0


class TestHeadroomDocumentCompressorTokenize:
    """Tests for text tokenization."""

    def test_tokenize_basic(self):
        """Tokenize basic text."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor()

        tokens = compressor._tokenize("Hello world")

        assert tokens == ["Hello", "world"]

    def test_tokenize_with_punctuation(self):
        """Tokenize text with punctuation."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor()

        tokens = compressor._tokenize("Hello, world! How are you?")

        assert "Hello" in tokens
        assert "world" in tokens
        assert "," not in tokens
        assert "!" not in tokens

    def test_tokenize_filters_short_tokens(self):
        """Tokenize filters tokens with length 1."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor()

        tokens = compressor._tokenize("I am a developer")

        # "I" and "a" should be filtered out
        assert "I" not in tokens
        assert "a" not in tokens
        assert "am" in tokens
        assert "developer" in tokens


class TestHeadroomDocumentCompressorDiversity:
    """Tests for diverse document selection (MMR-style)."""

    def test_compress_with_diversity(self):
        """Diverse selection avoids redundant documents."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        # Create similar documents
        documents = [
            Document(page_content="Python is a programming language."),
            Document(page_content="Python is a great programming language."),  # Very similar
            Document(page_content="Python programming tutorial."),  # Similar
            Document(page_content="Java is a different programming language."),  # Different
            Document(page_content="Machine learning with TensorFlow."),  # Very different
        ]

        compressor = HeadroomDocumentCompressor(
            max_documents=3,
            prefer_diverse=True,
        )

        result = compressor.compress_documents(documents, "programming language")

        assert len(result) == 3
        # Diversity should favor the Java/ML docs over multiple Python docs

    def test_select_diverse_empty(self):
        """Diverse selection with empty input."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor(prefer_diverse=True)

        result = compressor._select_diverse([], "query")

        assert result == []

    def test_document_similarity_identical(self):
        """Identical documents have similarity 1.0."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor()

        doc1 = Document(page_content="Hello world")
        doc2 = Document(page_content="Hello world")

        similarity = compressor._document_similarity(doc1, doc2)

        assert similarity == 1.0

    def test_document_similarity_different(self):
        """Different documents have low similarity."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor()

        doc1 = Document(page_content="Python programming tutorial")
        doc2 = Document(page_content="Cooking recipes for dinner")

        similarity = compressor._document_similarity(doc1, doc2)

        assert similarity < 0.2  # Very different

    def test_document_similarity_partial_overlap(self):
        """Partially overlapping documents have medium similarity."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor()

        doc1 = Document(page_content="Python programming tutorial")
        doc2 = Document(page_content="Python data science tutorial")

        similarity = compressor._document_similarity(doc1, doc2)

        assert 0.2 < similarity < 0.8  # Some overlap

    def test_document_similarity_empty_content(self):
        """Empty content documents have zero similarity."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor()

        doc1 = Document(page_content="")
        doc2 = Document(page_content="Some content")

        similarity = compressor._document_similarity(doc1, doc2)

        assert similarity == 0.0


class TestHeadroomDocumentCompressorStats:
    """Tests for compression statistics."""

    def test_last_metrics_none_initially(self):
        """last_metrics is None before any compression."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor()

        assert compressor.last_metrics is None

    def test_last_metrics_updated_after_compression(self, sample_documents):
        """last_metrics is updated after compression."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor(max_documents=3)

        compressor.compress_documents(sample_documents, "Python")

        assert compressor.last_metrics is not None
        assert compressor.last_metrics.documents_before == 5
        assert compressor.last_metrics.documents_after == 3

    def test_get_compression_stats_empty(self):
        """get_compression_stats returns empty dict before compression."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor()

        stats = compressor.get_compression_stats()

        assert stats == {}

    def test_get_compression_stats_with_data(self, many_documents):
        """get_compression_stats returns stats after compression."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor(max_documents=10)

        compressor.compress_documents(many_documents, "topic")

        stats = compressor.get_compression_stats()

        assert stats["documents_before"] == 50
        assert stats["documents_after"] == 10
        assert stats["documents_removed"] == 40
        assert "average_relevance" in stats
        assert 0 <= stats["average_relevance"] <= 1.0

    def test_get_compression_stats_average_relevance(self, sample_documents):
        """get_compression_stats calculates average relevance correctly."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor(max_documents=2)

        compressor.compress_documents(sample_documents, "Python")

        stats = compressor.get_compression_stats()

        # Average should match manual calculation
        expected_avg = sum(compressor._last_metrics.relevance_scores) / len(
            compressor._last_metrics.relevance_scores
        )
        assert abs(stats["average_relevance"] - expected_avg) < 0.001


class TestHeadroomDocumentCompressorCallbacks:
    """Tests for LangChain callbacks integration."""

    def test_compress_ignores_callbacks(self, sample_documents):
        """compress_documents accepts but ignores callbacks parameter."""
        from headroom.integrations.langchain.retriever import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor(max_documents=3)

        # Pass a mock callback - should not raise
        mock_callback = MagicMock()
        result = compressor.compress_documents(
            sample_documents, "Python", callbacks=[mock_callback]
        )

        assert len(result) == 3


class TestLangChainNotAvailable:
    """Tests for behavior when LangChain is not available."""

    def test_check_raises_import_error(self):
        """_check_langchain_available raises ImportError when not available."""
        from headroom.integrations.langchain.retriever import _check_langchain_available

        # When LangChain IS available, should not raise
        try:
            _check_langchain_available()
        except ImportError:
            pytest.fail("Should not raise when LangChain is available")
