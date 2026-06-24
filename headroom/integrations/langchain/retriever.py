"""Retriever integration for LangChain with intelligent document compression.

This module provides HeadroomDocumentCompressor, a LangChain BaseDocumentCompressor
that reduces retrieved documents based on relevance scoring while preserving
the most important information.

Example:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain_community.vectorstores import Chroma
    from headroom.integrations import HeadroomDocumentCompressor

    # Create vector store retriever
    vectorstore = Chroma.from_documents(documents, embeddings)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 50})

    # Wrap with Headroom compression
    compressor = HeadroomDocumentCompressor(max_documents=10)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    # Retrieve - automatically keeps most relevant documents
    docs = retriever.invoke("What is the capital of France?")
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from langchain_core.documents import Document

    LANGCHAIN_AVAILABLE = True

    class BaseDocumentCompressor:
        """Type-checking stub for LangChain's document compressor base."""

        def compress_documents(
            self, documents: Sequence[Any], query: str, callbacks: Any = None
        ) -> Sequence[Any]:
            raise NotImplementedError

# LangChain imports - these are optional dependencies
else:
    try:
        from langchain_core.callbacks import Callbacks
        from langchain_core.documents import Document

        # BaseDocumentCompressor location varies by langchain version
        try:
            from langchain.retrievers.document_compressors import BaseDocumentCompressor
        except ImportError:
            try:
                from langchain_core.documents.compressors import BaseDocumentCompressor
            except ImportError:
                # Fallback: create a minimal base class
                class BaseDocumentCompressor:
                    """Minimal base class for document compression."""

                    def compress_documents(
                        self, documents: Sequence[Any], query: str, callbacks: Any = None
                    ) -> Sequence[Any]:
                        raise NotImplementedError

        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        BaseDocumentCompressor = object  # type: ignore[misc,assignment]
        Document = object  # type: ignore[misc,assignment]
        Callbacks = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


def _check_langchain_available() -> None:
    """Raise ImportError if LangChain is not installed."""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is required for this integration. "
            "Install with: pip install headroom[langchain] "
            "or: pip install langchain-core"
        )


@dataclass
class CompressionMetrics:
    """Metrics from document compression."""

    documents_before: int
    documents_after: int
    documents_removed: int
    relevance_scores: list[float]


class HeadroomDocumentCompressor(BaseDocumentCompressor):
    """Compresses retrieved documents based on relevance to query.

    Uses BM25-style relevance scoring to keep only the most relevant
    documents from a larger retrieval set. This allows you to retrieve
    many documents initially (for recall) and then compress down to
    the most relevant ones (for precision).

    Works with LangChain's ContextualCompressionRetriever pattern.

    Example:
        from langchain.retrievers import ContextualCompressionRetriever
        from headroom.integrations import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor(
            max_documents=10,
            min_relevance=0.3,
        )

        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,  # Any retriever
        )

        # Retrieves top 10 most relevant docs
        docs = retriever.invoke("What is Python?")

    Attributes:
        max_documents: Maximum documents to return
        min_relevance: Minimum relevance score (0-1) to include
        prefer_diverse: Whether to prefer diverse results
    """

    max_documents: int = 10
    min_relevance: float = 0.0
    prefer_diverse: bool = False

    def __init__(
        self,
        max_documents: int = 10,
        min_relevance: float = 0.0,
        prefer_diverse: bool = False,
        **kwargs: Any,
    ):
        """Initialize HeadroomDocumentCompressor.

        Args:
            max_documents: Maximum number of documents to return. Default 10.
            min_relevance: Minimum relevance score (0-1) for a document to
                be included. Default 0.0 (no minimum).
            prefer_diverse: If True, use MMR-style selection to prefer
                diverse results over pure relevance. Default False.
            **kwargs: Additional arguments for BaseDocumentCompressor.
        """
        _check_langchain_available()

        super().__init__(**kwargs)
        self.max_documents = max_documents
        self.min_relevance = min_relevance
        self.prefer_diverse = prefer_diverse
        self._last_metrics: CompressionMetrics | None = None

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks = None,
    ) -> Sequence[Document]:
        """Compress documents based on relevance to query.

        Args:
            documents: Documents to compress.
            query: Query to score relevance against.
            callbacks: LangChain callbacks (unused).

        Returns:
            Compressed list of most relevant documents.
        """
        if not documents:
            self._last_metrics = CompressionMetrics(
                documents_before=0,
                documents_after=0,
                documents_removed=0,
                relevance_scores=[],
            )
            return []

        if len(documents) <= self.max_documents:
            # No compression needed
            scores = [self._score_document(doc, query) for doc in documents]
            self._last_metrics = CompressionMetrics(
                documents_before=len(documents),
                documents_after=len(documents),
                documents_removed=0,
                relevance_scores=scores,
            )
            return list(documents)

        # Score all documents
        scored = [(doc, self._score_document(doc, query)) for doc in documents]

        if self.prefer_diverse:
            # Use MMR-style selection for diversity
            selected = self._select_diverse(scored, query)
        else:
            # Sort by relevance score
            scored.sort(key=lambda x: x[1], reverse=True)
            selected = scored[: self.max_documents]

        # Filter by minimum relevance
        if self.min_relevance > 0:
            selected = [(doc, score) for doc, score in selected if score >= self.min_relevance]

        # Track metrics
        final_docs = [doc for doc, _ in selected]
        final_scores = [score for _, score in selected]

        self._last_metrics = CompressionMetrics(
            documents_before=len(documents),
            documents_after=len(final_docs),
            documents_removed=len(documents) - len(final_docs),
            relevance_scores=final_scores,
        )

        logger.info(
            f"HeadroomDocumentCompressor: {len(documents)} -> {len(final_docs)} documents "
            f"(avg relevance: {sum(final_scores) / len(final_scores) if final_scores else 0:.2f})"
        )

        return final_docs

    def _score_document(self, doc: Document, query: str) -> float:
        """Score a document's relevance to the query using BM25-style scoring.

        Args:
            doc: Document to score.
            query: Query to compare against.

        Returns:
            Relevance score between 0 and 1.
        """
        content = doc.page_content.lower()
        query_lower = query.lower()

        # Tokenize
        query_terms = self._tokenize(query_lower)
        doc_terms = self._tokenize(content)

        if not query_terms or not doc_terms:
            return 0.0

        # BM25-style scoring
        k1 = 1.5
        b = 0.75
        avg_dl = 100  # Assume average document length

        doc_len = len(doc_terms)
        term_freqs: dict[str, int] = {}
        for term in doc_terms:
            term_freqs[term] = term_freqs.get(term, 0) + 1

        score = 0.0
        for term in query_terms:
            if term in term_freqs:
                tf = term_freqs[term]
                # Simplified BM25 (without IDF since we don't have corpus stats)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / avg_dl))
                score += numerator / denominator

        # Normalize to 0-1 range
        max_possible = len(query_terms) * (k1 + 1)
        normalized = score / max_possible if max_possible > 0 else 0.0

        # Boost for exact phrase matches
        if query_lower in content:
            normalized = min(1.0, normalized + 0.3)

        return min(1.0, normalized)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into terms.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens.
        """
        # Simple tokenization: split on non-alphanumeric, filter short terms
        tokens = re.findall(r"\b\w+\b", text)
        return [t for t in tokens if len(t) > 1]

    def _select_diverse(
        self, scored_docs: list[tuple[Document, float]], query: str
    ) -> list[tuple[Document, float]]:
        """Select diverse documents using MMR-style approach.

        Balances relevance with diversity to avoid redundant results.

        Args:
            scored_docs: List of (document, relevance_score) tuples.
            query: Original query.

        Returns:
            Selected documents with diversity considered.
        """
        if not scored_docs:
            return []

        # Sort by initial relevance
        scored_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

        # Start with most relevant
        selected = [scored_docs[0]]
        remaining = scored_docs[1:]

        lambda_param = 0.5  # Balance between relevance and diversity

        while len(selected) < self.max_documents and remaining:
            best_score = -1.0
            best_idx = 0

            for i, (doc, rel_score) in enumerate(remaining):
                # Calculate max similarity to already selected docs
                max_sim = max(self._document_similarity(doc, sel_doc) for sel_doc, _ in selected)

                # MMR score: lambda * relevance - (1-lambda) * max_similarity
                mmr_score = lambda_param * rel_score - (1 - lambda_param) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(remaining[best_idx])
            remaining.pop(best_idx)

        return selected

    def _document_similarity(self, doc1: Document, doc2: Document) -> float:
        """Calculate similarity between two documents.

        Uses Jaccard similarity on terms for simplicity.

        Args:
            doc1: First document.
            doc2: Second document.

        Returns:
            Similarity score between 0 and 1.
        """
        terms1 = set(self._tokenize(doc1.page_content.lower()))
        terms2 = set(self._tokenize(doc2.page_content.lower()))

        if not terms1 or not terms2:
            return 0.0

        intersection = len(terms1 & terms2)
        union = len(terms1 | terms2)

        return intersection / union if union > 0 else 0.0

    @property
    def last_metrics(self) -> CompressionMetrics | None:
        """Get metrics from the last compression operation."""
        return self._last_metrics

    def get_compression_stats(self) -> dict[str, Any]:
        """Get statistics from the last compression.

        Returns:
            Dictionary with compression metrics, or empty if no compression yet.
        """
        if self._last_metrics is None:
            return {}

        return {
            "documents_before": self._last_metrics.documents_before,
            "documents_after": self._last_metrics.documents_after,
            "documents_removed": self._last_metrics.documents_removed,
            "average_relevance": (
                sum(self._last_metrics.relevance_scores) / len(self._last_metrics.relevance_scores)
                if self._last_metrics.relevance_scores
                else 0.0
            ),
        }
