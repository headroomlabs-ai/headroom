"""Semantic preservation scoring for Headroom compression.

Computes how well compression preserved the meaning of the original text
by comparing embedding similarity between input and output.

Uses fastembed (already a Headroom dependency) for lightweight ONNX-backed
embeddings — no additional heavy dependencies required.

Example::

    from headroom.transforms.semantic_scorer import score_semantic_preservation

    score = score_semantic_preservation("original text here", "compressed text here")
    # score: 0.85 (meaning well preserved)

    # Or with messages:
    from headroom.transforms.semantic_scorer import score_messages_semantic_preservation
    score = score_messages_semantic_preservation(original_messages, compressed_messages)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-loaded singleton — avoids re-initializing the model on every call
_model = None
_model_lock = None  # initialized lazily to avoid import-time threading overhead


def _get_model():
    """Get or create the fastembed TextEmbedding model (singleton).

    Returns None if fastembed is not installed — callers should handle
    this gracefully.
    """
    global _model, _model_lock

    if _model is not None:
        return _model

    try:
        import threading

        if _model_lock is None:
            _model_lock = threading.Lock()

        with _model_lock:
            if _model is not None:
                return _model

            from fastembed import TextEmbedding

            # Reuse the same default model as the relevance scorer
            # (BAAI/bge-small-en-v1.5 — 33M params, 384 dims, ~30 MB)
            from headroom.relevance.embedding import DEFAULT_MODEL_NAME, _pinned_revision

            revision = _pinned_revision(DEFAULT_MODEL_NAME)
            kwargs: dict[str, Any] = {}
            if revision is not None:
                kwargs["revision"] = revision

            _model = TextEmbedding(model_name=DEFAULT_MODEL_NAME, **kwargs)
            logger.debug("Semantic scorer model loaded: %s", DEFAULT_MODEL_NAME)
            return _model
    except ImportError:
        logger.debug(
            "fastembed not installed; semantic preservation scoring unavailable. "
            "Install with: pip install headroom[relevance]"
        )
        return None
    except Exception as exc:
        logger.debug("Failed to load semantic scorer model: %s", exc)
        return None


def is_available() -> bool:
    """Check if semantic scoring is available.

    Returns True if fastembed is installed and the model can be loaded.
    """
    return _get_model() is not None


def _cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector (numpy array).
        b: Second vector (numpy array).

    Returns:
        Cosine similarity in range [0.0, 1.0].
    """
    try:
        import numpy as np

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        similarity = float(np.dot(a, b) / (norm_a * norm_b))
        return max(0.0, min(1.0, similarity))
    except ImportError:
        # Fallback: manual dot product without numpy (slower but works)
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        similarity = dot / (norm_a * norm_b)
        return max(0.0, min(1.0, similarity))


def _extract_text(messages: list[dict[str, Any]]) -> str:
    """Extract concatenated text content from a message list.

    Handles both string content and list-of-parts content (Anthropic format).
    Truncates to a reasonable length to keep embedding fast.
    """
    parts: list[str] = []
    total_len = 0
    max_len = 50_000  # Cap to keep embedding fast; covers the semantic core

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
            total_len += len(content)
        elif isinstance(content, list):
            # Anthropic-style: list of {"type": "text", "text": "..."}
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    parts.append(text)
                    total_len += len(text)
                elif isinstance(block, str):
                    parts.append(block)
                    total_len += len(block)
        if total_len >= max_len:
            break

    combined = "\n".join(parts)
    if len(combined) > max_len:
        combined = combined[:max_len]
    return combined


def score_texts(
    original: str,
    compressed: str,
) -> float | None:
    """Score semantic preservation between original and compressed text.

    Args:
        original: The original text before compression.
        compressed: The compressed text after compression.

    Returns:
        Score from 0.0 (completely different meaning) to 1.0 (identical meaning),
        or None if scoring is unavailable (fastembed not installed, etc.).
    """
    if not original.strip() or not compressed.strip():
        return None

    model = _get_model()
    if model is None:
        return None

    try:
        embeddings = list(model.embed([original, compressed]))
        score = _cosine_similarity(embeddings[0], embeddings[1])
        return score
    except Exception as exc:
        logger.debug("Semantic scoring failed: %s", exc)
        return None


def score_messages(
    original_messages: list[dict[str, Any]],
    compressed_messages: list[dict[str, Any]],
) -> float | None:
    """Score semantic preservation between original and compressed messages.

    Extracts text from both message lists and computes embedding similarity.
    Designed to be lightweight — fastembed's ONNX inference is ~2-3x faster
    than sentence-transformers.

    Args:
        original_messages: Messages before compression.
        compressed_messages: Messages after compression.

    Returns:
        Score from 0.0 (completely different) to 1.0 (identical meaning),
        or None if scoring is unavailable.
    """
    original_text = _extract_text(original_messages)
    compressed_text = _extract_text(compressed_messages)

    if not original_text.strip() or not compressed_text.strip():
        return None

    return score_texts(original_text, compressed_text)
