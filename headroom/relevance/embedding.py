"""Embedding-based relevance scorer for Headroom SDK.

This module provides semantic relevance scoring using `fastembed`
(BAAI/bge-small-en-v1.5 by default — 33M params, 384 dims, ~30 MB
int8-quantized ONNX). Same library + same model used by the Rust
SmartCrusher (fastembed-rs crate) so embeddings agree byte-for-byte
across the language boundary.

Key features:
- Semantic understanding ("errors" matches "failed", "issues")
- Handles paraphrases and synonyms
- ONNX-backed inference (no PyTorch / no CUDA required)
- ~2-3x faster than sentence-transformers' all-MiniLM-L6-v2
- Outranks all-MiniLM-L6-v2 by ~6 MTEB points

Install with: pip install headroom[relevance]

History: this module previously wrapped `sentence-transformers`
(PyTorch). Switched to fastembed in Stage 3c.1 of the Rust port to:
1. Match the Rust embedding scorer byte-for-byte (both call into
   ONNX Runtime over the identical ONNX file).
2. Remove the torch dependency from the relevance/ path
   (Phase 6: "drop torch from Python").
3. Get a better default model (bge-small-en-v1.5 vs MiniLM-L6-v2).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from headroom.offline import OfflineEgressBlocked, guard_egress

from .base import RelevanceScore, RelevanceScorer

# numpy is an optional dependency - import lazily
_numpy = None


def _get_numpy():
    """Lazily import numpy."""
    global _numpy
    if _numpy is None:
        try:
            import numpy as np

            _numpy = np
        except ImportError as e:
            raise ImportError(
                "numpy is required for EmbeddingScorer. "
                "Install with: pip install headroom[relevance]"
            ) from e
    return _numpy


if TYPE_CHECKING:
    from fastembed import TextEmbedding

logger = logging.getLogger(__name__)


def _load_text_embedding(kwargs: dict[str, str]) -> TextEmbedding:
    """Build a fastembed ``TextEmbedding``, cache first and network second.

    ``onnx_runtime.hf_hub_download_local_first`` can hang the air-gap guard on
    an explicit ``local_files_only=True`` attempt because ``hf_hub_download``
    takes that flag. fastembed's constructor does not, so the same cache-hit /
    network-fallback split has to be made here: try the load with the whole
    HuggingFace stack forced offline, and only reach for the guard when that
    fails, which is exactly the case where a socket would have been opened.

    This keeps the behaviour an air-gapped deployment actually wants — a
    pre-seeded cache still loads under ``HEADROOM_OFFLINE`` — while making a
    cold cache refuse instead of quietly dialling huggingface.co. A plain
    unconditional guard would have broken the pre-seeded case, which is the
    reason this path was left unguarded before.

    The ``HF_HUB_OFFLINE`` flip is process-global for the duration of the
    call. That is acceptable here because loading is one-shot and memoised by
    the caller, and because the value it forces is *stricter* than whatever it
    replaces, so a concurrent HuggingFace call can only be refused locally,
    never sent somewhere it would not otherwise go.
    """
    from fastembed import TextEmbedding

    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        return TextEmbedding(**kwargs)
    except Exception as cache_miss:  # noqa: BLE001 - any local-lookup failure
        logger.debug("fastembed cache lookup failed, falling back to network: %s", cache_miss)
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous

    guard_egress(
        f"fastembed weight download for {kwargs['model_name']}",
        "huggingface.co",
    )
    return TextEmbedding(**kwargs)


# Default model name. Same string used by the Rust embedding scorer.
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Pinned revision of fastembed's underlying HF repo for the default model
# (fastembed resolves "BAAI/bge-small-en-v1.5" -> "qdrant/bge-small-en-v1.5-onnx-q").
# fastembed's TextEmbedding signature omits ``revision`` but forwards **kwargs to
# huggingface_hub.snapshot_download, so passing it pins the download for
# supply-chain integrity. Only the default model is pinned; custom models float.
# Set HEADROOM_HF_PIN=off to bypass (mirrors headroom.onnx_runtime pinning).
_DEFAULT_MODEL_PINNED_REVISION = "52398278842ec682c6f32300af41344b1c0b0bb2"


def _pinned_revision(model_name: str) -> str | None:
    """Return the pinned HF revision for ``model_name`` (default model only).

    Returns ``None`` for custom models or when ``HEADROOM_HF_PIN=off``.
    """
    if os.environ.get("HEADROOM_HF_PIN", "").strip().lower() in ("off", "0", "false", "no"):
        return None
    if model_name == DEFAULT_MODEL_NAME:
        return _DEFAULT_MODEL_PINNED_REVISION
    return None


def _cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector (numpy array).
        b: Second vector (numpy array).

    Returns:
        Cosine similarity in range [-1, 1], clamped to [0, 1].
    """
    np = _get_numpy()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = float(np.dot(a, b) / (norm_a * norm_b))
    # Clamp to [0, 1] since we only care about positive similarity
    return max(0.0, min(1.0, similarity))


class EmbeddingScorer(RelevanceScorer):
    """Semantic relevance scorer using fastembed (ONNX-backed).

    Default model: BAAI/bge-small-en-v1.5 (33M params, 384 dims).
    Auto-downloads from HuggingFace Hub on first use (~30 MB
    int8-quantized ONNX).

    Example:
        scorer = EmbeddingScorer()
        score = scorer.score(
            '{"status": "failed", "error": "connection refused"}',
            "show me the errors"
        )
        # score.score > 0.5 (semantic match between "failed"/"error" and "errors")

    Note:
        Requires fastembed: pip install headroom[relevance]
    """

    def __init__(
        self,
        model_name: str | None = None,
        cache_model: bool = True,  # Kept for API compatibility
    ):
        """Initialize embedding scorer.

        Args:
            model_name: Sentence-embedding model name. Default
                "BAAI/bge-small-en-v1.5". See fastembed's catalog for
                supported models (BGE, E5, MiniLM, jina, etc.).
            cache_model: Deprecated, models are always cached via
                fastembed's HF Hub cache.
        """
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.cache_model = cache_model
        self._model: TextEmbedding | None = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if fastembed is installed.

        Returns:
            True if the package is available.
        """
        try:
            import fastembed  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_model(self) -> TextEmbedding:
        """Get or load the fastembed text embedding model.

        Returns:
            Loaded TextEmbedding model.

        Raises:
            RuntimeError: If fastembed is not installed.
        """
        if not self.is_available():
            raise RuntimeError(
                "EmbeddingScorer requires fastembed. Install with: pip install headroom[relevance]"
            )

        if self._model is None:
            revision = _pinned_revision(self.model_name)
            kwargs = {"model_name": self.model_name}
            if revision is not None:
                # fastembed forwards **kwargs to snapshot_download(revision=...).
                kwargs["revision"] = revision
            try:
                self._model = _load_text_embedding(kwargs)
            except OfflineEgressBlocked as blocked:
                # Translate at the boundary that owns the degradation, the way
                # the Kompress and ONNX loaders do. Public model weights are
                # not data leaving the box, so what the caller needs to hear
                # is "this model is not available here and why", not an
                # air-gap type it has never seen.
                raise RuntimeError(
                    f"Embedding model {self.model_name!r} is unavailable: {blocked}. "
                    "Pre-seed the fastembed/HuggingFace cache on this host, or "
                    "use the BM25-only scorer."
                ) from None
        return self._model

    def _encode(self, texts: list[str]):
        """Encode texts to embeddings via fastembed.

        fastembed's `embed` returns an iterator yielding numpy arrays
        (one per text). We materialize to a list/np.array for the
        cosine-similarity downstream.

        Args:
            texts: List of texts to encode.

        Returns:
            numpy array of embeddings, shape (len(texts), embedding_dim).
        """
        np = _get_numpy()
        model = self._get_model()
        embeddings = list(model.embed(texts))
        return np.array(embeddings)

    def score(self, item: str, context: str) -> RelevanceScore:
        """Score item relevance to context using embeddings.

        Args:
            item: Item text.
            context: Query context.

        Returns:
            RelevanceScore with embedding-based similarity.
        """
        if not item or not context:
            return RelevanceScore(score=0.0, reason="Embedding: empty input")

        embeddings = self._encode([item, context])
        similarity = _cosine_similarity(embeddings[0], embeddings[1])

        return RelevanceScore(
            score=similarity,
            reason=f"Embedding: semantic similarity {similarity:.2f}",
        )

    def score_batch(self, items: list[str], context: str) -> list[RelevanceScore]:
        """Score multiple items efficiently using batch encoding.

        Encodes items + context in a single fastembed call. Mirrors the
        Rust scorer's batch behavior so both languages do the same
        amount of work for the same input.

        Args:
            items: List of items to score.
            context: Query context.

        Returns:
            List of RelevanceScore objects.
        """
        if not items:
            return []

        if not context:
            return [RelevanceScore(score=0.0, reason="Embedding: empty context") for _ in items]

        # Encode all texts in one batch
        all_texts = items + [context]
        embeddings = self._encode(all_texts)

        # Last embedding is the context
        context_emb = embeddings[-1]
        item_embs = embeddings[:-1]

        # Compute similarities
        results = []
        for emb in item_embs:
            similarity = _cosine_similarity(emb, context_emb)
            results.append(
                RelevanceScore(
                    score=similarity,
                    reason=f"Embedding: {similarity:.2f}",
                )
            )

        return results


# Convenience function for checking availability without instantiation
def embedding_available() -> bool:
    """Check if embedding scorer is available.

    Returns:
        True if fastembed is installed.
    """
    return EmbeddingScorer.is_available()
