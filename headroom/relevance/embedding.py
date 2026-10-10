"""Embedding-based relevance scorer for Headroom SDK.

This module provides semantic relevance scoring using `fastembed`
(BAAI/bge-small-en-v1.5 by default — 33M params, 384 dims, ~67 MB
quantized ONNX). The Rust scorer (fastembed-rs crate) loads the same
pinned Qdrant/bge-small-en-v1.5-onnx-Q snapshot, so both languages run
the same ONNX file and tokenizer. On the same ONNX Runtime build the two
agree within 1e-6 per component; across ORT versions they drift by up to
~4e-4, so they are close, not byte-equal. The gated parity fixture in
tests/parity/fixtures/embedding/ bounds that drift.

Key features:
- Semantic understanding ("errors" matches "failed", "issues")
- Handles paraphrases and synonyms
- ONNX-backed inference (no PyTorch / no CUDA required)
- ~2-3x faster than sentence-transformers' all-MiniLM-L6-v2
- Outranks all-MiniLM-L6-v2 by ~6 MTEB points

Install with: pip install headroom[relevance]

History: this module previously wrapped `sentence-transformers`
(PyTorch). Switched to fastembed in Stage 3c.1 of the Rust port to:
1. Run the same ONNX model as the Rust embedding scorer (both call
   into ONNX Runtime over the identical pinned ONNX file).
2. Remove the torch dependency from the relevance/ path
   (Phase 6: "drop torch from Python").
3. Get a better default model (bge-small-en-v1.5 vs MiniLM-L6-v2).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from headroom.offline import OfflineEgressBlocked, guard_egress
from headroom.onnx_runtime import hf_hub_download_local_first

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


@contextmanager
def _hf_hub_forced_offline() -> Iterator[None]:
    """Force the HuggingFace stack offline for the duration of the block.

    huggingface_hub reads HF_HUB_OFFLINE once, at import, into
    ``constants.HF_HUB_OFFLINE``; if it was imported before this call the env
    var alone changes nothing. Force both, and restore both on exit.
    """
    from huggingface_hub import constants as hf_constants

    previous = os.environ.get("HF_HUB_OFFLINE")
    previous_constant = hf_constants.HF_HUB_OFFLINE
    os.environ["HF_HUB_OFFLINE"] = "1"
    hf_constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        hf_constants.HF_HUB_OFFLINE = previous_constant
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous


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

    try:
        with _hf_hub_forced_offline():
            return TextEmbedding(**kwargs)
    except Exception as cache_miss:  # noqa: BLE001 - any local-lookup failure
        logger.debug("fastembed cache lookup failed, falling back to network: %s", cache_miss)

    guard_egress(
        f"fastembed weight download for {kwargs['model_name']}",
        "huggingface.co",
    )
    return TextEmbedding(**kwargs)


# Default model name. Same model the Rust embedding scorer loads.
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# The HF repo fastembed resolves DEFAULT_MODEL_NAME to, and every file a
# fastembed load of it reads. Its revision is pinned in
# ``onnx_runtime._PINNED_REVISIONS`` (HEADROOM_HF_PIN=off floats it). The Rust
# scorer loads the same snapshot: keep both equal to DEFAULT_MODEL_REPO and
# DEFAULT_MODEL_FILES in crates/headroom-core/src/relevance/embedding.rs.
DEFAULT_MODEL_REPO = "Qdrant/bge-small-en-v1.5-onnx-Q"
DEFAULT_MODEL_FILES = (
    "model_optimized.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
)


def _resolve_default_model_snapshot() -> str:
    """Return the local directory holding the pinned default-model snapshot.

    fastembed cannot pin this itself: ``TextEmbedding`` accepts ``revision``
    but drops it before ``snapshot_download`` (fastembed 0.9.0), so passing it
    left the download floating on the repo's ``main``. Each file is resolved
    here through ``hf_hub_download_local_first`` instead, which applies the
    pinned SHA, tries the cache first and guards the network fallback.

    The files land in the standard HuggingFace cache, which the Rust scorer
    searches too, rather than fastembed's default under the system temp dir,
    which macOS purges after a few idle days.

    Raises:
        OfflineEgressBlocked: ``HEADROOM_OFFLINE`` is set and a file is not cached.
        RuntimeError: the files resolved to more than one snapshot, which only
            happens when ``main`` moved mid-download with pinning turned off.
    """
    paths = [
        Path(hf_hub_download_local_first(DEFAULT_MODEL_REPO, name)) for name in DEFAULT_MODEL_FILES
    ]
    snapshot_dirs = {path.parent for path in paths}
    if len(snapshot_dirs) != 1:
        raise RuntimeError(
            f"{DEFAULT_MODEL_REPO} files resolved to more than one snapshot: "
            f"{sorted(str(d) for d in snapshot_dirs)}"
        )
    return str(snapshot_dirs.pop())


def _load_default_model() -> TextEmbedding:
    """Load DEFAULT_MODEL_NAME from its pinned snapshot, with no fastembed lookup.

    Given ``specific_model_path`` fastembed loads that directory without
    touching the Hub; the construction still runs with the Hub forced offline
    so that a fastembed that stopped honouring it fails here instead of
    downloading an unpinned copy.
    """
    from fastembed import TextEmbedding

    snapshot = _resolve_default_model_snapshot()
    with _hf_hub_forced_offline():
        return TextEmbedding(model_name=DEFAULT_MODEL_NAME, specific_model_path=snapshot)


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
    Auto-downloads its pinned snapshot from HuggingFace Hub on first
    use (~67 MB quantized ONNX).

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
            try:
                if self.model_name == DEFAULT_MODEL_NAME:
                    self._model = _load_default_model()
                else:
                    # Other catalog models are not pinned: fastembed resolves
                    # them on the repo's main, cache first.
                    self._model = _load_text_embedding({"model_name": self.model_name})
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
