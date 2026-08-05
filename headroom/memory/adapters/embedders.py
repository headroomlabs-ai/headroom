"""Embedder implementations for Headroom Memory.

Provides embedding generation via multiple backends:
- LocalEmbedder: sentence-transformers (local, no API needed)
- OpenAIEmbedder: OpenAI API (cloud, requires API key)
- OllamaEmbedder: Ollama API (local server)

All embedders return normalized float32 vectors for cosine similarity.
"""

from __future__ import annotations

import asyncio
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from headroom.models.config import ML_MODEL_DEFAULTS
from headroom.onnx_runtime import create_cpu_session_options, hf_hub_download_local_first

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Suppress HuggingFace Hub warnings about missing tokens and rate limits.
# These appear whenever hf_hub_download is called without HF_TOKEN set.
# We operate in an authenticated-optional mode; warnings are not actionable.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
# Also silence the huggingface_hub logger which emits rate-limit advisory messages.
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
# sentence_transformers uses httpx to check model file manifests on every startup.
# These HEAD/GET requests generate INFO lines per worker; suppress to WARNING.
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# =============================================================================
# Local (torch / sentence-transformers) CPU thread cap — issue #198
# =============================================================================
# A long-lived proxy serves many requests concurrently. Each torch ``encode()``
# fans out to BLAS (MKL / OpenBLAS / Accelerate) + OpenMP worker threads, which
# default to roughly ``os.cpu_count()``. Under concurrency this oversubscribes
# the CPU — N in-flight encodes x ~cpu_count threads each thrash the scheduler
# and starve the asyncio event loop, so liveness probes (``/livez``) spike to
# multiple seconds even though the loop itself is idle (issue #198).
#
# Capping intra-op parallelism makes a single encode modestly slower but lets
# concurrent encodes scale linearly without thread-pool thrash — the standard
# trade-off for serving torch models inside an async server. The ONNX embedder
# already caps its threads (see ``onnx_runtime.create_cpu_session_options``);
# this brings the torch path to parity.
#
# torch's OpenMP thread count is per-thread, and encodes run on executor worker
# threads, so a one-shot cap would miss most workers. Instead, CPU encodes run
# on a dedicated, size-limited executor whose ``initializer`` pins each worker's
# thread pool once. Total embedding threads are then bounded by
# ``workers (HEADROOM_EMBED_CONCURRENCY) x threads-per-encode
# (HEADROOM_EMBED_NUM_THREADS)``. Applies to the CPU device only (GPU/MPS do
# their compute off-CPU).
_EMBED_THREADS_ENV = "HEADROOM_EMBED_NUM_THREADS"
_DEFAULT_EMBED_THREADS = 1
_EMBED_CONCURRENCY_ENV = "HEADROOM_EMBED_CONCURRENCY"
_DEFAULT_EMBED_CONCURRENCY = 4
_BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _resolve_positive_int_env(env_var: str, default: int) -> int:
    """Read a positive integer from ``env_var``, falling back to ``default``.

    A non-positive or unparseable value logs a warning and returns a safe value
    (>= 1) rather than disabling the limit.
    """
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; falling back to %d.", env_var, raw, default)
        return default
    if value < 1:
        logger.warning("%s=%d is below 1; using 1.", env_var, value)
        return 1
    return value


def _resolve_embed_thread_cap() -> int:
    """Resolve the per-encode CPU thread cap (``HEADROOM_EMBED_NUM_THREADS``)."""
    return _resolve_positive_int_env(_EMBED_THREADS_ENV, _DEFAULT_EMBED_THREADS)


def _resolve_embed_concurrency() -> int:
    """Resolve the max concurrent CPU encodes (``HEADROOM_EMBED_CONCURRENCY``).

    Defaults to ``min(4, os.cpu_count())`` so embedding cannot occupy every core
    and starve the event loop, while still allowing useful parallelism.
    """
    cpu = os.cpu_count() or 1
    raw = os.environ.get(_EMBED_CONCURRENCY_ENV)
    if raw is None:
        return max(1, min(_DEFAULT_EMBED_CONCURRENCY, cpu))
    return _resolve_positive_int_env(
        _EMBED_CONCURRENCY_ENV, max(1, min(_DEFAULT_EMBED_CONCURRENCY, cpu))
    )


def _init_cpu_embed_worker() -> None:
    """Pin a CPU embed worker's thread pool (runs once per worker; issue #198).

    Sets BLAS/OpenMP env defaults (``setdefault`` never overrides an operator's
    explicit setting) and bounds torch's intra-op pool for this worker thread.
    Best-effort: failures never block embedding.
    """
    n = _resolve_embed_thread_cap()
    for var in _BLAS_THREAD_ENV_VARS:
        os.environ.setdefault(var, str(n))
    try:
        import torch

        torch.set_num_threads(n)
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover - defensive, never block embedding
        logger.debug("Could not cap torch intra-op thread pool: %s", exc)


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize embedding to unit vector for cosine similarity.

    Args:
        embedding: The embedding vector to normalize.

    Returns:
        Normalized embedding with L2 norm of 1.0.
    """
    norm = np.linalg.norm(embedding)
    if norm > 0:
        result: np.ndarray = (embedding / norm).astype(np.float32)
        return result
    result = embedding.astype(np.float32)
    return result


def _normalize_embeddings_batch(embeddings: np.ndarray) -> np.ndarray:
    """Normalize a batch of embeddings to unit vectors.

    Args:
        embeddings: 2D array of embeddings (batch_size, dimension).

    Returns:
        Normalized embeddings with L2 norm of 1.0 per row.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms > 0, norms, 1.0)
    result: np.ndarray = (embeddings / norms).astype(np.float32)
    return result


# =============================================================================
# LocalEmbedder - sentence-transformers
# =============================================================================


class LocalEmbedder:
    """Local embedding using sentence-transformers.

    Uses the sentence-transformers library for local embedding generation.
    No API calls needed - runs entirely on local hardware.

    Features:
    - Lazy model loading (loads on first use)
    - Automatic device selection (CUDA > MPS > CPU)
    - Batch embedding support
    - Returns normalized float32 vectors

    Default model: all-MiniLM-L6-v2 (384 dimensions)

    Usage:
        embedder = LocalEmbedder()
        embedding = await embedder.embed("Hello world")
        embeddings = await embedder.embed_batch(["Hello", "World"])
    """

    DEFAULT_DIMENSION = 384
    DEFAULT_MAX_TOKENS = 256

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize the local embedder.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Defaults to config's sentence_transformer setting.
            device: Device to run on ("cuda", "mps", "cpu", or None for auto).
                   If None, automatically selects the best available device.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        self._model_name = model_name or ML_MODEL_DEFAULTS.sentence_transformer
        self._requested_device = device
        self._model: SentenceTransformer | None = None
        self._device: str | None = None
        self._dimension: int | None = None
        self._lock = asyncio.Lock()
        # Dedicated single-worker executor, created only when the resolved device
        # is MPS (see _load_model). torch-MPS is not thread-safe, so every encode()
        # must run on one thread. Stays None for CPU/CUDA → default shared executor.
        self._executor: ThreadPoolExecutor | None = None

    def _check_dependencies(self) -> None:
        """Check that required dependencies are installed."""
        try:
            import sentence_transformers  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for LocalEmbedder. "
                "Install it with: pip install sentence-transformers"
            ) from e

    def _detect_device(self) -> str:
        """Auto-detect the best available device.

        Returns:
            Device string: "cuda", "mps", or "cpu".
        """
        import torch

        if torch.cuda.is_available():
            logger.info("CUDA device detected, using GPU")
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info("MPS device detected, using Apple Silicon GPU")
            return "mps"
        else:
            logger.info("No GPU detected, using CPU")
            return "cpu"

    def _load_model(self) -> None:
        """Load the sentence-transformers model lazily via MLModelRegistry."""
        if self._model is not None:
            return

        self._check_dependencies()
        from headroom.models.ml_models import MLModelRegistry

        # Determine device
        if self._requested_device:
            self._device = self._requested_device
        else:
            self._device = self._detect_device()

        # CPU: run encodes on a dedicated, size-limited executor whose workers
        # each pin their torch/BLAS/OpenMP thread pool (issue #198). Without this,
        # N concurrent encodes on the shared default executor each fan out to
        # ~os.cpu_count() BLAS threads and starve the asyncio event loop, spiking
        # /livez latency. Total embed threads are bounded by workers x per-encode
        # threads.
        if self._device == "cpu" and self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=_resolve_embed_concurrency(),
                thread_name_prefix="cpu-embed",
                initializer=_init_cpu_embed_worker,
            )

        # torch-MPS is not thread-safe: concurrent encode() calls from the default
        # multi-worker executor abort with "commit an already committed command
        # buffer" (verified). Funnel every encode through one worker thread when on
        # MPS so calls serialize; other devices keep the shared default executor.
        if self._device == "mps" and self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mps-embed")

        # Use centralized registry for shared model instances
        self._model = MLModelRegistry.get_sentence_transformer(self._model_name, self._device)

        # Get actual dimension from loaded model
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            f"Model loaded (shared): {self._model_name}, dimension={self._dimension}, device={self._device}"
        )

    async def embed(self, text: str) -> np.ndarray:
        """Generate an embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            Normalized embedding vector as float32 numpy array.
        """
        async with self._lock:
            # Load model if not already loaded
            if self._model is None:
                await asyncio.get_event_loop().run_in_executor(None, self._load_model)

        # Handle empty string
        if not text or not text.strip():
            return np.zeros(self.dimension, dtype=np.float32)

        # Run encoding in executor to avoid blocking
        # Model is guaranteed to be loaded after the lock check above
        assert self._model is not None
        model = self._model  # Local reference for lambda closure
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            self._executor,
            lambda: model.encode(text, convert_to_numpy=True, normalize_embeddings=False),
        )

        return _normalize_embedding(embedding)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of normalized embedding vectors.
        """
        if not texts:
            return []

        async with self._lock:
            # Load model if not already loaded
            if self._model is None:
                await asyncio.get_event_loop().run_in_executor(None, self._load_model)

        # Handle empty strings by tracking their indices
        non_empty_indices = []
        non_empty_texts = []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(text)

        # Initialize results with zeros for empty strings
        results: list[np.ndarray] = [
            np.zeros(self.dimension, dtype=np.float32) for _ in range(len(texts))
        ]

        if non_empty_texts:
            # Run batch encoding in executor
            # Model is guaranteed to be loaded after the lock check above
            assert self._model is not None
            model = self._model  # Local reference for lambda closure
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                self._executor,
                lambda: model.encode(
                    non_empty_texts, convert_to_numpy=True, normalize_embeddings=False
                ),
            )

            # Normalize batch
            normalized = _normalize_embeddings_batch(embeddings)

            # Place results at correct indices
            for idx, emb in zip(non_empty_indices, normalized):
                results[idx] = emb

        return results

    @property
    def dimension(self) -> int:
        """Return the dimension of generated embeddings."""
        if self._dimension is not None:
            return self._dimension
        # Return default dimension before model is loaded
        return self.DEFAULT_DIMENSION

    @property
    def model_name(self) -> str:
        """Return the name of the embedding model."""
        return self._model_name

    @property
    def max_tokens(self) -> int:
        """Return the maximum number of tokens the model can process."""
        return self.DEFAULT_MAX_TOKENS

    async def close(self) -> None:
        """Close resources: shut down the MPS serialization executor and drop the
        cached model reference so a later embed() fully re-initializes (and
        re-creates the serialized executor) instead of encoding on a torn-down pool.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        self._model = None


# =============================================================================
# OnnxLocalEmbedder - ONNX Runtime (no torch/sentence-transformers needed)
# =============================================================================


class OnnxLocalEmbedder:
    """Local embedding using ONNX Runtime — no torch dependency.

    Uses the same all-MiniLM-L6-v2 model as LocalEmbedder, but loaded
    via ONNX Runtime (~86 MB) instead of sentence-transformers + PyTorch (~2 GB).

    Dependencies: onnxruntime, tokenizers, huggingface_hub (all in proxy extras).
    Model auto-downloaded from HuggingFace on first use.

    Usage:
        embedder = OnnxLocalEmbedder()
        embedding = await embedder.embed("Hello world")
    """

    DEFAULT_DIMENSION = 384
    DEFAULT_MAX_TOKENS = 256
    ONNX_REPO = "Qdrant/all-MiniLM-L6-v2-onnx"
    MAX_BATCH_SIZE = 2

    def __init__(self, max_length: int = 256) -> None:
        self._max_length = max_length
        self._session: Any = None
        self._tokenizer: Any = None
        self._input_names: list[str] = []
        self._lock = asyncio.Lock()

    def _load_model(self) -> None:
        """Lazy-load the ONNX model and tokenizer."""
        if self._session is not None:
            return

        import onnxruntime as ort
        from tokenizers import Tokenizer

        logger.info("Loading ONNX embedding model (all-MiniLM-L6-v2, ~86MB)...")

        # Prefer local cache to avoid a redundant network HEAD on warm starts.
        model_path = hf_hub_download_local_first(self.ONNX_REPO, "model.onnx")
        tok_path = hf_hub_download_local_first(self.ONNX_REPO, "tokenizer.json")

        # Keep a small thread pool for Docker compatibility and disable ORT's
        # CPU memory arena/pattern caches so long-running proxy workers do not
        # retain large anonymous heaps after embedding bursts.
        sess_options = create_cpu_session_options(
            ort,
            intra_op_num_threads=1,
            inter_op_num_threads=1,
        )
        self._session = ort.InferenceSession(
            model_path, sess_options, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = Tokenizer.from_file(tok_path)
        self._tokenizer.enable_truncation(max_length=self._max_length)
        self._tokenizer.enable_padding(length=self._max_length)
        self._input_names = [inp.name for inp in self._session.get_inputs()]

        logger.info("ONNX embedding model loaded (384-dim, no torch)")

    def _build_feeds(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Build ONNX feeds for a token batch."""
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        feeds: dict[str, np.ndarray] = {}
        for name in self._input_names:
            if "input_ids" in name:
                feeds[name] = input_ids
            elif "attention_mask" in name:
                feeds[name] = attention_mask
            elif "token_type_ids" in name:
                feeds[name] = token_type_ids

        return feeds

    def _embed_many(self, texts: list[str]) -> np.ndarray:
        """Embed multiple non-empty text strings in one ONNX pass."""
        assert self._session is not None
        assert self._tokenizer is not None

        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([encoding.ids for encoding in encodings], dtype=np.int64)
        attention_mask = np.array(
            [encoding.attention_mask for encoding in encodings], dtype=np.int64
        )

        outputs = self._session.run(None, self._build_feeds(input_ids, attention_mask))
        token_embeddings = outputs[0]  # (batch, seq_len, 384)

        # Mean pooling over non-padding tokens
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        embeddings = summed / counts

        return _normalize_embeddings_batch(embeddings)

    def _embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        if not text or not text.strip():
            return np.zeros(self.DEFAULT_DIMENSION, dtype=np.float32)

        embedding = self._embed_many([text])[0]
        return cast(np.ndarray, embedding)

    async def embed(self, text: str) -> np.ndarray:
        """Generate an embedding for a single text."""
        async with self._lock:
            if self._session is None:
                await asyncio.get_event_loop().run_in_executor(None, self._load_model)

        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self._embed_single, text)
        return cast(np.ndarray, embedding)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        async with self._lock:
            if self._session is None:
                await asyncio.get_event_loop().run_in_executor(None, self._load_model)

        non_empty_indices: list[int] = []
        non_empty_texts: list[str] = []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(text)

        results: list[np.ndarray] = [
            np.zeros(self.dimension, dtype=np.float32) for _ in range(len(texts))
        ]
        if not non_empty_texts:
            return results

        loop = asyncio.get_event_loop()
        for start in range(0, len(non_empty_texts), self.MAX_BATCH_SIZE):
            batch_texts = non_empty_texts[start : start + self.MAX_BATCH_SIZE]
            batch_indices = non_empty_indices[start : start + self.MAX_BATCH_SIZE]
            embeddings = await loop.run_in_executor(None, self._embed_many, batch_texts)
            for idx, embedding in zip(batch_indices, embeddings):
                results[idx] = embedding

        return results

    @property
    def dimension(self) -> int:
        return self.DEFAULT_DIMENSION

    @property
    def model_name(self) -> str:
        return "all-MiniLM-L6-v2-onnx"

    @property
    def max_tokens(self) -> int:
        return self._max_length

    async def close(self) -> None:
        """Close resources."""
        self._session = None
        self._tokenizer = None


# =============================================================================
# OpenAIEmbedder - OpenAI API
# =============================================================================


class OpenAIEmbedder:
    """OpenAI API-based embedding generation.

    Uses OpenAI's text-embedding-3-small model for high-quality embeddings.
    Requires an API key (constructor parameter or OPENAI_API_KEY env var).

    Features:
    - Async API calls with retry logic
    - Batch support with automatic rate limiting
    - Returns normalized float32 vectors

    Default model: text-embedding-3-small (1536 dimensions)

    Usage:
        embedder = OpenAIEmbedder(api_key="sk-...")
        # Or use OPENAI_API_KEY environment variable
        embedder = OpenAIEmbedder()
        embedding = await embedder.embed("Hello world")
    """

    DEFAULT_MODEL = "text-embedding-3-small"
    DEFAULT_DIMENSION = 1536
    DEFAULT_MAX_TOKENS = 8191
    MAX_BATCH_SIZE = 2048  # OpenAI's limit
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 1.0  # Base delay in seconds for exponential backoff

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        max_retries: int | None = None,
    ) -> None:
        """Initialize the OpenAI embedder.

        Args:
            api_key: OpenAI API key. If not provided, will use OPENAI_API_KEY
                    environment variable.
            model_name: Model to use. Defaults to "text-embedding-3-small".
            max_retries: Maximum number of retries for transient failures.

        Raises:
            ImportError: If openai library is not installed.
            ValueError: If no API key is provided or found in environment.
        """
        self._check_dependencies()

        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Provide api_key parameter or set "
                "OPENAI_API_KEY environment variable."
            )

        self._model_name = model_name or self.DEFAULT_MODEL
        self._max_retries = max_retries if max_retries is not None else self.MAX_RETRIES
        self._client = None

    def _check_dependencies(self) -> None:
        """Check that required dependencies are installed."""
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "openai is required for OpenAIEmbedder. Install it with: pip install openai"
            ) from e

    @cached_property
    def _async_client(self) -> Any:
        """Lazy initialization of async OpenAI client."""
        from openai import AsyncOpenAI

        return AsyncOpenAI(api_key=self._api_key)

    async def _embed_with_retry(self, texts: list[str]) -> list[np.ndarray]:
        """Call OpenAI API with retry logic for transient failures.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            ConnectionError: If all retries fail.
        """
        from openai import APIConnectionError, APITimeoutError, RateLimitError

        last_error = None

        for attempt in range(self._max_retries):
            try:
                response = await self._async_client.embeddings.create(
                    model=self._model_name,
                    input=texts,
                )
                # Extract embeddings in order
                embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
                return embeddings

            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last_error = e
                delay = self.RETRY_DELAY_BASE * (2**attempt)
                logger.warning(
                    f"OpenAI API error (attempt {attempt + 1}/{self._max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

            except Exception as e:
                # Non-retryable error
                raise ConnectionError(f"OpenAI API error: {e}") from e

        # All retries exhausted
        raise ConnectionError(
            f"OpenAI API failed after {self._max_retries} retries: {last_error}"
        ) from last_error

    async def embed(self, text: str) -> np.ndarray:
        """Generate an embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            Normalized embedding vector as float32 numpy array.

        Raises:
            ConnectionError: If API call fails after retries.
        """
        # Handle empty string
        if not text or not text.strip():
            return np.zeros(self.dimension, dtype=np.float32)

        embeddings = await self._embed_with_retry([text])
        return _normalize_embedding(embeddings[0])

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts.

        Automatically handles batching for large inputs.

        Args:
            texts: List of texts to embed.

        Returns:
            List of normalized embedding vectors.

        Raises:
            ConnectionError: If API call fails after retries.
        """
        if not texts:
            return []

        # Handle empty strings by tracking their indices
        non_empty_indices = []
        non_empty_texts = []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(text)

        # Initialize results with zeros for empty strings
        results: list[np.ndarray] = [
            np.zeros(self.dimension, dtype=np.float32) for _ in range(len(texts))
        ]

        if not non_empty_texts:
            return results

        # Process in batches
        all_embeddings: list[np.ndarray] = []
        for batch_start in range(0, len(non_empty_texts), self.MAX_BATCH_SIZE):
            batch_end = min(batch_start + self.MAX_BATCH_SIZE, len(non_empty_texts))
            batch = non_empty_texts[batch_start:batch_end]

            batch_embeddings = await self._embed_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

        # Normalize and place results at correct indices
        for idx, emb in zip(non_empty_indices, all_embeddings):
            results[idx] = _normalize_embedding(emb)

        return results

    @property
    def dimension(self) -> int:
        """Return the dimension of generated embeddings."""
        return self.DEFAULT_DIMENSION

    @property
    def model_name(self) -> str:
        """Return the name of the embedding model."""
        return self._model_name

    @property
    def max_tokens(self) -> int:
        """Return the maximum number of tokens the model can process."""
        return self.DEFAULT_MAX_TOKENS

    async def close(self) -> None:
        """Close the OpenAI async client and its underlying httpx connection."""
        if "_async_client" in self.__dict__:
            await self._async_client.close()
            # Remove from cache to allow re-creation if needed
            del self.__dict__["_async_client"]


# =============================================================================
# OllamaEmbedder - Ollama API
# =============================================================================


class OllamaEmbedder:
    """Ollama API-based embedding generation.

    Uses a local Ollama server for embedding generation. No cloud API needed.

    Features:
    - Async HTTP calls via httpx
    - Batch support
    - Retry logic for transient failures
    - Returns normalized float32 vectors

    Default model: nomic-embed-text (768 dimensions)

    Usage:
        embedder = OllamaEmbedder()  # Uses localhost:11434
        embedder = OllamaEmbedder(base_url="http://remote:11434")
        embedding = await embedder.embed("Hello world")
    """

    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_DIMENSION = 768
    DEFAULT_MAX_TOKENS = 8192
    DEFAULT_BASE_URL = "http://localhost:11434"
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 0.5  # Base delay in seconds for exponential backoff
    REQUEST_TIMEOUT = 60.0  # Timeout for API requests

    # Known model dimensions (for models that don't report their dimension)
    KNOWN_DIMENSIONS = {
        "nomic-embed-text": 768,
        "all-minilm": 384,
        "mxbai-embed-large": 1024,
    }

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        max_retries: int | None = None,
        dimension: int | None = None,
    ) -> None:
        """Initialize the Ollama embedder.

        Args:
            model_name: Model to use. Defaults to "nomic-embed-text".
            base_url: Ollama server URL. Defaults to "http://localhost:11434".
            max_retries: Maximum number of retries for transient failures.
            dimension: Override embedding dimension. If not provided, uses
                      known dimension for model or probes the API.

        Raises:
            ImportError: If httpx library is not installed.
        """
        self._check_dependencies()

        self._model_name = model_name or self.DEFAULT_MODEL
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._max_retries = max_retries if max_retries is not None else self.MAX_RETRIES
        self._explicit_dimension = dimension
        self._detected_dimension: int | None = None
        self._client: Any = None  # httpx.AsyncClient when initialized
        self._lock = asyncio.Lock()

    def _check_dependencies(self) -> None:
        """Check that required dependencies are installed."""
        try:
            import httpx  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "httpx is required for OllamaEmbedder. Install it with: pip install httpx"
            ) from e

    async def _get_client(self) -> Any:
        """Get or create the httpx async client."""
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self.REQUEST_TIMEOUT,
            )
        return self._client

    async def _embed_single_with_retry(self, text: str) -> np.ndarray:
        """Call Ollama API with retry logic for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.

        Raises:
            ConnectionError: If all retries fail.
        """
        import httpx

        client = await self._get_client()
        last_error = None

        for attempt in range(self._max_retries):
            try:
                response = await client.post(
                    "/api/embeddings",
                    json={
                        "model": self._model_name,
                        "prompt": text,
                    },
                )
                response.raise_for_status()

                data = response.json()
                embedding = np.array(data["embedding"], dtype=np.float32)

                # Detect dimension from first successful response
                if self._detected_dimension is None:
                    self._detected_dimension = len(embedding)

                return embedding

            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_error = e
                delay = self.RETRY_DELAY_BASE * (2**attempt)
                logger.warning(
                    f"Ollama API error (attempt {attempt + 1}/{self._max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

            except Exception as e:
                # Non-retryable error
                raise ConnectionError(f"Ollama API error: {e}") from e

        # All retries exhausted
        raise ConnectionError(
            f"Ollama API failed after {self._max_retries} retries: {last_error}"
        ) from last_error

    async def embed(self, text: str) -> np.ndarray:
        """Generate an embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            Normalized embedding vector as float32 numpy array.

        Raises:
            ConnectionError: If API call fails after retries.
        """
        # Handle empty string
        if not text or not text.strip():
            return np.zeros(self.dimension, dtype=np.float32)

        embedding = await self._embed_single_with_retry(text)
        return _normalize_embedding(embedding)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts.

        Ollama API doesn't support batch embedding natively,
        so we make concurrent requests.

        Args:
            texts: List of texts to embed.

        Returns:
            List of normalized embedding vectors.

        Raises:
            ConnectionError: If API call fails after retries.
        """
        if not texts:
            return []

        # Handle empty strings by tracking their indices
        non_empty_indices = []
        non_empty_texts = []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(text)

        # Initialize results with zeros for empty strings
        results: list[np.ndarray] = [
            np.zeros(self.dimension, dtype=np.float32) for _ in range(len(texts))
        ]

        if not non_empty_texts:
            return results

        # Make concurrent requests for non-empty texts
        # Use a semaphore to limit concurrency and avoid overwhelming the server
        semaphore = asyncio.Semaphore(10)

        async def embed_with_semaphore(text: str) -> np.ndarray:
            async with semaphore:
                return await self._embed_single_with_retry(text)

        tasks = [embed_with_semaphore(text) for text in non_empty_texts]
        embeddings = await asyncio.gather(*tasks)

        # Normalize and place results at correct indices
        for idx, emb in zip(non_empty_indices, embeddings):
            results[idx] = _normalize_embedding(emb)

        return results

    @property
    def dimension(self) -> int:
        """Return the dimension of generated embeddings."""
        # Priority: explicit > detected > known > default
        if self._explicit_dimension is not None:
            return self._explicit_dimension
        if self._detected_dimension is not None:
            return self._detected_dimension
        if self._model_name in self.KNOWN_DIMENSIONS:
            return self.KNOWN_DIMENSIONS[self._model_name]
        return self.DEFAULT_DIMENSION

    @property
    def model_name(self) -> str:
        """Return the name of the embedding model."""
        return self._model_name

    @property
    def max_tokens(self) -> int:
        """Return the maximum number of tokens the model can process."""
        return self.DEFAULT_MAX_TOKENS

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> OllamaEmbedder:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
