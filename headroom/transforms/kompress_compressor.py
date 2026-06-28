"""Kompress: ModernBERT token compressor for structured tool outputs.

Auto-downloads the model from HuggingFace (chopratejas/kompress-v2-base)
on first use.

Requires the [ml] extra: pip install headroom-ai[ml]

Usage:
    >>> from headroom.transforms.kompress_compressor import KompressCompressor
    >>> compressor = KompressCompressor()
    >>> result = compressor.compress(long_tool_output)
    >>> print(result.compressed)
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

from ..config import TransformResult
from ..onnx_runtime import (
    create_cpu_session_options,
    hf_hub_download_local_first,
    trim_process_heap,
)
from ..tokenizer import Tokenizer
from .base import Transform

logger = logging.getLogger(__name__)

# Default HuggingFace model ID
HF_MODEL_ID = "chopratejas/kompress-v2-base"

# Tokens matching this pattern are always kept regardless of model score.
# Numbers, ALLCAPS identifiers, dotted paths, unix paths, file extensions,
# CLI flags, and CamelCase names carry semantic meaning that agents cannot
# reconstruct from context — dropping them degrades reasoning correctness.
# Disable with HEADROOM_KOMPRESS_MUST_KEEP=0.
_KOMPRESS_MUST_KEEP_RE = re.compile(
    r"\b0x[0-9A-Fa-f]+\b"  # hex addresses/IDs: 0x7fff2038
    r"|(?<![\w.])\d+(?:\.\d+)?(?![\w.])"  # standalone numbers: 42, 3.14
    r"|[A-Z_]{2,}"  # ALLCAPS: SIGILL, HTTP, EOF, ERROR
    r"|[a-z_][a-z0-9_]*\.[a-z0-9_]+"  # dotted.paths: libsystem_kernel.dylib
    r"|/[a-z0-9/._-]{2,}"  # unix paths: /usr/lib/python3.so
    r"|\.[a-z]{2,4}\b"  # extensions: .py .so .json
    r"|--?[a-z][\w-]*"  # flags: --verbose, -n
    r"|\b[A-Z][a-z]+[A-Z]\w*"  # CamelCase: EXC_BAD_INSTRUCTION, IndexError
)
_KOMPRESS_MUST_KEEP_ENV = "HEADROOM_KOMPRESS_MUST_KEEP"
KOMPRESS_BACKEND_ENV = "HEADROOM_KOMPRESS_BACKEND"
KOMPRESS_ONNX_FILENAME_ENV = "HEADROOM_KOMPRESS_ONNX_FILENAME"


def _add_kompress_must_keep_words(
    kept_ids: set[int],
    chunk_words: list[str],
    chunk_start: int,
) -> None:
    """Add semantically fragile words that should never be model-dropped."""
    if os.environ.get(_KOMPRESS_MUST_KEEP_ENV, "1") == "0":
        return
    for word_idx, word in enumerate(chunk_words):
        if _KOMPRESS_MUST_KEEP_RE.search(word):
            kept_ids.add(word_idx + chunk_start)


# ONNX artifacts are resolved against the model repo in this order, falling
# through on download miss OR session-load failure:
#
# - kompress-int8-wo.onnx: weight-only int8 (MatMulNBits), 261MB. Evaluated on
#   the labeled dataset_v2 test split (n=500): f1=0.9130 vs fp32's 0.9128,
#   must_keep_recall 0.9765 vs 0.9770, keep_rate 0.8097 vs 0.8100, 99.6%
#   keep-decision agreement — fp32-equivalent at 2.2x less memory. Uses the
#   com.microsoft MatMulNBits contrib op; older onnxruntime builds without the
#   8-bit kernel fail at session load and fall through to fp32.
# - kompress-fp32.onnx: lossless reference, 601MB.
# - kompress-int8.onnx: v1-era dynamic int8 (kept for custom domain repos).
#
# An operator can pin an exact file via HEADROOM_KOMPRESS_ONNX_FILENAME.
_DEFAULT_ONNX_FILENAMES = (
    "onnx/kompress-int8-wo.onnx",
    "onnx/kompress-fp32.onnx",
    "onnx/kompress-int8.onnx",
)
KOMPRESS_ONNX_INTRA_THREADS_ENV = "HEADROOM_KOMPRESS_ONNX_INTRA_THREADS"
KOMPRESS_ONNX_INTER_THREADS_ENV = "HEADROOM_KOMPRESS_ONNX_INTER_THREADS"
KOMPRESS_COREML_CACHE_DIR_ENV = "HEADROOM_KOMPRESS_COREML_CACHE_DIR"
KOMPRESS_MAX_CONCURRENT_ENV = "HEADROOM_KOMPRESS_MAX_CONCURRENT"
KOMPRESS_BATCH_SIZE_ENV = "HEADROOM_KOMPRESS_BATCH_SIZE"

KompressBackend = Literal["auto", "onnx", "onnx_cpu", "onnx_coreml", "pytorch", "pytorch_mps"]

# HuggingFace local-lookup errors that mean "asset not in cache" rather than a
# genuine failure. Caught when loading cache-only so startup can defer instead.
try:
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    _NOT_CACHED_ERRORS: tuple[type[BaseException], ...] = (
        LocalEntryNotFoundError,
        EntryNotFoundError,
        OSError,
    )
except Exception:  # pragma: no cover - huggingface_hub always present with [ml]
    _NOT_CACHED_ERRORS = (OSError,)


class KompressModelNotCached(RuntimeError):
    """Raised when a cache-only load is requested but the model is not cached.

    Used by startup eager-preload (``allow_download=False``) so the caller can
    defer the download to first use instead of blocking the proxy startup path
    on a network fetch.
    """


# Model cache: model_id -> (model, tokenizer, backend)
# Supports multiple models loaded simultaneously.
_kompress_cache: dict[str, tuple[Any, Any, str]] = {}
_kompress_lock = threading.Lock()
_execution_semaphores: dict[str, threading.BoundedSemaphore] = {}
_execution_semaphores_lock = threading.Lock()


def _selected_backend() -> KompressBackend:
    raw = os.environ.get(KOMPRESS_BACKEND_ENV, "auto").strip().lower().replace("-", "_")
    aliases = {
        "": "auto",
        "cpu": "onnx_cpu",
        "coreml": "onnx_coreml",
        "mps": "pytorch_mps",
        "torch": "pytorch",
        "torch_mps": "pytorch_mps",
        "onnx": "onnx",
        "onnx_cpu": "onnx_cpu",
        "onnx_coreml": "onnx_coreml",
        "pytorch": "pytorch",
        "pytorch_mps": "pytorch_mps",
        "auto": "auto",
    }
    backend = aliases.get(raw)
    if backend is None:
        logger.warning(
            "%s has unrecognized value %r; falling back to 'auto'. Valid values: %s",
            KOMPRESS_BACKEND_ENV,
            os.environ.get(KOMPRESS_BACKEND_ENV, ""),
            ", ".join(sorted(set(aliases.values()))),
        )
        return "auto"
    return backend  # type: ignore[return-value]


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s must be an integer, got %r; ignoring", name, raw)
        return None
    if value <= 0:
        logger.warning("%s must be positive, got %r; ignoring", name, raw)
        return None
    return value


def _onnx_session_options(ort: Any) -> Any:
    return create_cpu_session_options(
        ort,
        intra_op_num_threads=_env_int(KOMPRESS_ONNX_INTRA_THREADS_ENV),
        inter_op_num_threads=_env_int(KOMPRESS_ONNX_INTER_THREADS_ENV),
    )


def _model_device_type(model: Any, backend: str) -> str:
    if backend.startswith("onnx"):
        return backend
    if hasattr(model, "parameters"):
        try:
            return str(next(model.parameters()).device.type)
        except Exception:
            return "unknown"
    return "unknown"


def _default_max_concurrent(backend: str, device_type: str) -> int:
    # MPS/CUDA execution is usually serialized under the hood; letting many
    # Codex unit workers call the same model concurrently mostly adds queueing,
    # memory pressure, and timeout leaks. CPU defaults to 1 as well because ONNX
    # already owns its intra/inter-op threads.
    if backend.startswith("onnx"):
        return 1
    if backend == "pytorch" and device_type in {"cuda", "mps", "cpu"}:
        return 1
    return 1


def _execution_limit(backend: str, device_type: str) -> int:
    return _env_int(KOMPRESS_MAX_CONCURRENT_ENV) or _default_max_concurrent(backend, device_type)


def _execution_semaphore(backend: str, device_type: str) -> threading.BoundedSemaphore:
    limit = _execution_limit(backend, device_type)
    key = f"{backend}:{device_type}:{limit}"
    with _execution_semaphores_lock:
        semaphore = _execution_semaphores.get(key)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(limit)
            _execution_semaphores[key] = semaphore
        return semaphore


def _batch_size() -> int:
    return _env_int(KOMPRESS_BATCH_SIZE_ENV) or 32


def _bucket_count(value: int) -> str:
    """Return a coarse, privacy-preserving size bucket."""
    if value <= 0:
        return "0"
    lower = 1 << (value.bit_length() - 1)
    upper = lower << 1
    return f"{lower}-{upper}"


def _kompress_content_signature(content: str) -> Any:
    """Create a first-class TOIN signature for Kompress/plain-text content.

    This intentionally keys on shape, not values. Retrieval pressure should
    teach TOIN about this class of compressed content without storing the
    content or treating it as an anonymous fallback.
    """
    from ..telemetry.models import ToolSignature

    words = content.split()
    line_count = content.count("\n") + 1 if content else 0
    nonempty_lines = [line for line in content.splitlines() if line.strip()]
    avg_line_chars = (
        sum(len(line) for line in nonempty_lines) // len(nonempty_lines) if nonempty_lines else 0
    )
    has_paths = "/" in content or "\\" in content
    has_assignment_like_tokens = any("=" in word for word in words[:200])
    has_brackets = any(ch in content for ch in "{}[]()")
    has_error_terms = any(
        term in content.lower() for term in ("error", "exception", "traceback", "failed", "fatal")
    )
    shape = "|".join(
        (
            "kompress-text",
            f"chars:{_bucket_count(len(content))}",
            f"words:{_bucket_count(len(words))}",
            f"lines:{_bucket_count(line_count)}",
            f"avg_line:{_bucket_count(avg_line_chars)}",
            f"paths:{int(has_paths)}",
            f"assign:{int(has_assignment_like_tokens)}",
            f"brackets:{int(has_brackets)}",
            f"errors:{int(has_error_terms)}",
        )
    )
    structure_hash = hashlib.sha256(shape.encode()).hexdigest()[:24]
    return ToolSignature(
        structure_hash=structure_hash,
        field_count=0,
        has_nested_objects=False,
        has_arrays=False,
        max_depth=0,
        string_field_count=1,
        has_error_like_field=has_error_terms,
        has_message_like_field=True,
    )


def _is_onnx_available() -> bool:
    """Check if ONNX Runtime is available (lightweight, no torch needed)."""
    try:
        import onnxruntime  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _is_pytorch_available() -> bool:
    """Check if full PyTorch stack is available (requires [ml] extra)."""
    try:
        import safetensors  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def is_kompress_available() -> bool:
    """Check if Kompress can run — ONNX (lightweight) or PyTorch (full)."""
    return _is_onnx_available() or _is_pytorch_available()


# ── Model Architecture (must match training) ──────────────────────────
# torch/transformers are imported lazily — only when actually needed.
# This allows `from kompress_compressor import is_kompress_available`
# to work without torch installed.


def _get_model_class() -> type:
    """Return the HeadroomCompressorModel class, importing torch on demand."""
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class HeadroomCompressorModel(nn.Module):
        """Dual-head ModernBERT: token classification + span importance CNN."""

        def __init__(self, model_name: str = "answerdotai/ModernBERT-base"):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name, attn_implementation="eager")
            hidden_size = self.encoder.config.hidden_size  # 768

            # Head 1: Token keep/discard
            self.token_dropout = nn.Dropout(0.1)
            self.token_head = nn.Linear(hidden_size, 2)

            # Head 2: Span importance (1D CNN)
            self.span_conv = nn.Sequential(
                nn.Conv1d(hidden_size, 256, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(256, 1, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )

        def get_keep_mask(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
            """Get per-token keep/discard decision. True = keep."""
            with torch.no_grad():
                hidden = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state

                # Token head: binary classifier — argmax decides keep/discard
                token_logits = self.token_head(hidden)  # [B, L, 2]
                token_keep = (
                    token_logits[:, :, 1] > token_logits[:, :, 0]
                )  # True if class 1 > class 0

                # Span head: boost tokens in important spans
                # If a token is borderline but its span is important, keep it
                span_scores = self.span_conv(hidden.transpose(1, 2)).squeeze(1)
                span_boost = span_scores > 0.5  # span says this region matters

                # Keep if: token head says keep, OR token is borderline and span says keep
                token_probs = torch.softmax(token_logits, dim=-1)[:, :, 1]
                borderline = (token_probs > 0.3) & (token_probs <= 0.5)
                keep = token_keep | (borderline & span_boost)

                return keep  # type: ignore[no-any-return]

        def get_scores(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            """Get per-token importance scores (for ranking when target_ratio is set)."""
            with torch.no_grad():
                hidden = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
                token_probs = torch.softmax(self.token_head(hidden), dim=-1)[:, :, 1]
                span_scores = self.span_conv(hidden.transpose(1, 2)).squeeze(1)
                return token_probs * (0.5 + 0.5 * span_scores)  # type: ignore[no-any-return]

    return HeadroomCompressorModel


# ── Model Loading ─────────────────────────────────────────────────────


class _OnnxModel:
    """Thin wrapper so ONNX session has the same interface as PyTorch model."""

    def __init__(self, session: Any):
        self._session = session

    def get_scores(self, input_ids: Any, attention_mask: Any) -> Any:
        """Return [batch, seq] scores via ONNX Runtime."""
        import numpy as np

        scores = self._session.run(
            ["final_scores"],
            {
                "input_ids": np.asarray(input_ids, dtype=np.int64),
                "attention_mask": np.asarray(attention_mask, dtype=np.int64),
            },
        )
        return scores[0]  # [batch, seq] numpy array

    def get_keep_mask(self, input_ids: Any, attention_mask: Any) -> Any:
        """Return [batch, seq] boolean mask (score > 0.5)."""
        import numpy as np

        scores = self.get_scores(input_ids, attention_mask)
        return (np.array(scores) > 0.5).tolist()


def _onnx_filename_candidates() -> tuple[str, ...]:
    """ONNX repo paths to try, honoring an optional exact-file override."""
    override = os.environ.get(KOMPRESS_ONNX_FILENAME_ENV, "").strip()
    if override:
        # Put the override first but keep the defaults as a safety net.
        return (override, *(f for f in _DEFAULT_ONNX_FILENAMES if f != override))
    return _DEFAULT_ONNX_FILENAMES


def _create_onnx_session(
    model_id: str, providers: list[Any], *, allow_download: bool = True
) -> Any:
    """Resolve and load the model's ONNX artifact, trying candidates in order.

    A candidate is skipped on download miss (file not in the repo) or on
    session-load failure (e.g. the weight-only int8 artifact uses the
    MatMulNBits contrib op, which old onnxruntime builds can't run — those
    installs fall through to the fp32 artifact instead of losing Kompress).

    When ``allow_download`` is ``False`` candidates are resolved from the local
    cache only; if none is cached, :class:`KompressModelNotCached` is raised
    instead of hitting the network. ``onnxruntime`` is imported only after a
    candidate resolves, so a cache-only miss never requires it.
    """
    last_err: Exception | None = None
    cache_miss = False
    ort: Any = None
    for filename in _onnx_filename_candidates():
        try:
            onnx_path = hf_hub_download_local_first(
                model_id, filename, allow_network=allow_download
            )
        except Exception as exc:
            last_err = exc
            cache_miss = cache_miss or isinstance(exc, _NOT_CACHED_ERRORS)
            logger.debug("ONNX artifact %r unavailable for %s: %s", filename, model_id, exc)
            continue
        if ort is None:
            import onnxruntime

            ort = onnxruntime
        try:
            return ort.InferenceSession(
                onnx_path,
                _onnx_session_options(ort),
                providers=providers,
            )
        except Exception as exc:
            last_err = exc
            logger.warning(
                "ONNX artifact %r from %s failed to load (%s); trying next candidate",
                filename,
                model_id,
                exc,
            )
    if not allow_download and cache_miss:
        raise KompressModelNotCached(model_id) from last_err
    raise FileNotFoundError(
        f"No loadable ONNX artifact in {model_id}; tried {_onnx_filename_candidates()}"
    ) from last_err


def _load_kompress_onnx(
    model_id: str,
    *,
    use_coreml: bool = False,
    allow_download: bool = True,
) -> tuple[Any, Any, str]:
    """Download ONNX INT8 model from HuggingFace and load with onnxruntime.

    When ``allow_download`` is ``False`` the model and tokenizer are loaded from
    the local cache only; a cache miss raises :class:`KompressModelNotCached`
    instead of hitting the network.
    """
    with _kompress_lock:
        if model_id in _kompress_cache:
            return _kompress_cache[model_id]

        logger.info("Downloading Kompress ONNX model from %s ...", model_id)

        backend = "onnx_coreml" if use_coreml else "onnx"
        providers: list[Any]
        if use_coreml:
            from headroom import paths as _paths

            coreml_cache_dir = os.environ.get(KOMPRESS_COREML_CACHE_DIR_ENV, "").strip()
            cache_dir = (
                coreml_cache_dir
                if coreml_cache_dir
                else str(_paths.workspace_dir() / "cache" / "coreml")
            )
            os.makedirs(cache_dir, exist_ok=True)
            providers = [
                (
                    "CoreMLExecutionProvider",
                    {
                        "ModelFormat": "NeuralNetwork",
                        "MLComputeUnits": "ALL",
                        "RequireStaticInputShapes": "1",
                        "ModelCacheDirectory": cache_dir,
                    },
                ),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        session = _create_onnx_session(model_id, providers, allow_download=allow_download)
        model = _OnnxModel(session)

        from transformers import AutoTokenizer

        tokenizer = _load_modernbert_tokenizer(AutoTokenizer, allow_download=allow_download)

        _kompress_cache[model_id] = (model, tokenizer, backend)
        logger.info("Kompress ONNX loaded: %s backend=%s", model_id, backend)
        return model, tokenizer, backend


def _load_modernbert_tokenizer(auto_tokenizer: Any, *, allow_download: bool) -> Any:
    """Load the ModernBERT tokenizer, cache-only when ``allow_download`` is False."""
    try:
        return auto_tokenizer.from_pretrained(
            "answerdotai/ModernBERT-base", local_files_only=not allow_download
        )
    except _NOT_CACHED_ERRORS as exc:
        if not allow_download:
            raise KompressModelNotCached("answerdotai/ModernBERT-base") from exc
        raise


def _load_kompress_pytorch(
    model_id: str, device: str = "auto", *, allow_download: bool = True
) -> tuple[Any, Any, str]:
    """Download PyTorch model from HuggingFace and load with torch.

    When ``allow_download`` is ``False`` weights and tokenizer are loaded from
    the local cache only; a cache miss raises :class:`KompressModelNotCached`.
    """
    import torch
    from transformers import AutoTokenizer

    with _kompress_lock:
        if model_id in _kompress_cache:
            return _kompress_cache[model_id]

        logger.info("Downloading Kompress PyTorch model from %s ...", model_id)

        try:
            weights_path = hf_hub_download_local_first(
                model_id, "model.safetensors", allow_network=allow_download
            )
        except _NOT_CACHED_ERRORS as exc:
            if not allow_download:
                raise KompressModelNotCached(model_id) from exc
            raise

        HeadroomCompressorModel = _get_model_class()
        model = HeadroomCompressorModel()

        from safetensors.torch import load_file

        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=False)

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        model.to(device)
        model.eval()

        tokenizer = _load_modernbert_tokenizer(AutoTokenizer, allow_download=allow_download)
        _validate_pytorch_device(model, tokenizer, device)

        _kompress_cache[model_id] = (model, tokenizer, "pytorch")
        logger.info("Kompress PyTorch loaded on %s (%s)", device, model_id)
        return model, tokenizer, "pytorch"


def _validate_pytorch_device(model: Any, tokenizer: Any, device: str) -> None:
    if device == "cpu":
        return

    encoding = tokenizer(
        ["headroom", "kompress", "probe"],
        is_split_into_words=True,
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with _execution_semaphore("pytorch", device):
        scores = model.get_scores(input_ids, attention_mask)
        _ = scores[0].detach().cpu()


def _load_kompress(
    model_id: str = HF_MODEL_ID, device: str = "auto", *, allow_download: bool = True
) -> tuple[Any, Any, str]:
    """Load Kompress model, returns (model, tokenizer, backend).

    The default keeps the historic behavior: try ONNX CPU first
    (lightweight), then fall back to PyTorch. Operators can override via
    HEADROOM_KOMPRESS_BACKEND:

    - auto: ONNX CPU first, then PyTorch.
    - onnx / onnx_cpu: force ONNX CPU.
    - onnx_coreml: force ONNX Runtime CoreML provider with CPU fallback.
    - pytorch: force PyTorch with the configured device.
    - pytorch_mps: force PyTorch on Apple's MPS backend.

    When ``allow_download`` is ``False`` the model is loaded from the local
    cache only and a cache miss raises :class:`KompressModelNotCached` rather
    than fetching from the network.

    Models are cached by model_id — multiple models can coexist.
    """
    if model_id in _kompress_cache:
        return _kompress_cache[model_id]

    backend = _selected_backend()
    if backend in ("onnx", "onnx_cpu"):
        return _load_kompress_onnx(model_id, use_coreml=False, allow_download=allow_download)

    if backend == "onnx_coreml":
        return _load_kompress_onnx(model_id, use_coreml=True, allow_download=allow_download)

    if backend in ("pytorch", "pytorch_mps"):
        forced_device = "mps" if backend == "pytorch_mps" else device
        try:
            return _load_kompress_pytorch(model_id, forced_device, allow_download=allow_download)
        except KompressModelNotCached:
            raise
        except Exception as exc:
            if backend != "pytorch_mps":
                raise
            logger.warning(
                "Kompress PyTorch MPS validation failed for %s; falling back to ONNX CPU: %s",
                model_id,
                exc,
            )
            if _is_onnx_available():
                return _load_kompress_onnx(
                    model_id, use_coreml=False, allow_download=allow_download
                )
            return _load_kompress_pytorch(model_id, "cpu", allow_download=allow_download)

    # Auto mode: preserve stable default behavior. This avoids changing
    # compression quality/perf characteristics for existing installs while
    # allowing opt-in MPS/CoreML experiments via HEADROOM_KOMPRESS_BACKEND.
    if _is_onnx_available():
        try:
            return _load_kompress_onnx(model_id, use_coreml=False, allow_download=allow_download)
        except KompressModelNotCached:
            # Cache-only miss: don't trigger a PyTorch network download as a
            # fallback — propagate so the caller can defer.
            if not allow_download:
                raise
        except Exception as e:
            logger.warning("ONNX load failed for %s, trying PyTorch: %s", model_id, e)

    if _is_pytorch_available():
        return _load_kompress_pytorch(model_id, device, allow_download=allow_download)

    raise ImportError(
        "Kompress requires onnxruntime or torch. Install with: pip install headroom-ai[proxy]"
    )


def unload_kompress_model(model_id: str | None = None) -> bool:
    """Unload Kompress model(s) to free memory.

    Args:
        model_id: Specific model to unload. If None, unloads all cached models.
    """
    with _kompress_lock:
        if model_id is not None:
            if model_id in _kompress_cache:
                del _kompress_cache[model_id]
            else:
                return False
        elif _kompress_cache:
            _kompress_cache.clear()
        else:
            return False

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    gc.collect()
    trim_process_heap()
    return True


# ── Background model download ─────────────────────────────────────────
#
# The proxy request path must never block on a cold model download. A first
# deep-path request would otherwise resolve the 274MB ONNX artifact via an
# inline hf_hub_download on the request thread, where it races the proxy's
# compression timeout (HEADROOM_COMPRESSION_TIMEOUT_SECONDS, default 30s). The
# fetch is cancelled mid-transfer, the blob never finalizes in the HF cache,
# and every subsequent request re-hangs and fails open. Instead the request
# path resolves the model cache-only (allow_download=False) and pulls it down
# once here, in a daemon thread that the compression timeout does not bound.

_download_threads: dict[str, threading.Thread] = {}
_download_threads_lock = threading.Lock()


def _background_download(model_id: str, device: str) -> None:
    try:
        logger.info("Kompress: downloading model %s in the background ...", model_id)
        _load_kompress(model_id, device, allow_download=True)
        logger.info("Kompress: background model download complete for %s", model_id)
    except Exception as exc:
        logger.warning("Kompress: background model download failed for %s: %s", model_id, exc)


def ensure_background_download(model_id: str = HF_MODEL_ID, device: str = "auto") -> None:
    """Start a one-shot background download of the model if it isn't cached.

    Idempotent and non-blocking: at most one download thread runs per model_id,
    and a finished or failed thread is replaced on the next call so a transient
    network failure can be retried by a later request. Once the download
    completes the deep path activates on subsequent requests without ever
    blocking one on the network.
    """
    if model_id in _kompress_cache:
        return
    with _download_threads_lock:
        if model_id in _kompress_cache:
            return
        existing = _download_threads.get(model_id)
        if existing is not None and existing.is_alive():
            return
        thread = threading.Thread(
            target=_background_download,
            args=(model_id, device),
            name=f"kompress-download-{model_id.replace('/', '-')}",
            daemon=True,
        )
        _download_threads[model_id] = thread
        thread.start()


# ── Compressor ────────────────────────────────────────────────────────


@dataclass
class KompressConfig:
    """Configuration for Kompress compression.

    The model_id, chunk_words, and score_threshold are coupled: a model
    trained on 50-word chunks needs chunk_words=50 at inference. The
    defaults match kompress-v2-base. For domain-specific models, set all three.

    Example — financial documents::

        KompressConfig(
            model_id="chopratejas/kompress-finance",
            chunk_words=50,
            score_threshold=0.5,
        )
    """

    device: str = "auto"
    enable_ccr: bool = True
    model_id: str = HF_MODEL_ID
    chunk_words: int = 350
    score_threshold: float = 0.5


@dataclass
class KompressResult:
    """Result of Kompress compression."""

    compressed: str
    original: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    cache_key: str | None = None
    model_used: str = HF_MODEL_ID

    @property
    def tokens_saved(self) -> int:
        return max(0, self.original_tokens - self.compressed_tokens)

    @property
    def savings_percentage(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return (self.tokens_saved / self.original_tokens) * 100


class KompressCompressor(Transform):
    """Kompress: ModernBERT token compressor.

    Auto-downloads the model from HuggingFace on first use.
    Configure via KompressConfig to select model, chunk size, and threshold.
    """

    name: str = "kompress_compressor"

    def __init__(self, config: KompressConfig | None = None):
        self.config = config or KompressConfig()

    def preload(self, *, allow_download: bool = True) -> str:
        """Load the backing model/tokenizer and return the selected backend.

        When ``allow_download`` is ``False`` the model is loaded from the local
        cache only; if it is not cached, :class:`KompressModelNotCached` is
        raised so the caller can defer the download to first use. Startup eager
        preload uses this so a cold cache cannot block the proxy from binding
        its port.
        """

        _model, _tokenizer, backend = _load_kompress(
            self.config.model_id, self.config.device, allow_download=allow_download
        )
        return backend

    def is_ready(self) -> bool:
        """True if the model is loaded so :meth:`compress` won't touch the network.

        A plain cache-membership check — no lock, no I/O — safe to call on the
        hot request path to decide whether to run the deep compressor or skip it.
        """
        return self.config.model_id in _kompress_cache

    def ensure_background_load(self) -> None:
        """Kick off a one-shot, non-blocking background download of the model.

        No-op when the model is already cached or a download is already running.
        """
        ensure_background_download(self.config.model_id, self.config.device)

    def compress(
        self,
        content: str,
        context: str = "",
        content_type: str | None = None,
        question: str | None = None,
        target_ratio: float | None = None,
        *,
        allow_download: bool = True,
    ) -> KompressResult:
        """Compress content using Kompress model.

        Args:
            content: Text to compress.
            context: Optional surrounding context (unused by model).
            content_type: Ignored — model decides importance per content type.
            question: Ignored — reserved for future QA-aware compression.
            target_ratio: If None (default), model decides how much to keep using
                score threshold. If set (e.g. 0.3), forces that keep ratio.
                The proxy never sets this — only user-facing API does.
            allow_download: When False, load the model from the local cache only;
                a cache miss passes through instead of fetching from the network.
                The proxy sets this False so a cold model never blocks the request
                thread (see ``ensure_background_download``); direct callers keep
                the historic auto-download-on-first-use behavior.

        Returns:
            KompressResult with compressed text.
        """
        words = content.split()
        n_words = len(words)

        if n_words < 10:
            return self._passthrough(content, n_words)

        # Cooperative wall-clock budget (#1171): kompress ONNX inference is
        # O(tokens) and non-preemptible once the request's asyncio timeout fires,
        # so one large block can run for minutes holding a worker (the leak ->
        # executor-saturation -> queue-timeout cascade). Bail at the next chunk
        # boundary past this budget, keeping the unprocessed tail verbatim. 0
        # disables. Env HEADROOM_COMPRESSION_DEADLINE_MS overrides (default 20s).
        # Cached per instance: operator config, read once -- not per compress() call.
        deadline_s = getattr(self, "_deadline_s", None)
        if deadline_s is None:
            try:
                deadline_s = max(
                    0.0,
                    float(os.environ.get("HEADROOM_COMPRESSION_DEADLINE_MS", "20000")) / 1000.0,
                )
            except ValueError:
                deadline_s = 20.0
            self._deadline_s = deadline_s

        try:
            model, tokenizer, backend = _load_kompress(
                self.config.model_id, self.config.device, allow_download=allow_download
            )
            is_onnx = backend == "onnx"
            device_type = _model_device_type(model, backend)

            if self._should_batch_single_content(model, backend):
                batch_result = self.compress_batch(
                    [content],
                    context=context,
                    content_type=content_type,
                    question=question,
                    target_ratio=[target_ratio],
                    batch_size=_batch_size(),
                )
                if batch_result:
                    return batch_result[0]

            max_chunk_words = self.config.chunk_words
            kept_ids: set[int] = set()
            inference_ms = 0.0
            chunk_count = 0
            t_deadline = time.perf_counter()

            for chunk_start in range(0, n_words, max_chunk_words):
                if deadline_s and (time.perf_counter() - t_deadline) > deadline_s:
                    # Keep everything from here on verbatim and stop: a partial
                    # compression that returns NOW beats a full one that leaks a
                    # non-preemptible worker for minutes (#1171).
                    kept_ids.update(range(chunk_start, n_words))
                    logger.warning(
                        "Kompress hit %.1fs deadline after %d/%d words (%d chunks done); "
                        "kept remainder verbatim to free the request thread (#1171)",
                        deadline_s,
                        chunk_start,
                        n_words,
                        chunk_count,
                    )
                    break
                chunk_count += 1
                chunk_words = words[chunk_start : chunk_start + max_chunk_words]

                # ONNX uses numpy tensors, PyTorch uses torch tensors
                return_tensors = "np" if is_onnx else "pt"
                encoding = tokenizer(
                    chunk_words,
                    is_split_into_words=True,
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors=return_tensors,
                )

                input_ids = encoding["input_ids"]
                attention_mask = encoding["attention_mask"]
                word_ids = encoding.word_ids(batch_index=0)

                if not is_onnx:
                    device = next(model.parameters()).device
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)

                with _execution_semaphore(backend, device_type):
                    inference_started = time.perf_counter()
                    if target_ratio is not None:
                        scores = model.get_scores(input_ids, attention_mask)
                        if is_onnx:
                            score_list = scores[0]  # numpy: [seq_len]
                        else:
                            score_list = scores[0].cpu()
                    else:
                        keep_mask = model.get_keep_mask(input_ids, attention_mask)
                        if is_onnx:
                            mask_list = keep_mask[0]  # list of bools
                        else:
                            mask_list = keep_mask[0].cpu()
                    inference_ms += (time.perf_counter() - inference_started) * 1000

                if target_ratio is not None:
                    word_scores: dict[int, float] = {}
                    for idx, wid in enumerate(word_ids):
                        if wid is None:
                            continue
                        s = float(score_list[idx])
                        if wid not in word_scores or s > word_scores[wid]:
                            word_scores[wid] = s
                    if word_scores:
                        sorted_wids = sorted(
                            word_scores, key=lambda w: word_scores[w], reverse=True
                        )
                        num_keep = max(1, int(len(sorted_wids) * target_ratio))
                        for wid in sorted_wids[:num_keep]:
                            kept_ids.add(wid + chunk_start)
                else:
                    for idx, wid in enumerate(word_ids):
                        if wid is None:
                            continue
                        if bool(mask_list[idx]):
                            kept_ids.add(wid + chunk_start)

                # Hard override: always keep must-keep tokens regardless of model score.
                # Numbers, error names, paths, and flags carry meaning agents cannot
                # reconstruct from context. Disable via HEADROOM_KOMPRESS_MUST_KEEP=0.
                _add_kompress_must_keep_words(kept_ids, chunk_words, chunk_start)

            if not kept_ids:
                if inference_ms >= 1000.0:
                    logger.info(
                        "Kompress slow passthrough backend=%s device=%s words=%d chunks=%d "
                        "inference_ms=%.0f",
                        backend,
                        device_type,
                        n_words,
                        chunk_count,
                        inference_ms,
                    )
                return self._passthrough(content, n_words)

            compressed_words = [words[w] for w in sorted(kept_ids) if w < n_words]
            compressed = " ".join(compressed_words)
            compressed_count = len(compressed_words)
            ratio = compressed_count / n_words if n_words else 1.0

            result = KompressResult(
                compressed=compressed,
                original=content,
                original_tokens=n_words,
                compressed_tokens=compressed_count,
                compression_ratio=ratio,
                model_used=self.config.model_id,
            )

            # CCR marker
            if self.config.enable_ccr and ratio < 0.8:
                cache_key = self._store_in_ccr(content, compressed, n_words)
                if cache_key:
                    result.cache_key = cache_key
                    result.compressed += (
                        f"\n[{n_words} items compressed to {compressed_count}."
                        f" Retrieve more: hash={cache_key}]"
                    )

            if inference_ms >= 1000.0:
                logger.info(
                    "Kompress slow compress backend=%s device=%s words=%d chunks=%d "
                    "inference_ms=%.0f ratio=%.3f saved=%d",
                    backend,
                    device_type,
                    n_words,
                    chunk_count,
                    inference_ms,
                    ratio,
                    result.tokens_saved,
                )

            return result

        except KompressModelNotCached:
            logger.debug(
                "Kompress model %s not cached; passing through without compression",
                self.config.model_id,
            )
            return self._passthrough(content, n_words)
        except Exception as e:
            logger.warning("Kompress compression failed: %s", e)
            return self._passthrough(content, n_words)

    def compress_batch(
        self,
        contents: list[str],
        context: str = "",
        content_type: str | None = None,
        question: str | None = None,
        target_ratio: float | list[float | None] | None = None,
        batch_size: int = 32,
    ) -> list[KompressResult]:
        """Compress multiple texts. Uses batched inference on GPU, sequential on CPU.

        On GPU (PyTorch + CUDA / MPS), runs a single batched forward pass per
        chunk batch, amortizing model inference across N texts. On CPU (ONNX
        or PyTorch), falls back to sequential ``compress()`` calls because
        ONNX Runtime's CPU provider does not parallelize across the batch
        dimension for this model (empirically 0.7-0.9x vs sequential).

        The fallback is transparent: callers get the best available
        performance per device without needing to detect the backend
        themselves.

        Measured performance (RTX 3080 Ti, ~350-word inputs):

            GPU batched vs sequential:
                N=3:  1.76x speedup
                N=5:  2.08x speedup
                N=12: 2.18x speedup
                N=24: 2.34x speedup

            CPU (ONNX, 16 logical threads): falls back to sequential;
                net effect is parity with direct ``compress()`` in a loop.

        Args:
            contents: List of texts to compress. May contain short texts or
                empty strings — those pass through without a model call.
            context: Unused (parity with ``compress``).
            content_type: Unused (parity with ``compress``).
            question: Unused (parity with ``compress``).
            target_ratio: Compression target, one of:

                * ``None`` — model decides per text (same as :meth:`compress`).
                * ``float`` — applied uniformly to every text in the batch.
                * ``list`` of ``float | None`` — per-text ratio; must match
                  ``len(contents)``. ``None`` entries let the model decide for
                  that text.

            batch_size: Maximum number of chunks per forward pass on the
                batched path (GPU only — ignored on CPU fallback). Default
                ``32`` is a reasonable balance for ModernBERT on GPU.

        Returns:
            List of :class:`KompressResult`, one per input text, in input order.
            Empty input returns empty list. Failed texts fall back to
            passthrough rather than raising.

        Notes:
            On the batched GPU path, scoring uses ``get_scores`` uniformly
            (threshold at 0.5 when ``target_ratio`` is ``None``). This
            matches the ONNX non-batched behavior exactly. The PyTorch
            non-batched path applies an additional borderline + span-boost
            rule, so results may differ by a small fraction of tokens on
            ``target_ratio=None`` calls via the batched path vs direct
            :meth:`compress` on PyTorch. Call :meth:`compress` directly if
            the exact PyTorch borderline behavior is required.
        """
        n = len(contents)
        if n == 0:
            return []

        # Normalize target_ratio to a per-text list
        if isinstance(target_ratio, list):
            if len(target_ratio) != n:
                raise ValueError(
                    f"target_ratio list length {len(target_ratio)} does not match "
                    f"contents length {n}"
                )
            ratios: list[float | None] = list(target_ratio)
        else:
            ratios = [target_ratio] * n

        # Fast path: on backends where batch-dim parallelism does NOT help
        # (ONNX CPU, PyTorch CPU), fall back to sequential `compress()`
        # internally. This keeps the public API consistent while avoiding the
        # per-item slowdown measured on ONNX CPU (~0.7-0.9x vs sequential).
        # GPU users still benefit from the batched forward pass below.
        if self._should_use_sequential_fallback():
            return [
                self.compress(
                    content,
                    context=context,
                    content_type=content_type,
                    question=question,
                    target_ratio=r,
                )
                for content, r in zip(contents, ratios, strict=True)
            ]

        results: list[KompressResult | None] = [None] * n
        word_lists: list[list[str]] = [c.split() for c in contents]

        # Short texts short-circuit to passthrough — no model call needed.
        max_chunk_words = self.config.chunk_words
        chunk_queue: list[tuple[int, int, list[str], float | None]] = []
        for i, (words, ratio) in enumerate(zip(word_lists, ratios, strict=True)):
            if len(words) < 10:
                results[i] = self._passthrough(contents[i], len(words))
                continue
            for chunk_start in range(0, len(words), max_chunk_words):
                chunk_words = words[chunk_start : chunk_start + max_chunk_words]
                chunk_queue.append((i, chunk_start, chunk_words, ratio))

        if not chunk_queue:
            # Every input was short — all passthrough, no model needed.
            return [r for r in results if r is not None]

        # Load model once for the whole batch.
        try:
            model, tokenizer, backend = _load_kompress(self.config.model_id, self.config.device)
        except Exception as e:
            logger.warning("Kompress load failed for batch: %s — passthrough all", e)
            for i in range(n):
                if results[i] is None:
                    results[i] = self._passthrough(contents[i], len(word_lists[i]))
            return [r for r in results if r is not None]

        is_onnx = backend == "onnx"
        device_type = _model_device_type(model, backend)
        kept_ids_per_text: dict[int, set[int]] = {i: set() for i in range(n) if results[i] is None}
        inference_ms = 0.0

        for batch_start in range(0, len(chunk_queue), batch_size):
            batch = chunk_queue[batch_start : batch_start + batch_size]
            batch_word_lists = [c[2] for c in batch]

            try:
                return_tensors = "np" if is_onnx else "pt"
                encoding = tokenizer(
                    batch_word_lists,
                    is_split_into_words=True,
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors=return_tensors,
                )

                input_ids = encoding["input_ids"]
                attention_mask = encoding["attention_mask"]

                if not is_onnx:
                    device = next(model.parameters()).device
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)

                # Single forward pass for all chunks in this batch.
                with _execution_semaphore(backend, device_type):
                    inference_started = time.perf_counter()
                    scores = model.get_scores(input_ids, attention_mask)
                    inference_ms += (time.perf_counter() - inference_started) * 1000

                for batch_idx, (text_idx, chunk_start, chunk_words, ratio) in enumerate(batch):
                    word_ids = encoding.word_ids(batch_index=batch_idx)
                    score_list = scores[batch_idx] if is_onnx else scores[batch_idx].cpu()

                    # Token -> word reduction (max score per word).
                    word_scores: dict[int, float] = {}
                    for idx, wid in enumerate(word_ids):
                        if wid is None:
                            continue
                        s = float(score_list[idx])
                        if wid not in word_scores or s > word_scores[wid]:
                            word_scores[wid] = s

                    if not word_scores:
                        continue

                    if ratio is not None:
                        # Top-k by score.
                        sorted_wids = sorted(
                            word_scores, key=lambda w: word_scores[w], reverse=True
                        )
                        num_keep = max(1, int(len(sorted_wids) * ratio))
                        for wid in sorted_wids[:num_keep]:
                            kept_ids_per_text[text_idx].add(wid + chunk_start)
                    else:
                        # Threshold from config (default 0.5, matches ONNX get_keep_mask).
                        for wid, score in word_scores.items():
                            if score > self.config.score_threshold:
                                kept_ids_per_text[text_idx].add(wid + chunk_start)

                    _add_kompress_must_keep_words(
                        kept_ids_per_text[text_idx], chunk_words, chunk_start
                    )

            except Exception as e:
                logger.warning(
                    "Kompress batch forward pass failed: %s — passthrough affected texts", e
                )
                for text_idx, _, _, _ in batch:
                    if results[text_idx] is None:
                        results[text_idx] = self._passthrough(
                            contents[text_idx], len(word_lists[text_idx])
                        )
                        kept_ids_per_text.pop(text_idx, None)

        # Reconstruct compressed text for each non-passthrough result.
        for text_idx, kept_ids in kept_ids_per_text.items():
            if results[text_idx] is not None:
                continue
            content = contents[text_idx]
            words = word_lists[text_idx]
            n_words = len(words)

            if not kept_ids:
                results[text_idx] = self._passthrough(content, n_words)
                continue

            compressed_words = [words[w] for w in sorted(kept_ids) if w < n_words]
            compressed = " ".join(compressed_words)
            compressed_count = len(compressed_words)
            comp_ratio = compressed_count / n_words if n_words else 1.0

            result = KompressResult(
                compressed=compressed,
                original=content,
                original_tokens=n_words,
                compressed_tokens=compressed_count,
                compression_ratio=comp_ratio,
                model_used=self.config.model_id,
            )

            if self.config.enable_ccr and comp_ratio < 0.8:
                cache_key = self._store_in_ccr(content, compressed, n_words)
                if cache_key:
                    result.cache_key = cache_key
                    result.compressed += (
                        f"\n[{n_words} items compressed to {compressed_count}."
                        f" Retrieve more: hash={cache_key}]"
                    )

            results[text_idx] = result

        # Safety: every slot must be populated.
        final: list[KompressResult] = []
        for i, r in enumerate(results):
            if r is None:
                final.append(self._passthrough(contents[i], len(word_lists[i])))
            else:
                final.append(r)
        if inference_ms >= 1000.0:
            total_words = sum(len(words) for words in word_lists)
            total_saved = sum(r.tokens_saved for r in final)
            logger.info(
                "Kompress slow batch backend=%s device=%s items=%d chunks=%d "
                "batch_size=%d words=%d inference_ms=%.0f saved=%d",
                backend,
                device_type,
                n,
                len(chunk_queue),
                batch_size,
                total_words,
                inference_ms,
                total_saved,
            )
        return final

    def _should_batch_single_content(self, model: Any, backend: str) -> bool:
        if backend != "pytorch":
            return False
        device_type = _model_device_type(model, backend)
        return device_type in {"cuda", "mps"}

    def _should_use_sequential_fallback(self) -> bool:
        """Return True if batched inference wouldn't speed up on this backend.

        Empirically measured:
          - ONNX CPU: no batch-dim parallelism; batched is 0.7-0.9x vs sequential.
          - PyTorch CPU: typically similar (conservative fallback).
          - PyTorch + CUDA: 2.0-2.3x speedup at N>=3 — use batched path.

        If the model isn't loaded yet, we trigger loading so the backend
        is known. This is a no-op if the model is already in cache.
        """
        model_id = self.config.model_id
        if model_id not in _kompress_cache:
            try:
                _load_kompress(model_id, self.config.device)
            except Exception:
                return True

        if model_id not in _kompress_cache:
            return True

        model, _tokenizer, backend = _kompress_cache[model_id]

        if backend == "onnx":
            return True  # ONNX CPU provider doesn't parallelize batch dim
        if backend == "pytorch":
            try:
                import torch

                if hasattr(model, "parameters"):
                    device = next(model.parameters()).device
                    if device.type in ("cuda", "mps"):
                        return False  # GPU/MPS benefits from batching
                _ = torch
            except ImportError:
                return True
        return True  # Conservative default: sequential

    def _passthrough(self, content: str, n_words: int) -> KompressResult:
        return KompressResult(
            compressed=content,
            original=content,
            original_tokens=n_words,
            compressed_tokens=n_words,
            compression_ratio=1.0,
        )

    def apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> TransformResult:
        """Apply Kompress compression to messages (Transform interface)."""
        tokens_before = sum(tokenizer.count_text(str(m.get("content", ""))) for m in messages)
        transformed = []
        transforms_applied = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if not isinstance(content, str) or len(content.split()) < 10:
                transformed.append(message)
                continue

            # Compress tool outputs and long assistant messages
            # Model decides how much — no hardcoded ratios
            if role in ("tool", "assistant"):
                result = self.compress(content)
                if result.compression_ratio < 0.9:
                    transformed.append({**message, "content": result.compressed})
                    transforms_applied.append(f"kompress:{role}:{result.compression_ratio:.2f}")
                else:
                    transformed.append(message)
            else:
                transformed.append(message)

        tokens_after = sum(tokenizer.count_text(str(m.get("content", ""))) for m in transformed)

        return TransformResult(
            messages=transformed,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=transforms_applied or ["kompress:noop"],
        )

    def _store_in_ccr(self, original: str, compressed: str, original_tokens: int) -> str | None:
        try:
            from ..cache.compression_store import get_compression_store

            signature = _kompress_content_signature(original)
            compressed_tokens = len(compressed.split())
            store = get_compression_store()
            cache_key = store.store(
                original,
                compressed,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                original_item_count=original_tokens,
                compressed_item_count=compressed_tokens,
                tool_signature_hash=signature.structure_hash,
                compression_strategy="kompress",
            )
            with contextlib.suppress(Exception):
                from ..telemetry import get_toin

                get_toin().record_compression(
                    tool_signature=signature,
                    original_count=original_tokens,
                    compressed_count=compressed_tokens,
                    original_tokens=original_tokens,
                    compressed_tokens=compressed_tokens,
                    strategy="kompress",
                )
            return cache_key
        except Exception:
            return None
