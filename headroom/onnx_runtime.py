"""ONNX Runtime helpers for long-running Headroom processes."""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from typing import Any

from headroom.offline import guard_egress

logger = logging.getLogger(__name__)

# Override for the CPU memory-arena default below: "1"/"true" forces the
# arena ON, "0"/"false" forces it OFF, unset/"auto" uses the platform default.
ONNX_CPU_ARENA_ENV = "HEADROOM_ONNX_CPU_ARENA"
ONNX_ALLOW_SPINNING_ENV = "HEADROOM_ONNX_ALLOW_SPINNING"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


def _env_flag(name: str) -> bool | None:
    raw = os.environ.get(name, "").strip().lower()
    if raw in _TRUTHY:
        return True
    if raw in _FALSY:
        return False
    if raw and raw != "auto":
        logger.warning("%s must be a boolean or 'auto', got %r; using auto", name, raw)
    return None


def cpu_arena_enabled() -> bool:
    """Whether ONNX Runtime's CPU memory arena should stay enabled.

    Disabling the arena trades peak throughput for lower retained RSS, which
    is the right call on the small Linux VMs Headroom commonly runs on. On
    Windows the same setting is catastrophic: without the arena every
    ``Run()`` falls back to per-node VirtualAlloc/free, slowing transformer
    inference by 2-3 orders of magnitude (onnxruntime#11627). That turned
    every compression into a >30s timeout for Windows proxy users, so the
    arena stays at ORT's default (enabled) there.
    """
    override = _env_flag(ONNX_CPU_ARENA_ENV)
    if override is not None:
        return override
    return sys.platform == "win32"


def onnx_thread_spinning_enabled() -> bool:
    """Whether ONNX Runtime intra/inter-op thread pools may spin-wait when idle.

    ORT's thread pools spin-wait on every core between inferences by default, so
    a long-lived proxy that keeps compression/embedding models loaded pegs all
    cores even while completely idle — the machine slows to a crawl after a
    while (#2495). Default to blocking idle threads (spinning off). Set
    ``HEADROOM_ONNX_ALLOW_SPINNING=1`` to restore ORT's spinning for peak
    throughput on a dedicated/batch box.
    """
    override = _env_flag(ONNX_ALLOW_SPINNING_ENV)
    if override is not None:
        return override
    return False


# Pin model artifacts to immutable commit SHAs so a changed or compromised
# upstream HuggingFace repo cannot be pulled silently (supply-chain integrity).
# Repos not listed here fall back to the floating default ref. Set
# HEADROOM_HF_PIN=off to bypass pinning (e.g. when intentionally evaluating a
# newer model revision). To upgrade a model, bump its SHA here deliberately.
_PINNED_REVISIONS: dict[str, str] = {
    # chopratejas/kompress-v2-base @ 2026-06-10
    "chopratejas/kompress-v2-base": "b1563631b35bfdcee37587ad530147497d820d4c",
    # Kompress tokenizer + torch-path encoder base. The ONNX artifacts were
    # exported against this snapshot; a silently updated tokenizer would shift
    # token ids under the shipped weights.
    "answerdotai/ModernBERT-base": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
    "chopratejas/technique-router-onnx": "27b0b4bfa510a1cff66d888072c0b807082721a8",
    "chopratejas/siglip-image-encoder-onnx": "d0a9fbd66d4bd8c761bff592d44831f7c2ae184e",
    # Third-party repo — pinning matters most here.
    "Qdrant/all-MiniLM-L6-v2-onnx": "5f1b8cd78bc4fb444dd171e59b18f3a3af89a079",
    # Default relevance embedder (BAAI/bge-small-en-v1.5), third-party too.
    # The Rust scorer loads the same snapshot: keep this SHA equal to
    # DEFAULT_MODEL_REVISION in crates/headroom-core/src/relevance/embedding.rs
    # (tests/test_fastembed_pinned_snapshot.py enforces it).
    "Qdrant/bge-small-en-v1.5-onnx-Q": "52398278842ec682c6f32300af41344b1c0b0bb2",
}


def _resolve_revision(repo_id: str, revision: str | None) -> str | None:
    """Resolve the HF revision to download: explicit arg wins, else the pinned
    SHA for a known repo, else ``None`` (floating ref)."""
    if revision is not None:
        return revision
    if os.environ.get("HEADROOM_HF_PIN", "").strip().lower() in ("off", "0", "false", "no"):
        return None
    return _PINNED_REVISIONS.get(repo_id)


def hf_hub_download_local_first(
    repo_id: str,
    filename: str,
    *,
    allow_network: bool = True,
    revision: str | None = None,
) -> str:
    """Download a file from HuggingFace Hub, preferring the local cache.

    Tries ``local_files_only=True`` first to avoid a network HEAD request when
    the model is already cached.  Falls back to a normal (network-allowed)
    download on the first cold start.

    Args:
        repo_id: HuggingFace Hub repository identifier (e.g. ``"org/model"``).
        filename: Filename within the repository.
        allow_network: When ``False``, never fall back to a network download —
            a cache miss re-raises the local-lookup error. Used by startup
            preload so a cold cache cannot block (or, via native crashes in the
            download stack, kill) the process before it binds its port.
        revision: Explicit git revision (commit SHA / tag / branch). When
            ``None``, a pinned SHA is applied for known repos (see
            ``_PINNED_REVISIONS``) for supply-chain integrity; unknown repos use
            the floating default ref.

    Returns:
        Absolute path to the local cached file.

    Raises:
        OfflineEgressBlocked: when ``HEADROOM_OFFLINE`` is set and the file is
            not already cached. A cache HIT still succeeds — the guard sits on
            the network fallback only, so a pre-seeded air-gapped deployment
            keeps working, which is the whole point of pre-seeding.
        Any exception raised by ``hf_hub_download`` on a genuine download failure,
        or the local-lookup error when ``allow_network`` is ``False`` and the
        file is not cached.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    revision = _resolve_revision(repo_id, revision)

    try:
        return str(hf_hub_download(repo_id, filename, revision=revision, local_files_only=True))
    except (LocalEntryNotFoundError, EntryNotFoundError, OSError):
        if not allow_network:
            raise
        # Air-gap chokepoint: this is the Python half of the HuggingFace fetch
        # the Rust core already guards. ``apply_offline_env`` sets
        # ``HF_HUB_OFFLINE=1``, but only with ``setdefault`` (an explicit
        # ``HF_HUB_OFFLINE=0`` wins) and only inside the proxy process — the CLI
        # and library entry points never call it, so the flag alone is not the
        # guarantee. Guarding here is, and it costs nothing on the cache-hit
        # path above.
        guard_egress(
            f"HuggingFace download of {repo_id}/{filename}",
            "huggingface.co",
        )
        return str(hf_hub_download(repo_id, filename, revision=revision))


def hf_entry_known_absent(repo_id: str, filename: str, *, revision: str | None = None) -> bool:
    """True only if a prior network lookup already confirmed ``filename`` does
    not exist in ``repo_id`` at the resolved revision.

    Backed by ``huggingface_hub``'s own cache of negative lookups (the
    ``.no_exist`` marker written after a real 404), so this never makes a
    network call itself. Returns ``False`` both when the file is cached and
    when nothing is known yet about it, on purpose: callers in a cache-only
    (``allow_network=False``) code path can use this to tell "confirmed
    missing upstream, safe to use a fallback file" apart from "just never
    checked yet, do not guess."
    """
    from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache

    revision = _resolve_revision(repo_id, revision)
    result = try_to_load_from_cache(repo_id, filename, revision=revision)
    return result is _CACHED_NO_EXIST


def _hf_not_cached_errors() -> tuple[type[BaseException], ...]:
    """Errors a ``local_files_only=True`` load raises on a plain cache miss.

    Transformers and sentence-transformers both surface a miss as ``OSError``;
    huggingface_hub's own lookup errors are listed explicitly in case a loader
    lets one through unwrapped.
    """
    try:
        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
    except Exception:  # pragma: no cover - huggingface_hub ships with every HF loader
        return (OSError,)
    return (LocalEntryNotFoundError, EntryNotFoundError, OSError)


def hf_from_pretrained_local_first(
    loader: Any,
    name_or_path: str,
    *,
    purpose: str,
    allow_network: bool = True,
    **kwargs: Any,
) -> Any:
    """Load a HuggingFace model/tokenizer/processor, local cache first.

    The ``from_pretrained`` twin of :func:`hf_hub_download_local_first`, for
    every loader that takes ``local_files_only`` — ``AutoModel.from_pretrained``,
    ``AutoTokenizer.from_pretrained``, ``AutoProcessor.from_pretrained``,
    ``SentenceTransformer`` and friends. ``loader`` is that callable, passed
    uncalled; ``kwargs`` are forwarded to both attempts.

    1. ``local_files_only=True``: a pure cache lookup that cannot open a socket.
       A warm (or pre-seeded air-gapped) cache loads here and never touches the
       network, which also skips the Hub re-validation round-trips a plain
       ``from_pretrained`` makes on every load.
    2. On a cache miss, :func:`guard_egress` runs BEFORE the remote attempt.

    Why the guard and not ``HF_HUB_OFFLINE``: ``apply_offline_env`` sets it with
    ``setdefault``, so an explicit ``HF_HUB_OFFLINE=0`` wins, and only the proxy
    process calls it at all. Under that configuration huggingface_hub's offline
    constant is false even with ``TRANSFORMERS_OFFLINE=1``, and a bare
    ``from_pretrained`` downloads. ``HEADROOM_OFFLINE`` is the master switch;
    this guard is what makes it one for model loaders.

    Args:
        loader: The ``from_pretrained``-style callable.
        name_or_path: Hub repo id or local directory.
        purpose: What is being loaded, for the refusal message ("SigLIP model").
        allow_network: When ``False`` a cache miss re-raises the local-lookup
            error instead of falling back to a download.
        **kwargs: Forwarded to ``loader`` (``revision=``, ``device=``, ...).

    Raises:
        OfflineEgressBlocked: ``HEADROOM_OFFLINE`` is set and the model is not
            cached. Callers translate this into their own "model unavailable".
        The loader's own error on a cache miss when ``allow_network`` is
        ``False``, when ``name_or_path`` is a local directory (nothing remote
        to fall back to), or when the remote attempt itself fails.
    """
    kwargs.pop("local_files_only", None)
    try:
        return loader(name_or_path, local_files_only=True, **kwargs)
    except _hf_not_cached_errors():
        # A local directory that failed to load has no remote to fall back to;
        # surface the real error rather than a download attempt or a refusal.
        if not allow_network or os.path.isdir(name_or_path):
            raise
    guard_egress(f"HuggingFace download of {purpose} ({name_or_path})", "huggingface.co")
    return loader(name_or_path, local_files_only=False, **kwargs)


def create_cpu_session_options(
    ort: Any,
    *,
    intra_op_num_threads: int | None = None,
    inter_op_num_threads: int | None = None,
) -> Any:
    """Create CPU-oriented ONNX Runtime session options.

    Headroom runs as a long-lived proxy process, so we bias toward predictable
    memory usage over peak ONNX throughput. Disabling ORT's CPU arena and memory
    pattern caches reduces retained anonymous RSS after variable-size inference
    workloads, which is especially important on small VMs.

    The arena is left at ORT's default on Windows (see ``cpu_arena_enabled``),
    where disabling it degrades inference latency by orders of magnitude.
    """
    sess_options = ort.SessionOptions()

    if intra_op_num_threads is not None:
        sess_options.intra_op_num_threads = intra_op_num_threads
    if inter_op_num_threads is not None:
        sess_options.inter_op_num_threads = inter_op_num_threads

    if not onnx_thread_spinning_enabled():
        # ORT's thread pools spin-wait on all cores between inferences by
        # default, so idle-but-loaded models peg every core in a long-lived
        # proxy (#2495). Make idle threads block instead. Best-effort: older ORT
        # builds may not recognize a key.
        for spin_key in (
            "session.intra_op.allow_spinning",
            "session.inter_op.allow_spinning",
        ):
            try:
                sess_options.add_session_config_entry(spin_key, "0")
            except Exception:
                pass

    if not cpu_arena_enabled():
        if hasattr(sess_options, "enable_cpu_mem_arena"):
            sess_options.enable_cpu_mem_arena = False
        if hasattr(sess_options, "enable_mem_pattern"):
            sess_options.enable_mem_pattern = False

    return sess_options


def trim_process_heap() -> bool:
    """Ask glibc to return unused heap pages to the OS when available."""
    if not sys.platform.startswith("linux"):
        return False

    try:
        libc = ctypes.CDLL("libc.so.6")
    except OSError:
        return False

    try:
        return bool(libc.malloc_trim(0))
    except Exception:
        return False
