"""ONNX Runtime helpers for long-running Headroom processes."""

from __future__ import annotations

import ctypes
import os
import sys
from typing import Any

# Pin model artifacts to immutable commit SHAs so a changed or compromised
# upstream HuggingFace repo cannot be pulled silently (supply-chain integrity).
# Repos not listed here fall back to the floating default ref. Set
# HEADROOM_HF_PIN=off to bypass pinning (e.g. when intentionally evaluating a
# newer model revision). To upgrade a model, bump its SHA here deliberately.
_PINNED_REVISIONS: dict[str, str] = {
    # chopratejas/kompress-v2-base @ 2026-06-10
    "chopratejas/kompress-v2-base": "b1563631b35bfdcee37587ad530147497d820d4c",
    "chopratejas/technique-router-onnx": "27b0b4bfa510a1cff66d888072c0b807082721a8",
    "chopratejas/siglip-image-encoder-onnx": "d0a9fbd66d4bd8c761bff592d44831f7c2ae184e",
    # Third-party repo — pinning matters most here.
    "Qdrant/all-MiniLM-L6-v2-onnx": "5f1b8cd78bc4fb444dd171e59b18f3a3af89a079",
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
        return str(hf_hub_download(repo_id, filename, revision=revision))


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
    """
    sess_options = ort.SessionOptions()

    if intra_op_num_threads is not None:
        sess_options.intra_op_num_threads = intra_op_num_threads
    if inter_op_num_threads is not None:
        sess_options.inter_op_num_threads = inter_op_num_threads

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
