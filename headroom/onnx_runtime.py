"""ONNX Runtime helpers for long-running Headroom processes."""

from __future__ import annotations

import ctypes
import sys
from typing import Any


def hf_hub_download_local_first(repo_id: str, filename: str) -> str:
    """Download a file from HuggingFace Hub, preferring the local cache.

    Tries ``local_files_only=True`` first to avoid a network HEAD request when
    the model is already cached.  Falls back to a normal (network-allowed)
    download on the first cold start.

    Args:
        repo_id: HuggingFace Hub repository identifier (e.g. ``"org/model"``).
        filename: Filename within the repository.

    Returns:
        Absolute path to the local cached file.

    Raises:
        Any exception raised by ``hf_hub_download`` on a genuine download failure.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    try:
        return str(hf_hub_download(repo_id, filename, local_files_only=True))
    except (LocalEntryNotFoundError, EntryNotFoundError, OSError):
        return str(hf_hub_download(repo_id, filename))


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
