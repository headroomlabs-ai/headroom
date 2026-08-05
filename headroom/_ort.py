"""Pin the ONNX Runtime dylib for the Rust core.

Why this module exists
----------------------
``headroom._core`` consumers of the ``ort`` crate (magika content
detection, fastembed embeddings) are built with ``ort-load-dynamic`` on
every platform: the native ONNX Runtime library is resolved at runtime
rather than statically linked.

Static ``ort-download-binaries`` linking is risky on x86_64 Linux/macOS
because Microsoft's prebuilt ORT requires AVX2 and can execute at
extension load, SIGILLing ``import headroom._core`` on pre-AVX2 CPUs
before Headroom's runtime guard can fall back (#1278). On Windows, the
dynamic fallback can pick up ``C:\\Windows\\System32\\onnxruntime.dll``
from Windows ML and deadlock ORT session init on Windows 11 24H2+.

The fix: before anything can import ``headroom._core``, resolve the
pip-installed ``onnxruntime`` package's shared library
(``capi/onnxruntime.dll`` / ``capi/libonnxruntime.so*`` /
``capi/libonnxruntime*.dylib``) and export it via ``ORT_DYLIB_PATH``.
``headroom/__init__.py`` calls this hook, which guarantees ordering for
every package-level consumer.

Behavior contract
-----------------
- Active on all platforms; pins only when the ``onnxruntime`` package is present.
- Respects a pre-set ``ORT_DYLIB_PATH`` (user override wins).
- Locates the ``onnxruntime`` package via ``find_spec`` WITHOUT
  importing it (importing would load its native code; this hook must
  stay microsecond-scale and side-effect free).
- Never raises: import-time failure of an optional accelerator must
  not break ``import headroom``. Without a pin, detection still
  degrades gracefully through HEADROOM_MAGIKA_INIT_TIMEOUT_SECS and
  the non-ML tiers.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ENV_VAR = "ORT_DYLIB_PATH"

# Tri-state module cache: unset sentinel / resolved path / None (no pin).
_UNSET = object()
_pinned: object = _UNSET


def ensure_ort_dylib_pinned() -> str | None:
    """Export ``ORT_DYLIB_PATH`` for the Rust core's ort runtime.

    Returns the effective dylib path (pinned now or already present in
    the environment), or ``None`` when no ``onnxruntime`` package/native
    library is available. Idempotent and exception-free.
    """
    global _pinned
    if _pinned is not _UNSET:
        return _pinned  # type: ignore[return-value]
    _pinned = _resolve_and_pin()
    return _pinned  # type: ignore[return-value]


def _resolve_ort_native_library(capi_dir: Path) -> Path | None:
    """Return the platform's ONNX Runtime shared library inside ``capi_dir``."""
    if sys.platform.startswith("win"):
        candidate = capi_dir / "onnxruntime.dll"
        return candidate if candidate.is_file() else None

    patterns = (
        ("libonnxruntime*.dylib",)
        if sys.platform == "darwin"
        else ("libonnxruntime.so*", "libonnxruntime*.dylib")
    )
    for pattern in patterns:
        for candidate in sorted(capi_dir.glob(pattern)):
            if candidate.is_file():
                return candidate
    return None


def _resolve_and_pin() -> str | None:
    try:
        existing = os.environ.get(_ENV_VAR)
        if existing:
            logger.debug("%s already set; respecting user override: %s", _ENV_VAR, existing)
            return existing

        spec = importlib.util.find_spec("onnxruntime")
        if spec is None or not spec.origin:
            logger.debug(
                "onnxruntime package not found; %s left unset. Rust ML detection "
                "needs a pip-installed onnxruntime (install headroom-ai[proxy] "
                "or set %s explicitly).",
                _ENV_VAR,
                _ENV_VAR,
            )
            return None

        capi_dir = Path(spec.origin).parent / "capi"
        native = _resolve_ort_native_library(capi_dir)
        if native is None:
            logger.debug(
                "onnxruntime package found but no native library under %s; %s left unset",
                capi_dir,
                _ENV_VAR,
            )
            return None

        os.environ[_ENV_VAR] = str(native)
        logger.info("Pinned %s to bundled ONNX Runtime: %s", _ENV_VAR, native)
        return str(native)
    except Exception as exc:  # never break `import headroom` over an accelerator pin
        logger.debug("ort dylib pin skipped: %s: %s", type(exc).__name__, exc)
        return None
