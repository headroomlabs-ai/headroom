"""Pin the ONNX Runtime dylib for the Rust core on dynamic-ORT platforms.

Why this module exists
----------------------
On Windows and Intel macOS (``x86_64-apple-darwin``), ``headroom._core``
consumers of the ``ort`` crate (magika content detection, fastembed
embeddings) are built with ``ort-load-dynamic``: the native ONNX Runtime
library is resolved at *runtime*.

Windows: unless ``ORT_DYLIB_PATH`` is set, ort falls back to a bare
``LoadLibrary("onnxruntime.dll")`` and the Windows DLL search order
applies — and ``C:\\Windows\\System32`` wins. Windows 11 24H2+ ships
``System32\\onnxruntime.dll`` as part of Windows ML (observed:
1.17.2603 "os-germanium"). Initializing an ort 2.x session against that
OS build does not fail — it deadlocks indefinitely at 0% CPU, which the
tiered detection fallback cannot catch (a hang is not an ``Err``).
Reproduced and bracketed with ``scripts/diag_magika_windows.py``: the
identical session inits in ~400ms when ``ORT_DYLIB_PATH`` points at the
``onnxruntime`` pip package's DLL (which ``headroom-ai[proxy]`` already
depends on).

Intel macOS: ``ort-sys 2.0.0-rc.12`` does not ship prebuilt ONNX Runtime
binaries for ``x86_64-apple-darwin``, so the wheel/sdist build uses
``ort-load-dynamic`` and expects a pip-installed ``onnxruntime`` dylib
at runtime (same contract as Windows).

The fix: before anything can import ``headroom._core``, resolve the
pip-installed ``onnxruntime`` native library and export it via
``ORT_DYLIB_PATH``. ``headroom/__init__.py`` calls this hook, which
guarantees ordering for every package-level consumer.

Behavior contract
-----------------
- Active on Windows and Intel macOS only; a no-op elsewhere.
- Respects a pre-set ``ORT_DYLIB_PATH`` (user override wins).
- Locates the ``onnxruntime`` package via ``find_spec`` WITHOUT
  importing it (importing would load its native code; this hook must
  stay ~microseconds and side-effect free).
- Never raises: import-time failure of an optional accelerator must
  not break ``import headroom``. Without a pin, detection still
  degrades gracefully through HEADROOM_MAGIKA_INIT_TIMEOUT_SECS and
  the non-ML tiers.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import platform
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
    the environment), or ``None`` when no pin applies (platforms that
    bundle ORT at build time, or no ``onnxruntime`` package to point at).
    Idempotent and exception-free.
    """
    global _pinned
    if _pinned is not _UNSET:
        return _pinned  # type: ignore[return-value]
    _pinned = _resolve_and_pin()
    return _pinned  # type: ignore[return-value]


def _needs_ort_dylib_pin() -> bool:
    if sys.platform.startswith("win"):
        return True
    return sys.platform == "darwin" and platform.machine() == "x86_64"


def _resolve_ort_native_library(capi_dir: Path) -> Path | None:
    if sys.platform.startswith("win"):
        candidate = capi_dir / "onnxruntime.dll"
        return candidate if candidate.is_file() else None

    matches = sorted(capi_dir.glob("libonnxruntime*.dylib"))
    return matches[0] if matches else None


def _resolve_and_pin() -> str | None:
    if not _needs_ort_dylib_pin():
        return None

    try:
        existing = os.environ.get(_ENV_VAR)
        if existing:
            logger.debug("%s already set; respecting user override: %s", _ENV_VAR, existing)
            return existing

        spec = importlib.util.find_spec("onnxruntime")
        if spec is None or not spec.origin:
            logger.debug(
                "onnxruntime package not found; %s left unset. Rust ML detection "
                "needs a pip-installed onnxruntime on this platform (install "
                "headroom-ai[proxy] or set %s explicitly).",
                _ENV_VAR,
                _ENV_VAR,
            )
            return None

        native = _resolve_ort_native_library(Path(spec.origin).parent / "capi")
        if native is None:
            logger.debug(
                "onnxruntime package found but no native library under %s; %s left unset",
                Path(spec.origin).parent / "capi",
                _ENV_VAR,
            )
            return None

        os.environ[_ENV_VAR] = str(native)
        logger.info("Pinned %s to bundled ONNX Runtime: %s", _ENV_VAR, native)
        return str(native)
    except Exception as exc:  # never break `import headroom` over an accelerator pin
        logger.debug("ort dylib pin skipped: %s: %s", type(exc).__name__, exc)
        return None
