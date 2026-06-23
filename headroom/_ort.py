"""Pin the ONNX Runtime dylib for the Rust core on Windows.

Why this module exists
----------------------
On Windows, ``headroom._core`` consumers of the ``ort`` crate (magika
content detection, fastembed embeddings) are built with
``ort-load-dynamic``: the native ``onnxruntime.dll`` is resolved at
*runtime*. Unless ``ORT_DYLIB_PATH`` is set, ort falls back to a bare
``LoadLibrary("onnxruntime.dll")`` and the Windows DLL search order
applies — and ``C:\\Windows\\System32`` wins.

Windows 11 24H2+ ships ``System32\\onnxruntime.dll`` as part of Windows
ML (observed: 1.17.2603 "os-germanium"). Initializing an ort 2.x
session against that OS build does not fail — it deadlocks
indefinitely at 0% CPU, which the tiered detection fallback cannot
catch (a hang is not an ``Err``). Reproduced and bracketed with
``scripts/diag_magika_windows.py``: the identical session inits in
~400ms when ``ORT_DYLIB_PATH`` points at the ``onnxruntime`` pip
package's DLL (which ``headroom-ai[proxy]`` already depends on).

The fix: before anything can import ``headroom._core``, resolve the
pip-installed ``onnxruntime\\capi\\onnxruntime.dll`` and export it via
``ORT_DYLIB_PATH``. ``headroom/__init__.py`` calls this hook, which
guarantees ordering for every package-level consumer.

Behavior contract
-----------------
- Windows-only; a no-op everywhere else.
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
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ENV_VAR = "ORT_DYLIB_PATH"

# Tri-state module cache: unset sentinel / resolved path / None (no pin).
_UNSET = object()
_pinned: object = _UNSET


def ensure_ort_dylib_pinned() -> str | None:
    """Export ``ORT_DYLIB_PATH`` for the Rust core's ort runtime (Windows).

    Returns the effective dylib path (pinned now or already present in
    the environment), or ``None`` when no pin applies (non-Windows, or
    no ``onnxruntime`` package to point at). Idempotent and exception-free.
    """
    global _pinned
    if _pinned is not _UNSET:
        return _pinned  # type: ignore[return-value]
    _pinned = _resolve_and_pin()
    return _pinned  # type: ignore[return-value]


def _resolve_and_pin() -> str | None:
    if not sys.platform.startswith("win"):
        return None

    try:
        existing = os.environ.get(_ENV_VAR)
        if existing:
            logger.debug("%s already set; respecting user override: %s", _ENV_VAR, existing)
            return existing

        spec = importlib.util.find_spec("onnxruntime")
        if spec is None or not spec.origin:
            logger.debug(
                "onnxruntime package not found; %s left unset. The Rust ML detection "
                "may pick up the Windows ML System32 onnxruntime.dll, which is known "
                "to deadlock ort init on Windows 11 24H2+ (it then degrades to non-ML "
                "tiers via HEADROOM_MAGIKA_INIT_TIMEOUT_SECS). Install onnxruntime or "
                "set %s explicitly.",
                _ENV_VAR,
                _ENV_VAR,
            )
            return None

        dll = Path(spec.origin).parent / "capi" / "onnxruntime.dll"
        if not dll.is_file():
            logger.debug(
                "onnxruntime package found but %s is missing; %s left unset", dll, _ENV_VAR
            )
            return None

        os.environ[_ENV_VAR] = str(dll)
        logger.info("Pinned %s to bundled ONNX Runtime: %s", _ENV_VAR, dll)
        return str(dll)
    except Exception as exc:  # never break `import headroom` over an accelerator pin
        logger.debug("ort dylib pin skipped: %s: %s", type(exc).__name__, exc)
        return None
