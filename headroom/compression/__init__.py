"""Universal compression with ML-based content detection.

.. deprecated::
    The package API (``compress``, ``UniversalCompressor``,
    ``UniversalCompressorConfig``, ``CompressionResult``, ``MagikaDetector``,
    ``ContentType``, ``StructureMask``) is unused inside Headroom and will be
    removed in a future release. Use ``headroom.compress()`` for one-call
    compression, or ``ContentRouter`` from ``headroom.transforms``.

``headroom.compression.detector`` is still used by the proxy (Magika preload
at startup) and importing it does not warn.
"""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from headroom.compression.detector import ContentType, MagikaDetector
    from headroom.compression.masks import StructureMask
    from headroom.compression.universal import (
        CompressionResult,
        UniversalCompressor,
        UniversalCompressorConfig,
        compress,
    )

__all__ = [
    # Simple API
    "compress",
    # Full API
    "UniversalCompressor",
    "UniversalCompressorConfig",
    "CompressionResult",
    # Advanced
    "MagikaDetector",
    "ContentType",
    "StructureMask",
]

_DEPRECATED_EXPORTS: dict[str, tuple[str, str]] = {
    "compress": ("headroom.compression.universal", "compress"),
    "UniversalCompressor": ("headroom.compression.universal", "UniversalCompressor"),
    "UniversalCompressorConfig": ("headroom.compression.universal", "UniversalCompressorConfig"),
    "CompressionResult": ("headroom.compression.universal", "CompressionResult"),
    "MagikaDetector": ("headroom.compression.detector", "MagikaDetector"),
    "ContentType": ("headroom.compression.detector", "ContentType"),
    "StructureMask": ("headroom.compression.masks", "StructureMask"),
}

# On importlib.reload() the module dict is reused; drop cached exports so the
# next access resolves (and warns) against the current submodules.
for _name in _DEPRECATED_EXPORTS:
    globals().pop(_name, None)
del _name


def __getattr__(name: str) -> object:
    try:
        module_name, attr_name = _DEPRECATED_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    warnings.warn(
        f"headroom.compression.{name} is deprecated and will be removed in a future release; "
        "use headroom.compress() for one-call compression, or ContentRouter from headroom.transforms.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Warn once, here, rather than again from universal.py's import-time warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        module = import_module(module_name)
    value = getattr(module, attr_name)
    # Cache it, so `from headroom.compression import X` (which looks the name
    # up twice) warns once, and later accesses stay quiet.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
