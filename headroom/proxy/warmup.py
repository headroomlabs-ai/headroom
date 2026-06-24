"""Warmup registry for proxy cold-start state.

Holds references to preloaded heavy assets (ML compressors, content detectors,
memory backends, embedders) that are eagerly initialized during
``HeadroomProxy.startup`` so that request-time use shares one in-process
singleton and concurrent first-use callers do not trigger N parallel loads.

The registry is populated by ``HeadroomProxy.startup`` and exposed as
``proxy.warmup``. Unit 5's ``/debug/warmup`` endpoint serializes this state.

Slot status semantics
---------------------
Each slot exposes a :class:`WarmupSlot` with a ``status`` that must be one of:

``loaded``
    The asset was loaded successfully and its handle is stored on the slot.
``loading``
    Load is in progress (used when the warmup is async; currently only
    applies to memory embedder warm-up, everything else is synchronous).
``null``
    Preload did not run for this slot. Either preload was disabled
    (``optimize=False``) or the component is not configured / unavailable.
    This is *not* an error — the request-time lazy path should still work.
``error``
    Preload was attempted but failed. An ``error`` string is populated.
    The request-time lazy path is still the fallback and may succeed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

WarmupStatus = Literal["loaded", "loading", "null", "error"]


@dataclass
class WarmupSlot:
    """Status record for one warmed-up component.

    ``handle`` is the concrete asset (compressor instance, backend instance,
    etc.) when ``status == "loaded"``. Otherwise it is ``None``.
    """

    status: WarmupStatus = "null"
    handle: Any | None = None
    error: str | None = None
    # Free-form extra info surfaced by /debug/warmup (e.g. tree-sitter language
    # list, model id, embedder backend name). Kept small and JSON-serializable.
    info: dict[str, Any] = field(default_factory=dict)

    def mark_loaded(self, handle: Any = None, **info: Any) -> None:
        self.status = "loaded"
        self.handle = handle
        self.error = None
        if info:
            self.info.update(info)

    def mark_loading(self) -> None:
        self.status = "loading"
        self.handle = None
        self.error = None

    def mark_null(self) -> None:
        self.status = "null"
        self.handle = None
        self.error = None

    def mark_error(self, error: str) -> None:
        self.status = "error"
        self.handle = None
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        """Serialize for /debug/warmup. Never includes the raw handle."""
        payload: dict[str, Any] = {"status": self.status}
        if self.error:
            payload["error"] = self.error
        if self.info:
            payload["info"] = dict(self.info)
        return payload


@dataclass
class WarmupRegistry:
    """Shared preloaded asset registry populated by ``HeadroomProxy.startup``.

    One instance per ``HeadroomProxy``. All writes happen on the startup
    task; reads may come from request handlers or ``/debug/warmup``. Python
    dict/dataclass attribute access is thread-safe enough for our read
    patterns (single-writer on startup, readers never mutate).
    """

    kompress: WarmupSlot = field(default_factory=WarmupSlot)
    magika: WarmupSlot = field(default_factory=WarmupSlot)
    code_aware: WarmupSlot = field(default_factory=WarmupSlot)
    tree_sitter: WarmupSlot = field(default_factory=WarmupSlot)
    smart_crusher: WarmupSlot = field(default_factory=WarmupSlot)
    memory_backend: WarmupSlot = field(default_factory=WarmupSlot)
    memory_embedder: WarmupSlot = field(default_factory=WarmupSlot)

    def merge_transform_status(self, status: dict[str, str]) -> None:
        """Merge a ``eager_load_compressors`` status dict into slots.

        Only promotes a slot to ``loaded`` if the status string is
        ``enabled`` / ``ready`` / starts with ``loaded``. Any other value
        is treated as ``null`` with the string surfaced via ``info``.
        ``error`` statuses must be set explicitly by the caller.
        """

        def _apply(slot: WarmupSlot, value: str | None) -> None:
            if value is None:
                return
            v = value.strip().lower()
            if v in {"enabled", "ready"} or v.startswith("loaded"):
                # Preserve whatever handle/info was set already.
                if slot.status != "loaded":
                    slot.mark_loaded(handle=slot.handle, source_status=value)
                else:
                    slot.info.setdefault("source_status", value)
            else:
                if slot.status != "loaded":
                    slot.info["source_status"] = value

        _apply(self.kompress, status.get("kompress"))
        _apply(self.magika, status.get("magika"))
        _apply(self.code_aware, status.get("code_aware"))
        _apply(self.tree_sitter, status.get("tree_sitter"))
        _apply(self.smart_crusher, status.get("smart_crusher"))

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Serialize the whole registry (for ``/debug/warmup``)."""
        return {
            "kompress": self.kompress.to_dict(),
            "magika": self.magika.to_dict(),
            "code_aware": self.code_aware.to_dict(),
            "tree_sitter": self.tree_sitter.to_dict(),
            "smart_crusher": self.smart_crusher.to_dict(),
            "memory_backend": self.memory_backend.to_dict(),
            "memory_embedder": self.memory_embedder.to_dict(),
        }


__all__ = ["WarmupRegistry", "WarmupSlot", "WarmupStatus"]
