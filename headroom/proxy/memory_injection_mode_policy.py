"""Memory-injection mode resolution policy."""

from __future__ import annotations

from typing import Literal, cast

MEMORY_INJECTION_MODE_ENV = "HEADROOM_MEMORY_INJECTION_MODE"
MEMORY_INJECTION_MODE_DEFAULT: Literal["live_zone_tail", "disabled"] = "live_zone_tail"
MemoryInjectionMode = Literal["live_zone_tail", "disabled"]


def resolve_memory_injection_mode(raw: str | None) -> MemoryInjectionMode:
    """Resolve the active memory-injection routing mode from an optional value."""
    normalized = (raw or "").strip().lower()
    if not normalized:
        return MEMORY_INJECTION_MODE_DEFAULT
    if normalized in ("live_zone_tail", "disabled"):
        return cast(MemoryInjectionMode, normalized)
    raise ValueError(
        f"Invalid {MEMORY_INJECTION_MODE_ENV}={normalized!r}; "
        "expected 'live_zone_tail' or 'disabled'"
    )
