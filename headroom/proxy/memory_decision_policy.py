"""Pure memory-injection decision policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from headroom.proxy.helpers import _headroom_bypass_enabled


@dataclass(frozen=True)
class MemoryInjectionDecision:
    """Raw memory-injection decision before wrapping in public value types."""

    inject: bool
    skip_reason: str | None
    bypass_header_set: bool
    memory_handler_present: bool
    memory_user_id_present: bool
    mode_name: str


def decide_memory_injection(
    *,
    headers: Any,
    memory_handler_present: bool,
    memory_user_id_present: bool,
    mode_name: str,
) -> MemoryInjectionDecision:
    """Compute the canonical memory-injection decision."""
    bypass = _headroom_bypass_enabled(headers)

    if bypass:
        reason: str | None = "bypass_header"
        inject = False
    elif not memory_handler_present:
        reason = "no_handler"
        inject = False
    elif not memory_user_id_present:
        reason = "no_user_id"
        inject = False
    elif mode_name == "disabled":
        reason = "mode_disabled"
        inject = False
    elif mode_name == "tool":
        reason = "mode_tool"
        inject = False
    else:
        reason = None
        inject = True

    return MemoryInjectionDecision(
        inject=inject,
        skip_reason=reason,
        bypass_header_set=bypass,
        memory_handler_present=memory_handler_present,
        memory_user_id_present=memory_user_id_present,
        mode_name=mode_name,
    )


def apply_memory_skip_reason(tags: dict[str, str], skip_reason: str | None) -> None:
    """Stamp memory skip reason into tags when present."""
    if skip_reason is not None:
        tags["memory_skip_reason"] = skip_reason
