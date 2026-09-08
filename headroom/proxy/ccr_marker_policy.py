"""CCR marker freshness policy.

Retrieval-tool injection is decided by ``apply_session_sticky_ccr_tool`` in
``headroom.proxy.helpers``, from what the session has actually forwarded.
"""

from __future__ import annotations

from typing import Any, Literal


def has_new_ccr_markers(
    *,
    current_detected_hashes: list[str],
    previous_forwarded_messages: list[dict[str, Any]] | None,
    provider: Literal["anthropic", "openai", "google"],
) -> bool:
    """Return whether current CCR hashes contain hashes not previously forwarded."""

    current = set(current_detected_hashes)
    if not current:
        return False
    if not previous_forwarded_messages:
        return True

    from headroom.ccr.tool_injection import CCRToolInjector

    previous = CCRToolInjector(
        provider=provider,
        inject_tool=False,
        inject_system_instructions=False,
    )
    previous.scan_for_markers(previous_forwarded_messages)
    return bool(current - set(previous.detected_hashes))
