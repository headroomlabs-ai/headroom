"""Logging policy for proxy tool-injection decisions."""

from __future__ import annotations

import logging
from typing import Literal

ToolInjectionDecision = Literal[
    "inject_first_time",
    "inject_sticky_replay",
    "skip",
    "skip_disabled_via_env",
]


def log_tool_injection_decision(
    *,
    logger: logging.Logger,
    provider: str,
    session_id: str | None,
    decision: ToolInjectionDecision,
    tool_definition_bytes_count: int,
    request_id: str | None,
) -> None:
    """Emit a cache-affecting tool-injection decision without tool contents."""

    logger.info(
        "event=tool_injection_decision provider=%s session_id=%s "
        "decision=%s tool_definition_bytes_count=%d request_id=%s",
        provider,
        session_id or "",
        decision,
        tool_definition_bytes_count,
        request_id or "",
    )
