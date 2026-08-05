from __future__ import annotations

import pytest

from headroom.proxy.tool_injection_policy import (
    TOOL_INJECTION_STICKY_ENV,
    TOOL_TRACKER_MAX_SESSIONS_DEFAULT,
    TOOL_TRACKER_MAX_SESSIONS_ENV,
    resolve_tool_injection_sticky_mode,
    resolve_tool_tracker_max_sessions,
)


def test_resolve_tool_injection_sticky_mode_defaults_to_enabled() -> None:
    assert resolve_tool_injection_sticky_mode(None) == "enabled"
    assert resolve_tool_injection_sticky_mode("  ") == "enabled"


def test_resolve_tool_injection_sticky_mode_accepts_known_values() -> None:
    assert resolve_tool_injection_sticky_mode("ENABLED") == "enabled"
    assert resolve_tool_injection_sticky_mode(" disabled ") == "disabled"


def test_resolve_tool_injection_sticky_mode_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match=TOOL_INJECTION_STICKY_ENV):
        resolve_tool_injection_sticky_mode("maybe")


def test_resolve_tool_tracker_max_sessions_defaults() -> None:
    assert resolve_tool_tracker_max_sessions(None) == TOOL_TRACKER_MAX_SESSIONS_DEFAULT
    assert resolve_tool_tracker_max_sessions("  ") == TOOL_TRACKER_MAX_SESSIONS_DEFAULT


def test_resolve_tool_tracker_max_sessions_accepts_positive_int() -> None:
    assert resolve_tool_tracker_max_sessions("42") == 42
    assert resolve_tool_tracker_max_sessions(" 7 ") == 7


@pytest.mark.parametrize("raw", ["0", "-1", "not-int", "1.5"])
def test_resolve_tool_tracker_max_sessions_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(ValueError, match=TOOL_TRACKER_MAX_SESSIONS_ENV):
        resolve_tool_tracker_max_sessions(raw)
