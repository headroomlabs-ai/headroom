from __future__ import annotations

import pytest

from headroom.proxy.memory_injection_mode_policy import (
    MEMORY_INJECTION_MODE_DEFAULT,
    resolve_memory_injection_mode,
)


def test_resolve_memory_injection_mode_defaults_for_missing_value() -> None:
    assert resolve_memory_injection_mode(None) == MEMORY_INJECTION_MODE_DEFAULT
    assert resolve_memory_injection_mode("") == MEMORY_INJECTION_MODE_DEFAULT
    assert resolve_memory_injection_mode("   ") == MEMORY_INJECTION_MODE_DEFAULT


def test_resolve_memory_injection_mode_accepts_known_values_case_insensitively() -> None:
    assert resolve_memory_injection_mode("live_zone_tail") == "live_zone_tail"
    assert resolve_memory_injection_mode(" DISABLED ") == "disabled"
    assert resolve_memory_injection_mode(" Tool ") == "tool"


def test_get_memory_injection_mode_accepts_tool_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Documented in memory_decision.py (mode_tool); must not raise per request.
    from headroom.proxy.helpers import get_memory_injection_mode

    monkeypatch.setenv("HEADROOM_MEMORY_INJECTION_MODE", "tool")
    assert get_memory_injection_mode() == "tool"


def test_resolve_memory_injection_mode_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="HEADROOM_MEMORY_INJECTION_MODE"):
        resolve_memory_injection_mode("system_prompt")


async def test_tool_mode_skips_context_search_but_keeps_decision_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tool mode: no auto-injected context, memory tools stay registrable."""
    from headroom.proxy.memory_decision import MemoryDecision
    from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler

    monkeypatch.setenv("HEADROOM_MEMORY_INJECTION_MODE", "tool")
    handler = MemoryHandler(MemoryConfig(enabled=True, backend="local"))
    messages = [{"role": "user", "content": "what did I say about databases?"}]

    assert await handler.search_and_format_context("u1", messages) is None
    assert handler._initialized is False  # returned before any backend lookup

    decision = MemoryDecision.decide(
        headers={}, memory_handler=handler, memory_user_id="u1", mode_name="tool"
    )
    assert decision.inject is True
