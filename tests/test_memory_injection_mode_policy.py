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


def test_resolve_memory_injection_mode_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="HEADROOM_MEMORY_INJECTION_MODE"):
        resolve_memory_injection_mode("system_prompt")
