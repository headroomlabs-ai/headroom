from __future__ import annotations

import pytest

from headroom.proxy.helpers import (
    get_tool_injection_sticky_mode as helper_get_tool_injection_sticky_mode,
)
from headroom.proxy.helpers import (
    get_tool_tracker_max_sessions as helper_get_tool_tracker_max_sessions,
)
from headroom.proxy.tool_injection_config import (
    get_tool_injection_sticky_mode,
    get_tool_tracker_max_sessions,
)


def test_sticky_mode_defaults_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_TOOL_INJECTION_STICKY", raising=False)

    assert get_tool_injection_sticky_mode() == "enabled"


def test_sticky_mode_accepts_enabled_and_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_TOOL_INJECTION_STICKY", "enabled")
    assert get_tool_injection_sticky_mode() == "enabled"

    monkeypatch.setenv("HEADROOM_TOOL_INJECTION_STICKY", " DISABLED ")
    assert get_tool_injection_sticky_mode() == "disabled"


def test_sticky_mode_rejects_unknown_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_TOOL_INJECTION_STICKY", "maybe")

    with pytest.raises(ValueError, match="HEADROOM_TOOL_INJECTION_STICKY"):
        get_tool_injection_sticky_mode()


def test_tracker_max_sessions_defaults_to_1000(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_TOOL_TRACKER_MAX_SESSIONS", raising=False)

    assert get_tool_tracker_max_sessions() == 1000


def test_tracker_max_sessions_accepts_positive_int(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_TOOL_TRACKER_MAX_SESSIONS", "42")

    assert get_tool_tracker_max_sessions() == 42


@pytest.mark.parametrize("raw", ["0", "-1", "not-int"])
def test_tracker_max_sessions_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    monkeypatch.setenv("HEADROOM_TOOL_TRACKER_MAX_SESSIONS", raw)

    with pytest.raises(ValueError, match="HEADROOM_TOOL_TRACKER_MAX_SESSIONS"):
        get_tool_tracker_max_sessions()


def test_helpers_keep_existing_config_import_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_TOOL_INJECTION_STICKY", "disabled")
    monkeypatch.setenv("HEADROOM_TOOL_TRACKER_MAX_SESSIONS", "12")

    assert helper_get_tool_injection_sticky_mode() == get_tool_injection_sticky_mode()
    assert helper_get_tool_tracker_max_sessions() == get_tool_tracker_max_sessions()
