from __future__ import annotations

import pytest

from headroom.proxy.beta_header_policy import (
    BETA_HEADER_STICKY_ENV,
    BETA_TRACKER_MAX_SESSIONS_DEFAULT,
    BETA_TRACKER_MAX_SESSIONS_ENV,
    resolve_beta_header_sticky_mode,
    resolve_beta_tracker_max_sessions,
)


def test_resolve_beta_header_sticky_mode_defaults_to_enabled() -> None:
    assert resolve_beta_header_sticky_mode(None) == "enabled"
    assert resolve_beta_header_sticky_mode("  ") == "enabled"


def test_resolve_beta_header_sticky_mode_accepts_known_values() -> None:
    assert resolve_beta_header_sticky_mode("ENABLED") == "enabled"
    assert resolve_beta_header_sticky_mode(" disabled ") == "disabled"


def test_resolve_beta_header_sticky_mode_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match=BETA_HEADER_STICKY_ENV):
        resolve_beta_header_sticky_mode("maybe")


def test_resolve_beta_tracker_max_sessions_defaults() -> None:
    assert resolve_beta_tracker_max_sessions(None) == BETA_TRACKER_MAX_SESSIONS_DEFAULT
    assert resolve_beta_tracker_max_sessions("  ") == BETA_TRACKER_MAX_SESSIONS_DEFAULT


def test_resolve_beta_tracker_max_sessions_accepts_positive_int() -> None:
    assert resolve_beta_tracker_max_sessions("42") == 42
    assert resolve_beta_tracker_max_sessions(" 7 ") == 7


@pytest.mark.parametrize("raw", ["0", "-1", "not-int", "1.5"])
def test_resolve_beta_tracker_max_sessions_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(ValueError, match=BETA_TRACKER_MAX_SESSIONS_ENV):
        resolve_beta_tracker_max_sessions(raw)
