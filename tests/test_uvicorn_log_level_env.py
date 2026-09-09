"""HEADROOM_LOG_LEVEL resolution for uvicorn's log level.

uvicorn's level was previously hardcoded to "warning", so a deployed proxy had
no way to turn request logging on — no env var, no CLI flag. These tests pin the
override and, more importantly, pin that an operator typo cannot stop the proxy
booting (uvicorn raises on an unknown level).
"""

from __future__ import annotations

import pytest

from headroom.proxy.server import _resolve_uvicorn_log_level


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_LOG_LEVEL", raising=False)


def test_defaults_to_warning_when_unset() -> None:
    assert _resolve_uvicorn_log_level() == "warning"


@pytest.mark.parametrize("level", ["critical", "error", "warning", "info", "debug", "trace"])
def test_accepts_every_uvicorn_level(level: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_LOG_LEVEL", level)

    assert _resolve_uvicorn_log_level() == level


def test_normalizes_case_and_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_LOG_LEVEL", "  INFO \n")

    assert _resolve_uvicorn_log_level() == "info"


@pytest.mark.parametrize("bad", ["verbose", "WARN", "", "10"])
def test_unrecognized_value_falls_back_instead_of_raising(
    bad: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typo must degrade to the old default, never fail the boot."""
    monkeypatch.setenv("HEADROOM_LOG_LEVEL", bad)

    assert _resolve_uvicorn_log_level() == "warning"
