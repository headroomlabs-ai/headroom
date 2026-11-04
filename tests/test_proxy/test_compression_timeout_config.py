from __future__ import annotations

import importlib
import logging
import os

import pytest

import headroom.proxy.helpers as helpers


@pytest.fixture(autouse=True)
def _restore_default_timeout():
    """Each test reloads helpers under a chosen env; reset to the default afterwards
    so the reloaded module constant doesn't leak into the rest of the suite."""
    yield
    os.environ.pop("HEADROOM_COMPRESSION_TIMEOUT_SECONDS", None)
    importlib.reload(helpers)


def test_default_timeout_is_30(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_COMPRESSION_TIMEOUT_SECONDS", raising=False)
    importlib.reload(helpers)
    assert helpers.COMPRESSION_TIMEOUT_SECONDS == 30.0


def test_env_overrides_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_COMPRESSION_TIMEOUT_SECONDS", "75")
    importlib.reload(helpers)
    assert helpers.COMPRESSION_TIMEOUT_SECONDS == 75.0


def test_fractional_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_COMPRESSION_TIMEOUT_SECONDS", "12.5")
    importlib.reload(helpers)
    assert helpers.COMPRESSION_TIMEOUT_SECONDS == 12.5


def test_bad_env_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_COMPRESSION_TIMEOUT_SECONDS", "not-a-number")
    importlib.reload(helpers)
    assert helpers.COMPRESSION_TIMEOUT_SECONDS == 30.0


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno == logging.WARNING:
            self.messages.append(record.getMessage())


def _reload_warnings() -> list[str]:
    capture = _Capture()
    proxy_logger = logging.getLogger("headroom.proxy")
    proxy_logger.addHandler(capture)
    try:
        importlib.reload(helpers)
    finally:
        proxy_logger.removeHandler(capture)
    return [m for m in capture.messages if "HEADROOM_COMPRESSION_TIMEOUT_SECONDS" in m]


def test_bad_env_says_it_was_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo in the knob used to revert to 30s without a word."""
    monkeypatch.setenv("HEADROOM_COMPRESSION_TIMEOUT_SECONDS", "not-a-number")
    assert _reload_warnings() == [
        "Ignoring invalid HEADROOM_COMPRESSION_TIMEOUT_SECONDS='not-a-number'; using 30.0"
    ]


def test_valid_env_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_COMPRESSION_TIMEOUT_SECONDS", "75")
    assert _reload_warnings() == []
