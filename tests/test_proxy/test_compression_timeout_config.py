from __future__ import annotations

import importlib
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
