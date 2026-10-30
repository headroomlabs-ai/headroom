"""Shared fixtures for the CLI test suite.

``headroom wrap`` may fetch helper binaries (e.g. tokensave) over the network
via ``headroom.binaries``. Force offline across CLI tests so a missing binary
resolves locally instead of reaching out to GitHub releases. Tests that
exercise a binary-present path patch the relevant resolver directly and are
unaffected by this guard.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _tokensave_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_BINARIES_OFFLINE", "1")


@pytest.fixture(autouse=True)
def _no_host_claude_managed_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Keep the machine's own Claude managed-settings file out of wrap tests."""
    from headroom.cli import wrap as wrap_mod

    missing = tmp_path_factory.mktemp("managed") / "managed-settings.json"
    monkeypatch.setattr(wrap_mod, "_claude_managed_settings_path", lambda: missing)
