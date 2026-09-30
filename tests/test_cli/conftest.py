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
