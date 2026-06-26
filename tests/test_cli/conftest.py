"""Shared fixtures for the CLI test suite.

tokensave is now the primary coding-task compressor, so a default
``headroom wrap`` tries to fetch the tokensave release binary. Force offline
across CLI tests so a missing binary resolves to ``None`` (→ Serena fallback)
instead of reaching out to GitHub releases. Tests that exercise the
tokensave-present path patch ``_ensure_tokensave_binary`` / ``ensure_tokensave``
directly and are unaffected by this guard. This env only gates the new
tokensave installer (``headroom.graph.tokensave_installer``); rtk and
codebase-memory-mcp installers do not read it.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _tokensave_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_BINARIES_OFFLINE", "1")
