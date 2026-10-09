"""Tests for the ``ccr``-default of ``_requested_agy_fr_mode``.

Scope (headroom-37g.32, WU-CCRDEFAULT): ``HEADROOM_AGY_FR_MODE`` defaults to
``ccr`` -- both when unset and when set to an invalid value -- so agy users get
tool-output savings by default (WU2 retrieve-exemption converges voluntary
retrieval). ``lossless`` remains available but must be requested explicitly.
The unrecoverable-marker safety net is preserved downstream: ``_resolve_agy_fr_mode``
still downgrades ccr->lossless when the retrieve MCP is not wired.
"""

from __future__ import annotations

import pytest

from headroom.proxy.handlers.gemini import _requested_agy_fr_mode


class TestRequestedAgyFrModeDefault:
    def test_unset_defaults_to_ccr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HEADROOM_AGY_FR_MODE", raising=False)
        assert _requested_agy_fr_mode() == "ccr"

    def test_invalid_value_falls_back_to_ccr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_AGY_FR_MODE", "xyz")
        assert _requested_agy_fr_mode() == "ccr"

    def test_explicit_ccr_is_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_AGY_FR_MODE", "ccr")
        assert _requested_agy_fr_mode() == "ccr"

    def test_explicit_lossless_is_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_AGY_FR_MODE", "lossless")
        assert _requested_agy_fr_mode() == "lossless"

    def test_normalizes_case_and_whitespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_AGY_FR_MODE", " CCR ")
        assert _requested_agy_fr_mode() == "ccr"
