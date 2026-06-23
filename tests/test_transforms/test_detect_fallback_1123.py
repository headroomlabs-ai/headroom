"""Native (Rust) content-detector failures must degrade to the pure-Python
detector instead of propagating out as an HTTP 500. Regression test for #1123."""

from __future__ import annotations

import asyncio

import headroom._core as core
from headroom.transforms import content_router as cr


def test_falls_back_on_rust_exception(monkeypatch):
    """An ordinary exception from the native detector degrades to regex."""

    def _boom(_content):
        raise RuntimeError("simulated native failure")

    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "rust")
    monkeypatch.setattr(core, "detect_content_type", _boom)
    monkeypatch.setattr(cr, "_detect_panic_warned", False, raising=False)

    # Must not raise; returns a usable detection result from the regex path.
    result = cr._detect_content('{"a": 1, "b": [1, 2, 3]}')
    assert result is not None
    assert result.content_type is not None


def test_falls_back_on_baseexception_panic(monkeypatch):
    """A BaseException-derived panic (like pyo3's PanicException) is caught too."""

    class FakePanic(BaseException):
        pass

    def _panic(_content):
        raise FakePanic("simulated pyo3 panic")

    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "rust")
    monkeypatch.setattr(core, "detect_content_type", _panic)
    monkeypatch.setattr(cr, "_detect_panic_warned", False, raising=False)

    result = cr._detect_content("some plain text content here")
    assert result is not None


def test_control_flow_exceptions_propagate(monkeypatch):
    """KeyboardInterrupt/SystemExit must not be swallowed by the fallback."""

    def _interrupt(_content):
        raise KeyboardInterrupt

    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "rust")
    monkeypatch.setattr(core, "detect_content_type", _interrupt)
    monkeypatch.setattr(cr, "_detect_panic_warned", False, raising=False)

    import pytest

    with pytest.raises(KeyboardInterrupt):
        cr._detect_content("content")


def test_cancelled_error_propagates(monkeypatch):
    """asyncio.CancelledError must propagate, not be swallowed as a fallback."""

    def _cancel(_content):
        raise asyncio.CancelledError()

    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "rust")
    monkeypatch.setattr(core, "detect_content_type", _cancel)
    monkeypatch.setattr(cr, "_detect_panic_warned", False, raising=False)

    import pytest

    with pytest.raises(asyncio.CancelledError):
        cr._detect_content("content")
