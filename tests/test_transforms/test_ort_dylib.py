"""Tests for headroom._ort -- the ORT_DYLIB_PATH auto-pin.

The resolver points the Rust core's ort-load-dynamic runtime at the pip
onnxruntime package's shared library on every platform: on Windows it
guards against the DLL search picking up the Windows ML System32
onnxruntime.dll, and on Linux/macOS it avoids static ORT import-time
CPU feature faults on older x86_64 CPUs (#1278). The platform is
monkeypatched so every branch runs on any CI OS.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import headroom._ort as _ort


@pytest.fixture(autouse=True)
def _fresh_resolver(monkeypatch):
    """Reset the module-level cache and scrub the env before every test."""
    monkeypatch.setattr(_ort, "_pinned", _ort._UNSET)
    monkeypatch.delenv("ORT_DYLIB_PATH", raising=False)


def _force_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")


def _fake_spec_for(monkeypatch, package_dir):
    """Make find_spec('onnxruntime') resolve to a fake package directory."""
    spec = SimpleNamespace(origin=str(package_dir / "__init__.py"))
    monkeypatch.setattr(
        _ort.importlib.util,
        "find_spec",
        lambda name: spec if name == "onnxruntime" else None,
    )


def test_pins_versioned_so_on_linux(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    pkg = tmp_path / "onnxruntime"
    capi = pkg / "capi"
    capi.mkdir(parents=True)
    so = capi / "libonnxruntime.so.1.22.0"
    so.write_bytes(b"not really a shared object")
    _fake_spec_for(monkeypatch, pkg)

    assert _ort.ensure_ort_dylib_pinned() == str(so)
    assert _ort.os.environ["ORT_DYLIB_PATH"] == str(so)


def test_pins_dylib_on_macos(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "darwin")
    pkg = tmp_path / "onnxruntime"
    capi = pkg / "capi"
    capi.mkdir(parents=True)
    dylib = capi / "libonnxruntime.1.23.2.dylib"
    dylib.write_bytes(b"not really a dylib")
    _fake_spec_for(monkeypatch, pkg)

    assert _ort.ensure_ort_dylib_pinned() == str(dylib)
    assert _ort.os.environ["ORT_DYLIB_PATH"] == str(dylib)


def test_respects_existing_env(monkeypatch):
    _force_windows(monkeypatch)
    monkeypatch.setenv("ORT_DYLIB_PATH", r"C:\custom\onnxruntime.dll")
    assert _ort.ensure_ort_dylib_pinned() == r"C:\custom\onnxruntime.dll"
    assert _ort.os.environ["ORT_DYLIB_PATH"] == r"C:\custom\onnxruntime.dll"


def test_pins_to_package_capi_dll(monkeypatch, tmp_path):
    _force_windows(monkeypatch)
    pkg = tmp_path / "onnxruntime"
    capi = pkg / "capi"
    capi.mkdir(parents=True)
    dll = capi / "onnxruntime.dll"
    dll.write_bytes(b"not really a dll")
    _fake_spec_for(monkeypatch, pkg)

    assert _ort.ensure_ort_dylib_pinned() == str(dll)
    assert _ort.os.environ["ORT_DYLIB_PATH"] == str(dll)


def test_idempotent_after_first_resolution(monkeypatch, tmp_path):
    _force_windows(monkeypatch)
    pkg = tmp_path / "onnxruntime"
    (pkg / "capi").mkdir(parents=True)
    (pkg / "capi" / "onnxruntime.dll").write_bytes(b"x")
    _fake_spec_for(monkeypatch, pkg)

    first = _ort.ensure_ort_dylib_pinned()
    # Resolution must not re-run: break find_spec and call again.
    monkeypatch.setattr(
        _ort.importlib.util,
        "find_spec",
        lambda name: pytest.fail("resolution ran twice"),
    )
    assert _ort.ensure_ort_dylib_pinned() == first


def test_no_pin_when_package_missing(monkeypatch):
    _force_windows(monkeypatch)
    monkeypatch.setattr(_ort.importlib.util, "find_spec", lambda name: None)
    assert _ort.ensure_ort_dylib_pinned() is None
    assert "ORT_DYLIB_PATH" not in _ort.os.environ


def test_no_pin_when_native_library_absent(monkeypatch, tmp_path):
    _force_windows(monkeypatch)
    pkg = tmp_path / "onnxruntime"
    pkg.mkdir()  # package exists, but no capi/onnxruntime.dll inside
    _fake_spec_for(monkeypatch, pkg)
    assert _ort.ensure_ort_dylib_pinned() is None
    assert "ORT_DYLIB_PATH" not in _ort.os.environ


def test_never_raises(monkeypatch):
    _force_windows(monkeypatch)

    def boom(name):
        raise RuntimeError("synthetic find_spec failure")

    monkeypatch.setattr(_ort.importlib.util, "find_spec", boom)
    assert _ort.ensure_ort_dylib_pinned() is None
    assert "ORT_DYLIB_PATH" not in _ort.os.environ
