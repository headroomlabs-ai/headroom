"""The headroom.compression package API is deprecated; detector stays quiet."""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest


def test_package_exports_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "headroom.compression.universal", raising=False)
    import headroom.compression as compression

    with pytest.warns(
        DeprecationWarning, match="compression.UniversalCompressor is deprecated"
    ) as record:
        compressor_cls = compression.UniversalCompressor
    assert compressor_cls.__name__ == "UniversalCompressor"
    # One warning, attributed to this file rather than to importlib.
    assert len(record) == 1
    assert record[0].filename == __file__

    with pytest.warns(DeprecationWarning, match="use headroom.compress"):
        from headroom.compression import compress  # noqa: F401


def test_universal_module_import_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "headroom.compression.universal", raising=False)
    with pytest.warns(DeprecationWarning, match="compression.universal is deprecated"):
        importlib.import_module("headroom.compression.universal")


def test_detector_import_stays_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "headroom.compression", raising=False)
    monkeypatch.delitem(sys.modules, "headroom.compression.detector", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        detector = importlib.import_module("headroom.compression.detector")
    assert callable(detector._magika_available)
