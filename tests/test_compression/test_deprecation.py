"""The headroom.compression package API is deprecated; detector stays quiet."""

from __future__ import annotations

import importlib
import sys
import warnings
from collections.abc import Iterator

import pytest

import headroom.compression as compression


@pytest.fixture
def fresh(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Drop cached exports and the universal module for the test.

    Exports cached during the test are dropped again afterwards; monkeypatch
    then restores whatever was cached before.
    """
    for name in compression.__all__:
        monkeypatch.delitem(compression.__dict__, name, raising=False)
    monkeypatch.delitem(sys.modules, "headroom.compression.universal", raising=False)
    monkeypatch.delattr(compression, "universal", raising=False)
    yield
    for name in compression.__all__:
        compression.__dict__.pop(name, None)


@pytest.mark.usefixtures("fresh")
def test_attribute_access_warns_once() -> None:
    with pytest.warns(
        DeprecationWarning, match="compression.UniversalCompressor is deprecated"
    ) as record:
        compressor_cls = compression.UniversalCompressor
        assert compression.UniversalCompressor is compressor_cls  # cached: no second warning
    assert compressor_cls.__name__ == "UniversalCompressor"
    # One warning, attributed to this file rather than to importlib.
    assert len(record) == 1
    assert record[0].filename == __file__


@pytest.mark.usefixtures("fresh")
def test_from_import_warns_once() -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        from headroom.compression import ContentType  # noqa: F401
    deprecations = [w for w in record if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1
    assert "use headroom.compress" in str(deprecations[0].message)


@pytest.mark.usefixtures("fresh")
def test_wildcard_import_warns_once_per_name() -> None:
    namespace: dict[str, object] = {}
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        exec("from headroom.compression import *", namespace)
    messages = [str(w.message) for w in record if issubclass(w.category, DeprecationWarning)]
    assert len(messages) == len(compression.__all__)
    assert set(compression.__all__) <= set(namespace)


@pytest.mark.usefixtures("fresh")
def test_universal_module_import_warns() -> None:
    with pytest.warns(DeprecationWarning, match="compression.universal is deprecated"):
        importlib.import_module("headroom.compression.universal")


def test_detector_import_stays_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "headroom.compression.detector", raising=False)
    monkeypatch.delattr(compression, "detector", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        detector = importlib.import_module("headroom.compression.detector")
    assert callable(detector._magika_available)


@pytest.mark.usefixtures("fresh")
def test_reload_drops_cached_exports() -> None:
    import headroom.compression.detector as detector

    real = detector.ContentType
    with pytest.warns(DeprecationWarning):
        assert compression.ContentType is real
    stand_in = type("ReloadedContentType", (), {})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(detector, "ContentType", stand_in)
        importlib.reload(compression)
        with pytest.warns(DeprecationWarning):
            assert compression.ContentType is stand_in
    # Back on the real class: a second reload must drop the cached stand-in.
    importlib.reload(compression)
    with pytest.warns(DeprecationWarning):
        assert compression.ContentType is real
