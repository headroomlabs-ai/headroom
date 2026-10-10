from __future__ import annotations

import pytest

import headroom.transforms as transforms


def test_transforms_getattr_and_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transforms, "_HTML_EXTRACTOR_AVAILABLE", True)
    monkeypatch.setattr(
        transforms,
        "_LAZY_EXPORTS",
        {"FakeExport": ("fake.module", "VALUE")},
    )
    monkeypatch.setattr(transforms, "import_module", lambda name: type("M", (), {"VALUE": 123})())

    assert transforms.__getattr__("_HTML_EXTRACTOR_AVAILABLE") is True
    assert transforms.__getattr__("FakeExport") == 123
    assert transforms.FakeExport == 123
    assert "FakeExport" in transforms.__dir__()

    with pytest.raises(AttributeError, match="__path__"):
        transforms.__getattr__("__path__")

    with pytest.raises(AttributeError, match="MissingExport"):
        transforms.__getattr__("MissingExport")


def test_anchor_selector_exports_are_deprecated(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import warnings

    monkeypatch.delitem(sys.modules, "headroom.transforms.anchor_selector", raising=False)
    monkeypatch.delitem(transforms.__dict__, "AnchorSelector", raising=False)

    # A live export stays quiet.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert transforms.ContentRouter is not None

    with pytest.warns(DeprecationWarning, match="AnchorSelector is deprecated") as record:
        selector_cls = transforms.AnchorSelector
    assert selector_cls.__name__ == "AnchorSelector"
    # One warning, attributed to this file rather than to importlib.
    assert len(record) == 1
    assert record[0].filename == __file__


def test_anchor_selector_module_import_is_deprecated(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import sys

    monkeypatch.delitem(sys.modules, "headroom.transforms.anchor_selector", raising=False)
    with pytest.warns(DeprecationWarning, match="anchor_selector is deprecated"):
        importlib.import_module("headroom.transforms.anchor_selector")
