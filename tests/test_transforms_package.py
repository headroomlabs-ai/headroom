from __future__ import annotations

import importlib
import sys
import warnings

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


_ANCHOR_EXPORTS = (
    "AnchorSelector",
    "AnchorStrategy",
    "AnchorWeights",
    "DataPattern",
    "calculate_information_score",
    "compute_item_hash",
)


@pytest.fixture
def fresh_anchor_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop cached exports and the module; monkeypatch restores both afterwards."""
    for name in _ANCHOR_EXPORTS:
        monkeypatch.delitem(transforms.__dict__, name, raising=False)
    monkeypatch.delitem(sys.modules, "headroom.transforms.anchor_selector", raising=False)
    monkeypatch.delattr(transforms, "anchor_selector", raising=False)


@pytest.mark.usefixtures("fresh_anchor_selector")
def test_anchor_selector_attribute_access_warns_once() -> None:
    # A live export stays quiet.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert transforms.ContentRouter is not None

    with pytest.warns(DeprecationWarning, match="AnchorSelector is deprecated") as record:
        selector_cls = transforms.AnchorSelector
        assert transforms.AnchorSelector is selector_cls  # cached: no second warning
    assert selector_cls.__name__ == "AnchorSelector"
    # One warning, attributed to this file rather than to importlib.
    assert len(record) == 1
    assert record[0].filename == __file__


@pytest.mark.usefixtures("fresh_anchor_selector")
def test_anchor_selector_from_import_warns_once() -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        from headroom.transforms import AnchorSelector  # noqa: F401
    deprecations = [w for w in record if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1
    assert deprecations[0].filename == __file__


@pytest.mark.usefixtures("fresh_anchor_selector")
def test_anchor_selector_import_statement_is_deprecated() -> None:
    with pytest.warns(DeprecationWarning, match="anchor_selector is deprecated") as record:
        import headroom.transforms.anchor_selector  # noqa: F401
    assert record[0].filename == __file__


@pytest.mark.usefixtures("fresh_anchor_selector")
def test_anchor_selector_dynamic_import_is_deprecated() -> None:
    with pytest.warns(DeprecationWarning, match="anchor_selector is deprecated") as record:
        importlib.import_module("headroom.transforms.anchor_selector")
    if sys.version_info >= (3, 12):
        # Older Pythons cannot skip importlib's frame, so the warning names importlib.
        assert record[0].filename == __file__
