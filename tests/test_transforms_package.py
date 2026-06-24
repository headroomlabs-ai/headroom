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
