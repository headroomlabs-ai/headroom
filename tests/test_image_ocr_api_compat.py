"""OCR backend API-compat regression tests for issue #372.

The rapidocr ecosystem split after 1.4.x:

* `rapidocr-onnxruntime` 1.4.x — Python <3.13 only; tuple result.
* `rapidocr` 3.x — Python 3.13+; `RapidOCROutput` dataclass result.

`headroom/image/compressor.py` adapts both at runtime via
`_resolve_rapidocr` + per-version branches in `_ocr_extract`. These
tests pin both branches so a future "let me clean up the v1 path"
refactor doesn't silently break Python <3.13 users.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from headroom.image import compressor as compressor_module
from headroom.image.compressor import ImageCompressor


def _install_fake_module(monkeypatch: pytest.MonkeyPatch, name: str, attrs: dict[str, Any]) -> None:
    """Inject a synthetic module into sys.modules so import sees it."""
    mod = ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)


def _hide_module(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Force ImportError when `name` is imported."""
    monkeypatch.setitem(sys.modules, name, None)


@pytest.fixture(autouse=True)
def _reset_resolver_cache() -> None:
    compressor_module._reset_resolved_ocr_for_tests()
    yield
    compressor_module._reset_resolved_ocr_for_tests()


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def test_resolve_rapidocr_prefers_v1_when_both_available(monkeypatch: pytest.MonkeyPatch) -> None:
    class _V1RapidOCR:
        pass

    class _V3RapidOCR:
        pass

    _install_fake_module(monkeypatch, "rapidocr_onnxruntime", {"RapidOCR": _V1RapidOCR})
    _install_fake_module(monkeypatch, "rapidocr", {"RapidOCR": _V3RapidOCR})

    cls, api = compressor_module._resolve_rapidocr()
    assert cls is _V1RapidOCR
    assert api == "v1"


def test_resolve_rapidocr_falls_back_to_v3_when_v1_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    class _V3RapidOCR:
        pass

    _hide_module(monkeypatch, "rapidocr_onnxruntime")
    _install_fake_module(monkeypatch, "rapidocr", {"RapidOCR": _V3RapidOCR})

    cls, api = compressor_module._resolve_rapidocr()
    assert cls is _V3RapidOCR
    assert api == "v3"


def test_resolve_rapidocr_returns_none_when_neither_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hide_module(monkeypatch, "rapidocr_onnxruntime")
    _hide_module(monkeypatch, "rapidocr")

    cls, api = compressor_module._resolve_rapidocr()
    assert cls is None
    assert api is None


# ---------------------------------------------------------------------------
# _ocr_extract — v1 tuple shape
# ---------------------------------------------------------------------------


def _make_compressor() -> ImageCompressor:
    """Build an ImageCompressor with the heavy ML deps stubbed.

    `ImageCompressor.__init__` lazy-loads the trained router; we don't
    exercise it here. Constructing a bare instance bypasses the load.
    """
    return ImageCompressor.__new__(ImageCompressor)


def test_ocr_extract_v1_tuple_shape_parses_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    class _V1Engine:
        def __call__(self, _image_data: bytes) -> tuple[list[tuple[Any, str, float]], float]:
            return (
                [
                    (None, "hello", 0.95),
                    (None, "world", 0.90),
                ],
                0.123,
            )

    class _V1RapidOCR:
        def __init__(self) -> None: ...
        def __call__(self, image_data: bytes) -> Any:  # noqa: D401
            return _V1Engine()(image_data)

    _install_fake_module(monkeypatch, "rapidocr_onnxruntime", {"RapidOCR": _V1RapidOCR})

    c = _make_compressor()
    text = c._ocr_extract(b"\x89PNG fake")
    assert text == "hello\nworld"


def test_ocr_extract_v1_low_confidence_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    class _V1RapidOCR:
        def __init__(self) -> None: ...
        def __call__(self, _image_data: bytes) -> tuple[list[Any], float]:
            return ([(None, "blurry", 0.3), (None, "smudge", 0.4)], 0.0)

    _install_fake_module(monkeypatch, "rapidocr_onnxruntime", {"RapidOCR": _V1RapidOCR})

    c = _make_compressor()
    assert c._ocr_extract(b"\x89PNG fake", min_confidence=0.7) is None


def test_ocr_extract_v1_empty_result_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    class _V1RapidOCR:
        def __init__(self) -> None: ...
        def __call__(self, _image_data: bytes) -> tuple[list[Any] | None, float]:
            return ([], 0.0)

    _install_fake_module(monkeypatch, "rapidocr_onnxruntime", {"RapidOCR": _V1RapidOCR})

    c = _make_compressor()
    assert c._ocr_extract(b"\x89PNG fake") is None


# ---------------------------------------------------------------------------
# _ocr_extract — v3 dataclass shape
# ---------------------------------------------------------------------------


def test_ocr_extract_v3_dataclass_shape_parses_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    class _V3RapidOCR:
        def __init__(self) -> None: ...
        def __call__(self, _image_data: bytes) -> Any:
            # Mirror the real `rapidocr.RapidOCROutput` minimal surface.
            return SimpleNamespace(
                txts=["hello", "world"],
                scores=[0.95, 0.90],
                boxes=[None, None],
            )

    _hide_module(monkeypatch, "rapidocr_onnxruntime")
    _install_fake_module(monkeypatch, "rapidocr", {"RapidOCR": _V3RapidOCR})

    c = _make_compressor()
    text = c._ocr_extract(b"\x89PNG fake")
    assert text == "hello\nworld"


def test_ocr_extract_v3_low_confidence_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    class _V3RapidOCR:
        def __init__(self) -> None: ...
        def __call__(self, _image_data: bytes) -> Any:
            return SimpleNamespace(txts=["blurry", "smudge"], scores=[0.3, 0.4], boxes=[None, None])

    _hide_module(monkeypatch, "rapidocr_onnxruntime")
    _install_fake_module(monkeypatch, "rapidocr", {"RapidOCR": _V3RapidOCR})

    c = _make_compressor()
    assert c._ocr_extract(b"\x89PNG fake", min_confidence=0.7) is None


def test_ocr_extract_v3_none_attrs_when_no_text_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real-world v3 behavior: when detection finds nothing, RapidOCROutput
    has txts=None and scores=None. Verified in dispatch smoke test against
    rapidocr 3.8.1.
    """

    class _V3RapidOCR:
        def __init__(self) -> None: ...
        def __call__(self, _image_data: bytes) -> Any:
            return SimpleNamespace(txts=None, scores=None, boxes=None)

    _hide_module(monkeypatch, "rapidocr_onnxruntime")
    _install_fake_module(monkeypatch, "rapidocr", {"RapidOCR": _V3RapidOCR})

    c = _make_compressor()
    assert c._ocr_extract(b"\x89PNG fake") is None


def test_ocr_extract_v3_mismatched_lengths_logs_and_returns_none(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class _V3RapidOCR:
        def __init__(self) -> None: ...
        def __call__(self, _image_data: bytes) -> Any:
            return SimpleNamespace(txts=["a", "b", "c"], scores=[0.9, 0.8], boxes=[None] * 3)

    _hide_module(monkeypatch, "rapidocr_onnxruntime")
    _install_fake_module(monkeypatch, "rapidocr", {"RapidOCR": _V3RapidOCR})

    c = _make_compressor()
    with caplog.at_level("WARNING", logger="headroom.image.compressor"):
        result = c._ocr_extract(b"\x89PNG fake")
    assert result is None
    assert any("event=ocr_unknown_api_shape" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Backend missing
# ---------------------------------------------------------------------------


def test_ocr_extract_returns_none_when_no_backend_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hide_module(monkeypatch, "rapidocr_onnxruntime")
    _hide_module(monkeypatch, "rapidocr")

    c = _make_compressor()
    assert c._ocr_extract(b"\x89PNG fake") is None
