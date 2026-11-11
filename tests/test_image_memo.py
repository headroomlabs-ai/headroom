"""OCR and SigLIP run once per image, not once per turn.

Every turn resends the whole conversation, so each screenshot in history
reached the image stack again on every request: one proxy OCR'd the same
screenshot 114 times in two hours.
"""

from __future__ import annotations

import base64
import sys
import types
from typing import Any

import pytest

from headroom.image import compressor as compressor_module
from headroom.image.compressor import ImageCompressor
from headroom.image.image_types import ImageMemo, ImageSignals, Technique
from headroom.image.onnx_router import OnnxTechniqueRouter


def _image(data: bytes) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(data).decode(),
        },
    }


def test_transcode_ocrs_each_image_once_across_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bytes] = []

    class _RapidOCR:
        def __call__(self, image_data: bytes) -> Any:
            calls.append(image_data)
            return [(None, f"text of {image_data.decode()}", 0.99)], 0.0

    fake = types.ModuleType("rapidocr_onnxruntime")
    fake.RapidOCR = _RapidOCR  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", fake)
    monkeypatch.setattr(compressor_module, "_RESOLVED_OCR", None)

    compressor = ImageCompressor(use_siglip=False)
    turn1 = [{"role": "user", "content": [_image(b"a")]}]
    turn2 = [
        *turn1,
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [_image(b"b")]},
    ]

    first = compressor._apply_compression(turn1, Technique.TRANSCODE, "anthropic")
    second = compressor._apply_compression(turn2, Technique.TRANSCODE, "anthropic")

    assert calls == [b"a", b"b"]
    assert second[0] == first[0]
    assert first[0]["content"] == [{"type": "text", "text": "[OCR from image]\ntext of a"}]
    assert second[2]["content"] == [{"type": "text", "text": "[OCR from image]\ntext of b"}]


def test_past_deadline_ocrs_nothing_new_but_reuses_the_memo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A cold history that would outrun the isolation timeout stops starting
    # OCR at the deadline, so the call returns, the worker and its memo
    # survive, and the next turn picks up where this one stopped.
    calls: list[bytes] = []
    compressor = ImageCompressor(use_siglip=False)
    monkeypatch.setattr(
        compressor, "_ocr_extract", lambda data: calls.append(data) or f"text of {data.decode()}"
    )
    seen = [{"role": "user", "content": [_image(b"a")]}]
    compressor._apply_compression(seen, Technique.TRANSCODE, "anthropic")
    history = [{"role": "user", "content": [_image(b"a"), _image(b"b")]}]
    calls.clear()

    late = compressor._apply_compression(history, Technique.TRANSCODE, "anthropic", deadline=0.0)
    on_time = compressor._apply_compression(history, Technique.TRANSCODE, "anthropic")

    assert late[0]["content"][0] == {"type": "text", "text": "[OCR from image]\ntext of a"}
    assert late[0]["content"][1]["type"] == "image"
    assert on_time[0]["content"][1] == {"type": "text", "text": "[OCR from image]\ntext of b"}
    assert calls == [b"b"]


def test_long_history_reuses_ocr_across_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    # Each transcode turn scans the history oldest-first. With fewer slots
    # than images, every miss evicts the very image the scan reaches next, so
    # nothing was ever reused; 300 screenshots must fit.
    calls: list[bytes] = []
    compressor = ImageCompressor(use_siglip=False)
    monkeypatch.setattr(compressor, "_ocr_extract", lambda data: calls.append(data) or "text")
    images = [_image(str(i).encode()) for i in range(300)]
    history = [{"role": "user", "content": images}]

    for _ in range(2):
        compressor._apply_compression(history, Technique.TRANSCODE, "anthropic")

    assert len(calls) == 300


def test_ocr_engine_failure_is_retried_but_no_text_is_remembered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # "No confident text" is a property of the image; an engine that raised
    # says nothing about it, so remembering that would skip the image for the
    # worker's lifetime.
    calls: list[bytes] = []
    engine_up = False

    class _RapidOCR:
        def __call__(self, image_data: bytes) -> Any:
            calls.append(image_data)
            if not engine_up:
                raise RuntimeError("onnxruntime session lost")
            return ([], 0.0) if image_data == b"blank" else ([(None, "hi", 0.99)], 0.0)

    fake = types.ModuleType("rapidocr_onnxruntime")
    fake.RapidOCR = _RapidOCR  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", fake)
    monkeypatch.setattr(compressor_module, "_RESOLVED_OCR", None)

    compressor = ImageCompressor(use_siglip=False)
    history = [{"role": "user", "content": [_image(b"blank"), _image(b"text")]}]

    down = compressor._apply_compression(history, Technique.TRANSCODE, "anthropic")
    engine_up = True
    up = compressor._apply_compression(history, Technique.TRANSCODE, "anthropic")
    again = compressor._apply_compression(history, Technique.TRANSCODE, "anthropic")

    assert [part["type"] for part in down[0]["content"]] == ["image", "image"]
    assert up[0]["content"][1] == {"type": "text", "text": "[OCR from image]\nhi"}
    assert again == up
    assert calls == [b"blank", b"text", b"blank", b"text"]


def test_classify_analyzes_each_image_once(monkeypatch: pytest.MonkeyPatch) -> None:
    router = OnnxTechniqueRouter()
    analyzed: list[bytes] = []

    def encode(image_data: bytes) -> ImageSignals:
        analyzed.append(image_data)
        if len(analyzed) == 1:
            raise RuntimeError("siglip session lost")
        return ImageSignals(has_text=0.9, is_document=0.1, is_complex=0.1, has_small_details=0.1)

    monkeypatch.setattr(router, "classify_query", lambda query: (Technique.PRESERVE, 0.9))
    monkeypatch.setattr(router, "_load_siglip", lambda: None)
    monkeypatch.setattr(router, "_encode_image", encode)

    # The query changes every turn; the image analysis does not need to. The
    # first analysis of "a" fails, so it is retried rather than remembered.
    turns = ((b"a", "what"), (b"a", "and now?"), (b"a", "and?"), (b"b", "what"))
    for image_data, query in turns:
        router.classify(image_data, query)

    assert analyzed == [b"a", b"a", b"b"]


def test_image_memo_evicts_least_recently_used() -> None:
    memo = ImageMemo[str](max_entries=2)
    computed: list[bytes] = []

    def get(data: bytes) -> str:
        return memo.get(data, lambda: computed.append(data) or data.decode())

    for data in (b"a", b"b", b"a", b"c", b"a", b"b"):
        assert get(data) == data.decode()

    # "a" stayed hot, so "c" evicted "b", and "b" evicted "c".
    assert computed == [b"a", b"b", b"c", b"b"]


def test_image_memo_holds_at_most_max_bytes_of_values() -> None:
    # Entry count alone let a few thousand text-heavy screenshots pin hundreds
    # of MB of OCR text in a worker that outlives the conversation.
    text = "x" * 1000
    memo = ImageMemo[str](max_bytes=3 * sys.getsizeof(text))
    computed: list[bytes] = []

    def get(data: bytes, value: str = text) -> str:
        return memo.get(data, lambda: computed.append(data) or value)

    for data in (b"a", b"b", b"c", b"d", b"b"):
        get(data)
    assert computed == [b"a", b"b", b"c", b"d"]  # "d" evicted "a"; "b" still held
    get(b"a")
    assert computed[-1] == b"a"

    # A value bigger than the whole budget is returned but evicts nothing.
    huge = "y" * 10_000
    assert get(b"huge", huge) == huge
    get(b"huge", huge)
    assert computed[-2:] == [b"huge", b"huge"]
    assert b"b" in memo and b"a" in memo
