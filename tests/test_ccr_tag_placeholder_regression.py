"""Regression: CCR must store the pre-protection original, not the
``{{HEADROOM_TAG_N}}`` placeholder intermediate, for tag-protected Kompress inputs.

Before the fix, ``ContentRouter._try_ml_compressor`` passed the tag-protected
(placeholdered) text into ``KompressCompressor.compress`` without an original, so
CCR stored the placeholder as the entry's ``original_content``. A later *full*
retrieval then returned ``{{HEADROOM_TAG_N}}`` and the protected block (e.g. a
``<system-reminder>`` instruction) was lost from the retrieval path — even
though the immediate upstream request was correctly restored by ``restore_tags``.

These tests pin the contract deterministically without loading the 274MB model:
the router half (it forwards the raw original as ``ccr_original``) and the
compressor half (``compress`` stores ``ccr_original`` rather than the protected
``content``).
"""

from __future__ import annotations

from headroom.transforms.content_router import ContentRouter, ContentRouterConfig
from headroom.transforms.kompress_compressor import KompressCompressor


def _kompress_router() -> ContentRouter:
    return ContentRouter(
        ContentRouterConfig(
            enable_kompress=True,
            enable_code_aware=False,
            enable_smart_crusher=False,
        )
    )


def test_router_forwards_raw_original_as_ccr_original_for_tagged_content(monkeypatch):
    """ContentRouter passes the pre-protection content as ``ccr_original`` while
    the model only ever sees the placeholdered text."""
    router = _kompress_router()
    captured: dict[str, object] = {}

    class RecordingResult:
        compressed = "kept words"
        compressed_tokens = 2

    class RecordingKompress:
        def is_ready(self) -> bool:
            return True

        def ensure_background_load(self) -> None:  # pragma: no cover - guard
            raise AssertionError("must not fetch when the model is already cached")

        def compress(
            self,
            content,
            *,
            context="",
            question=None,
            target_ratio=None,
            allow_download=True,
            ccr_original=None,
        ):
            captured["model_input"] = content
            captured["ccr_original"] = ccr_original
            return RecordingResult()

    monkeypatch.setattr(router, "_get_kompress", lambda: RecordingKompress())

    raw = (
        "<system-reminder>CRITICAL: invoke the skill before responding</system-reminder> "
        + " ".join(["filler"] * 40)
    )
    router._try_ml_compressor(raw, context="")

    # The model only ever sees the tag-protected placeholder.
    assert "{{HEADROOM_TAG" in captured["model_input"]
    assert "CRITICAL" not in captured["model_input"]

    # CCR must receive the raw pre-protection original, so a later full retrieval
    # returns the real <system-reminder> content rather than the placeholder.
    assert captured["ccr_original"] == raw
    assert "{{HEADROOM_TAG" not in str(captured["ccr_original"])


def test_router_omits_ccr_original_when_no_tags(monkeypatch):
    """Untagged content keeps the historic call shape — the kwarg is only added
    when tags were actually protected (backward-compat for direct callers)."""
    router = _kompress_router()
    captured: dict[str, object] = {"called": False}

    class RecordingResult:
        compressed = "kept"
        compressed_tokens = 1

    class RecordingKompress:
        def is_ready(self) -> bool:
            return True

        def ensure_background_load(self) -> None:  # pragma: no cover - guard
            raise AssertionError("must not fetch when ready")

        def compress(
            self,
            content,
            *,
            context="",
            question=None,
            target_ratio=None,
            allow_download=True,
            ccr_original="__unset__",
        ):
            captured["called"] = True
            captured["ccr_original"] = ccr_original
            return RecordingResult()

    monkeypatch.setattr(router, "_get_kompress", lambda: RecordingKompress())

    router._try_ml_compressor(" ".join(["plain"] * 40), context="")

    assert captured["called"] is True
    # Sentinel default untouched => router did not pass the kwarg for untagged text.
    assert captured["ccr_original"] == "__unset__"


def test_compress_batch_validates_ccr_originals_length():
    """compress_batch rejects a mismatched ``ccr_originals`` length, pinning the
    per-item plumbing the batched/GPU CCR-store path relies on."""
    import pytest

    kc = KompressCompressor.__new__(KompressCompressor)
    with pytest.raises(ValueError, match="ccr_originals"):
        kc.compress_batch(
            ["one input"],
            ccr_originals=["a", "b"],  # length 2 != 1 input
        )


# ── compressor store-site coverage ───────────────────────────────────────────
# The two tests above pin the router→compressor boundary and the ``compress_batch``
# API. These drive the *real* ``compress()`` / ``compress_batch()`` all the way to
# the CCR store call and capture the value handed to ``_store_in_ccr`` — directly
# pinning the fixed lines (``ccr_source = ccr_original if ... else content``) so a
# regression that stores the placeholdered ``content`` again would be caught here,
# not just at the boundary. A tiny fake model stands in for the 274MB ModernBERT:
# it keeps the first two words of each chunk so the compression ratio clears the
# ``< 0.8`` CCR-store threshold.

_RAW = "<system-reminder>CRITICAL: invoke the skill</system-reminder> " + " ".join(["filler"] * 40)
_PLACEHOLDER = "{{HEADROOM_TAG_0}} " + " ".join(["filler"] * 40)


class _FakeEncoding:
    def __init__(self, word_lists: list[list[str]]):
        self._word_lists = word_lists
        self._ids = [[0] * len(w) for w in word_lists]

    def __getitem__(self, key):
        return {"input_ids": self._ids, "attention_mask": self._ids}[key]

    def word_ids(self, batch_index: int = 0):
        return list(range(len(self._word_lists[batch_index])))


class _FakeTokenizer:
    def __call__(self, words, **_kwargs):
        # Single-content path passes a flat list[str]; batched passes list[list[str]].
        batch = words if (words and isinstance(words[0], list)) else [words]
        return _FakeEncoding(batch)


class _FakeModel:
    """Keep the first two words of each chunk -> ratio < 0.8 (CCR fires)."""

    def get_keep_mask(self, input_ids, _attention_mask):  # inline compress() path
        return [[i < 2 for i in range(len(input_ids[0]))]]

    def get_scores(self, input_ids, _attention_mask):  # batched compress_batch() path
        return [[1.0 if i < 2 else 0.0 for i in range(len(row))] for row in input_ids]


def _capture_store(compressor, monkeypatch):
    captured: dict[str, object] = {}

    def fake_store(original, compressed, original_tokens):
        captured["original"] = original
        return "fakehash"

    monkeypatch.setattr(compressor, "_store_in_ccr", fake_store)
    monkeypatch.setattr(
        "headroom.transforms.kompress_compressor._load_kompress",
        lambda *a, **k: (_FakeModel(), _FakeTokenizer(), "onnx"),
    )
    return captured


def test_compress_inline_stores_ccr_original_not_placeholder(monkeypatch):
    """The inline ``compress()`` CCR-store stores the raw original, not the
    placeholdered ``content`` the model compressed."""
    compressor = KompressCompressor()
    captured = _capture_store(compressor, monkeypatch)

    compressor.compress(_PLACEHOLDER, ccr_original=_RAW)

    assert captured["original"] == _RAW
    assert "{{HEADROOM_TAG" not in str(captured["original"])


def test_compress_batch_batched_path_stores_ccr_original(monkeypatch):
    """The batched (GPU) ``compress_batch()`` CCR-store path stores the raw
    per-item original. Force the batched branch (ONNX defaults to the sequential
    fallback, which routes through ``compress()`` covered above)."""
    compressor = KompressCompressor()
    captured = _capture_store(compressor, monkeypatch)
    monkeypatch.setattr(compressor, "_should_use_sequential_fallback", lambda: False)

    compressor.compress_batch([_PLACEHOLDER], ccr_originals=[_RAW])

    assert captured["original"] == _RAW
    assert "{{HEADROOM_TAG" not in str(captured["original"])
