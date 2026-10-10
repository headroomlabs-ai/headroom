"""Regression: CCR must store the pre-protection original, not the
``{{HEADROOM_TAG_N}}`` placeholder intermediate, for tag-protected Kompress inputs.

Before the fix, ``ContentRouter._try_ml_compressor`` passed the tag-protected
(placeholdered) text into ``KompressCompressor.compress`` without an original, so
CCR stored the placeholder as the entry's ``original_content``. A later *full*
retrieval then returned ``{{HEADROOM_TAG_N}}`` and the protected block (e.g. a
``<system-reminder>`` instruction) was lost from the retrieval path — even
though the immediate upstream request was correctly restored by ``restore_tags``.

These tests exercise actual CCR storage and retrieval with a deterministic
lightweight model instead of downloading ModernBERT. Single, batch, and the
router's Kompress path must recover the exact protected original.
"""

from __future__ import annotations

from headroom.transforms.content_router import ContentRouter, ContentRouterConfig
from headroom.transforms.kompress_compressor import KompressCompressor, KompressConfig


def _kompress_router() -> ContentRouter:
    return ContentRouter(
        ContentRouterConfig(
            enable_kompress=True,
            enable_code_aware=False,
            enable_smart_crusher=False,
        )
    )


def test_compress_batch_validates_ccr_originals_length():
    """compress_batch rejects a mismatched ``ccr_originals`` length, pinning the
    per-item plumbing the batched/GPU CCR-store path relies on."""
    import pytest

    kc = KompressCompressor.__new__(KompressCompressor)
    with pytest.raises(ValueError):
        kc.compress_batch(
            ["one input"],
            ccr_originals=["a", "b"],  # length 2 != 1 input
        )


# 100 filler words: keeping two saves ~99, well past the marker cost gate.
_RAW = "<system-reminder>CRITICAL: invoke the skill</system-reminder> " + " ".join(["filler"] * 100)
_PLACEHOLDER = "{{HEADROOM_TAG_0}} " + " ".join(["filler"] * 100)


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


def _isolated_store_and_model(monkeypatch):
    from headroom.cache.compression_store import CompressionStore
    from headroom.telemetry import TOINConfig, ToolIntelligenceNetwork

    store = CompressionStore(enable_feedback=False)
    toin = ToolIntelligenceNetwork(config=TOINConfig(enabled=False, storage_path=""))
    model = (_FakeModel(), _FakeTokenizer(), "onnx")
    monkeypatch.setattr("headroom.cache.compression_store.get_compression_store", lambda: store)
    monkeypatch.setattr("headroom.telemetry.get_toin", lambda: toin)
    monkeypatch.setattr(
        "headroom.transforms.kompress_compressor._load_kompress", lambda *a, **k: model
    )
    monkeypatch.setattr(
        "headroom.transforms.kompress_compressor._kompress_cache",
        {KompressConfig().model_id: model},
    )
    return store


def test_inline_ccr_retrieval_recovers_protected_original(monkeypatch):
    from headroom.ccr.tool_injection import CCRToolInjector

    store = _isolated_store_and_model(monkeypatch)
    compressor = KompressCompressor(KompressConfig(min_input_words=10))
    result = compressor.compress(_PLACEHOLDER, ccr_original=_RAW)
    injector = CCRToolInjector(
        provider="anthropic", inject_tool=False, inject_system_instructions=False
    )
    injector.scan_for_markers([{"role": "user", "content": result.compressed}])
    assert injector.detected_hashes == [result.cache_key]
    entry = store.retrieve(injector.detected_hashes[0])
    assert entry is not None and entry.original_content == _RAW


def test_batch_ccr_retrieval_recovers_each_protected_original(monkeypatch):
    from headroom.ccr.tool_injection import CCRToolInjector

    store = _isolated_store_and_model(monkeypatch)
    compressor = KompressCompressor(KompressConfig(min_input_words=10))
    monkeypatch.setattr(compressor, "_should_use_sequential_fallback", lambda: False)
    originals = [_RAW, _RAW.replace("CRITICAL", "ESSENTIAL")]
    results = compressor.compress_batch([_PLACEHOLDER, _PLACEHOLDER], ccr_originals=originals)
    recovered = []
    for result in results:
        injector = CCRToolInjector(
            provider="anthropic", inject_tool=False, inject_system_instructions=False
        )
        injector.scan_for_markers([{"role": "user", "content": result.compressed}])
        assert injector.detected_hashes == [result.cache_key]
        entry = store.retrieve(injector.detected_hashes[0])
        assert entry is not None
        recovered.append(entry.original_content)
    assert recovered == originals


def test_router_kompress_ccr_retrieval_recovers_protected_original(monkeypatch):
    from headroom.ccr.tool_injection import CCRToolInjector

    store = _isolated_store_and_model(monkeypatch)
    compressor = KompressCompressor(KompressConfig(min_input_words=10))
    router = _kompress_router()
    monkeypatch.setattr(router, "_get_kompress", lambda: compressor)
    compressed, _ = router._try_ml_compressor(_RAW, context="")
    injector = CCRToolInjector(
        provider="anthropic", inject_tool=False, inject_system_instructions=False
    )
    injector.scan_for_markers([{"role": "user", "content": compressed}])
    entry = store.retrieve(injector.detected_hashes[0])
    assert entry is not None and entry.original_content == _RAW
