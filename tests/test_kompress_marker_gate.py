"""The CCR retrieval marker follows the saving, and the saving must be real.

A lossy Kompress result that shipped without a marker was discarded by the
router's lossy-unrecoverable guard (#1307), so every pass that shrank a block
by less than 20 percent ran the model and saved nothing. The gate now compares
the whole original with the whole candidate-plus-marker in one token unit
(cl100k_base, an estimate for non-OpenAI providers; one token per character
without an encoder) and passes the candidate through unless it is smaller. A
marked result reports that same measurement. Word counts cannot do this: the
marker is 36-45 tokens for 12 words, dropped words can be one token each, and
the word left at the head of the candidate can tokenize differently from its
space-prefixed form in the source.
"""

from __future__ import annotations

import hashlib

import pytest

from headroom.transforms import kompress_compressor as kc
from headroom.transforms.kompress_compressor import (
    KompressCompressor,
    KompressConfig,
    ccr_retrieval_marker,
    payload_tokens,
)


class _Enc(dict):
    def word_ids(self, batch_index=0):
        return self["_word_ids"][batch_index]


class _Tok:
    def __call__(self, chunk_words, **kw):
        batch_words = (
            chunk_words if chunk_words and isinstance(chunk_words[0], list) else [chunk_words]
        )
        return _Enc(
            input_ids=[[0] * len(words) for words in batch_words],
            attention_mask=[[1] * len(words) for words in batch_words],
            _word_ids=[list(range(len(words))) for words in batch_words],
        )


class _DropFirst:
    """Keep every word except the first ``drop`` of each row."""

    def __init__(self, drop: int) -> None:
        self.drop = drop

    def get_keep_mask(self, input_ids, attention_mask):
        return [[idx >= self.drop for idx, _ in enumerate(row)] for row in input_ids]

    def get_scores(self, input_ids, attention_mask):
        return [[0.0 if idx < self.drop else 1.0 for idx, _ in enumerate(row)] for row in input_ids]


def _install(monkeypatch, drop: int) -> None:
    monkeypatch.setattr(kc, "_load_kompress", lambda *a, **k: (_DropFirst(drop), _Tok(), "onnx"))
    monkeypatch.setattr(kc, "_model_device_type", lambda *a, **k: "cpu")


def _prose(n: int) -> str:
    # Plain lowercase words: nothing the must-keep override would pin.
    return " ".join(f"w{chr(97 + i % 26)}{chr(97 + (i // 26) % 26)}" for i in range(n))


def _real_key(source: str) -> str:
    """The CCR store's own key: the SHA-256 prefix of the stored source."""
    return hashlib.sha256(source.encode()).hexdigest()[:24]


def _compressor(monkeypatch, **config) -> KompressCompressor:
    compressor = KompressCompressor(KompressConfig(min_input_words=10, **config))
    monkeypatch.setattr(compressor, "_should_batch_single_content", lambda *a, **k: False)
    monkeypatch.setattr(compressor, "_should_use_sequential_fallback", lambda: False)
    monkeypatch.setattr(compressor, "_store_in_ccr", lambda source, *a, **k: _real_key(source))
    return compressor


# 100 single-token words: the tightest source there is. Dropping 41 saves 41
# tokens, and the marker for it (with its real source-derived hash) costs 43.
SINGLE_TOKEN_SOURCE = " ".join(["alpha"] * 99 + ["nfs"])
# Dropping 36 leaves "bureaucratic" at the head of the candidate, where it
# tokenizes differently from " bureaucratic" in the source: 100 -> 103 with
# the marker, though a marker-only cost against saved words called it 99.
HEAD_WORD_SOURCE = " ".join(["alpha"] * 36 + ["bureaucratic"] + ["alpha"] * 63)


def _tok(text: str) -> int:
    enc = pytest.importorskip("tiktoken").get_encoding("cl100k_base")
    return len(enc.encode(text))


def test_payload_tokens_is_cl100k_or_one_per_character(monkeypatch):
    marker = ccr_retrieval_marker(100, 59, SINGLE_TOKEN_SOURCE, _real_key(SINGLE_TOKEN_SOURCE))
    assert _real_key(SINGLE_TOKEN_SOURCE) == "7dbb8f8de9f3e1d7c6f3a6e1"
    assert payload_tokens(marker) == _tok(marker) == 43
    assert payload_tokens(SINGLE_TOKEN_SOURCE) == _tok(SINGLE_TOKEN_SOURCE) == 100
    monkeypatch.setattr(kc, "_payload_encoder", False)
    assert payload_tokens(marker) == len(marker)


@pytest.mark.parametrize(
    ("source", "drop"),
    [(SINGLE_TOKEN_SOURCE, 41), (HEAD_WORD_SOURCE, 36)],
    ids=["single-token-words", "retokenized-head-word"],
)
def test_candidates_that_do_not_shrink_the_whole_payload_pass_through(monkeypatch, source, drop):
    kept = " ".join(source.split()[drop:])
    marked = kept + ccr_retrieval_marker(100, 100 - drop, source, _real_key(source))
    assert _tok(marked) > _tok(source)  # the reviewer's measurement
    _install(monkeypatch, drop=drop)
    result = _compressor(monkeypatch).compress(source)
    assert result.compressed == source
    assert result.cache_key is None
    assert result.compression_ratio == 1.0
    [batched] = _compressor(monkeypatch).compress_batch([source], batch_size=8)
    assert batched.compressed == source and batched.compression_ratio == 1.0


def test_marked_result_reports_the_whole_payload_measurement(monkeypatch):
    # 300 single-token words, drop 60: 240 kept plus a ~43-token marker is
    # smaller than 300, and the accounting is that measurement, not words.
    source = " ".join(["alpha"] * 299 + ["nfs"])
    _install(monkeypatch, drop=60)
    result = _compressor(monkeypatch).compress(source)
    marked = " ".join(source.split()[60:]) + ccr_retrieval_marker(
        300, 240, source, _real_key(source)
    )
    assert result.compressed == marked
    assert result.cache_key == _real_key(source)
    assert (result.original_tokens, result.compressed_tokens) == (_tok(source), _tok(marked))
    assert result.compression_ratio == _tok(marked) / _tok(source)
    [batched] = _compressor(monkeypatch).compress_batch([source], batch_size=8)
    assert batched.compressed == result.compressed
    assert (batched.original_tokens, batched.compressed_tokens) == (
        result.original_tokens,
        result.compressed_tokens,
    )


def test_no_ccr_mode_ships_unmarked_lossy_unchanged(monkeypatch):
    # Without CCR there is no marker and no whole-payload gate: the deliberate
    # output is the bare lossy result in word counts, on both paths, as before.
    _install(monkeypatch, drop=41)
    compressor = _compressor(monkeypatch, enable_ccr=False)
    result = compressor.compress(SINGLE_TOKEN_SOURCE)
    assert (result.original_tokens, result.compressed_tokens) == (100, 59)
    assert "Retrieve more" not in result.compressed
    [batched] = compressor.compress_batch([SINGLE_TOKEN_SOURCE], batch_size=8)
    assert batched.compressed_tokens == 59
