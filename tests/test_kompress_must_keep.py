"""Tests for the must-keep token override in kompress_compressor."""

from __future__ import annotations

import os

from headroom.transforms import kompress_compressor as kc
from headroom.transforms.kompress_compressor import (
    _KOMPRESS_MUST_KEEP_ENV,
    _KOMPRESS_MUST_KEEP_RE,
    KompressCompressor,
    KompressConfig,
)


class _Enc(dict):
    def word_ids(self, batch_index=0):
        return self["_word_ids"][batch_index]


class _Tok:
    def __call__(self, chunk_words, **kw):
        if chunk_words and isinstance(chunk_words[0], list):
            batch_words = chunk_words
        else:
            batch_words = [chunk_words]
        return _Enc(
            input_ids=[[0] * len(words) for words in batch_words],
            attention_mask=[[1] * len(words) for words in batch_words],
            _word_ids=[list(range(len(words))) for words in batch_words],
        )


class _Model:
    def get_keep_mask(self, input_ids, attention_mask):
        return [[idx == 0 for idx, _ in enumerate(row)] for row in input_ids]

    def get_scores(self, input_ids, attention_mask):
        return [[1.0 if idx == 0 else 0.0 for idx, _ in enumerate(row)] for row in input_ids]


def _install_fake_kompress(monkeypatch):
    monkeypatch.setattr(kc, "_load_kompress", lambda *a, **k: (_Model(), _Tok(), "onnx"))
    monkeypatch.setattr(kc, "_model_device_type", lambda *a, **k: "cpu")


class TestMustKeepRegex:
    def test_numbers(self):
        assert _KOMPRESS_MUST_KEEP_RE.search("42")
        assert _KOMPRESS_MUST_KEEP_RE.search("3.14")
        assert _KOMPRESS_MUST_KEEP_RE.search("0x7fff2038")
        assert not _KOMPRESS_MUST_KEEP_RE.search("word0")

    def test_allcaps(self):
        assert _KOMPRESS_MUST_KEEP_RE.search("SIGILL")
        assert _KOMPRESS_MUST_KEEP_RE.search("HTTP")
        assert _KOMPRESS_MUST_KEEP_RE.search("EOF")

    def test_dotted_paths(self):
        assert _KOMPRESS_MUST_KEEP_RE.search("libsystem_kernel.dylib")
        assert _KOMPRESS_MUST_KEEP_RE.search("torch.nn")

    def test_unix_paths(self):
        assert _KOMPRESS_MUST_KEEP_RE.search("/usr/lib/python3")
        assert _KOMPRESS_MUST_KEEP_RE.search("/workspace/ultrawhale")

    def test_extensions(self):
        assert _KOMPRESS_MUST_KEEP_RE.search("model.py")
        assert _KOMPRESS_MUST_KEEP_RE.search("weights.so")

    def test_flags(self):
        assert _KOMPRESS_MUST_KEEP_RE.search("--verbose")
        assert _KOMPRESS_MUST_KEEP_RE.search("-n")

    def test_camelcase(self):
        assert _KOMPRESS_MUST_KEEP_RE.search("IndexError")
        assert _KOMPRESS_MUST_KEEP_RE.search("EXC_BAD_INSTRUCTION")

    def test_plain_words_not_matched(self):
        assert not _KOMPRESS_MUST_KEEP_RE.search("the")
        assert not _KOMPRESS_MUST_KEEP_RE.search("process")
        assert not _KOMPRESS_MUST_KEEP_RE.search("raised")


class TestMustKeepEnvVar:
    def test_env_var_name(self):
        assert _KOMPRESS_MUST_KEEP_ENV == "HEADROOM_KOMPRESS_MUST_KEEP"

    def test_env_var_default_is_enabled(self, monkeypatch):
        monkeypatch.delenv(_KOMPRESS_MUST_KEEP_ENV, raising=False)
        assert os.environ.get(_KOMPRESS_MUST_KEEP_ENV, "1") != "0"

    def test_env_var_can_disable(self, monkeypatch):
        monkeypatch.setenv(_KOMPRESS_MUST_KEEP_ENV, "0")
        assert os.environ.get(_KOMPRESS_MUST_KEEP_ENV, "1") == "0"


class TestMustKeepCompression:
    def test_compress_keeps_must_keep_word_when_model_drops_it(self, monkeypatch):
        _install_fake_kompress(monkeypatch)
        monkeypatch.delenv(_KOMPRESS_MUST_KEEP_ENV, raising=False)

        compressor = KompressCompressor(KompressConfig(enable_ccr=False))
        monkeypatch.setattr(compressor, "_should_batch_single_content", lambda *a, **k: False)

        result = compressor.compress(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa 0x7fff2038 omega"
        )

        assert result.compressed.split() == ["alpha", "0x7fff2038"]

    def test_compress_can_disable_must_keep_override(self, monkeypatch):
        _install_fake_kompress(monkeypatch)
        monkeypatch.setenv(_KOMPRESS_MUST_KEEP_ENV, "0")

        compressor = KompressCompressor(KompressConfig(enable_ccr=False))
        monkeypatch.setattr(compressor, "_should_batch_single_content", lambda *a, **k: False)

        result = compressor.compress(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa 0x7fff2038 omega"
        )

        assert result.compressed.split() == ["alpha"]

    def test_compress_batch_keeps_must_keep_word_when_score_is_low(self, monkeypatch):
        _install_fake_kompress(monkeypatch)
        monkeypatch.delenv(_KOMPRESS_MUST_KEEP_ENV, raising=False)

        compressor = KompressCompressor(KompressConfig(enable_ccr=False))
        monkeypatch.setattr(compressor, "_should_use_sequential_fallback", lambda: False)

        [result] = compressor.compress_batch(
            ["alpha beta gamma delta epsilon zeta eta theta iota kappa 0x7fff2038 omega"],
            batch_size=8,
        )

        assert result.compressed.split() == ["alpha", "0x7fff2038"]
