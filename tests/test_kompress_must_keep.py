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

    def test_boolean_connectives(self):
        assert _KOMPRESS_MUST_KEEP_RE.search("or")
        assert _KOMPRESS_MUST_KEEP_RE.search("and")
        assert _KOMPRESS_MUST_KEEP_RE.search("nor")
        assert _KOMPRESS_MUST_KEEP_RE.search("xor")
        assert _KOMPRESS_MUST_KEEP_RE.search("or,")
        assert _KOMPRESS_MUST_KEEP_RE.search("(and)")
        # Tokens that merely contain a connective are not connectives.
        for word in ("orange", "random", "android", "core", "sort", "concat", "node"):
            assert not _KOMPRESS_MUST_KEEP_RE.search(word), word


class TestBooleanConnectivesCompression:
    """`and`/`or` hold a condition together, so dropping one changes the
    predicate rather than degrading the sentence. Issue #3545 measured 12 of 40
    `or` lost from a repeated Python return line on the released compressor,
    while `not` from the very same line survived because negation is already
    pinned -- the result still reads as a line of Python and nothing marks it
    as altered."""

    CODE = 'return name == "__init__" or not name.startswith("_") alpha beta gamma delta'

    def _compress(self, monkeypatch, env):
        _install_fake_kompress(monkeypatch)
        if env:
            monkeypatch.delenv(_KOMPRESS_MUST_KEEP_ENV, raising=False)
        else:
            monkeypatch.setenv(_KOMPRESS_MUST_KEEP_ENV, "0")
        compressor = KompressCompressor(KompressConfig(enable_ccr=False, min_input_words=10))
        monkeypatch.setattr(compressor, "_should_batch_single_content", lambda *a, **k: False)
        return compressor.compress(self.CODE)

    def test_keeps_both_operands_of_a_negated_disjunction(self, monkeypatch):
        kept = self._compress(monkeypatch, env=True).compressed.split()
        assert "not" in kept
        assert "or" in kept

    def test_batch_path_keeps_connectives_too(self, monkeypatch):
        _install_fake_kompress(monkeypatch)
        monkeypatch.delenv(_KOMPRESS_MUST_KEEP_ENV, raising=False)
        compressor = KompressCompressor(KompressConfig(enable_ccr=False, min_input_words=10))
        monkeypatch.setattr(compressor, "_should_use_sequential_fallback", lambda: False)

        [result] = compressor.compress_batch(
            ["alpha beta gamma delta epsilon zeta eta theta iota kappa one and two or three"],
            batch_size=8,
        )

        kept = result.compressed.split()
        assert "and" in kept
        assert "or" in kept

    def test_disabling_must_keep_leaves_the_connective_to_the_model(self, monkeypatch):
        kept = self._compress(monkeypatch, env=False).compressed.split()
        assert "or" not in kept
        assert "not" not in kept


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

        compressor = KompressCompressor(KompressConfig(enable_ccr=False, min_input_words=10))
        monkeypatch.setattr(compressor, "_should_batch_single_content", lambda *a, **k: False)

        result = compressor.compress(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa 0x7fff2038 omega"
        )

        assert result.compressed.split() == ["alpha", "0x7fff2038"]

    def test_compress_can_disable_must_keep_override(self, monkeypatch):
        _install_fake_kompress(monkeypatch)
        monkeypatch.setenv(_KOMPRESS_MUST_KEEP_ENV, "0")

        compressor = KompressCompressor(KompressConfig(enable_ccr=False, min_input_words=10))
        monkeypatch.setattr(compressor, "_should_batch_single_content", lambda *a, **k: False)

        result = compressor.compress(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa 0x7fff2038 omega"
        )

        assert result.compressed.split() == ["alpha"]

    def test_compress_batch_keeps_must_keep_word_when_score_is_low(self, monkeypatch):
        _install_fake_kompress(monkeypatch)
        monkeypatch.delenv(_KOMPRESS_MUST_KEEP_ENV, raising=False)

        compressor = KompressCompressor(KompressConfig(enable_ccr=False, min_input_words=10))
        monkeypatch.setattr(compressor, "_should_use_sequential_fallback", lambda: False)

        [result] = compressor.compress_batch(
            ["alpha beta gamma delta epsilon zeta eta theta iota kappa 0x7fff2038 omega"],
            batch_size=8,
        )

        assert result.compressed.split() == ["alpha", "0x7fff2038"]


class _TruncatingTok:
    """Tokenizer that covers only the first ``max_words`` words of a chunk.

    Mimics ``truncation=True, max_length=512`` for a chunk whose words are
    token-heavy: words past the budget get no ``word_id`` at all, exactly as
    the real tokenizer reports them.
    """

    def __init__(self, max_words: int):
        self.max_words = max_words

    def __call__(self, chunk_words, **kw):
        batch_words = chunk_words if (chunk_words and isinstance(chunk_words[0], list)) else [chunk_words]
        word_ids = []
        for words in batch_words:
            covered = min(len(words), self.max_words)
            word_ids.append(list(range(covered)))
        return _Enc(
            input_ids=[[0] * len(wids) for wids in word_ids],
            attention_mask=[[1] * len(wids) for wids in word_ids],
            _word_ids=word_ids,
        )


class TestTruncatedWordsCompression:
    """Words past the tokenizer's 512-token cut are kept, not dropped (#4309).

    Chunks are cut by words (``chunk_words``) but tokenized with
    ``truncation=True, max_length=512``. A chunk of token-heavy words (logs,
    IDs, paths, numbers) hits the token limit well before its word budget, so
    the tail of the chunk had no score and was dropped silently from the
    default proxy path (``target_ratio=None``).
    """

    # 13 words clears the compressor's 10-word input gate.
    WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
    # The fake model keeps only the first word of what it was shown.
    KEPT_BY_MODEL = ["alpha"]

    def _compress(self, monkeypatch, max_words):
        monkeypatch.setattr(kc, "_load_kompress", lambda *a, **k: (_Model(), _TruncatingTok(max_words), "onnx"))
        monkeypatch.setattr(kc, "_model_device_type", lambda *a, **k: "cpu")
        monkeypatch.delenv(_KOMPRESS_MUST_KEEP_ENV, raising=False)
        compressor = KompressCompressor(KompressConfig(enable_ccr=False, min_input_words=10))
        monkeypatch.setattr(compressor, "_should_batch_single_content", lambda *a, **k: False)
        return compressor.compress(self.WORDS).compressed.split()

    def test_words_past_the_truncation_point_survive(self, monkeypatch):
        # The tokenizer covers only the first two words, so every word from
        # the third on was truncated: the model never saw them and they used
        # to vanish from the output entirely.
        kept = self._compress(monkeypatch, max_words=2)
        words = self.WORDS.split()
        # The first word the model saw is kept; the second it dropped; the
        # truncated tail is kept verbatim rather than dropped.
        assert kept == [words[0]] + words[2:]

    def test_fully_covered_chunk_is_unchanged(self, monkeypatch):
        # Nothing is truncated, so the model's own choice stands: it kept the
        # first word and dropped the rest.
        kept = self._compress(monkeypatch, max_words=99)
        assert kept == self.KEPT_BY_MODEL

    def test_batch_path_keeps_truncated_words_too(self, monkeypatch):
        monkeypatch.setattr(kc, "_load_kompress", lambda *a, **k: (_Model(), _TruncatingTok(2), "onnx"))
        monkeypatch.setattr(kc, "_model_device_type", lambda *a, **k: "cpu")
        monkeypatch.delenv(_KOMPRESS_MUST_KEEP_ENV, raising=False)
        compressor = KompressCompressor(KompressConfig(enable_ccr=False, min_input_words=10))
        monkeypatch.setattr(compressor, "_should_use_sequential_fallback", lambda: False)

        [result] = compressor.compress_batch([self.WORDS], batch_size=8)

        words = self.WORDS.split()
        assert result.compressed.split() == [words[0]] + words[2:]
