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
    """Tokenizer that models ``truncation=True, max_length=512``.

    Every word costs ``tokens_per_word`` tokens, which is how a chunk of
    token-heavy words (logs, IDs, paths, numbers) runs past the token limit
    before its word budget is used up. Words past the limit get no
    ``word_id`` at all, exactly as the real tokenizer reports them. The cut
    can also land inside a word: that word keeps the tokens that fit and
    the rest of it never reaches the tokenizer.
    """

    def __init__(self, tokens_per_word: int = 1, max_length: int = 512):
        self.tokens_per_word = tokens_per_word
        self.max_length = max_length

    def __call__(self, chunk_words, **kw):
        batch_words = chunk_words if (chunk_words and isinstance(chunk_words[0], list)) else [chunk_words]
        word_ids = []
        for words in batch_words:
            ids: list[int] = []
            used = 0
            for idx in range(len(words)):
                room = self.max_length - used
                if room <= 0:
                    break
                take = min(self.tokens_per_word, room)
                ids.extend([idx] * take)
                used += take
            word_ids.append(ids)
        return _Enc(
            input_ids=[[0] * len(ids) for ids in word_ids],
            attention_mask=[[1] * len(ids) for ids in word_ids],
            _word_ids=word_ids,
        )


class TestTruncatedWordsCompression:
    """Words past the tokenizer's 512-token cut are kept, not dropped (#4309).

    Chunks are cut by words (``chunk_words``) but tokenized with
    ``truncation=True, max_length=512``. A chunk of token-heavy words hits
    the token limit well before its word budget, so the tail of the chunk had
    no score and was dropped silently from the default proxy path
    (``target_ratio=None``).
    """

    # One chunk is `chunk_words` words long, so 350 words is exactly one.
    CHUNK_WORDS = 350

    def _words(self, count: int) -> list[str]:
        return [f"word{i}" for i in range(count)]

    def _install(self, monkeypatch, tokens_per_word):
        monkeypatch.setattr(
            kc,
            "_load_kompress",
            lambda *a, **k: (_Model(), _TruncatingTok(tokens_per_word), "onnx"),
        )
        monkeypatch.setattr(kc, "_model_device_type", lambda *a, **k: "cpu")
        monkeypatch.delenv(_KOMPRESS_MUST_KEEP_ENV, raising=False)

    def _compress(self, monkeypatch, tokens_per_word):
        self._install(monkeypatch, tokens_per_word)
        words = self._words(self.CHUNK_WORDS)
        compressor = KompressCompressor(KompressConfig(enable_ccr=False, min_input_words=10))
        monkeypatch.setattr(compressor, "_should_batch_single_content", lambda *a, **k: False)
        kept = compressor.compress(" ".join(words)).compressed.split()
        return kept, words

    def test_words_past_the_truncation_point_survive(self, monkeypatch):
        # Two tokens per word: the 350-word chunk needs 700 tokens, so the
        # tokenizer cuts it at 512. The 256 words it covered were scored, the
        # other 94 were never seen and used to vanish from the output.
        kept, words = self._compress(monkeypatch, tokens_per_word=2)
        # 256 words were covered; the cut landed at a word boundary here, but
        # the last covered word is kept too, because the cut may have fallen
        # inside it.
        assert kept == [words[0]] + words[255:]

    def test_the_word_the_cut_landed_in_is_kept(self, monkeypatch):
        # Three tokens per word: the cut at 512 falls inside word 170, whose
        # last token never reached the tokenizer. The model only saw part of
        # that word, so dropping it would take its unseen suffix with it.
        kept, words = self._compress(monkeypatch, tokens_per_word=3)
        assert kept == [words[0]] + words[170:]

    def test_fully_covered_chunk_is_unchanged(self, monkeypatch):
        # One token per word: the chunk fits in 512 tokens, so nothing was
        # truncated and the model's own choice stands. It kept the first word
        # and dropped the rest.
        kept, words = self._compress(monkeypatch, tokens_per_word=1)
        assert kept == [words[0]]

    def test_batch_path_keeps_truncated_words_too(self, monkeypatch):
        self._install(monkeypatch, 2)
        words = self._words(self.CHUNK_WORDS)
        compressor = KompressCompressor(KompressConfig(enable_ccr=False, min_input_words=10))
        monkeypatch.setattr(compressor, "_should_use_sequential_fallback", lambda: False)

        [result] = compressor.compress_batch([" ".join(words)], batch_size=8)

        assert result.compressed.split() == [words[0]] + words[255:]

    def test_a_forced_ratio_still_gets_its_budget(self, monkeypatch):
        # A forced ratio is a hard budget the caller asked for, so the
        # unscored tail is not added back on that path.
        self._install(monkeypatch, 2)
        words = self._words(self.CHUNK_WORDS)
        compressor = KompressCompressor(KompressConfig(enable_ccr=False, min_input_words=10))
        monkeypatch.setattr(compressor, "_should_batch_single_content", lambda *a, **k: False)

        kept = compressor.compress(" ".join(words), target_ratio=0.5).compressed.split()

        # The ratio applied to the 256 scored words: the first word (the only
        # one the model scored) plus the next 127. The unscored tail is not
        # added back, so the budget the caller asked for still holds.
        assert len(kept) == 128
        assert kept[0] == words[0]
        assert words[255] not in kept
