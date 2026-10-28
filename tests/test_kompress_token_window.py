"""Words the model never saw whole must not be dropped.

A chunk is 350 words, but the tokenizer truncates at 512 tokens. URLs, code and
citation markers take several tokens per word, so the tail of a dense chunk
never reached the model and every word in it was dropped, even when the model
would have kept it. The cut can also land inside a word, leaving the model a
prefix to judge.
"""

from __future__ import annotations

import numpy as np
import pytest

from headroom.transforms import kompress_compressor as kc
from headroom.transforms.kompress_compressor import KompressCompressor, KompressConfig

WINDOW = 7  # stands in for 512 tokens
CONTENT = " ".join(f"w{i}" for i in range(25))


class _Enc(dict):
    def word_ids(self, batch_index=0):
        return self["_word_ids"][batch_index]


class _Tok:
    """Gives each word ``tokens_per_word(word)`` tokens and cuts each row at WINDOW."""

    def __init__(self, tokens_per_word):
        self.tokens_per_word = tokens_per_word

    def __call__(self, chunk_words, **kw):
        batch = chunk_words if chunk_words and isinstance(chunk_words[0], list) else [chunk_words]
        rows = [
            [i for i, w in enumerate(words) for _ in range(self.tokens_per_word(w))][:WINDOW]
            for words in batch
        ]
        width = max(len(row) for row in rows)
        return _Enc(
            input_ids=np.zeros((len(rows), width), dtype=np.int64),
            attention_mask=np.array([[1] * len(row) + [0] * (width - len(row)) for row in rows]),
            _word_ids=[row + [None] * (width - len(row)) for row in rows],
        )


class _KeepAllModel:
    """Keeps every token it is shown."""

    def get_keep_mask(self, input_ids, attention_mask):
        return [[True] * len(row) for row in input_ids]

    def get_scores(self, input_ids, attention_mask):
        return [[1.0] * len(row) for row in input_ids]


class _RejectsLastTokenModel:
    """Rejects the last token of each row and keeps the rest.

    With two tokens per word a whole word survives on its first token. Only a
    word the cut split, present as a single trailing token, is rejected.
    """

    @staticmethod
    def _keep(mask):
        n = int(mask.sum())
        return [i < n - 1 for i in range(len(mask))]

    def get_keep_mask(self, input_ids, attention_mask):
        return [self._keep(mask) for mask in attention_mask]

    def get_scores(self, input_ids, attention_mask):
        return [[1.0 if k else 0.0 for k in self._keep(mask)] for mask in attention_mask]


class _PrefixOutscoresModel:
    """Scores a row's last token highest, its first word next, the rest low.

    A split word shows up as that last token. Under a forced ratio its prefix
    score would take the one slot a whole word earned.
    """

    def get_keep_mask(self, input_ids, attention_mask):
        return [[s > 0.5 for s in row] for row in self.get_scores(input_ids, attention_mask)]

    def get_scores(self, input_ids, attention_mask):
        rows = []
        for mask in attention_mask:
            n = int(mask.sum())
            rows.append([1.0 if i == n - 1 else 0.9 if i < 2 else 0.1 for i in range(len(mask))])
        return rows


def _compress(monkeypatch, path, tokenizer, model, **kwargs):
    """Run CONTENT through one path with the fake model and tokenizer."""
    monkeypatch.setattr(kc, "_load_kompress", lambda *a, **k: (model, tokenizer, "onnx"))
    # The compressor must agree with the fake on where the window ends.
    monkeypatch.setattr(kc, "_KOMPRESS_MAX_TOKENS", WINDOW)
    monkeypatch.setattr(kc, "_model_device_type", lambda *a, **k: "cpu")
    monkeypatch.setenv(kc._KOMPRESS_MUST_KEEP_ENV, "0")
    compressor = KompressCompressor(
        KompressConfig(enable_ccr=False, min_input_words=10, chunk_words=10)
    )
    if path == "compress":
        monkeypatch.setattr(compressor, "_should_batch_single_content", lambda *a, **k: False)
        return compressor.compress(CONTENT, **kwargs)
    monkeypatch.setattr(compressor, "_should_use_sequential_fallback", lambda: False)
    [result] = compressor.compress_batch([CONTENT], batch_size=8, **kwargs)
    return result


CASES = {
    "cut between words": (_Tok(lambda w: 1), _KeepAllModel()),
    "cut inside a word": (_Tok(lambda w: 2), _RejectsLastTokenModel()),
    "word longer than the window": (
        _Tok(lambda w: WINDOW + 2 if w == "w5" else 2),
        _RejectsLastTokenModel(),
    ),
}


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("path", ["compress", "compress_batch"])
def test_words_the_model_did_not_see_whole_are_kept(monkeypatch, path, case):
    result = _compress(monkeypatch, path, *CASES[case])

    # The model keeps every word it sees whole, so nothing may go missing.
    assert result.compressed.split() == CONTENT.split()


# Two tokens per word and a seven-token window, ratio 0.34 keeps one word of
# each scored group. The single path advances three words at a time and keeps
# the first of each group; the batch path keeps the first of each ten-word
# chunk plus the seven it never saw whole.
RATIO_EXPECTED = {
    "compress": ["w0", "w3", "w6", "w9", "w12", "w15", "w18", "w21", "w24"],
    "compress_batch": [
        *["w0", "w3", "w4", "w5", "w6", "w7", "w8", "w9"],
        *["w10", "w13", "w14", "w15", "w16", "w17", "w18", "w19"],
        *["w20", "w23", "w24"],
    ],
}


@pytest.mark.parametrize("path", ["compress", "compress_batch"])
def test_a_split_word_takes_no_slot_under_a_forced_ratio(monkeypatch, path):
    result = _compress(
        monkeypatch, path, _Tok(lambda w: 2), _PrefixOutscoresModel(), target_ratio=0.34
    )

    assert result.compressed.split() == RATIO_EXPECTED[path]


def test_row_token_counts_tell_padding_from_special_tokens():
    # [CLS] w0 w0 w1 [SEP] and [CLS] w0 [SEP] [PAD] [PAD]: the longest row has no
    # padding, so its trailing run of None is the special-token count.
    rows = [[None, 0, 0, 1, None], [None, 0, None, None, None]]

    assert kc._row_token_counts(rows) == [5, 3]
