"""Regression tests for issue #3099: compression merging words across line
boundaries, and dropping fields from tabular rows inconsistently row to row.
"""

from __future__ import annotations

import pytest

from headroom.transforms import kompress_compressor as kc
from headroom.transforms.kompress_compressor import KompressCompressor, KompressConfig


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


class _KeepIndicesModel:
    """Fake model that keeps exactly the given (global, single-chunk) word indices."""

    def __init__(self, keep_indices: set[int]):
        self._keep = keep_indices

    def get_keep_mask(self, input_ids, attention_mask):
        return [[idx in self._keep for idx in range(len(row))] for row in input_ids]

    def get_scores(self, input_ids, attention_mask):
        return [[1.0 if idx in self._keep else 0.0 for idx in range(len(row))] for row in input_ids]


def _install_fake_kompress(monkeypatch, keep_indices: set[int]) -> None:
    model = _KeepIndicesModel(keep_indices)
    monkeypatch.setattr(kc, "_load_kompress", lambda *a, **k: (model, _Tok(), "onnx"))
    monkeypatch.setattr(kc, "_model_device_type", lambda *a, **k: "cpu")
    monkeypatch.delenv(kc._KOMPRESS_MUST_KEEP_ENV, raising=False)


def _compress(monkeypatch, content: str, path: str):
    compressor = KompressCompressor(KompressConfig(enable_ccr=False))
    if path == "compress":
        monkeypatch.setattr(compressor, "_should_batch_single_content", lambda *a, **k: False)
        return compressor.compress(content)
    monkeypatch.setattr(compressor, "_should_use_sequential_fallback", lambda: False)
    [result] = compressor.compress_batch([content], batch_size=8)
    return result


@pytest.mark.parametrize("path", ["compress", "compress_batch"])
def test_compress_does_not_merge_words_across_lines(monkeypatch, path):
    """Reproduces #3099 bug 1: two commands' output on separate lines.

    The model keeps only the last word of line 1 ("rg") and the first word of
    line 2 ("/opt/homebrew/bin/ast-grep"). Before the fix, these landed adjacent
    in the flat kept-word list and were joined with a space onto one line,
    fabricating "rg /opt/homebrew/bin/ast-grep" -- read literally, that implies
    rg resolves to that path, which is false.
    """
    line1 = "alpha beta gamma delta epsilon zeta eta rg"
    line2 = "/opt/homebrew/bin/ast-grep theta iota kappa lambda mu nu xi"
    content = f"{line1}\n{line2}"

    _install_fake_kompress(monkeypatch, keep_indices={7, 8})

    result = _compress(monkeypatch, content, path)

    assert "rg /opt/homebrew/bin/ast-grep" not in result.compressed
    assert result.compressed.split("\n") == ["rg", "/opt/homebrew/bin/ast-grep"]


@pytest.mark.parametrize("path", ["compress", "compress_batch"])
def test_compress_keeps_tabular_row_intact(monkeypatch, path):
    """Reproduces #3099 bug 2: an `ls -la`-shaped listing.

    The model keeps nothing on its own, so absent row protection only the
    must-keep regex would save permission bits/numbers/filenames -- exactly
    the reported symptom of some fields (the group column, the month)
    disappearing inconsistently. Once a row is part of a >=3-row run of
    consistent fixed-width columns, the whole row must survive intact.
    """
    rows = [
        "-rw-r--r--@  1  user  staff   4921  Aug 12 20:14  daemon.log",
        "-rw-r--r--@  1  user  staff  10485  Aug 12 20:14  other.log",
        "-rw-r--r--@  1  user  staff    512  Aug 12 20:15  third.log",
    ]
    content = "\n".join(rows)

    _install_fake_kompress(monkeypatch, keep_indices=set())

    result = _compress(monkeypatch, content, path)

    expected = "\n".join(" ".join(row.split()) for row in rows)
    assert result.compressed == expected
