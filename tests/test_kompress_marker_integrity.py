from __future__ import annotations

import headroom.transforms.kompress_compressor as kc
from headroom.transforms.kompress_compressor import KompressCompressor, KompressConfig

_HASH = "abcdefabcdefabcdefabcdef"
# Stay above the current production minimum so the test exercises compression
# and marker emission rather than the small-input passthrough.
_CONTENT = " ".join(["ordinary"] * 80)


class _FakeEncoding:
    def __init__(self, rows: list[list[str]]):
        self._rows = rows

    def __getitem__(self, key: str):
        if key in {"input_ids", "attention_mask"}:
            return [[0] * len(row) for row in self._rows]
        raise KeyError(key)

    def word_ids(self, batch_index: int = 0):
        return list(range(len(self._rows[batch_index])))


class _FakeTokenizer:
    def __call__(self, words, **_kwargs):  # noqa: ANN001, ANN202
        rows = words if words and isinstance(words[0], list) else [words]
        return _FakeEncoding(rows)


class _FakeModel:
    def get_keep_mask(self, input_ids, _attention_mask):  # noqa: ANN001, ANN202
        return [[index < 2 for index in range(len(input_ids[0]))]]

    def get_scores(self, input_ids, _attention_mask):  # noqa: ANN001, ANN202
        return [[1.0 if index < 2 else 0.0 for index in range(len(row))] for row in input_ids]


def _compressor(monkeypatch) -> KompressCompressor:  # noqa: ANN001
    monkeypatch.setattr(
        kc,
        "_load_kompress",
        lambda *args, **kwargs: (_FakeModel(), _FakeTokenizer(), "onnx"),
    )
    compressor = KompressCompressor(KompressConfig(enable_ccr=True))
    monkeypatch.setattr(compressor, "_store_in_ccr", lambda *args, **kwargs: _HASH)
    return compressor


def _assert_integrity_marker(compressed: str) -> None:
    assert "Original content is intact" in compressed
    assert f"headroom_retrieve(hash={_HASH})" in compressed


def test_single_compress_marks_original_as_intact(monkeypatch) -> None:
    result = _compressor(monkeypatch).compress(_CONTENT)

    _assert_integrity_marker(result.compressed)


def test_batch_compress_marks_original_as_intact(monkeypatch) -> None:
    compressor = _compressor(monkeypatch)
    monkeypatch.setattr(compressor, "_should_use_sequential_fallback", lambda: False)

    result = compressor.compress_batch([_CONTENT])[0]

    _assert_integrity_marker(result.compressed)
