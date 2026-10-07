from __future__ import annotations

from headroom.ccr.tool_injection import CCRToolInjector
from headroom.proxy.ccr_marker_policy import has_new_ccr_markers


def _hashes(*contents: str) -> list[str]:
    injector = CCRToolInjector(
        provider="anthropic",
        inject_tool=False,
        inject_system_instructions=False,
    )
    injector.scan_for_markers([{"role": "user", "content": content} for content in contents])
    return injector.detected_hashes


def test_has_new_ccr_markers_filters_replayed_forwarded_markers() -> None:
    marker = "[100 items compressed to 10. Retrieve more: hash=abc123def456abc123def456]"

    assert (
        has_new_ccr_markers(
            current_detected_hashes=_hashes(marker),
            previous_forwarded_messages=[{"role": "user", "content": marker}],
            provider="anthropic",
        )
        is False
    )


def test_source_line_span_marker_is_still_detected() -> None:
    # The compressor annotates the count with a source-line span (#2586); the
    # retrieval hash must still be extracted from the enriched marker.
    marker = "[122 items compressed to 27 (from 5 source lines). Retrieve more: hash=c00eb437e5e5c00eb437e5e5]"
    assert _hashes(marker) == ["c00eb437e5e5c00eb437e5e5"]


def test_new_marker_with_integrity_clause_is_detected() -> None:
    # Issue #3098: marker includes "Original content preserved."
    marker = (
        "[122 items compressed to 27 (from 5 source lines). "
        "Original content preserved. Retrieve more: hash=c00eb437e5e5c00eb437e5e5]"
    )
    assert _hashes(marker) == ["c00eb437e5e5c00eb437e5e5"]


class _FakeEncoding:
    def __init__(self, word_lists: list[list[str]]):
        self._word_lists = word_lists
        self._ids = [[0] * len(w) for w in word_lists]

    def __getitem__(self, key: str):
        return {"input_ids": self._ids, "attention_mask": self._ids}[key]

    def word_ids(self, batch_index: int = 0) -> list[int]:
        return list(range(len(self._word_lists[batch_index])))


class _FakeTokenizer:
    def __call__(self, words: list[str] | list[list[str]], **_kwargs) -> _FakeEncoding:
        batch = words if (words and isinstance(words[0], list)) else [words]  # type: ignore[list-item]
        return _FakeEncoding(batch)  # type: ignore[arg-type]


class _FakeModel:
    """Keep the first two words of each chunk -> ratio < 0.8 (CCR fires)."""

    def get_keep_mask(
        self, input_ids: list[list[int]], _attention_mask: object
    ) -> list[list[bool]]:
        return [[i < 2 for i in range(len(input_ids[0]))]]

    def get_scores(self, input_ids: list[list[int]], _attention_mask: object) -> list[list[float]]:
        return [[1.0 if i < 2 else 0.0 for i in range(len(row))] for row in input_ids]


def test_remote_kompress_compressor_emits_original_content_preserved(monkeypatch) -> None:
    from headroom.transforms.kompress_remote import RemoteKompressCompressor

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"compressed": "short text", "compression_ratio": 0.2, "original_tokens": 100}

    class _FakeClient:
        def post(self, url, headers=None, json=None):  # noqa: A002, ANN001
            return _FakeResponse()

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "headroom.transforms.kompress_remote.store_kompress_in_ccr",
        lambda original, compressed, tokens: "cafebabecafebabecafebabe",
    )
    c = RemoteKompressCompressor("https://ml.example.invalid")
    c._client = _FakeClient()
    c.config.enable_ccr = True

    result = c.compress("line one\nline two\nline three " * 20)
    assert (
        "Original content preserved. Retrieve more: hash=cafebabecafebabecafebabe"
        in result.compressed
    )


def test_kompress_compressor_emits_original_content_preserved(monkeypatch) -> None:
    from headroom.transforms.kompress_compressor import KompressCompressor

    compressor = KompressCompressor()
    compressor.config.enable_ccr = True
    monkeypatch.setattr(
        compressor,
        "_store_in_ccr",
        lambda original, compressed, tokens: "cafebabecafebabecafebabe",
    )
    monkeypatch.setattr(
        "headroom.transforms.kompress_compressor._load_kompress",
        lambda *a, **k: (_FakeModel(), _FakeTokenizer(), "onnx"),
    )

    result = compressor.compress("line one\nline two\nline three " * 20)
    assert (
        "Original content preserved. Retrieve more: hash=cafebabecafebabecafebabe"
        in result.compressed
    )


def test_kompress_compressor_batch_emits_original_content_preserved(monkeypatch) -> None:
    from headroom.transforms.kompress_compressor import KompressCompressor

    compressor = KompressCompressor()
    compressor.config.enable_ccr = True
    monkeypatch.setattr(compressor, "_should_use_sequential_fallback", lambda: False)
    monkeypatch.setattr(
        compressor,
        "_store_in_ccr",
        lambda original, compressed, tokens: "cafebabecafebabecafebabe",
    )
    monkeypatch.setattr(
        "headroom.transforms.kompress_compressor._load_kompress",
        lambda *a, **k: (_FakeModel(), _FakeTokenizer(), "onnx"),
    )

    results = compressor.compress_batch(["line one\nline two\nline three " * 20])
    assert len(results) == 1
    assert (
        "Original content preserved. Retrieve more: hash=cafebabecafebabecafebabe"
        in results[0].compressed
    )


def test_has_new_ccr_markers_detects_hash_not_seen_in_previous_forward() -> None:
    old = "[100 items compressed to 10. Retrieve more: hash=abc123def456abc123def456]"
    new = "[50 items compressed to 5. Retrieve more: hash=deadbeefdeadbeefdeadbeef]"

    assert (
        has_new_ccr_markers(
            current_detected_hashes=_hashes(old, new),
            previous_forwarded_messages=[{"role": "user", "content": old}],
            provider="anthropic",
        )
        is True
    )


def test_has_new_ccr_markers_treats_missing_previous_forward_as_new() -> None:
    marker = "[100 items compressed to 10. Retrieve more: hash=abc123def456abc123def456]"

    assert (
        has_new_ccr_markers(
            current_detected_hashes=_hashes(marker),
            previous_forwarded_messages=None,
            provider="anthropic",
        )
        is True
    )


def test_has_new_ccr_markers_returns_false_without_current_hashes() -> None:
    assert (
        has_new_ccr_markers(
            current_detected_hashes=[],
            previous_forwarded_messages=None,
            provider="anthropic",
        )
        is False
    )
