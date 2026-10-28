"""compress() must not report a previous request's result for a request it did not touch (#4340).

The per-worker ImageCompressor is a singleton (proxy/image_isolation.py reuses one
instance across requests) and _compress_messages_worker reports last_result whenever
it is not None. Before the fix, compress() left last_result set from the previous call
on its early-return paths, so a request whose images were never touched inherited the
previous request's technique and savings and was marked image_compression-mutated.
"""

from __future__ import annotations

from headroom.image.compressor import CompressionResult, ImageCompressor, Technique


def _stale() -> CompressionResult:
    return CompressionResult(
        technique=Technique.PRESERVE,
        original_tokens=765,
        compressed_tokens=2295,
        confidence=1.0,
    )


def test_compress_clears_last_result_when_there_are_no_images() -> None:
    compressor = ImageCompressor()
    compressor.last_result = _stale()

    messages = [{"role": "user", "content": "plain text, no images"}]
    out = compressor.compress(messages)

    assert out == messages
    assert compressor.last_result is None


def test_compress_clears_last_result_for_url_only_images() -> None:
    # A URL image (no inline base64 data) reaches the "no image_data" early return;
    # it must not report the previous request's result either.
    compressor = ImageCompressor()
    compressor.last_result = _stale()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
            ],
        }
    ]
    compressor.compress(messages)

    assert compressor.last_result is None
