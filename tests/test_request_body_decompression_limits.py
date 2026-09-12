"""Decompressed request bodies must obey the same size ceiling as plain ones.

The handlers gate on ``Content-Length`` — the *compressed* wire size — so a
high-ratio body bought unlimited capacity: ~2MB of zeros expands to 2GB and the
process dies allocating it, from a single unauthenticated request (#3284).

The property under test is not merely "an oversized body is rejected" but
"it is rejected *without being materialized first*" — a guard that checks the
size after a one-shot ``decompress()`` has already OOM'd. So the memory tests
below assert on peak allocation, not just on the exception.

Bombs are built by streaming a compressor, so the fixtures themselves stay
small; the cap is monkeypatched down so the tests neither allocate nor scan
100MB to prove the point.
"""

from __future__ import annotations

import gzip
import io
import tracemalloc
import zlib

import pytest


def _helpers():
    """Resolve the helpers module at call time, not at import time.

    Other suites swap ``headroom.proxy`` in and out of ``sys.modules`` for their
    own isolation, which can leave two live copies of this module — and two
    distinct ``RequestBodyTooLarge`` classes, so a class captured at import time
    stops matching the one actually raised. Production never reimports this way,
    and callers catch the builtin ``ValueError`` regardless, so this is a test
    concern only. Resolving through ``sys.modules`` keeps the helper and the
    exception it raises on the same module object.
    """
    import importlib

    return importlib.import_module("headroom.proxy.helpers")


MB = 1024 * 1024
PAYLOAD = b'{"model":"claude-sonnet-5","messages":[{"role":"user","content":"hi"}]}'

# Expands to 64MB. Every test that uses it caps well below that, so the refusal
# has to happen mid-stream rather than at the end.
BOMB_PLAIN_SIZE = 64 * MB
SMALL_CAP = 1 * MB


def _gzip_bomb(total: int = BOMB_PLAIN_SIZE) -> bytes:
    """A gzip stream expanding to ``total`` bytes, built without holding them."""
    compressor = zlib.compressobj(9, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
    buf = io.BytesIO()
    chunk = b"\x00" * MB
    for _ in range(total // MB):
        buf.write(compressor.compress(chunk))
    buf.write(compressor.flush())
    return buf.getvalue()


def _deflate_bomb(total: int = BOMB_PLAIN_SIZE) -> bytes:
    compressor = zlib.compressobj(9)
    buf = io.BytesIO()
    chunk = b"\x00" * MB
    for _ in range(total // MB):
        buf.write(compressor.compress(chunk))
    buf.write(compressor.flush())
    return buf.getvalue()


class _Request:
    """Minimal stand-in for the Starlette Request the reader actually takes."""

    def __init__(
        self,
        body: bytes,
        content_encoding: str = "",
        *,
        stream_chunk_size: int = 64 * 1024,
    ) -> None:
        self._body = body
        self.headers = {"content-encoding": content_encoding}
        self._stream_chunk_size = stream_chunk_size
        self.chunks_yielded = 0

    async def body(self) -> bytes:
        return self._body

    async def stream(self):
        # Real ASGI servers hand the body over in chunks, not one blob, which
        # is exactly what lets `_read_request_body_bounded` refuse a body
        # before all of it has arrived.
        for start in range(0, len(self._body), self._stream_chunk_size):
            self.chunks_yielded += 1
            yield self._body[start : start + self._stream_chunk_size]


@pytest.fixture
def small_cap(monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(_helpers(), "MAX_DECOMPRESSED_BODY_SIZE", SMALL_CAP)
    return SMALL_CAP


@pytest.fixture
def small_raw_cap(monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(_helpers(), "MAX_REQUEST_BODY_SIZE", SMALL_CAP)
    return SMALL_CAP


# ───────────────────────────── the ceiling itself ──────────────────────────


def test_decompressed_ceiling_matches_the_plain_body_ceiling() -> None:
    """A client must not get more room by arriving compressed."""
    assert _helpers().MAX_DECOMPRESSED_BODY_SIZE == _helpers().MAX_REQUEST_BODY_SIZE


def test_too_large_is_a_value_error() -> None:
    """Every existing call site catches ValueError; the subclass must not escape it."""
    assert issubclass(_helpers().RequestBodyTooLarge, ValueError)


# ───────────────────────────── refusal, per codec ──────────────────────────


def test_gzip_bomb_is_refused(small_cap: int) -> None:
    with pytest.raises(_helpers().RequestBodyTooLarge):
        _helpers()._inflate_bounded(
            _gzip_bomb(), wbits=16 + zlib.MAX_WBITS, label="gzip", multi_member=True
        )


def test_deflate_bomb_is_refused(small_cap: int) -> None:
    with pytest.raises(_helpers().RequestBodyTooLarge):
        _helpers()._inflate_bounded(_deflate_bomb(), wbits=zlib.MAX_WBITS, label="deflate")


def test_zstd_bomb_is_refused(small_cap: int) -> None:
    zstandard = pytest.importorskip("zstandard")
    bomb = zstandard.ZstdCompressor(level=19).compress(b"\x00" * BOMB_PLAIN_SIZE)
    with pytest.raises(_helpers().RequestBodyTooLarge):
        _helpers()._zstd_bounded(bomb)


def test_brotli_bomb_is_refused(small_cap: int) -> None:
    brotli = pytest.importorskip("brotli")
    bomb = brotli.compress(b"\x00" * BOMB_PLAIN_SIZE)
    with pytest.raises(_helpers().RequestBodyTooLarge):
        _helpers()._brotli_bounded(bomb)


# ─────────────────────── refused *before* materializing ────────────────────


@pytest.mark.parametrize(
    ("make_bomb", "call"),
    [
        (
            _gzip_bomb,
            lambda b: _helpers()._inflate_bounded(
                b, wbits=16 + zlib.MAX_WBITS, label="gzip", multi_member=True
            ),
        ),
        (
            _deflate_bomb,
            lambda b: _helpers()._inflate_bounded(b, wbits=zlib.MAX_WBITS, label="deflate"),
        ),
    ],
)
def test_bomb_never_materializes(make_bomb, call, small_cap: int) -> None:
    """Peak allocation stays near the cap, not near the expansion.

    This is the whole fix: a check placed after a one-shot decompress() would
    pass the refusal assertions above while still OOM-ing the process.
    """
    bomb = make_bomb()

    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    try:
        # reset_peak so a peak set by whatever ran before this test cannot be
        # mistaken for the bomb materializing here.
        tracemalloc.reset_peak()
        with pytest.raises(_helpers().RequestBodyTooLarge):
            call(bomb)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not already_tracing:
            tracemalloc.stop()

    # Generous headroom over the 1MB cap; the unbounded path would peak near
    # the 64MB expansion (and higher still, with the copy decompress() makes).
    assert peak < 8 * MB, f"peak {peak / MB:.1f}MB suggests the bomb was materialized"


# ─────────────────────────── ordinary bodies still work ────────────────────


def test_gzip_round_trips() -> None:
    assert (
        _helpers()._inflate_bounded(
            gzip.compress(PAYLOAD), wbits=16 + zlib.MAX_WBITS, label="gzip", multi_member=True
        )
        == PAYLOAD
    )


def test_deflate_round_trips() -> None:
    assert (
        _helpers()._inflate_bounded(zlib.compress(PAYLOAD), wbits=zlib.MAX_WBITS, label="deflate")
        == PAYLOAD
    )


def test_zstd_round_trips() -> None:
    zstandard = pytest.importorskip("zstandard")
    assert _helpers()._zstd_bounded(zstandard.ZstdCompressor().compress(PAYLOAD)) == PAYLOAD


def test_brotli_round_trips() -> None:
    brotli = pytest.importorskip("brotli")
    assert _helpers()._brotli_bounded(brotli.compress(PAYLOAD)) == PAYLOAD


def test_multi_member_gzip_is_still_concatenated() -> None:
    """`gzip.decompress` joins members; a single-member decompressobj would not."""
    stream = gzip.compress(b'{"a":1}') + gzip.compress(b'{"b":2}')

    assert (
        _helpers()._inflate_bounded(
            stream, wbits=16 + zlib.MAX_WBITS, label="gzip", multi_member=True
        )
        == b'{"a":1}{"b":2}'
    )


def test_gzip_trailing_nul_padding_is_tolerated() -> None:
    """Real clients pad gzip bodies with NULs and `gzip.decompress` accepts it.

    A member-restart loop that treats the padding as a fresh member rejects
    bodies the one-shot call happily decoded — a silent compatibility break
    that no bomb test would have caught.
    """
    padded = gzip.compress(PAYLOAD) + b"\x00" * 16

    assert gzip.decompress(padded) == PAYLOAD  # the behavior being matched
    assert (
        _helpers()._inflate_bounded(
            padded, wbits=16 + zlib.MAX_WBITS, label="gzip", multi_member=True
        )
        == PAYLOAD
    )


def test_gzip_padding_between_members_is_skipped() -> None:
    stream = gzip.compress(b'{"a":1}') + b"\x00" * 8 + gzip.compress(b'{"b":2}')

    assert (
        _helpers()._inflate_bounded(
            stream, wbits=16 + zlib.MAX_WBITS, label="gzip", multi_member=True
        )
        == b'{"a":1}{"b":2}'
    )


def test_empty_gzip_body_matches_the_one_shot_call() -> None:
    """`gzip.decompress(b"")` returns b""; refusing it would be a new rejection."""
    assert gzip.decompress(b"") == b""
    assert (
        _helpers()._inflate_bounded(b"", wbits=16 + zlib.MAX_WBITS, label="gzip", multi_member=True)
        == b""
    )


def test_gzip_body_of_only_padding_is_still_rejected() -> None:
    """Tolerating padding must not turn a bodyless run of NULs into success."""
    with pytest.raises(gzip.BadGzipFile):
        gzip.decompress(b"\x00" * 8)
    # zlib.error, not ValueError: only `_read_request_body_bytes` wraps the
    # codec's own error, and it turns this into the same 400 as before.
    with pytest.raises(zlib.error):
        _helpers()._inflate_bounded(
            b"\x00" * 8, wbits=16 + zlib.MAX_WBITS, label="gzip", multi_member=True
        )


def test_truncated_gzip_still_errors() -> None:
    """Losing the one-shot call must not turn a corrupt body into a silent empty one."""
    truncated = gzip.compress(PAYLOAD)[:-5]

    with pytest.raises(ValueError):
        _helpers()._inflate_bounded(
            truncated, wbits=16 + zlib.MAX_WBITS, label="gzip", multi_member=True
        )


# ───────────────── raw body ceiling (the chunked-transfer bypass) ──────────
#
# The handlers only ever see the *compressed*, wire-level size via
# `Content-Length`, and that header is absent on a chunked-transfer-encoding
# request. `_read_request_body_bounded` is the backstop the handlers fall
# through to regardless of whether that header was there (#3326).


async def test_raw_body_over_the_cap_is_refused(small_raw_cap: int) -> None:
    oversized = b"a" * (small_raw_cap + 1)
    with pytest.raises(_helpers().RequestBodyTooLarge):
        await _helpers()._read_request_body_bounded(_Request(oversized))


async def test_raw_body_refused_without_consuming_the_whole_stream(small_raw_cap: int) -> None:
    """The point is refusing early, not just refusing eventually."""
    request = _Request(b"a" * (small_raw_cap * 10), stream_chunk_size=small_raw_cap // 4)
    total_chunks = -(-len(request._body) // request._stream_chunk_size)  # ceil div

    with pytest.raises(_helpers().RequestBodyTooLarge):
        await _helpers()._read_request_body_bounded(request)

    assert request.chunks_yielded < total_chunks


async def test_raw_body_at_or_under_the_cap_is_accepted(small_raw_cap: int) -> None:
    assert await _helpers()._read_request_body_bounded(_Request(b"a" * small_raw_cap)) == (
        b"a" * small_raw_cap
    )


async def test_raw_body_reader_round_trips_an_ordinary_body() -> None:
    assert await _helpers()._read_request_body_bounded(_Request(PAYLOAD)) == PAYLOAD


# ──────────────────────────── through the entry point ──────────────────────


async def test_reader_refuses_a_bomb(small_cap: int) -> None:
    with pytest.raises(_helpers().RequestBodyTooLarge):
        await _helpers()._read_request_body_bytes(_Request(_gzip_bomb(), "gzip"))


async def test_reader_passes_an_ordinary_compressed_body(small_cap: int) -> None:
    assert (
        await _helpers()._read_request_body_bytes(_Request(gzip.compress(PAYLOAD), "gzip"))
        == PAYLOAD
    )


async def test_reader_leaves_uncompressed_bodies_alone(small_cap: int) -> None:
    assert await _helpers()._read_request_body_bytes(_Request(PAYLOAD, "")) == PAYLOAD
    assert await _helpers()._read_request_body_bytes(_Request(PAYLOAD, "identity")) == PAYLOAD


async def test_reader_refuses_an_oversized_plain_body_with_no_content_length(
    small_raw_cap: int,
) -> None:
    """The regression this PR fixes: no `Content-Length` header, no compression.

    Before this fix, `_read_request_body_bytes` called `request.body()`
    directly, so a handler's `Content-Length` precheck was the only gate and a
    chunked request (no `Content-Length` to check) reached this function
    regardless of size.
    """
    oversized = b"a" * (small_raw_cap + 1)
    with pytest.raises(_helpers().RequestBodyTooLarge):
        await _helpers()._read_request_body_bytes(_Request(oversized, ""))


async def test_reader_still_rejects_an_unknown_encoding() -> None:
    with pytest.raises(ValueError, match="Unsupported Content-Encoding"):
        await _helpers()._read_request_body_bytes(_Request(PAYLOAD, "snappy"))
