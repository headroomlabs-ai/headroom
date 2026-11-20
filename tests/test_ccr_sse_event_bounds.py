"""SSE limits apply to individual events, independent of transport read size."""

import pytest

from headroom.cache.compression_store import CompressionStore
from headroom.ccr import egress, stream_splice


@pytest.fixture(params=["egress", "splice"])
def decoder(request, monkeypatch):
    module = egress if request.param == "egress" else stream_splice
    monkeypatch.setattr(module, "MAX_CCR_SSE_EVENT_BYTES", 256)
    if request.param == "egress":
        return egress.CCRMarkerEgressFilter(store=CompressionStore())
    return stream_splice._SSEEventDecoder()


def test_large_read_of_small_complete_events_preserves_every_event(decoder):
    event = b'data: {"text":"valid"}\n\n'
    assert decoder.feed(event * 40) == [event] * 40
    assert decoder.finish() == []


def test_large_read_with_small_unterminated_tail_preserves_tail(decoder):
    event = b'data: {"text":"valid"}\n\n'
    assert decoder.feed(event * 40 + event[:-2]) == [event] * 40
    assert decoder.feed(b"\n\n") == [event]


def test_oversized_complete_event_is_rejected(decoder):
    with pytest.raises(ValueError):
        decoder.feed(b"data: " + b"x" * 257 + b"\n\n")


def test_oversized_unterminated_event_is_rejected_across_reads(decoder):
    assert decoder.feed(b"data: " + b"x" * 100) == []
    with pytest.raises(ValueError):
        decoder.feed(b"x" * 157)
