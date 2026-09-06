from __future__ import annotations

import json

import pytest

from headroom.cache.compression_store import get_compression_store, reset_compression_store
from headroom.ccr.egress import (
    MAX_CCR_SSE_EVENT_BYTES,
    CCRMarkerEgressFilter,
    CCRMarkerEgressOverflow,
    scrub_sse_chunks,
)


@pytest.fixture(autouse=True)
def reset_store():
    reset_compression_store()
    yield
    reset_compression_store()


def _store_entry(original: str) -> str:
    return get_compression_store().store(
        original=original,
        compressed="[]",
        original_item_count=1,
        compressed_item_count=0,
    )


def test_scrubber_holds_split_marker_until_complete_sse_event():
    hash_key = _store_entry("restored payload")
    event = (
        "event: content_block_delta\n"
        + "data: "
        + json.dumps(
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": f"x <<ccr:{hash_key},text,1KB>> y"},
            }
        )
        + "\n\n"
    ).encode()
    marker_offset = event.index(b"<<ccr:") + 8

    output = b"".join(scrub_sse_chunks([event[:marker_offset], event[marker_offset:]]))

    assert b"<<ccr:" not in output
    assert b"compressed content: text,1KB" in output
    assert b"restored payload" not in output
    assert output.endswith(b"\n\n")


def test_scrubber_handles_crlf_and_tool_argument_json_string():
    hash_key = _store_entry("full source")
    argument_delta = json.dumps({"path": "x", "body": f"<<ccr:{hash_key},code,11B>>"})
    event = (
        "event: response.function_call_arguments.delta\r\n"
        + "data: "
        + json.dumps({"type": "response.function_call_arguments.delta", "delta": argument_delta})
        + "\r\n\r\n"
    ).encode()

    output = b"".join(scrub_sse_chunks([event]))
    payload_line = next(line for line in output.splitlines() if line.startswith(b"data:"))
    payload = json.loads(payload_line.removeprefix(b"data:").strip())

    assert "<<ccr:" not in payload["delta"]
    assert json.loads(payload["delta"])["body"] == "full source"
    assert output.endswith(b"\r\n\r\n")


def test_scrubber_fails_closed_for_missing_marker_in_malformed_event():
    chunks = [b"data: not-json <<ccr:deadbeefdead", b"beef,text,1KB>>\n\n"]

    output = b"".join(scrub_sse_chunks(chunks))

    assert b"<<ccr:" not in output
    assert b"compressed content unavailable" in output


def test_scrubber_flushes_truncated_tail_without_marker_leak():
    output = b"".join(scrub_sse_chunks([b'data: {"text":"<<ccr:deadbeefdeadbeef,text,1KB>>"}']))

    assert b"<<ccr:" not in output
    assert b"compressed content unavailable" in output


def test_scrubber_fails_closed_on_unbounded_unterminated_event():
    marker_filter = CCRMarkerEgressFilter()

    with pytest.raises(CCRMarkerEgressOverflow):
        marker_filter.feed(b"x" * (MAX_CCR_SSE_EVENT_BYTES + 1))

    assert marker_filter.finish() == []
