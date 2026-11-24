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


def _sse(event_type: str, **fields: object) -> bytes:
    data = json.dumps({"type": event_type, **fields})
    return f"event: {event_type}\ndata: {data}\n\n".encode()


def _data_events(output: list[bytes]) -> list[dict]:
    return [
        json.loads(line.removeprefix(b"data:").strip())
        for chunk in output
        for line in chunk.splitlines()
        if line.startswith(b"data:")
    ]


def test_scrubber_resolves_marker_split_across_anthropic_argument_deltas():
    hash_key = _store_entry("full source")
    marker = f"<<ccr:{hash_key},code,11B>>"
    cut = 9
    delta = lambda text: {"type": "input_json_delta", "partial_json": text}  # noqa: E731
    output = scrub_sse_chunks(
        [
            _sse("content_block_delta", index=1, delta=delta('{"body": "' + marker[:cut])),
            _sse("content_block_delta", index=1, delta=delta(marker[cut:] + '"}')),
            _sse("content_block_stop", index=1),
        ]
    )

    assert b"<<ccr:" not in b"".join(output)
    fragments = "".join(
        event["delta"]["partial_json"]
        for event in _data_events(output)
        if event["type"] == "content_block_delta"
    )
    assert json.loads(fragments) == {"body": "full source"}


def test_scrubber_resolves_marker_split_across_responses_argument_deltas():
    hash_key = _store_entry("full source")
    marker = f"<<ccr:{hash_key},code,11B>>"
    output = scrub_sse_chunks(
        [
            _sse(
                "response.function_call_arguments.delta",
                item_id="fc_1",
                delta='{"body": "<<c',
            ),
            _sse(
                "response.function_call_arguments.delta",
                item_id="fc_1",
                delta=marker[3:] + '"}',
            ),
        ]
    )

    assert b"<<ccr:" not in b"".join(output)
    assert json.loads("".join(e["delta"] for e in _data_events(output))) == {"body": "full source"}


def test_scrubber_flushes_held_argument_tail_before_call_closes():
    output = scrub_sse_chunks(
        [
            _sse(
                "content_block_delta",
                index=0,
                delta={"type": "input_json_delta", "partial_json": '{"a": "<<c'},
            ),
            _sse("content_block_stop", index=0),
        ]
    )

    events = _data_events(output)
    assert [event["type"] for event in events] == [
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
    ]
    assert "".join(e["delta"]["partial_json"] for e in events[:2]) == '{"a": "<<c'


def test_scrubber_fails_closed_on_unbounded_unterminated_event():
    marker_filter = CCRMarkerEgressFilter()

    with pytest.raises(CCRMarkerEgressOverflow):
        marker_filter.feed(b"x" * (MAX_CCR_SSE_EVENT_BYTES + 1))

    assert marker_filter.finish() == []
