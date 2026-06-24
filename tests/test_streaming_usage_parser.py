"""Unit tests for the online SSE usage parser used by the
OpenAI-via-backend streaming path.

These tests pin the per-chunk parsing contract so streaming memory
stays O(1) regardless of stream length — the prior implementation
buffered the entire response just to scan the trailing usage frame.
"""

from __future__ import annotations

from headroom.proxy.handlers.streaming import _parse_completion_tokens_from_sse_chunk


def test_returns_completion_tokens_from_usage_frame() -> None:
    chunk = b'data: {"id":"x","usage":{"prompt_tokens":10,"completion_tokens":42}}\n\n'
    assert _parse_completion_tokens_from_sse_chunk(chunk) == 42


def test_returns_none_for_content_only_chunk() -> None:
    chunk = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
    assert _parse_completion_tokens_from_sse_chunk(chunk) is None


def test_returns_none_for_done_marker() -> None:
    assert _parse_completion_tokens_from_sse_chunk(b"data: [DONE]\n\n") is None


def test_returns_none_for_invalid_json() -> None:
    assert _parse_completion_tokens_from_sse_chunk(b"data: not-json\n\n") is None


def test_returns_none_for_empty_chunk() -> None:
    assert _parse_completion_tokens_from_sse_chunk(b"") is None


def test_handles_chunk_with_multiple_frames() -> None:
    # SSE frames can batch across a single chunk write.
    chunk = (
        b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"b"}}],"usage":{"completion_tokens":7}}\n\n'
    )
    assert _parse_completion_tokens_from_sse_chunk(chunk) == 7


def test_treats_zero_completion_tokens_as_zero_not_none() -> None:
    chunk = b'data: {"usage":{"completion_tokens":0}}\n\n'
    assert _parse_completion_tokens_from_sse_chunk(chunk) == 0


def test_handles_non_dict_data_payload() -> None:
    # Edge case: a JSON array or scalar where a dict was expected.
    chunk = b"data: [1,2,3]\n\n"
    assert _parse_completion_tokens_from_sse_chunk(chunk) is None


def test_handles_invalid_utf8_bytes_without_crashing() -> None:
    # Leading invalid UTF-8 bytes corrupt the "data: " prefix; parser
    # should skip the malformed line and return None rather than raise.
    chunk = b'\xff\xfedata: {"usage":{"completion_tokens":3}}\n\n'
    assert _parse_completion_tokens_from_sse_chunk(chunk) is None
