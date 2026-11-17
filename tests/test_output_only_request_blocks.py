"""Output-only content blocks must be stripped from request messages.

Anthropic's server-side refusal-fallback feature emits an output-only
``{"type": "fallback", ...}`` block inside an assistant response. It is valid on
the response path but rejected on the request path, so replaying that assistant
turn 400s the whole request. The shared body readers must drop it before
forwarding. See ``strip_output_only_request_blocks`` in ``headroom.proxy.helpers``.
"""

import asyncio
import json

from headroom.proxy.helpers import (
    read_request_json_with_bytes,
    strip_output_only_request_blocks,
)

_FALLBACK = {
    "type": "fallback",
    "from": {"model": "claude-fable-5"},
    "to": {"model": "claude-opus-4-8"},
}


class _FakeHeaders:
    def __init__(self, d=None):
        self._d = {k.lower(): v for k, v in (d or {}).items()}

    def get(self, k, default=None):
        return self._d.get(k.lower(), default)


class _FakeRequest:
    def __init__(self, raw, headers=None):
        self._raw = raw
        self.headers = _FakeHeaders(headers)

    async def body(self):
        return self._raw

    async def stream(self):
        yield self._raw


def _has_fallback(messages):
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "fallback":
                    return True
    return False


def test_strip_removes_fallback_and_backfills_emptied_turn():
    messages = [
        {"role": "user", "content": "hi"},
        # assistant turn that is ONLY a fallback signal (the crash case)
        {"role": "assistant", "content": [dict(_FALLBACK)]},
        # fallback prefix + real content
        {"role": "assistant", "content": [dict(_FALLBACK), {"type": "text", "text": "A."}]},
    ]
    assert strip_output_only_request_blocks(messages) is True
    assert not _has_fallback(messages)
    # emptied turn is backfilled with a single benign text block
    assert messages[1]["content"] == [{"type": "text", "text": "(model fallback)"}]
    # mixed turn keeps only the real content
    assert [b["type"] for b in messages[2]["content"]] == ["text"]
    # idempotent
    assert strip_output_only_request_blocks(messages) is False


def test_strip_is_noop_on_clean_or_invalid_input():
    assert strip_output_only_request_blocks(None) is False
    assert strip_output_only_request_blocks([{"role": "user", "content": "hi"}]) is False
    assert (
        strip_output_only_request_blocks(
            [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
        )
        is False
    )


def test_reader_strips_and_reencodes_raw_bytes():
    body = {
        "model": "claude-fable-5",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [dict(_FALLBACK)]},
        ],
    }
    raw = json.dumps(body).encode("utf-8")
    result, out_raw = asyncio.run(read_request_json_with_bytes(_FakeRequest(raw)))
    assert not _has_fallback(result["messages"])
    # raw bytes re-encoded so byte-faithful passthrough cannot leak the pre-strip body
    assert not _has_fallback(json.loads(out_raw)["messages"])
    assert json.loads(out_raw) == result


def test_reader_leaves_clean_requests_byte_identical():
    raw = json.dumps({"model": "x", "messages": [{"role": "user", "content": "hi"}]}).encode(
        "utf-8"
    )
    _, out_raw = asyncio.run(read_request_json_with_bytes(_FakeRequest(raw)))
    assert out_raw == raw


# ---------------------------------------------------------------------------
# Streaming-only ``index`` keys (2026-09-27 perf audit): both shared body
# readers must canonicalize them in place so the snapshot, the forwarded body,
# and the recorded/replayed prefix are all schema-valid for EVERY provider,
# not just the Anthropic handler. ``index`` is a streaming-RESPONSE field;
# echoing it into request content 400s upstreams that reject extra inputs.
# Idempotent and a no-op for well-formed requests (byte-identical passthrough
# is preserved — the readers only re-encode raw when output-only blocks are
# removed).
# ---------------------------------------------------------------------------


def test_with_bytes_reader_strips_streaming_index_keys():
    body = {
        "model": "claude-fable-5",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "hi", "index": 0},
                    {"type": "tool_use", "id": "t1", "name": "f", "input": {}, "index": 1},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": "ok", "index": 0}],
                        "index": 0,
                    }
                ],
            },
        ],
    }
    raw = json.dumps(body).encode("utf-8")
    result, out_raw = asyncio.run(read_request_json_with_bytes(_FakeRequest(raw)))
    for msg in result["messages"]:
        for block in msg["content"]:
            assert "index" not in block
            if block.get("type") == "tool_result":
                for inner in block.get("content", []):
                    assert "index" not in inner


def test_with_bytes_reader_index_strip_preserves_clean_bytes():
    """No fallback blocks + index stripped: raw bytes stay the original wire bytes."""
    body = {
        "model": "x",
        "messages": [
            {"role": "assistant", "content": [{"type": "text", "text": "hi", "index": 0}]},
        ],
    }
    raw = json.dumps(body).encode("utf-8")
    result, out_raw = asyncio.run(read_request_json_with_bytes(_FakeRequest(raw)))
    assert result["messages"][0]["content"][0] == {"type": "text", "text": "hi"}
    # Dict is stripped; raw is NOT re-encoded (index strip alone must not
    # force a canonical re-serialization — byte-faithful passthrough holds).
    assert out_raw == raw


def test_bytes_less_reader_strips_streaming_index_keys():
    """The bytes-less reader (Gemini + OpenAI chat paths) canonicalizes too."""
    from headroom.proxy.helpers import _read_request_json

    body = {
        "model": "x",
        "messages": [
            {"role": "assistant", "content": [{"type": "text", "text": "hi", "index": 3}]},
        ],
    }
    raw = json.dumps(body).encode("utf-8")
    result = asyncio.run(_read_request_json(_FakeRequest(raw)))
    assert result["messages"][0]["content"][0] == {"type": "text", "text": "hi"}


def test_strip_streaming_index_is_idempotent():
    """The in-place canonicalizer must be safe to re-run (the readers apply it,
    and the Anthropic handler re-runs its private copy downstream)."""
    from headroom.utils import strip_streaming_only_content_fields_in_place

    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "hi", "index": 0},
                {"type": "tool_use", "id": "t1", "name": "f", "input": {}, "index": 1},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": "ok", "index": 0}],
                    "index": 0,
                }
            ],
        },
        {"role": "user", "content": "plain string content"},
    ]
    first = strip_streaming_only_content_fields_in_place(messages)
    second = strip_streaming_only_content_fields_in_place(messages)
    # returns None (in-place); idempotency is asserted by the walk below
    assert first is None and second is None
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                assert "index" not in block
                if isinstance(block.get("content"), list):
                    for inner in block["content"]:
                        assert "index" not in inner
    # non-list input is a safe no-op
    strip_streaming_only_content_fields_in_place(None)
    strip_streaming_only_content_fields_in_place({"not": "a list"})
