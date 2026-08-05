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
