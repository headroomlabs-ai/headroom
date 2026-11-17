"""Live validation of Mechanism B's cache behaviour against the real API.

Tested empirically:

1. Request A holds a fresh Read verbatim and forwards the client's tail
   breakpoint on it → the provider's cache_creation includes the read
   content (the held Read is cached with its turn).
2. Request B (one turn later, file quiet, but the Read is now inside the
   provider-confirmed prefix) → the Read is NOT matured, and the provider
   reports a cache READ covering request A's cached prefix — nothing was
   busted.

Skipped without ANTHROPIC_API_KEY. Costs ~15K haiku tokens per run.
"""

from __future__ import annotations

import os

import httpx
import pytest

from headroom.config import ReadMaturationConfig
from headroom.transforms.read_maturation import ReadMaturationManager

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)

MODEL = "claude-haiku-4-5-20251001"
API_URL = "https://api.anthropic.com/v1/messages"

# System pad: must clear the model's minimum cacheable prefix (haiku:
# 2048 tokens) on its own, so request A caches system+early messages.
SYSTEM_PAD = (
    "You are a coding assistant. Policy clause %d: always be precise and "
    "verify against the source before answering. " * 400
) % tuple(range(400))

# Read content: big enough to dominate the message tokens (~4K tokens),
# so its presence/absence in cache numbers is unambiguous.
FILE_CONTENT = "".join(
    f"   {i}\tdef func_{i}(): return {i}  # padding comment line {i}\n" for i in range(700)
)

READ_TOOL = {
    "name": "Read",
    "description": "Read a file",
    "input_schema": {
        "type": "object",
        "properties": {"file_path": {"type": "string"}},
        "required": ["file_path"],
    },
}


def call(messages: list[dict]) -> dict:
    resp = httpx.post(
        API_URL,
        json={
            "model": MODEL,
            "max_tokens": 50,
            "system": [
                {"type": "text", "text": SYSTEM_PAD, "cache_control": {"type": "ephemeral"}}
            ],
            "tools": [READ_TOOL],
            "messages": messages,
        },
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
        },
        timeout=120,
    )
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:500]}"
    return resp.json()["usage"]


def conv_base() -> list[dict]:
    return [
        {"role": "user", "content": [{"type": "text", "text": "Read /src/pad.py please"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_r1",
                    "name": "Read",
                    "input": {"file_path": "/src/pad.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_r1",
                    "content": FILE_CONTENT,
                    # Claude Code-style tail breakpoint on the newest block.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]


class TestNoBustInvariantLive:
    def test_held_read_is_cached_and_not_rewritten(self):
        mgr = ReadMaturationManager(ReadMaturationConfig(enabled=True, quiesce_turns=1))

        # ── Request A: fresh read → held verbatim, client breakpoint kept.
        msgs_a = conv_base()
        res_a = mgr.apply(msgs_a)
        assert res_a.holding_msg_indices == [2], "fixture must trigger holding"
        fwd_a = res_a.messages
        assert "cache_control" in fwd_a[2]["content"][0]

        usage_a = call(fwd_a)
        created_a = usage_a.get("cache_creation_input_tokens", 0)
        input_a = usage_a.get("input_tokens", 0)
        # The held read (~4K tokens) is in the cache write, not the
        # uncached input: the client's tail breakpoint sits on it.
        assert created_a > 3000, f"held read was not cache-written: {usage_a}"
        assert input_a < 1000, f"held read was sent uncached: {usage_a}"

        # ── Request B: one assistant turn later, file quiet, Read cached.
        msgs_b = [
            *conv_base(),
            {"role": "assistant", "content": [{"type": "text", "text": "Read it."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Thanks. Reply with the single word: done",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
        ]
        # The client breakpoint moved to the new tail; the old read block
        # no longer carries one.
        del msgs_b[2]["content"][0]["cache_control"]

        # Request A cached messages 0-2, so the provider-confirmed prefix
        # now covers the Read: it stays verbatim instead of maturing.
        res_b = mgr.apply(msgs_b, frozen_message_count=3)
        assert res_b.newly_matured == 0, "a cached read must not mature"
        fwd_b = res_b.messages
        assert fwd_b[2]["content"][0]["content"] == FILE_CONTENT

        usage_b = call(fwd_b)
        read_b = usage_b.get("cache_read_input_tokens", 0)

        # Request A's cached prefix, Read included, must still be valid.
        assert read_b >= created_a * 0.9, (
            f"request B read {read_b} cached tokens but request A created "
            f"{created_a}: the cached prefix was busted. A={usage_a} B={usage_b}"
        )
        assert usage_b.get("input_tokens", 0) < 2500, (
            f"request B carried heavy uncached input: {usage_b}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
