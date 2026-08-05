"""Live validation of Mechanism B's no-bust invariant against the real API.

The design's central claim, tested empirically:

1. Request A holds a fresh Read out of the cache (breakpoint relocated
   to just before it) → the provider's cache_creation must NOT include
   the read content.
2. Request B (one turn later, read matured into a marker, breakpoint
   back at the tail) → the provider must report a cache READ covering
   request A's cached prefix — proving the prefix survived the read's
   replacement, i.e. nothing was busted.

If assertion 2 fails, breakpoint relocation breaks prefix matching and
the mechanism needs redesign before it is enabled anywhere.

Skipped without ANTHROPIC_API_KEY. Costs ~15K haiku tokens per run.
"""

from __future__ import annotations

import os

import httpx
import pytest

from headroom.config import ReadMaturationConfig
from headroom.transforms.read_maturation import (
    ReadMaturationManager,
    relocate_cache_breakpoint,
)

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
    def test_hold_then_mature_preserves_cache(self):
        mgr = ReadMaturationManager(ReadMaturationConfig(enabled=True, quiesce_turns=1))

        # ── Request A: fresh read → held, breakpoint relocated before it.
        msgs_a = conv_base()
        res_a = mgr.apply(msgs_a)
        assert res_a.holding_msg_indices == [2], "fixture must trigger holding"
        fwd_a = relocate_cache_breakpoint(res_a.messages, res_a.holding_msg_indices)
        assert "cache_control" not in fwd_a[2]["content"][0]

        usage_a = call(fwd_a)
        created_a = usage_a.get("cache_creation_input_tokens", 0)
        input_a = usage_a.get("input_tokens", 0)
        # The held read (~4K tokens of input) must NOT be in the cache
        # write. created_a covers system pad + first two messages only.
        assert created_a > 0, f"prefix did not cache at all: {usage_a}"
        assert input_a > 3000, f"read content missing from input: {usage_a}"
        total_a = created_a + input_a + usage_a.get("cache_read_input_tokens", 0)
        assert created_a < total_a * 0.7, (
            f"cache write covered the held read — hold failed: {usage_a}"
        )

        # ── Request B: one assistant turn later, file quiet → matured.
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

        res_b = mgr.apply(msgs_b)
        assert res_b.newly_matured == 1, "read must mature after quiesce"
        fwd_b = relocate_cache_breakpoint(res_b.messages, res_b.holding_msg_indices)
        marker = fwd_b[2]["content"][0]["content"]
        assert "Retrieve original: hash=" in marker

        usage_b = call(fwd_b)
        read_b = usage_b.get("cache_read_input_tokens", 0)

        # THE invariant: request A's cached prefix must still be valid —
        # the matured read sat outside it, so replacing it busts nothing.
        assert read_b >= created_a * 0.9, (
            f"NO-BUST INVARIANT FAILED: request B read {read_b} cached tokens "
            f"but request A created {created_a} — breakpoint relocation broke "
            f"prefix matching. A={usage_a} B={usage_b}"
        )
        # And the matured form is small: B's uncached input should be far
        # below the read size (marker + two short turns, not 4K tokens).
        assert usage_b.get("input_tokens", 0) < 2500, (
            f"matured request still carried heavy uncached input: {usage_b}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
