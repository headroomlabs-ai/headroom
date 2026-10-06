"""Small MCP tool results are forwarded verbatim (HEADROOM_MCP_RESULT_MIN_CHARS)."""

import json

from headroom.tokenizers import get_tokenizer
from headroom.transforms.content_router import ContentRouter

TICKET = json.dumps(
    {
        "key": "BENCH-42",
        "summary": "Discount applied twice at checkout",
        "status": "Open",
        "description": "Customers with a 10% coupon are charged 81% of the price instead of 90%. "
        "The bug is in shop/pricing.py, function apply_discount.",
        "comments": [{"author": f"user{i}", "body": "seen this too " * 3} for i in range(8)],
    }
)


def _messages(tool_name):
    return [
        {"role": "user", "content": "fix BENCH-42"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": tool_name,
                    "input": {"issue_key": "BENCH-42"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": TICKET}],
                }
            ],
        },
        {"role": "assistant", "content": "Reading the ticket."},
        {"role": "user", "content": "go on"},
    ]


def _tool_result(out):
    return out.messages[2]["content"][0]


def _apply(tool_name):
    return ContentRouter().apply(
        _messages(tool_name),
        get_tokenizer("claude-sonnet-5-5"),
        min_tokens_to_compress=10,
        min_chars_for_block_compression=25,
    )


def test_small_mcp_result_is_verbatim(monkeypatch):
    monkeypatch.delenv("HEADROOM_MCP_RESULT_MIN_CHARS", raising=False)
    out = _apply("mcp__jira__get_issue")
    assert _tool_result(out)["content"][0]["text"] == TICKET
    assert "router:mcp_small_result_verbatim" in out.transforms_applied


def test_disabled_floor_lets_the_router_see_it(monkeypatch):
    monkeypatch.setenv("HEADROOM_MCP_RESULT_MIN_CHARS", "0")
    out = _apply("mcp__jira__get_issue")
    assert "router:mcp_small_result_verbatim" not in out.transforms_applied


def test_floor_can_be_disabled(monkeypatch):
    from headroom.transforms import content_router as cr

    monkeypatch.setenv("HEADROOM_MCP_RESULT_MIN_CHARS", "0")
    assert cr._mcp_result_min_chars() == 0
    monkeypatch.setenv("HEADROOM_MCP_RESULT_MIN_CHARS", "junk")
    assert cr._mcp_result_min_chars() == 4000
