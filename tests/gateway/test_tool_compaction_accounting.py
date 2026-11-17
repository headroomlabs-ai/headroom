"""/v1/compress counts the tool-schema compaction it ships.

The gateway compacts tool schemas (``compact_gateway_tools``) and returns the
compacted array in ``body``, but the message counts never see tool bytes. The
chat handlers fold the compaction delta into both endpoints, so
``tokens_saved`` covers it (``tool_schema_savings_policy``: compaction "is
ALREADY inside tokens_saved"); this path must do the same.
"""

from __future__ import annotations

import json
from typing import Any

from headroom.tokenizers import get_tokenizer
from tests.gateway.conftest import compress

MODEL = "claude-sonnet-4-5"


def _annotated_tool(i: int) -> dict[str, Any]:
    # `title` and `examples` are annotation keys schema compaction strips.
    return {
        "name": f"tool_{i}",
        "description": "Look something up.",
        "input_schema": {
            "type": "object",
            "title": f"Tool {i} input, a long title compaction removes",
            "properties": {
                "q": {
                    "type": "string",
                    "title": "Query",
                    "examples": ["first example query", "second example query"],
                }
            },
        },
    }


def _compress(client: Any, tools: list[dict[str, Any]]) -> dict[str, Any]:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": tools,
        "gateway": {"can_redrive": False, "can_relay_response": False},
    }
    resp = compress(client, body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_compaction_is_inside_tokens_saved(make_headroom_client) -> None:
    client = make_headroom_client(optimize=False)
    sent = [_annotated_tool(i) for i in range(3)]
    data = _compress(client, sent)

    assert "tool_schema_compaction" in data["transforms_applied"]
    tok = get_tokenizer(MODEL)
    shipped = data["body"]["tools"]
    removed = tok.count_text(json.dumps(sent, default=str)) - tok.count_text(
        json.dumps(shipped, default=str)
    )
    assert removed > 0
    # optimize=False leaves the messages alone: the whole saving is the tools.
    assert data["tokens_saved"] == removed
    assert data["tokens_before"] - data["tokens_after"] == data["tokens_saved"]


def test_tools_compaction_leaves_alone_count_nothing(make_headroom_client) -> None:
    client = make_headroom_client(optimize=False)
    plain = {"name": "plain", "description": "x", "input_schema": {"type": "object"}}
    data = _compress(client, [plain])

    assert "tool_schema_compaction" not in data["transforms_applied"]
    assert data["tokens_saved"] == 0


def test_compacting_client_deferred_tools_counts_nothing(make_headroom_client) -> None:
    # A tool the client marked defer_loading is not in the billed context.
    client = make_headroom_client(optimize=False)
    deferred = [{**_annotated_tool(i), "defer_loading": True} for i in range(3)]
    data = _compress(client, deferred)

    assert "tool_schema_compaction" in data["transforms_applied"]
    assert data["tokens_saved"] == 0
