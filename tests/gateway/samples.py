"""Deterministic sample payloads shared by the gateway contract tests.

Everything here is a pure function of its arguments so two calls produce
byte-identical JSON - the byte-stability invariants depend on that.
"""

from __future__ import annotations

import json
from typing import Any

# --------------------------------------------------------------------------- #
# Conversations                                                                #
# --------------------------------------------------------------------------- #


def big_tool_history(n_items: int = 200, call_id: str = "c1") -> list[dict[str, Any]]:
    """A chat-completions conversation whose tool result is large enough to be
    compressed (mirrors ``tests/test_compress_session_mode.py::_big_tool_history``)."""
    items = [
        {
            "id": i,
            "score": 0.99 if i % 30 == 0 else 0.6,
            "msg": f"Result {i:03d}{' error' if i % 30 == 0 else ' ok'}",
            "blob": f"payload-{i:04d}-" + "".join(chr(97 + (i * 7 + j) % 26) for j in range(240)),
        }
        for i in range(n_items)
    ]
    return [
        {"role": "user", "content": "Get items"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "get_items", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": json.dumps(items)},
    ]


def anthropic_tool_history(n_items: int = 200) -> list[dict[str, Any]]:
    """The same conversation in Anthropic Messages shape."""
    items = [
        {"id": i, "msg": f"Result {i:03d}", "blob": f"payload-{i:04d}-" + "x" * 200}
        for i in range(n_items)
    ]
    return [
        {"role": "user", "content": "Get items"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_items", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": json.dumps(items)}
            ],
        },
    ]


# --------------------------------------------------------------------------- #
# Tools                                                                        #
# --------------------------------------------------------------------------- #


def _verbose_params(prefix: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    f"The {prefix} query string. This description is deliberately long so "
                    "that the tool schema compaction pass has something to remove. " * 3
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return. " * 4,
                "default": 10,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }


def openai_tools(deferred_prefix: str = "deferred_") -> list[dict[str, Any]]:
    """Chat-completions tools: one core tool plus two deferrable ones.

    ``deferred_prefix`` matches ``redrive_hook_ext.RedriveHook``'s default so a
    test can register the hook with no arguments.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "get_items",
                "description": "Fetch the item list from the catalogue service.",
                "parameters": _verbose_params("catalogue"),
            },
        },
        {
            "type": "function",
            "function": {
                "name": f"{deferred_prefix}create_issue",
                "description": "Create a GitHub issue in the given repository.",
                "parameters": _verbose_params("issue"),
            },
        },
        {
            "type": "function",
            "function": {
                "name": f"{deferred_prefix}query_db",
                "description": "Run a read-only SQL query against Postgres.",
                "parameters": _verbose_params("sql"),
            },
        },
    ]


def anthropic_tools(deferred_prefix: str = "deferred_") -> list[dict[str, Any]]:
    """The same three tools in Anthropic Messages shape."""
    return [
        {
            "name": "get_items",
            "description": "Fetch the item list from the catalogue service.",
            "input_schema": _verbose_params("catalogue"),
        },
        {
            "name": f"{deferred_prefix}create_issue",
            "description": "Create a GitHub issue in the given repository.",
            "input_schema": _verbose_params("issue"),
        },
        {
            "name": f"{deferred_prefix}query_db",
            "description": "Run a read-only SQL query against Postgres.",
            "input_schema": _verbose_params("sql"),
        },
    ]


def tool_names(tools: Any) -> list[str]:
    """Names of every tool in either provider shape (order preserved)."""
    out: list[str] = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            out.append(str(fn["name"]))
        elif t.get("name"):
            out.append(str(t["name"]))
    return out


def canonical(value: Any) -> str:
    """The byte-stability yardstick: sorted-key JSON."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


SYSTEM_PROMPT = "You are a terse assistant. Answer in one sentence."
