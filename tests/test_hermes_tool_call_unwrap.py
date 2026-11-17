"""Tests for Hermes deferred-tool (`tool_call` wrapper) unwrapping.

Hermes Agent loads on-demand tools via a `tool_search`/`tool_describe`/
`tool_call` indirection: on the wire the emitted tool call is named
`tool_call` and the REAL tool name lives in the arguments payload
(`{"name": "...", "arguments": {...}}`). Tool exclusion / protect lists
match on the real name, so `_build_tool_name_map` must unwrap the bridge
or whitelists silently no-op for all deferred tools.

These tests pin the `unwrap_tool_call_name` helper and its integration
into `ContentRouter._build_tool_name_map` (OpenAI + Anthropic paths).
"""

from __future__ import annotations

import json

from headroom.config import (
    DEFAULT_EXCLUDE_TOOLS,
    is_tool_excluded,
    unwrap_tool_call,
    unwrap_tool_call_name,
)
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_unwrap_passthrough_plain_name() -> None:
    assert unwrap_tool_call_name("read_file", '{"path": "/x"}') == "read_file"


def test_unwrap_passthrough_none_arguments() -> None:
    assert unwrap_tool_call_name("tool_call", None) == "tool_call"


def test_unwrap_passthrough_bad_json() -> None:
    assert unwrap_tool_call_name("tool_call", "bad json") == "tool_call"


def test_unwrap_passthrough_missing_name_key() -> None:
    assert unwrap_tool_call_name("tool_call", '{"no_name": true}') == "tool_call"


def test_unwrap_passthrough_empty_name() -> None:
    assert unwrap_tool_call_name("", None) == ""


def test_unwrap_web_search() -> None:
    assert (
        unwrap_tool_call_name("tool_call", '{"name": "web_search", "arguments": {}}')
        == "web_search"
    )


def test_unwrap_read_file() -> None:
    assert (
        unwrap_tool_call_name("tool_call", '{"name": "read_file", "arguments": {"path": "/x"}}')
        == "read_file"
    )


def test_unwrap_mcp_tool() -> None:
    assert (
        unwrap_tool_call_name(
            "tool_call", '{"name": "mcp__codebase_memory__search", "arguments": {}}'
        )
        == "mcp__codebase_memory__search"
    )


def test_unwrap_dict_arguments_form() -> None:
    """Arguments may arrive as a dict (not JSON string) on some paths."""
    assert (
        unwrap_tool_call_name("tool_call", {"name": "search_files", "arguments": {"pattern": "x"}})
        == "search_files"
    )


def test_unwrap_whitelist_activation() -> None:
    """Unwrapped names must activate the DEFAULT_EXCLUDE_TOOLS whitelist."""
    assert is_tool_excluded("web_search", DEFAULT_EXCLUDE_TOOLS) is True
    unwrapped = unwrap_tool_call_name("tool_call", '{"name": "web_search", "arguments": {}}')
    assert is_tool_excluded(unwrapped, DEFAULT_EXCLUDE_TOOLS) is True


# ---------------------------------------------------------------------------
# _build_tool_name_map integration tests
# ---------------------------------------------------------------------------


def _router(exclude_tools: set[str] | None = None) -> ContentRouter:
    config = ContentRouterConfig(
        min_section_tokens=10,
        enable_kompress=False,
        exclude_tools=exclude_tools,
    )
    return ContentRouter(config)


def test_build_tool_name_map_openai_wrapped() -> None:
    """OpenAI-format assistant tool_calls with Hermes tool_call wrapper."""
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_wrapped_1",
                    "type": "function",
                    "function": {
                        "name": "tool_call",
                        "arguments": '{"name": "read_file", "arguments": {"path": "/x"}}',
                    },
                },
                {
                    "id": "call_plain_2",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query": "q"}'},
                },
            ],
        }
    ]
    router = _router()
    mapping = router._build_tool_name_map(messages)
    assert mapping["call_wrapped_1"] == "read_file", (
        "wrapped tool_call must map to the real tool name"
    )
    assert mapping["call_plain_2"] == "web_search", "plain tool names must pass through unchanged"


def test_build_tool_name_map_anthropic_wrapped() -> None:
    """Anthropic-format tool_use blocks with Hermes tool_call wrapper."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_wrapped_1",
                    "name": "tool_call",
                    "input": {"name": "headroom_retrieve", "arguments": {"hash": "abc"}},
                },
                {
                    "type": "tool_use",
                    "id": "toolu_plain_2",
                    "name": "Read",
                    "input": {"file_path": "/x"},
                },
            ],
        }
    ]
    router = _router()
    mapping = router._build_tool_name_map(messages)
    assert mapping["toolu_wrapped_1"] == "headroom_retrieve", (
        "wrapped tool_call must map to the real tool name"
    )
    assert mapping["toolu_plain_2"] == "Read", "plain tool names must pass through unchanged"


def test_build_tool_name_map_wrapped_not_excluded_before_unwrap() -> None:
    """Sanity: without unwrapping, a wrapped tool is NOT excluded.

    This documents the failure mode the fix addresses: `tool_call` is not in
    DEFAULT_EXCLUDE_TOOLS, so a whitelist match would never fire.

    The inner name is `codebase_search` rather than `read_file` purely because
    the second assertion needs a tool that is genuinely absent from the
    defaults, and `read_file` (Cursor's `Read`) has since been added to them on
    purpose. The property under test is the WRAPPER's name not matching, which
    is the first assertion; the inner name is only an example.
    """
    assert is_tool_excluded("tool_call", DEFAULT_EXCLUDE_TOOLS) is False
    assert is_tool_excluded("codebase_search", DEFAULT_EXCLUDE_TOOLS) is False


def test_build_tool_name_map_exclusion_after_unwrap() -> None:
    """Unwrapped names feed is_tool_excluded for whitelist decisions."""
    router = _router(exclude_tools=set(DEFAULT_EXCLUDE_TOOLS) | {"read_file"})
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_rf_1",
                    "type": "function",
                    "function": {
                        "name": "tool_call",
                        "arguments": '{"name": "read_file", "arguments": {"path": "/x"}}',
                    },
                }
            ],
        }
    ]
    mapping = router._build_tool_name_map(messages)
    assert mapping["call_rf_1"] == "read_file"
    assert is_tool_excluded(mapping["call_rf_1"], router.config.exclude_tools or set()) is True


# ---------------------------------------------------------------------------
# Batch shape: {"calls": [{"name", "arguments"}, ...]} (#3837)
#
# Hermes advertises `tool_call(calls)`; the legacy top-level {"name", ...} form
# is only tolerated. Mirrors Hermes' normalize_tool_call_entries.
# ---------------------------------------------------------------------------


def test_unwrap_batch_of_one_dict() -> None:
    args = {"calls": [{"name": "headroom_retrieve", "arguments": {"hash": "abc"}}]}
    assert unwrap_tool_call("tool_call", args) == ("headroom_retrieve", {"hash": "abc"})


def test_unwrap_batch_of_one_json_string() -> None:
    args = '{"calls": [{"name": "read_file", "arguments": {"path": "/x"}}]}'
    assert unwrap_tool_call("tool_call", args) == ("read_file", {"path": "/x"})


def test_unwrap_batch_calls_as_json_string() -> None:
    args = {"calls": '[{"name": "web_search", "arguments": {"query": "q"}}]'}
    assert unwrap_tool_call_name("tool_call", args) == "web_search"


def test_unwrap_batch_calls_as_bare_object() -> None:
    args = {"calls": {"name": "web_search", "arguments": {}}}
    assert unwrap_tool_call_name("tool_call", args) == "web_search"


def test_unwrap_batch_same_name_keeps_wrapper_arguments() -> None:
    args = {
        "calls": [
            {"name": "connectors__gh__search", "arguments": {"q": "a"}},
            {"name": "connectors__gh__search", "arguments": {"q": "b"}},
        ]
    }
    assert unwrap_tool_call("tool_call", args) == ("connectors__gh__search", args)


def test_unwrap_batch_mixed_names_fails_open() -> None:
    args = {
        "calls": [
            {"name": "connectors__gh__search", "arguments": {}},
            {"name": "connectors__jira__search", "arguments": {}},
        ]
    }
    assert unwrap_tool_call("tool_call", args) == ("tool_call", args)


def test_unwrap_batch_malformed_fails_open() -> None:
    for args in (
        {"calls": []},
        {"calls": "not json"},
        {"calls": [{"arguments": {}}]},
        {"calls": ["headroom_retrieve"]},
    ):
        assert unwrap_tool_call("tool_call", args) == ("tool_call", args)


def test_unwrap_legacy_shape_returns_inner_arguments() -> None:
    args = {"name": "read_file", "arguments": {"path": "/x"}}
    assert unwrap_tool_call("tool_call", args) == ("read_file", {"path": "/x"})


def test_unwrap_plain_name_keeps_arguments() -> None:
    assert unwrap_tool_call("Read", {"file_path": "/x"}) == ("Read", {"file_path": "/x"})


def test_build_tool_name_map_batch_shape_both_formats() -> None:
    batch = {"calls": [{"name": "terminal", "arguments": {"command": "cat a.py"}}]}
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_b1",
                    "type": "function",
                    "function": {"name": "tool_call", "arguments": json.dumps(batch)},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_b2", "name": "tool_call", "input": batch}
            ],
        },
    ]
    router = _router()
    mapping = router._build_tool_name_map(messages)
    assert mapping == {"call_b1": "terminal", "toolu_b2": "terminal"}
    # The per-call arguments, not the wrapper envelope, feed read protection
    # and the relevance query.
    assert router._tool_call_commands == {"call_b1": "cat a.py", "toolu_b2": "cat a.py"}
    assert router._tool_call_args["toolu_b2"] == "cat a.py"


def _retrieve_rows_payload() -> str:
    rows = [
        {"id": i, "name": f"item-{i}", "status": "ok" if i % 3 else "fail", "value": i * 7}
        for i in range(300)
    ]
    return json.dumps(rows)


def _anthropic_retrieve_conversation(tool_use: dict, payload: str) -> list[dict]:
    return [
        {"role": "user", "content": "get the data"},
        {"role": "assistant", "content": [tool_use]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use["id"], "content": payload}],
        },
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "continue"},
    ]


def test_batch_wrapped_headroom_retrieve_result_is_not_recompressed() -> None:
    """#3837: a retrieve issued through Hermes' batch bridge must pass through.

    Before the fix the name map resolved to `tool_call`, the ccr_retrieve guard
    never fired, and SmartCrusher re-compressed the retrieved rows back into a
    marker, leaving the original unreachable.
    """
    from headroom.providers import AnthropicProvider
    from headroom.tokenizer import Tokenizer

    tokenizer = Tokenizer(AnthropicProvider().get_token_counter("claude-sonnet-4-5"))
    payload = _retrieve_rows_payload()
    for retrieve_name in ("headroom_retrieve", "mcp_headroom_headroom_retrieve"):
        tool_use = {
            "type": "tool_use",
            "id": "toolu_retrieve",
            "name": "tool_call",
            "input": {"calls": [{"name": retrieve_name, "arguments": {"hash": "abc123"}}]},
        }
        result = ContentRouter(ContentRouterConfig()).apply(
            _anthropic_retrieve_conversation(tool_use, payload), tokenizer
        )
        assert result.messages[2]["content"][0]["content"] == payload, retrieve_name
        assert "router:excluded:ccr_retrieve" in result.transforms_applied, retrieve_name

    # Control: the same payload behind a non-retrieve deferred tool does
    # compress, so the assertion above is not vacuous.
    control = {
        "type": "tool_use",
        "id": "toolu_retrieve",
        "name": "tool_call",
        "input": {"calls": [{"name": "mcp_db_query", "arguments": {"q": "all"}}]},
    }
    result = ContentRouter(ContentRouterConfig()).apply(
        _anthropic_retrieve_conversation(control, payload), tokenizer
    )
    assert result.messages[2]["content"][0]["content"] != payload
