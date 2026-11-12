"""Literal ``cache_control`` fields inside model-visible payloads (#4272).

``cache_control`` is overloaded: a transport prompt-cache directive on content
blocks, and ordinary application data inside tool inputs, JSON Schema bodies and
their literal values. Stripping it everywhere made two requests that differ in
model-visible data share one key, so the second replayed the first's response.
"""

from __future__ import annotations

import copy

import pytest

from headroom.proxy.semantic_cache import SemanticCache
from headroom.proxy.semantic_cache_key_policy import (
    compute_semantic_cache_key,
    strip_cache_control,
)

MODEL = "fixture"
PARTITION = "caller"
RESPONSE = b'{"id":"first-response","choices":[{"message":{"content":"first"}}]}'

# Fields whose value Headroom may legitimately receive as a decoded dict and
# where ``cache_control`` is application data, not a transport directive. A JSON
# *string* value (what OpenAI actually sends for ``arguments``) was never at risk:
# the stripper leaves strings alone at any depth.
LITERAL_DICT_LOCATIONS = (
    "tool_input",  # Anthropic tool_use input
    "tool_arguments_dict",  # OpenAI function.arguments, decoded to a dict
    "tool_schema",  # Anthropic input_schema default
    "response_schema",  # OpenAI response_format json_schema const
)


def _payloads(location: str) -> tuple[list[dict], dict]:
    """(messages, key_fields) holding a literal ``cache_control`` at ``location``."""
    messages: list[dict] = [{"role": "user", "content": "Configure the application."}]
    fields: dict = {}
    if location == "tool_input":
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call1",
                        "name": "configure",
                        "input": {"cache_control": "first"},
                    }
                ],
            }
        ]
    elif location == "tool_arguments_dict":
        # OpenAI sends ``arguments`` as a JSON *string*, which the old stripper
        # left alone (see the string guard below). Headroom also tolerates a
        # decoded dict there -- ``handlers/openai.py`` parses either shape -- and
        # for that shape the old stripper did erase the field, so this is the
        # shape that demonstrates the OpenAI-path regression.
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call1",
                        "type": "function",
                        "function": {"name": "configure", "arguments": {"cache_control": "first"}},
                    }
                ],
            }
        ]
    elif location == "tool_schema":
        fields = {
            "tools": [
                {
                    "name": "configure",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "object", "default": {"cache_control": "first"}}
                        },
                    },
                }
            ]
        }
    elif location == "response_schema":
        fields = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": {"const": {"cache_control": "first"}}},
            }
        }
    else:  # pragma: no cover - guards the parametrization
        raise AssertionError(f"unknown location {location}")
    return messages, fields


def _with_changed_value(messages: list[dict], fields: dict, location: str) -> tuple[list, dict]:
    messages, fields = copy.deepcopy((messages, fields))
    if location == "tool_input":
        messages[0]["content"][0]["input"]["cache_control"] = "second"
    elif location == "tool_arguments_dict":
        messages[0]["tool_calls"][0]["function"]["arguments"]["cache_control"] = "second"
    elif location == "tool_schema":
        fields["tools"][0]["input_schema"]["properties"]["value"]["default"]["cache_control"] = (
            "second"
        )
    else:
        fields["response_format"]["json_schema"]["schema"]["const"]["cache_control"] = "second"
    return messages, fields


@pytest.mark.parametrize("location", LITERAL_DICT_LOCATIONS)
@pytest.mark.asyncio
async def test_literal_cache_control_does_not_alias_cached_responses(location: str) -> None:
    messages, fields = _payloads(location)
    changed_messages, changed_fields = _with_changed_value(messages, fields, location)

    cache = SemanticCache()
    await cache.set(messages, MODEL, RESPONSE, {}, partition=PARTITION, **fields)

    # The changed request must miss: its tool data / schema really differs.
    assert await cache.get(changed_messages, MODEL, partition=PARTITION, **changed_fields) is None
    # ...and the original still hits, so the fix does not merely widen the key.
    original = await cache.get(messages, MODEL, partition=PARTITION, **fields)
    assert original is not None and original.response_body == RESPONSE


@pytest.mark.parametrize("location", LITERAL_DICT_LOCATIONS)
def test_literal_cache_control_changes_the_key(location: str) -> None:
    messages, fields = _payloads(location)
    changed_messages, changed_fields = _with_changed_value(messages, fields, location)
    assert compute_semantic_cache_key(
        messages, MODEL, partition=PARTITION, **fields
    ) != compute_semantic_cache_key(changed_messages, MODEL, partition=PARTITION, **changed_fields)


def test_openai_arguments_string_is_not_rewritten() -> None:
    """OpenAI sends ``arguments`` as a JSON string; it must pass through verbatim.

    This shape was never aliased by the old stripper (strings are returned
    as-is at any depth), so it is a regression guard, not a bug demonstration.
    """
    first = '{"cache_control": "first"}'
    second = '{"cache_control": "second"}'
    payload = {"function": {"name": "configure", "arguments": first}}
    assert strip_cache_control(payload) == payload
    assert compute_semantic_cache_key(
        [{"role": "assistant", "content": None, "tool_calls": [{"id": "c", **payload}]}],
        MODEL,
        partition=PARTITION,
    ) != compute_semantic_cache_key(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c", "function": {"name": "configure", "arguments": second}}],
            }
        ],
        MODEL,
        partition=PARTITION,
    )


def test_transport_cache_control_is_still_ignored() -> None:
    """A moved breakpoint must not fragment the key (#327 intent)."""
    block = {"type": "text", "text": "sys"}
    with_block = [{**block, "cache_control": {"type": "ephemeral"}}]
    assert compute_semantic_cache_key(
        [{"role": "user", "content": "hello"}],
        MODEL,
        partition=PARTITION,
        system=with_block,
        tools=[{"name": "read", "cache_control": {"type": "ephemeral"}}],
    ) == compute_semantic_cache_key(
        [{"role": "user", "content": "hello"}],
        MODEL,
        partition=PARTITION,
        system=[block],
        tools=[{"name": "read"}],
    )


def test_strip_cache_control_keeps_schema_property_named_cache_control() -> None:
    schema = {
        "type": "object",
        "properties": {
            "cache_control": {"type": "boolean", "description": "Flush the application cache"}
        },
    }
    assert strip_cache_control(schema) == schema


def test_strip_cache_control_still_drops_nested_transport_annotations() -> None:
    assert strip_cache_control(
        {
            "cache_control": "drop-root",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "input": {"cache_control": "keep"}},
                        {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}},
                    ],
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ) == {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "input": {"cache_control": "keep"}},
                    {"type": "text", "text": "hi"},
                ],
            }
        ]
    }
