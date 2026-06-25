"""Tests that HeadroomAgnoModel normalizes streaming tool_call objects to dicts.

Regression for https://github.com/headroomlabs-ai/headroom/issues/1312:
During streaming the OpenAI SDK populates tool_calls with Pydantic model
objects (ChoiceDeltaToolCall, ChatCompletionMessageToolCall), not plain dicts.
Headroom's parser and tokenizer call dict methods (.get, key-in) on tool_call
items; passing raw Pydantic models raises AttributeError and crashes the agent.
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

pytest.importorskip("agno", reason="agno not installed")

from headroom.integrations.agno.model import HeadroomAgnoModel


# ── Minimal stubs for OpenAI SDK streaming objects ──────────────────────────

@dataclass
class _FunctionDelta:
    name: str | None = "search_web"
    arguments: str | None = '{"query": "test"}'


@dataclass
class _ChoiceDeltaToolCall:
    """Minimal stub of openai.types.chat.chat_completion_chunk.ChoiceDeltaToolCall."""
    id: str = "call_abc123"
    type: str = "function"
    function: _FunctionDelta = None

    def __post_init__(self):
        if self.function is None:
            self.function = _FunctionDelta()


@dataclass
class _ChatCompletionMessageToolCall:
    """Minimal stub of openai.types.chat.ChatCompletionMessageToolCall."""
    id: str = "call_xyz789"
    type: str = "function"
    function: _FunctionDelta = None

    def __post_init__(self):
        if self.function is None:
            self.function = _FunctionDelta()


class _PydanticLikeFunctionDelta:
    """Simulates Pydantic v2 model_dump() on function delta."""
    def __init__(self):
        self.name = "get_weather"
        self.arguments = '{"city": "Paris"}'

    def model_dump(self):
        return {"name": self.name, "arguments": self.arguments}


class _PydanticLikeToolCall:
    """Simulates Pydantic v2 model with model_dump()."""
    def __init__(self):
        self.id = "call_pydantic"
        self.type = "function"
        self.function = _PydanticLikeFunctionDelta()

    def model_dump(self):
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function.model_dump(),
        }


# ── _normalize_tool_call unit tests ─────────────────────────────────────────

def test_normalize_plain_dict_passthrough():
    """Plain dicts must pass through unchanged."""
    tc = {"id": "call_1", "type": "function", "function": {"name": "foo", "arguments": "{}"}}
    model = HeadroomAgnoModel.__new__(HeadroomAgnoModel)
    result = model._normalize_tool_call(tc)
    assert result == tc
    assert result is tc  # same object, no copy


def test_normalize_dataclass_style_object():
    """Dataclass-style streaming objects (no model_dump) must be flattened."""
    tc = _ChoiceDeltaToolCall()
    model = HeadroomAgnoModel.__new__(HeadroomAgnoModel)
    result = model._normalize_tool_call(tc)

    assert isinstance(result, dict)
    assert result["id"] == "call_abc123"
    assert result["type"] == "function"
    assert isinstance(result["function"], dict)
    assert result["function"]["name"] == "search_web"
    assert result["function"]["arguments"] == '{"query": "test"}'


def test_normalize_pydantic_v2_model():
    """Pydantic v2 objects with model_dump() must use model_dump()."""
    tc = _PydanticLikeToolCall()
    model = HeadroomAgnoModel.__new__(HeadroomAgnoModel)
    result = model._normalize_tool_call(tc)

    assert isinstance(result, dict)
    assert result["id"] == "call_pydantic"
    assert result["function"] == {"name": "get_weather", "arguments": '{"city": "Paris"}'}


def test_normalize_partial_delta_missing_fields():
    """Stream deltas often have None fields — must not raise."""
    @dataclass
    class _PartialDelta:
        id: str | None = None
        type: str = "function"
        function: Any = None

    tc = _PartialDelta()
    model = HeadroomAgnoModel.__new__(HeadroomAgnoModel)
    result = model._normalize_tool_call(tc)

    assert isinstance(result, dict)
    # Must not raise — None fields are fine


# ── Integration: _convert_messages_to_openai ────────────────────────────────

def test_convert_messages_normalizes_tool_calls_in_agno_message():
    """tool_calls on an Agno Message object must be normalized to dicts."""
    from agno.models.message import Message as AgnoMessage

    streaming_tc = _ChoiceDeltaToolCall()
    msg = MagicMock(spec=AgnoMessage)
    msg.role = "assistant"
    msg.content = None
    msg.tool_calls = [streaming_tc]
    msg.tool_call_id = None
    msg.reasoning_content = None
    msg.redacted_reasoning_content = None
    msg.provider_data = None

    # We only need the conversion method — create a bare instance
    model = HeadroomAgnoModel.__new__(HeadroomAgnoModel)
    result = model._convert_messages_to_openai([msg])

    assert len(result) == 1
    tool_calls = result[0]["tool_calls"]
    assert isinstance(tool_calls, list)
    assert all(isinstance(tc, dict) for tc in tool_calls), \
        "All tool_calls must be plain dicts after normalization"

    tc = tool_calls[0]
    assert tc["id"] == "call_abc123"
    # These dict operations must not raise — this was the bug
    assert tc.get("id") == "call_abc123"
    assert "function" in tc
    assert tc["function"].get("name") == "search_web"


def test_parser_does_not_raise_on_normalized_tool_calls():
    """headroom.parser must not raise AttributeError after normalization."""
    from headroom.parser import parse_messages
    from headroom.tokenizers import get_tokenizer

    tokenizer = get_tokenizer("gpt-4o")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q":"test"}'},
                }
            ],
        }
    ]
    # Must not raise — parse_messages returns (blocks_per_msg, ...) tuple
    result = parse_messages(messages, tokenizer)
    assert result is not None
