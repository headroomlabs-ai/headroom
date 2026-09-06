"""Provider-message metadata preservation contracts."""

from __future__ import annotations

from headroom.backends.anyllm import AnyLLMBackend
from headroom.backends.litellm import LiteLLMBackend
from headroom.message_contract import PRESERVED_MESSAGE_FIELDS, preserve_message_fields


def test_preserve_message_fields_copies_known_provider_metadata() -> None:
    source = {
        "role": "assistant",
        "content": "thinking",
        "reasoning_content": "private chain",
        "redacted_reasoning_content": "redacted",
        "provider_data": {"vendor": "deepseek"},
        "not_preserved": "discard",
    }
    target = {"role": "assistant", "content": "answer"}

    result = preserve_message_fields(source, target)

    assert result is target
    for field in PRESERVED_MESSAGE_FIELDS:
        if field in source:
            assert target[field] == source[field]
    assert "not_preserved" not in target


def test_litellm_conversion_preserves_reasoning_content_on_string_message() -> None:
    backend = LiteLLMBackend.__new__(LiteLLMBackend)
    messages = [
        {
            "role": "assistant",
            "content": "tool call explanation",
            "reasoning_content": "deepseek thinking state",
        }
    ]

    converted = backend._convert_messages_for_litellm(messages)

    assert converted[0]["reasoning_content"] == "deepseek thinking state"


def test_litellm_conversion_preserves_reasoning_content_on_tool_call_message() -> None:
    backend = LiteLLMBackend.__new__(LiteLLMBackend)
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will call the tool."},
                {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": "x"}},
            ],
            "reasoning_content": "deepseek thinking state",
        }
    ]

    converted = backend._convert_messages_for_litellm(messages)

    assert converted[0]["role"] == "assistant"
    assert converted[0]["tool_calls"][0]["id"] == "call_1"
    assert converted[0]["reasoning_content"] == "deepseek thinking state"


def test_anyllm_conversion_preserves_provider_data() -> None:
    backend = AnyLLMBackend.__new__(AnyLLMBackend)
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "answer"}],
            "provider_data": {"trace_id": "abc"},
        }
    ]

    converted = backend._convert_messages(messages)

    assert converted[0]["provider_data"] == {"trace_id": "abc"}
