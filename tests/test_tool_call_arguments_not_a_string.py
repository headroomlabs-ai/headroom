"""Token counters must survive a tool_call whose fields aren't strings (GH #2782).

``function.arguments`` is a JSON *string* per the OpenAI spec, but
OpenAI-compatible upstreams do emit ``None`` or a raw object there. Every counter
passed the value straight to ``tiktoken.encode()``, which raises
``TypeError: expected string or buffer`` — so ``/v1/compress`` failed the whole
request with a 503. Worse, the malformed message stays in conversation history,
so every later request replaying that history failed too, regardless of provider.

``arguments: None`` stopped raising once ``count_text`` grew its falsy guard, but
any *truthy* non-string (``{"path": "x"}``, ``5``) still crashed all four
counters. The fix is ``coerce_countable_text`` at the tool-call field sites, so a
dict is priced roughly like the JSON string it should have been rather than
either crashing or silently counting as zero.
"""

from __future__ import annotations

import json

import pytest

from headroom.providers.anthropic import AnthropicProvider
from headroom.providers.openai import OpenAITokenCounter
from headroom.providers.openai_compatible import OpenAICompatibleTokenCounter
from headroom.tokenizers.base import coerce_countable_text
from headroom.tokenizers.tiktoken_counter import TiktokenCounter


def _counters():
    return {
        "openai": OpenAITokenCounter("gpt-4o"),
        "openai_compatible": OpenAICompatibleTokenCounter("gpt-4o"),
        "anthropic": AnthropicProvider(warn=False).get_token_counter("claude-sonnet-4-6"),
        "tiktoken": TiktokenCounter("gpt-4o"),
    }


def _message(arguments: object) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": arguments},
            }
        ],
    }


@pytest.mark.parametrize(
    "arguments",
    [None, {"path": "x"}, ["a", "b"], 5, True, 1.5],
    ids=["none", "dict", "list", "int", "bool", "float"],
)
def test_non_string_arguments_do_not_raise(arguments: object) -> None:
    """The reported crash: 503 + TypeError out of tiktoken.encode."""
    for name, counter in _counters().items():
        got = counter.count_messages([_message(arguments)])
        assert got > 0, f"{name} priced the whole message at {got}"


def test_object_arguments_are_priced_like_their_json_form() -> None:
    """Not just non-crashing: a dict must not silently count as zero."""
    payload = {"path": "src/very/long/path/to/a/file.py", "start": 1, "end": 400}
    for name, counter in _counters().items():
        as_object = counter.count_messages([_message(payload)])
        as_json = counter.count_messages([_message(json.dumps(payload))])
        assert abs(as_object - as_json) <= 5, f"{name}: {as_object} vs {as_json}"


def test_null_function_and_id_do_not_raise() -> None:
    """``{"function": null}`` / ``{"id": null}`` reach the same encode path."""
    message = {"role": "assistant", "tool_calls": [{"id": None, "function": None}]}
    for name, counter in _counters().items():
        assert counter.count_messages([message]) > 0, name


def test_legacy_function_call_with_object_arguments_does_not_raise() -> None:
    message = {"role": "assistant", "function_call": {"name": "f", "arguments": {"a": 1}}}
    for name, counter in _counters().items():
        assert counter.count_messages([message]) > 0, name


def test_oversized_object_arguments_are_bounded() -> None:
    """A malformed upstream must not turn an estimate into a megabyte encode."""
    huge = {"blob": "x" * 5_000_000}
    assert len(coerce_countable_text(huge)) <= 200_000


def test_string_arguments_are_untouched() -> None:
    """The control: the spec-compliant shape must not move."""
    args = json.dumps({"path": "a.py"})
    assert coerce_countable_text(args) == args
    assert coerce_countable_text(None) == ""
