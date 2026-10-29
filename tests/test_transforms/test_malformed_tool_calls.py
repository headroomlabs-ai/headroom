"""Malformed assistant tool calls must not raise out of ContentRouter.apply.

A request body can carry ``"function": null``, ``arguments`` that decode to a
non-object, or a ``tool_use`` whose ``file_path`` is not a string. Those used to
raise out of ``ReadLifecycleManager`` / ``ReadMaturationManager`` /
``ContentRouter._build_tool_name_map``; ``TransformPipeline.apply`` re-raised
and counted a circuit-breaker failure, so three such requests in a row switched
compression off for the breaker cooldown.
"""

from __future__ import annotations

import pytest

from headroom.config import ReadLifecycleConfig, ReadMaturationConfig
from headroom.tokenizers import EstimatingTokenCounter
from headroom.transforms.content_router import ContentRouter
from headroom.transforms.read_lifecycle import ReadLifecycleManager
from headroom.transforms.read_maturation import ReadMaturationManager


def _openai(function: object) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c1", "type": "function", "function": function}],
    }


def _anthropic(tool_input: object) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "c1", "name": "Read", "input": tool_input}],
    }


MALFORMED = {
    "function_null": _openai(None),
    "function_string": _openai("Read"),
    "arguments_list": _openai({"name": "Read", "arguments": "[1]"}),
    "arguments_string": _openai({"name": "Read", "arguments": '"s"'}),
    "arguments_number": _openai({"name": "Read", "arguments": "5"}),
    "arguments_invalid_json": _openai({"name": "Read", "arguments": "{"}),
    "arguments_file_path_list": _openai({"name": "Read", "arguments": '{"file_path": ["a"]}'}),
    "tool_use_file_path_list": _anthropic({"file_path": ["a"]}),
    "tool_use_path_dict": _anthropic({"path": {"a": 1}}),
    "tool_use_input_list": _anthropic([1]),
}


# Compressible log text: a well-formed call to an unknown tool would be compressed,
# so an unchanged result shows the malformed call's output was left alone.
RESULT = "".join(f"2026-10-24 INFO worker {i % 7} processed item {i} ok\n" for i in range(400))


def _conversation(assistant: dict, quiet_turns: int = 0) -> list[dict]:
    result = RESULT
    if "tool_calls" in assistant:
        tool_msg = {"role": "tool", "tool_call_id": "c1", "content": result}
    else:
        tool_msg = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "c1", "content": result}],
        }
    messages = [{"role": "user", "content": "go"}, assistant, tool_msg]
    for _ in range(quiet_turns):
        messages += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "next"}]
    return messages


def _tool_result_text(messages: list[dict]) -> str:
    content = messages[2]["content"]
    return content if isinstance(content, str) else content[0]["content"]


@pytest.mark.parametrize("assistant", MALFORMED.values(), ids=MALFORMED.keys())
def test_read_lifecycle_tolerates_malformed_tool_calls(assistant: dict) -> None:
    messages = _conversation(assistant)
    result = ReadLifecycleManager(ReadLifecycleConfig(enabled=True)).apply(messages)
    assert result.messages == messages


@pytest.mark.parametrize("assistant", MALFORMED.values(), ids=MALFORMED.keys())
def test_read_maturation_tolerates_malformed_tool_calls(assistant: dict) -> None:
    # Enough quiet turns that a well-formed Read would mature; a call whose file
    # cannot be read from its arguments must never be matured.
    messages = _conversation(assistant, quiet_turns=8)
    result = ReadMaturationManager(ReadMaturationConfig(enabled=True)).apply(messages)
    assert _tool_result_text(result.messages) == RESULT


@pytest.mark.parametrize("assistant", MALFORMED.values(), ids=MALFORMED.keys())
def test_content_router_apply_tolerates_malformed_tool_calls(assistant: dict) -> None:
    messages = _conversation(assistant)
    result = ContentRouter().apply(messages, EstimatingTokenCounter())
    assert len(result.messages) == len(messages)
    assert _tool_result_text(result.messages) == RESULT
