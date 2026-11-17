"""A tool call with a null ``function`` must not crash memory tool-call
detection in ``MemoryHandler``.

``tc.get("function", {}).get("name")`` raises ``AttributeError`` on an explicit
``{"function": null}`` (the default only applies to a missing key). Both
``has_memory_tool_calls`` and the arg extraction in ``handle_tool_calls`` read
that shape from the untrusted upstream response. ``has_memory_tool_calls`` and
``_extract_tool_calls`` use no instance state, so we exercise them on a bare
instance via ``object.__new__``.
"""

from __future__ import annotations

from headroom.proxy.memory_handler import MemoryHandler

_handler = object.__new__(MemoryHandler)


def _openai_response(tool_calls):
    return {"choices": [{"message": {"tool_calls": tool_calls}}]}


def test_has_memory_tool_calls_survives_null_function():
    response = _openai_response(
        [
            {"id": "c1", "type": "function", "function": None},
            {"id": "c2", "type": "function", "function": {"name": "memory_save"}},
        ]
    )
    # Must not raise, and must still see the real memory tool call.
    assert _handler.has_memory_tool_calls(response, "openai") is True


def test_has_memory_tool_calls_all_null_functions_is_false():
    response = _openai_response([{"id": "c1", "type": "function", "function": None}])
    assert _handler.has_memory_tool_calls(response, "openai") is False


def test_extract_tool_calls_survives_null_choice_element():
    # An OpenAI-compatible gateway can send `choices: [null]` on a
    # content-filtered / usage-only response. `choices[0].get(...)` on the null
    # element would raise AttributeError; detection must treat it as no tool
    # calls, matching the sibling guard in CCRResponseHandler.
    for choices in ([None], ["not-a-dict"], [{"message": None}]):
        response = {"choices": choices}
        assert _handler._extract_tool_calls(response, "openai") == []
        assert _handler.has_memory_tool_calls(response, "openai") is False


def test_extract_tool_calls_falls_through_to_responses_output():
    # A null first choice must not shadow a Responses-API `output[]` payload in
    # the same response; the function_call there is still detected.
    response = {
        "choices": [None],
        "output": [{"type": "function_call", "name": "memory_save", "arguments": "{}"}],
    }
    assert _handler.has_memory_tool_calls(response, "openai") is True


def test_extract_tool_calls_survives_non_dict_anthropic_block():
    # A reconstructed Anthropic response may carry a null content block; the
    # `block.get("type")` filter must skip it instead of raising.
    response = {"content": [None, {"type": "tool_use", "id": "t1", "name": "memory_save"}]}
    calls = _handler._extract_tool_calls(response, "anthropic")
    assert [c["id"] for c in calls] == ["t1"]


def test_extract_tool_calls_skips_non_dict_openai_tool_call_elements():
    # A null / string element inside message.tool_calls is passed straight to
    # `.get` downstream (has_memory_tool_calls / handle_memory_tool_calls), so a
    # malformed element must be dropped while a valid tool call is still seen.
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        None,
                        "oops",
                        {"id": "c1", "type": "function", "function": {"name": "memory_save"}},
                    ]
                }
            }
        ]
    }
    calls = _handler._extract_tool_calls(response, "openai")
    assert [c["id"] for c in calls] == ["c1"]
    assert _handler.has_memory_tool_calls(response, "openai") is True
