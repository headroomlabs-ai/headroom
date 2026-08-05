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
