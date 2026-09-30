"""A tool call with a null ``function`` / ``arguments`` must not crash the
memory tool adapter's provider-format parsing.

``dict.get("function", {})`` returns ``None`` for a present-but-null key, so the
following ``.get`` raised ``AttributeError``; a null ``arguments`` makes
``json.loads(None)`` raise ``TypeError`` that the bare ``JSONDecodeError`` catch
missed. Both are reachable from the untrusted upstream response. The parse
helpers read only the ``tool_call`` argument, so we exercise them on a bare
instance via ``object.__new__``.
"""

from __future__ import annotations

from headroom.proxy.memory_tool_adapter import MemoryToolAdapter

_adapter = object.__new__(MemoryToolAdapter)


def test_get_tool_name_survives_null_function():
    tc = {"id": "c1", "type": "function", "function": None}
    assert _adapter._get_tool_name(tc, "openai") == ""
    assert _adapter._get_tool_id(tc, "openai") == "c1"
    assert _adapter._get_tool_input(tc, "openai") == {}


def test_get_tool_input_survives_null_arguments():
    # json.loads(None) raises TypeError, not JSONDecodeError.
    tc = {"function": {"name": "memory_save", "arguments": None}}
    assert _adapter._get_tool_input(tc, "openai") == {}


def test_get_tool_helpers_still_parse_real_calls():
    tc = {"function": {"name": "memory_save", "arguments": '{"content": "hi"}'}}
    assert _adapter._get_tool_name(tc, "openai") == "memory_save"
    assert _adapter._get_tool_input(tc, "openai") == {"content": "hi"}
