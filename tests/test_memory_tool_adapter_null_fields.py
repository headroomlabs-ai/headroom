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


def test_extract_tool_calls_survives_malformed_openai_choices():
    # `choices: [null]` (a content-filtered / usage-only turn from an
    # OpenAI-compatible gateway) once crashed choices[0].get. It must yield no
    # tool calls, across the explicit openai branch and the generic fallback.
    for choices in ([None], ["str"], [{"message": None}]):
        response = {"choices": choices}
        assert _adapter._extract_tool_calls(response, "openai") == []
        assert _adapter._extract_tool_calls(response, "other") == []


def test_extract_tool_calls_survives_malformed_gemini_candidates():
    # A null candidate / null content / null parts must not crash the Gemini
    # branch's chained .get calls.
    for response in (
        {"candidates": [None]},
        {"candidates": [{"content": None}]},
        {"candidates": [{"content": {"parts": None}}]},
    ):
        assert _adapter._extract_tool_calls(response, "gemini") == []

    # A real functionCall part is still returned; a non-dict part is skipped.
    response = {
        "candidates": [{"content": {"parts": [None, {"functionCall": {"name": "memory_save"}}]}}]
    }
    calls = _adapter._extract_tool_calls(response, "gemini")
    assert calls == [{"functionCall": {"name": "memory_save"}}]


def test_extract_tool_calls_survives_non_dict_anthropic_block():
    response = {"content": [None, {"type": "tool_use", "id": "t1", "name": "memory_save"}]}
    calls = _adapter._extract_tool_calls(response, "anthropic")
    assert [c["id"] for c in calls] == ["t1"]


def test_extract_tool_calls_still_parses_real_openai_call():
    response = {"choices": [{"message": {"tool_calls": [{"id": "c1"}]}}]}
    assert _adapter._extract_tool_calls(response, "openai") == [{"id": "c1"}]


def test_extract_tool_calls_skips_non_dict_openai_tool_call_elements():
    # A null / string element inside message.tool_calls is passed straight to
    # `.get` downstream (_get_tool_name / _get_tool_input), so a malformed
    # element must be dropped while a valid tool call is still returned.
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
    calls = _adapter._extract_tool_calls(response, "openai")
    assert [c["id"] for c in calls] == ["c1"]
    assert _adapter.has_memory_tool_calls(response, "openai") is True
