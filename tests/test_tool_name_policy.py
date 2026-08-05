from __future__ import annotations

from headroom.proxy.helpers import _extract_tool_name
from headroom.proxy.tool_name_policy import extract_tool_name


def test_extracts_anthropic_custom_tool_name() -> None:
    assert extract_tool_name({"name": "memory_save"}) == "memory_save"


def test_extracts_openai_function_tool_name() -> None:
    assert (
        extract_tool_name({"type": "function", "function": {"name": "memory_search"}})
        == "memory_search"
    )


def test_extracts_native_tool_type_when_name_absent() -> None:
    assert extract_tool_name({"type": "memory_20250818"}) == "memory_20250818"


def test_prefers_explicit_name_over_function_and_type() -> None:
    assert (
        extract_tool_name(
            {
                "name": "headroom_retrieve",
                "type": "function",
                "function": {"name": "memory_save"},
            }
        )
        == "headroom_retrieve"
    )


def test_ignores_empty_or_non_string_names() -> None:
    assert extract_tool_name({"name": "", "function": {"name": ""}, "type": ""}) is None
    assert extract_tool_name({"name": 123, "function": {"name": 456}, "type": []}) is None


def test_helpers_private_wrapper_keeps_existing_import_path() -> None:
    tool_definition = {"function": {"name": "memory_update"}}

    assert _extract_tool_name(tool_definition) == extract_tool_name(tool_definition)
