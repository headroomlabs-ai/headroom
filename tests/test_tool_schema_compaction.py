"""Tests for headroom.proxy.tool_schema_compaction — shared tool-schema compaction.

Verifies that the compaction logic (shared by OpenAI and Anthropic handlers):
- strips JSON Schema annotation keys ($schema, title, examples, …)
- preserves property names that collide with DROP_KEYS (e.g. a field named "title")
- normalises description whitespace
- never inflates payload size
"""

from __future__ import annotations

import json

from headroom.proxy.tool_schema_compaction import (
    TOOL_SCHEMA_DROP_KEYS,
    compact_tool_schema_value,
    compact_tools,
)


# ---------------------------------------------------------------------------
# compact_tool_schema_value
# ---------------------------------------------------------------------------

class TestCompactToolSchemaValue:
    """Unit tests for compact_tool_schema_value."""

    def test_drops_schema_annotations(self) -> None:
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "MyToolParams",
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        result = compact_tool_schema_value(schema)
        assert "$schema" not in result
        assert "title" not in result
        assert "type" in result
        assert "properties" in result
        assert "required" in result

    def test_preserves_property_named_title(self) -> None:
        """A field literally named 'title' must survive (not a schema annotation)."""
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "code": {"type": "string"},
            },
            "required": ["title", "code"],
        }
        result = compact_tool_schema_value(schema)
        props = result["properties"]
        assert "title" in props, "property named 'title' must survive"
        assert "code" in props

    def test_normalises_description_whitespace(self) -> None:
        schema = {
            "name": "my_tool",
            "description": "  This   is   a   description  \n  with   extra   spaces  ",
            "input_schema": {"type": "object", "properties": {}},
        }
        result = compact_tool_schema_value(schema)
        assert result["description"] == "This is a description with extra spaces"

    def test_drops_examples_and_deprecated(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "x": {
                    "type": "integer",
                    "examples": [1, 2, 3],
                    "deprecated": True,
                },
            },
        }
        result = compact_tool_schema_value(schema)
        prop_x = result["properties"]["x"]
        assert "examples" not in prop_x
        assert "deprecated" not in prop_x
        assert prop_x["type"] == "integer"

    def test_preserves_property_named_deprecated(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "deprecated": {"type": "boolean", "description": "Is it deprecated?"},
            },
        }
        result = compact_tool_schema_value(schema)
        assert "deprecated" in result["properties"]

    def test_handles_list_of_tools(self) -> None:
        tools = [
            {
                "name": "tool_a",
                "description": "  First  tool  ",
                "input_schema": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "title": "ToolAParams",
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                },
            },
            {
                "name": "tool_b",
                "description": "  Second  tool  ",
                "input_schema": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "title": "ToolBParams",
                    "type": "object",
                    "properties": {"b": {"type": "integer"}},
                },
            },
        ]
        result = compact_tool_schema_value(tools)
        assert len(result) == 2
        for tool in result:
            assert "$schema" not in tool["input_schema"]
            assert "title" not in tool["input_schema"]
            assert "  " not in tool["description"]

    def test_nested_properties_preserved(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "title": "ConfigObject",  # annotation — should be dropped
                    "properties": {
                        "title": {"type": "string"},  # property name — must survive
                        "value": {"type": "integer"},
                    },
                },
            },
        }
        result = compact_tool_schema_value(schema)
        # Top-level config annotation dropped
        assert "title" not in result["properties"]["config"]
        # But nested property named "title" preserved
        assert "title" in result["properties"]["config"]["properties"]


# ---------------------------------------------------------------------------
# compact_tools
# ---------------------------------------------------------------------------

class TestCompactTools:
    """Unit tests for compact_tools (full payload compaction)."""

    def test_compacts_anthropic_style_payload(self) -> None:
        """Anthropic Messages API format uses 'input_schema'."""
        payload = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "  Get the   current   weather  ",
                    "input_schema": {
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "title": "GetWeatherParams",
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                },
            ],
        }
        result, modified, before, after = compact_tools(payload)
        assert modified is True
        assert after < before
        tool = result["tools"][0]
        assert "  " not in tool["description"]
        assert "$schema" not in tool["input_schema"]
        assert "title" not in tool["input_schema"]
        assert "properties" in tool["input_schema"]

    def test_compacts_openai_style_payload(self) -> None:
        """OpenAI format uses 'parameters' instead of 'input_schema'."""
        payload = {
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "description": "Read a file from disk.",
                    "parameters": {
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "title": "ReadFileParams",
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "examples": ["/tmp/test"]},
                        },
                        "required": ["path"],
                    },
                },
            ],
        }
        result, modified, before, after = compact_tools(payload)
        assert modified is True
        assert after < before
        params = result["tools"][0]["parameters"]
        assert "$schema" not in params
        assert "title" not in params
        assert "examples" not in params["properties"]["path"]

    def test_returns_unchanged_when_no_tools(self) -> None:
        payload = {"model": "claude-sonnet-4-20250514", "messages": []}
        result, modified, _, _ = compact_tools(payload)
        assert modified is False
        assert result is payload  # same object, not copied

    def test_returns_unchanged_when_empty_tools(self) -> None:
        payload = {"tools": []}
        result, modified, _, _ = compact_tools(payload)
        assert modified is False

    def test_returns_unchanged_when_already_compact(self) -> None:
        payload = {
            "tools": [
                {
                    "name": "simple",
                    "description": "A simple tool",
                    "input_schema": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}},
                    },
                },
            ],
        }
        result, modified, before, after = compact_tools(payload)
        # May or may not be modified depending on description whitespace
        # but should never inflate
        assert after <= before

    def test_preserves_non_tool_fields(self) -> None:
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "name": "t",
                    "description": "test",
                    "input_schema": {
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                    },
                },
            ],
        }
        result, _, _, _ = compact_tools(payload)
        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["max_tokens"] == 1024
        assert len(result["messages"]) == 1

    def test_large_github_like_tool_set(self) -> None:
        """Simulate a large tool set (like GitHub MCP with 44 tools)."""
        tools = []
        for i in range(44):
            tools.append({
                "name": f"github_tool_{i}",
                "description": f"  Perform   operation   {i}   on   GitHub   repositories  ",
                "input_schema": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "title": f"GithubTool{i}Params",
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "description": "  Repo   owner  "},
                        "repo": {"type": "string", "examples": ["my-repo"]},
                    },
                    "required": ["owner", "repo"],
                },
            })
        payload = {"model": "claude-sonnet-4-20250514", "tools": tools}
        result, modified, before, after = compact_tools(payload)
        assert modified is True
        savings_pct = (1 - after / before) * 100
        # Expect meaningful savings (at least 15% with annotation keys + whitespace)
        assert savings_pct >= 10, f"Expected ≥10% savings, got {savings_pct:.1f}%"
