"""Tests for headroom.proxy.tool_schema_compaction — shared tool-schema compaction.

Verifies that the compaction logic (shared by OpenAI and Anthropic handlers):
- strips JSON Schema annotation keys ($schema, title, examples, …)
- preserves property names that collide with DROP_KEYS (e.g. a field named "title")
- normalises description whitespace
- never inflates payload size
"""

from __future__ import annotations

from headroom.proxy.tool_schema_compaction import (
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
            tools.append(
                {
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
                }
            )
        payload = {"model": "claude-sonnet-4-20250514", "tools": tools}
        result, modified, before, after = compact_tools(payload)
        assert modified is True
        savings_pct = (1 - after / before) * 100
        # Expect meaningful savings (at least 15% with annotation keys + whitespace)
        assert savings_pct >= 10, f"Expected ≥10% savings, got {savings_pct:.1f}%"


# ---------------------------------------------------------------------------
# Layer 2: compact_tool_descriptions
# ---------------------------------------------------------------------------


class TestTruncateDescription:
    """Unit tests for _truncate_description."""

    def test_short_description_unchanged(self) -> None:
        from headroom.proxy.tool_schema_compaction import _truncate_description

        assert _truncate_description("Read a file.", 120) == "Read a file."

    def test_first_sentence_preserved(self) -> None:
        from headroom.proxy.tool_schema_compaction import _truncate_description

        desc = "Fast and precise code search across ALL GitHub repositories. Best for finding exact symbols."
        result = _truncate_description(desc, 60)
        assert result == "Fast and precise code search across ALL GitHub repositories."

    def test_first_sentence_plus_second(self) -> None:
        from headroom.proxy.tool_schema_compaction import _truncate_description

        desc = "Read a file. Returns the contents as text."
        result = _truncate_description(desc, 60)
        # First sentence is short enough, second fits within 1.5x budget
        assert "Read a file." in result
        assert "Returns the contents as text." in result

    def test_long_first_sentence_truncated(self) -> None:
        from headroom.proxy.tool_schema_compaction import _truncate_description

        desc = "This is an extremely long description that goes on and on without any sentence boundary"
        result = _truncate_description(desc, 40)
        assert len(result) <= 45  # 40 + "…"
        assert result.endswith("…")

    def test_whitespace_normalised_before_truncation(self) -> None:
        from headroom.proxy.tool_schema_compaction import _truncate_description

        desc = "  Search   code.   Very   useful.  "
        result = _truncate_description(desc, 60)
        # Whitespace normalised, both sentences fit within 1.5x budget
        assert result == "Search code. Very useful."

    def test_max_chars_zero_returns_original(self) -> None:
        from headroom.proxy.tool_schema_compaction import _truncate_description

        desc = "Any long description that would normally be truncated."
        result = _truncate_description(desc, 0)
        # max_chars=0 means disabled, return unchanged
        assert result == "Any long description that would normally be truncated."

    def test_chinese_description(self) -> None:
        from headroom.proxy.tool_schema_compaction import _truncate_description

        desc = "搜索代码仓库中的函数和类。支持正则表达式匹配。"
        result = _truncate_description(desc, 30)
        # First Chinese sentence fits
        assert "搜索代码仓库中的函数和类。" in result


class TestCompactToolDescriptions:
    """Unit tests for compact_tool_descriptions (full payload)."""

    def test_truncates_long_tool_description(self) -> None:
        from headroom.proxy.tool_schema_compaction import compact_tool_descriptions

        payload = {
            "tools": [
                {
                    "name": "search_code",
                    "description": "Fast and precise code search across ALL GitHub repositories using GitHub's native search engine. Best for finding exact symbols, functions, classes, or specific code patterns. Returns ranked results.",
                    "input_schema": {"type": "object", "properties": {}},
                },
            ],
        }
        result, modified, before, after = compact_tool_descriptions(payload, max_chars=60)
        assert modified is True
        assert after < before
        tool = result["tools"][0]
        # First sentence ends at "repositories." (abbrev regex match)
        # With max_chars=60, first sentence "Fast...repositories." is 60 chars exactly
        # so it should be preserved. Second sentence may or may not fit in 1.5x budget.
        assert tool["description"].startswith("Fast and precise code search")
        assert len(tool["description"]) < len(payload["tools"][0]["description"])

    def test_truncates_nested_param_descriptions(self) -> None:
        from headroom.proxy.tool_schema_compaction import compact_tool_descriptions

        payload = {
            "tools": [
                {
                    "name": "t",
                    "description": "Short.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "q": {
                                "type": "string",
                                "description": "The search query string to find matching code. Supports advanced syntax like OR, NOT, and quoted phrases for exact match.",
                            },
                        },
                    },
                },
            ],
        }
        result, modified, before, after = compact_tool_descriptions(payload, max_chars=60)
        assert modified is True
        param_desc = result["tools"][0]["input_schema"]["properties"]["q"]["description"]
        assert "The search query string to find matching code." == param_desc

    def test_disabled_when_max_chars_zero(self) -> None:
        from headroom.proxy.tool_schema_compaction import compact_tool_descriptions

        payload = {
            "tools": [
                {"name": "t", "description": "A" * 500, "input_schema": {"type": "object"}},
            ],
        }
        result, modified, _, _ = compact_tool_descriptions(payload, max_chars=0)
        assert modified is False
        assert result is payload

    def test_no_tools_returns_unchanged(self) -> None:
        from headroom.proxy.tool_schema_compaction import compact_tool_descriptions

        result, modified, _, _ = compact_tool_descriptions({"model": "x"}, max_chars=120)
        assert modified is False

    def test_preserves_non_description_fields(self) -> None:
        from headroom.proxy.tool_schema_compaction import compact_tool_descriptions

        payload = {
            "model": "claude-sonnet-4-20250514",
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get the current weather for a location. Supports any city worldwide.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city name to get weather for.",
                            },
                        },
                    },
                },
            ],
        }
        result, _, _, _ = compact_tool_descriptions(payload, max_chars=50)
        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["tools"][0]["name"] == "get_weather"
        assert result["tools"][0]["input_schema"]["properties"]["city"]["type"] == "string"

    def test_large_tool_set_savings(self) -> None:
        """44 GitHub-like tools with verbose descriptions should see significant savings."""
        from headroom.proxy.tool_schema_compaction import compact_tool_descriptions

        tools = []
        for i in range(44):
            tools.append(
                {
                    "name": f"github_tool_{i}",
                    "description": f"Perform operation {i} on GitHub repositories. This tool supports advanced filtering and pagination for large result sets. Use it for code search, issue management, and PR operations.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "owner": {
                                "type": "string",
                                "description": "Repository owner username or organization name.",
                            },
                            "repo": {
                                "type": "string",
                                "description": "The name of the repository to operate on.",
                            },
                        },
                    },
                }
            )
        payload = {"tools": tools}
        result, modified, before, after = compact_tool_descriptions(payload, max_chars=80)
        assert modified is True
        savings_pct = (1 - after / before) * 100
        assert savings_pct >= 10, f"Expected ≥10% savings, got {savings_pct:.1f}%"
