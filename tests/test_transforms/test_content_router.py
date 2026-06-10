"""Tests for ContentRouter - intelligent content-based compression routing.

Comprehensive tests covering:
- ContentRouterConfig: Configuration validation and defaults
- ContentRouter: Core routing functionality
- Strategy detection: Code, JSON, search, logs, text
- Mixed content handling: Split, route, reassemble
- Transform interface: apply(), should_apply() methods
"""

from types import SimpleNamespace

import pytest

from headroom.transforms.content_detector import ContentType
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    RouterCompressionResult,
    RoutingDecision,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Default ContentRouterConfig for testing."""
    return ContentRouterConfig(
        min_section_tokens=10,  # Low threshold for tests
    )


@pytest.fixture
def router(default_config):
    """ContentRouter instance with default config."""
    return ContentRouter(default_config)


@pytest.fixture
def tokenizer():
    """Get a tokenizer for Transform interface tests."""
    from headroom.providers import OpenAIProvider
    from headroom.tokenizer import Tokenizer

    provider = OpenAIProvider()
    token_counter = provider.get_token_counter("gpt-4o")
    return Tokenizer(token_counter, "gpt-4o")


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_python_code(n_functions: int = 5) -> str:
    """Generate Python code for testing."""
    lines = [
        '"""Module with functions."""',
        "import os",
        "from typing import Any",
        "",
    ]
    for i in range(n_functions):
        lines.extend(
            [
                f"def function_{i}(arg: Any) -> str:",
                f'    """Process argument {i}."""',
                "    return str(arg)",
                "",
            ]
        )
    return "\n".join(lines)


def generate_json_data(n_items: int = 20) -> str:
    """Generate JSON content for testing."""
    import json

    items = [
        {"id": i, "name": f"Item {i}", "value": i * 10, "active": i % 2 == 0}
        for i in range(n_items)
    ]
    return json.dumps(items, indent=2)


def generate_search_results(n_results: int = 10) -> str:
    """Generate grep/search-like results for testing."""
    lines = []
    for i in range(n_results):
        lines.append(f"src/module{i}.py:42: def process_data(input: str) -> str:")
        lines.append(f"src/module{i}.py:43:     return transform(input)")
    return "\n".join(lines)


def generate_log_output(n_lines: int = 30) -> str:
    """Generate build/test log output for testing."""
    lines = [
        "Running tests...",
        "=== Test Suite: Unit Tests ===",
    ]
    for i in range(n_lines):
        if i % 10 == 0:
            lines.append(f"PASS tests/test_module{i}.py::test_function")
        elif i % 15 == 0:
            lines.append(f"FAIL tests/test_module{i}.py::test_failing")
        else:
            lines.append(f"  Running test_{i}... ok")
    lines.append("=== Summary ===")
    lines.append(f"Tests: {n_lines}, Passed: {n_lines - 2}, Failed: 2")
    return "\n".join(lines)


def generate_mixed_content() -> str:
    """Generate content with mixed types (markdown with code)."""
    return """# Documentation

This is a README file with code examples.

## Python Example

```python
def example():
    return "hello"
```

## JSON Configuration

```json
{"key": "value", "number": 42}
```

## Usage

Run the following command:
```bash
python main.py --verbose
```

That's all!
"""


def test_force_kompress_routes_anthropic_tool_result_to_targeted_kompress(
    router, tokenizer, monkeypatch
):
    captured: dict[str, object] = {}

    class FakeKompress:
        def compress(self, content, **kwargs):
            captured.update(kwargs)
            compressed = " ".join(content.split()[:20])
            return SimpleNamespace(
                compressed=compressed,
                compressed_tokens=len(compressed.split()),
            )

    monkeypatch.setattr(router, "_get_kompress", lambda: FakeKompress())
    tool_content = " ".join(
        f'{{"file":"src/module_{i}.py","line":{i},"text":"repeated search payload"}}'
        for i in range(160)
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_search_1",
                    "content": tool_content,
                }
            ],
        }
    ]

    result = router.apply(
        messages,
        tokenizer,
        force_kompress=True,
        target_ratio=0.10,
        compress_user_messages=True,
        min_tokens_to_compress=10,
        read_protection_window=0,
    )

    assert result.messages[0]["content"][0]["content"] != tool_content
    assert result.transforms_applied == ["router:tool_result:kompress"]
    assert captured["target_ratio"] == 0.10


# =============================================================================
# TestContentRouterConfig
# =============================================================================


class TestContentRouterConfig:
    """Tests for ContentRouterConfig dataclass."""

    def test_default_values(self):
        """Default config values are sensible."""
        config = ContentRouterConfig()

        assert config.enable_code_aware is False  # Disabled by default; use code graph MCP instead
        assert config.enable_kompress is True
        assert config.enable_smart_crusher is True
        assert config.enable_search_compressor is True
        assert config.enable_log_compressor is True
        assert config.min_section_tokens == 20
        assert config.fallback_strategy == CompressionStrategy.KOMPRESS

    def test_custom_values(self):
        """Custom config values are applied."""
        config = ContentRouterConfig(
            min_section_tokens=50,
            enable_code_aware=False,
            fallback_strategy=CompressionStrategy.TEXT,
        )

        assert config.min_section_tokens == 50
        assert config.enable_code_aware is False
        assert config.fallback_strategy == CompressionStrategy.TEXT

    def test_all_strategies_in_enum(self):
        """All expected strategies are in the enum."""
        expected = [
            "CODE_AWARE",
            "SMART_CRUSHER",
            "SEARCH",
            "LOG",
            "TEXT",
            "MIXED",
            "PASSTHROUGH",
        ]
        actual = [s.name for s in CompressionStrategy]
        for strategy in expected:
            assert strategy in actual, f"Missing strategy: {strategy}"


# =============================================================================
# TestRouterCompressionResult
# =============================================================================


class TestRouterCompressionResult:
    """Tests for RouterCompressionResult dataclass."""

    def test_tokens_saved_from_routing_log(self):
        """tokens_saved property calculates correctly from routing log."""
        result = RouterCompressionResult(
            compressed="short",
            original="long content here",
            strategy_used=CompressionStrategy.CODE_AWARE,
            routing_log=[
                RoutingDecision(
                    content_type=ContentType.SOURCE_CODE,
                    strategy=CompressionStrategy.CODE_AWARE,
                    confidence=0.9,
                    original_tokens=100,
                    compressed_tokens=30,
                )
            ],
            sections_processed=1,
        )

        assert result.tokens_saved == 70

    def test_tokens_saved_no_negative(self):
        """tokens_saved never returns negative."""
        result = RouterCompressionResult(
            compressed="expanded",
            original="short",
            strategy_used=CompressionStrategy.PASSTHROUGH,
            routing_log=[
                RoutingDecision(
                    content_type=ContentType.PLAIN_TEXT,
                    strategy=CompressionStrategy.PASSTHROUGH,
                    confidence=1.0,
                    original_tokens=10,
                    compressed_tokens=20,  # Expanded
                )
            ],
            sections_processed=1,
        )

        # Should be 0 not negative
        assert result.tokens_saved == 0

    def test_savings_percentage(self):
        """savings_percentage property calculates correctly."""
        result = RouterCompressionResult(
            compressed="short",
            original="long content",
            strategy_used=CompressionStrategy.TEXT,
            routing_log=[
                RoutingDecision(
                    content_type=ContentType.PLAIN_TEXT,
                    strategy=CompressionStrategy.TEXT,
                    confidence=0.8,
                    original_tokens=100,
                    compressed_tokens=25,
                )
            ],
            sections_processed=1,
        )

        assert result.savings_percentage == 75.0

    def test_empty_routing_log(self):
        """Handles empty routing log gracefully."""
        result = RouterCompressionResult(
            compressed="content",
            original="content",
            strategy_used=CompressionStrategy.PASSTHROUGH,
            routing_log=[],
            sections_processed=0,
        )

        assert result.total_original_tokens == 0
        assert result.total_compressed_tokens == 0
        assert result.savings_percentage == 0.0


# =============================================================================
# TestStrategyDetection
# =============================================================================


class TestStrategyDetection:
    """Tests for content type and strategy detection."""

    def test_detect_python_code(self, router):
        """Python code is detected."""
        code = generate_python_code(5)
        strategy = router._determine_strategy(code)
        # Should be either CODE_AWARE or fallback
        assert strategy in CompressionStrategy

    def test_detect_json_content(self, router):
        """JSON content is detected."""
        json_data = generate_json_data(20)
        strategy = router._determine_strategy(json_data)
        assert strategy in CompressionStrategy

    def test_detect_search_results(self, router):
        """Search/grep results are detected."""
        search_results = generate_search_results(10)
        strategy = router._determine_strategy(search_results)
        assert strategy in CompressionStrategy

    def test_detect_log_output(self, router):
        """Build/test logs are detected."""
        logs = generate_log_output(30)
        strategy = router._determine_strategy(logs)
        assert strategy in CompressionStrategy

    def test_detect_plain_text(self, router):
        """Plain text detection."""
        text = "This is just plain text without any special formatting."
        strategy = router._determine_strategy(text)
        assert strategy in CompressionStrategy


# =============================================================================
# TestContentRouter
# =============================================================================


class TestContentRouter:
    """Tests for ContentRouter core functionality."""

    def test_init_with_default_config(self):
        """Router initializes with default config."""
        router = ContentRouter()
        assert router.config is not None
        assert router.config.enable_code_aware is False  # Disabled by default

    def test_init_with_custom_config(self, default_config):
        """Router initializes with custom config."""
        router = ContentRouter(default_config)
        assert router.config == default_config

    def test_compress_empty_content(self, router):
        """Empty content returns passthrough."""
        result = router.compress("")
        assert result.compressed == ""
        assert result.strategy_used == CompressionStrategy.PASSTHROUGH

    def test_compress_small_content(self, router):
        """Small content returns same content."""
        result = router.compress("small")
        assert result.compressed == "small"
        # Small content might use TEXT or PASSTHROUGH strategy
        assert result.strategy_used in (
            CompressionStrategy.PASSTHROUGH,
            CompressionStrategy.TEXT,
        )

    def test_compress_returns_result(self, router):
        """compress() returns RouterCompressionResult."""
        content = generate_python_code(10)
        result = router.compress(content)

        assert isinstance(result, RouterCompressionResult)
        assert result.original == content
        assert result.strategy_used is not None

    def test_name_property(self, router):
        """Router has correct name."""
        assert router.name == "content_router"


# =============================================================================
# TestTransformInterface
# =============================================================================


class TestTransformInterface:
    """Tests for Transform interface (apply, should_apply)."""

    def test_should_apply_returns_bool(self, default_config, tokenizer):
        """should_apply returns a boolean."""
        router = ContentRouter(default_config)
        messages = [{"role": "user", "content": "small"}]

        result = router.should_apply(messages, tokenizer)
        assert isinstance(result, bool)

    def test_should_apply_returns_true_for_large_content(self, default_config, tokenizer):
        """should_apply returns True for large content."""
        router = ContentRouter(default_config)
        content = generate_python_code(20)
        messages = [{"role": "tool", "tool_call_id": "call_1", "content": content}]

        assert router.should_apply(messages, tokenizer)

    def test_apply_returns_transform_result(self, default_config, tokenizer):
        """apply() returns proper TransformResult."""
        router = ContentRouter(default_config)
        content = generate_python_code(10)
        messages = [{"role": "tool", "tool_call_id": "call_1", "content": content}]

        result = router.apply(messages, tokenizer)

        assert result is not None
        assert result.tokens_before > 0
        assert len(result.messages) == 1

    def test_apply_passes_through_non_tool_messages(self, default_config, tokenizer):
        """apply() passes through non-tool messages unchanged."""
        router = ContentRouter(default_config)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = router.apply(messages, tokenizer)

        assert result.messages[0]["content"] == "Hello"
        assert result.messages[1]["content"] == "Hi there!"


# =============================================================================
# TestCompressorDisabling
# =============================================================================


class TestCompressorDisabling:
    """Tests for disabling specific compressors.

    Note: These tests verify the config is accepted, not that the router
    actually respects the disable flags (which may not be fully implemented).
    """

    def test_config_accepts_disable_code_compression(self):
        """Config accepts enable_code_aware=False."""
        config = ContentRouterConfig(
            enable_code_aware=False,
            min_section_tokens=10,
        )
        router = ContentRouter(config)
        code = generate_python_code(10)

        # Should not crash
        result = router.compress(code)
        assert result is not None

    def test_config_accepts_disable_search_compression(self):
        """Config accepts enable_search_compressor=False."""
        config = ContentRouterConfig(
            enable_search_compressor=False,
            min_section_tokens=10,
        )
        router = ContentRouter(config)
        search_results = generate_search_results(10)

        # Should not crash
        result = router.compress(search_results)
        assert result is not None

    def test_config_accepts_disable_log_compression(self):
        """Config accepts enable_log_compressor=False."""
        config = ContentRouterConfig(
            enable_log_compressor=False,
            min_section_tokens=10,
        )
        router = ContentRouter(config)
        logs = generate_log_output(30)

        # Should not crash
        result = router.compress(logs)
        assert result is not None


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for ContentRouter."""

    def test_whitespace_only_content(self, router):
        """Whitespace-only content is handled gracefully."""
        result = router.compress("   \n\t\n   ")
        assert result.strategy_used == CompressionStrategy.PASSTHROUGH

    def test_unicode_content(self, router):
        """Unicode content is handled correctly."""
        content = "This has unicode: \u4e2d\u6587 \u65e5\u672c\u8a9e " * 50
        result = router.compress(content)
        assert result is not None

    def test_very_long_content(self, router):
        """Very long content is handled."""
        content = generate_python_code(100)
        result = router.compress(content)
        assert result is not None


# =============================================================================
# TestRoutingLog
# =============================================================================


class TestRoutingLog:
    """Tests for routing log functionality."""

    def test_routing_log_populated(self, router):
        """Routing log is populated with decisions."""
        content = generate_python_code(10)
        result = router.compress(content)

        # Routing log should be a list
        assert isinstance(result.routing_log, list)

    def test_routing_log_entries_have_strategy(self, router):
        """Routing log entries contain strategy."""
        content = generate_python_code(10)
        result = router.compress(content)

        for entry in result.routing_log:
            assert hasattr(entry, "strategy")
            assert entry.strategy in CompressionStrategy


# =============================================================================
# TestSummary
# =============================================================================


class TestSummary:
    """Tests for result summary generation."""

    def test_summary_property(self, router):
        """Summary property exists and is callable or returns string."""
        content = generate_python_code(10)
        result = router.compress(content)

        # Check summary property exists
        assert hasattr(result, "summary")

        # Get summary (call if callable)
        summary = result.summary
        if callable(summary):
            summary = summary()

        # Should be a string
        assert summary is not None


# =============================================================================
# TestExcludeTools
# =============================================================================


class TestExcludeTools:
    """Tests for exclude_tools feature - bypassing compression for specific tools."""

    @pytest.fixture
    def tokenizer(self):
        """Get a tokenizer for tests."""
        from headroom.providers import OpenAIProvider
        from headroom.tokenizer import Tokenizer

        provider = OpenAIProvider()
        token_counter = provider.get_token_counter("gpt-4o")
        return Tokenizer(token_counter, "gpt-4o")

    def test_default_exclude_tools_uses_defaults(self, tokenizer):
        """Default config excludes DEFAULT_EXCLUDE_TOOLS (Read, Glob, etc)."""
        config = ContentRouterConfig(min_section_tokens=10)
        router = ContentRouter(config)

        # Create message with tool call from "Read" tool (should be excluded)
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_read_1",
                        "type": "function",
                        "function": {"name": "Read", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_read_1",
                "content": generate_python_code(20),  # Large content that would normally compress
            },
        ]

        result = router.apply(messages, tokenizer)

        # Content should be unchanged (passed through, not compressed)
        assert result.messages[1]["content"] == messages[1]["content"]
        # Check transform was marked as excluded
        assert "router:excluded:tool" in result.transforms_applied

    def test_custom_exclude_tools(self, tokenizer):
        """Custom exclude_tools set is respected."""
        config = ContentRouterConfig(
            min_section_tokens=10,
            exclude_tools={"MyCustomTool"},  # Only exclude this tool
        )
        router = ContentRouter(config)

        # Create message with MyCustomTool (should be excluded)
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_custom_1",
                        "type": "function",
                        "function": {"name": "MyCustomTool", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_custom_1",
                "content": generate_json_data(50),
            },
        ]

        result = router.apply(messages, tokenizer)

        # Content should be unchanged
        assert result.messages[1]["content"] == messages[1]["content"]
        assert "router:excluded:tool" in result.transforms_applied

    def test_non_excluded_tools_are_compressed(self, tokenizer):
        """Tools not in exclude_tools set are still compressed."""
        config = ContentRouterConfig(
            min_section_tokens=10,
            exclude_tools={"Read"},  # Only exclude Read, not OtherTool
        )
        router = ContentRouter(config)

        original_content = generate_json_data(100)  # Large JSON array

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_other_1",
                        "type": "function",
                        "function": {"name": "OtherTool", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_other_1",
                "content": original_content,
            },
        ]

        result = router.apply(messages, tokenizer)

        # Content should be compressed (different from original)
        # Note: Compression may or may not change the content depending on strategy
        # But it should NOT have the excluded marker
        assert "router:excluded:tool" not in result.transforms_applied

    def test_empty_exclude_tools_compresses_all(self, tokenizer):
        """Empty exclude_tools set means no tools are excluded."""
        config = ContentRouterConfig(
            min_section_tokens=10,
            exclude_tools=set(),  # Empty set - exclude nothing
        )
        router = ContentRouter(config)

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_read_1",
                        "type": "function",
                        "function": {"name": "Read", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_read_1",
                "content": generate_python_code(20),
            },
        ]

        result = router.apply(messages, tokenizer)

        # Should NOT be excluded (empty set means compress everything)
        assert "router:excluded:tool" not in result.transforms_applied

    def test_anthropic_format_tool_result_exclusion(self, tokenizer):
        """Anthropic format tool_result blocks are also excluded."""
        config = ContentRouterConfig(
            min_section_tokens=10,
            exclude_tools={"Glob"},
        )
        router = ContentRouter(config)

        # Anthropic format with tool_use and tool_result in content blocks
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_glob_1",
                        "name": "Glob",
                        "input": {"pattern": "*.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_glob_1",
                        "content": generate_search_results(50),
                    }
                ],
            },
        ]

        result = router.apply(messages, tokenizer)

        # Find the tool_result block and verify content unchanged
        user_msg = result.messages[1]
        tool_result_block = next(
            (b for b in user_msg["content"] if b.get("type") == "tool_result"), None
        )
        assert tool_result_block is not None
        assert tool_result_block["content"] == messages[1]["content"][0]["content"]
        # Verify exclusion was tracked (consistent with OpenAI format)
        assert "router:excluded:tool" in result.transforms_applied

    def test_anthropic_tool_result_runtime_window_allows_old_excluded_tools(self, tokenizer):
        """Agent profiles can shrink the protected window for Claude tool results."""
        config = ContentRouterConfig(
            min_section_tokens=10,
            min_chars_for_block_compression=10,
            exclude_tools={"Glob"},
        )
        router = ContentRouter(config)

        old_tool_content = generate_search_results(80)
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_glob_old",
                        "name": "Glob",
                        "input": {"pattern": "*.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_glob_old",
                        "content": old_tool_content,
                    }
                ],
            },
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "ack"},
        ]

        result = router.apply(messages, tokenizer, read_protection_window=2)

        assert "router:excluded:tool" not in result.transforms_applied

    def test_mixed_excluded_and_non_excluded_tools(self, tokenizer):
        """Multiple tools in same conversation - only excluded ones pass through."""
        config = ContentRouterConfig(
            min_section_tokens=10,
            exclude_tools={"Read"},  # Only exclude Read
        )
        router = ContentRouter(config)

        read_content = generate_python_code(20)
        other_content = generate_json_data(100)

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_read_1",
                        "type": "function",
                        "function": {"name": "Read", "arguments": "{}"},
                    },
                    {
                        "id": "call_other_1",
                        "type": "function",
                        "function": {"name": "OtherTool", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_read_1",
                "content": read_content,
            },
            {
                "role": "tool",
                "tool_call_id": "call_other_1",
                "content": other_content,
            },
        ]

        result = router.apply(messages, tokenizer)

        # Read tool content should be unchanged (excluded)
        read_result = next(m for m in result.messages if m.get("tool_call_id") == "call_read_1")
        assert read_result["content"] == read_content

        # OtherTool may or may not be compressed, but should be processed
        # (we just verify it wasn't excluded)
        assert "router:excluded:tool" in result.transforms_applied
