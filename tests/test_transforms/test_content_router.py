"""Tests for ContentRouter - intelligent content-based compression routing.

Comprehensive tests covering:
- ContentRouterConfig: Configuration validation and defaults
- ContentRouter: Core routing functionality
- Strategy detection: Code, JSON, search, logs, text
- Mixed content handling: Split, route, reassemble
- Transform interface: apply(), should_apply() methods
"""

import json
import logging
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
from headroom.transforms.lossless_compaction import search_unheading

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
        def is_ready(self) -> bool:
            return True

        def ensure_background_load(self) -> None:
            pass

        def compress(self, content, **kwargs):
            captured.update(kwargs)
            # Real Kompress appends a CCR retrieval marker when CCR is enabled,
            # keeping the lossy result recoverable. Include one so the router's
            # reversibility gate (tool ground truth must stay recoverable, #1307)
            # accepts the compression instead of reverting to verbatim.
            compressed = " ".join(content.split()[:20]) + " Retrieve more: hash=deadbeef"
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


def test_skip_kompress_routes_around_ml_stage(router, tokenizer, monkeypatch):
    """skip_kompress (cold-start fast pass) must never invoke the Kompress ML
    stage — units that would route there take the same fallback as when the
    model isn't ready — and wins over force_kompress."""
    calls: list[str] = []

    class FakeKompress:
        def is_ready(self) -> bool:
            return True

        def ensure_background_load(self) -> None:
            pass

        def compress(self, content, **kwargs):
            calls.append(content)
            compressed = " ".join(content.split()[:20]) + " Retrieve more: hash=deadbeef"
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
        skip_kompress=True,
        target_ratio=0.10,
        compress_user_messages=True,
        min_tokens_to_compress=10,
        read_protection_window=0,
    )

    assert calls == []
    assert result.transforms_applied != ["router:tool_result:kompress"]
    # The pass still completes and returns a well-formed message list.
    assert result.messages[0]["content"][0]["content"]


def test_anthropic_tool_result_lossy_without_marker_stays_verbatim(router, tokenizer, monkeypatch):
    """Reversibility gate (#1307): a lossy Kompress result on a tool_result block
    with no CCR retrieval marker is unrecoverable, so the router must keep the
    original verbatim rather than hand the agent a fabricated summary. This is
    the block-path counterpart to the string/`role=="tool"` guard."""

    class FakeKompress:
        def is_ready(self) -> bool:
            return True

        def ensure_background_load(self) -> None:
            pass

        def compress(self, content, **kwargs):
            # Lossy summary with NO retrieval marker → unrecoverable.
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

    # Unrecoverable lossy compression is rejected → original kept verbatim.
    assert result.messages[0]["content"][0]["content"] == tool_content


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

    def test_compress_diff_accepts_none_context_with_debug(self, router, caplog):
        """None context is normalized before debug logging and compressor dispatch."""

        class FakeDiffCompressor:
            def compress(self, content, context):
                assert context == ""
                return SimpleNamespace(compressed="diff summary")

        diff = "diff --git a/file.py b/file.py\n@@ -1 +1 @@\n-old\n+new\n"
        router._diff_compressor = FakeDiffCompressor()

        caplog.set_level(logging.DEBUG, logger="headroom.transforms.content_router")
        result = router.compress(diff, context=None)

        assert result.compressed == "diff summary"
        assert result.strategy_used == CompressionStrategy.DIFF

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

        # Excluded from *lossy* compression, but JSON still gets a data-lossless
        # minify (same object, fewer tokens). Assert recovery, not byte-identity.
        assert json.loads(result.messages[1]["content"]) == json.loads(messages[1]["content"])
        assert "router:excluded:lossless_json" in result.transforms_applied

    def test_glob_exclude_tools(self, tokenizer):
        """Glob patterns in exclude_tools match by prefix (issue #870)."""
        config = ContentRouterConfig(
            min_section_tokens=10,
            exclude_tools={"mcp__*"},  # One pattern excludes every MCP tool
        )
        router = ContentRouter(config)

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_mcp_1",
                        "type": "function",
                        "function": {
                            "name": "mcp__build123d__measure",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_mcp_1",
                "content": generate_json_data(50),
            },
        ]

        result = router.apply(messages, tokenizer)

        # The MCP tool result matched the glob → excluded from lossy compression.
        # Its JSON still gets a data-lossless minify; assert recovery.
        assert json.loads(result.messages[1]["content"]) == json.loads(messages[1]["content"])
        assert "router:excluded:lossless_json" in result.transforms_applied

    def test_anthropic_mcp_alias_exclude_tools(self, tokenizer):
        """Single-underscore MCP names from custom agents honor documented MCP globs."""
        config = ContentRouterConfig(
            min_section_tokens=10,
            exclude_tools={"mcp__*"},
        )
        router = ContentRouter(config)

        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_mcp_1",
                        "name": "mcp_CursorTaskRegistry_cursor_list_tasks",
                        "input": {"project": "headroom"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_mcp_1",
                        "content": generate_json_data(50),
                    }
                ],
            },
        ]

        result = router.apply(messages, tokenizer)

        tool_result_block = result.messages[1]["content"][0]
        assert json.loads(tool_result_block["content"]) == json.loads(
            messages[1]["content"][0]["content"]
        )
        assert "router:excluded:lossless_json" in result.transforms_applied

    def test_anthropic_mcp_bare_tool_alias_exclude_tools(self, tokenizer):
        """Bare tool exclusions match custom-agent MCP wrappers (#1822).

        Any MCP wrapper's bare tool name can be excluded via config — this test
        uses a fictitious "HeadroomZai" server name to prove the alias match is
        server-name-agnostic. ``headroom_retrieve`` specifically is now also an
        unconditional, config-independent exclusion (see the fix for the
        ContentRouter self-recompression bug: SmartCrusher.apply() already
        guarded #1077 on its own call path, but ContentRouter called
        SmartCrusher.crush() directly, bypassing it). That guard fires before
        the config-driven `excluded_tool_ids` check below, giving byte-identical
        passthrough rather than the lossless-JSON fold a narrower custom
        `exclude_tools` used to produce for this specific tool name.
        """
        config = ContentRouterConfig(
            min_section_tokens=10,
            exclude_tools={"headroom_retrieve"},
        )
        router = ContentRouter(config)

        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_retrieve_1",
                        "name": "mcp_HeadroomZai_headroom_retrieve",
                        "input": {"key": "abc123"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_retrieve_1",
                        "content": generate_json_data(50),
                    }
                ],
            },
        ]

        result = router.apply(messages, tokenizer)

        tool_result_block = result.messages[1]["content"][0]
        # Byte-identical, not just JSON-semantically-equal: the unconditional
        # ccr_retrieve guard passes the original block through untouched.
        assert tool_result_block["content"] == messages[1]["content"][0]["content"]
        assert "router:excluded:ccr_retrieve" in result.transforms_applied

    def test_anthropic_mcp_bare_tool_alias_exclude_tools_generic(self, tokenizer):
        """General #1822 coverage: bare-name alias matching through the
        config-driven ``excluded_tool_ids``/``DEFAULT_VERBATIM_EXCLUDE_TOOLS``
        path for an arbitrary tool that is NOT ``headroom_retrieve`` (which now
        has its own unconditional guard that would otherwise mask this path —
        see ``test_anthropic_mcp_bare_tool_alias_exclude_tools`` above)."""
        config = ContentRouterConfig(
            min_section_tokens=10,
            exclude_tools={"measure"},
        )
        router = ContentRouter(config)

        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_measure_1",
                        "name": "mcp_build123d_measure",
                        "input": {"key": "abc123"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_measure_1",
                        "content": generate_json_data(50),
                    }
                ],
            },
        ]

        result = router.apply(messages, tokenizer)

        tool_result_block = result.messages[1]["content"][0]
        assert json.loads(tool_result_block["content"]) == json.loads(
            messages[1]["content"][0]["content"]
        )
        assert "router:excluded:lossless_json" in result.transforms_applied

    def test_is_tool_excluded_helper(self):
        """is_tool_excluded: exact (case-insensitive) and glob matching."""
        from headroom.config import is_tool_excluded

        # Glob entry covers a whole MCP server; unrelated tools are untouched.
        assert is_tool_excluded("mcp__build123d__measure", {"mcp__*"})
        assert is_tool_excluded("mcp_CursorTaskRegistry_cursor_list_tasks", {"mcp__*"})
        assert not is_tool_excluded("Bash", {"mcp__*"})
        # Plain entries keep exact, case-insensitive membership.
        assert is_tool_excluded("Read", {"read"})
        assert is_tool_excluded("MCP__X", {"mcp__*"})
        # MCP wrapper aliases can still be excluded by their bare tool name.
        assert is_tool_excluded("mcp_HeadroomZai_headroom_retrieve", {"headroom_retrieve"})
        assert is_tool_excluded("mcp__Headroom__headroom_retrieve", {"headroom_retrieve"})
        # Empty set never excludes.
        assert not is_tool_excluded("Read", set())

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
        # Excluded from lossy compression; search results get a byte-lossless
        # heading fold. Verify byte-exact recovery (Anthropic block format).
        original = messages[1]["content"][0]["content"]
        assert search_unheading(tool_result_block["content"]) == original
        assert "router:excluded:lossless_search" in result.transforms_applied

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

    def test_protect_recent_reads_fraction_zero_overrides_runtime_window(self, tokenizer):
        """protect_recent_reads_fraction == 0.0 (the --protect-tool-results
        sentinel) means "protect all excluded-tool output forever". A
        profile-derived read_protection_window kwarg must not be allowed to
        shrink that back down -- regression test for the precedence bug
        where the runtime kwarg unconditionally overrode this config-level
        guarantee."""
        config = ContentRouterConfig(
            min_section_tokens=10,
            min_chars_for_block_compression=10,
            exclude_tools={"Glob"},
            protect_recent_reads_fraction=0.0,
        )
        router = ContentRouter(config)

        # Plain unstructured text (not grep/log/json-shaped) so
        # _lossless_compact_excluded returns None and the router takes the
        # bare "protect as before" branch, matching the tag this test
        # asserts on.
        old_tool_content = "\n".join(
            f"line {i}: some output from a glob command that is long enough to compress"
            for i in range(80)
        )
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

        assert "router:excluded:tool" in result.transforms_applied

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

    def test_bash_not_in_default_exclude_tools(self):
        """Bash is NOT excluded by default — its outputs (build logs, test
        output) are ideal compression targets. Regression test for PR #704.

        This test validates the DEFAULT_EXCLUDE_TOOLS frozenset directly
        (pure config check — no Rust dependency).
        """
        from headroom.config import DEFAULT_EXCLUDE_TOOLS

        assert "Bash" not in DEFAULT_EXCLUDE_TOOLS, (
            "Bash should NOT be in DEFAULT_EXCLUDE_TOOLS — "
            "its outputs (build logs, test output) are ideal compression targets"
        )
        assert "bash" not in DEFAULT_EXCLUDE_TOOLS, "'bash' should NOT be in DEFAULT_EXCLUDE_TOOLS"

    def test_bash_lowercase_not_in_exclude_tools(self):
        """Lowercase 'bash' is also NOT in default exclude tools."""
        from headroom.config import DEFAULT_EXCLUDE_TOOLS

        assert "bash" not in DEFAULT_EXCLUDE_TOOLS

    def test_default_exclude_tools_membership(self):
        """Verify all expected exclude tools and their lowercase variants."""
        from headroom.config import DEFAULT_EXCLUDE_TOOLS

        # Tools that SHOULD be excluded (fresh Read/Write/Edit/Glob/Grep outputs)
        for tool in ("Read", "Glob", "Grep", "Write", "Edit"):
            assert tool in DEFAULT_EXCLUDE_TOOLS, f"{tool} should be in DEFAULT_EXCLUDE_TOOLS"
            assert tool.lower() in DEFAULT_EXCLUDE_TOOLS, (
                f"{tool.lower()} should be in DEFAULT_EXCLUDE_TOOLS"
            )

        # Tools that should NOT be excluded
        for tool in ("Bash", "bash", "TodoWrite", "todo_write"):
            assert tool not in DEFAULT_EXCLUDE_TOOLS, (
                f"{tool} should NOT be in DEFAULT_EXCLUDE_TOOLS"
            )


# =============================================================================
# TestSmartCrusherFallback — PR #704 regression suite
# =============================================================================


class TestSmartCrusherFallback:
    """Verify SmartCrusher→Kompress→Log fallback chain.

    The post-strategy unified fallback block (added in PR #704) replaces
    inline duplicate Kompress invocations. When SmartCrusher returns no
    savings, the unified block tries Kompress, then Log (structurally
    repetitive content), without double-invoking Kompress.

    Uses ``_apply_strategy_to_content`` + monkeypatched fallback
    compressors to avoid network/ML-model downloads in test environments.
    """

    def test_smart_crusher_with_no_savings_triggers_kompress_fallback(self, router, monkeypatch):
        """When SmartCrusher produces no savings (returns content unchanged),
        the unified post-strategy block must fire Kompress fallback.

        Monkeypatches ``_get_smart_crusher`` to return a mock whose
        ``crush()`` returns *content* unchanged — this simulates "ran
        but produced no savings" without depending on the Rust
        ``headroom._core`` extension or an LLM round-trip.
        """
        from unittest.mock import MagicMock

        import headroom.transforms.content_router as crm
        from headroom.transforms.smart_crusher import CrushResult

        content = "this is repetitive text. " * 300

        # Mock SmartCrusher: ran successfully but returned content as-is
        # (no savings), so the unified fallback block is entered.
        mock_crush_result = CrushResult(
            compressed=content,
            original=content,
            was_modified=False,
            strategy="passthrough",
        )
        mock_crusher = MagicMock()
        mock_crusher.crush.return_value = mock_crush_result
        monkeypatch.setattr(
            crm.ContentRouter,
            "_get_smart_crusher",
            lambda self: mock_crusher,
        )

        # Patch _try_ml_compressor to simulate Kompress also returning
        # unchanged (no savings), forcing the full chain to exercise
        monkeypatch.setattr(
            crm.ContentRouter,
            "_try_ml_compressor",
            lambda self, c, context="", question=None: (
                c,
                len(c.split()),
            ),
        )

        compressed, compressed_tokens, strategy_chain = router._apply_strategy_to_content(
            content,
            CompressionStrategy.SMART_CRUSHER,
            context="",
        )

        # Strategy chain must include smart_crusher
        assert "smart_crusher" in strategy_chain
        # Kompress fallback should have been attempted
        assert "kompress" in strategy_chain, (
            f"Expected kompress in chain {strategy_chain} — "
            f"unified post-strategy block should have fired"
        )

    def test_smart_crusher_json_compresses_directly(self, router, monkeypatch):
        """When SmartCrusher successfully compresses JSON, the chain is
        just [smart_crusher] with no fallback entries.

        Uses a mock SmartCrusher to avoid depending on the Rust
        ``headroom._core`` extension in test environments.
        """
        import json
        from unittest.mock import MagicMock

        import headroom.transforms.content_router as crm
        from headroom.transforms.smart_crusher import CrushResult

        content = json.dumps([{"id": i, "name": f"item_{i}", "value": i * 10} for i in range(100)])

        # Mock SmartCrusher: simulated compression (shorter output)
        mock_compressed = json.dumps([{"id": i, "name": f"item_{i}"} for i in range(50)])
        mock_crush_result = CrushResult(
            compressed=mock_compressed,
            original=content,
            was_modified=True,
            strategy="smart_crusher",
        )
        mock_crusher = MagicMock()
        mock_crusher.crush.return_value = mock_crush_result
        monkeypatch.setattr(
            crm.ContentRouter,
            "_get_smart_crusher",
            lambda self: mock_crusher,
        )

        compressed, compressed_tokens, strategy_chain = router._apply_strategy_to_content(
            content,
            CompressionStrategy.SMART_CRUSHER,
            context="",
        )

        # SmartCrusher should handle JSON directly
        assert "smart_crusher" in strategy_chain
        # With real savings, no fallback should be triggered
        assert "kompress" not in strategy_chain
        assert len(compressed.strip()) > 0

    def test_post_strategy_block_no_duplicate_kompress(self, router, monkeypatch):
        """The unified post-strategy block must NOT produce duplicate
        'kompress' entries in the strategy chain.

        Pre-PR #704: an inline duplicate Kompress fallback existed for
        SmartCrusher that could fire alongside the post-strategy block,
        causing 'kompress' to appear twice in the chain.

        Uses a mock SmartCrusher returning no savings so the fallback
        block is entered deterministically, without depending on the
        Rust ``headroom._core`` extension.
        """
        from unittest.mock import MagicMock

        import headroom.transforms.content_router as crm
        from headroom.transforms.smart_crusher import CrushResult

        repetitive = "line " * 300 + "\n"

        # Mock SmartCrusher: ran successfully but returned content as-is
        # (no savings) — fallback block must fire.
        mock_crush_result = CrushResult(
            compressed=repetitive,
            original=repetitive,
            was_modified=False,
            strategy="passthrough",
        )
        mock_crusher = MagicMock()
        mock_crusher.crush.return_value = mock_crush_result
        monkeypatch.setattr(
            crm.ContentRouter,
            "_get_smart_crusher",
            lambda self: mock_crusher,
        )

        # Monkeypatch Kompress to return unchanged (no savings),
        # forcing the full fallback chain without network access
        monkeypatch.setattr(
            crm.ContentRouter,
            "_try_ml_compressor",
            lambda self, c, context="", question=None: (
                c,
                len(c.split()),
            ),
        )

        compressed, compressed_tokens, strategy_chain = router._apply_strategy_to_content(
            repetitive,
            CompressionStrategy.SMART_CRUSHER,
            context="",
        )

        # The chain must include the requested strategy
        assert "smart_crusher" in strategy_chain

        # No duplicate "kompress" entries — the key regression check
        kompress_count = strategy_chain.count("kompress")
        assert kompress_count <= 1, (
            f"Kompress appeared {kompress_count} times in chain; "
            f"duplicate fallback suggests inline+post-strategy both fired: "
            f"{strategy_chain}"
        )

    def test_code_aware_fallback_also_uses_unified_block(self, router, monkeypatch):
        """CodeAware strategy also uses the unified fallback block.
        Verify it doesn't double-invoke Kompress either."""
        import headroom.transforms.content_router as crm

        monkeypatch.setattr(
            crm.ContentRouter,
            "_try_ml_compressor",
            lambda self, content, context="", question=None: (
                content,
                len(content.split()),
            ),
        )

        plain = "This is just plain text. " * 200

        compressed, compressed_tokens, strategy_chain = router._apply_strategy_to_content(
            plain,
            CompressionStrategy.CODE_AWARE,
            context="",
        )

        # CodeAware should be in the chain
        assert "code_aware" in strategy_chain

        # No duplicate fallback entries
        kompress_count = strategy_chain.count("kompress")
        assert kompress_count <= 1, (
            f"Kompress appeared {kompress_count} times; "
            f"duplicate fallback in CodeAware path: {strategy_chain}"
        )


# =============================================================================
# TestCompressBlockContent — PR #704 shared-path regression
# =============================================================================


class TestCompressBlockContent:
    """Verify `_compress_block_content` shared path for tool_result and text blocks.

    Before PR #704, the two block paths had ~60 lines of duplicate cache
    logic each. The shared helper ensures both paths stay in sync (cache
    expiry, pinning, ratio gating).

    Tests target the two-tier ``CompressionCache`` (content_router-local,
    line 191) and the ``_compress_block_content`` method directly,
    avoiding the Rust content-detection extension by pre-populating
    the cache and verifying cache-hit/skip behaviour.
    """

    @pytest.fixture
    def router_with_cache(self):
        """ContentRouter with all compressors enabled."""
        config = ContentRouterConfig(
            enable_smart_crusher=True,
            enable_kompress=True,
            enable_log_compressor=True,
            min_section_tokens=10,
        )
        return ContentRouter(config)

    def test_skip_set_prevents_recompression(self, router_with_cache):
        """Tier 1 (skip set): content_key in the skip set returns
        (None, False) immediately — no compression attempted."""
        cache = router_with_cache._cache
        key = hash("test-content-that-wont-compress")

        # Mark as skipped
        cache.mark_skip(key)
        assert cache.is_skipped(key) is True

        # _compress_block_content should return early on skip
        compressed, was_compressed = router_with_cache._compress_block_content(
            content="test-content-that-wont-compress",
            content_key=key,
            context="",
            bias=1.0,
            min_ratio=0.5,
            compressor_timing=None,
            transforms_applied=[],
            route_counts=None,
            compressed_details=None,
            strategy_label="test",
            details_prefix="test",
        )

        assert compressed is None
        assert was_compressed is False

    def test_result_cache_hit_returns_cached(self, router_with_cache):
        """Tier 2 (result cache): cached content is returned without
        re-running compression."""
        cache = router_with_cache._cache
        key = hash("cacheable-content")
        original = "compressed-version-of-content"

        # Populate result cache
        cache.put(key, original, ratio=0.3, strategy="kompress")
        assert cache.get(key) == (original, 0.3, "kompress")

        # _compress_block_content should return cached result
        compressed, was_compressed = router_with_cache._compress_block_content(
            content="cacheable-content",
            content_key=key,
            context="",
            bias=1.0,
            min_ratio=0.5,
            compressor_timing=None,
            transforms_applied=[],
            route_counts=None,
            compressed_details=None,
            strategy_label="test",
            details_prefix="test",
        )

        assert compressed == original
        assert was_compressed is True

    def test_result_cache_ratio_above_min_moves_to_skip(self, router_with_cache):
        """When the cached ratio is ≥ min_ratio, the entry is moved from
        Tier 2 to Tier 1 (skip set) — ratio threshold has tightened."""
        cache = router_with_cache._cache
        key = hash("borderline-content")

        # Cached with ratio 0.8 (high — barely compressed)
        cache.put(key, "slightly-compressed", ratio=0.8, strategy="text")

        # min_ratio=0.7 — cached ratio (0.8) ≥ threshold → move to skip
        compressed, was_compressed = router_with_cache._compress_block_content(
            content="borderline-content",
            content_key=key,
            context="",
            bias=1.0,
            min_ratio=0.7,
            compressor_timing=None,
            transforms_applied=[],
            route_counts=None,
            compressed_details=None,
            strategy_label="test",
            details_prefix="test",
        )

        assert compressed is None, "Should move to skip when ratio ≥ min_ratio"
        assert was_compressed is False
        assert cache.is_skipped(key), "Entry should now be in skip set"
        assert cache.get(key) is None, "Entry should be removed from result cache"

    def test_compress_block_content_route_counts_mutated(self, router_with_cache):
        """route_counts dict is mutated in-place with cache hit/miss info."""
        cache = router_with_cache._cache
        key_skip = hash("skip-content")
        key_hit = hash("hit-content")

        cache.mark_skip(key_skip)
        cache.put(key_hit, "compressed", ratio=0.3, strategy="kompress")

        route_counts: dict[str, int] = {}

        # Skip hit
        router_with_cache._compress_block_content(
            content="skip-content",
            content_key=key_skip,
            context="",
            bias=1.0,
            min_ratio=0.5,
            compressor_timing=None,
            transforms_applied=[],
            route_counts=route_counts,
            compressed_details=None,
            strategy_label="test",
            details_prefix="test",
        )
        assert route_counts.get("ratio_too_high", 0) >= 1
        assert route_counts.get("cache_hit", 0) >= 1

        # Cache hit
        router_with_cache._compress_block_content(
            content="hit-content",
            content_key=key_hit,
            context="",
            bias=1.0,
            min_ratio=0.5,
            compressor_timing=None,
            transforms_applied=[],
            route_counts=route_counts,
            compressed_details=None,
            strategy_label="test",
            details_prefix="test",
        )

    def test_compress_block_content_transforms_applied_mutated(self, router_with_cache):
        """transforms_applied list is mutated with strategy info on cache hit."""
        cache = router_with_cache._cache
        key = hash("transform-test-content")
        cache.put(key, "short", ratio=0.25, strategy="kompress")

        transforms_applied: list[str] = []
        router_with_cache._compress_block_content(
            content="transform-test-content",
            content_key=key,
            context="",
            bias=1.0,
            min_ratio=0.5,
            compressor_timing=None,
            transforms_applied=transforms_applied,
            route_counts=None,
            compressed_details=None,
            strategy_label="tool_result",
            details_prefix="tool",
        )

        assert any("router:tool_result" in t for t in transforms_applied), (
            f"Expected router:tool_result:* in transforms, got: {transforms_applied}"
        )


# =============================================================================
# Mixed content: custom-tag protection (system-reminder mangling regression)
# =============================================================================


class TestMixedContentTagProtection:
    """_compress_mixed must protect custom-tag blocks BEFORE section split.

    Splitting first lands the open/close tags of a
    ``<system-reminder>...</system-reminder>`` pair in different sections;
    per-section protection then sees only unmatched tags (which protect
    nothing) and the block's content — Claude Code ships CLAUDE.md this way —
    is lossy-compressed and arrives word-dropped.
    """

    REMINDER = (
        "<system-reminder>\n"
        "Instruction prose that must survive byte-exact.\n\n"
        "```bash\nrtk gain\n```\n\n"
        "More instructions after the fence, also byte-exact.\n"
        "</system-reminder>"
    )

    @staticmethod
    def _mangling_router() -> ContentRouter:
        """Router whose per-section compressor visibly mangles everything."""
        router = ContentRouter(ContentRouterConfig(min_section_tokens=1))

        def mangle(content, strategy, context, language=None, question=None, bias=1.0):
            return "MANGLED", 1, None

        router._apply_strategy_to_content = mangle  # type: ignore[method-assign]
        return router

    def test_reminder_block_survives_mixed_compression_verbatim(self):
        router = self._mangling_router()
        content = (
            "Prose before the reminder that may compress.\n\n"
            + self.REMINDER
            + "\n\nProse after the reminder that may compress."
        )

        result = router._compress_mixed(content, context="")

        # The tag block (fence and all) is byte-exact in the output...
        assert self.REMINDER in result.compressed
        # ...while content outside it still went through the compressor.
        assert "MANGLED" in result.compressed

    def test_reminder_only_content_passes_through(self):
        router = self._mangling_router()

        result = router._compress_mixed(self.REMINDER, context="")

        assert self.REMINDER in result.compressed
        assert "MANGLED" not in result.compressed

    def test_untagged_mixed_content_still_compresses(self):
        router = self._mangling_router()
        content = "Plain prose section.\n\n```python\nprint('hi')\n```\n\nMore prose."

        result = router._compress_mixed(content, context="")

        assert "MANGLED" in result.compressed
        assert result.strategy_used == CompressionStrategy.MIXED
