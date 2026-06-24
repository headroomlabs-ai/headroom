"""Tests for MCP (Model Context Protocol) integration.

These tests verify that Headroom correctly compresses MCP tool outputs
while preserving 100% of critical data (errors, anomalies).
"""

import json
import random
from datetime import datetime, timedelta

import pytest

from headroom.integrations.mcp import (
    HeadroomMCPClientWrapper,
    HeadroomMCPCompressor,
    MCPCompressionResult,
    MCPToolProfile,
    compress_tool_result,
    compress_tool_result_with_metrics,
)
from headroom.providers import OpenAIProvider
from headroom.transforms.smart_crusher import strip_ccr_sentinels

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mcp_compressor():
    """Create MCP compressor with default settings."""
    return HeadroomMCPCompressor()


@pytest.fixture
def tokenizer():
    """Create token counter."""
    provider = OpenAIProvider()
    return provider.get_token_counter("gpt-4o")


def generate_slack_messages(count: int, error_rate: float = 0.1) -> str:
    """Generate realistic Slack search results."""
    channels = ["#engineering", "#incidents", "#support", "#general", "#alerts"]
    users = ["alice", "bob", "charlie", "diana", "eve"]

    messages = []
    for i in range(count):
        is_error = random.random() < error_rate
        if is_error:
            text = random.choice(
                [
                    "ERROR: Database connection failed at 2:30am",
                    "CRITICAL: API latency spike detected",
                    "Exception: NullPointerException in AuthService",
                    "FAILED: Build pipeline broke on main branch",
                    "BUG: Users can't login - investigating now",
                ]
            )
        else:
            text = random.choice(
                [
                    "Reviewed the PR, looks good to merge",
                    "Updated the docs with new API endpoints",
                    "Meeting notes from standup attached",
                    "Can someone review my changes?",
                    "Deployed v2.3.1 to staging",
                ]
            )

        messages.append(
            {
                "id": f"msg_{i}",
                "channel": random.choice(channels),
                "user": random.choice(users),
                "text": text,
                "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
                "reactions": random.randint(0, 10),
            }
        )

    return json.dumps({"messages": messages, "total": count})


def generate_database_results(count: int, null_rate: float = 0.1) -> str:
    """Generate realistic database query results."""
    rows = []
    for i in range(count):
        has_null = random.random() < null_rate
        has_error = random.random() < 0.05  # 5% error rate

        row = {
            "id": i + 1,
            "user_id": f"user_{random.randint(1000, 9999)}",
            "email": f"user{i}@example.com",
            "status": "ERROR: validation failed"
            if has_error
            else random.choice(["active", "inactive", "pending"]),
            "created_at": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
            "balance": None if has_null else round(random.uniform(0, 10000), 2),
        }
        rows.append(row)

    return json.dumps({"rows": rows, "count": count})


def generate_log_entries(count: int, error_rate: float = 0.15) -> str:
    """Generate realistic log entries."""
    services = ["api-gateway", "auth-service", "payment-service", "user-service"]

    entries = []
    for i in range(count):
        if random.random() < error_rate:
            level = random.choice(["ERROR", "FATAL"])
            message = random.choice(
                [
                    "Connection timeout to database",
                    "Failed to process payment: insufficient funds",
                    "Authentication failed for user",
                    "Memory limit exceeded",
                    "Unhandled exception in request handler",
                ]
            )
        else:
            level = random.choice(["DEBUG", "INFO", "INFO", "INFO", "WARN"])
            message = random.choice(
                [
                    "Request processed successfully",
                    "Cache hit for user data",
                    "Starting health check",
                    "Connection pool recycled",
                    "Metrics exported",
                ]
            )

        entries.append(
            {
                "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                "level": level,
                "service": random.choice(services),
                "message": message,
                "trace_id": f"trace_{random.randint(100000, 999999)}",
            }
        )

    return json.dumps({"entries": entries})


def generate_github_issues(count: int, bug_rate: float = 0.2) -> str:
    """Generate realistic GitHub issues."""
    labels_pool = ["enhancement", "documentation", "question", "good first issue"]
    bug_labels = ["bug", "critical", "urgent", "blocker"]

    issues = []
    for i in range(count):
        is_bug = random.random() < bug_rate
        labels = (
            random.sample(bug_labels, k=random.randint(1, 2))
            if is_bug
            else random.sample(labels_pool, k=random.randint(0, 2))
        )

        issues.append(
            {
                "number": i + 1,
                "title": f"{'BUG: ' if is_bug else ''}{random.choice(['Fix login flow', 'Update API docs', 'Add dark mode', 'Improve performance'])}",
                "state": random.choice(["open", "closed"]),
                "labels": labels,
                "author": f"user{random.randint(1, 100)}",
                "created_at": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                "comments": random.randint(0, 20),
            }
        )

    return json.dumps({"issues": issues, "total_count": count})


# ============================================================================
# Test Classes
# ============================================================================


class TestMCPToolProfiles:
    """Test tool profile matching."""

    def test_slack_profile_match(self, mcp_compressor):
        """Slack tools should match slack profile."""
        profile = mcp_compressor.get_profile("mcp__slack__search")
        assert "slack" in profile.tool_name_pattern
        assert profile.max_items == 25

    def test_database_profile_match(self, mcp_compressor):
        """Database tools should match database profile."""
        profile = mcp_compressor.get_profile("mcp__database__query")
        assert "database" in profile.tool_name_pattern or "sql" in profile.tool_name_pattern
        assert profile.max_items == 30

    def test_github_profile_match(self, mcp_compressor):
        """GitHub tools should match github profile."""
        profile = mcp_compressor.get_profile("mcp__github__list_issues")
        assert "github" in profile.tool_name_pattern

    def test_log_profile_match(self, mcp_compressor):
        """Log tools should match log profile with higher max_items."""
        profile = mcp_compressor.get_profile("search_logs")
        assert "log" in profile.tool_name_pattern
        assert profile.max_items == 40  # Logs get more items

    def test_fallback_profile(self, mcp_compressor):
        """Unknown tools should get fallback profile."""
        profile = mcp_compressor.get_profile("some_random_tool")
        assert profile.max_items == 20  # Default


class TestMCPCompressionBasics:
    """Test basic compression functionality."""

    def test_compress_returns_result(self, mcp_compressor):
        """Compression should return MCPCompressionResult."""
        content = generate_slack_messages(100)
        result = mcp_compressor.compress(
            content=content,
            tool_name="slack_search",
            user_query="find errors",
        )
        assert isinstance(result, MCPCompressionResult)

    def test_compress_reduces_tokens(self, mcp_compressor):
        """Compression should reduce token count."""
        content = generate_slack_messages(200)
        result = mcp_compressor.compress(
            content=content,
            tool_name="slack_search",
            user_query="find errors",
        )
        assert result.compressed_tokens < result.original_tokens
        assert result.tokens_saved > 0

    def test_compress_tracks_metrics(self, mcp_compressor):
        """Compression should track items before/after."""
        content = generate_slack_messages(100)
        result = mcp_compressor.compress(
            content=content,
            tool_name="slack_search",
        )
        assert result.items_before == 100
        assert result.items_after is not None
        assert result.items_after < result.items_before

    def test_small_content_not_compressed(self, mcp_compressor):
        """Small content should not be compressed."""
        content = generate_slack_messages(5)  # Very small
        result = mcp_compressor.compress(
            content=content,
            tool_name="slack_search",
        )
        assert result.was_compressed is False
        assert result.compressed_content == content


class TestMCPErrorPreservation:
    """Test error preservation - SmartCrusher detects errors via field patterns."""

    def test_all_log_errors_preserved(self, mcp_compressor):
        """100% of ERROR/FATAL log entries must be preserved.

        SmartCrusher detects errors via the 'level' field containing ERROR/FATAL.
        This is the strongest error preservation guarantee.
        """
        random.seed(43)
        content = generate_log_entries(200, error_rate=0.25)
        data = json.loads(content)

        original_errors = [e for e in data["entries"] if e["level"] in ["ERROR", "FATAL"]]

        result = mcp_compressor.compress(
            content=content,
            tool_name="search_logs",
            user_query="find errors",
        )

        compressed_data = json.loads(result.compressed_content)
        compressed_errors = [
            e
            for e in strip_ccr_sentinels(compressed_data["entries"])
            if e["level"] in ["ERROR", "FATAL"]
        ]

        # CRITICAL: 100% of errors must be preserved
        assert len(compressed_errors) >= len(original_errors), (
            f"Lost errors: {len(original_errors)} -> {len(compressed_errors)}"
        )

    def test_slack_significant_compression_with_content(self, mcp_compressor):
        """Slack messages should compress while preserving error keywords in text."""
        random.seed(42)
        content = generate_slack_messages(200, error_rate=0.2)

        result = mcp_compressor.compress(
            content=content,
            tool_name="slack_search",
            user_query="find errors",
        )

        # Should achieve significant compression
        assert result.compression_ratio > 0.5

        compressed_data = json.loads(result.compressed_content)
        # Should preserve some messages with error keywords (SmartCrusher detects these)
        error_msgs = [
            m
            for m in strip_ccr_sentinels(compressed_data["messages"])
            if any(kw in m["text"].lower() for kw in ["error", "failed", "exception"])
        ]
        assert len(error_msgs) > 0, "Should preserve some error messages"

    def test_database_error_status_preserved(self, mcp_compressor):
        """Database rows with ERROR status should be preserved."""
        random.seed(44)
        content = generate_database_results(150, null_rate=0.15)
        data = json.loads(content)

        original_errors = [r for r in data["rows"] if "error" in str(r["status"]).lower()]

        result = mcp_compressor.compress(
            content=content,
            tool_name="database_query",
            user_query="find errors",
        )

        compressed_data = json.loads(result.compressed_content)
        compressed_errors = [
            r
            for r in strip_ccr_sentinels(compressed_data["rows"])
            if "error" in str(r["status"]).lower()
        ]

        # Should preserve most error rows
        assert len(compressed_errors) >= len(original_errors) * 0.8, (
            f"Lost too many errors: {len(original_errors)} -> {len(compressed_errors)}"
        )

    def test_github_bugs_partial_preservation(self, mcp_compressor):
        """GitHub bug issues should have partial preservation."""
        random.seed(45)
        content = generate_github_issues(100, bug_rate=0.3)

        result = mcp_compressor.compress(
            content=content,
            tool_name="github_issues",
            user_query="find bugs",
        )

        compressed_data = json.loads(result.compressed_content)
        # Should preserve at least some bugs
        compressed_bugs = [
            i
            for i in strip_ccr_sentinels(compressed_data["issues"])
            if any(label in ["bug", "critical", "urgent", "blocker"] for label in i["labels"])
        ]

        # At least 5 bugs should be preserved
        assert len(compressed_bugs) >= 5, "Should preserve at least 5 bug issues"


class TestMCPStandaloneFunction:
    """Test the standalone compress_tool_result function."""

    def test_standalone_returns_string(self):
        """Standalone function should return compressed string."""
        content = generate_slack_messages(100)
        result = compress_tool_result(
            content=content,
            tool_name="slack_search",
            tool_args={"query": "errors"},
            user_query="find errors in slack",
        )
        assert isinstance(result, str)
        # Should be valid JSON
        json.loads(result)

    def test_standalone_with_metrics(self):
        """Standalone function with metrics should return MCPCompressionResult."""
        content = generate_log_entries(150)
        result = compress_tool_result_with_metrics(
            content=content,
            tool_name="search_logs",
            tool_args={"service": "api"},
            user_query="find errors",
        )
        assert isinstance(result, MCPCompressionResult)
        assert result.tool_name == "search_logs"


class TestMCPClientWrapper:
    """Test the async client wrapper."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create a mock MCP client."""

        class MockMCPClient:
            async def call_tool(self, name: str, arguments: dict | None = None) -> str:
                if "slack" in name:
                    return generate_slack_messages(100)
                elif "log" in name:
                    return generate_log_entries(150)
                else:
                    return generate_database_results(80)

        return MockMCPClient()

    @pytest.mark.asyncio
    async def test_wrapper_compresses_automatically(self, mock_mcp_client):
        """Wrapper should automatically compress tool results."""
        wrapper = HeadroomMCPClientWrapper(mock_mcp_client)

        result = await wrapper.call_tool("slack_search", {"query": "test"})

        # Result should be valid JSON
        data = json.loads(result)
        # Should be compressed (fewer items)
        assert len(data["messages"]) < 100

    @pytest.mark.asyncio
    async def test_wrapper_tracks_metrics(self, mock_mcp_client):
        """Wrapper should track compression metrics."""
        wrapper = HeadroomMCPClientWrapper(mock_mcp_client)

        await wrapper.call_tool("slack_search", {"query": "test"})
        await wrapper.call_tool("search_logs", {"service": "api"})

        metrics = wrapper.get_metrics()
        assert len(metrics) == 2
        assert metrics[0].tool_name == "slack_search"
        assert metrics[1].tool_name == "search_logs"

    @pytest.mark.asyncio
    async def test_wrapper_total_tokens_saved(self, mock_mcp_client):
        """Wrapper should track total tokens saved."""
        wrapper = HeadroomMCPClientWrapper(mock_mcp_client)

        await wrapper.call_tool("slack_search", {"query": "test"})
        await wrapper.call_tool("search_logs", {"service": "api"})

        total_saved = wrapper.get_total_tokens_saved()
        assert total_saved > 0


class TestMCPCompressionRatio:
    """Test compression efficiency."""

    def test_significant_compression_slack(self, mcp_compressor):
        """Slack results should compress well (>50%)."""
        content = generate_slack_messages(200)
        result = mcp_compressor.compress(
            content=content,
            tool_name="slack_search",
        )
        assert result.compression_ratio > 0.5, (
            f"Compression ratio too low: {result.compression_ratio:.2%}"
        )

    def test_significant_compression_logs(self, mcp_compressor):
        """Log entries should compress well (>50%)."""
        content = generate_log_entries(200)
        result = mcp_compressor.compress(
            content=content,
            tool_name="search_logs",
        )
        assert result.compression_ratio > 0.5, (
            f"Compression ratio too low: {result.compression_ratio:.2%}"
        )

    def test_compression_efficiency_increases_with_size(self, mcp_compressor):
        """Larger outputs should compress more efficiently."""
        small = generate_slack_messages(50)
        large = generate_slack_messages(500)

        small_result = mcp_compressor.compress(small, "slack_search")
        large_result = mcp_compressor.compress(large, "slack_search")

        # Large should have higher compression ratio
        assert large_result.compression_ratio >= small_result.compression_ratio


class TestMCPSchemaPreservation:
    """Test that JSON schema is preserved."""

    def test_schema_preserved_slack(self, mcp_compressor):
        """Slack message schema should be preserved."""
        content = generate_slack_messages(100)
        result = mcp_compressor.compress(content, "slack_search")

        compressed_data = json.loads(result.compressed_content)
        assert "messages" in compressed_data
        assert len(compressed_data["messages"]) > 0

        # Check first message has all fields
        msg = compressed_data["messages"][0]
        assert "id" in msg
        assert "channel" in msg
        assert "user" in msg
        assert "text" in msg

    def test_schema_preserved_logs(self, mcp_compressor):
        """Log entry schema should be preserved."""
        content = generate_log_entries(100)
        result = mcp_compressor.compress(content, "search_logs")

        compressed_data = json.loads(result.compressed_content)
        assert "entries" in compressed_data
        assert len(compressed_data["entries"]) > 0

        entry = compressed_data["entries"][0]
        assert "timestamp" in entry
        assert "level" in entry
        assert "service" in entry
        assert "message" in entry


class TestMCPEdgeCases:
    """Test edge cases."""

    def test_empty_array(self, mcp_compressor):
        """Empty array should pass through unchanged."""
        content = json.dumps({"messages": []})
        result = mcp_compressor.compress(content, "slack_search")
        assert result.was_compressed is False

    def test_single_item(self, mcp_compressor):
        """Single item should pass through unchanged."""
        content = json.dumps({"messages": [{"id": 1, "text": "test"}]})
        result = mcp_compressor.compress(content, "slack_search")
        compressed = json.loads(result.compressed_content)
        assert len(compressed["messages"]) == 1

    def test_non_json_passthrough(self, mcp_compressor):
        """Non-JSON content should pass through unchanged."""
        content = "This is plain text, not JSON"
        result = mcp_compressor.compress(content, "some_tool")
        assert result.compressed_content == content
        assert result.was_compressed is False

    def test_malformed_json_passthrough(self, mcp_compressor):
        """Malformed JSON should pass through unchanged."""
        content = '{"messages": [broken json'
        result = mcp_compressor.compress(content, "slack_search")
        assert result.compressed_content == content
        assert result.was_compressed is False


class TestMCPContextUsage:
    """Test context extraction for relevance."""

    def test_context_from_user_query(self, mcp_compressor):
        """User query should be used for context."""
        content = generate_slack_messages(100)
        result = mcp_compressor.compress(
            content=content,
            tool_name="slack_search",
            user_query="find authentication errors",
        )
        assert "authentication errors" in result.context_used

    def test_context_from_tool_args(self, mcp_compressor):
        """Tool args should be included in context."""
        content = generate_slack_messages(100)
        result = mcp_compressor.compress(
            content=content,
            tool_name="slack_search",
            tool_args={"channel": "#incidents", "query": "outage"},
        )
        assert "incidents" in result.context_used or "outage" in result.context_used

    def test_combined_context(self, mcp_compressor):
        """Both user query and tool args should be combined."""
        content = generate_slack_messages(100)
        result = mcp_compressor.compress(
            content=content,
            tool_name="slack_search",
            tool_args={"channel": "#alerts"},
            user_query="find database errors",
        )
        assert "database errors" in result.context_used


class TestMCPCustomProfiles:
    """Test custom tool profiles."""

    def test_custom_profile(self):
        """Custom profiles should override defaults."""
        custom_profiles = [
            MCPToolProfile(
                tool_name_pattern=r".*custom.*",
                max_items=10,
                min_tokens_to_compress=100,
            ),
        ]
        compressor = HeadroomMCPCompressor(profiles=custom_profiles)

        profile = compressor.get_profile("custom_tool")
        assert profile.max_items == 10

    def test_profile_disabled(self):
        """Disabled profiles should not compress."""
        custom_profiles = [
            MCPToolProfile(
                tool_name_pattern=r".*",
                enabled=False,
            ),
        ]
        compressor = HeadroomMCPCompressor(profiles=custom_profiles)

        content = generate_slack_messages(200)
        result = compressor.compress(content, "any_tool")
        assert result.was_compressed is False
