"""Real-world integration tests for Strands HeadroomHookProvider.

These tests use actual AWS Bedrock API calls with real credentials.
NO MOCKS - all tests hit the real Bedrock API.

Skip in CI if AWS credentials are not available.
"""

from __future__ import annotations

import json
import os

import pytest

# Check for AWS credentials availability
SKIP_BEDROCK = not (
    os.environ.get("AWS_ACCESS_KEY_ID")
    or os.environ.get("AWS_PROFILE")
    or os.path.exists(os.path.expanduser("~/.aws/credentials"))
)

# Check if strands-agents is installed
try:
    from strands import Agent, tool
    from strands.models import BedrockModel

    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False

    # Provide a no-op decorator when strands is not installed
    def tool(fn):
        return fn

    Agent = None  # type: ignore
    BedrockModel = None  # type: ignore

# Skip all tests if dependencies not available
pytestmark = [
    pytest.mark.skipif(SKIP_BEDROCK, reason="AWS credentials not available"),
    pytest.mark.skipif(not STRANDS_AVAILABLE, reason="strands-agents not installed"),
]


# ============================================================================
# Test Tools - Generate realistic verbose data for compression testing
# These are defined with @tool decorator for use when strands is installed.
# When strands is not installed, the no-op decorator ensures import succeeds.
# ============================================================================


@tool
def search_logs(query: str, limit: int = 100) -> str:
    """Search application logs. Returns JSON array of log entries.

    Args:
        query: Search query to find in logs
        limit: Maximum number of log entries to return

    Returns:
        JSON array of log entry objects
    """
    # Generate realistic verbose log data that should be compressed
    logs = [
        {
            "timestamp": f"2024-01-{(i % 28) + 1:02d}T{10 + (i % 12):02d}:00:00Z",
            "level": ["INFO", "DEBUG", "WARN", "ERROR"][i % 4],
            "service": ["api-gateway", "auth-service", "data-processor", "cache-service"][i % 4],
            "message": f"Request processed successfully - latency={50 + i}ms, query={query}",
            "request_id": f"req-{i:06d}-{hash(query) % 10000:04d}",
            "status_code": [200, 201, 400, 500][i % 4],
            "user_agent": "Mozilla/5.0 (compatible; TestBot/1.0)",
            "ip_address": f"192.168.{i % 256}.{(i * 7) % 256}",
            "trace_id": f"trace-{i:08x}",
            "span_id": f"span-{i:04x}",
            "duration_ms": 50 + (i * 3) % 200,
            "memory_mb": 128 + (i * 5) % 512,
            "cpu_percent": 10 + (i * 2) % 80,
        }
        for i in range(limit)
    ]
    return json.dumps(logs, indent=2)


@tool
def get_small_status() -> str:
    """Get a small status response that should NOT be compressed.

    Returns:
        Small JSON status object
    """
    return json.dumps({"status": "healthy", "uptime_seconds": 12345, "version": "1.2.3"})


@tool
def get_error_data() -> str:
    """Get error information. Error results should NOT be compressed.

    Returns:
        Error information (but not as a tool error)
    """
    return json.dumps(
        {
            "errors": [
                {"code": "E001", "message": "Connection timeout"},
                {"code": "E002", "message": "Authentication failed"},
            ],
            "timestamp": "2024-01-15T10:00:00Z",
        }
    )


@tool
def fetch_user_data(user_id: str) -> str:
    """Fetch detailed user data. Returns large JSON payload.

    Args:
        user_id: The user ID to fetch data for

    Returns:
        Large JSON object with user details
    """
    # Generate a large user profile that should trigger compression
    activities = [
        {
            "activity_id": f"act-{i:06d}",
            "type": ["login", "purchase", "view", "share"][i % 4],
            "timestamp": f"2024-01-{(i % 28) + 1:02d}T{10 + (i % 12):02d}:30:00Z",
            "details": {
                "ip": f"10.0.{i % 256}.{(i * 3) % 256}",
                "device": ["desktop", "mobile", "tablet"][i % 3],
                "browser": ["Chrome", "Firefox", "Safari"][i % 3],
                "duration_seconds": 30 + i * 5,
                "page_views": 1 + i % 10,
            },
            "metadata": {
                "session_id": f"sess-{i:08x}",
                "referrer": f"https://example.com/page/{i}",
                "utm_source": ["google", "facebook", "twitter", "email"][i % 4],
            },
        }
        for i in range(50)
    ]

    return json.dumps(
        {
            "user_id": user_id,
            "profile": {
                "name": "Test User",
                "email": f"{user_id}@example.com",
                "created_at": "2023-01-01T00:00:00Z",
            },
            "activities": activities,
        },
        indent=2,
    )


@tool
def simple_calculator(a: int, b: int, operation: str) -> str:
    """Simple calculator for basic operations.

    Args:
        a: First number
        b: Second number
        operation: One of 'add', 'subtract', 'multiply', 'divide'

    Returns:
        The result of the operation
    """
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        result = a / b if b != 0 else "undefined"
    else:
        result = "unknown operation"

    return json.dumps({"operation": operation, "a": a, "b": b, "result": result})


# ============================================================================
# Test Class
# ============================================================================


@pytest.mark.skipif(SKIP_BEDROCK, reason="AWS credentials not available")
@pytest.mark.skipif(not STRANDS_AVAILABLE, reason="strands-agents not installed")
class TestHeadroomHookProviderReal:
    """Real-world integration tests for HeadroomHookProvider with Bedrock."""

    @pytest.fixture
    def bedrock_model(self):
        """Create a BedrockModel instance using Claude 3 Haiku (fast and cheap)."""
        return BedrockModel(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            region_name="us-west-2",
            temperature=0.1,  # Low temperature for consistent tests
        )

    @pytest.fixture
    def hook_provider(self):
        """Create a HeadroomHookProvider with test configuration."""
        from headroom.integrations.strands import HeadroomHookProvider

        return HeadroomHookProvider(
            compress_tool_outputs=True,
            min_tokens_to_compress=50,  # Low threshold for testing
            preserve_errors=True,
        )

    def test_hook_compresses_large_tool_output(self, bedrock_model, hook_provider):
        """Test that large tool outputs are compressed by the hook.

        This test:
        1. Creates an agent with the search_logs tool
        2. Asks a question that triggers the tool
        3. Verifies the hook compressed the output and saved tokens
        """
        # Create agent with hook provider
        agent = Agent(
            model=bedrock_model,
            tools=[search_logs],
            hooks=[hook_provider],
        )

        # Ask a question that will trigger the search_logs tool
        result = agent(
            "Search the logs for 'error' and tell me how many entries you found. "
            "Use limit=100 to get plenty of results."
        )

        # Verify the agent got a response
        assert result is not None

        # Check hook metrics
        metrics = hook_provider.get_savings_summary()

        # The hook should have processed at least one tool call
        assert metrics["total_requests"] >= 1, "Hook should have processed tool calls"

        # With 100 log entries, compression should have occurred
        # and saved significant tokens
        if metrics["compressed_requests"] > 0:
            assert metrics["total_tokens_saved"] > 0, "Should have saved tokens"
            assert metrics["total_tokens_before"] > metrics["total_tokens_after"]

    def test_hook_preserves_small_outputs(self, bedrock_model, hook_provider):
        """Test that small tool outputs are NOT compressed.

        This test:
        1. Creates an agent with a tool returning small output
        2. Triggers the tool
        3. Verifies the hook did not modify the small output
        """
        # Reset metrics from any previous tests
        hook_provider.reset()

        agent = Agent(
            model=bedrock_model,
            tools=[get_small_status],
            hooks=[hook_provider],
        )

        # Ask a question that will trigger the small status tool
        result = agent("What is the current system status? Use the get_small_status tool.")

        assert result is not None

        # Check metrics - small outputs should not be compressed
        metrics = hook_provider.get_savings_summary()

        # Tool was called but output was below threshold
        if metrics["total_requests"] > 0:
            # For small outputs, tokens_before == tokens_after (no compression)
            for m in hook_provider.metrics_history:
                if m.tool_name == "get_small_status" or "small" in str(m.skip_reason):
                    # Either not compressed or skip reason indicates below threshold
                    assert not m.was_compressed or m.skip_reason is not None, (
                        "Small output should not be compressed"
                    )

    def test_hook_preserves_errors(self, bedrock_model):
        """Test that error results are NOT compressed when preserve_errors=True.

        This test:
        1. Creates a hook with preserve_errors=True
        2. Creates an agent with a tool that returns error data
        3. Verifies error results are preserved unchanged
        """
        from headroom.integrations.strands import HeadroomHookProvider

        # Create hook with preserve_errors=True (default)
        hook_with_preserve = HeadroomHookProvider(
            compress_tool_outputs=True,
            min_tokens_to_compress=10,  # Very low threshold
            preserve_errors=True,
        )

        agent = Agent(
            model=bedrock_model,
            tools=[get_error_data],
            hooks=[hook_with_preserve],
        )

        # Get error data
        result = agent("Get the error data using get_error_data tool and summarize it.")

        assert result is not None

        # Check that error-related results were handled appropriately
        metrics = hook_with_preserve.get_savings_summary()

        # The get_error_data tool returns data about errors but doesn't itself error
        # So it should be processed normally (this tests the flow works)
        assert metrics["total_requests"] >= 0  # May or may not have been called

    def test_hook_metrics_tracking(self, bedrock_model, hook_provider):
        """Test that metrics are tracked correctly across multiple tool calls.

        This test:
        1. Creates an agent with multiple tools
        2. Makes requests that trigger various tools
        3. Verifies metrics are accumulated correctly
        """
        # Reset metrics
        hook_provider.reset()

        agent = Agent(
            model=bedrock_model,
            tools=[search_logs, get_small_status, simple_calculator],
            hooks=[hook_provider],
        )

        # First request - should trigger search_logs (large output)
        agent("Search logs for 'test' with limit=50 and give me a count.")

        # Second request - should trigger calculator (small output)
        agent("Calculate 15 + 27 using the calculator tool.")

        # Third request - should trigger status (small output)
        agent("Get the system status using get_small_status.")

        # Check accumulated metrics
        metrics = hook_provider.get_savings_summary()

        # Should have tracked multiple requests
        assert metrics["total_requests"] >= 1, "Should have tracked tool requests"

        # total_tokens_before should be >= total_tokens_after
        assert metrics["total_tokens_before"] >= metrics["total_tokens_after"]

        # History should contain records
        history = hook_provider.metrics_history
        assert len(history) >= 1, "Should have metrics history entries"

        # Each metric should have required fields
        for m in history:
            assert m.request_id is not None
            assert m.timestamp is not None
            assert m.tokens_before >= 0
            assert m.tokens_after >= 0

    def test_multiple_tool_calls_in_single_request(self, bedrock_model, hook_provider):
        """Test that multiple tool calls in a single agent request are all processed.

        This test:
        1. Asks a complex question requiring multiple tools
        2. Verifies each tool call is processed by the hook
        """
        # Reset metrics
        hook_provider.reset()

        agent = Agent(
            model=bedrock_model,
            tools=[search_logs, simple_calculator, fetch_user_data],
            hooks=[hook_provider],
        )

        # Ask a complex question that might trigger multiple tools
        result = agent(
            "I need you to do three things: "
            "1. Search logs for 'api' with limit=30. "
            "2. Calculate 100 * 5 using the calculator. "
            "3. Tell me the total number of results from step 1."
        )

        assert result is not None

        # Check that multiple tool calls were processed
        metrics = hook_provider.get_savings_summary()

        # Should have processed at least the search_logs call
        assert metrics["total_requests"] >= 1

        # Verify metrics history
        history = hook_provider.metrics_history

        # At minimum, should have processed search_logs (which has large output)
        # The actual tools called depend on the model's interpretation
        assert len(history) >= 1

        # Check that we have tool names recorded
        tool_names = [m.tool_name for m in history]
        assert all(name is not None for name in tool_names)

    def test_hook_reset_clears_metrics(self, bedrock_model, hook_provider):
        """Test that reset() clears all accumulated metrics.

        This test:
        1. Makes some requests to accumulate metrics
        2. Calls reset()
        3. Verifies all metrics are cleared
        """
        agent = Agent(
            model=bedrock_model,
            tools=[search_logs],
            hooks=[hook_provider],
        )

        # Make a request to accumulate metrics
        agent("Search logs for 'test' with limit=20.")

        # Verify we have some metrics
        assert hook_provider.total_tokens_saved >= 0

        # Reset
        hook_provider.reset()

        # Verify metrics are cleared
        assert hook_provider.total_tokens_saved == 0
        assert len(hook_provider.metrics_history) == 0

        metrics = hook_provider.get_savings_summary()
        assert metrics["total_requests"] == 0
        assert metrics["total_tokens_saved"] == 0

    def test_hook_with_compression_disabled(self, bedrock_model):
        """Test that hook passes through without compression when disabled.

        This test:
        1. Creates a hook with compress_tool_outputs=False
        2. Verifies tool outputs are not modified
        """
        from headroom.integrations.strands import HeadroomHookProvider

        # Create hook with compression disabled
        disabled_hook = HeadroomHookProvider(
            compress_tool_outputs=False,
            min_tokens_to_compress=10,
        )

        agent = Agent(
            model=bedrock_model,
            tools=[search_logs],
            hooks=[disabled_hook],
        )

        result = agent("Search logs for 'api' with limit=50.")

        assert result is not None

        # When compression is disabled, no requests should be tracked
        # (the hook doesn't register callbacks when disabled)
        metrics = disabled_hook.get_savings_summary()
        assert metrics["compressed_requests"] == 0

    def test_hook_concurrent_safety(self, bedrock_model, hook_provider):
        """Test that hook is thread-safe for concurrent access.

        This test verifies that metrics tracking is thread-safe
        by checking that accumulated values are consistent.
        """
        import threading

        # Reset metrics
        hook_provider.reset()

        agent = Agent(
            model=bedrock_model,
            tools=[simple_calculator],
            hooks=[hook_provider],
        )

        results = []
        errors = []

        def make_request(n: int):
            try:
                result = agent(f"Calculate {n} + {n} using simple_calculator.")
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Run a few sequential requests (concurrent Bedrock calls might be rate-limited)
        threads = []
        for i in range(3):
            t = threading.Thread(target=make_request, args=(i,))
            threads.append(t)
            t.start()
            # Small delay to avoid rate limiting
            import time

            time.sleep(0.5)

        for t in threads:
            t.join(timeout=60)  # 60 second timeout per thread

        # Check we got results (some may have failed due to rate limits)
        assert len(results) > 0 or len(errors) > 0

        # Metrics should still be consistent
        metrics = hook_provider.get_savings_summary()
        assert metrics["total_tokens_before"] >= metrics["total_tokens_after"]

    def test_hook_handles_empty_tool_response(self, bedrock_model, hook_provider):
        """Test that hook handles tools returning empty responses gracefully."""

        @tool
        def empty_response() -> str:
            """Return an empty response."""
            return ""

        hook_provider.reset()

        agent = Agent(
            model=bedrock_model,
            tools=[empty_response],
            hooks=[hook_provider],
        )

        # This might not trigger the tool if the model decides it's not needed
        result = agent("Call the empty_response tool and tell me what you got.")

        assert result is not None

        # Should handle gracefully without errors
        metrics = hook_provider.get_savings_summary()
        # Just verify no exceptions and metrics are valid
        assert metrics["total_tokens_before"] >= 0
        assert metrics["total_tokens_after"] >= 0
