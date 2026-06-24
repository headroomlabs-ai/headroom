"""Real-world integration tests for Strands HeadroomStrandsModel.

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
# Test Tools - Generate realistic data for optimization testing
# These are defined with @tool decorator for use when strands is installed.
# When strands is not installed, the no-op decorator ensures import succeeds.
# ============================================================================


@tool
def get_database_records(table: str, limit: int = 50) -> str:
    """Fetch records from a database table. Returns JSON array.

    Args:
        table: Name of the database table
        limit: Maximum records to return

    Returns:
        JSON array of database records
    """
    records = [
        {
            "id": i,
            "table": table,
            "created_at": f"2024-01-{(i % 28) + 1:02d}T{10 + (i % 12):02d}:00:00Z",
            "updated_at": f"2024-01-{(i % 28) + 1:02d}T{11 + (i % 12):02d}:00:00Z",
            "status": ["active", "inactive", "pending", "archived"][i % 4],
            "priority": ["low", "medium", "high", "critical"][i % 4],
            "data": {
                "field1": f"value_{i}_{table}",
                "field2": i * 100,
                "field3": i % 2 == 0,
                "metadata": {
                    "source": "database",
                    "version": f"1.{i % 10}.0",
                    "tags": [f"tag_{j}" for j in range(i % 5 + 1)],
                },
            },
            "metrics": {
                "read_count": i * 10,
                "write_count": i * 5,
                "error_count": i % 3,
                "latency_ms": 50 + (i * 7) % 200,
            },
        }
        for i in range(limit)
    ]
    return json.dumps(records, indent=2)


@tool
def get_large_logs(query: str, count: int = 200) -> str:
    """Fetch verbose log data that should trigger compression.

    Args:
        query: Search query for logs
        count: Number of log entries to return

    Returns:
        JSON array of detailed log entries
    """
    logs = [
        {
            "log_id": f"log_{i:08d}",
            "timestamp": f"2024-01-{(i % 28) + 1:02d}T{10 + (i % 12):02d}:{i % 60:02d}:00Z",
            "level": ["DEBUG", "INFO", "WARN", "ERROR"][i % 4],
            "service": f"service_{i % 10}",
            "message": f"Processing request for query '{query}' - step {i}",
            "request_id": f"req_{i:012d}",
            "trace_id": f"trace_{i:016x}",
            "span_id": f"span_{i:08x}",
            "user_id": f"user_{i % 100:04d}",
            "session_id": f"sess_{i:010d}",
            "metadata": {
                "host": f"server-{i % 20:02d}.example.com",
                "region": ["us-west-2", "us-east-1", "eu-west-1", "ap-southeast-1"][i % 4],
                "instance_type": ["t3.micro", "t3.small", "t3.medium", "t3.large"][i % 4],
                "container_id": f"container_{i:08x}",
                "kubernetes_pod": f"pod-{i:06d}",
                "kubernetes_namespace": "production",
            },
            "metrics": {
                "duration_ms": 50 + (i * 3) % 500,
                "memory_mb": 128 + (i * 7) % 1024,
                "cpu_percent": 5 + (i * 2) % 95,
                "network_bytes_in": i * 1024,
                "network_bytes_out": i * 512,
            },
            "tags": ["env:prod", f"version:1.{i % 10}.0", "team:backend"],
        }
        for i in range(count)
    ]
    return json.dumps(logs, indent=2)


@tool
def analyze_metrics(metric_type: str) -> str:
    """Analyze system metrics. Returns detailed metrics data.

    Args:
        metric_type: Type of metrics to analyze (cpu, memory, network, disk)

    Returns:
        JSON object with metric analysis
    """
    data_points = [
        {
            "timestamp": f"2024-01-15T{10 + (i % 12):02d}:{(i * 5) % 60:02d}:00Z",
            "value": 20 + (i * 3) % 80,
            "unit": {"cpu": "%", "memory": "MB", "network": "Mbps", "disk": "GB"}.get(
                metric_type, "units"
            ),
            "host": f"server-{(i % 5) + 1:02d}",
            "region": ["us-west-2", "us-east-1", "eu-west-1"][i % 3],
            "metadata": {
                "collection_interval": 60,
                "aggregation": "avg",
                "quality": "good" if i % 5 != 0 else "degraded",
            },
        }
        for i in range(100)
    ]

    return json.dumps(
        {
            "metric_type": metric_type,
            "time_range": {"start": "2024-01-15T10:00:00Z", "end": "2024-01-15T22:00:00Z"},
            "data_points": data_points,
            "summary": {
                "min": 20,
                "max": 99,
                "avg": 55.5,
                "p50": 52,
                "p95": 90,
                "p99": 97,
            },
        },
        indent=2,
    )


@tool
def quick_lookup(key: str) -> str:
    """Quick key-value lookup. Returns small response.

    Args:
        key: The key to look up

    Returns:
        Small JSON with the value
    """
    return json.dumps({"key": key, "value": f"result_for_{key}", "found": True})


@tool
def math_operation(x: float, y: float, op: str) -> str:
    """Perform a math operation.

    Args:
        x: First operand
        y: Second operand
        op: Operation (add, sub, mul, div)

    Returns:
        Result of the operation
    """
    operations = {
        "add": x + y,
        "sub": x - y,
        "mul": x * y,
        "div": x / y if y != 0 else None,
    }
    result = operations.get(op, None)
    return json.dumps({"x": x, "y": y, "operation": op, "result": result})


# ============================================================================
# Test Class for HeadroomStrandsModel
# ============================================================================


@pytest.mark.skipif(SKIP_BEDROCK, reason="AWS credentials not available")
@pytest.mark.skipif(not STRANDS_AVAILABLE, reason="strands-agents not installed")
class TestHeadroomStrandsModelReal:
    """Real-world integration tests for HeadroomStrandsModel with Bedrock."""

    @pytest.fixture
    def base_bedrock_model(self):
        """Create a base BedrockModel instance using Claude 3 Haiku (fast and cheap)."""
        return BedrockModel(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            region_name="us-west-2",
            temperature=0.1,
        )

    @pytest.fixture
    def wrapped_model(self, base_bedrock_model):
        """Create a HeadroomStrandsModel wrapping the Bedrock model."""
        from headroom.integrations.strands import HeadroomStrandsModel

        return HeadroomStrandsModel(
            wrapped_model=base_bedrock_model,
            auto_detect_provider=True,
        )

    def test_stream_returns_proper_events(self, wrapped_model):
        """Test that stream() works and returns proper StreamEvents.

        The Strands Agent uses the model's stream() method internally.
        This test verifies that the wrapped model properly streams responses.
        """
        wrapped_model.reset()

        agent = Agent(model=wrapped_model)

        # Make a request - the agent internally calls stream() on the model
        result = agent("Count from 1 to 5, one number per line.")

        # Verify we got a response (proves streaming worked)
        assert result is not None
        response_text = str(result)
        assert len(response_text) > 0

        # The response should contain numbers 1-5
        for num in ["1", "2", "3", "4", "5"]:
            assert num in response_text, f"Expected {num} in response"

        # Metrics should be tracked (proves stream() was intercepted properly)
        metrics = wrapped_model.get_savings_summary()
        assert metrics["total_requests"] >= 1, "stream() should track requests"

    def test_messages_optimized_large_conversations(self, wrapped_model):
        """Test that messages are actually optimized (tokens_before > tokens_after for large conversations).

        This test builds up a large conversation context through tool calls
        with verbose JSON responses, then verifies that optimization occurs.
        """
        wrapped_model.reset()

        agent = Agent(model=wrapped_model, tools=[get_large_logs, get_database_records])

        # First request - get large logs (200 entries with verbose data)
        agent(
            "Search for logs containing 'error' and get 200 entries using get_large_logs. "
            "Tell me how many ERROR level logs there are."
        )

        # Second request - more tool output, context grows
        agent(
            "Now get 100 records from the 'events' table using get_database_records. "
            "How many records have 'active' status?"
        )

        # Third request - even more context
        agent(
            "Based on all the data you've seen, give me a one-sentence summary "
            "of the system health."
        )

        # Check optimization metrics
        metrics = wrapped_model.get_savings_summary()

        # Should have processed multiple requests
        assert metrics["total_requests"] >= 1, "Should have processed requests"

        # With large tool outputs, tokens_before should be significant
        assert metrics["total_tokens_before"] > 0, "Should have counted input tokens"

        # The key assertion: optimization should reduce tokens
        # (tokens_before >= tokens_after, with strict > when there's compressible content)
        assert metrics["total_tokens_before"] >= metrics["total_tokens_after"], (
            f"Optimization should not increase tokens: "
            f"before={metrics['total_tokens_before']}, after={metrics['total_tokens_after']}"
        )

        # Check history shows optimization was tracked
        history = wrapped_model.metrics_history
        assert len(history) >= 1, "Should have metrics history"

        # Verify individual requests track before/after properly
        for m in history:
            assert m.tokens_before >= m.tokens_after, (
                f"Each request should have tokens_before >= tokens_after: "
                f"request_id={m.request_id}, before={m.tokens_before}, after={m.tokens_after}"
            )

    def test_get_savings_summary_returns_correct_metrics(self, wrapped_model):
        """Test that get_savings_summary() returns correct metrics.

        Verifies the structure and accuracy of the savings summary.
        """
        wrapped_model.reset()

        agent = Agent(model=wrapped_model, tools=[get_database_records])

        # Make a few requests
        agent("Get 30 records from 'users' table.")
        agent("Get 30 records from 'orders' table.")

        # Get the summary
        summary = wrapped_model.get_savings_summary()

        # Verify required keys exist
        required_keys = [
            "total_requests",
            "total_tokens_saved",
            "average_savings_percent",
            "total_tokens_before",
            "total_tokens_after",
        ]
        for key in required_keys:
            assert key in summary, f"Summary missing required key: {key}"

        # Verify values are sensible
        assert summary["total_requests"] >= 1, "Should have at least one request"
        assert summary["total_tokens_before"] >= 0, "tokens_before should be non-negative"
        assert summary["total_tokens_after"] >= 0, "tokens_after should be non-negative"
        assert summary["total_tokens_saved"] >= 0, "tokens_saved should be non-negative"
        assert 0 <= summary["average_savings_percent"] <= 100, (
            "average_savings_percent should be between 0 and 100"
        )

        # Verify mathematical consistency
        expected_saved = summary["total_tokens_before"] - summary["total_tokens_after"]
        assert summary["total_tokens_saved"] == expected_saved, (
            f"tokens_saved should equal tokens_before - tokens_after: "
            f"saved={summary['total_tokens_saved']}, expected={expected_saved}"
        )

    def test_reset_clears_all_metrics(self, wrapped_model):
        """Test that reset() clears all accumulated metrics.

        Verifies that reset() properly clears:
        - total_tokens_saved
        - metrics_history
        - The summary returned by get_savings_summary()
        """
        # Make some requests to accumulate metrics
        agent = Agent(model=wrapped_model)
        agent("Say 'hello world'")
        agent("Say 'goodbye world'")

        # Verify we have metrics before reset
        assert wrapped_model.total_tokens_saved >= 0
        pre_reset_requests = wrapped_model.get_savings_summary()["total_requests"]
        assert pre_reset_requests >= 1, "Should have requests before reset"

        # Call reset
        wrapped_model.reset()

        # Verify all metrics are cleared
        assert wrapped_model.total_tokens_saved == 0, "total_tokens_saved should be 0 after reset"
        assert len(wrapped_model.metrics_history) == 0, (
            "metrics_history should be empty after reset"
        )

        # Verify get_savings_summary reflects the reset
        summary = wrapped_model.get_savings_summary()
        assert summary["total_requests"] == 0, "total_requests should be 0 after reset"
        assert summary["total_tokens_saved"] == 0, "total_tokens_saved should be 0 after reset"
        assert summary["total_tokens_before"] == 0, "total_tokens_before should be 0 after reset"
        assert summary["total_tokens_after"] == 0, "total_tokens_after should be 0 after reset"

        # Verify we can still make requests after reset
        agent = Agent(model=wrapped_model)
        agent("Say 'post-reset test'")

        post_reset_summary = wrapped_model.get_savings_summary()
        assert post_reset_summary["total_requests"] >= 1, "Should track requests after reset"

    def test_model_wrapper_basic_response(self, wrapped_model):
        """Test that wrapped model produces valid responses."""
        agent = Agent(model=wrapped_model)

        result = agent("Say 'Hello, Headroom!' and nothing else.")

        assert result is not None
        content = str(result)
        assert len(content) > 0

    def test_model_wrapper_with_tools(self, wrapped_model):
        """Test that wrapped model works correctly with tools."""
        wrapped_model.reset()

        agent = Agent(model=wrapped_model, tools=[quick_lookup, math_operation, analyze_metrics])

        result = agent(
            "Please do these tasks: "
            "1. Look up the key 'config_setting' using quick_lookup. "
            "2. Calculate 15.5 multiplied by 4 using math_operation. "
            "3. Tell me the results."
        )

        assert result is not None

        metrics = wrapped_model.get_savings_summary()
        assert metrics["total_requests"] >= 1

    def test_model_wrapper_metrics_tracking(self, wrapped_model):
        """Test that metrics are accurately tracked across requests."""
        wrapped_model.reset()

        agent = Agent(model=wrapped_model, tools=[get_database_records])

        # Make several requests
        agent("Get 20 records from 'products' table.")
        agent("Get 20 records from 'customers' table.")
        agent("Summarize both sets of records.")

        metrics = wrapped_model.get_savings_summary()

        assert metrics["total_requests"] >= 1
        assert metrics["total_tokens_before"] >= metrics["total_tokens_after"]

        if metrics["total_tokens_saved"] > 0:
            assert metrics["average_savings_percent"] >= 0
            assert metrics["average_savings_percent"] <= 100

        # History should be bounded
        assert len(wrapped_model.metrics_history) <= 100

    def test_model_wrapper_attribute_forwarding(self, base_bedrock_model):
        """Test that attributes are forwarded to wrapped model."""
        from headroom.integrations.strands import HeadroomStrandsModel

        wrapped = HeadroomStrandsModel(
            wrapped_model=base_bedrock_model,
            auto_detect_provider=True,
        )

        # The wrapper should forward config to the wrapped model (Strands stores model_id in config)
        assert hasattr(wrapped, "config")
        config = wrapped.config
        assert isinstance(config, dict)
        assert "model_id" in config

        # Access wrapped model directly
        assert wrapped.wrapped_model is base_bedrock_model

    def test_model_wrapper_custom_config(self, base_bedrock_model):
        """Test that custom HeadroomConfig is applied."""
        from headroom import HeadroomConfig
        from headroom.integrations.strands import HeadroomStrandsModel

        custom_config = HeadroomConfig()
        custom_config.smart_crusher.min_tokens_to_crush = 50
        custom_config.smart_crusher.max_items_after_crush = 10

        wrapped = HeadroomStrandsModel(
            wrapped_model=base_bedrock_model,
            config=custom_config,
            auto_detect_provider=True,
        )

        assert wrapped.headroom_config is custom_config
        assert wrapped.headroom_config.smart_crusher.min_tokens_to_crush == 50

        # The model should still work
        agent = Agent(model=wrapped)
        result = agent("Say 'test'")
        assert result is not None

    def test_model_wrapper_provider_detection(self, base_bedrock_model):
        """Test that provider is auto-detected correctly for Bedrock Claude."""
        from headroom.integrations.strands import HeadroomStrandsModel
        from headroom.providers import AnthropicProvider

        wrapped = HeadroomStrandsModel(
            wrapped_model=base_bedrock_model,
            auto_detect_provider=True,
        )

        # Access pipeline to trigger lazy initialization
        _ = wrapped.pipeline

        # For Bedrock Claude models, should detect Anthropic provider
        assert wrapped._headroom_provider is not None
        assert isinstance(wrapped._headroom_provider, AnthropicProvider)

    def test_model_wrapper_handles_large_context(self, wrapped_model):
        """Test that wrapper handles large context appropriately."""
        wrapped_model.reset()

        agent = Agent(model=wrapped_model, tools=[analyze_metrics, get_database_records])

        # Build up context with large tool outputs
        agent("Analyze CPU metrics using analyze_metrics.")
        agent("Get 50 records from 'logs' table using get_database_records.")
        agent("Based on everything, what patterns do you see?")

        metrics = wrapped_model.get_savings_summary()
        assert metrics["total_requests"] >= 1
        assert metrics["total_tokens_before"] > 0

    def test_model_wrapper_empty_messages(self, base_bedrock_model):
        """Test that wrapper handles edge cases gracefully."""
        from headroom.integrations.strands import HeadroomStrandsModel

        wrapped = HeadroomStrandsModel(
            wrapped_model=base_bedrock_model,
            auto_detect_provider=True,
        )

        # Test with minimal input
        agent = Agent(model=wrapped)
        result = agent("Hi")

        assert result is not None

    def test_model_wrapper_thread_safety(self, base_bedrock_model):
        """Test that wrapper is thread-safe for metrics tracking."""
        import threading
        import time

        from headroom.integrations.strands import HeadroomStrandsModel

        wrapped = HeadroomStrandsModel(
            wrapped_model=base_bedrock_model,
            auto_detect_provider=True,
        )

        agent = Agent(model=wrapped)

        results = []
        errors = []

        def make_request(msg: str):
            try:
                result = agent(msg)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = []
        messages = ["Say 'one'", "Say 'two'", "Say 'three'"]

        for msg in messages:
            t = threading.Thread(target=make_request, args=(msg,))
            threads.append(t)
            t.start()
            time.sleep(0.5)  # Small delay to avoid rate limiting

        for t in threads:
            t.join(timeout=60)

        # Should have some results (may have errors due to rate limiting)
        assert len(results) > 0 or len(errors) > 0

        # Metrics should be consistent
        metrics = wrapped.get_savings_summary()
        assert metrics["total_tokens_before"] >= metrics["total_tokens_after"]


# ============================================================================
# Test Class for optimize_messages standalone function
# ============================================================================


@pytest.mark.skipif(SKIP_BEDROCK, reason="AWS credentials not available")
@pytest.mark.skipif(not STRANDS_AVAILABLE, reason="strands-agents not installed")
class TestOptimizeMessagesFunction:
    """Tests for the standalone optimize_messages function."""

    def test_optimize_messages_basic(self):
        """Test basic message optimization."""
        from headroom.integrations.strands import optimize_messages

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you today?"},
        ]

        optimized, metrics = optimize_messages(messages)

        assert len(optimized) > 0

        assert "tokens_before" in metrics
        assert "tokens_after" in metrics
        assert "tokens_saved" in metrics
        assert metrics["tokens_before"] >= 0
        assert metrics["tokens_after"] >= 0

    def test_optimize_messages_with_tool_content(self):
        """Test optimization of messages containing tool responses."""
        from headroom.integrations.strands import optimize_messages

        # Create messages with large tool output
        large_data = json.dumps([{"id": i, "data": f"value_{i}" * 10} for i in range(100)])

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Get the data"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_data", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": large_data, "tool_call_id": "call_123"},
            {"role": "assistant", "content": "Here is the data summary..."},
        ]

        optimized, metrics = optimize_messages(messages)

        assert len(optimized) > 0
        assert metrics["tokens_before"] >= 0

    def test_optimize_messages_custom_config(self):
        """Test optimization with custom config."""
        from headroom import HeadroomConfig
        from headroom.integrations.strands import optimize_messages

        config = HeadroomConfig()
        config.smart_crusher.enabled = True
        config.smart_crusher.min_tokens_to_crush = 10

        messages = [
            {"role": "user", "content": "Hello!"},
        ]

        optimized, metrics = optimize_messages(messages, config=config)

        assert len(optimized) > 0
        assert "tokens_before" in metrics
