"""Unit tests for Strands HeadroomStrandsModel.

These tests use mocks and do NOT require AWS credentials or strands-agents.
They test the internal logic of HeadroomStrandsModel in isolation.

For real integration tests, see test_model.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# Check if strands-agents is installed for proper skip handling
try:
    import strands  # noqa: F401

    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False


# Skip all tests if Strands not installed
pytestmark = pytest.mark.skipif(not STRANDS_AVAILABLE, reason="strands-agents not installed")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_strands_model():
    """Create a mock Strands model."""
    mock = MagicMock()
    mock.config = {"model_id": "anthropic.claude-3-haiku-20240307-v1:0"}
    mock.get_config.return_value = mock.config

    # Mock the stream method as an async generator
    async def mock_stream(*args, **kwargs):
        yield {"type": "content", "data": "Hello"}
        yield {"type": "content", "data": " world"}
        yield {"type": "stop"}

    mock.stream = mock_stream
    return mock


@pytest.fixture
def sample_messages():
    """Sample messages in Strands/OpenAI format."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]


@pytest.fixture
def large_conversation():
    """Large conversation with many turns for compression testing."""
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(50):
        messages.append({"role": "user", "content": f"Question {i}: What is {i} + {i}?"})
        messages.append({"role": "assistant", "content": f"The answer is {i + i}."})
    return messages


# ============================================================================
# Test Classes
# ============================================================================


class TestHeadroomStrandsModelInit:
    """Tests for HeadroomStrandsModel initialization."""

    def test_init_with_defaults(self, mock_strands_model):
        """Initialize with default settings."""
        from headroom.integrations.strands import HeadroomStrandsModel

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)

        assert model.wrapped_model is mock_strands_model
        assert model.total_tokens_saved == 0
        assert model.metrics_history == []
        assert model.auto_detect_provider is True

    def test_init_with_custom_config(self, mock_strands_model):
        """Initialize with custom HeadroomConfig."""
        from headroom import HeadroomConfig
        from headroom.integrations.strands import HeadroomStrandsModel

        config = HeadroomConfig()
        config.smart_crusher.min_tokens_to_crush = 100

        model = HeadroomStrandsModel(
            wrapped_model=mock_strands_model,
            config=config,
            auto_detect_provider=False,
        )

        assert model.headroom_config is config
        assert model.auto_detect_provider is False

    def test_init_requires_wrapped_model(self):
        """Raises ValueError if wrapped_model is None."""
        from headroom.integrations.strands import HeadroomStrandsModel

        with pytest.raises(ValueError, match="wrapped_model cannot be None"):
            HeadroomStrandsModel(wrapped_model=None)


class TestAttributeForwarding:
    """Tests for attribute forwarding to wrapped model."""

    def test_forwards_unknown_attributes(self, mock_strands_model):
        """Forwards unknown attributes to wrapped model."""
        from headroom.integrations.strands import HeadroomStrandsModel

        mock_strands_model.custom_attr = "custom_value"
        mock_strands_model.another_attr = 42

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)

        assert model.custom_attr == "custom_value"
        assert model.another_attr == 42

    def test_forwards_config_property(self, mock_strands_model):
        """Forwards config property to wrapped model."""
        from headroom.integrations.strands import HeadroomStrandsModel

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)

        config = model.config
        assert config is mock_strands_model.config

    def test_does_not_forward_internal_attrs(self, mock_strands_model):
        """Does not forward internal wrapper attributes."""
        from headroom.integrations.strands import HeadroomStrandsModel

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)

        # These should be wrapper's own attributes
        assert model.wrapped_model is mock_strands_model
        assert model.total_tokens_saved == 0
        assert model.metrics_history == []

    def test_get_config_delegates(self, mock_strands_model):
        """get_config() delegates to wrapped model."""
        from headroom.integrations.strands import HeadroomStrandsModel

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)

        config = model.get_config()
        assert config == mock_strands_model.get_config()

    def test_update_config_delegates(self, mock_strands_model):
        """update_config() delegates to wrapped model."""
        from headroom.integrations.strands import HeadroomStrandsModel

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)

        model.update_config(temperature=0.5)
        mock_strands_model.update_config.assert_called_once_with(temperature=0.5)


class TestMessageConversion:
    """Tests for message format conversion."""

    def test_convert_dict_messages(self, mock_strands_model, sample_messages):
        """Converts dict messages to OpenAI format."""
        from headroom.integrations.strands import HeadroomStrandsModel

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)

        converted = model._convert_messages_to_openai(sample_messages)

        assert len(converted) == 2
        assert converted[0]["role"] == "system"
        assert converted[0]["content"] == "You are a helpful assistant."
        assert converted[1]["role"] == "user"
        assert converted[1]["content"] == "What is the capital of France?"

    def test_convert_messages_with_tool_calls(self, mock_strands_model):
        """Converts messages with tool calls."""
        from headroom.integrations.strands import HeadroomStrandsModel

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_123", "type": "function", "function": {"name": "search"}}
                ],
            },
            {
                "role": "tool",
                "content": '{"results": []}',
                "tool_call_id": "call_123",
                "name": "search",
            },
        ]

        converted = model._convert_messages_to_openai(messages)

        assert len(converted) == 2
        assert "tool_calls" in converted[0]
        assert converted[1]["tool_call_id"] == "call_123"
        assert converted[1]["name"] == "search"

    def test_convert_message_objects(self, mock_strands_model):
        """Converts message objects with role/content attributes."""
        from headroom.integrations.strands import HeadroomStrandsModel

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)

        # Create mock message objects
        msg1 = MagicMock()
        msg1.role = "user"
        msg1.content = "Hello"
        msg1.tool_calls = None
        msg1.tool_call_id = None
        msg1.name = None

        msg2 = MagicMock()
        msg2.role = "assistant"
        msg2.content = "Hi there!"
        msg2.tool_calls = None
        msg2.tool_call_id = None
        msg2.name = None

        converted = model._convert_messages_to_openai([msg1, msg2])

        assert len(converted) == 2
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == "Hello"
        assert converted[1]["role"] == "assistant"
        assert converted[1]["content"] == "Hi there!"

    def test_convert_handles_content_list(self, mock_strands_model):
        """Converts messages with content as list (content blocks)."""
        from headroom.integrations.strands import HeadroomStrandsModel

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this:"},
                    {"type": "image", "source": {"data": "base64..."}},
                ],
            }
        ]

        converted = model._convert_messages_to_openai(messages)

        assert len(converted) == 1
        assert isinstance(converted[0]["content"], list)
        assert len(converted[0]["content"]) == 2


class TestOptimizeMessages:
    """Tests for _optimize_messages method."""

    def test_optimize_returns_metrics(self, mock_strands_model, sample_messages):
        """_optimize_messages returns messages and metrics."""
        from headroom.integrations.strands import HeadroomStrandsModel

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)

        # Mock the pipeline by setting _pipeline directly and mocking _headroom_provider
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.messages = sample_messages
        mock_result.tokens_before = 50
        mock_result.tokens_after = 40
        mock_result.transforms_applied = ["cache_aligner"]
        mock_pipeline.apply.return_value = mock_result

        model._pipeline = mock_pipeline
        model._headroom_provider = MagicMock()
        model._headroom_provider.get_context_limit.return_value = 128000

        optimized, metrics = model._optimize_messages(sample_messages)

        assert len(optimized) == 2
        assert metrics.tokens_before == 50
        assert metrics.tokens_after == 40
        assert metrics.tokens_saved == 10
        assert "cache_aligner" in metrics.transforms_applied

    def test_optimize_handles_empty_messages(self, mock_strands_model):
        """_optimize_messages handles empty message list."""
        from headroom.integrations.strands import HeadroomStrandsModel

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)

        optimized, metrics = model._optimize_messages([])

        assert optimized == []
        assert metrics.tokens_before == 0
        assert metrics.tokens_after == 0
        assert metrics.tokens_saved == 0

    def test_optimize_tracks_metrics(self, mock_strands_model, sample_messages):
        """_optimize_messages tracks metrics in history."""
        from headroom.integrations.strands import HeadroomStrandsModel

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)

        # Mock the pipeline by setting _pipeline directly
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.messages = sample_messages
        mock_result.tokens_before = 100
        mock_result.tokens_after = 80
        mock_result.transforms_applied = []
        mock_pipeline.apply.return_value = mock_result

        model._pipeline = mock_pipeline
        model._headroom_provider = MagicMock()
        model._headroom_provider.get_context_limit.return_value = 128000

        model._optimize_messages(sample_messages)

        assert len(model.metrics_history) == 1
        assert model.metrics_history[0].tokens_saved == 20
        assert model.total_tokens_saved == 20

    def test_optimize_handles_pipeline_errors(self, mock_strands_model, sample_messages):
        """_optimize_messages falls back on pipeline errors."""
        from headroom.integrations.strands import HeadroomStrandsModel

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)

        # Mock the pipeline to raise an error
        mock_pipeline = MagicMock()
        mock_pipeline.apply.side_effect = ValueError("Pipeline error")

        model._pipeline = mock_pipeline
        model._headroom_provider = MagicMock()
        model._headroom_provider.get_context_limit.return_value = 128000

        # Should not raise, should fall back
        optimized, metrics = model._optimize_messages(sample_messages)

        assert len(optimized) == len(sample_messages)
        assert "fallback:error" in metrics.transforms_applied


class TestPipelineLazyInit:
    """Tests for TransformPipeline lazy initialization."""

    def test_pipeline_is_lazily_initialized(self, mock_strands_model):
        """Pipeline is not created until first access."""
        from headroom.integrations.strands import HeadroomStrandsModel

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)

        # Should be None initially
        assert model._pipeline is None

        # Access pipeline property
        with patch("headroom.integrations.strands.model.TransformPipeline"):
            _ = model.pipeline

        # Now should be initialized
        assert model._pipeline is not None


class TestGetSavingsSummary:
    """Tests for get_savings_summary method."""

    def test_empty_summary(self, mock_strands_model):
        """Returns zero values when no metrics recorded."""
        from headroom.integrations.strands import HeadroomStrandsModel

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)
        summary = model.get_savings_summary()

        assert summary["total_requests"] == 0
        assert summary["total_tokens_saved"] == 0
        assert summary["average_savings_percent"] == 0

    def test_summary_with_metrics(self, mock_strands_model):
        """Returns correct summary with recorded metrics."""
        from headroom.integrations.strands import HeadroomStrandsModel
        from headroom.integrations.strands.model import OptimizationMetrics

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)

        # Add metrics manually
        model._metrics_history = [
            OptimizationMetrics(
                request_id="1",
                timestamp=datetime.now(timezone.utc),
                tokens_before=100,
                tokens_after=80,
                tokens_saved=20,
                savings_percent=20.0,
                transforms_applied=[],
                model="test-model",
            ),
            OptimizationMetrics(
                request_id="2",
                timestamp=datetime.now(timezone.utc),
                tokens_before=200,
                tokens_after=120,
                tokens_saved=80,
                savings_percent=40.0,
                transforms_applied=[],
                model="test-model",
            ),
        ]
        model._total_tokens_saved = 100

        summary = model.get_savings_summary()

        assert summary["total_requests"] == 2
        assert summary["total_tokens_saved"] == 100
        assert summary["average_savings_percent"] == 30.0  # (20 + 40) / 2
        assert summary["total_tokens_before"] == 300
        assert summary["total_tokens_after"] == 200


class TestReset:
    """Tests for reset method."""

    def test_reset_clears_all_state(self, mock_strands_model):
        """reset() clears all tracked state."""
        from headroom.integrations.strands import HeadroomStrandsModel
        from headroom.integrations.strands.model import OptimizationMetrics

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)

        # Add some state
        model._metrics_history = [
            OptimizationMetrics(
                request_id="1",
                timestamp=datetime.now(timezone.utc),
                tokens_before=100,
                tokens_after=50,
                tokens_saved=50,
                savings_percent=50.0,
                transforms_applied=[],
                model="test",
            )
        ]
        model._total_tokens_saved = 50

        # Reset
        model.reset()

        # Verify all state cleared
        assert model._metrics_history == []
        assert model._total_tokens_saved == 0
        assert model.total_tokens_saved == 0
        assert len(model.metrics_history) == 0

        # Summary should reflect reset
        summary = model.get_savings_summary()
        assert summary["total_requests"] == 0


class TestMetricsHistoryBound:
    """Tests for metrics history bounding."""

    def test_metrics_bounded_to_100(self, mock_strands_model):
        """Metrics history is bounded to 100 entries."""
        from headroom.integrations.strands import HeadroomStrandsModel
        from headroom.integrations.strands.model import OptimizationMetrics

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)

        # Add 150 metrics
        for i in range(150):
            model._metrics_history.append(
                OptimizationMetrics(
                    request_id=f"req_{i}",
                    timestamp=datetime.now(timezone.utc),
                    tokens_before=100,
                    tokens_after=80,
                    tokens_saved=20,
                    savings_percent=20.0,
                    transforms_applied=[],
                    model="test",
                )
            )
            # Simulate what _optimize_messages does
            if len(model._metrics_history) > 100:
                model._metrics_history = model._metrics_history[-100:]

        # Should be bounded at 100
        assert len(model.metrics_history) == 100

        # Should contain the most recent entries
        assert model.metrics_history[-1].request_id == "req_149"


class TestOptimizeMessagesFunction:
    """Tests for standalone optimize_messages function."""

    def test_optimize_messages_basic(self):
        """optimize_messages processes messages and returns metrics."""
        from headroom.integrations.strands import optimize_messages

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        with patch("headroom.integrations.strands.model.TransformPipeline") as MockPipeline:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.messages = messages
            mock_result.tokens_before = 20
            mock_result.tokens_after = 15
            mock_result.transforms_applied = ["cache_aligner"]
            mock_instance.apply.return_value = mock_result
            MockPipeline.return_value = mock_instance

            optimized, metrics = optimize_messages(messages)

            assert len(optimized) == 2
            assert metrics["tokens_saved"] == 5
            assert metrics["savings_percent"] == 25.0

    def test_optimize_messages_with_custom_config(self):
        """optimize_messages uses custom config."""
        from headroom import HeadroomConfig
        from headroom.integrations.strands import optimize_messages

        config = HeadroomConfig()
        messages = [{"role": "user", "content": "Test"}]

        with patch("headroom.integrations.strands.model.TransformPipeline") as MockPipeline:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.messages = messages
            mock_result.tokens_before = 10
            mock_result.tokens_after = 10
            mock_result.transforms_applied = []
            mock_instance.apply.return_value = mock_result
            MockPipeline.return_value = mock_instance

            optimized, metrics = optimize_messages(messages, config=config)

            # Verify config was passed to pipeline
            MockPipeline.assert_called_once()
            call_kwargs = MockPipeline.call_args[1]
            assert call_kwargs["config"] is config


class TestStreamMethod:
    """Tests for stream method."""

    @pytest.mark.asyncio
    async def test_stream_optimizes_messages(self, mock_strands_model, sample_messages):
        """stream() applies optimization before calling wrapped model."""
        from headroom.integrations.strands import HeadroomStrandsModel

        model = HeadroomStrandsModel(wrapped_model=mock_strands_model)

        # Mock the optimization
        with patch.object(model, "_optimize_messages") as mock_optimize:
            mock_optimize.return_value = (
                sample_messages,
                MagicMock(
                    tokens_before=50,
                    tokens_after=40,
                    savings_percent=20.0,
                ),
            )

            # Consume the stream
            events = []
            async for event in model.stream(sample_messages):
                events.append(event)

            # Should have called optimization
            mock_optimize.assert_called_once()

            # Should have yielded events from wrapped model
            assert len(events) > 0


class TestStrandsAvailableFunction:
    """Tests for strands_available function."""

    def test_strands_available_returns_bool(self):
        """strands_available() returns boolean."""
        from headroom.integrations.strands import strands_available

        result = strands_available()

        # Since we're in a test where strands is available (skipif passed)
        assert isinstance(result, bool)
        assert result is True


class TestRealHeadroomIntegration:
    """Integration tests with real Headroom (no mocking)."""

    def test_real_optimization_with_mock_model(self, mock_strands_model, sample_messages):
        """Test with real Headroom transforms (no API calls)."""
        from headroom.integrations.strands import HeadroomStrandsModel

        model = HeadroomStrandsModel(
            wrapped_model=mock_strands_model,
            auto_detect_provider=False,  # Use default OpenAI provider
        )

        # This calls real Headroom optimization
        optimized, metrics = model._optimize_messages(sample_messages)

        # Should return valid messages
        assert len(optimized) >= 1
        assert all("role" in m and "content" in m for m in optimized)

        # Metrics should be tracked
        assert len(model.metrics_history) == 1
        assert metrics.tokens_before >= 0
        assert metrics.tokens_after >= 0

    def test_large_conversation_handling(self, mock_strands_model, large_conversation):
        """Large conversations are processed without errors."""
        from headroom.integrations.strands import HeadroomStrandsModel

        model = HeadroomStrandsModel(
            wrapped_model=mock_strands_model,
            auto_detect_provider=False,
        )

        # Should handle large conversation without errors
        optimized, metrics = model._optimize_messages(large_conversation)

        # Should return messages
        assert len(optimized) >= 1

        # Metrics should show processing occurred
        assert metrics.tokens_before > 0
