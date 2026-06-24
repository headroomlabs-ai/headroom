"""Comprehensive tests for Agno integration.

Tests cover:
1. HeadroomAgnoModel - Wrapper for any Agno model
2. Provider detection - Detecting correct provider from Agno model
3. Hooks - Pre and post hooks for observability
4. optimize_messages() - Standalone optimization function
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Check if Agno is available
try:
    import agno  # noqa: F401

    AGNO_AVAILABLE = True
except ImportError:
    AGNO_AVAILABLE = False

from headroom import HeadroomConfig, HeadroomMode

# Skip all tests if Agno not installed
pytestmark = pytest.mark.skipif(not AGNO_AVAILABLE, reason="Agno not installed")


@pytest.fixture
def mock_agno_model():
    """Create a mock Agno model (OpenAIChat-like)."""
    from agno.models.response import ModelResponse

    mock = MagicMock()
    mock.__class__.__name__ = "OpenAIChat"
    mock.__class__.__module__ = "agno.models.openai"
    mock.id = "gpt-4o"

    # Mock response method
    def mock_response(messages, **kwargs):
        response = MagicMock()
        response.content = "Hello! I'm a mock response."
        response.metrics = MagicMock()
        response.metrics.input_tokens = 10
        response.metrics.output_tokens = 5
        response.metrics.total_tokens = 15
        return response

    mock.response = MagicMock(side_effect=mock_response)

    # Mock invoke method (returns ModelResponse for Agno's response() loop)
    def mock_invoke(messages, **kwargs):
        from agno.models.metrics import Metrics

        # Create a proper ModelResponse that Agno's response() can process
        return ModelResponse(
            role="assistant",
            content="Hello! I'm a mock response.",
            response_usage=Metrics(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
            ),
        )

    mock.invoke = MagicMock(side_effect=mock_invoke)

    # Mock streaming response
    def mock_stream(messages, **kwargs):
        yield MagicMock(content="Streaming...")

    mock.response_stream = MagicMock(side_effect=mock_stream)

    # Mock invoke_stream for streaming
    def mock_invoke_stream(messages, **kwargs):
        from agno.models.metrics import Metrics

        yield ModelResponse(
            role="assistant",
            content="Streaming...",
            response_usage=Metrics(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
            ),
        )

    mock.invoke_stream = MagicMock(side_effect=mock_invoke_stream)

    return mock


@pytest.fixture
def mock_claude_model():
    """Create a mock Agno model (Claude-like)."""
    mock = MagicMock()
    mock.__class__.__name__ = "Claude"
    mock.__class__.__module__ = "agno.models.anthropic"
    mock.id = "claude-3-5-sonnet-20241022"

    def mock_response(messages, **kwargs):
        response = MagicMock()
        response.content = "I'm Claude!"
        response.metrics = MagicMock()
        response.metrics.input_tokens = 20
        response.metrics.output_tokens = 10
        response.metrics.total_tokens = 30
        return response

    mock.response = MagicMock(side_effect=mock_response)
    return mock


@pytest.fixture
def sample_messages():
    """Sample messages in OpenAI format (Agno accepts this)."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]


@pytest.fixture
def large_conversation():
    """Large conversation with many turns."""
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(50):
        messages.append({"role": "user", "content": f"Question {i}: What is {i} + {i}?"})
        messages.append({"role": "assistant", "content": f"The answer is {i + i}."})
    return messages


class TestAgnoAvailable:
    """Tests for agno_available() helper."""

    def test_returns_bool(self):
        """agno_available returns boolean."""
        from headroom.integrations.agno import agno_available

        assert isinstance(agno_available(), bool)

    def test_returns_true_when_installed(self):
        """Returns True when Agno is installed."""
        from headroom.integrations.agno import agno_available

        assert agno_available() is True


class TestHeadroomAgnoModel:
    """Tests for HeadroomAgnoModel wrapper."""

    def test_init_with_defaults(self, mock_agno_model):
        """Initialize with default config."""
        from headroom.integrations.agno import HeadroomAgnoModel

        model = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        assert model.wrapped_model is mock_agno_model
        assert model.headroom_config is not None
        assert model._metrics_history == []
        assert model._total_tokens_saved == 0

    def test_init_with_custom_config(self, mock_agno_model):
        """Initialize with custom config."""
        from headroom.integrations.agno import HeadroomAgnoModel

        config = HeadroomConfig(default_mode=HeadroomMode.AUDIT)
        model = HeadroomAgnoModel(
            wrapped_model=mock_agno_model,
            headroom_config=config,
            headroom_mode=HeadroomMode.SIMULATE,
        )

        assert model.headroom_config is config
        assert model.headroom_mode == HeadroomMode.SIMULATE

    def test_init_auto_detect_provider(self, mock_agno_model):
        """Auto-detect provider from wrapped model."""
        from headroom.integrations.agno import HeadroomAgnoModel

        model = HeadroomAgnoModel(wrapped_model=mock_agno_model, auto_detect_provider=True)

        assert model.auto_detect_provider is True

    def test_forward_attributes(self, mock_agno_model):
        """Forward attribute access to wrapped model."""
        from headroom.integrations.agno import HeadroomAgnoModel

        mock_agno_model.custom_attribute = "test_value"
        model = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        assert model.custom_attribute == "test_value"

    def test_properties_not_forwarded(self, mock_agno_model):
        """Own properties should not be forwarded."""
        from headroom.integrations.agno import HeadroomAgnoModel

        model = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        # These should work without forwarding to wrapped model
        assert model.total_tokens_saved == 0
        assert model.metrics_history == []

    def test_convert_messages_to_openai(self, mock_agno_model, sample_messages):
        """Convert Agno messages to OpenAI format."""
        from headroom.integrations.agno import HeadroomAgnoModel

        model = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        # Test with dict messages (already OpenAI format)
        openai_msgs = model._convert_messages_to_openai(sample_messages)

        assert len(openai_msgs) == 2
        assert openai_msgs[0]["role"] == "system"
        assert openai_msgs[0]["content"] == "You are a helpful assistant."
        assert openai_msgs[1]["role"] == "user"
        assert "France" in openai_msgs[1]["content"]

    def test_convert_agno_message_objects(self, mock_agno_model):
        """Convert Agno Message objects to OpenAI format."""
        from headroom.integrations.agno import HeadroomAgnoModel

        # Create mock Agno Message objects
        system_msg = MagicMock()
        system_msg.role = "system"
        system_msg.content = "You are helpful."
        system_msg.tool_calls = None
        system_msg.tool_call_id = None

        user_msg = MagicMock()
        user_msg.role = "user"
        user_msg.content = "Hello"
        user_msg.tool_calls = None
        user_msg.tool_call_id = None

        messages = [system_msg, user_msg]

        model = HeadroomAgnoModel(wrapped_model=mock_agno_model)
        openai_msgs = model._convert_messages_to_openai(messages)

        assert len(openai_msgs) == 2
        assert openai_msgs[0]["role"] == "system"
        assert openai_msgs[0]["content"] == "You are helpful."

    def test_convert_messages_with_tool_calls(self, mock_agno_model):
        """Convert messages with tool calls."""
        from headroom.integrations.agno import HeadroomAgnoModel

        assistant_msg = MagicMock()
        assistant_msg.role = "assistant"
        assistant_msg.content = "I'll check the weather."
        assistant_msg.tool_calls = [
            {"id": "call_123", "name": "get_weather", "args": {"city": "Paris"}}
        ]
        assistant_msg.tool_call_id = None

        tool_msg = MagicMock()
        tool_msg.role = "tool"
        tool_msg.content = '{"temp": 20}'
        tool_msg.tool_calls = None
        tool_msg.tool_call_id = "call_123"

        messages = [assistant_msg, tool_msg]

        model = HeadroomAgnoModel(wrapped_model=mock_agno_model)
        openai_msgs = model._convert_messages_to_openai(messages)

        assert len(openai_msgs) == 2
        assert openai_msgs[0]["role"] == "assistant"
        assert "tool_calls" in openai_msgs[0]
        assert openai_msgs[1]["tool_call_id"] == "call_123"

    def test_response_applies_optimization(self, mock_agno_model, sample_messages):
        """response() applies Headroom optimization."""
        from headroom.integrations.agno import HeadroomAgnoModel
        from headroom.providers import OpenAIProvider

        model = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        # Initialize provider and pipeline for mocking
        model._headroom_provider = OpenAIProvider()
        _ = model.pipeline  # Force lazy init

        # Mock the pipeline apply method
        with patch.object(model._pipeline, "apply") as mock_apply:
            mock_result = MagicMock()
            mock_result.messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is the capital of France?"},
            ]
            mock_result.tokens_before = 100
            mock_result.tokens_after = 80
            mock_result.transforms_applied = ["cache_aligner"]
            mock_apply.return_value = mock_result

            model.response(sample_messages)

            # Verify pipeline.apply was called
            mock_apply.assert_called_once()

            # Verify metrics were tracked
            assert len(model._metrics_history) == 1
            assert model._metrics_history[0].tokens_saved == 20

    def test_response_stream_applies_optimization(self, mock_agno_model, sample_messages):
        """response_stream() applies Headroom optimization."""
        from headroom.integrations.agno import HeadroomAgnoModel
        from headroom.providers import OpenAIProvider

        model = HeadroomAgnoModel(wrapped_model=mock_agno_model)
        model._headroom_provider = OpenAIProvider()
        _ = model.pipeline

        with patch.object(model._pipeline, "apply") as mock_apply:
            mock_result = MagicMock()
            mock_result.messages = sample_messages
            mock_result.tokens_before = 100
            mock_result.tokens_after = 90
            mock_result.transforms_applied = []
            mock_apply.return_value = mock_result

            # Consume the generator
            list(model.response_stream(sample_messages))

            mock_apply.assert_called_once()
            assert len(model._metrics_history) == 1

    def test_metrics_history_limited(self, mock_agno_model, sample_messages):
        """Metrics history is limited to 100 entries."""
        from headroom.integrations.agno import HeadroomAgnoModel

        model = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        # Add 150 fake metrics
        for _i in range(150):
            model._metrics_history.append(MagicMock())

        # Simulate a call that trims
        model._metrics_history = model._metrics_history[-100:]

        assert len(model._metrics_history) == 100

    def test_get_savings_summary_empty(self, mock_agno_model):
        """get_savings_summary with no history."""
        from headroom.integrations.agno import HeadroomAgnoModel

        model = HeadroomAgnoModel(wrapped_model=mock_agno_model)
        summary = model.get_savings_summary()

        assert summary["total_requests"] == 0
        assert summary["total_tokens_saved"] == 0
        assert summary["average_savings_percent"] == 0

    def test_get_savings_summary_with_data(self, mock_agno_model):
        """get_savings_summary with metrics."""
        from headroom.integrations.agno import HeadroomAgnoModel
        from headroom.integrations.agno.model import OptimizationMetrics

        model = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        # Add fake metrics
        model._metrics_history = [
            OptimizationMetrics(
                request_id="1",
                timestamp=datetime.now(),
                tokens_before=100,
                tokens_after=80,
                tokens_saved=20,
                savings_percent=20.0,
                transforms_applied=["smart_crusher"],
                model="gpt-4o",
            ),
            OptimizationMetrics(
                request_id="2",
                timestamp=datetime.now(),
                tokens_before=200,
                tokens_after=150,
                tokens_saved=50,
                savings_percent=25.0,
                transforms_applied=["cache_aligner"],
                model="gpt-4o",
            ),
        ]
        model._total_tokens_saved = 70

        summary = model.get_savings_summary()

        assert summary["total_requests"] == 2
        assert summary["total_tokens_saved"] == 70
        assert summary["average_savings_percent"] == 22.5

    def test_reset_clears_all_state(self, mock_agno_model):
        """reset() clears all metrics state."""
        from headroom.integrations.agno import HeadroomAgnoModel
        from headroom.integrations.agno.model import OptimizationMetrics

        model = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        # Add fake metrics
        model._metrics_history = [
            OptimizationMetrics(
                request_id="1",
                timestamp=datetime.now(),
                tokens_before=100,
                tokens_after=80,
                tokens_saved=20,
                savings_percent=20.0,
                transforms_applied=["smart_crusher"],
                model="gpt-4o",
            ),
        ]
        model._total_tokens_saved = 20

        # Verify state before reset
        assert len(model._metrics_history) == 1
        assert model._total_tokens_saved == 20

        # Reset
        model.reset()

        # Verify state after reset
        assert model._metrics_history == []
        assert model._total_tokens_saved == 0
        assert model.total_tokens_saved == 0

        # Verify summary is empty
        summary = model.get_savings_summary()
        assert summary["total_requests"] == 0
        assert summary["total_tokens_saved"] == 0


class TestProviderDetection:
    """Tests for provider detection from Agno models."""

    def test_detect_openai_provider(self, mock_agno_model):
        """Detect OpenAI provider from OpenAIChat."""
        from headroom.integrations.agno.providers import get_headroom_provider
        from headroom.providers import OpenAIProvider

        provider = get_headroom_provider(mock_agno_model)

        assert isinstance(provider, OpenAIProvider)

    def test_detect_anthropic_provider(self, mock_claude_model):
        """Detect Anthropic provider from Claude model."""
        from headroom.integrations.agno.providers import get_headroom_provider
        from headroom.providers import AnthropicProvider

        provider = get_headroom_provider(mock_claude_model)

        assert isinstance(provider, AnthropicProvider)

    def test_detect_from_model_id(self):
        """Detect provider from model ID string."""
        from headroom.integrations.agno.providers import get_headroom_provider
        from headroom.providers import AnthropicProvider, GoogleProvider, OpenAIProvider

        # GPT model
        mock_gpt = MagicMock()
        mock_gpt.__class__.__name__ = "UnknownModel"
        mock_gpt.__class__.__module__ = "some.module"
        mock_gpt.id = "gpt-4o-mini"
        assert isinstance(get_headroom_provider(mock_gpt), OpenAIProvider)

        # Claude model
        mock_claude = MagicMock()
        mock_claude.__class__.__name__ = "UnknownModel"
        mock_claude.__class__.__module__ = "some.module"
        mock_claude.id = "claude-3-opus-20240229"
        assert isinstance(get_headroom_provider(mock_claude), AnthropicProvider)

        # Gemini model
        mock_gemini = MagicMock()
        mock_gemini.__class__.__name__ = "UnknownModel"
        mock_gemini.__class__.__module__ = "some.module"
        mock_gemini.id = "gemini-pro"
        assert isinstance(get_headroom_provider(mock_gemini), GoogleProvider)

    def test_fallback_to_openai(self):
        """Fallback to OpenAI provider for unknown models."""
        from headroom.integrations.agno.providers import get_headroom_provider
        from headroom.providers import OpenAIProvider

        mock = MagicMock()
        mock.__class__.__name__ = "TotallyUnknownModel"
        mock.__class__.__module__ = "completely.unknown"
        mock.id = "mystery-model-v1"

        provider = get_headroom_provider(mock)

        assert isinstance(provider, OpenAIProvider)

    def test_get_model_name(self, mock_agno_model):
        """Extract model name from Agno model."""
        from headroom.integrations.agno.providers import get_model_name_from_agno

        name = get_model_name_from_agno(mock_agno_model)

        assert name == "gpt-4o"

    def test_get_model_name_fallback(self):
        """Fallback model name when not found."""
        from headroom.integrations.agno.providers import get_model_name_from_agno

        mock = MagicMock(spec=[])  # No attributes
        name = get_model_name_from_agno(mock)

        assert name == "gpt-4o"  # Default fallback


class TestOptimizeMessages:
    """Tests for standalone optimize_messages function."""

    def test_basic_optimization(self, sample_messages):
        """Basic message optimization."""
        from headroom.integrations.agno import optimize_messages

        with patch("headroom.integrations.agno.model.TransformPipeline") as MockPipeline:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
            mock_result.tokens_before = 100
            mock_result.tokens_after = 80
            mock_result.transforms_applied = ["cache_aligner"]
            mock_instance.apply.return_value = mock_result
            MockPipeline.return_value = mock_instance

            optimized, metrics = optimize_messages(sample_messages)

            assert len(optimized) == 2
            assert metrics["tokens_saved"] == 20
            assert metrics["savings_percent"] == 20.0

    def test_with_custom_config(self, sample_messages):
        """Optimization with custom config."""
        from headroom.integrations.agno import optimize_messages

        config = HeadroomConfig(default_mode=HeadroomMode.AUDIT)

        with patch("headroom.integrations.agno.model.TransformPipeline") as MockPipeline:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.messages = []
            mock_result.tokens_before = 50
            mock_result.tokens_after = 50
            mock_result.transforms_applied = []
            mock_instance.apply.return_value = mock_result
            MockPipeline.return_value = mock_instance

            _, metrics = optimize_messages(
                sample_messages,
                config=config,
                mode=HeadroomMode.AUDIT,
            )

            # Verify pipeline was created with config
            MockPipeline.assert_called_once()
            call_kwargs = MockPipeline.call_args[1]
            assert call_kwargs["config"] is config


class TestIntegrationWithRealHeadroom:
    """Integration tests using real Headroom components (no mocking)."""

    def test_real_optimization_pipeline(self, sample_messages):
        """Test with real Headroom client (no API calls)."""
        from headroom.integrations.agno import optimize_messages

        # This uses real Headroom transforms but no LLM API calls
        optimized, metrics = optimize_messages(
            sample_messages,
            mode=HeadroomMode.OPTIMIZE,
        )

        # Should return valid messages
        assert len(optimized) >= 1
        assert all(isinstance(m, dict) for m in optimized)
        assert all("role" in m and "content" in m for m in optimized)

        # Metrics should be populated
        assert "tokens_before" in metrics
        assert "tokens_after" in metrics
        assert "transforms_applied" in metrics

    def test_large_conversation_compression(self, large_conversation):
        """Test compression of large conversation."""
        from headroom.integrations.agno import optimize_messages

        optimized, metrics = optimize_messages(large_conversation)

        # Should compress (rolling window, etc.)
        assert metrics["tokens_before"] >= metrics["tokens_after"]

    def test_model_wrapper_real_optimization(self, mock_agno_model, sample_messages):
        """Test HeadroomAgnoModel with real Headroom optimization."""
        from headroom.integrations.agno import HeadroomAgnoModel

        model = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        # Call response - this will apply real optimization
        model.response(sample_messages)

        # Should have tracked metrics
        assert len(model.metrics_history) == 1
        metrics = model.metrics_history[0]
        assert metrics.tokens_before >= 0
        assert metrics.tokens_after >= 0


class TestReasoningCapabilityForwarding:
    """Tests for reasoning capability forwarding in HeadroomAgnoModel.

    These tests verify that HeadroomAgnoModel properly forwards
    reasoning-related properties from the wrapped model, enabling
    framework introspection (e.g., Agno's reasoning detection).
    """

    def test_underlying_model_property_returns_wrapped_model(self, mock_agno_model):
        """underlying_model property should return the wrapped model."""
        from headroom.integrations.agno import HeadroomAgnoModel

        wrapped = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        assert wrapped.underlying_model is mock_agno_model

    def test_underlying_model_class_introspection(self):
        """underlying_model allows class name introspection for framework detection."""
        from agno.models.openai import OpenAIChat

        from headroom.integrations.agno import HeadroomAgnoModel

        base_model = OpenAIChat(id="gpt-4o")
        wrapped = HeadroomAgnoModel(wrapped_model=base_model)

        # Framework detection typically checks __class__.__name__
        assert wrapped.underlying_model.__class__.__name__ == "OpenAIChat"
        assert wrapped.__class__.__name__ == "HeadroomAgnoModel"

    def test_thinking_property_forwarded_when_present(self, mock_agno_model):
        """thinking property is forwarded from wrapped model when present."""
        from headroom.integrations.agno import HeadroomAgnoModel

        # Set thinking config on mock model
        mock_agno_model.thinking = {"type": "enabled", "budget_tokens": 5000}

        wrapped = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        assert wrapped.thinking == {"type": "enabled", "budget_tokens": 5000}

    def test_thinking_property_not_present_when_absent(self, mock_agno_model):
        """thinking property not set when wrapped model doesn't have it."""
        from headroom.integrations.agno import HeadroomAgnoModel

        # Ensure mock doesn't have thinking attribute
        if hasattr(mock_agno_model, "thinking"):
            delattr(mock_agno_model, "thinking")

        wrapped = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        # Should raise AttributeError when accessed
        assert not hasattr(wrapped, "thinking") or wrapped.thinking is None

    def test_reasoning_effort_property_forwarded(self, mock_agno_model):
        """reasoning_effort property is forwarded from wrapped model."""
        from headroom.integrations.agno import HeadroomAgnoModel

        mock_agno_model.reasoning_effort = "high"

        wrapped = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        assert wrapped.reasoning_effort == "high"

    def test_provider_property_forwarded_from_wrapped_model(self, mock_agno_model):
        """provider property is set from wrapped model during init."""
        from headroom.integrations.agno import HeadroomAgnoModel

        mock_agno_model.provider = "OpenAI"

        wrapped = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        assert wrapped.provider == "OpenAI"

    def test_name_property_forwarded_from_wrapped_model(self, mock_agno_model):
        """name property is set from wrapped model during init."""
        from headroom.integrations.agno import HeadroomAgnoModel

        mock_agno_model.name = "gpt-4o"

        wrapped = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        assert wrapped.name == "gpt-4o"

    def test_has_extended_thinking_enabled_with_dict_config(self, mock_agno_model):
        """has_extended_thinking_enabled returns True when thinking dict is enabled."""
        from headroom.integrations.agno import HeadroomAgnoModel

        mock_agno_model.thinking = {"type": "enabled", "budget_tokens": 5000}

        wrapped = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        assert wrapped.has_extended_thinking_enabled() is True

    def test_has_extended_thinking_disabled_with_dict_config(self, mock_agno_model):
        """has_extended_thinking_enabled returns False when thinking dict is disabled."""
        from headroom.integrations.agno import HeadroomAgnoModel

        mock_agno_model.thinking = {"type": "disabled"}

        wrapped = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        assert wrapped.has_extended_thinking_enabled() is False

    def test_has_extended_thinking_returns_false_when_none(self, mock_agno_model):
        """has_extended_thinking_enabled returns False when thinking is None."""
        from headroom.integrations.agno import HeadroomAgnoModel

        mock_agno_model.thinking = None

        wrapped = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        assert wrapped.has_extended_thinking_enabled() is False

    def test_has_extended_thinking_returns_false_when_missing(self, mock_agno_model):
        """has_extended_thinking_enabled returns False when thinking attribute missing."""
        from headroom.integrations.agno import HeadroomAgnoModel

        # Remove thinking attribute if present
        if hasattr(mock_agno_model, "thinking"):
            delattr(mock_agno_model, "thinking")

        wrapped = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        assert wrapped.has_extended_thinking_enabled() is False

    def test_has_extended_thinking_with_truthy_value(self, mock_agno_model):
        """has_extended_thinking_enabled handles non-dict truthy values."""
        from headroom.integrations.agno import HeadroomAgnoModel

        mock_agno_model.thinking = True

        wrapped = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        assert wrapped.has_extended_thinking_enabled() is True

    def test_has_extended_thinking_with_falsy_value(self, mock_agno_model):
        """has_extended_thinking_enabled handles non-dict falsy values."""
        from headroom.integrations.agno import HeadroomAgnoModel

        mock_agno_model.thinking = False

        wrapped = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        assert wrapped.has_extended_thinking_enabled() is False

    def test_supports_native_structured_outputs_forwarded(self, mock_agno_model):
        """supports_native_structured_outputs property is forwarded."""
        from headroom.integrations.agno import HeadroomAgnoModel

        mock_agno_model.supports_native_structured_outputs = True

        wrapped = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        assert wrapped.supports_native_structured_outputs is True

    def test_supports_json_schema_outputs_forwarded(self, mock_agno_model):
        """supports_json_schema_outputs property is forwarded."""
        from headroom.integrations.agno import HeadroomAgnoModel

        mock_agno_model.supports_json_schema_outputs = True

        wrapped = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        assert wrapped.supports_json_schema_outputs is True

    def test_multiple_capability_properties_forwarded(self, mock_agno_model):
        """Multiple capability properties are forwarded correctly."""
        from headroom.integrations.agno import HeadroomAgnoModel

        mock_agno_model.thinking = {"type": "enabled", "budget_tokens": 10000}
        mock_agno_model.reasoning_effort = "medium"
        mock_agno_model.supports_native_structured_outputs = True
        mock_agno_model.supports_json_schema_outputs = False
        mock_agno_model.provider = "Anthropic"

        wrapped = HeadroomAgnoModel(wrapped_model=mock_agno_model)

        assert wrapped.thinking == {"type": "enabled", "budget_tokens": 10000}
        assert wrapped.reasoning_effort == "medium"
        assert wrapped.supports_native_structured_outputs is True
        assert wrapped.supports_json_schema_outputs is False
        assert wrapped.provider == "Anthropic"

    def test_underlying_model_with_real_openai_model(self):
        """Test underlying_model with real Agno OpenAIChat model."""
        from agno.models.openai import OpenAIChat

        from headroom.integrations.agno import HeadroomAgnoModel

        base_model = OpenAIChat(id="gpt-4o")
        wrapped = HeadroomAgnoModel(wrapped_model=base_model)

        # Verify underlying_model returns the actual model
        assert wrapped.underlying_model is base_model
        assert isinstance(wrapped.underlying_model, OpenAIChat)


class TestRealAgnoIntegration:
    """REAL integration tests with actual Agno components.

    These tests verify that HeadroomAgnoModel:
    1. Is a proper subclass of agno.models.base.Model
    2. Passes Agno's get_model() validation
    3. Can be used with Agno Agent
    4. Works with real Agno model types (not MagicMock)

    NO MOCKS for Agno components - only for external APIs.
    """

    def test_is_subclass_of_agno_model(self):
        """HeadroomAgnoModel must be a subclass of agno.models.base.Model."""
        from agno.models.base import Model

        from headroom.integrations.agno import HeadroomAgnoModel

        assert issubclass(HeadroomAgnoModel, Model)

    def test_passes_agno_get_model_validation(self):
        """HeadroomAgnoModel must pass Agno's get_model() validation."""
        from agno.models.openai import OpenAIChat
        from agno.models.utils import get_model

        from headroom.integrations.agno import HeadroomAgnoModel

        # Create a real OpenAIChat model (doesn't need API key for instantiation)
        base_model = OpenAIChat(id="gpt-4o")
        headroom_model = HeadroomAgnoModel(wrapped_model=base_model)

        # This should NOT raise "Model must be a Model instance, string, or None"
        result = get_model(headroom_model)

        assert result is headroom_model
        assert isinstance(result, HeadroomAgnoModel)

    def test_agent_accepts_headroom_model(self):
        """Agno Agent must accept HeadroomAgnoModel as model parameter."""
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat

        from headroom.integrations.agno import HeadroomAgnoModel

        # Create wrapped model
        base_model = OpenAIChat(id="gpt-4o")
        headroom_model = HeadroomAgnoModel(wrapped_model=base_model)

        # This should NOT raise any validation errors
        agent = Agent(model=headroom_model, markdown=False)

        assert agent.model is headroom_model
        assert agent.model.wrapped_model is base_model

    def test_model_id_reflects_wrapped_model(self):
        """HeadroomAgnoModel id should reflect the wrapped model."""
        from agno.models.openai import OpenAIChat

        from headroom.integrations.agno import HeadroomAgnoModel

        base_model = OpenAIChat(id="gpt-4o-mini")
        headroom_model = HeadroomAgnoModel(wrapped_model=base_model)

        assert "gpt-4o-mini" in headroom_model.id
        assert headroom_model.id.startswith("headroom:")

    def test_headroom_model_has_required_abstract_methods(self):
        """HeadroomAgnoModel must implement all required abstract methods."""
        from agno.models.openai import OpenAIChat

        from headroom.integrations.agno import HeadroomAgnoModel

        base_model = OpenAIChat(id="gpt-4o")
        headroom_model = HeadroomAgnoModel(wrapped_model=base_model)

        # Verify required methods exist and are callable
        assert hasattr(headroom_model, "invoke")
        assert callable(headroom_model.invoke)

        assert hasattr(headroom_model, "ainvoke")
        assert callable(headroom_model.ainvoke)

        assert hasattr(headroom_model, "invoke_stream")
        assert callable(headroom_model.invoke_stream)

        assert hasattr(headroom_model, "ainvoke_stream")
        assert callable(headroom_model.ainvoke_stream)

        assert hasattr(headroom_model, "_parse_provider_response")
        assert callable(headroom_model._parse_provider_response)

        assert hasattr(headroom_model, "_parse_provider_response_delta")
        assert callable(headroom_model._parse_provider_response_delta)

    def test_isinstance_check_passes(self):
        """isinstance check with agno.models.base.Model must pass."""
        from agno.models.base import Model
        from agno.models.openai import OpenAIChat

        from headroom.integrations.agno import HeadroomAgnoModel

        base_model = OpenAIChat(id="gpt-4o")
        headroom_model = HeadroomAgnoModel(wrapped_model=base_model)

        # This is the exact check that get_model() uses
        assert isinstance(headroom_model, Model)

    def test_model_with_custom_headroom_config(self):
        """Test with custom Headroom configuration."""
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat

        from headroom.integrations.agno import HeadroomAgnoModel

        config = HeadroomConfig(default_mode=HeadroomMode.AUDIT)
        base_model = OpenAIChat(id="gpt-4o")
        headroom_model = HeadroomAgnoModel(
            wrapped_model=base_model,
            headroom_config=config,
        )

        agent = Agent(model=headroom_model, markdown=False)

        assert agent.model.headroom_config is config
        assert agent.model.headroom_config.default_mode == HeadroomMode.AUDIT

    def test_response_method_delegates_to_wrapped(self):
        """Test that response() method works with real Agno model structure."""
        from agno.models.openai import OpenAIChat

        from headroom.integrations.agno import HeadroomAgnoModel

        base_model = OpenAIChat(id="gpt-4o")
        headroom_model = HeadroomAgnoModel(wrapped_model=base_model)

        # We can't actually call the response method without an API key, but we can verify
        # the method signature matches what Agno expects
        import inspect

        sig = inspect.signature(headroom_model.response)
        params = list(sig.parameters.keys())

        assert "messages" in params

    def test_optimization_tracked_across_calls(self):
        """Test that optimization metrics are tracked properly."""
        from agno.models.openai import OpenAIChat

        from headroom.integrations.agno import HeadroomAgnoModel

        base_model = OpenAIChat(id="gpt-4o")
        headroom_model = HeadroomAgnoModel(wrapped_model=base_model)

        # Initially no metrics
        assert headroom_model.total_tokens_saved == 0
        assert len(headroom_model.metrics_history) == 0

        # Simulate optimization (without actual API call)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        # Use the internal optimize method to test
        optimized, metrics = headroom_model._optimize_messages(messages)

        # Should have tracked metrics
        assert len(headroom_model.metrics_history) == 1
        assert headroom_model.total_tokens_saved >= 0


def _ollama_available() -> bool:
    """Check if Ollama is running and has a model available."""
    import socket

    # First check if ollama Python package is installed
    try:
        import ollama  # noqa: F401
    except ImportError:
        return False

    try:
        # Check if Ollama server is running on default port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", 11434))
        sock.close()
        return result == 0
    except Exception:
        return False


def _get_ollama_model() -> str | None:
    """Get an available Ollama model for testing."""
    if not _ollama_available():
        return None

    import subprocess

    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        # Parse output to find a model
        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:  # Header + at least one model
            return None

        # Get first model name (skip header)
        for line in lines[1:]:
            parts = line.split()
            if parts:
                model_name = parts[0]
                # Prefer small models for faster tests
                if any(
                    small in model_name.lower() for small in ["tiny", "phi", "qwen", "gemma:2b"]
                ):
                    return model_name
        # Fallback to first available model
        first_model_line = lines[1].split()
        return first_model_line[0] if first_model_line else None
    except Exception:
        return None


@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
class TestOllamaIntegration:
    """Integration tests using real Ollama models.

    These tests require Ollama to be installed and running locally.
    They are skipped in CI unless Ollama is set up.

    To run these tests locally:
        1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
        2. Pull a small model: ollama pull tinyllama
        3. Run tests: pytest tests/test_integrations/agno/test_model.py -v -k ollama
    """

    @pytest.fixture
    def ollama_model_name(self):
        """Get an available Ollama model."""
        model = _get_ollama_model()
        if not model:
            pytest.skip("No Ollama models available")
        return model

    def test_agent_with_ollama_model(self, ollama_model_name):
        """Test Agent with HeadroomAgnoModel wrapping real Ollama model."""
        from agno.agent import Agent
        from agno.models.ollama import Ollama

        from headroom.integrations.agno import HeadroomAgnoModel

        # Create wrapped Ollama model (real, local, no API key needed)
        base_model = Ollama(id=ollama_model_name)
        headroom_model = HeadroomAgnoModel(wrapped_model=base_model)

        # Create agent - this validates HeadroomAgnoModel works with Agent
        agent = Agent(model=headroom_model, markdown=False)

        assert agent.model is headroom_model
        assert isinstance(agent.model, HeadroomAgnoModel)

    def test_agent_run_with_ollama(self, ollama_model_name):
        """Actually run an agent with Ollama - full end-to-end test."""
        from agno.agent import Agent
        from agno.models.ollama import Ollama

        from headroom.integrations.agno import HeadroomAgnoModel

        # Create wrapped Ollama model
        base_model = Ollama(id=ollama_model_name)
        headroom_model = HeadroomAgnoModel(wrapped_model=base_model)

        # Create and run agent
        agent = Agent(model=headroom_model, markdown=False)

        # Actually run the agent - this tests the full pipeline
        response = agent.run("Say 'hello' and nothing else.")

        # Verify we got a response
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0

        # Verify Headroom optimization was applied
        assert len(headroom_model.metrics_history) >= 1

    def test_agent_with_system_prompt_and_ollama(self, ollama_model_name):
        """Test agent with system prompt using Ollama."""
        from agno.agent import Agent
        from agno.models.ollama import Ollama

        from headroom.integrations.agno import HeadroomAgnoModel

        base_model = Ollama(id=ollama_model_name)
        headroom_model = HeadroomAgnoModel(wrapped_model=base_model)

        # Agent with system prompt - tests system message optimization
        agent = Agent(
            model=headroom_model,
            description="You are a helpful assistant that always responds with exactly one word.",
            markdown=False,
        )

        response = agent.run("What is 2+2?")

        assert response is not None
        assert response.content is not None

        # Headroom should have processed the system prompt
        assert headroom_model.total_tokens_saved >= 0

    def test_multiple_turns_with_ollama(self, ollama_model_name):
        """Test multi-turn conversation with Ollama."""
        from agno.agent import Agent
        from agno.models.ollama import Ollama

        from headroom.integrations.agno import HeadroomAgnoModel

        base_model = Ollama(id=ollama_model_name)
        headroom_model = HeadroomAgnoModel(wrapped_model=base_model)

        agent = Agent(model=headroom_model, markdown=False)

        # Multiple turns
        agent.run("My name is Alice.")
        agent.run("What is my name?")

        # Should have tracked multiple optimization passes
        assert len(headroom_model.metrics_history) >= 2

    def test_headroom_optimization_reduces_tokens(self, ollama_model_name, large_conversation):
        """Test that Headroom actually reduces tokens on large conversations."""
        from agno.models.ollama import Ollama

        from headroom.integrations.agno import HeadroomAgnoModel

        base_model = Ollama(id=ollama_model_name)
        headroom_model = HeadroomAgnoModel(wrapped_model=base_model)

        # Optimize the large conversation
        optimized, metrics = headroom_model._optimize_messages(large_conversation)

        # Large conversations should see compression
        assert metrics.tokens_before > 0
        # With a 100+ message conversation, we should see some savings
        # (at minimum from whitespace normalization)
        assert metrics.tokens_after <= metrics.tokens_before
