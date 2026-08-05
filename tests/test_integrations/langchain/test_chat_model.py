"""Comprehensive tests for LangChain integration.

Tests cover:
1. HeadroomChatModel - Wrapper for any BaseChatModel
2. HeadroomCallbackHandler - Metrics and observability
3. HeadroomRunnable - LCEL chain composition
4. optimize_messages() - Standalone optimization function
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

# Check if LangChain is available
try:
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatResult

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from headroom import HeadroomConfig, HeadroomMode

# Skip all tests if LangChain not installed
pytestmark = pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")


@pytest.fixture
def mock_chat_model():
    """Create a mock LangChain chat model."""
    mock = MagicMock()
    mock._llm_type = "mock-chat"
    mock._identifying_params = {"model": "mock-model"}
    mock.model_name = "gpt-4o"

    # Mock _generate to return a ChatResult
    def mock_generate(messages, **kwargs):
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content="Hello! I'm a mock response."),
                )
            ],
            llm_output={
                "token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            },
        )

    mock._generate = MagicMock(side_effect=mock_generate)
    mock._stream = MagicMock(
        return_value=iter([ChatGeneration(message=AIMessage(content="Streaming..."))])
    )

    return mock


@pytest.fixture
def sample_messages():
    """Sample LangChain messages for testing."""
    return [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?"),
    ]


@pytest.fixture
def large_tool_output():
    """Large tool output that should trigger compression."""
    items = [
        {"id": i, "name": f"Item {i}", "value": i * 100, "status": "active"} for i in range(100)
    ]
    return json.dumps(items)


class TestLangchainAvailable:
    """Tests for langchain_available() helper."""

    def test_returns_bool(self):
        """langchain_available returns boolean."""
        from headroom.integrations.langchain import langchain_available

        assert isinstance(langchain_available(), bool)

    def test_returns_true_when_installed(self):
        """Returns True when LangChain is installed."""
        from headroom.integrations.langchain import langchain_available

        assert langchain_available() is True


class TestHeadroomChatModel:
    """Tests for HeadroomChatModel wrapper."""

    def test_init_with_defaults(self, mock_chat_model):
        """Initialize with default config."""
        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(mock_chat_model)

        assert model.wrapped_model is mock_chat_model
        assert model.mode == HeadroomMode.OPTIMIZE
        assert model._metrics_history == []
        assert model._total_tokens_saved == 0

    def test_init_with_custom_config(self, mock_chat_model):
        """Initialize with custom config."""
        from headroom.integrations import HeadroomChatModel

        config = HeadroomConfig(default_mode=HeadroomMode.AUDIT)
        model = HeadroomChatModel(
            mock_chat_model,
            config=config,
            mode=HeadroomMode.SIMULATE,
        )

        assert model.headroom_config is config
        assert model.mode == HeadroomMode.SIMULATE

    def test_llm_type(self, mock_chat_model):
        """_llm_type includes wrapped model type."""
        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(mock_chat_model)
        assert "headroom" in model._llm_type
        assert "mock-chat" in model._llm_type

    def test_identifying_params(self, mock_chat_model):
        """_identifying_params includes wrapped model params."""
        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(mock_chat_model)
        params = model._identifying_params

        assert "wrapped_model" in params
        assert "headroom_mode" in params

    def test_convert_messages_to_openai(self, mock_chat_model, sample_messages):
        """Convert LangChain messages to OpenAI format."""
        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(mock_chat_model)
        openai_msgs = model._convert_messages_to_openai(sample_messages)

        assert len(openai_msgs) == 2
        assert openai_msgs[0]["role"] == "system"
        assert openai_msgs[0]["content"] == "You are a helpful assistant."
        assert openai_msgs[1]["role"] == "user"
        assert "France" in openai_msgs[1]["content"]

    def test_convert_messages_with_tool_calls(self, mock_chat_model):
        """Convert messages with tool calls."""
        from headroom.integrations import HeadroomChatModel

        messages = [
            HumanMessage(content="Get the weather"),
            AIMessage(
                content="I'll check the weather.",
                tool_calls=[{"id": "call_123", "name": "get_weather", "args": {"city": "Paris"}}],
            ),
            ToolMessage(content='{"temp": 20}', tool_call_id="call_123"),
        ]

        model = HeadroomChatModel(mock_chat_model)
        openai_msgs = model._convert_messages_to_openai(messages)

        assert len(openai_msgs) == 3
        assert openai_msgs[1]["role"] == "assistant"
        assert "tool_calls" in openai_msgs[1]
        assert openai_msgs[2]["role"] == "tool"
        assert openai_msgs[2]["tool_call_id"] == "call_123"

    def test_convert_messages_from_openai(self, mock_chat_model):
        """Convert OpenAI format back to LangChain."""
        from headroom.integrations import HeadroomChatModel

        openai_msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        model = HeadroomChatModel(mock_chat_model)
        lc_msgs = model._convert_messages_from_openai(openai_msgs)

        assert len(lc_msgs) == 3
        assert isinstance(lc_msgs[0], SystemMessage)
        assert isinstance(lc_msgs[1], HumanMessage)
        assert isinstance(lc_msgs[2], AIMessage)

    def test_generate_applies_optimization(self, mock_chat_model, sample_messages):
        """_generate applies Headroom optimization."""
        from headroom.integrations import HeadroomChatModel
        from headroom.providers import OpenAIProvider

        model = HeadroomChatModel(mock_chat_model)

        # Initialize provider and pipeline for mocking
        model._provider = OpenAIProvider()
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

            model._generate(sample_messages)

            # Verify pipeline.apply was called
            mock_apply.assert_called_once()

            # Verify metrics were tracked
            assert len(model._metrics_history) == 1
            assert model._metrics_history[0].tokens_saved == 20

    def test_metrics_history_limited(self, mock_chat_model, sample_messages):
        """Metrics history is limited to 100 entries."""
        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(mock_chat_model)

        # Add 150 fake metrics
        for _i in range(150):
            model._metrics_history.append(MagicMock())

        # Simulate a call that trims
        model._metrics_history = model._metrics_history[-100:]

        assert len(model._metrics_history) == 100

    def test_get_savings_summary_empty(self, mock_chat_model):
        """get_savings_summary with no history."""
        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(mock_chat_model)
        summary = model.get_savings_summary()

        assert summary["total_requests"] == 0
        assert summary["total_tokens_saved"] == 0
        assert summary["average_savings_percent"] == 0

    def test_get_savings_summary_with_data(self, mock_chat_model):
        """get_savings_summary with metrics."""
        from headroom.integrations import HeadroomChatModel
        from headroom.integrations.langchain import OptimizationMetrics

        model = HeadroomChatModel(mock_chat_model)

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


class TestHeadroomCallbackHandler:
    """Tests for HeadroomCallbackHandler."""

    def test_init_defaults(self):
        """Initialize with default settings."""
        from headroom.integrations import HeadroomCallbackHandler

        handler = HeadroomCallbackHandler()

        assert handler.log_level == "INFO"
        assert handler.token_alert_threshold is None
        assert handler.total_tokens == 0
        assert handler.total_requests == 0

    def test_init_with_thresholds(self):
        """Initialize with alert thresholds."""
        from headroom.integrations import HeadroomCallbackHandler

        handler = HeadroomCallbackHandler(
            token_alert_threshold=10000,
            cost_alert_threshold=1.0,
        )

        assert handler.token_alert_threshold == 10000
        assert handler.cost_alert_threshold == 1.0

    def test_on_chat_model_start(self):
        """Track chat model start."""
        from headroom.integrations import HeadroomCallbackHandler

        handler = HeadroomCallbackHandler()

        messages = [[HumanMessage(content="Hello, how are you?")]]
        handler.on_chat_model_start(
            serialized={"name": "ChatOpenAI", "id": ["langchain", "ChatOpenAI"]},
            messages=messages,
        )

        assert handler._current_request is not None
        assert "start_time" in handler._current_request
        assert handler._current_request["message_count"] == 1

    def test_on_chat_model_start_triggers_alert(self):
        """Alert triggered when tokens exceed threshold."""
        from headroom.integrations import HeadroomCallbackHandler

        handler = HeadroomCallbackHandler(token_alert_threshold=5)

        # Long message to exceed threshold
        messages = [[HumanMessage(content="A" * 100)]]
        handler.on_chat_model_start(
            serialized={"name": "ChatOpenAI"},
            messages=messages,
        )

        assert len(handler.alerts) > 0
        assert "Token alert" in handler.alerts[0]

    def test_on_llm_end_tracks_tokens(self):
        """Track tokens on LLM completion."""
        from headroom.integrations import HeadroomCallbackHandler

        handler = HeadroomCallbackHandler()
        handler._current_request = {"start_time": datetime.now()}

        response = MagicMock()
        response.llm_output = {
            "token_usage": {
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "total_tokens": 70,
            }
        }

        handler.on_llm_end(response)

        assert handler.total_tokens == 70
        assert handler.total_requests == 1
        assert handler.requests[0]["total_tokens"] == 70

    def test_on_llm_error(self):
        """Track errors."""
        from headroom.integrations import HeadroomCallbackHandler

        handler = HeadroomCallbackHandler()
        handler._current_request = {"start_time": datetime.now()}

        handler.on_llm_error(ValueError("Test error"), run_id=uuid4())

        assert handler.total_requests == 1
        assert "error" in handler.requests[0]

    def test_get_summary(self):
        """Get summary statistics."""
        from headroom.integrations import HeadroomCallbackHandler

        handler = HeadroomCallbackHandler()

        # Add some requests
        handler._requests = [
            {"total_tokens": 100, "duration_ms": 500},
            {"total_tokens": 200, "duration_ms": 300},
            {"error": "failed", "duration_ms": 0},
        ]

        summary = handler.get_summary()

        assert summary["total_requests"] == 3
        assert summary["successful_requests"] == 2
        assert summary["total_tokens"] == 300
        assert summary["errors"] == 1

    def test_reset(self):
        """Reset clears all state."""
        from headroom.integrations import HeadroomCallbackHandler

        handler = HeadroomCallbackHandler()
        handler._requests = [{"test": 1}]
        handler._total_tokens = 100
        handler._alerts = ["alert"]

        handler.reset()

        assert handler.total_requests == 0
        assert handler.total_tokens == 0
        assert len(handler.alerts) == 0


class TestHeadroomRunnable:
    """Tests for HeadroomRunnable LCEL component."""

    def test_init_defaults(self):
        """Initialize with defaults."""
        from headroom.integrations.langchain import HeadroomRunnable

        runnable = HeadroomRunnable()

        assert runnable.mode == HeadroomMode.OPTIMIZE
        assert runnable.config is not None

    def test_init_custom_config(self):
        """Initialize with custom config."""
        from headroom.integrations.langchain import HeadroomRunnable

        config = HeadroomConfig(default_mode=HeadroomMode.AUDIT)
        runnable = HeadroomRunnable(config=config, mode=HeadroomMode.SIMULATE)

        assert runnable.config is config
        assert runnable.mode == HeadroomMode.SIMULATE

    def test_as_runnable(self):
        """Convert to LangChain Runnable."""
        from langchain_core.runnables import RunnableLambda

        from headroom.integrations.langchain import HeadroomRunnable

        runnable = HeadroomRunnable()
        lc_runnable = runnable.as_runnable()

        assert isinstance(lc_runnable, RunnableLambda)

    def test_optimize_messages(self, sample_messages):
        """Optimize list of messages."""
        from headroom.integrations.langchain import HeadroomRunnable
        from headroom.providers import OpenAIProvider

        runnable = HeadroomRunnable()

        # Initialize provider and pipeline for mocking
        runnable._provider = OpenAIProvider()
        _ = runnable.pipeline  # Force lazy init

        with patch.object(runnable._pipeline, "apply") as mock_apply:
            mock_result = MagicMock()
            mock_result.messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
            mock_result.tokens_before = 50
            mock_result.tokens_after = 40
            mock_result.transforms_applied = []
            mock_apply.return_value = mock_result

            result = runnable._optimize(sample_messages)

            assert len(result) == 2
            assert isinstance(result[0], SystemMessage)


class TestOptimizeMessages:
    """Tests for standalone optimize_messages function."""

    def test_basic_optimization(self, sample_messages):
        """Basic message optimization."""
        from headroom.integrations import optimize_messages

        with patch("headroom.integrations.langchain.chat_model.TransformPipeline") as MockPipeline:
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
        from headroom.integrations import optimize_messages

        config = HeadroomConfig(default_mode=HeadroomMode.AUDIT)

        with patch("headroom.integrations.langchain.chat_model.TransformPipeline") as MockPipeline:
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

    def test_with_tool_messages(self):
        """Optimization with tool messages."""
        from headroom.integrations import optimize_messages

        messages = [
            HumanMessage(content="Get weather"),
            AIMessage(
                content="Checking...",
                tool_calls=[{"id": "1", "name": "weather", "args": {}}],
            ),
            ToolMessage(content="Sunny", tool_call_id="1"),
        ]

        with patch("headroom.integrations.langchain.chat_model.TransformPipeline") as MockPipeline:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.messages = [
                {"role": "user", "content": "Get weather"},
                {
                    "role": "assistant",
                    "content": "Checking...",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {"name": "weather", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "1", "content": "Sunny"},
            ]
            mock_result.tokens_before = 100
            mock_result.tokens_after = 90
            mock_result.transforms_applied = []
            mock_instance.apply.return_value = mock_result
            MockPipeline.return_value = mock_instance

            optimized, metrics = optimize_messages(messages)

            assert len(optimized) == 3
            assert isinstance(optimized[2], ToolMessage)


class TestIntegrationWithRealHeadroom:
    """Integration tests using real Headroom components (no mocking)."""

    def test_real_optimization_pipeline(self, sample_messages):
        """Test with real Headroom client (no API calls)."""
        from headroom.integrations import optimize_messages

        # This uses real Headroom transforms but no LLM API calls
        optimized, metrics = optimize_messages(
            sample_messages,
            mode=HeadroomMode.OPTIMIZE,
        )

        # Should return valid messages
        assert len(optimized) >= 1
        assert all(
            isinstance(m, (SystemMessage, HumanMessage, AIMessage, ToolMessage)) for m in optimized
        )

        # Metrics should be populated
        assert "tokens_before" in metrics
        assert "tokens_after" in metrics
        assert "transforms_applied" in metrics

    def test_large_conversation_compression(self):
        """Test compression of large conversation."""
        from headroom.integrations import optimize_messages

        # Create large conversation
        messages = [SystemMessage(content="You are a helpful assistant.")]
        for i in range(50):
            messages.append(HumanMessage(content=f"Question {i}: What is {i} + {i}?"))
            messages.append(AIMessage(content=f"The answer is {i + i}."))

        optimized, metrics = optimize_messages(messages)

        # Should compress (rolling window, etc.)
        assert metrics["tokens_before"] >= metrics["tokens_after"]


class TestAinvokeStreamingTrue:
    """Tests for ainvoke() / _agenerate() when wrapped model has streaming=True.

    Reproduces and verifies the fix for GitHub #1285:
    AttributeError: 'AsyncStream' object has no attribute 'model_dump'
    """

    @pytest.fixture
    def streaming_mock_model(self):
        """Create a mock model with streaming=True that simulates the bug.

        When streaming=True, _agenerate returns a raw AsyncStream-like object
        (no model_dump). When streaming=False, it returns a proper ChatResult.
        This mirrors ChatOpenAI's real behavior with streaming=True.

        The mock supports model_copy() so that the per-call copy approach
        works correctly: the copy gets streaming=False and its own
        _agenerate that references the copy (not the original).
        """
        mock = MagicMock()
        mock._llm_type = "mock-streaming"
        mock._identifying_params = {"model": "mock-streaming-model"}
        mock.model_name = "gpt-4o"
        mock.streaming = True

        class FakeAsyncStream:
            """Simulates openai.AsyncStream — has no model_dump attribute."""

            async def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        def make_agenerate(model_ref):
            """Create an _agenerate that checks model_ref.streaming.

            This closure pattern lets the per-call copy's _agenerate
            see the copy's streaming=False, while the original's
            _agenerate sees streaming=True.
            """

            async def agenerate(messages, **kwargs):
                if model_ref.streaming:
                    # Simulate the bug: return raw AsyncStream instead of ChatResult
                    return FakeAsyncStream()
                # Non-streaming path returns a proper ChatResult
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(content="Hello from non-streaming path!"),
                        )
                    ],
                    llm_output={
                        "token_usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        }
                    },
                )

            return agenerate

        mock._agenerate = make_agenerate(mock)

        # Also mock _generate for completeness
        def mock_generate(messages, **kwargs):
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content="Hello from sync path!"),
                    )
                ],
                llm_output={
                    "token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                },
            )

        mock._generate = MagicMock(side_effect=mock_generate)
        mock._stream = MagicMock(
            return_value=iter([ChatGeneration(message=AIMessage(content="Streaming..."))])
        )

        # Configure model_copy to return a shallow copy with updated streaming
        # and its own _agenerate that references the copy (not the original).
        def mock_model_copy(update=None, **kwargs):
            new_mock = MagicMock()
            new_mock._llm_type = mock._llm_type
            new_mock._identifying_params = mock._identifying_params
            new_mock.model_name = mock.model_name
            new_mock.streaming = mock.streaming
            new_mock._generate = mock._generate
            new_mock._stream = mock._stream
            if update:
                for key, value in update.items():
                    setattr(new_mock, key, value)
            new_mock._agenerate = make_agenerate(new_mock)
            return new_mock

        mock.model_copy = mock_model_copy

        return mock

    @pytest.fixture
    def _patched_pipeline(self, streaming_mock_model):
        """Create a HeadroomChatModel with a mocked optimization pipeline."""
        from headroom.integrations import HeadroomChatModel
        from headroom.providers import OpenAIProvider

        model = HeadroomChatModel(streaming_mock_model)
        model._provider = OpenAIProvider()
        _ = model.pipeline  # Force lazy init

        mock_result = MagicMock()
        mock_result.messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is the capital of France?"},
        ]
        mock_result.tokens_before = 100
        mock_result.tokens_after = 80
        mock_result.transforms_applied = ["cache_aligner"]

        ctx = patch.object(model._pipeline, "apply", return_value=mock_result)
        ctx.model = model  # type: ignore[attr-defined]
        return ctx

    async def test_agenerate_returns_chatresult_with_streaming_true(
        self, streaming_mock_model, sample_messages
    ):
        """_agenerate() returns a ChatResult (not AsyncStream) when streaming=True.

        This is the core fix for #1285 — without the fix, the mock's _agenerate
        returns a FakeAsyncStream and the test would fail the isinstance check.
        """
        from headroom.integrations import HeadroomChatModel
        from headroom.providers import OpenAIProvider

        model = HeadroomChatModel(streaming_mock_model)
        model._provider = OpenAIProvider()
        _ = model.pipeline

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

            result = await model._agenerate(sample_messages)

            # Must be a ChatResult, not a FakeAsyncStream
            assert isinstance(result, ChatResult)
            assert len(result.generations) == 1
            assert isinstance(result.generations[0].message, AIMessage)
            assert result.generations[0].message.content == "Hello from non-streaming path!"

    async def test_streaming_never_changed_after_agenerate(
        self, streaming_mock_model, sample_messages
    ):
        """streaming=True is never changed on the wrapped model during/after _agenerate()."""
        from headroom.integrations import HeadroomChatModel
        from headroom.providers import OpenAIProvider

        model = HeadroomChatModel(streaming_mock_model)
        model._provider = OpenAIProvider()
        _ = model.pipeline

        with patch.object(model._pipeline, "apply") as mock_apply:
            mock_result = MagicMock()
            mock_result.messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
            mock_result.tokens_before = 50
            mock_result.tokens_after = 40
            mock_result.transforms_applied = []
            mock_apply.return_value = mock_result

            # Before the call, streaming is True
            assert streaming_mock_model.streaming is True

            await model._agenerate(sample_messages)

            # After the call, streaming must still be True — never mutated
            assert streaming_mock_model.streaming is True

    async def test_original_streaming_unchanged_during_agenerate(
        self, streaming_mock_model, sample_messages
    ):
        """Original model's streaming is never changed during _agenerate().

        With the per-call copy approach, the original model is never mutated.
        The copy gets streaming=False, which is why we still get a ChatResult.
        """
        from headroom.integrations import HeadroomChatModel
        from headroom.providers import OpenAIProvider

        model = HeadroomChatModel(streaming_mock_model)
        model._provider = OpenAIProvider()
        _ = model.pipeline

        # Track the original model's streaming state when model_copy is called
        streaming_states_when_copy: list[bool] = []
        original_model_copy = streaming_mock_model.model_copy

        def tracking_model_copy(update=None, **kwargs):
            streaming_states_when_copy.append(streaming_mock_model.streaming)
            return original_model_copy(update=update, **kwargs)

        streaming_mock_model.model_copy = tracking_model_copy

        with patch.object(model._pipeline, "apply") as mock_apply:
            mock_result = MagicMock()
            mock_result.messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
            mock_result.tokens_before = 50
            mock_result.tokens_after = 40
            mock_result.transforms_applied = []
            mock_apply.return_value = mock_result

            result = await model._agenerate(sample_messages)

            # The original model's streaming was True when model_copy was called
            assert streaming_states_when_copy == [True]
            # And it's still True after the call
            assert streaming_mock_model.streaming is True
            # The copy had streaming=False, so we got a ChatResult
            assert isinstance(result, ChatResult)

    async def test_streaming_unchanged_on_exception(self, streaming_mock_model, sample_messages):
        """Original model's streaming is unchanged even if _agenerate raises."""
        from headroom.integrations import HeadroomChatModel
        from headroom.providers import OpenAIProvider

        model = HeadroomChatModel(streaming_mock_model)
        model._provider = OpenAIProvider()
        _ = model.pipeline

        # Make the copy's _agenerate raise
        async def failing_agenerate(messages, **kwargs):
            raise RuntimeError("upstream error")

        original_model_copy = streaming_mock_model.model_copy

        def failing_model_copy(update=None, **kwargs):
            new_mock = original_model_copy(update=update, **kwargs)
            new_mock._agenerate = failing_agenerate
            return new_mock

        streaming_mock_model.model_copy = failing_model_copy

        with patch.object(model._pipeline, "apply") as mock_apply:
            mock_result = MagicMock()
            mock_result.messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
            mock_result.tokens_before = 50
            mock_result.tokens_after = 40
            mock_result.transforms_applied = []
            mock_apply.return_value = mock_result

            with pytest.raises(RuntimeError, match="upstream error"):
                await model._agenerate(sample_messages)

            # Original model's streaming is unchanged — never mutated
            assert streaming_mock_model.streaming is True

    async def test_concurrent_ainvoke_no_race_condition(
        self, streaming_mock_model, sample_messages
    ):
        """Concurrent _agenerate() calls don't race on shared model state.

        With the per-call copy approach, two overlapping _agenerate() calls
        each get their own copy with streaming=False, so the original model's
        streaming is never mutated. Both calls return ChatResult.
        """
        from headroom.integrations import HeadroomChatModel
        from headroom.providers import OpenAIProvider

        model = HeadroomChatModel(streaming_mock_model)
        model._provider = OpenAIProvider()
        _ = model.pipeline

        # Wrap model_copy to add a delay inside _agenerate, forcing overlap
        original_model_copy = streaming_mock_model.model_copy

        def slow_model_copy(update=None, **kwargs):
            new_mock = original_model_copy(update=update, **kwargs)
            base_agenerate = new_mock._agenerate

            async def slow_agenerate(messages, **kwargs):
                await asyncio.sleep(0.1)
                return await base_agenerate(messages, **kwargs)

            new_mock._agenerate = slow_agenerate
            return new_mock

        streaming_mock_model.model_copy = slow_model_copy

        with patch.object(model._pipeline, "apply") as mock_apply:
            mock_result = MagicMock()
            mock_result.messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
            mock_result.tokens_before = 50
            mock_result.tokens_after = 40
            mock_result.transforms_applied = []
            mock_apply.return_value = mock_result

            # Original streaming must be True before
            assert streaming_mock_model.streaming is True

            # Two concurrent calls
            results = await asyncio.gather(
                model._agenerate(sample_messages),
                model._agenerate(sample_messages),
            )

            # Original streaming was never mutated — still True
            assert streaming_mock_model.streaming is True

            # Both calls returned ChatResult (not AsyncStream)
            for r in results:
                assert isinstance(r, ChatResult)
                assert len(r.generations) == 1
                assert isinstance(r.generations[0].message, AIMessage)

    async def test_agenerate_no_streaming_attr_passthrough(self, sample_messages):
        """_agenerate() works when wrapped model has no streaming attribute."""
        from headroom.integrations import HeadroomChatModel
        from headroom.providers import OpenAIProvider

        # Mock without streaming attribute — MagicMock auto-generates
        # attributes, so we must explicitly delete streaming to simulate
        # a model that genuinely lacks it.
        mock = MagicMock()
        mock._llm_type = "mock-no-streaming"
        mock._identifying_params = {"model": "mock-model"}
        mock.model_name = "gpt-4o"
        del mock.streaming  # simulate no streaming attribute

        async def mock_agenerate(messages, **kwargs):
            return ChatResult(
                generations=[
                    ChatGeneration(message=AIMessage(content="No streaming attr!")),
                ],
                llm_output={
                    "token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                },
            )

        mock._agenerate = mock_agenerate

        model = HeadroomChatModel(mock)
        model._provider = OpenAIProvider()
        _ = model.pipeline

        with patch.object(model._pipeline, "apply") as mock_apply:
            mock_result = MagicMock()
            mock_result.messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
            mock_result.tokens_before = 50
            mock_result.tokens_after = 40
            mock_result.transforms_applied = []
            mock_apply.return_value = mock_result

            result = await model._agenerate(sample_messages)

            assert isinstance(result, ChatResult)
            assert result.generations[0].message.content == "No streaming attr!"


# ============================================================================
# Real Ollama Integration Tests (no mocks, actual LLM calls)
# ============================================================================


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


def _langchain_ollama_available() -> bool:
    """Check if langchain-ollama package is installed."""
    try:
        from langchain_ollama import ChatOllama  # noqa: F401

        return True
    except ImportError:
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


@pytest.mark.skipif(
    not (_ollama_available() and _langchain_ollama_available()),
    reason="Ollama not running or langchain-ollama not installed",
)
class TestOllamaIntegration:
    """Integration tests using real Ollama models with LangChain.

    These tests require Ollama to be installed and running locally.
    They are skipped in CI unless Ollama is set up.

    To run these tests locally:
        1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
        2. Pull a small model: ollama pull llama2
        3. Run tests: pytest tests/test_integrations/langchain/test_chat_model.py -v -k ollama
    """

    @pytest.fixture
    def ollama_model_name(self):
        """Get an available Ollama model."""
        model = _get_ollama_model()
        if not model:
            pytest.skip("No Ollama models available")
        return model

    def test_headroom_chat_model_with_ollama(self, ollama_model_name):
        """Test HeadroomChatModel wrapping real ChatOllama model."""
        from langchain_ollama import ChatOllama

        from headroom.integrations import HeadroomChatModel

        # Create real Ollama model (no API key needed)
        base_model = ChatOllama(model=ollama_model_name)
        headroom_model = HeadroomChatModel(base_model)

        assert headroom_model.wrapped_model is base_model
        assert isinstance(headroom_model, HeadroomChatModel)

    def test_invoke_with_ollama(self, ollama_model_name):
        """Actually invoke an LLM call with Ollama - full end-to-end test."""
        from langchain_ollama import ChatOllama

        from headroom.integrations import HeadroomChatModel

        base_model = ChatOllama(model=ollama_model_name)
        headroom_model = HeadroomChatModel(base_model)

        messages = [
            SystemMessage(content="You are a helpful assistant. Be very brief."),
            HumanMessage(content="What is 2+2? Answer with just the number."),
        ]

        # This makes a real LLM call
        result = headroom_model.invoke(messages)

        assert result is not None
        assert result.content is not None
        assert len(result.content) > 0

    def test_generate_with_ollama(self, ollama_model_name):
        """Test _generate method with real Ollama model."""
        from langchain_ollama import ChatOllama

        from headroom.integrations import HeadroomChatModel

        base_model = ChatOllama(model=ollama_model_name)
        headroom_model = HeadroomChatModel(base_model)

        messages = [
            HumanMessage(content="Say 'hello' and nothing else."),
        ]

        result = headroom_model._generate(messages)

        assert result is not None
        assert len(result.generations) > 0
        assert result.generations[0].message.content is not None

    def test_optimization_tracked_with_ollama(self, ollama_model_name):
        """Test that optimization metrics are tracked with real calls."""
        from langchain_ollama import ChatOllama

        from headroom.integrations import HeadroomChatModel

        base_model = ChatOllama(model=ollama_model_name)
        headroom_model = HeadroomChatModel(base_model)

        # Make a call with some messages
        messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hi"),
        ]

        headroom_model.invoke(messages)

        # Metrics should be tracked
        assert len(headroom_model._metrics_history) >= 1

    def test_multiple_turns_with_ollama(self, ollama_model_name):
        """Test multi-turn conversation with real Ollama."""
        from langchain_ollama import ChatOllama

        from headroom.integrations import HeadroomChatModel

        base_model = ChatOllama(model=ollama_model_name)
        headroom_model = HeadroomChatModel(base_model)

        # First turn
        messages = [
            SystemMessage(content="You are a helpful assistant. Be brief."),
            HumanMessage(content="My name is Alice."),
        ]
        response1 = headroom_model.invoke(messages)

        # Second turn - add previous exchange
        messages.append(AIMessage(content=response1.content))
        messages.append(HumanMessage(content="What is my name?"))

        response2 = headroom_model.invoke(messages)

        assert response2 is not None
        assert response2.content is not None
        # Model should remember the name from context
        assert len(response2.content) > 0

    def test_headroom_optimization_reduces_tokens(self, ollama_model_name):
        """Test that Headroom optimization actually reduces token count."""
        from langchain_ollama import ChatOllama

        from headroom.integrations import HeadroomChatModel

        base_model = ChatOllama(model=ollama_model_name)
        headroom_model = HeadroomChatModel(base_model, mode=HeadroomMode.OPTIMIZE)

        # Create a conversation with repetitive content that should be compressed
        messages = [SystemMessage(content="You are a helpful assistant.")]
        for i in range(20):
            messages.append(HumanMessage(content=f"Question {i}: What is {i} + {i}?"))
            messages.append(AIMessage(content=f"The answer to {i} + {i} is {i + i}."))
        messages.append(HumanMessage(content="What was question 5?"))

        # This should trigger compression
        headroom_model.invoke(messages)

        # Check that some optimization was tracked
        if headroom_model._metrics_history:
            metrics = headroom_model._metrics_history[-1]
            # With a large conversation, we expect some savings
            assert metrics.tokens_before >= metrics.tokens_after

    def test_callback_handler_with_ollama(self, ollama_model_name):
        """Test HeadroomCallbackHandler with real Ollama calls."""
        from langchain_ollama import ChatOllama

        from headroom.integrations import HeadroomCallbackHandler, HeadroomChatModel

        base_model = ChatOllama(model=ollama_model_name)
        headroom_model = HeadroomChatModel(base_model)
        handler = HeadroomCallbackHandler()

        messages = [
            HumanMessage(content="Say 'test' and nothing else."),
        ]

        # Invoke with callback
        headroom_model.invoke(messages, config={"callbacks": [handler]})

        # Handler should have tracked the request
        assert handler.total_requests >= 1


@pytest.mark.skipif(
    not (_ollama_available() and _langchain_ollama_available()),
    reason="Ollama not running or langchain-ollama not installed",
)
class TestRealLangChainIntegration:
    """Real integration tests that validate LangChain components work together.

    These use real Ollama but focus on LangChain-specific functionality.
    """

    @pytest.fixture
    def ollama_model_name(self):
        """Get an available Ollama model."""
        model = _get_ollama_model()
        if not model:
            pytest.skip("No Ollama models available")
        return model

    def test_lcel_chain_with_headroom(self, ollama_model_name):
        """Test LCEL chain composition with HeadroomChatModel."""
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_ollama import ChatOllama

        from headroom.integrations import HeadroomChatModel

        # Create chain: prompt -> headroom model -> output parser
        base_model = ChatOllama(model=ollama_model_name)
        headroom_model = HeadroomChatModel(base_model)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant. Be very brief."),
                ("human", "{input}"),
            ]
        )

        chain = prompt | headroom_model | StrOutputParser()

        # Invoke the chain
        result = chain.invoke({"input": "What is 1+1? Just the number."})

        assert result is not None
        assert len(result) > 0

    def test_optimize_messages_standalone_with_ollama_types(self, ollama_model_name):
        """Test standalone optimize_messages function with real message types."""
        from headroom.integrations import optimize_messages

        # Use real LangChain message types
        messages = [
            SystemMessage(content="You are a math tutor."),
            HumanMessage(content="What is calculus?"),
            AIMessage(content="Calculus is the mathematical study of continuous change."),
            HumanMessage(content="Can you give me an example?"),
        ]

        optimized, metrics = optimize_messages(messages)

        # Should return valid LangChain messages
        assert len(optimized) >= 1
        assert all(
            isinstance(m, (SystemMessage, HumanMessage, AIMessage, ToolMessage)) for m in optimized
        )
        assert "tokens_before" in metrics
        assert "tokens_after" in metrics
