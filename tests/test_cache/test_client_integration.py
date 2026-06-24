"""Tests for HeadroomClient cache optimizer integration."""

import os
import tempfile
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from headroom import (
    AnthropicCacheOptimizer,
    HeadroomClient,
)
from headroom.cache.base import CacheMetrics, CacheResult


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield f"sqlite:///{path}"
    if os.path.exists(path):
        os.unlink(path)


class MockTokenCounter:
    """Mock token counter for testing."""

    def count_text(self, text: str) -> int:
        """Count tokens in text (required by Tokenizer interface)."""
        return len(text) // 4

    def count_tokens(self, text: str) -> int:
        """Alias for count_text."""
        return self.count_text(text)

    def count_message(self, message: dict) -> int:
        """Count tokens in a single message."""
        content = message.get("content", "")
        if isinstance(content, str):
            return len(content) // 4
        elif isinstance(content, list):
            total = 0
            for block in content:
                if isinstance(block, dict):
                    total += len(block.get("text", "")) // 4
            return total
        return 0

    def count_messages(self, messages: list) -> int:
        """Count tokens in messages."""
        return sum(self.count_message(msg) for msg in messages)


class MockAnthropicProvider:
    """Mock Anthropic provider for testing."""

    name = "anthropic"

    def get_token_counter(self, model: str):
        return MockTokenCounter()

    def get_context_limit(self, model: str) -> int:
        return 200000


class MockOpenAIProvider:
    """Mock OpenAI provider for testing."""

    name = "openai"

    def get_token_counter(self, model: str):
        return MockTokenCounter()

    def get_context_limit(self, model: str) -> int:
        return 128000


# Mock response classes for testing (avoid MagicMock in sqlite)


@dataclass
class MockTextBlock:
    """Mock text block for Anthropic response."""

    type: str = "text"
    text: str = "Hello!"


@dataclass
class MockUsage:
    """Mock usage for Anthropic response."""

    input_tokens: int = 100
    output_tokens: int = 20


@dataclass
class MockAnthropicResponse:
    """Mock Anthropic API response."""

    content: list = None
    usage: MockUsage = None
    model: str = "claude-sonnet-4-20250514"
    id: str = "msg_123"
    stop_reason: str = "end_turn"

    def __post_init__(self):
        if self.content is None:
            self.content = [MockTextBlock()]
        if self.usage is None:
            self.usage = MockUsage()


class TestHeadroomClientCacheIntegration:
    """Test HeadroomClient cache optimizer integration."""

    def test_auto_detect_anthropic_optimizer(self, temp_db):
        """Test that Anthropic optimizer is auto-detected."""
        mock_client = MagicMock()
        provider = MockAnthropicProvider()

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
            enable_cache_optimizer=True,
        )

        assert client._cache_optimizer is not None
        assert client._cache_optimizer.name == "anthropic-cache-optimizer"

    def test_auto_detect_openai_optimizer(self, temp_db):
        """Test that OpenAI optimizer is auto-detected."""
        mock_client = MagicMock()
        provider = MockOpenAIProvider()

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
            enable_cache_optimizer=True,
        )

        assert client._cache_optimizer is not None
        assert client._cache_optimizer.name == "openai-prefix-stabilizer"

    def test_custom_optimizer(self, temp_db):
        """Test using a custom optimizer."""
        mock_client = MagicMock()
        provider = MockAnthropicProvider()
        custom_optimizer = AnthropicCacheOptimizer()

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
            cache_optimizer=custom_optimizer,
        )

        assert client._cache_optimizer is custom_optimizer

    def test_disable_cache_optimizer(self, temp_db):
        """Test disabling cache optimizer."""
        mock_client = MagicMock()
        provider = MockAnthropicProvider()

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
            enable_cache_optimizer=False,
        )

        assert client._cache_optimizer is None

    def test_semantic_cache_layer_creation(self, temp_db):
        """Test semantic cache layer is created when enabled."""
        mock_client = MagicMock()
        provider = MockAnthropicProvider()

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
            enable_cache_optimizer=True,
            enable_semantic_cache=True,
        )

        assert client._semantic_cache_layer is not None
        assert client._cache_optimizer is not None

    def test_extract_query_from_string_content(self, temp_db):
        """Test query extraction from string content."""
        mock_client = MagicMock()
        provider = MockAnthropicProvider()

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
        )

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        query = client._extract_query(messages)
        assert query == "What is 2+2?"

    def test_extract_query_from_content_blocks(self, temp_db):
        """Test query extraction from content block format."""
        mock_client = MagicMock()
        provider = MockAnthropicProvider()

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
        )

        messages = [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is 2+2?"}],
            },
        ]

        query = client._extract_query(messages)
        assert query == "What is 2+2?"

    def test_extract_query_last_user_message(self, temp_db):
        """Test that query extraction uses last user message."""
        mock_client = MagicMock()
        provider = MockAnthropicProvider()

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
        )

        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]

        query = client._extract_query(messages)
        assert query == "Second question"

    def test_config_propagation(self, temp_db):
        """Test that config is properly propagated."""
        mock_client = MagicMock()
        provider = MockAnthropicProvider()

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
            enable_cache_optimizer=True,
            enable_semantic_cache=True,
        )

        assert client._config.cache_optimizer.enabled is True
        assert client._config.cache_optimizer.enable_semantic_cache is True


class TestCacheOptimizerInvocation:
    """Test that cache optimizer is actually INVOKED during chat completion.

    These tests catch bugs where the optimizer is assigned but never called
    in the production code path.
    """

    @patch("headroom.storage.sqlite.SQLiteStorage.save")
    def test_optimizer_optimize_is_called_during_chat(self, mock_save, temp_db):
        """CRITICAL: Verify optimizer.optimize() is called during chat completion.

        This test catches the gap where tests verify assignment but not invocation.
        Note: Cache optimizer is only invoked in OPTIMIZE mode, not AUDIT mode (the default).
        """
        from headroom import HeadroomMode

        # Use module-level mock classes to avoid sqlite issues with MagicMock
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockAnthropicResponse()

        provider = MockAnthropicProvider()

        # Create a spy optimizer to track calls
        real_optimizer = AnthropicCacheOptimizer()
        spy_optimize = MagicMock(
            return_value=CacheResult(
                messages=[{"role": "user", "content": "test"}],
                metrics=CacheMetrics(
                    cacheable_tokens=100,
                    breakpoints_inserted=1,
                    estimated_cache_hit=False,
                    estimated_savings_percent=0.0,
                ),
                transforms_applied=["test_transform"],
            )
        )
        real_optimizer.optimize = spy_optimize

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
            cache_optimizer=real_optimizer,
        )

        # Make a chat completion call in OPTIMIZE mode (cache optimizer only runs in OPTIMIZE mode)
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
        ]

        client.chat.completions.create(
            model="claude-sonnet-4-20250514",
            messages=messages,
            max_tokens=100,
            headroom_mode=HeadroomMode.OPTIMIZE,
        )

        # CRITICAL: Verify optimizer.optimize() was actually called
        assert spy_optimize.called, (
            "Cache optimizer.optimize() should be called during chat completion. "
            "If this fails, the optimizer is assigned but never invoked."
        )

        # Verify it was called with the right arguments
        call_args = spy_optimize.call_args
        assert call_args is not None
        optimized_messages, context = call_args[0]
        assert len(optimized_messages) >= 1, "Should pass messages to optimizer"

    @patch("headroom.storage.sqlite.SQLiteStorage.save")
    def test_optimizer_transforms_applied_in_response(self, mock_save, temp_db):
        """Verify optimizer transforms are reported in the response metadata."""
        from headroom import HeadroomMode

        # Use module-level mock classes to avoid sqlite issues with MagicMock
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockAnthropicResponse()

        provider = MockAnthropicProvider()

        # Create optimizer that applies a transform
        real_optimizer = AnthropicCacheOptimizer()
        real_optimizer.optimize = MagicMock(
            return_value=CacheResult(
                messages=[{"role": "user", "content": "test"}],
                metrics=CacheMetrics(
                    cacheable_tokens=500,
                    breakpoints_inserted=2,
                    estimated_cache_hit=True,
                    estimated_savings_percent=0.5,
                ),
                transforms_applied=["add_cache_control"],
            )
        )

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
            cache_optimizer=real_optimizer,
        )

        messages = [
            {"role": "user", "content": "x" * 1000},  # Large message
        ]

        # Use OPTIMIZE mode so cache optimizer is invoked
        result = client.chat.completions.create(
            model="claude-sonnet-4-20250514",
            messages=messages,
            max_tokens=100,
            headroom_mode=HeadroomMode.OPTIMIZE,
        )

        # Verify the response includes cache optimizer info
        assert hasattr(result, "headroom"), "Response should have headroom metadata"
        headroom_meta = result.headroom

        # Check that cache optimizer was reported
        assert headroom_meta.cache_optimizer_used is not None or any(
            "cache_optimizer" in t for t in (headroom_meta.transforms_applied or [])
        ), "Cache optimizer usage should be reported in metadata"

    @patch("headroom.storage.sqlite.SQLiteStorage.save")
    def test_optimizer_not_called_in_audit_mode(self, mock_save, temp_db):
        """Verify optimizer is NOT called in AUDIT mode (observe only)."""
        from headroom import HeadroomMode

        # Use module-level mock classes to avoid sqlite issues with MagicMock
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockAnthropicResponse()

        provider = MockAnthropicProvider()

        spy_optimize = MagicMock(
            return_value=CacheResult(
                messages=[{"role": "user", "content": "test"}],
                metrics=CacheMetrics(),
            )
        )
        real_optimizer = AnthropicCacheOptimizer()
        real_optimizer.optimize = spy_optimize

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
            cache_optimizer=real_optimizer,
        )

        messages = [{"role": "user", "content": "Hello"}]

        # Make call in AUDIT mode (observe only, no modifications)
        client.chat.completions.create(
            model="claude-sonnet-4-20250514",
            messages=messages,
            max_tokens=100,
            headroom_mode=HeadroomMode.AUDIT,
        )

        # Optimizer should NOT be called in AUDIT mode
        assert not spy_optimize.called, "Cache optimizer should NOT be called in AUDIT mode"


class TestSemanticCacheIntegration:
    """Test semantic cache integration with HeadroomClient.

    These tests verify the full production code path for semantic caching,
    including that cache hits actually return cached responses without calling
    the underlying API.
    """

    @patch("headroom.storage.sqlite.SQLiteStorage.save")
    def test_semantic_cache_hit_returns_cached_response_without_api_call(self, mock_save, temp_db):
        """CRITICAL: Verify semantic cache hit returns cached response without API call.

        This test catches the gap where semantic cache is enabled but cached
        responses are never actually returned (API is always called).
        """
        from headroom import HeadroomMode

        # Mock OpenAI-style response (chat.completions.create uses OpenAI API style)
        mock_client = MagicMock()
        mock_openai_response = MagicMock()
        mock_openai_response.choices = [MagicMock(message=MagicMock(content="4"))]
        mock_openai_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        mock_openai_response.model = "claude-sonnet-4-20250514"
        mock_openai_response.id = "chatcmpl-123"
        mock_client.chat.completions.create.return_value = mock_openai_response

        provider = MockAnthropicProvider()

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
            enable_cache_optimizer=True,
            enable_semantic_cache=True,
        )

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        # First call - should call API and potentially cache
        client.chat.completions.create(
            model="claude-sonnet-4-20250514",
            messages=messages,
            max_tokens=100,
            headroom_mode=HeadroomMode.OPTIMIZE,
        )

        first_call_count = mock_client.chat.completions.create.call_count
        assert first_call_count == 1, "First call should hit API"

        # Manually store response in semantic cache for test
        if client._semantic_cache_layer is not None:
            from headroom.cache import OptimizationContext

            context = OptimizationContext(
                provider="anthropic",
                model="claude-sonnet-4-20250514",
                query="What is 2+2?",
            )
            client._semantic_cache_layer.store_response(
                messages,
                {"text": "4", "role": "assistant"},
                context,
            )

        # Second call with same messages - should hit cache, NOT call API
        client.chat.completions.create(
            model="claude-sonnet-4-20250514",
            messages=messages,
            max_tokens=100,
            headroom_mode=HeadroomMode.OPTIMIZE,
        )

        second_call_count = mock_client.chat.completions.create.call_count

        # If semantic cache is working, API should NOT be called again
        assert second_call_count == 1, (
            f"Semantic cache hit should NOT call API. "
            f"Expected 1 API call, got {second_call_count}. "
            "If this fails, cached responses are not being returned."
        )


class TestSessionStatsTracking:
    """Test session statistics tracking in HeadroomClient.

    These tests verify that session stats are actually updated during
    chat completion calls.
    """

    @patch("headroom.storage.sqlite.SQLiteStorage.save")
    def test_session_stats_incremented_after_request(self, mock_save, temp_db):
        """CRITICAL: Verify session stats are incremented after requests."""
        from headroom import HeadroomMode

        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockAnthropicResponse()

        provider = MockAnthropicProvider()

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
        )

        # Get initial stats
        initial_stats = client.get_stats()
        initial_requests = initial_stats["session"]["requests_total"]

        # Make a request in AUDIT mode
        messages = [{"role": "user", "content": "Hello"}]
        client.chat.completions.create(
            model="claude-sonnet-4-20250514",
            messages=messages,
            max_tokens=100,
            headroom_mode=HeadroomMode.AUDIT,
        )

        # Verify stats were updated
        after_stats = client.get_stats()
        after_requests = after_stats["session"]["requests_total"]

        assert after_requests == initial_requests + 1, (
            f"requests_total should increment. Before: {initial_requests}, After: {after_requests}"
        )
        assert after_stats["session"]["requests_audit"] >= 1, (
            "requests_audit should be at least 1 after AUDIT mode request"
        )

    @patch("headroom.storage.sqlite.SQLiteStorage.save")
    def test_session_stats_tracks_optimize_mode(self, mock_save, temp_db):
        """Verify session stats track OPTIMIZE mode requests separately."""
        from headroom import HeadroomMode

        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockAnthropicResponse()

        provider = MockAnthropicProvider()

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
        )

        messages = [{"role": "user", "content": "Hello"}]

        # Make request in OPTIMIZE mode
        client.chat.completions.create(
            model="claude-sonnet-4-20250514",
            messages=messages,
            max_tokens=100,
            headroom_mode=HeadroomMode.OPTIMIZE,
        )

        stats = client.get_stats()

        assert stats["session"]["requests_optimized"] >= 1, (
            "requests_optimized should be at least 1 after OPTIMIZE mode request"
        )

    @patch("headroom.storage.sqlite.SQLiteStorage.save")
    def test_session_stats_tracks_tokens_saved(self, mock_save, temp_db):
        """Verify session stats track tokens saved."""
        from headroom import HeadroomMode

        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockAnthropicResponse()

        provider = MockAnthropicProvider()

        client = HeadroomClient(
            original_client=mock_client,
            provider=provider,
            store_url=temp_db,
        )

        # Create a conversation that will trigger some optimization
        messages = [
            {"role": "system", "content": "You are helpful. " * 100},
            {"role": "user", "content": "Hello"},
        ]

        client.chat.completions.create(
            model="claude-sonnet-4-20250514",
            messages=messages,
            max_tokens=100,
            headroom_mode=HeadroomMode.OPTIMIZE,
        )

        stats = client.get_stats()

        # tokens_saved_total should be tracked (may be 0 if no compression)
        assert "tokens_saved_total" in stats["session"], (
            "Session stats should track tokens_saved_total"
        )
