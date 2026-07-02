"""Tests for issue #1577: single message compression with ratio threshold.

When compression fails to meet the min_ratio threshold, the router should
add a `router:skipped:ratio_too_high:<slot_idx>` marker to transforms_applied
instead of returning an empty list (which results in router:noop).

This test deterministically drives the ratio-too-high branch by mocking
ContentRouter.compress() to return a result with compression_ratio >= min_ratio.
"""

from unittest.mock import MagicMock, patch

import pytest

from headroom.tokenizer import Tokenizer
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig


@pytest.fixture
def router():
    """Create a ContentRouter with minimal configuration."""
    config = ContentRouterConfig(
        skip_user_messages=False,
        protect_recent_code=0,
        protect_analysis_context=False,
    )
    return ContentRouter(config=config)


class MockTokenCounter:
    """Mock token counter that returns fixed values."""

    def count_text(self, text: str) -> int:
        return 100

    def count_message(self, message: dict) -> int:
        return 100

    def count_messages(self, messages: list) -> int:
        return 100


@pytest.fixture
def tokenizer():
    """Create a Tokenizer instance with a mock counter."""
    mock_counter = MockTokenCounter()
    return Tokenizer(token_counter=mock_counter)


def test_single_message_ratio_too_high_adds_marker(router, tokenizer):
    """
    Test that single message compression adds router:skipped:ratio_too_high
    marker when compression fails to meet the min_ratio threshold.
    """
    # Create a message with enough tokens to trigger compression
    large_text = "A" * 1000
    messages = [{"role": "user", "content": large_text}]

    # Mock compress() to return a result with high ratio (compression failed)
    mock_result = MagicMock()
    mock_result.compression_ratio = 0.95  # Higher than min_ratio threshold
    mock_result.strategy_used.value = "kompress"

    with patch.object(router, "compress", return_value=mock_result):
        result = router.apply(
            messages,
            tokenizer,
            compress_user_messages=True,
            protect_recent=0,
            min_tokens_to_compress=10,
        )

    # Verify transforms_applied contains the ratio_too_high marker
    assert "router:skipped:ratio_too_high:0" in result.transforms_applied
    assert "router:noop" not in result.transforms_applied


def test_multiple_messages_each_get_ratio_too_high_marker(router, tokenizer):
    """
    Test that each message that fails compression gets its own
    router:skipped:ratio_too_high marker with correct slot index.
    """
    messages = [
        {"role": "user", "content": "A" * 1000},
        {"role": "assistant", "content": "B" * 1000},
        {"role": "user", "content": "C" * 1000},
    ]

    # Mock compress() to return high ratio for all messages
    mock_result = MagicMock()
    mock_result.compression_ratio = 0.90  # Higher than min_ratio threshold
    mock_result.strategy_used.value = "kompress"

    with patch.object(router, "compress", return_value=mock_result):
        result = router.apply(
            messages,
            tokenizer,
            compress_user_messages=True,
            protect_recent=0,
            min_tokens_to_compress=10,
        )

    # Verify each message gets its own marker with correct slot index
    assert "router:skipped:ratio_too_high:0" in result.transforms_applied
    assert "router:skipped:ratio_too_high:1" in result.transforms_applied
    assert "router:skipped:ratio_too_high:2" in result.transforms_applied
    assert "router:noop" not in result.transforms_applied


def test_ratio_too_high_preserves_original_message(router, tokenizer):
    """
    Test that when compression fails due to ratio threshold,
    the original message is preserved in the output.
    """
    original_text = "Original content " * 100
    messages = [{"role": "user", "content": original_text}]

    # Mock compress() to return high ratio (compression failed)
    mock_result = MagicMock()
    mock_result.compression_ratio = 0.88  # Higher than min_ratio threshold
    mock_result.strategy_used.value = "kompress"

    with patch.object(router, "compress", return_value=mock_result):
        result = router.apply(
            messages,
            tokenizer,
            compress_user_messages=True,
            protect_recent=0,
            min_tokens_to_compress=10,
        )

    # Verify original message is preserved
    assert len(result.messages) == 1
    assert result.messages[0]["content"] == original_text
    # When no compression occurs, tokens_before should equal tokens_after
    assert result.tokens_before == result.tokens_after
