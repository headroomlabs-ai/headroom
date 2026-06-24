"""Tests for CCR feedback loop and pattern learning."""

import time

import pytest

from headroom.cache.compression_feedback import (
    CompressionFeedback,
    LocalToolPattern,
    get_compression_feedback,
    reset_compression_feedback,
)
from headroom.cache.compression_store import (
    CompressionStore,
    RetrievalEvent,
    reset_compression_store,
)


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global state before each test."""
    reset_compression_feedback()
    reset_compression_store()
    yield
    reset_compression_feedback()
    reset_compression_store()


class TestCompressionFeedback:
    """Test CompressionFeedback analyzer."""

    def test_record_compression(self):
        """Recording compression events updates tool patterns."""
        feedback = CompressionFeedback()

        feedback.record_compression("test_tool", 100, 10)
        feedback.record_compression("test_tool", 200, 20)

        patterns = feedback.get_all_patterns()
        assert "test_tool" in patterns
        assert patterns["test_tool"].total_compressions == 2

    def test_record_retrieval(self):
        """Recording retrieval events updates patterns."""
        feedback = CompressionFeedback()
        feedback.record_compression("test_tool", 100, 10)

        event = RetrievalEvent(
            hash="abc123",
            query="find errors",
            items_retrieved=50,
            total_items=100,
            tool_name="test_tool",
            timestamp=time.time(),
            retrieval_type="search",
        )
        feedback.record_retrieval(event)

        patterns = feedback.get_all_patterns()
        assert patterns["test_tool"].total_retrievals == 1
        assert patterns["test_tool"].search_retrievals == 1

    def test_retrieval_rate_calculation(self):
        """Retrieval rate is calculated correctly."""
        feedback = CompressionFeedback()

        # 10 compressions
        for _ in range(10):
            feedback.record_compression("test_tool", 100, 10)

        # 5 retrievals (50% retrieval rate)
        for _ in range(5):
            event = RetrievalEvent(
                hash="abc123",
                query=None,
                items_retrieved=100,
                total_items=100,
                tool_name="test_tool",
                timestamp=time.time(),
                retrieval_type="full",
            )
            feedback.record_retrieval(event)

        pattern = feedback.get_all_patterns()["test_tool"]
        assert pattern.retrieval_rate == 0.5
        assert pattern.full_retrieval_rate == 1.0  # All were full retrievals

    def test_hints_default_with_no_data(self):
        """Default hints returned when no data exists."""
        feedback = CompressionFeedback()

        hints = feedback.get_compression_hints("unknown_tool")

        assert hints.max_items == 15  # Default
        assert hints.skip_compression is False
        assert "No pattern data" in hints.reason

    def test_hints_insufficient_samples(self):
        """Default hints returned with insufficient samples."""
        feedback = CompressionFeedback()

        # Only 3 compressions (need 5 for hints)
        for _ in range(3):
            feedback.record_compression("test_tool", 100, 10)

        hints = feedback.get_compression_hints("test_tool")

        assert hints.max_items == 15  # Default
        assert "Insufficient data" in hints.reason

    def test_hints_high_retrieval_rate_less_aggressive(self):
        """High retrieval rate results in less aggressive compression."""
        feedback = CompressionFeedback()

        # 10 compressions
        for _ in range(10):
            feedback.record_compression("test_tool", 100, 10)

        # 6 retrievals (60% retrieval rate - HIGH)
        for _ in range(6):
            event = RetrievalEvent(
                hash="abc123",
                query="search query",
                items_retrieved=50,
                total_items=100,
                tool_name="test_tool",
                timestamp=time.time(),
                retrieval_type="search",
            )
            feedback.record_retrieval(event)

        hints = feedback.get_compression_hints("test_tool")

        assert hints.max_items > 15  # Should be more than default
        assert hints.aggressiveness < 0.7  # Less aggressive
        assert "High retrieval rate" in hints.reason or "less aggressive" in hints.reason.lower()

    def test_hints_very_high_full_retrieval_skips_compression(self):
        """Very high full retrieval rate recommends skipping compression."""
        feedback = CompressionFeedback()

        # 10 compressions
        for _ in range(10):
            feedback.record_compression("test_tool", 100, 10)

        # 9 FULL retrievals (90% retrieval rate, all full)
        for _ in range(9):
            event = RetrievalEvent(
                hash="abc123",
                query=None,
                items_retrieved=100,
                total_items=100,
                tool_name="test_tool",
                timestamp=time.time(),
                retrieval_type="full",
            )
            feedback.record_retrieval(event)

        hints = feedback.get_compression_hints("test_tool")

        assert hints.skip_compression is True
        assert "skip compression" in hints.reason.lower()

    def test_hints_low_retrieval_rate_aggressive(self):
        """Low retrieval rate means current compression is effective."""
        feedback = CompressionFeedback()

        # 10 compressions
        for _ in range(10):
            feedback.record_compression("test_tool", 100, 10)

        # Only 1 retrieval (10% - LOW)
        event = RetrievalEvent(
            hash="abc123",
            query=None,
            items_retrieved=100,
            total_items=100,
            tool_name="test_tool",
            timestamp=time.time(),
            retrieval_type="full",
        )
        feedback.record_retrieval(event)

        hints = feedback.get_compression_hints("test_tool")

        assert hints.max_items == 15  # Default/aggressive
        assert "effective" in hints.reason.lower() or "Low retrieval" in hints.reason

    def test_common_queries_tracked(self):
        """Common search queries are tracked per tool."""
        feedback = CompressionFeedback()
        feedback.record_compression("test_tool", 100, 10)

        queries = ["find errors", "find errors", "status:failed", "error"]
        for q in queries:
            event = RetrievalEvent(
                hash="abc123",
                query=q,
                items_retrieved=10,
                total_items=100,
                tool_name="test_tool",
                timestamp=time.time(),
                retrieval_type="search",
            )
            feedback.record_retrieval(event)

        pattern = feedback.get_all_patterns()["test_tool"]
        assert "find errors" in pattern.common_queries
        assert pattern.common_queries["find errors"] == 2

    def test_queried_fields_extracted(self):
        """Field names are extracted from queries."""
        feedback = CompressionFeedback()
        feedback.record_compression("test_tool", 100, 10)

        # Query with field:value patterns
        event = RetrievalEvent(
            hash="abc123",
            query="status:error id=12345",
            items_retrieved=10,
            total_items=100,
            tool_name="test_tool",
            timestamp=time.time(),
            retrieval_type="search",
        )
        feedback.record_retrieval(event)

        pattern = feedback.get_all_patterns()["test_tool"]
        assert "status" in pattern.queried_fields
        assert "id" in pattern.queried_fields

    def test_preserve_fields_in_hints(self):
        """Frequently queried fields appear in hints."""
        feedback = CompressionFeedback()

        # Multiple compressions
        for _ in range(10):
            feedback.record_compression("test_tool", 100, 10)

        # Multiple queries with same fields
        for _ in range(5):
            event = RetrievalEvent(
                hash="abc123",
                query="status:error code:500",
                items_retrieved=10,
                total_items=100,
                tool_name="test_tool",
                timestamp=time.time(),
                retrieval_type="search",
            )
            feedback.record_retrieval(event)

        hints = feedback.get_compression_hints("test_tool")

        # Even if retrieval rate triggers hints, preserve_fields should be populated
        assert len(hints.preserve_fields) > 0

    def test_stats_returns_overview(self):
        """get_stats returns comprehensive overview."""
        feedback = CompressionFeedback()

        feedback.record_compression("tool_a", 100, 10)
        feedback.record_compression("tool_b", 200, 20)

        stats = feedback.get_stats()

        assert stats["total_compressions"] == 2
        assert stats["tools_tracked"] == 2
        assert "tool_a" in stats["tool_patterns"]
        assert "tool_b" in stats["tool_patterns"]

    def test_clear_resets_state(self):
        """clear() removes all learned patterns."""
        feedback = CompressionFeedback()
        feedback.record_compression("test_tool", 100, 10)

        feedback.clear()

        assert len(feedback.get_all_patterns()) == 0
        stats = feedback.get_stats()
        assert stats["total_compressions"] == 0


class TestLocalToolPattern:
    """Test LocalToolPattern dataclass."""

    def test_retrieval_rate_zero_compressions(self):
        """Retrieval rate is 0 when no compressions."""
        pattern = LocalToolPattern(tool_name="test")
        assert pattern.retrieval_rate == 0.0

    def test_full_retrieval_rate_zero_retrievals(self):
        """Full retrieval rate is 0 when no retrievals."""
        pattern = LocalToolPattern(tool_name="test")
        assert pattern.full_retrieval_rate == 0.0

    def test_search_rate_calculation(self):
        """Search rate is calculated correctly."""
        pattern = LocalToolPattern(
            tool_name="test",
            total_retrievals=10,
            full_retrievals=3,
            search_retrievals=7,
        )
        assert pattern.search_rate == 0.7


class TestGlobalFeedback:
    """Test global feedback singleton."""

    def test_singleton_returns_same_instance(self):
        """get_compression_feedback returns same instance."""
        fb1 = get_compression_feedback()
        fb2 = get_compression_feedback()
        assert fb1 is fb2

    def test_reset_clears_singleton(self):
        """reset_compression_feedback creates new instance."""
        fb1 = get_compression_feedback()
        fb1.record_compression("test", 100, 10)

        reset_compression_feedback()

        fb2 = get_compression_feedback()
        assert len(fb2.get_all_patterns()) == 0


class TestFeedbackIntegrationWithStore:
    """Test feedback integration with CompressionStore."""

    def test_store_notifies_feedback_on_retrieval(self):
        """CompressionStore adds events to pending for feedback processing."""
        store = CompressionStore()

        # Store content
        hash_key = store.store(
            original='[{"id": 1}, {"id": 2}]',
            compressed='[{"id": 1}]',
            original_item_count=2,
            compressed_item_count=1,
            tool_name="test_tool",
        )

        # Retrieve (should log event)
        store.retrieve(hash_key)

        # Process pending events (uses global feedback)
        store.process_pending_feedback()

        # Now global feedback should have the retrieval
        feedback = get_compression_feedback()
        patterns = feedback.get_all_patterns()
        assert "test_tool" in patterns
        assert patterns["test_tool"].total_retrievals == 1

    def test_search_retrieval_tracked_separately(self):
        """Search retrievals are tracked as search type."""
        store = CompressionStore()

        hash_key = store.store(
            original='[{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]',
            compressed='[{"id": 1}]',
            original_item_count=2,
            compressed_item_count=1,
            tool_name="test_tool",
        )

        # Search (should log as search type)
        store.search(hash_key, "alice")

        # Check events
        events = store.get_retrieval_events()
        assert len(events) > 0
        assert events[0].retrieval_type == "search"
        assert events[0].query == "alice"
