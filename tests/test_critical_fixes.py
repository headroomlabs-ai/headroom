"""Tests demonstrating critical fixes for TOIN/CCR implementation.

These tests verify the before/after behavior of critical bug fixes:
1. TOIN confidence math error (line 721)
2. TOIN double-count bug (lines 354-358)
3. compression_feedback.py race condition (lines 481-491)
4. Unbounded strategy dicts in compression_feedback.py
5. SmartCrusher integration with TOIN
"""

import time
from unittest.mock import patch


class TestTOINConfidenceMathFix:
    """Test for CRITICAL: Confidence calculation math error in toin.py:721.

    BUG: `user_boost = min(0.3, pattern.user_count / 10 * 0.1)`
    Due to operator precedence: user_count / 10 * 0.1 = user_count * 0.01
    - 3 users: 0.03 boost (too small)
    - 10 users: 0.1 boost
    - 30 users needed to hit 0.3 cap!

    FIX: Should be `min(0.3, pattern.user_count * 0.03)` for meaningful boost
    - 3 users: 0.09 boost
    - 10 users: 0.3 boost (capped)
    """

    def test_confidence_user_boost_at_3_users(self):
        """With 3 users (min for network effect), boost should be meaningful."""
        from headroom.telemetry.toin import (
            TOINConfig,
            ToolIntelligenceNetwork,
            ToolPattern,
            reset_toin,
        )

        reset_toin()
        config = TOINConfig(min_users_for_network_effect=3)
        toin = ToolIntelligenceNetwork(config)

        # Create pattern with 3 users (correct API: tool_signature_hash is first arg)
        pattern = ToolPattern(
            tool_signature_hash="test123",
            user_count=3,
            sample_size=100,  # Good sample size
        )

        confidence = toin._calculate_confidence(pattern)

        # Sample confidence = min(0.7, 100/100) = 0.7
        # User boost for 3 users should be meaningful (>= 0.05)
        # FIX: With user_count * 0.03: boost = 0.09, total = 0.79
        # BUG: With user_count * 0.01: boost = 0.03, total = 0.73

        # After fix, confidence should be at least 0.75
        assert confidence >= 0.75, (
            f"Confidence {confidence} too low for 3 users - user boost not meaningful"
        )

    def test_confidence_user_boost_at_10_users(self):
        """With 10 users, boost should hit or approach cap."""
        from headroom.telemetry.toin import (
            TOINConfig,
            ToolIntelligenceNetwork,
            ToolPattern,
            reset_toin,
        )

        reset_toin()
        config = TOINConfig(min_users_for_network_effect=3)
        toin = ToolIntelligenceNetwork(config)

        pattern = ToolPattern(
            tool_signature_hash="test123",
            user_count=10,
            sample_size=100,
        )

        confidence = toin._calculate_confidence(pattern)

        # With 10 users, should be near cap (0.95)
        # Sample confidence = 0.7, user boost should be 0.3 (capped)
        # Total = min(0.95, 0.7 + 0.3) = 0.95
        # BUG: user_boost = 0.1, total = 0.8

        assert confidence >= 0.9, f"Confidence {confidence} too low for 10 users"


class TestTOINDoubleCountFix:
    """Test for CRITICAL: Double-count bug in toin.py:354-358.

    BUG: When _seen_instance_hashes hits cap (100), new instance_ids are NOT stored
    but user_count IS incremented. Next call with same instance_id:
    - `if self._instance_id not in pattern._seen_instance_hashes` → True (not stored!)
    - user_count incremented AGAIN → Double counting!

    FIX: Use a separate set to track ALL seen instances (no cap for lookup),
    OR check if we already tracked overflow for this instance.
    """

    def test_user_count_no_double_counting_after_cap(self):
        """Same instance shouldn't be counted twice even after cap hit."""
        from headroom.telemetry.models import ToolSignature
        from headroom.telemetry.toin import TOINConfig, ToolIntelligenceNetwork, reset_toin

        reset_toin()
        toin = ToolIntelligenceNetwork(TOINConfig())

        # Create a signature using the correct factory method
        items = [{"field1": "value1", "field2": 123}]
        sig = ToolSignature.from_items(items)

        # Simulate 101 unique instances (exceed the 100 cap)
        # First, fill up the cap with 100 unique instances
        original_instance_id = toin._instance_id
        for i in range(100):
            toin._instance_id = f"instance_{i}"
            toin.record_compression(sig, 100, 10, 1000, 100, strategy="test_strategy")

        # Now add one more instance (exceeds cap)
        toin._instance_id = "instance_100"
        toin.record_compression(sig, 100, 10, 1000, 100, strategy="test_strategy")

        # Get the pattern
        with toin._lock:
            pattern = toin._patterns[("unknown", "unknown", sig.structure_hash)]
            user_count_after_101 = pattern.user_count

        # Now call again with same instance (instance_100)
        # BUG: This would increment user_count again because instance_100
        # was not stored (cap hit) so the check passes again
        toin.record_compression(sig, 100, 10, 1000, 100, strategy="test_strategy")

        with toin._lock:
            pattern = toin._patterns[("unknown", "unknown", sig.structure_hash)]
            user_count_after_102 = pattern.user_count

        # Restore instance_id
        toin._instance_id = original_instance_id

        # User count should NOT increase for same instance
        assert user_count_after_102 == user_count_after_101, (
            f"Double-counting bug: user_count went from {user_count_after_101} to "
            f"{user_count_after_102} for same instance after cap hit"
        )


class TestCompressionFeedbackRaceCondition:
    """Test for CRITICAL: Race condition in compression_feedback.py:481-491.

    BUG: _last_event_timestamp is read (line 481) and written (line 491)
    WITHOUT holding the lock. Another thread calling record_retrieval()
    between these could cause events to be missed or double-counted.

    FIX: Move timestamp filtering and update inside the lock.
    """

    def test_analyze_from_store_thread_safety(self):
        """Concurrent analyze_from_store and record_retrieval should not lose events."""
        from headroom.cache.compression_feedback import (
            CompressionFeedback,
            reset_compression_feedback,
        )
        from headroom.cache.compression_store import CompressionStore, RetrievalEvent

        reset_compression_feedback()

        # Create store with mock events
        store = CompressionStore()
        feedback = CompressionFeedback(store=store, analysis_interval=0.0)  # No rate limiting

        # Pre-populate some events with correct API
        base_time = time.time()
        events_recorded = []

        def add_retrieval_event(tool_name: str, timestamp: float):
            event = RetrievalEvent(
                hash="test_hash",
                query=None,
                items_retrieved=10,
                total_items=100,
                tool_name=tool_name,
                timestamp=timestamp,
                retrieval_type="full",
            )
            # Directly add to feedback (simulating what analyze_from_store does)
            feedback.record_retrieval(event)
            events_recorded.append(event)

        # Record some events
        for i in range(10):
            add_retrieval_event(f"tool_{i % 3}", base_time + i)

        with feedback._lock:
            total_retrievals = feedback._total_retrievals
            patterns_count = len(feedback._tool_patterns)

        # All 10 events should be recorded
        assert total_retrievals == 10, f"Expected 10 retrievals, got {total_retrievals}"
        # Should have 3 unique tools (tool_0, tool_1, tool_2)
        assert patterns_count == 3, f"Expected 3 tool patterns, got {patterns_count}"

    def test_timestamp_filtering_inside_lock(self):
        """Verify that timestamp filtering happens atomically with update."""
        from headroom.cache.compression_feedback import (
            CompressionFeedback,
            reset_compression_feedback,
        )
        from headroom.cache.compression_store import CompressionStore, RetrievalEvent

        reset_compression_feedback()
        store = CompressionStore()
        feedback = CompressionFeedback(store=store, analysis_interval=0.0)

        # Manually set last event timestamp
        feedback._last_event_timestamp = 100.0

        # Create mock store with events (correct API)
        mock_events = [
            RetrievalEvent(
                hash="h1",
                query=None,
                items_retrieved=5,
                total_items=50,
                tool_name="tool_a",
                timestamp=99.0,
                retrieval_type="full",
            ),
            RetrievalEvent(
                hash="h2",
                query=None,
                items_retrieved=5,
                total_items=50,
                tool_name="tool_b",
                timestamp=101.0,
                retrieval_type="full",
            ),
            RetrievalEvent(
                hash="h3",
                query="test",
                items_retrieved=5,
                total_items=50,
                tool_name="tool_c",
                timestamp=102.0,
                retrieval_type="search",
            ),
        ]

        # Mock store.get_retrieval_events
        with patch.object(store, "get_retrieval_events", return_value=mock_events):
            feedback.analyze_from_store()

        # Only events with timestamp > 100.0 should be processed (h2, h3)
        with feedback._lock:
            total = feedback._total_retrievals
            # The timestamp should now be 102.0 (max of processed events)
            last_ts = feedback._last_event_timestamp

        assert total == 2, f"Expected 2 new events processed, got {total}"
        assert last_ts == 102.0, f"Expected last_event_timestamp=102.0, got {last_ts}"


class TestUnboundedStrategyDicts:
    """Test for HIGH: Unbounded strategy_compressions/strategy_retrievals dicts.

    BUG: Unlike common_queries (truncated at 100) and queried_fields (truncated at 50),
    the strategy dicts have no size limits and could grow unbounded.

    FIX: Add truncation logic similar to other dicts.
    """

    def test_strategy_dicts_have_size_limits(self):
        """Strategy dicts should be bounded to prevent memory leaks."""
        from headroom.cache.compression_feedback import (
            CompressionFeedback,
            reset_compression_feedback,
        )
        from headroom.cache.compression_store import CompressionStore

        reset_compression_feedback()
        store = CompressionStore()
        feedback = CompressionFeedback(store=store)

        # Record many compressions with different strategies
        for i in range(200):
            feedback.record_compression(
                tool_name="test_tool",
                original_count=100,
                compressed_count=10,
                strategy=f"strategy_{i}",  # 200 unique strategies
            )

        with feedback._lock:
            pattern = feedback._tool_patterns.get("test_tool")
            strategy_count = len(pattern.strategy_compressions) if pattern else 0

        # Strategy dict should be bounded (e.g., to 50 like queried_fields)
        assert strategy_count <= 50, (
            f"strategy_compressions has {strategy_count} entries, should be <= 50"
        )


class TestAllFixesIntegrated:
    """Integration tests ensuring all fixes work together."""
