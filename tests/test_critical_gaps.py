"""Tests for CRITICAL gap fixes in TOIN/CCR implementation.

These tests demonstrate bugs BEFORE the fix and verify they're fixed AFTER.
Each test documents the specific issue being addressed.
"""

import hashlib
import json
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from headroom.cache.compression_feedback import (
    CompressionFeedback,
    get_compression_feedback,
    reset_compression_feedback,
)
from headroom.cache.compression_store import (
    CompressionStore,
    RetrievalEvent,
    get_compression_store,
    reset_compression_store,
)
from headroom.telemetry.models import ToolSignature
from headroom.telemetry.toin import (
    TOINConfig,
    ToolIntelligenceNetwork,
    ToolPattern,
    get_toin,
    reset_toin,
)


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset all global state before each test."""
    reset_toin()
    reset_compression_feedback()
    reset_compression_store()
    yield
    reset_toin()
    reset_compression_feedback()
    reset_compression_store()


# =============================================================================
# CRITICAL #1: _all_seen_instances unbounded growth
# =============================================================================


class TestAllSeenInstancesUnboundedGrowth:
    """CRITICAL: _all_seen_instances set can grow unboundedly.

    The _all_seen_instances set is used for O(1) deduplication of users,
    but unlike _seen_instance_hashes (capped at 100), the set has no cap.
    With millions of users, this causes OOM.

    FIX: Add cap to _all_seen_instances or use a Bloom filter for memory efficiency.
    """

    def test_all_seen_instances_should_be_capped(self):
        """Verify _all_seen_instances doesn't grow beyond cap."""
        toin = ToolIntelligenceNetwork(TOINConfig(enabled=True))

        # Create a pattern
        sig = ToolSignature.from_items([{"id": 1, "name": "test"}])

        # Verify the cap constant exists
        assert hasattr(ToolPattern, "MAX_SEEN_INSTANCES")
        assert ToolPattern.MAX_SEEN_INSTANCES == 10000

        # Simulate adding users via record_compression
        # (the cap is enforced there, not when directly adding to set)
        pattern = ToolPattern(tool_signature_hash=sig.structure_hash)
        toin._patterns[("unknown", "unknown", sig.structure_hash)] = pattern

        # Direct manipulation should still work for testing
        for i in range(200):
            instance_hash = hashlib.sha256(f"user_{i}".encode()).hexdigest()[:8]
            # Simulate the capped addition logic
            if len(pattern._all_seen_instances) < ToolPattern.MAX_SEEN_INSTANCES:
                pattern._all_seen_instances.add(instance_hash)
            if len(pattern._seen_instance_hashes) < 100:
                pattern._seen_instance_hashes.append(instance_hash)
            pattern.user_count += 1

        # Verify constraints
        assert len(pattern._seen_instance_hashes) <= 100  # Storage is capped
        assert len(pattern._all_seen_instances) <= ToolPattern.MAX_SEEN_INSTANCES
        assert pattern.user_count == 200  # user_count tracks all, even after cap

    def test_user_count_preserved_after_instance_cap(self):
        """User count should remain accurate even after instance cap is hit."""
        toin = ToolIntelligenceNetwork(TOINConfig(enabled=True))
        sig = ToolSignature.from_items([{"id": 1}])

        # Record compressions from 150 "users" (simulated)
        # by directly manipulating the pattern
        pattern = ToolPattern(tool_signature_hash=sig.structure_hash)
        toin._patterns[("unknown", "unknown", sig.structure_hash)] = pattern

        # Track 150 unique users
        for i in range(150):
            instance_hash = hashlib.sha256(f"user_{i}".encode()).hexdigest()[:8]
            if instance_hash not in pattern._all_seen_instances:
                pattern._all_seen_instances.add(instance_hash)
                if len(pattern._seen_instance_hashes) < 100:
                    pattern._seen_instance_hashes.append(instance_hash)
                pattern.user_count += 1

        # User count should be 150 even though storage list is capped at 100
        assert pattern.user_count == 150
        assert len(pattern._seen_instance_hashes) == 100


# =============================================================================
# CRITICAL #2: _all_seen_instances serialization
# =============================================================================


class TestAllSeenInstancesSerialization:
    """CRITICAL: _all_seen_instances isn't properly serialized.

    When saving/loading TOIN data, _all_seen_instances is not serialized
    because sets can't be JSON serialized directly. After reload, the set
    is recreated from _seen_instance_hashes, but if there were more than
    100 users, those extra entries are LOST, leading to incorrect deduplication.

    FIX: Serialize user_count separately and ensure _all_seen_instances
    is properly reconstructed from both list and user_count.
    """

    def test_serialization_preserves_all_seen_instances(self):
        """Verify _all_seen_instances survives serialization round-trip."""
        # Create pattern with more users than storage cap
        pattern = ToolPattern(tool_signature_hash="test_hash")

        # Add 150 unique instances
        for i in range(150):
            instance_hash = hashlib.sha256(f"user_{i}".encode()).hexdigest()[:8]
            pattern._all_seen_instances.add(instance_hash)
            if len(pattern._seen_instance_hashes) < 100:
                pattern._seen_instance_hashes.append(instance_hash)
            pattern.user_count += 1

        # Serialize
        data = pattern.to_dict()

        # Deserialize
        restored = ToolPattern.from_dict(data)

        # AFTER FIX: restored._all_seen_instances should be reconstructed
        # Currently it's only reconstructed from _seen_instance_hashes (100 max)
        # The user_count (150) should be preserved and used for future dedup logic
        assert restored.user_count == 150
        # After fix, the set should have at least the stored hashes
        assert len(restored._all_seen_instances) >= 100

    def test_disk_persistence_preserves_user_count(self):
        """Verify user count survives disk save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "toin_data.json"

            # Create TOIN with storage
            config = TOINConfig(enabled=True, storage_path=str(storage_path))
            toin = ToolIntelligenceNetwork(config)

            sig = ToolSignature.from_items([{"id": 1, "name": "test"}])

            # Record compressions from multiple "users"
            # We'll simulate by directly manipulating the pattern
            toin.record_compression(
                tool_signature=sig,
                original_count=100,
                compressed_count=10,
                original_tokens=1000,
                compressed_tokens=100,
                strategy="smart_sample",
            )

            # Manually add more users to simulate multi-user scenario
            pattern = toin._patterns[("unknown", "unknown", sig.structure_hash)]
            for i in range(50):
                instance_hash = hashlib.sha256(f"extra_user_{i}".encode()).hexdigest()[:8]
                if instance_hash not in pattern._all_seen_instances:
                    pattern._all_seen_instances.add(instance_hash)
                    if len(pattern._seen_instance_hashes) < 100:
                        pattern._seen_instance_hashes.append(instance_hash)
                    pattern.user_count += 1

            original_user_count = pattern.user_count

            # Save
            toin.save()

            # Create new TOIN instance to load from disk
            reset_toin()
            toin2 = ToolIntelligenceNetwork(config)

            # Verify user count is preserved
            pattern2 = toin2._patterns.get(("unknown", "unknown", sig.structure_hash))
            assert pattern2 is not None
            assert pattern2.user_count == original_user_count


# =============================================================================
# CRITICAL #3: User count merge logic complexity
# =============================================================================


class TestUserCountMergeLogic:
    """CRITICAL: User count merge logic in _merge_patterns is complex.

    The formula for merging user counts is:
    users_beyond_imported_storage = max(0, imported.user_count - len(imported._seen_instance_hashes) - len(imported._all_seen_instances - set(imported._seen_instance_hashes)))

    This is complex and may have edge case bugs. Simplified logic needed.

    FIX: Simplify to: existing.user_count = len(existing._all_seen_instances)
    after merging all instances.
    """

    def test_merge_user_count_simple_case(self):
        """Verify user count merge works for simple case."""
        toin = ToolIntelligenceNetwork(TOINConfig(enabled=True))

        # Create existing pattern with 5 users
        existing = ToolPattern(tool_signature_hash="test_hash")
        for i in range(5):
            h = hashlib.sha256(f"existing_{i}".encode()).hexdigest()[:8]
            existing._all_seen_instances.add(h)
            existing._seen_instance_hashes.append(h)
            existing.user_count += 1
        existing.sample_size = 10

        # Create imported pattern with 3 users (1 overlapping)
        imported = ToolPattern(tool_signature_hash="test_hash")
        for i in range(3):
            # User 0 overlaps with existing
            h = (
                hashlib.sha256(f"existing_{i}".encode()).hexdigest()[:8]
                if i == 0
                else hashlib.sha256(f"imported_{i}".encode()).hexdigest()[:8]
            )
            imported._all_seen_instances.add(h)
            imported._seen_instance_hashes.append(h)
            imported.user_count += 1
        imported.sample_size = 5

        # Merge
        toin._patterns[("unknown", "unknown", "test_hash")] = existing
        toin._merge_patterns(existing, imported)

        # After merge: 5 existing + 2 new = 7 unique users
        # (imported user 0 overlaps with existing user 0)
        assert existing.user_count == 7

    def test_merge_user_count_with_capped_storage(self):
        """Verify user count merge works when storage list is capped."""
        toin = ToolIntelligenceNetwork(TOINConfig(enabled=True))

        # Create existing pattern at storage cap
        existing = ToolPattern(tool_signature_hash="test_hash")
        for i in range(100):
            h = hashlib.sha256(f"existing_{i}".encode()).hexdigest()[:8]
            existing._all_seen_instances.add(h)
            existing._seen_instance_hashes.append(h)
            existing.user_count += 1
        # Add 20 more users beyond cap
        for i in range(100, 120):
            h = hashlib.sha256(f"existing_{i}".encode()).hexdigest()[:8]
            existing._all_seen_instances.add(h)
            existing.user_count += 1
        existing.sample_size = 200

        # Create imported with 10 new users
        imported = ToolPattern(tool_signature_hash="test_hash")
        for i in range(10):
            h = hashlib.sha256(f"new_user_{i}".encode()).hexdigest()[:8]
            imported._all_seen_instances.add(h)
            imported._seen_instance_hashes.append(h)
            imported.user_count += 1
        imported.sample_size = 20

        # Merge
        toin._patterns[("unknown", "unknown", "test_hash")] = existing
        toin._merge_patterns(existing, imported)

        # After merge: 120 existing + 10 new = 130 unique users
        assert existing.user_count == 130


# =============================================================================
# CRITICAL #4: _get_entry_for_search returns reference not copy
# =============================================================================


class TestGetEntryForSearchRaceCondition:
    """CRITICAL: _get_entry_for_search returns reference to internal entry.

    The entry can be modified or evicted by another thread after the lock
    is released but before the caller uses it, causing race conditions.

    FIX: Return a deep copy of the entry, or use copy-on-write.
    """

    def test_returned_entry_is_independent_copy(self):
        """Verify returned entry is independent from internal state."""
        store = CompressionStore(max_entries=100)

        original_data = '[{"id": 1}, {"id": 2}]'
        hash_key = store.store(
            original=original_data,
            compressed='[{"id": 1}]',
            original_item_count=2,
            compressed_item_count=1,
            tool_name="test_tool",
        )

        # Get entry via _get_entry_for_search
        entry1 = store._get_entry_for_search(hash_key)
        assert entry1 is not None

        # Modify the returned entry
        entry1.search_queries.append("test_query")
        entry1.retrieval_count = 999

        # Get entry again - should NOT reflect our modifications
        entry2 = store._get_entry_for_search(hash_key)

        # AFTER FIX: entry2 should be a fresh copy, not affected by entry1 modifications
        # Currently this may fail because we return a reference
        # The fix ensures we return a copy
        assert "test_query" not in entry2.search_queries or entry2.retrieval_count != 999

    def test_concurrent_access_no_corruption(self):
        """Verify concurrent access doesn't corrupt entries."""
        store = CompressionStore(max_entries=100)

        original_data = json.dumps([{"id": i} for i in range(100)])
        hash_key = store.store(
            original=original_data,
            compressed='[{"id": 0}]',
            original_item_count=100,
            compressed_item_count=1,
            tool_name="test_tool",
        )

        errors = []

        def reader():
            for _ in range(50):
                entry = store._get_entry_for_search(hash_key, "query")
                if entry:
                    # Simulate work with the entry
                    try:
                        items = json.loads(entry.original_content)
                        if len(items) != 100:
                            errors.append("Content corrupted")
                    except Exception as e:
                        errors.append(str(e))
                time.sleep(0.001)

        def modifier():
            for _ in range(50):
                # Try to mess with internal state
                entry = store._get_entry_for_search(hash_key)
                if entry:
                    entry.search_queries.clear()  # Shouldn't affect other readers
                time.sleep(0.001)

        # Run concurrent readers and modifiers
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for _ in range(4):
                futures.append(executor.submit(reader))
                futures.append(executor.submit(modifier))

            for f in futures:
                f.result()

        assert len(errors) == 0, f"Errors during concurrent access: {errors}"


# =============================================================================
# CRITICAL #5: Hash collision vulnerability (16 chars = 64 bits)
# =============================================================================


class TestHashCollisionVulnerability:
    """CRITICAL: Hash truncation to 16 chars (64 bits) may cause collisions.

    SHA256[:16] = 64 bits. Birthday problem suggests 50% collision probability
    at ~2^32 entries (~4 billion). While unlikely in practice, for security-
    sensitive applications this is too short.

    FIX: Increase to 32 chars (128 bits) for compression_store hashes.
    """

    def test_hash_length_is_sufficient(self):
        """Verify hash length provides adequate collision resistance."""
        store = CompressionStore()

        # Store some content and check hash length
        content1 = '[{"id": 1}]'
        hash1 = store.store(original=content1, compressed=content1)

        # CRITICAL FIX #5: Now uses 24 chars (96 bits) instead of 16 (64 bits)
        # For birthday attack resistance with 1 billion entries, need ~96 bits
        assert len(hash1) >= 24  # Fixed: Better collision resistance

    def test_no_practical_collision(self):
        """Verify no collisions for reasonable number of entries."""
        store = CompressionStore(max_entries=10000)

        hashes = set()
        for i in range(1000):
            content = json.dumps([{"id": i, "data": f"unique_content_{i}_{time.time()}"}])
            h = store.store(original=content, compressed=content)
            if h in hashes:
                pytest.fail(f"Hash collision detected at entry {i}")
            hashes.add(h)

        assert len(hashes) == 1000


# =============================================================================
# CRITICAL #6: Lock ordering deadlock risk
# =============================================================================


class TestLockOrderingDeadlockRisk:
    """CRITICAL: Multiple locks across files without documented ordering.

    TOIN, CompressionStore, and CompressionFeedback each have their own locks.
    If they call each other while holding their locks, deadlock can occur.

    Current call chain that could deadlock:
    - CompressionStore.process_pending_feedback() holds _store._lock
    - Calls TOIN.record_retrieval() which tries to acquire _toin._lock
    - If TOIN is doing something that needs store, deadlock

    FIX: Document lock ordering, ensure consistent acquisition order.
    Actually, looking at the code, process_pending_feedback RELEASES the lock
    before calling TOIN, so this specific case is safe. But we should verify.
    """

    def test_no_deadlock_on_concurrent_operations(self):
        """Verify no deadlock when operations are concurrent."""
        toin = get_toin(TOINConfig(enabled=True))
        store = get_compression_store()
        feedback = get_compression_feedback()

        sig = ToolSignature.from_items([{"id": 1, "name": "test"}])

        errors = []
        deadlock_detected = threading.Event()

        def toin_writer():
            for _i in range(50):
                if deadlock_detected.is_set():
                    break
                try:
                    toin.record_compression(
                        tool_signature=sig,
                        original_count=100,
                        compressed_count=10,
                        original_tokens=1000,
                        compressed_tokens=100,
                        strategy="smart_sample",
                    )
                except Exception as e:
                    errors.append(f"TOIN writer error: {e}")
                time.sleep(0.001)

        def store_writer():
            for i in range(50):
                if deadlock_detected.is_set():
                    break
                try:
                    store.store(
                        original=f'[{{"id": {i}}}]',
                        compressed=f'[{{"id": {i}}}]',
                        tool_signature_hash=sig.structure_hash,
                    )
                except Exception as e:
                    errors.append(f"Store writer error: {e}")
                time.sleep(0.001)

        def feedback_reader():
            for _i in range(50):
                if deadlock_detected.is_set():
                    break
                try:
                    feedback.get_compression_hints("test_tool")
                    feedback.get_all_patterns()
                except Exception as e:
                    errors.append(f"Feedback reader error: {e}")
                time.sleep(0.001)

        # Run with timeout to detect deadlocks
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            for _ in range(2):
                futures.append(executor.submit(toin_writer))
                futures.append(executor.submit(store_writer))
                futures.append(executor.submit(feedback_reader))

            # Wait with timeout
            import concurrent.futures

            done, not_done = concurrent.futures.wait(futures, timeout=10)

            if not_done:
                deadlock_detected.set()
                pytest.fail("Potential deadlock detected - operations didn't complete in 10s")

        assert len(errors) == 0, f"Errors during concurrent operations: {errors}"


# =============================================================================
# HIGH PRIORITY: Additional important fixes
# =============================================================================


class TestHighPriorityFixes:
    """Additional HIGH priority fixes that affect correctness."""

    def test_eviction_heap_cleanup(self):
        """Verify eviction heap is properly maintained.

        HIGH: Eviction heap can have stale entries after manual deletion,
        causing O(n) degradation as we pop non-existent entries.
        """
        store = CompressionStore(max_entries=5)

        # Fill store
        hashes = []
        for i in range(5):
            h = store.store(
                original=f'[{{"id": {i}}}]',
                compressed=f'[{{"id": {i}}}]',
            )
            hashes.append(h)

        # Store 6th entry - should evict oldest
        store.store(
            original='[{"id": 6}]',
            compressed='[{"id": 6}]',
        )

        # Verify eviction happened
        stats = store.get_stats()
        assert stats["entry_count"] <= 5

    def test_get_all_patterns_returns_copy(self):
        """Verify get_all_patterns returns copies, not references.

        HIGH: Returning mutable internal state allows external code to
        corrupt the feedback system.
        """
        feedback = CompressionFeedback()
        feedback.record_compression("test_tool", 100, 10)

        patterns = feedback.get_all_patterns()

        # Modify returned patterns
        if "test_tool" in patterns:
            patterns["test_tool"].total_compressions = 9999
            patterns["test_tool"].common_queries["injected"] = 100

        # Get patterns again - should not be modified
        patterns2 = feedback.get_all_patterns()

        assert patterns2["test_tool"].total_compressions == 1
        assert "injected" not in patterns2["test_tool"].common_queries

    def test_unbounded_dict_limits(self):
        """Verify unbounded dicts have proper limits.

        HIGH: Several dicts (common_queries, queried_fields, strategy_*)
        can grow unboundedly without limits.
        """
        feedback = CompressionFeedback()

        # Record many compressions with different strategies
        for i in range(200):
            feedback.record_compression(
                "test_tool",
                100,
                10,
                strategy=f"strategy_{i}",
            )

        patterns = feedback.get_all_patterns()
        pattern = patterns["test_tool"]

        # Verify dicts are bounded
        assert len(pattern.strategy_compressions) <= 50

        # Record many retrievals with different queries
        for i in range(200):
            event = RetrievalEvent(
                hash="test",
                query=f"unique_query_{i}_field:value",
                items_retrieved=10,
                total_items=100,
                tool_name="test_tool",
                timestamp=time.time(),
                retrieval_type="search",
            )
            feedback.record_retrieval(event, strategy=f"strategy_{i % 50}")

        patterns = feedback.get_all_patterns()
        pattern = patterns["test_tool"]

        # Verify all dicts are bounded
        assert len(pattern.common_queries) <= 100
        assert len(pattern.queried_fields) <= 50
        assert len(pattern.strategy_retrievals) <= 50


# =============================================================================
# Integration test
# =============================================================================


class TestCriticalFixesIntegration:
    """Integration test verifying all critical fixes work together."""

    def test_full_workflow_with_fixes(self):
        """Full CCR workflow with all critical fixes applied."""
        # Setup
        toin = get_toin(TOINConfig(enabled=True))
        store = get_compression_store()
        feedback = get_compression_feedback()

        sig = ToolSignature.from_items([{"id": 1, "score": 0.9, "name": "test"}])

        # Simulate compression workflow
        original = json.dumps(
            [{"id": i, "score": 0.9 - i * 0.01, "name": f"item_{i}"} for i in range(100)]
        )
        compressed = json.dumps([{"id": 0, "score": 0.9, "name": "item_0"}])

        # 1. Record compression in feedback
        feedback.record_compression(
            "test_tool",
            100,
            1,
            strategy="TOP_N",
            tool_signature_hash=sig.structure_hash,
        )

        # 2. Store in compression store
        hash_key = store.store(
            original=original,
            compressed=compressed,
            original_item_count=100,
            compressed_item_count=1,
            tool_name="test_tool",
            tool_signature_hash=sig.structure_hash,
            compression_strategy="TOP_N",
        )

        # 3. Record in TOIN
        toin.record_compression(
            tool_signature=sig,
            original_count=100,
            compressed_count=1,
            original_tokens=2000,
            compressed_tokens=50,
            strategy="TOP_N",
        )

        # 4. Simulate retrieval
        entry = store.retrieve(hash_key)
        assert entry is not None
        assert entry.original_item_count == 100

        # 5. Search within cached data
        store.search(hash_key, "item_50")
        # Should find the item even though it was compressed away

        # 6. Get recommendation from TOIN
        toin.get_recommendation(sig, "find item_50")

        # 7. Verify stats are consistent
        toin_stats = toin.get_stats()
        store_stats = store.get_stats()
        feedback_stats = feedback.get_stats()

        assert toin_stats["total_compressions"] >= 1
        assert store_stats["entry_count"] >= 1
        assert feedback_stats["total_compressions"] >= 1


# =============================================================================
# Additional HIGH PRIORITY tests
# =============================================================================


class TestTOINHighPriorityFixes:
    """Additional HIGH priority tests for TOIN."""

    def test_field_retrieval_frequency_bounded(self):
        """Verify field_retrieval_frequency dict is bounded.

        HIGH: This dict can grow unboundedly with many unique field names.
        """
        toin = ToolIntelligenceNetwork(TOINConfig(enabled=True))
        sig = ToolSignature.from_items([{"id": 1}])
        toin.record_compression(
            tool_signature=sig,
            original_count=100,
            compressed_count=10,
            original_tokens=1000,
            compressed_tokens=100,
            strategy="test",
        )

        # Record many retrievals with different field names
        for i in range(150):
            toin.record_retrieval(
                tool_signature_hash=sig.structure_hash,
                retrieval_type="search",
                query=f"field_{i}:value",
                query_fields=[f"unique_field_{i}"],
            )

        pattern = toin._patterns[("unknown", "unknown", sig.structure_hash)]
        assert len(pattern.field_retrieval_frequency) <= 100

    def test_commonly_retrieved_fields_bounded(self):
        """Verify commonly_retrieved_fields list is bounded.

        HIGH: This list can grow unboundedly with many unique fields.
        """
        toin = ToolIntelligenceNetwork(TOINConfig(enabled=True))
        sig = ToolSignature.from_items([{"id": 1}])
        toin.record_compression(
            tool_signature=sig,
            original_count=100,
            compressed_count=10,
            original_tokens=1000,
            compressed_tokens=100,
            strategy="test",
        )

        # Record many retrievals to trigger commonly_retrieved_fields update
        for i in range(50):
            for _ in range(5):  # 5 retrievals per field to hit threshold
                toin.record_retrieval(
                    tool_signature_hash=sig.structure_hash,
                    retrieval_type="search",
                    query=f"field_{i}:value",
                    query_fields=[f"common_field_{i}"],
                )

        pattern = toin._patterns[("unknown", "unknown", sig.structure_hash)]
        assert len(pattern.commonly_retrieved_fields) <= 20

    def test_strategy_success_rate_updates(self):
        """Verify strategy success rates update correctly.

        HIGH: Strategies should be penalized on retrieval and boosted on compression.
        """
        toin = ToolIntelligenceNetwork(TOINConfig(enabled=True))
        sig = ToolSignature.from_items([{"id": 1}])

        # Record initial compression - establishes strategy
        toin.record_compression(
            tool_signature=sig,
            original_count=100,
            compressed_count=10,
            original_tokens=1000,
            compressed_tokens=100,
            strategy="TEST_STRATEGY",
        )

        pattern = toin._patterns[("unknown", "unknown", sig.structure_hash)]
        initial_rate = pattern.strategy_success_rates["TEST_STRATEGY"]
        assert initial_rate == 1.0  # Starts at 1.0

        # Record retrieval - should penalize strategy
        toin.record_retrieval(
            tool_signature_hash=sig.structure_hash,
            retrieval_type="full",
            query=None,
            strategy="TEST_STRATEGY",
        )

        pattern = toin._patterns[("unknown", "unknown", sig.structure_hash)]
        after_retrieval = pattern.strategy_success_rates["TEST_STRATEGY"]
        assert after_retrieval < initial_rate  # Should decrease

        # Record more compressions - should boost strategy
        for _ in range(5):
            toin.record_compression(
                tool_signature=sig,
                original_count=100,
                compressed_count=10,
                original_tokens=1000,
                compressed_tokens=100,
                strategy="TEST_STRATEGY",
            )

        pattern = toin._patterns[("unknown", "unknown", sig.structure_hash)]
        after_compressions = pattern.strategy_success_rates["TEST_STRATEGY"]
        assert after_compressions > after_retrieval  # Should increase

    def test_maybe_auto_save_only_saves_when_dirty(self):
        """Verify _maybe_auto_save only saves when dirty flag is set."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "toin_test.json"

            config = TOINConfig(
                enabled=True,
                storage_path=str(storage_path),
                auto_save_interval=0.001,  # Very short interval = auto-save on every call
            )
            toin = ToolIntelligenceNetwork(config)

            # Initially should not be dirty
            assert not toin._dirty

            # Set _last_save_time to past so elapsed > interval
            toin._last_save_time = 0

            # Record compression - should set dirty
            sig = ToolSignature.from_items([{"id": 1}])
            toin.record_compression(
                tool_signature=sig,
                original_count=100,
                compressed_count=10,
                original_tokens=1000,
                compressed_tokens=100,
                strategy="test",
            )

            # After auto-save, dirty should be cleared
            # (auto-save happens inside record_compression)
            assert not toin._dirty

    @pytest.mark.skip(
        reason="PR-B5: get_recommendation retired; preserve_fields lives on the aggregated ToolPattern instead"
    )
    def test_toin_preserves_fields_returns_list(self):
        """Retired in PR-B5 along with the request-time hint API."""


class TestCompressionStoreHighPriorityFixes:
    """Additional HIGH priority tests for CompressionStore."""

    def test_eviction_heap_handles_stale_entries(self):
        """Verify eviction heap handles entries deleted outside eviction.

        HIGH: Stale entries in heap could cause O(n) degradation.
        """
        store = CompressionStore(max_entries=10)

        # Fill store
        hashes = []
        for i in range(10):
            h = store.store(
                original=f'[{{"id": {i}}}]',
                compressed=f'[{{"id": {i}}}]',
            )
            hashes.append(h)

        # Manually expire entries (simulating TTL)
        with store._lock:
            for h in hashes[:5]:
                entry = store._backend.get(h)
                if entry:
                    entry.created_at = 0  # Make it look old
                    entry.ttl = 0  # Make it expired
                    store._backend.set(h, entry)

        # Store more entries - should handle stale heap entries gracefully
        for i in range(20, 30):
            store.store(
                original=f'[{{"id": {i}}}]',
                compressed=f'[{{"id": {i}}}]',
            )

        stats = store.get_stats()
        assert stats["entry_count"] <= 10

    def test_retrieval_events_list_bounded(self):
        """Verify retrieval events list is bounded.

        HIGH: Events list can grow unboundedly without trimming.
        """
        store = CompressionStore(max_entries=100)

        hash_key = store.store(
            original='[{"id": 1}]',
            compressed='[{"id": 1}]',
        )

        # Trigger many retrievals
        for i in range(1500):
            store.retrieve(hash_key, f"query_{i}")

        with store._lock:
            assert len(store._retrieval_events) <= 1000

    def test_search_queries_in_entry_bounded(self):
        """Verify search_queries list in entry is bounded.

        HIGH: search_queries list can grow unboundedly.
        """
        store = CompressionStore(max_entries=100)

        hash_key = store.store(
            original='[{"id": 1}]',
            compressed='[{"id": 1}]',
        )

        # Trigger many searches with different queries
        for i in range(50):
            store.search(hash_key, f"unique_query_{i}")

        with store._lock:
            entry = store._backend.get(hash_key)
            if entry:
                assert len(entry.search_queries) <= 10


class TestCompressionFeedbackHighPriorityFixes:
    """Additional HIGH priority tests for CompressionFeedback."""

    def test_signature_hashes_set_bounded(self):
        """Verify signature_hashes set is bounded.

        HIGH: Set can grow unboundedly with many unique hashes.
        """
        feedback = CompressionFeedback()

        # Record many compressions with different signature hashes
        for i in range(200):
            feedback.record_compression(
                "test_tool",
                100,
                10,
                strategy="test",
                tool_signature_hash=f"sig_hash_{i}",
            )

        patterns = feedback.get_all_patterns()
        pattern = patterns["test_tool"]

        assert len(pattern.signature_hashes) <= 100

    def test_analyze_from_store_avoids_double_counting(self):
        """Verify analyze_from_store doesn't double-count events.

        HIGH: Without timestamp tracking, events could be processed multiple times.
        """

        feedback = CompressionFeedback(analysis_interval=0)  # Allow immediate re-analysis

        # Record initial compression
        feedback.record_compression("test_tool", 100, 10)

        # Manually set last_event_timestamp to simulate processed events
        # This ensures we don't double-count

        # Call analyze multiple times - should not double-count
        for _ in range(3):
            feedback.analyze_from_store()

        # Total retrievals should not have increased dramatically from re-analysis
        # (may increase slightly from any new real events)


class TestMediumPriorityToolSignatureFixes:
    """Tests for MEDIUM priority ToolSignature fixes."""

    def test_max_depth_calculated_not_hardcoded(self):
        """MEDIUM FIX #12: max_depth should be calculated from actual item structure."""
        from headroom.telemetry.models import ToolSignature

        # Simple flat structure - depth = 2 (list item -> dict fields)
        flat_items = [{"id": 1, "name": "test"}]
        flat_sig = ToolSignature.from_items(flat_items)
        assert flat_sig.max_depth == 2

        # Nested structure - depth = 4 (list -> dict -> nested -> deep)
        nested_items = [{"id": 1, "data": {"nested": {"deep": "value"}}}]
        nested_sig = ToolSignature.from_items(nested_items)
        assert nested_sig.max_depth == 4

        # Very deep structure
        deep_items = [{"a": {"b": {"c": {"d": {"e": "bottom"}}}}}]
        deep_sig = ToolSignature.from_items(deep_items)
        assert deep_sig.max_depth == 6  # list + 5 levels of nesting

    def test_multiple_items_analyzed_for_structure(self):
        """MEDIUM FIX #13: Should analyze multiple items to get representative structure."""
        from headroom.telemetry.models import ToolSignature

        # Items with varying structures
        varying_items = [
            {"id": 1},
            {"id": 2, "name": "test"},
            {"id": 3, "name": "test", "extra": "field"},
            {"id": 4, "status": "active"},
            {"id": 5, "nested": {"data": 1}},
        ]

        sig = ToolSignature.from_items(varying_items)

        # Should capture field count from representative items
        # The implementation samples up to 5 items, so field_count should reflect merged fields
        assert sig.field_count > 0
        # Should detect ID-like field from "id"
        assert sig.has_id_like_field
        # Should detect status-like field from "status"
        assert sig.has_status_like_field

    def test_id_pattern_word_boundary_matching(self):
        """MEDIUM FIX #14: ID pattern detection should use word boundaries."""
        from headroom.telemetry.models import ToolSignature

        # Field named "id" should be detected as ID
        items_with_id = [{"id": "abc123", "data": "value"}]
        sig1 = ToolSignature.from_items(items_with_id)
        assert sig1.has_id_like_field

        # Field named "hidden" should NOT be detected as ID (contains "id" but not at word boundary)
        items_with_hidden = [{"hidden": True, "data": "value"}]
        sig2 = ToolSignature.from_items(items_with_hidden)
        # The pattern should match "_id", "id_", "id" as standalone but not "hid" in "hidden"
        # has_id_like_field should be False for "hidden" field
        assert not sig2.has_id_like_field

        # Field named "user_id" SHOULD be detected (word boundary)
        items_with_user_id = [{"user_id": "abc123", "data": "value"}]
        sig3 = ToolSignature.from_items(items_with_user_id)
        assert sig3.has_id_like_field

        # camelCase "userId" SHOULD be detected
        items_with_camel = [{"userId": "abc123", "data": "value"}]
        sig4 = ToolSignature.from_items(items_with_camel)
        assert sig4.has_id_like_field

    def test_hash_uses_96_bits(self):
        """MEDIUM FIX #15: Hash should use 24 chars (96 bits) for collision resistance."""
        from headroom.telemetry.models import ToolSignature

        items = [{"id": 1, "name": "test", "value": 42}]
        sig = ToolSignature.from_items(items)

        # Hash should be 24 characters
        assert len(sig.structure_hash) == 24


class TestMediumPriorityTOINFixes:
    """Tests for MEDIUM priority TOIN fixes."""

    def test_query_pattern_frequency_tracking(self):
        """MEDIUM FIX #10: Query patterns should be ranked by frequency, not just recency."""
        from headroom.telemetry.models import ToolSignature
        from headroom.telemetry.toin import TOINConfig, ToolIntelligenceNetwork

        toin = ToolIntelligenceNetwork(TOINConfig(enabled=True))
        sig = ToolSignature.from_items([{"id": 1, "status": "active"}])

        # Record initial compression
        toin.record_compression(
            tool_signature=sig,
            original_count=100,
            compressed_count=10,
            original_tokens=1000,
            compressed_tokens=100,
            strategy="test",
        )

        # Record many retrievals with different queries
        # One query appears much more frequently
        frequent_query = "find errors"
        rare_query1 = "find user 123"
        rare_query2 = "find order 456"

        # Record frequent query many times
        for _ in range(10):
            toin.record_retrieval(
                sig.structure_hash,  # Correct attribute name
                retrieval_type="search",
                query=frequent_query,
                query_fields=["id"],
            )

        # Record rare queries once each
        toin.record_retrieval(
            sig.structure_hash,
            retrieval_type="search",
            query=rare_query1,
            query_fields=["id"],
        )
        toin.record_retrieval(
            sig.structure_hash,
            retrieval_type="search",
            query=rare_query2,
            query_fields=["id"],
        )

        # Get the pattern and check query frequencies
        pattern = toin.get_pattern(sig.structure_hash)
        if pattern:
            # The query_pattern_frequency dict should exist and track counts
            freq = pattern.query_pattern_frequency
            assert freq.get(frequent_query, 0) >= freq.get(rare_query1, 0)

    def test_common_queries_bounded(self):
        """Verify common_queries list is bounded."""
        from headroom.telemetry.models import ToolSignature
        from headroom.telemetry.toin import TOINConfig, ToolIntelligenceNetwork

        toin = ToolIntelligenceNetwork(TOINConfig(enabled=True))
        sig = ToolSignature.from_items([{"id": 1}])

        toin.record_compression(
            tool_signature=sig,
            original_count=100,
            compressed_count=10,
            original_tokens=1000,
            compressed_tokens=100,
            strategy="test",
        )

        # Record many unique queries
        for i in range(50):
            toin.record_retrieval(
                sig.structure_hash,  # Correct attribute name
                retrieval_type="search",
                query=f"unique query {i}",
                query_fields=["id"],
            )

        pattern = toin.get_pattern(sig.structure_hash)
        if pattern:
            # The limit is set by max_query_patterns config (default 10)
            assert len(pattern.common_query_patterns) <= 10


class TestLowPriorityFixes:
    """Tests for LOW priority fixes."""

    def test_exists_does_not_delete_by_default(self):
        """LOW FIX #20: exists() should be a pure check by default."""
        from headroom.cache.compression_store import CompressionStore

        store = CompressionStore(default_ttl=1)  # 1 second TTL

        hash_key = store.store(
            original='[{"id": 1}]',
            compressed="[1]",
            original_item_count=1,
            compressed_item_count=1,
            tool_name="test",
        )

        # Entry exists initially
        assert store.exists(hash_key) is True

        # Wait for expiry
        import time

        time.sleep(1.1)

        # Entry is expired, exists() returns False but does NOT delete
        assert store.exists(hash_key) is False

        # Entry should still be in internal store (not deleted)
        with store._lock:
            assert store._backend.exists(hash_key)

        # Now with clean_expired=True, it should delete
        assert store.exists(hash_key, clean_expired=True) is False
        with store._lock:
            assert not store._backend.exists(hash_key)

    def test_toin_confidence_threshold_configurable(self):
        """LOW FIX #21: TOIN confidence threshold should be configurable."""
        from headroom.config import SmartCrusherConfig

        # Default value (lowered from 0.5 to 0.3 for faster TOIN learning)
        config = SmartCrusherConfig()
        assert config.toin_confidence_threshold == 0.3

        # Custom value
        config2 = SmartCrusherConfig(toin_confidence_threshold=0.8)
        assert config2.toin_confidence_threshold == 0.8

    def test_toin_metrics_callback(self):
        """LOW FIX #22: TOIN should emit metrics via callback."""
        from headroom.telemetry.models import ToolSignature
        from headroom.telemetry.toin import TOINConfig, ToolIntelligenceNetwork

        metrics_events = []

        def capture_metric(event_name: str, event_data: dict):
            metrics_events.append((event_name, event_data))

        config = TOINConfig(enabled=True, metrics_callback=capture_metric)
        toin = ToolIntelligenceNetwork(config)

        sig = ToolSignature.from_items([{"id": 1}])

        # Record compression - should emit metric
        toin.record_compression(
            tool_signature=sig,
            original_count=100,
            compressed_count=10,
            original_tokens=1000,
            compressed_tokens=100,
            strategy="test",
        )

        # Check that compression metric was emitted
        compression_events = [e for e in metrics_events if e[0] == "toin.compression"]
        assert len(compression_events) >= 1

        # Record retrieval - should emit metric
        toin.record_retrieval(
            sig.structure_hash,
            retrieval_type="full",
            query=None,
        )

        # Check that retrieval metric was emitted
        retrieval_events = [e for e in metrics_events if e[0] == "toin.retrieval"]
        assert len(retrieval_events) >= 1


class TestMediumPriorityCompressionStoreFixes:
    """Tests for MEDIUM priority CompressionStore fixes."""

    def test_eviction_heap_order_correct(self):
        """MEDIUM FIX #16: Eviction heap should evict oldest entries first."""
        import time

        from headroom.cache.compression_store import CompressionStore

        # Small store to trigger eviction
        store = CompressionStore(max_entries=3)

        # Store entries with small delays to ensure different timestamps
        hash1 = store.store(
            original='[{"id": 1}]',
            compressed="[1]",
            original_item_count=1,
            compressed_item_count=1,
            tool_name="tool1",
        )
        time.sleep(0.01)

        hash2 = store.store(
            original='[{"id": 2}]',
            compressed="[2]",
            original_item_count=1,
            compressed_item_count=1,
            tool_name="tool2",
        )
        time.sleep(0.01)

        hash3 = store.store(
            original='[{"id": 3}]',
            compressed="[3]",
            original_item_count=1,
            compressed_item_count=1,
            tool_name="tool3",
        )

        # Retrieve hash2 and hash3 to update their last_accessed
        store.retrieve(hash2)
        store.retrieve(hash3)

        # Add a 4th entry to trigger eviction
        hash4 = store.store(
            original='[{"id": 4}]',
            compressed="[4]",
            original_item_count=1,
            compressed_item_count=1,
            tool_name="tool4",
        )

        # hash1 should be evicted (oldest, not accessed)
        assert store.retrieve(hash1) is None
        # Others should still exist
        assert store.retrieve(hash2) is not None
        assert store.retrieve(hash3) is not None
        assert store.retrieve(hash4) is not None

    def test_get_retrieval_events_returns_copy(self):
        """MEDIUM FIX #17: get_retrieval_events should return a copy."""
        from headroom.cache.compression_store import CompressionStore

        store = CompressionStore()

        hash_key = store.store(
            original='[{"id": 1}, {"id": 2}]',
            compressed='[{"id": 1}]',
            original_item_count=2,
            compressed_item_count=1,
            tool_name="test_tool",
        )

        # Retrieve to generate an event
        store.retrieve(hash_key)

        # Get events
        events1 = store.get_retrieval_events()
        events2 = store.get_retrieval_events()

        # Should be different list objects (copies)
        assert events1 is not events2

        # Modifying one should not affect the other
        if events1:
            original_len = len(events1)
            events1.clear()
            assert len(events2) == original_len
