"""Integration tests for memory tracking with real stores.

These tests verify that memory tracking works correctly with actual
store implementations - no mocks, no simulations.
"""

from __future__ import annotations

import pytest

from headroom.memory.tracker import MemoryTracker

# Check HNSW availability for skipping tests
try:
    from headroom.memory.adapters.hnsw import _check_hnswlib_available

    HNSW_AVAILABLE = _check_hnswlib_available()
except ImportError:
    HNSW_AVAILABLE = False


class TestCompressionStoreMemoryTracking:
    """Tests for CompressionStore memory tracking integration."""

    @pytest.fixture(autouse=True)
    def reset_tracker(self):
        """Reset the tracker singleton before each test."""
        MemoryTracker.reset()
        yield
        MemoryTracker.reset()

    def test_compression_store_reports_memory_stats(self):
        """Test that CompressionStore correctly reports memory stats."""
        from headroom.cache.compression_store import CompressionStore

        store = CompressionStore(max_entries=100)

        # Add some data - store(original, compressed, ...)
        store.store("original content 1" * 100, "compressed1")
        store.store("original content 2" * 100, "compressed2")

        stats = store.get_memory_stats()

        assert stats.name == "compression_store"
        assert stats.entry_count == 2
        assert stats.size_bytes > 0
        # budget_bytes is None since CompressionStore uses entry count limit not byte limit

    def test_compression_store_tracks_hits(self):
        """Test that CompressionStore tracks cache hits."""
        from headroom.cache.compression_store import CompressionStore

        store = CompressionStore(max_entries=100)

        # Store and retrieve (hit)
        hash_key = store.store("original content", "compressed")
        store.retrieve(hash_key)  # Hit - increments retrieval_count
        store.retrieve(hash_key)  # Another retrieval
        store.retrieve("nonexistent_hash")  # Miss (not tracked)

        stats = store.get_memory_stats()

        # Hits counts entries with retrieval_count > 0, not total retrievals
        assert stats.hits == 1  # 1 entry has been retrieved
        # CompressionStore doesn't track misses
        assert stats.misses == 0

    def test_compression_store_registers_with_tracker(self):
        """Test that CompressionStore can register with MemoryTracker."""
        from headroom.cache.compression_store import CompressionStore

        tracker = MemoryTracker.get()
        store = CompressionStore(max_entries=100)

        # Register the store
        tracker.register("compression_store", store.get_memory_stats)

        # Verify it's registered
        assert "compression_store" in tracker.registered_components

        # Get stats through tracker
        stats = tracker.get_component_stats("compression_store")
        assert stats is not None
        assert stats.name == "compression_store"


class TestBatchContextStoreMemoryTracking:
    """Tests for BatchContextStore memory tracking integration."""

    @pytest.fixture(autouse=True)
    def reset_tracker(self):
        """Reset the tracker singleton before each test."""
        MemoryTracker.reset()
        yield
        MemoryTracker.reset()

    def test_batch_context_store_reports_memory_stats(self):
        """Test that BatchContextStore correctly reports memory stats."""
        from headroom.ccr.batch_store import (
            BatchContext,
            BatchContextStore,
            BatchRequestContext,
        )

        store = BatchContextStore(ttl=3600, max_contexts=100)

        # Add some batch contexts
        ctx1 = BatchContext(batch_id="batch_1", provider="anthropic")
        ctx1.add_request(
            BatchRequestContext(
                custom_id="req_1",
                messages=[{"role": "user", "content": "Hello world"}],
                model="claude-3-opus",
            )
        )

        ctx2 = BatchContext(batch_id="batch_2", provider="openai")
        ctx2.add_request(
            BatchRequestContext(
                custom_id="req_2",
                messages=[{"role": "user", "content": "Test message"}],
                model="gpt-4",
            )
        )

        # Store them (sync for testing - accessing internal dict)
        store._contexts["batch_1"] = ctx1
        store._contexts["batch_2"] = ctx2

        stats = store.get_memory_stats()

        assert stats.name == "batch_context_store"
        assert stats.entry_count == 2
        assert stats.size_bytes > 0

    def test_batch_context_store_registers_with_tracker(self):
        """Test that BatchContextStore can register with MemoryTracker."""
        from headroom.ccr.batch_store import BatchContextStore

        tracker = MemoryTracker.get()
        store = BatchContextStore()

        # Register the store
        tracker.register("batch_context_store", store.get_memory_stats)

        # Verify it's registered
        assert "batch_context_store" in tracker.registered_components


class TestGraphStoreMemoryTracking:
    """Tests for InMemoryGraphStore memory tracking integration."""

    @pytest.fixture(autouse=True)
    def reset_tracker(self):
        """Reset the tracker singleton before each test."""
        MemoryTracker.reset()
        yield
        MemoryTracker.reset()

    @pytest.mark.asyncio
    async def test_graph_store_reports_memory_stats(self):
        """Test that InMemoryGraphStore correctly reports memory stats."""
        from headroom.memory.adapters.graph import InMemoryGraphStore
        from headroom.memory.adapters.graph_models import Entity, Relationship

        store = InMemoryGraphStore()

        # Add some entities using the correct API
        entity1 = Entity(id="node1", user_id="test", name="Test Entity 1", entity_type="entity")
        entity2 = Entity(id="node2", user_id="test", name="Test Entity 2", entity_type="entity")
        entity3 = Entity(id="node3", user_id="test", name="Test Concept", entity_type="concept")

        await store.add_entity(entity1)
        await store.add_entity(entity2)
        await store.add_entity(entity3)

        # Add a relationship
        rel = Relationship(
            source_id="node1",
            target_id="node2",
            relation_type="related_to",
            user_id="test",
        )
        await store.add_relationship(rel)

        stats = store.get_memory_stats()

        assert stats.name == "graph_store"
        assert stats.entry_count == 4  # 3 entities + 1 relationship
        assert stats.size_bytes > 0

    @pytest.mark.asyncio
    async def test_graph_store_size_grows_with_data(self):
        """Test that reported size grows as data is added."""
        from headroom.memory.adapters.graph import InMemoryGraphStore
        from headroom.memory.adapters.graph_models import Entity

        store = InMemoryGraphStore()

        # Get initial size
        initial_stats = store.get_memory_stats()
        initial_size = initial_stats.size_bytes

        # Add data
        for i in range(100):
            entity = Entity(
                id=f"node_{i}",
                user_id="test",
                name=f"Entity {i}",
                entity_type="entity",
                properties={"data": "x" * 100},
            )
            await store.add_entity(entity)

        # Get new size
        final_stats = store.get_memory_stats()

        assert final_stats.size_bytes > initial_size
        assert final_stats.entry_count == 100


@pytest.mark.skipif(not HNSW_AVAILABLE, reason="hnswlib not available")
class TestHNSWVectorIndexMemoryTracking:
    """Tests for HNSWVectorIndex memory tracking integration."""

    @pytest.fixture(autouse=True)
    def reset_tracker(self):
        """Reset the tracker singleton before each test."""
        MemoryTracker.reset()
        yield
        MemoryTracker.reset()

    @pytest.mark.asyncio
    async def test_hnsw_index_reports_memory_stats(self):
        """Test that HNSWVectorIndex correctly reports memory stats."""
        from headroom.memory.adapters.hnsw import HNSWVectorIndex
        from headroom.memory.models import Memory

        index = HNSWVectorIndex(dimension=128)

        # Add some vectors using Memory objects
        import numpy as np

        for i in range(10):
            embedding = np.random.rand(128).astype(np.float32).tolist()
            memory = Memory(
                id=f"mem_{i}",
                content=f"Test memory {i}",
                user_id="test_user",
                embedding=embedding,
            )
            await index.index(memory)

        stats = index.get_memory_stats()

        assert stats.name == "vector_index"
        assert stats.entry_count == 10
        assert stats.size_bytes > 0

    @pytest.mark.asyncio
    async def test_hnsw_index_size_grows_with_vectors(self):
        """Test that reported size grows as vectors are added."""
        from headroom.memory.adapters.hnsw import HNSWVectorIndex
        from headroom.memory.models import Memory

        index = HNSWVectorIndex(dimension=256)

        # Get initial size
        initial_stats = index.get_memory_stats()
        initial_size = initial_stats.size_bytes

        # Add vectors
        import numpy as np

        for i in range(100):
            embedding = np.random.rand(256).astype(np.float32).tolist()
            memory = Memory(
                id=f"mem_{i}",
                content=f"Test memory {i}",
                user_id="test_user",
                embedding=embedding,
            )
            await index.index(memory)

        # Get new size
        final_stats = index.get_memory_stats()

        assert final_stats.size_bytes > initial_size
        assert final_stats.entry_count == 100


class TestTrackerIntegrationWithMultipleStores:
    """Tests for MemoryTracker with multiple real stores."""

    @pytest.fixture(autouse=True)
    def reset_tracker(self):
        """Reset the tracker singleton before each test."""
        MemoryTracker.reset()
        yield
        MemoryTracker.reset()

    @pytest.mark.asyncio
    async def test_tracker_aggregates_multiple_stores(self):
        """Test that tracker correctly aggregates stats from multiple stores."""
        from headroom.cache.compression_store import CompressionStore
        from headroom.ccr.batch_store import BatchContextStore
        from headroom.memory.adapters.graph import InMemoryGraphStore
        from headroom.memory.adapters.graph_models import Entity

        tracker = MemoryTracker.get()

        # Create stores
        compression_store = CompressionStore(max_entries=100)
        batch_store = BatchContextStore()
        graph_store = InMemoryGraphStore()

        # Add some data
        compression_store.store("original content" * 50, "compressed")
        entity = Entity(id="node1", user_id="test", name="Test", entity_type="entity")
        await graph_store.add_entity(entity)

        # Register all stores
        tracker.register("compression_store", compression_store.get_memory_stats)
        tracker.register("batch_context_store", batch_store.get_memory_stats)
        tracker.register("graph_store", graph_store.get_memory_stats)

        # Get total
        total = tracker.get_total_tracked_bytes()

        # Should be sum of all stores
        cs_stats = compression_store.get_memory_stats()
        bs_stats = batch_store.get_memory_stats()
        gs_stats = graph_store.get_memory_stats()

        expected_total = cs_stats.size_bytes + bs_stats.size_bytes + gs_stats.size_bytes
        assert total == expected_total

    @pytest.mark.asyncio
    async def test_full_memory_report(self):
        """Test generating a full memory report with real stores."""
        from headroom.cache.compression_store import CompressionStore
        from headroom.memory.adapters.graph import InMemoryGraphStore
        from headroom.memory.adapters.graph_models import Entity

        tracker = MemoryTracker.get(target_budget_mb=100.0)

        # Create and register stores
        compression_store = CompressionStore(max_entries=1000)
        graph_store = InMemoryGraphStore()

        # Add data
        for i in range(10):
            compression_store.store(f"original content {i}" * 100, f"compressed_{i}")
            entity = Entity(
                id=f"node_{i}",
                user_id="test",
                name=f"Entity {i}",
                entity_type="entity",
            )
            await graph_store.add_entity(entity)

        tracker.register("compression_store", compression_store.get_memory_stats)
        tracker.register("graph_store", graph_store.get_memory_stats)

        # Get full report
        report = tracker.get_report()

        # Verify report structure
        assert report.process is not None
        assert report.process.rss_bytes >= 0
        assert len(report.components) == 2
        assert "compression_store" in report.components
        assert "graph_store" in report.components
        assert report.total_tracked_bytes > 0
        assert report.target_budget_bytes == 100 * 1024 * 1024

        # Verify serialization
        d = report.to_dict()
        assert "process" in d
        assert "components" in d
        assert "total_tracked_mb" in d


class TestMemoryBudgetEnforcement:
    """Tests for memory budget checking."""

    @pytest.fixture(autouse=True)
    def reset_tracker(self):
        """Reset the tracker singleton before each test."""
        MemoryTracker.reset()
        yield
        MemoryTracker.reset()

    def test_under_budget(self):
        """Test that under-budget is correctly detected."""
        from headroom.cache.compression_store import CompressionStore

        tracker = MemoryTracker.get(target_budget_mb=100.0)  # 100 MB budget

        store = CompressionStore(max_entries=10)  # Small store
        store.store("original", "compressed")

        tracker.register("compression_store", store.get_memory_stats)

        report = tracker.get_report()

        # Small store should be under budget
        assert report.is_over_budget is False

    def test_over_budget_detection(self):
        """Test that over-budget is correctly detected."""
        tracker = MemoryTracker.get(target_budget_mb=0.001)  # Very small budget (1 KB)

        # Create a component that reports large size
        from headroom.memory.tracker import ComponentStats

        def large_component_stats() -> ComponentStats:
            return ComponentStats(
                name="large_component",
                entry_count=1000,
                size_bytes=10 * 1024 * 1024,  # 10 MB
            )

        tracker.register("large_component", large_component_stats)

        report = tracker.get_report()

        # Should be over budget
        assert report.is_over_budget is True


class TestProcessStatsCollection:
    """Tests for process-level memory stats."""

    @pytest.fixture(autouse=True)
    def reset_tracker(self):
        """Reset the tracker singleton before each test."""
        MemoryTracker.reset()
        yield
        MemoryTracker.reset()

    def test_process_stats_collected(self):
        """Test that process stats are collected from the real process."""
        tracker = MemoryTracker.get()

        stats = tracker.get_process_stats()

        # psutil is optional — without it, stats are all zeros (graceful degradation)
        try:
            import psutil  # noqa: F401

            assert stats.rss_bytes > 0  # Process must use some memory
            assert stats.vms_bytes > 0
        except ImportError:
            assert stats.rss_bytes == 0  # No psutil → zeros expected
        assert stats.percent >= 0  # Could be 0 on some systems

    def test_process_stats_in_report(self):
        """Test that process stats are included in report."""
        tracker = MemoryTracker.get()

        report = tracker.get_report()

        try:
            import psutil  # noqa: F401

            assert report.process.rss_bytes > 0
            assert report.process.rss_mb > 0
        except ImportError:
            # Without psutil, process stats are zeros — that's expected
            assert report.process.rss_bytes == 0
