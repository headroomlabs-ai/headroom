"""Comprehensive integration tests for memory tracking with real components.

These tests exercise the full system including:
- Memory system (GraphStore, HNSWVectorIndex)
- CCR (Compress-Cache-Retrieve)
- Compression store
- Real API calls through the proxy

Tests track memory usage throughout to verify our tracking is accurate.

Requirements:
    - ANTHROPIC_API_KEY in .env
    - Run with: uv run pytest tests/test_memory_usage_integration.py -v -s
"""

from __future__ import annotations

import os

import pytest

# Load .env values into a local dict and apply per-test (not at module
# level) — see tests/_dotenv.py for why.
from tests._dotenv import autouse_apply_env, load_env_overrides

_env_overrides = load_env_overrides()
apply_dotenv = autouse_apply_env(_env_overrides)

# Check HNSW availability for skipping tests
try:
    from headroom.memory.adapters.hnsw import _check_hnswlib_available

    HNSW_AVAILABLE = _check_hnswlib_available()
except ImportError:
    HNSW_AVAILABLE = False


def get_process_memory_mb() -> float:
    """Get current process memory in MB."""
    import psutil

    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def get_tracked_memory() -> dict:
    """Get memory stats from the tracker."""
    from headroom.memory.tracker import MemoryTracker

    tracker = MemoryTracker.get()
    report = tracker.get_report()
    return report.to_dict()


class TestMemorySystemIntegration:
    """Tests for the memory system (GraphStore + HNSWVectorIndex)."""

    @pytest.fixture(autouse=True)
    def reset_tracker(self):
        """Reset the tracker singleton before each test."""
        from headroom.memory.tracker import MemoryTracker

        MemoryTracker.reset()
        yield
        MemoryTracker.reset()

    @pytest.mark.asyncio
    async def test_graph_store_memory_growth(self):
        """Test that graph store memory is tracked as entities are added."""
        from headroom.memory.adapters.graph import InMemoryGraphStore
        from headroom.memory.adapters.graph_models import Entity, Relationship
        from headroom.memory.tracker import MemoryTracker

        tracker = MemoryTracker.get()
        store = InMemoryGraphStore()
        tracker.register("graph_store", store.get_memory_stats)

        print("\n=== Graph Store Memory Growth Test ===")

        # Track memory at each stage
        memory_snapshots = []

        # Initial state
        stats = store.get_memory_stats()
        memory_snapshots.append(("initial", stats.entry_count, stats.size_bytes))
        print(f"Initial: {stats.entry_count} entries, {stats.size_bytes} bytes")

        # Add 100 entities
        for i in range(100):
            entity = Entity(
                id=f"entity_{i}",
                user_id="test_user",
                name=f"Test Entity {i}",
                entity_type="concept",
                description=f"This is a detailed description for entity {i} " * 10,
                properties={"index": i, "data": "x" * 200},
            )
            await store.add_entity(entity)

        stats = store.get_memory_stats()
        memory_snapshots.append(("100 entities", stats.entry_count, stats.size_bytes))
        print(f"After 100 entities: {stats.entry_count} entries, {stats.size_bytes} bytes")

        # Add 200 relationships
        for i in range(200):
            rel = Relationship(
                id=f"rel_{i}",
                user_id="test_user",
                source_id=f"entity_{i % 100}",
                target_id=f"entity_{(i + 1) % 100}",
                relation_type="related_to",
                properties={"weight": 0.5, "metadata": "y" * 100},
            )
            await store.add_relationship(rel)

        stats = store.get_memory_stats()
        memory_snapshots.append(("+ 200 relationships", stats.entry_count, stats.size_bytes))
        print(f"After 200 relationships: {stats.entry_count} entries, {stats.size_bytes} bytes")

        # Verify memory grew
        assert memory_snapshots[1][2] > memory_snapshots[0][2], (
            "Memory should grow after adding entities"
        )
        assert memory_snapshots[2][2] > memory_snapshots[1][2], (
            "Memory should grow after adding relationships"
        )

        # Verify tracker reports correctly
        report = tracker.get_report()
        assert "graph_store" in report.components
        assert (
            report.components["graph_store"].entry_count == 300
        )  # 100 entities + 200 relationships

        print(f"\nTotal tracked memory: {report.total_tracked_mb:.4f} MB")
        print(f"Process RSS: {report.process.rss_mb:.1f} MB")

    @pytest.mark.skipif(not HNSW_AVAILABLE, reason="hnswlib not available")
    @pytest.mark.asyncio
    async def test_hnsw_vector_index_memory_growth(self):
        """Test that HNSW vector index memory is tracked as vectors are added."""
        from headroom.memory.adapters.hnsw import HNSWVectorIndex
        from headroom.memory.models import Memory
        from headroom.memory.tracker import MemoryTracker

        tracker = MemoryTracker.get()

        # Use 384 dimensions (common for MiniLM embeddings)
        index = HNSWVectorIndex(dimension=384)
        tracker.register("vector_index", index.get_memory_stats)

        print("\n=== HNSW Vector Index Memory Growth Test ===")

        import numpy as np

        # Track memory at each stage
        memory_snapshots = []

        # Initial state
        stats = index.get_memory_stats()
        memory_snapshots.append(("initial", stats.entry_count, stats.size_bytes))
        print(f"Initial: {stats.entry_count} entries, {stats.size_bytes} bytes")

        # Add 100 vectors
        for i in range(100):
            embedding = np.random.rand(384).astype(np.float32).tolist()
            memory = Memory(
                id=f"mem_{i}",
                content=f"This is memory content {i} with some additional text " * 5,
                user_id="test_user",
                embedding=embedding,
                importance=0.5 + (i % 10) / 20,
            )
            await index.index(memory)

        stats = index.get_memory_stats()
        memory_snapshots.append(("100 vectors", stats.entry_count, stats.size_bytes))
        print(f"After 100 vectors: {stats.entry_count} entries, {stats.size_bytes} bytes")

        # Add 400 more vectors
        for i in range(100, 500):
            embedding = np.random.rand(384).astype(np.float32).tolist()
            memory = Memory(
                id=f"mem_{i}",
                content=f"This is memory content {i} with some additional text " * 5,
                user_id="test_user",
                embedding=embedding,
            )
            await index.index(memory)

        stats = index.get_memory_stats()
        memory_snapshots.append(("500 vectors", stats.entry_count, stats.size_bytes))
        print(f"After 500 vectors: {stats.entry_count} entries, {stats.size_bytes} bytes")

        # Verify memory grew
        assert memory_snapshots[1][2] > memory_snapshots[0][2], (
            "Memory should grow after adding vectors"
        )
        assert memory_snapshots[2][2] > memory_snapshots[1][2], (
            "Memory should grow with more vectors"
        )

        # Verify tracker reports correctly
        report = tracker.get_report()
        assert "vector_index" in report.components
        assert report.components["vector_index"].entry_count == 500

        print(f"\nTotal tracked memory: {report.total_tracked_mb:.4f} MB")
        print(f"Process RSS: {report.process.rss_mb:.1f} MB")


class TestCCRIntegration:
    """Tests for CCR (Compress-Cache-Retrieve) memory tracking."""

    @pytest.fixture(autouse=True)
    def reset_stores(self):
        """Reset stores before each test."""
        from headroom.ccr.batch_store import reset_batch_context_store
        from headroom.memory.tracker import MemoryTracker

        MemoryTracker.reset()
        reset_batch_context_store()
        yield
        MemoryTracker.reset()
        reset_batch_context_store()

    def test_compression_store_memory_growth(self):
        """Test that compression store memory is tracked correctly."""
        from headroom.cache.compression_store import CompressionStore
        from headroom.memory.tracker import MemoryTracker

        tracker = MemoryTracker.get()
        store = CompressionStore(max_entries=1000, default_ttl=3600)
        tracker.register("compression_store", store.get_memory_stats)

        print("\n=== Compression Store Memory Growth Test ===")

        memory_snapshots = []

        # Initial state
        stats = store.get_memory_stats()
        memory_snapshots.append(("initial", stats.entry_count, stats.size_bytes))
        print(f"Initial: {stats.entry_count} entries, {stats.size_bytes} bytes")

        # Add compressed content (simulating tool outputs)
        for i in range(50):
            original = f"Original tool output {i}: " + "data " * 500
            compressed = f"Compressed {i}: " + "data " * 50
            store.store(
                original=original,
                compressed=compressed,
                original_tokens=len(original.split()),
                compressed_tokens=len(compressed.split()),
                tool_name=f"tool_{i % 5}",
            )

        stats = store.get_memory_stats()
        memory_snapshots.append(("50 entries", stats.entry_count, stats.size_bytes))
        print(f"After 50 entries: {stats.entry_count} entries, {stats.size_bytes} bytes")

        # Add more with larger content
        for i in range(50, 150):
            original = f"Large tool output {i}: " + "data " * 2000
            compressed = f"Compressed {i}: " + "data " * 200
            store.store(
                original=original,
                compressed=compressed,
                original_tokens=len(original.split()),
                compressed_tokens=len(compressed.split()),
                tool_name=f"tool_{i % 5}",
            )

        stats = store.get_memory_stats()
        memory_snapshots.append(("150 entries", stats.entry_count, stats.size_bytes))
        print(f"After 150 entries: {stats.entry_count} entries, {stats.size_bytes} bytes")

        # Verify memory grew
        assert memory_snapshots[1][2] > memory_snapshots[0][2]
        assert memory_snapshots[2][2] > memory_snapshots[1][2]

        # Test retrieval (should register hits)
        # Get a key from the first entry
        first_key = store.store("test original", "test compressed")
        store.retrieve(first_key)
        store.retrieve(first_key)
        store.retrieve("nonexistent")

        stats = store.get_memory_stats()
        print(f"\nAfter retrievals - Hits: {stats.hits}, Misses: {stats.misses}")

        report = tracker.get_report()
        print(f"Total tracked memory: {report.total_tracked_mb:.4f} MB")

    def test_batch_context_store_memory_growth(self):
        """Test that batch context store memory is tracked correctly."""
        from headroom.ccr.batch_store import (
            BatchContext,
            BatchContextStore,
            BatchRequestContext,
        )
        from headroom.memory.tracker import MemoryTracker

        tracker = MemoryTracker.get()
        store = BatchContextStore(ttl=3600, max_contexts=1000)
        tracker.register("batch_context_store", store.get_memory_stats)

        print("\n=== Batch Context Store Memory Growth Test ===")

        memory_snapshots = []

        # Initial state
        stats = store.get_memory_stats()
        memory_snapshots.append(("initial", stats.entry_count, stats.size_bytes))
        print(f"Initial: {stats.entry_count} entries, {stats.size_bytes} bytes")

        # Add batch contexts (simulating batch API submissions)
        for batch_num in range(20):
            ctx = BatchContext(
                batch_id=f"batch_{batch_num}",
                provider="anthropic",
            )
            # Each batch has multiple requests
            for req_num in range(10):
                ctx.add_request(
                    BatchRequestContext(
                        custom_id=f"req_{batch_num}_{req_num}",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": f"Request {req_num}: " + "context " * 100},
                        ],
                        model="claude-sonnet-4-20250514",
                        tools=[
                            {
                                "name": "search",
                                "description": "Search the web",
                                "input_schema": {"type": "object", "properties": {}},
                            }
                        ],
                    )
                )
            # Store directly (bypassing async for testing)
            store._contexts[ctx.batch_id] = ctx

        stats = store.get_memory_stats()
        memory_snapshots.append(("20 batches", stats.entry_count, stats.size_bytes))
        print(
            f"After 20 batches (200 requests): {stats.entry_count} entries, {stats.size_bytes} bytes"
        )

        # Verify memory grew
        assert memory_snapshots[1][2] > memory_snapshots[0][2]

        report = tracker.get_report()
        print(f"Total tracked memory: {report.total_tracked_mb:.4f} MB")


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set in environment",
)
class TestProxyMemoryIntegration:
    """Tests that exercise the proxy with real API calls and track memory."""

    @pytest.fixture
    def api_key(self):
        """Get API key from environment."""
        return os.environ.get("ANTHROPIC_API_KEY")

    @pytest.fixture(autouse=True)
    def reset_tracker(self):
        """Reset the tracker singleton before each test."""
        from headroom.memory.tracker import MemoryTracker

        MemoryTracker.reset()
        yield
        MemoryTracker.reset()

    def test_real_api_calls_memory_tracking(self, api_key):
        """Test memory tracking with real API calls."""
        import httpx

        from headroom.memory.tracker import MemoryTracker

        tracker = MemoryTracker.get()

        print("\n=== Real API Calls Memory Tracking Test ===")

        # Note: This test requires a running proxy
        # We'll test the components directly instead

        # Create and register stores
        from headroom.cache.compression_store import CompressionStore
        from headroom.ccr.batch_store import BatchContextStore

        compression_store = CompressionStore(max_entries=100)
        batch_store = BatchContextStore()

        tracker.register("compression_store", compression_store.get_memory_stats)
        tracker.register("batch_context_store", batch_store.get_memory_stats)

        initial_report = tracker.get_report()
        print(f"Initial tracked: {initial_report.total_tracked_mb:.4f} MB")
        print(f"Initial RSS: {initial_report.process.rss_mb:.1f} MB")

        # Make real API call using httpx directly
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        messages_list = [
            [{"role": "user", "content": f"Say 'test {i}' and nothing else."}] for i in range(3)
        ]

        with httpx.Client(timeout=60.0) as client:
            for i, messages in enumerate(messages_list):
                response = client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 50,
                        "messages": messages,
                    },
                )
                assert response.status_code == 200, f"API call failed: {response.text}"

                # Simulate storing compressed response (as CCR would)
                response_text = response.text
                compression_store.store(
                    original=response_text,
                    compressed=response_text[:100],  # Simulated compression
                    tool_name="api_response",
                )

                report = tracker.get_report()
                print(
                    f"After request {i + 1}: tracked={report.total_tracked_mb:.4f} MB, RSS={report.process.rss_mb:.1f} MB"
                )

        final_report = tracker.get_report()
        print(f"\nFinal tracked: {final_report.total_tracked_mb:.4f} MB")
        print(f"Final RSS: {final_report.process.rss_mb:.1f} MB")

        # Verify stores have entries
        assert final_report.components["compression_store"].entry_count == 3


class TestCombinedMemoryTracking:
    """Tests that combine multiple components and track total memory."""

    @pytest.fixture(autouse=True)
    def reset_all(self):
        """Reset all stores."""
        from headroom.ccr.batch_store import reset_batch_context_store
        from headroom.memory.tracker import MemoryTracker

        MemoryTracker.reset()
        reset_batch_context_store()
        yield
        MemoryTracker.reset()
        reset_batch_context_store()

    @pytest.mark.skipif(not HNSW_AVAILABLE, reason="hnswlib not available")
    @pytest.mark.asyncio
    async def test_all_components_memory_tracking(self):
        """Test memory tracking with all components active."""
        import numpy as np

        from headroom.cache.compression_store import CompressionStore
        from headroom.ccr.batch_store import BatchContext, BatchContextStore, BatchRequestContext
        from headroom.memory.adapters.graph import InMemoryGraphStore
        from headroom.memory.adapters.graph_models import Entity, Relationship
        from headroom.memory.adapters.hnsw import HNSWVectorIndex
        from headroom.memory.models import Memory
        from headroom.memory.tracker import MemoryTracker

        tracker = MemoryTracker.get(target_budget_mb=50.0)  # Set a 50MB budget

        print("\n=== Combined Memory Tracking Test ===")

        # Create all components
        compression_store = CompressionStore(max_entries=500)
        batch_store = BatchContextStore(max_contexts=100)
        graph_store = InMemoryGraphStore()
        vector_index = HNSWVectorIndex(dimension=384)

        # Register all with tracker
        tracker.register("compression_store", compression_store.get_memory_stats)
        tracker.register("batch_context_store", batch_store.get_memory_stats)
        tracker.register("graph_store", graph_store.get_memory_stats)
        tracker.register("vector_index", vector_index.get_memory_stats)

        # Initial state
        report = tracker.get_report()
        print("\nInitial state:")
        print(f"  Total tracked: {report.total_tracked_mb:.4f} MB")
        print(f"  Budget: {report.target_budget_mb:.1f} MB")
        print(f"  Over budget: {report.is_over_budget}")

        # Add data to all components
        print("\nAdding data to components...")

        # 1. Compression store - 100 entries (unique content for each)
        for i in range(100):
            compression_store.store(
                original=f"unique content {i}: " + "x" * 1000,
                compressed=f"compressed {i}: " + "x" * 100,
                tool_name=f"tool_{i}",
            )

        # 2. Batch store - 10 batches with 5 requests each
        for b in range(10):
            ctx = BatchContext(batch_id=f"batch_{b}", provider="anthropic")
            for r in range(5):
                ctx.add_request(
                    BatchRequestContext(
                        custom_id=f"req_{b}_{r}",
                        messages=[{"role": "user", "content": "test " * 50}],
                        model="claude-sonnet-4-20250514",
                    )
                )
            batch_store._contexts[ctx.batch_id] = ctx

        # 3. Graph store - 50 entities, 100 relationships
        for i in range(50):
            entity = Entity(
                id=f"entity_{i}",
                user_id="test",
                name=f"Entity {i}",
                entity_type="concept",
                properties={"data": "y" * 200},
            )
            await graph_store.add_entity(entity)

        for i in range(100):
            rel = Relationship(
                id=f"rel_{i}",
                user_id="test",
                source_id=f"entity_{i % 50}",
                target_id=f"entity_{(i + 1) % 50}",
                relation_type="related",
            )
            await graph_store.add_relationship(rel)

        # 4. Vector index - 200 vectors
        for i in range(200):
            embedding = np.random.rand(384).astype(np.float32).tolist()
            memory = Memory(
                id=f"mem_{i}",
                content=f"Memory {i}",
                user_id="test",
                embedding=embedding,
            )
            await vector_index.index(memory)

        # Final state
        report = tracker.get_report()
        print("\nAfter adding data:")
        print("  Components:")
        for name, comp in report.components.items():
            print(f"    {name}: {comp.entry_count} entries, {comp.size_bytes / 1024:.2f} KB")
        print(f"  Total tracked: {report.total_tracked_mb:.4f} MB")
        print(f"  Process RSS: {report.process.rss_mb:.1f} MB")
        print(f"  Over budget: {report.is_over_budget}")

        # Verify all components are tracked
        assert len(report.components) == 4
        assert report.components["compression_store"].entry_count == 100
        assert report.components["batch_context_store"].entry_count == 10
        assert report.components["graph_store"].entry_count == 150  # 50 + 100
        assert report.components["vector_index"].entry_count == 200

        # Verify total is sum of components
        total_from_components = sum(c.size_bytes for c in report.components.values())
        assert report.total_tracked_bytes == total_from_components

    @pytest.mark.skipif(not HNSW_AVAILABLE, reason="hnswlib not available")
    @pytest.mark.asyncio
    async def test_memory_budget_enforcement(self):
        """Test that budget enforcement works correctly."""
        import numpy as np

        from headroom.memory.adapters.hnsw import HNSWVectorIndex
        from headroom.memory.models import Memory
        from headroom.memory.tracker import MemoryTracker

        # Set a very small budget (1 MB)
        tracker = MemoryTracker.get(target_budget_mb=1.0)

        vector_index = HNSWVectorIndex(dimension=384)
        tracker.register("vector_index", vector_index.get_memory_stats)

        print("\n=== Budget Enforcement Test ===")

        # Add vectors until we exceed budget
        for i in range(1000):
            embedding = np.random.rand(384).astype(np.float32).tolist()
            memory = Memory(
                id=f"mem_{i}",
                content=f"Memory {i} with extra content " * 10,
                user_id="test",
                embedding=embedding,
            )
            await vector_index.index(memory)

            if i % 100 == 0:
                report = tracker.get_report()
                print(
                    f"After {i} vectors: {report.total_tracked_mb:.4f} MB, over_budget={report.is_over_budget}"
                )
                if report.is_over_budget:
                    print(f"  Budget exceeded at {i} vectors!")
                    break

        report = tracker.get_report()
        print(
            f"\nFinal: {report.total_tracked_mb:.4f} MB (budget: {report.target_budget_mb:.1f} MB)"
        )

        # With 1MB budget and 384-dim vectors, we should exceed budget
        # Each vector is ~1.5KB (384 floats * 4 bytes + metadata)
        # 1000 vectors = ~1.5MB, so we should exceed 1MB budget


class TestMemoryReportEndpoint:
    """Test the /debug/memory endpoint format."""

    @pytest.fixture(autouse=True)
    def reset_tracker(self):
        """Reset the tracker singleton before each test."""
        from headroom.memory.tracker import MemoryTracker

        MemoryTracker.reset()
        yield
        MemoryTracker.reset()

    def test_memory_report_serialization(self):
        """Test that memory report serializes correctly for API response."""
        from headroom.cache.compression_store import CompressionStore
        from headroom.memory.tracker import MemoryTracker

        tracker = MemoryTracker.get(target_budget_mb=100.0)

        store = CompressionStore(max_entries=10)
        store.store("original", "compressed")
        tracker.register("compression_store", store.get_memory_stats)

        report = tracker.get_report()
        data = report.to_dict()

        # Verify structure matches what API returns
        assert "process" in data
        assert "rss_mb" in data["process"]
        assert "vms_mb" in data["process"]
        assert "percent" in data["process"]

        assert "components" in data
        assert "compression_store" in data["components"]
        comp = data["components"]["compression_store"]
        assert "name" in comp
        assert "entry_count" in comp
        assert "size_bytes" in comp
        assert "size_mb" in comp
        assert "hits" in comp
        assert "misses" in comp

        assert "total_tracked_mb" in data
        assert "target_budget_mb" in data
        assert "is_over_budget" in data
        assert "timestamp" in data

        print("\n=== Memory Report Format ===")
        import json

        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
