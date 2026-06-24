"""Tests for memory tracking functionality.

These tests verify that the MemoryTracker correctly tracks memory usage
across all components without mocks or simulations.
"""

from __future__ import annotations

import sys

import pytest

from headroom.memory.tracker import (
    ComponentStats,
    MemoryReport,
    MemoryTracker,
    ProcessStats,
    estimate_object_size,
)


class TestComponentStats:
    """Tests for ComponentStats dataclass."""

    def test_basic_properties(self):
        """Test basic property calculations."""
        stats = ComponentStats(
            name="test_store",
            entry_count=100,
            size_bytes=1024 * 1024,  # 1 MB
            budget_bytes=2 * 1024 * 1024,  # 2 MB
            hits=80,
            misses=20,
            evictions=5,
        )

        assert stats.name == "test_store"
        assert stats.entry_count == 100
        assert stats.size_mb == 1.0
        assert stats.budget_mb == 2.0
        assert stats.budget_used_percent == 50.0
        assert stats.hit_rate == 80.0

    def test_no_budget(self):
        """Test when no budget is set."""
        stats = ComponentStats(
            name="test_store",
            entry_count=100,
            size_bytes=1024 * 1024,
            budget_bytes=None,
        )

        assert stats.budget_mb is None
        assert stats.budget_used_percent is None

    def test_no_hits_misses(self):
        """Test when no hits or misses recorded."""
        stats = ComponentStats(
            name="test_store",
            entry_count=100,
            size_bytes=1024,
            hits=0,
            misses=0,
        )

        assert stats.hit_rate is None

    def test_to_dict(self):
        """Test serialization to dictionary."""
        stats = ComponentStats(
            name="test_store",
            entry_count=100,
            size_bytes=1024 * 1024,
            budget_bytes=2 * 1024 * 1024,
            hits=80,
            misses=20,
            evictions=5,
        )

        d = stats.to_dict()

        assert d["name"] == "test_store"
        assert d["entry_count"] == 100
        assert d["size_bytes"] == 1024 * 1024
        assert d["size_mb"] == 1.0
        assert d["budget_mb"] == 2.0
        assert d["budget_used_percent"] == 50.0
        assert d["hit_rate"] == 80.0


class TestProcessStats:
    """Tests for ProcessStats dataclass."""

    def test_basic_properties(self):
        """Test basic property calculations."""
        stats = ProcessStats(
            rss_bytes=500 * 1024 * 1024,  # 500 MB
            vms_bytes=1024 * 1024 * 1024,  # 1 GB
            percent=5.0,
            available_bytes=8 * 1024 * 1024 * 1024,  # 8 GB
            total_bytes=16 * 1024 * 1024 * 1024,  # 16 GB
        )

        assert stats.rss_mb == 500.0
        assert stats.vms_mb == 1024.0
        assert stats.percent == 5.0
        assert stats.available_mb == 8192.0
        assert stats.total_mb == 16384.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        stats = ProcessStats(
            rss_bytes=500 * 1024 * 1024,
            vms_bytes=1024 * 1024 * 1024,
            percent=5.0,
            available_bytes=8 * 1024 * 1024 * 1024,
            total_bytes=16 * 1024 * 1024 * 1024,
        )

        d = stats.to_dict()

        assert d["rss_mb"] == 500.0
        assert d["vms_mb"] == 1024.0
        assert d["percent"] == 5.0


class TestMemoryTracker:
    """Tests for MemoryTracker singleton."""

    @pytest.fixture(autouse=True)
    def reset_tracker(self):
        """Reset the tracker singleton before each test."""
        MemoryTracker.reset()
        yield
        MemoryTracker.reset()

    def test_singleton_pattern(self):
        """Test that MemoryTracker is a singleton."""
        tracker1 = MemoryTracker.get()
        tracker2 = MemoryTracker.get()

        assert tracker1 is tracker2

    def test_reset(self):
        """Test singleton reset."""
        tracker1 = MemoryTracker.get()
        MemoryTracker.reset()
        tracker2 = MemoryTracker.get()

        assert tracker1 is not tracker2

    def test_register_component(self):
        """Test registering a component."""
        tracker = MemoryTracker.get()

        def get_stats() -> ComponentStats:
            return ComponentStats(
                name="test_component",
                entry_count=10,
                size_bytes=1024,
            )

        tracker.register("test_component", get_stats)

        assert "test_component" in tracker.registered_components

    def test_unregister_component(self):
        """Test unregistering a component."""
        tracker = MemoryTracker.get()

        def get_stats() -> ComponentStats:
            return ComponentStats(
                name="test_component",
                entry_count=10,
                size_bytes=1024,
            )

        tracker.register("test_component", get_stats)
        assert "test_component" in tracker.registered_components

        result = tracker.unregister("test_component")
        assert result is True
        assert "test_component" not in tracker.registered_components

        # Unregistering non-existent component returns False
        result = tracker.unregister("non_existent")
        assert result is False

    def test_get_component_stats(self):
        """Test getting stats for a specific component."""
        tracker = MemoryTracker.get()

        def get_stats() -> ComponentStats:
            return ComponentStats(
                name="test_component",
                entry_count=42,
                size_bytes=2048,
            )

        tracker.register("test_component", get_stats)

        stats = tracker.get_component_stats("test_component")

        assert stats is not None
        assert stats.name == "test_component"
        assert stats.entry_count == 42
        assert stats.size_bytes == 2048

    def test_get_component_stats_not_found(self):
        """Test getting stats for non-existent component."""
        tracker = MemoryTracker.get()

        stats = tracker.get_component_stats("non_existent")

        assert stats is None

    def test_get_all_component_stats(self):
        """Test getting stats for all components."""
        tracker = MemoryTracker.get()

        def get_stats_a() -> ComponentStats:
            return ComponentStats(name="component_a", entry_count=10, size_bytes=1024)

        def get_stats_b() -> ComponentStats:
            return ComponentStats(name="component_b", entry_count=20, size_bytes=2048)

        tracker.register("component_a", get_stats_a)
        tracker.register("component_b", get_stats_b)

        all_stats = tracker.get_all_component_stats()

        assert len(all_stats) == 2
        assert "component_a" in all_stats
        assert "component_b" in all_stats
        assert all_stats["component_a"].entry_count == 10
        assert all_stats["component_b"].entry_count == 20

    def test_get_process_stats(self):
        """Test getting process-level stats."""
        tracker = MemoryTracker.get()

        stats = tracker.get_process_stats()

        # Should return ProcessStats (may be zero if psutil not available)
        assert isinstance(stats, ProcessStats)
        assert stats.rss_bytes >= 0
        assert stats.vms_bytes >= 0

    def test_get_total_tracked_bytes(self):
        """Test getting total tracked bytes."""
        tracker = MemoryTracker.get()

        def get_stats_a() -> ComponentStats:
            return ComponentStats(name="a", entry_count=10, size_bytes=1000)

        def get_stats_b() -> ComponentStats:
            return ComponentStats(name="b", entry_count=20, size_bytes=2000)

        tracker.register("a", get_stats_a)
        tracker.register("b", get_stats_b)

        total = tracker.get_total_tracked_bytes()

        assert total == 3000

    def test_get_report(self):
        """Test getting full memory report."""
        tracker = MemoryTracker.get(target_budget_mb=100.0)

        def get_stats() -> ComponentStats:
            return ComponentStats(name="test", entry_count=10, size_bytes=50 * 1024 * 1024)

        tracker.register("test", get_stats)

        report = tracker.get_report()

        assert isinstance(report, MemoryReport)
        assert isinstance(report.process, ProcessStats)
        assert "test" in report.components
        assert report.total_tracked_bytes == 50 * 1024 * 1024
        assert report.target_budget_bytes == 100 * 1024 * 1024
        assert report.is_over_budget is False

    def test_is_over_budget(self):
        """Test budget checking."""
        tracker = MemoryTracker.get(target_budget_mb=10.0)  # 10 MB budget

        def get_stats() -> ComponentStats:
            return ComponentStats(
                name="large", entry_count=10, size_bytes=20 * 1024 * 1024
            )  # 20 MB

        tracker.register("large", get_stats)

        report = tracker.get_report()

        assert report.is_over_budget is True

    def test_set_target_budget(self):
        """Test setting target budget after creation."""
        tracker = MemoryTracker.get()

        assert tracker.target_budget_mb is None

        tracker.set_target_budget(500.0)

        assert tracker.target_budget_mb == 500.0


class TestEstimateObjectSize:
    """Tests for the estimate_object_size utility function."""

    def test_simple_types(self):
        """Test size estimation for simple types."""
        # Integer
        int_size = estimate_object_size(42)
        assert int_size == sys.getsizeof(42)

        # String
        s = "hello world"
        str_size = estimate_object_size(s)
        assert str_size == sys.getsizeof(s)

    def test_dict(self):
        """Test size estimation for dictionaries."""
        d = {"a": 1, "b": 2, "c": 3}
        size = estimate_object_size(d)

        # Size should be at least the base dict size
        assert size >= sys.getsizeof(d)
        # Size should include keys and values
        assert size > sys.getsizeof({})

    def test_list(self):
        """Test size estimation for lists."""
        lst = [1, 2, 3, "hello", "world"]
        size = estimate_object_size(lst)

        assert size >= sys.getsizeof(lst)
        assert size > sys.getsizeof([])

    def test_nested_structure(self):
        """Test size estimation for nested structures."""
        nested = {
            "items": [{"id": 1, "name": "first"}, {"id": 2, "name": "second"}],
            "metadata": {"count": 2, "tags": ["a", "b", "c"]},
        }
        size = estimate_object_size(nested)

        # Should be larger than just the outer dict
        assert size > sys.getsizeof(nested)

    def test_circular_reference(self):
        """Test that circular references don't cause infinite loop."""
        d: dict = {"a": 1}
        d["self"] = d  # Circular reference

        # Should not hang or crash
        size = estimate_object_size(d)
        assert size > 0


class TestMemoryReportSerialization:
    """Tests for MemoryReport serialization."""

    def test_to_dict(self):
        """Test that MemoryReport serializes correctly."""
        process = ProcessStats(
            rss_bytes=100 * 1024 * 1024,
            vms_bytes=200 * 1024 * 1024,
            percent=1.0,
            available_bytes=8 * 1024 * 1024 * 1024,
            total_bytes=16 * 1024 * 1024 * 1024,
        )

        components = {
            "store_a": ComponentStats(name="store_a", entry_count=100, size_bytes=10 * 1024 * 1024),
            "store_b": ComponentStats(name="store_b", entry_count=200, size_bytes=20 * 1024 * 1024),
        }

        report = MemoryReport(
            process=process,
            components=components,
            total_tracked_bytes=30 * 1024 * 1024,
            target_budget_bytes=50 * 1024 * 1024,
        )

        d = report.to_dict()

        assert "process" in d
        assert "components" in d
        assert "total_tracked_mb" in d
        assert "target_budget_mb" in d
        assert "is_over_budget" in d

        assert d["process"]["rss_mb"] == 100.0
        assert d["total_tracked_mb"] == 30.0
        assert d["target_budget_mb"] == 50.0
        assert d["is_over_budget"] is False
