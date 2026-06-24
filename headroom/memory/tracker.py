"""Memory tracking infrastructure for headroom.

This module provides centralized memory tracking across all components,
enabling observability into memory usage patterns and budget enforcement.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Try to import psutil for process memory tracking
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class ComponentStats:
    """Statistics for a single memory component."""

    name: str
    entry_count: int
    size_bytes: int
    budget_bytes: int | None = None
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    last_updated: float = field(default_factory=time.time)

    @property
    def size_mb(self) -> float:
        """Size in megabytes."""
        return self.size_bytes / (1024 * 1024)

    @property
    def budget_mb(self) -> float | None:
        """Budget in megabytes."""
        return self.budget_bytes / (1024 * 1024) if self.budget_bytes else None

    @property
    def budget_used_percent(self) -> float | None:
        """Percentage of budget used."""
        if self.budget_bytes and self.budget_bytes > 0:
            return (self.size_bytes / self.budget_bytes) * 100
        return None

    @property
    def hit_rate(self) -> float | None:
        """Cache hit rate as percentage."""
        total = self.hits + self.misses
        if total > 0:
            return (self.hits / total) * 100
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "entry_count": self.entry_count,
            "size_bytes": self.size_bytes,
            "size_mb": round(self.size_mb, 2),
            "budget_bytes": self.budget_bytes,
            "budget_mb": round(self.budget_mb, 2) if self.budget_mb else None,
            "budget_used_percent": round(self.budget_used_percent, 2)
            if self.budget_used_percent
            else None,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 2) if self.hit_rate else None,
            "last_updated": self.last_updated,
        }


@dataclass
class ProcessStats:
    """Process-level memory statistics."""

    rss_bytes: int
    vms_bytes: int
    percent: float
    available_bytes: int
    total_bytes: int

    @property
    def rss_mb(self) -> float:
        """Resident set size in MB."""
        return self.rss_bytes / (1024 * 1024)

    @property
    def vms_mb(self) -> float:
        """Virtual memory size in MB."""
        return self.vms_bytes / (1024 * 1024)

    @property
    def available_mb(self) -> float:
        """Available system memory in MB."""
        return self.available_bytes / (1024 * 1024)

    @property
    def total_mb(self) -> float:
        """Total system memory in MB."""
        return self.total_bytes / (1024 * 1024)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "rss_bytes": self.rss_bytes,
            "rss_mb": round(self.rss_mb, 2),
            "vms_bytes": self.vms_bytes,
            "vms_mb": round(self.vms_mb, 2),
            "percent": round(self.percent, 2),
            "available_mb": round(self.available_mb, 2),
            "total_mb": round(self.total_mb, 2),
        }


@dataclass
class MemoryReport:
    """Complete memory report including process and component stats."""

    process: ProcessStats
    components: dict[str, ComponentStats]
    total_tracked_bytes: int
    target_budget_bytes: int | None
    timestamp: float = field(default_factory=time.time)

    @property
    def total_tracked_mb(self) -> float:
        """Total tracked memory in MB."""
        return self.total_tracked_bytes / (1024 * 1024)

    @property
    def target_budget_mb(self) -> float | None:
        """Target budget in MB."""
        return self.target_budget_bytes / (1024 * 1024) if self.target_budget_bytes else None

    @property
    def is_over_budget(self) -> bool:
        """Check if tracked memory exceeds target budget."""
        if self.target_budget_bytes:
            return self.total_tracked_bytes > self.target_budget_bytes
        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "process": self.process.to_dict(),
            "components": {name: stats.to_dict() for name, stats in self.components.items()},
            "total_tracked_bytes": self.total_tracked_bytes,
            "total_tracked_mb": round(self.total_tracked_mb, 2),
            "target_budget_bytes": self.target_budget_bytes,
            "target_budget_mb": round(self.target_budget_mb, 2) if self.target_budget_mb else None,
            "is_over_budget": self.is_over_budget,
            "timestamp": self.timestamp,
        }


class MemoryTracker:
    """Singleton that tracks memory usage across all components.

    Usage:
        # Register a component
        tracker = MemoryTracker.get()
        tracker.register("my_store", my_store.get_memory_stats)

        # Get all stats
        report = tracker.get_report()

        # Get specific component
        stats = tracker.get_component_stats("my_store")
    """

    _instance: MemoryTracker | None = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, target_budget_mb: float | None = None):
        """Initialize the tracker.

        Args:
            target_budget_mb: Target memory budget in MB for all tracked components.
        """
        self._components: dict[str, Callable[[], ComponentStats]] = {}
        self._target_budget_bytes: int | None = (
            int(target_budget_mb * 1024 * 1024) if target_budget_mb else None
        )
        self._component_lock = threading.Lock()

    @classmethod
    def get(cls, target_budget_mb: float | None = None) -> MemoryTracker:
        """Get or create the singleton instance.

        Args:
            target_budget_mb: Target memory budget (only used on first call).

        Returns:
            The singleton MemoryTracker instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(target_budget_mb=target_budget_mb)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance. Useful for testing."""
        with cls._lock:
            cls._instance = None

    def set_target_budget(self, budget_mb: float) -> None:
        """Set the target memory budget.

        Args:
            budget_mb: Target budget in megabytes.
        """
        self._target_budget_bytes = int(budget_mb * 1024 * 1024)

    def register(self, name: str, stats_fn: Callable[[], ComponentStats]) -> None:
        """Register a component's stats function.

        Args:
            name: Unique name for the component.
            stats_fn: Function that returns ComponentStats for this component.
        """
        with self._component_lock:
            self._components[name] = stats_fn

    def unregister(self, name: str) -> bool:
        """Unregister a component.

        Args:
            name: Name of the component to unregister.

        Returns:
            True if component was unregistered, False if not found.
        """
        with self._component_lock:
            if name in self._components:
                del self._components[name]
                return True
            return False

    def get_component_stats(self, name: str) -> ComponentStats | None:
        """Get stats for a specific component.

        Args:
            name: Name of the component.

        Returns:
            ComponentStats or None if component not found.
        """
        with self._component_lock:
            if name in self._components:
                try:
                    return self._components[name]()
                except Exception:
                    return None
        return None

    def get_all_component_stats(self) -> dict[str, ComponentStats]:
        """Get stats for all registered components.

        Returns:
            Dictionary mapping component names to their stats.
        """
        stats: dict[str, ComponentStats] = {}
        with self._component_lock:
            for name, fn in self._components.items():
                try:
                    stats[name] = fn()
                except Exception:
                    # Skip components that fail to report stats
                    pass
        return stats

    def get_process_stats(self) -> ProcessStats:
        """Get process-level memory statistics.

        Returns:
            ProcessStats with current memory usage.
        """
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            mem_info = process.memory_info()
            sys_mem = psutil.virtual_memory()
            return ProcessStats(
                rss_bytes=mem_info.rss,
                vms_bytes=mem_info.vms,
                percent=process.memory_percent(),
                available_bytes=sys_mem.available,
                total_bytes=sys_mem.total,
            )
        else:
            # Fallback when psutil not available
            return ProcessStats(
                rss_bytes=0,
                vms_bytes=0,
                percent=0.0,
                available_bytes=0,
                total_bytes=0,
            )

    def get_total_tracked_bytes(self) -> int:
        """Get total memory used by all tracked components.

        Returns:
            Total bytes used by tracked components.
        """
        stats = self.get_all_component_stats()
        return sum(s.size_bytes for s in stats.values())

    def get_report(self) -> MemoryReport:
        """Get a complete memory report.

        Returns:
            MemoryReport with process and component statistics.
        """
        process_stats = self.get_process_stats()
        component_stats = self.get_all_component_stats()
        total_tracked = sum(s.size_bytes for s in component_stats.values())

        return MemoryReport(
            process=process_stats,
            components=component_stats,
            total_tracked_bytes=total_tracked,
            target_budget_bytes=self._target_budget_bytes,
        )

    @property
    def registered_components(self) -> list[str]:
        """Get list of registered component names."""
        with self._component_lock:
            return list(self._components.keys())

    @property
    def target_budget_mb(self) -> float | None:
        """Get target budget in MB."""
        return self._target_budget_bytes / (1024 * 1024) if self._target_budget_bytes else None


def estimate_object_size(obj: Any, seen: set | None = None) -> int:
    """Estimate the memory size of a Python object recursively.

    This provides a rough estimate by traversing the object graph.
    For more accurate measurements, use tracemalloc or memory_profiler.

    Args:
        obj: Object to measure.
        seen: Set of already-seen object ids (for cycle detection).

    Returns:
        Estimated size in bytes.
    """
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(
            estimate_object_size(k, seen) + estimate_object_size(v, seen) for k, v in obj.items()
        )
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(estimate_object_size(item, seen) for item in obj)
    elif hasattr(obj, "__dict__"):
        size += estimate_object_size(obj.__dict__, seen)
    elif hasattr(obj, "__slots__"):
        size += sum(
            estimate_object_size(getattr(obj, slot, None), seen)
            for slot in obj.__slots__
            if hasattr(obj, slot)
        )

    return size
