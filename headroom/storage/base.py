"""Base storage interface for Headroom SDK."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime
from typing import Any

from ..config import RequestMetrics


class Storage(ABC):
    """Abstract base class for metrics storage."""

    @abstractmethod
    def save(self, metrics: RequestMetrics) -> None:
        """
        Save request metrics.

        Args:
            metrics: RequestMetrics to save.
        """
        pass

    @abstractmethod
    def get(self, request_id: str) -> RequestMetrics | None:
        """
        Get metrics by request ID.

        Args:
            request_id: The request ID.

        Returns:
            RequestMetrics or None if not found.
        """
        pass

    @abstractmethod
    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        model: str | None = None,
        mode: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RequestMetrics]:
        """
        Query metrics with filters.

        Args:
            start_time: Filter by timestamp >= start_time.
            end_time: Filter by timestamp <= end_time.
            model: Filter by model name.
            mode: Filter by mode (audit/optimize).
            limit: Maximum results to return.
            offset: Number of results to skip.

        Returns:
            List of matching RequestMetrics.
        """
        pass

    @abstractmethod
    def count(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        model: str | None = None,
        mode: str | None = None,
    ) -> int:
        """
        Count metrics matching filters.

        Args:
            start_time: Filter by timestamp >= start_time.
            end_time: Filter by timestamp <= end_time.
            model: Filter by model name.
            mode: Filter by mode.

        Returns:
            Count of matching records.
        """
        pass

    @abstractmethod
    def iter_all(self) -> Iterator[RequestMetrics]:
        """
        Iterate over all stored metrics.

        Yields:
            RequestMetrics objects.
        """
        pass

    @abstractmethod
    def get_summary_stats(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Get summary statistics.

        Args:
            start_time: Filter by timestamp >= start_time.
            end_time: Filter by timestamp <= end_time.

        Returns:
            Dict with summary stats (total_requests, tokens_saved, etc.)
        """
        pass

    def close(self) -> None:  # noqa: B027
        """Close storage connection if applicable."""
        pass

    def __enter__(self) -> Storage:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
