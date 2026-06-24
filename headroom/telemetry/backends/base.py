"""Base protocol for TOIN storage backends.

This protocol defines the minimal interface that TOIN storage backends must
implement. The interface is intentionally simple — it only handles serialized
pattern data load/save. All TOIN logic (pattern aggregation, recommendations,
merging) stays in ToolIntelligenceNetwork.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TOINBackend(Protocol):
    """Protocol for TOIN storage backends.

    Implementations can use any storage mechanism: filesystem, Redis,
    PostgreSQL, S3, etc.

    Design Principles:
    - Two operations only: load and save
    - Data is a serialized dict (from ToolIntelligenceNetwork.export_patterns)
    - Backend owns atomicity and durability
    - Thread-safety is implementation's responsibility

    Example implementation:
        class MyBackend:
            def load(self) -> dict[str, Any]:
                return json.loads(self._redis.get("toin_data") or "{}")

            def save(self, data: dict[str, Any]) -> None:
                self._redis.set("toin_data", json.dumps(data))
    """

    def load(self) -> dict[str, Any]:
        """Load serialized TOIN data.

        Returns:
            Dict with TOIN data (as produced by export_patterns),
            or empty dict if no data exists.
        """
        ...

    def save(self, data: dict[str, Any]) -> None:
        """Save serialized TOIN data.

        The implementation must ensure atomicity — a failed save must not
        corrupt existing data.

        Args:
            data: Serialized TOIN data (from export_patterns).
        """
        ...
