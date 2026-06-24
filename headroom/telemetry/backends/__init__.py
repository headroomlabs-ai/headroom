"""Storage backends for TOIN (Tool Output Intelligence Network).

This module provides pluggable storage backends for TOIN pattern persistence.
The default is filesystem storage (JSON), but alternative backends can be
implemented for distributed/multi-tenant scenarios (Redis, PostgreSQL, etc.).

Usage:
    from headroom.telemetry.backends import TOINBackend, FileSystemTOINBackend

    # Use default filesystem backend
    backend = FileSystemTOINBackend("/path/to/toin.json")

    # Use custom backend (e.g. from a SaaS adapter package)
    class RedisBackend:
        # Implement TOINBackend protocol
        ...

    toin = ToolIntelligenceNetwork(config, backend=RedisBackend(...))
"""

from .base import TOINBackend
from .filesystem import FileSystemTOINBackend

__all__ = [
    "TOINBackend",
    "FileSystemTOINBackend",
]
