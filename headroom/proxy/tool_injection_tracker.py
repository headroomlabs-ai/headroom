"""Session-scoped state for sticky memory tool injection."""

from __future__ import annotations

import threading
from collections import OrderedDict


class SessionToolTracker:
    """Bounded LRU tracker recording per-session memory-tool injection state."""

    def __init__(self, max_sessions: int) -> None:
        if max_sessions <= 0:
            raise ValueError("max_sessions must be > 0")
        self._max_sessions: int = max_sessions
        self._lock = threading.RLock()
        self._sessions: OrderedDict[tuple[str, str], OrderedDict[str, bytes]] = OrderedDict()

    @property
    def active_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _key(self, provider: str, session_id: str) -> tuple[str, str]:
        return (provider, session_id)

    def should_inject(self, provider: str, session_id: str) -> bool:
        """Return True when this session has previously injected memory tools."""

        if not provider:
            raise ValueError("provider must be non-empty")
        if not session_id:
            raise ValueError("session_id must be non-empty")
        key = self._key(provider, session_id)
        with self._lock:
            entry = self._sessions.get(key)
            if entry is None:
                return False
            self._sessions.move_to_end(key)
            return len(entry) > 0

    def get_golden_definitions(
        self, provider: str, session_id: str
    ) -> list[tuple[str, bytes]] | None:
        """Return the previously recorded (tool_name, bytes) pairs for a session."""

        if not provider:
            raise ValueError("provider must be non-empty")
        if not session_id:
            raise ValueError("session_id must be non-empty")
        key = self._key(provider, session_id)
        with self._lock:
            entry = self._sessions.get(key)
            if entry is None:
                return None
            self._sessions.move_to_end(key)
            return [(name, golden_bytes) for name, golden_bytes in entry.items()]

    def record_injection(
        self,
        provider: str,
        session_id: str,
        tool_name: str,
        tool_definition_bytes: bytes,
    ) -> None:
        """Record golden bytes for a memory tool in this provider/session."""

        if not provider:
            raise ValueError("provider must be non-empty")
        if not session_id:
            raise ValueError("session_id must be non-empty")
        if not tool_name:
            raise ValueError("tool_name must be non-empty")
        if not tool_definition_bytes:
            raise ValueError("tool_definition_bytes must be non-empty")

        key = self._key(provider, session_id)
        with self._lock:
            entry = self._sessions.get(key)
            if entry is None:
                entry = OrderedDict()
                self._sessions[key] = entry
            if tool_name not in entry:
                entry[tool_name] = tool_definition_bytes
            self._sessions.move_to_end(key)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)

    def reset(self) -> None:
        """Clear all session state."""

        with self._lock:
            self._sessions.clear()
