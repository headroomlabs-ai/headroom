"""Session-scoped state for sticky CCR retrieval tool injection."""

from __future__ import annotations

import threading
from collections import OrderedDict


class SessionCcrTracker:
    """Bounded LRU tracker recording per-provider/session CCR state."""

    def __init__(self, max_sessions: int) -> None:
        if max_sessions <= 0:
            raise ValueError("max_sessions must be > 0")
        self._max_sessions = max_sessions
        self._lock = threading.RLock()
        self._sessions: OrderedDict[tuple[str, str], tuple[bool, bytes | None]] = OrderedDict()

    @property
    def active_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _key(self, provider: str, session_id: str) -> tuple[str, str]:
        return (provider, session_id)

    def has_done_ccr(self, provider: str, session_id: str) -> bool:
        """Return True when this session has previously performed CCR."""

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
            return entry[0]

    def get_golden_tool_bytes(self, provider: str, session_id: str) -> bytes | None:
        """Return recorded golden CCR tool-definition bytes, if any."""

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
            return entry[1]

    def record_ccr_done(
        self,
        provider: str,
        session_id: str,
        golden_tool_bytes: bytes,
    ) -> None:
        """Mark the session as having performed CCR and pin golden tool bytes."""

        if not provider:
            raise ValueError("provider must be non-empty")
        if not session_id:
            raise ValueError("session_id must be non-empty")
        if not golden_tool_bytes:
            raise ValueError("golden_tool_bytes must be non-empty")
        key = self._key(provider, session_id)
        with self._lock:
            existing = self._sessions.get(key)
            if existing is None:
                self._sessions[key] = (True, golden_tool_bytes)
            else:
                pinned = existing[1] if existing[1] is not None else golden_tool_bytes
                self._sessions[key] = (True, pinned)
            self._sessions.move_to_end(key)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)

    def reset(self) -> None:
        """Clear all session state."""

        with self._lock:
            self._sessions.clear()


class SessionExpansionDedupTracker:
    """Bounded LRU tracker for per-session CCR proactive-expansion dedup.

    Prevents the same compressed-content hash from being proactively
    re-injected more than once per session (#2186): a same-messages
    re-request or a continued conversation must not receive duplicate
    expansion blocks for content the agent already saw this session.
    Dedup is per-session, not global — a different conversation may
    legitimately need the same expansion.
    """

    def __init__(self, max_sessions: int) -> None:
        if max_sessions <= 0:
            raise ValueError("max_sessions must be > 0")
        self._max_sessions = max_sessions
        self._lock = threading.RLock()
        self._sessions: OrderedDict[str, set[str]] = OrderedDict()

    def claim(self, session_id: str, hash_keys: list[str]) -> list[str]:
        """Atomically reserve and return the hash_keys still eligible here.

        Selection and recording happen under ONE lock hold: a claimed hash
        is invisible to every other request for this session the moment it
        is handed out. A check-then-act split (select under the lock, record
        after the append) lets two concurrent turns of the same session — a
        normal path for shared-session agents — both select the same hash
        and both append it, which is the duplicate this tracker exists to
        prevent.

        The claim IS the commit; there is no separate commit call. A caller
        that ends up not appending MUST hand the hashes back via
        :meth:`release`, otherwise they stay reserved for the session's
        lifetime. Failing closed that way is deliberate: a dropped release
        costs one missed optimization, a dropped claim costs a permanent
        per-turn token tax (#2186).

        Claiming counts as a use: an active session that keeps being
        *asked* about without reserving anything new must not age out
        behind other sessions, or its already-seen hashes come back and get
        proactively expanded a second time.
        """

        if not session_id:
            raise ValueError("session_id must be non-empty")
        with self._lock:
            seen = self._sessions.get(session_id)
            if seen is None:
                if not hash_keys:
                    return []
                seen = set()
                self._sessions[session_id] = seen
            claimed = [h for h in hash_keys if h not in seen]
            seen.update(claimed)
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
            return claimed

    def release(self, session_id: str, hash_keys: list[str]) -> None:
        """Hand claimed hash_keys back to the eligible pool.

        For every path that claims before knowing whether the append will
        happen: expansion execution returning nothing or a subset,
        cache-mode gating, the already-forwarded target guard, an
        ineligible tail. Only pass hashes THIS request received from
        :meth:`claim` — releasing another request's claim re-opens the
        duplicate window.
        """

        if not session_id:
            raise ValueError("session_id must be non-empty")
        if not hash_keys:
            return
        with self._lock:
            seen = self._sessions.get(session_id)
            if seen is None:
                return
            seen.difference_update(hash_keys)
            self._sessions.move_to_end(session_id)

    def reset(self) -> None:
        """Clear all session state."""

        with self._lock:
            self._sessions.clear()
