"""WebSocket session registry for Codex relay lifecycle tracking.

Unit 3 of the Codex-proxy resilience plan. Every ``/v1/responses`` WS
session is explicitly registered on accept and deregistered in the
outermost ``finally`` of the handler. The registry provides:

* First-class visibility of active sessions (``/debug/ws-sessions``
  from Unit 5 consumes :meth:`WebSocketSessionRegistry.snapshot`).
* Gauges for Prometheus (``active_ws_sessions``, ``active_relay_tasks``).
* A home for relay-task references so the handler's orchestrator can
  attach them at creation time for introspection, without the registry
  itself owning or cancelling them — cancellation is the handler's job.

Design notes
------------
* Single-event-loop usage. All mutations happen on the proxy's event
  loop from within the handler coroutine, so no asyncio.Lock / asyncio
  primitives are needed. Python dict mutations are atomic under the
  GIL; snapshot copies happen under the event loop's cooperative
  scheduling, so iteration + mutation cannot interleave across an
  ``await`` point.
* ``deregister`` must be idempotent. The handler calls it from the
  outermost ``finally`` — if upstream never connected, the session may
  have been registered or may not have been, depending on how far
  handshake got. Either way ``deregister`` is safe.
* ``deregister`` clears the handle's ``relay_tasks`` list so the
  registry does not retain references to task coroutine frames after a
  session ends.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

logger = logging.getLogger(__name__)

__all__ = [
    "WebSocketSessionRegistry",
    "WSSessionHandle",
    "TerminationCause",
]


TerminationCause = Literal[
    "client_disconnect",
    "upstream_disconnect",
    "upstream_error",
    "client_error",
    "client_cancel",
    "response_completed",
    "client_timeout",
    "unknown",
]


class _TaskLike(Protocol):
    def done(self) -> bool: ...
    def cancel(self) -> bool: ...
    def get_name(self) -> str: ...


@dataclass
class WSSessionHandle:
    """Per-session state entry held in :class:`WebSocketSessionRegistry`.

    Timestamps use :func:`time.perf_counter` so age computations are
    monotonic and independent of wall-clock adjustments.
    """

    session_id: str
    request_id: str
    client_addr: str | None = None
    upstream_url: str | None = None
    started_at: float = field(default_factory=time.perf_counter)
    last_activity_at: float = field(default_factory=time.perf_counter)
    relay_tasks: list[_TaskLike] = field(default_factory=list)
    termination_cause: TerminationCause | None = None

    def mark_activity(self) -> None:
        self.last_activity_at = time.perf_counter()

    def age_seconds(self) -> float:
        return max(0.0, time.perf_counter() - self.started_at)

    def to_snapshot_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "request_id": self.request_id,
            "client_addr": self.client_addr,
            "upstream_url": self.upstream_url,
            "age_seconds": self.age_seconds(),
            "idle_seconds": max(0.0, time.perf_counter() - self.last_activity_at),
            "relay_task_count": len(self.relay_tasks),
            "relay_task_names": [t.get_name() for t in self.relay_tasks],
            "termination_cause": self.termination_cause,
        }


class WebSocketSessionRegistry:
    """In-memory registry of active Codex WS sessions.

    Methods are safe to call from the event loop. Unit 5's
    ``/debug/ws-sessions`` endpoint consumes :meth:`snapshot`; the
    Prometheus exporter reads :meth:`active_count` and
    :meth:`active_relay_task_count`.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, WSSessionHandle] = {}
        # Tracked separately from ``sum(len(h.relay_tasks))`` so that
        # repeated snapshotting is O(1) rather than O(N * tasks).
        self._active_relay_tasks = 0

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    def register(self, handle: WSSessionHandle) -> None:
        """Register a session. Idempotent by ``session_id``.

        If the session id is already present, the existing entry is
        replaced and the active count stays the same. This matches the
        invariant that one ``session_id`` corresponds to at most one
        active session at a time.
        """
        existing = self._sessions.get(handle.session_id)
        if existing is not None:
            # Re-registration: release any tasks we were tracking on the
            # old handle before replacing it.
            self._active_relay_tasks -= len(existing.relay_tasks)
            existing.relay_tasks.clear()
        self._sessions[handle.session_id] = handle
        self._active_relay_tasks += len(handle.relay_tasks)

    def deregister(
        self, session_id: str, cause: TerminationCause = "unknown"
    ) -> WSSessionHandle | None:
        """Remove a session. Idempotent: returns ``None`` if unknown.

        Also clears the handle's ``relay_tasks`` list so the registry
        stops holding references to coroutine frames after the session
        ends.

        The handle is returned with ``relay_tasks`` already cleared, but
        the caller can still observe the number of tasks the registry
        released via the ``released_tasks`` attribute set on the handle
        before the clear. Prefer :meth:`deregister_and_count` for new
        callers that need that number for Prometheus dec calls — it
        couples the handle pop and the count read so they cannot drift.
        """
        handle, _count = self._deregister_internal(session_id, cause)
        return handle

    def deregister_and_count(
        self, session_id: str, cause: TerminationCause = "unknown"
    ) -> tuple[WSSessionHandle | None, int]:
        """Deregister and return (handle, released_task_count).

        This is the preferred API for handlers that need to decrement a
        Prometheus gauge by the number of relay tasks that were attached
        to the session: capturing the count separately (via
        ``len(handle.relay_tasks)`` before :meth:`deregister`) and then
        calling :meth:`deregister` risks drift if the registry's
        bookkeeping is ever changed. ``deregister_and_count`` returns
        both atomically.

        Returns ``(None, 0)`` when the session was not registered.
        """
        return self._deregister_internal(session_id, cause)

    def _deregister_internal(
        self, session_id: str, cause: TerminationCause
    ) -> tuple[WSSessionHandle | None, int]:
        handle = self._sessions.pop(session_id, None)
        if handle is None:
            return None, 0
        handle.termination_cause = cause
        released = len(handle.relay_tasks)
        self._active_relay_tasks = max(0, self._active_relay_tasks - released)
        handle.relay_tasks.clear()
        return handle, released

    def attach_tasks(self, session_id: str, tasks: Iterable[_TaskLike]) -> None:
        """Attach relay tasks to an existing session (merge, not replace).

        If ``session_id`` is not registered, this is a no-op (handler
        should have registered before spawning tasks; defensive so a
        race during deregister doesn't crash the handler).
        """
        handle = self._sessions.get(session_id)
        if handle is None:
            return
        task_list = list(tasks)
        handle.relay_tasks.extend(task_list)
        self._active_relay_tasks += len(task_list)
        handle.mark_activity()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get(self, session_id: str) -> WSSessionHandle | None:
        return self._sessions.get(session_id)

    def active_count(self) -> int:
        return len(self._sessions)

    def active_relay_task_count(self) -> int:
        return self._active_relay_tasks

    def snapshot(self) -> list[dict[str, Any]]:
        """JSON-serializable view of the registry (for ``/debug/ws-sessions``)."""
        return [handle.to_snapshot_dict() for handle in self._sessions.values()]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._sessions

    def __len__(self) -> int:
        return len(self._sessions)
