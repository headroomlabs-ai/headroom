"""Request coalescer — merge rapid compatible prompts into batched requests.

When an agent sends multiple *identical* non-streaming requests within a
configurable window (default 5s), buffer them by compatibility key
(the complete request: model, system, tools, generation parameters,
messages, and auth scope) and dispatch once through the configured
upstream sender.

Only requests whose complete inputs are equivalent may share a dispatch.
Two requests under one credential are still two independent sessions: a
shared model/settings/auth scope does not make their answers
interchangeable, and one upstream response cannot be demultiplexed into
per-caller answers. Deduplication therefore requires an identical payload.

Benefits:
- Reduces API call overhead for duplicate prompts (retry storms, fan-out
  of the same question)
- Never substitutes one caller's answer for another's
- Dispatch lifetime is independent of any single waiter
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DispatchFn = Callable[[dict[str, Any]], Awaitable[Any]]

_INTERNAL_KEYS = frozenset({"_coalesce_scope"})


def _canonical_json(value: Any) -> str:
    """Deterministic serialization for request compatibility & prefix comparison."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return repr(value)


def _compatibility_key(payload: dict[str, Any], scope: str | None = None) -> str:
    """Compute a compatibility key from the complete request inputs + scope.

    Requests only batch together when *every* field matches, messages
    included. Matching settings and auth scope do not make two prompts
    interchangeable, so the conversation is part of the identity.
    """
    fields = {k: v for k, v in payload.items() if k not in _INTERNAL_KEYS}
    effective_scope = scope if scope is not None else str(payload.get("_coalesce_scope", ""))
    return _canonical_json({"scope": effective_scope, "fields": fields})


def _strip_internal_keys(payload: dict[str, Any]) -> dict[str, Any]:
    if not any(k in payload for k in _INTERNAL_KEYS):
        return dict(payload)
    return {k: v for k, v in payload.items() if k not in _INTERNAL_KEYS}


@dataclass
class CoalescedRequest:
    """A request waiting to be batched."""

    payload: dict[str, Any]
    scope: str | None = None
    dispatch_fn: DispatchFn | None = None
    future: asyncio.Future[dict[str, Any]] = field(init=False)
    arrived_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.future = asyncio.get_running_loop().create_future()


class RequestCoalescer:
    """Buffer rapid identical requests and merge them into batches.

    Requests are grouped by their complete inputs (settings *and* messages)
    plus auth scope, so a batch only ever contains the same question asked
    more than once. Dispatch runs in its own task: it is never owned by —
    or cancellable through — any single waiting caller.
    """

    def __init__(
        self,
        window_sec: float = 5.0,
        max_batch: int = 5,
        dispatch_fn: DispatchFn | None = None,
    ):
        self.window_sec = window_sec
        self.max_batch = max_batch
        self.dispatch_fn = dispatch_fn
        self._pending: list[CoalescedRequest] = []
        self._lock = asyncio.Lock()
        self._timer_task: asyncio.Task[None] | None = None
        self._flushing: set[asyncio.Task[None]] = set()
        self._stats: dict[str, int] = {
            "requests_submitted": 0,
            "requests_coalesced": 0,
            "batches_flushed": 0,
            "api_calls_saved": 0,
            "cancelled_requests": 0,
            "dispatch_errors": 0,
        }

    @property
    def _flush_task(self) -> asyncio.Task[None] | None:
        """Most recent in-flight flush task, if any (observability/compat)."""
        for task in self._flushing:
            if not task.done():
                return task
        return None

    async def __aenter__(self) -> RequestCoalescer:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.close()

    def get_stats(self) -> dict[str, int]:
        """Return a snapshot of coalescer operational counters."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset coalescer operational counters."""
        for key in self._stats:
            self._stats[key] = 0

    async def submit(
        self,
        payload: dict[str, Any],
        *,
        scope: str | None = None,
        dispatch_fn: DispatchFn | None = None,
    ) -> dict[str, Any]:
        """Submit a request. May be batched with identical rapid requests."""
        req = CoalescedRequest(payload=payload, scope=scope, dispatch_fn=dispatch_fn)

        async with self._lock:
            self._stats["requests_submitted"] += 1
            self._pending.append(req)
            active_count = sum(1 for r in self._pending if not r.future.done())

            if active_count >= self.max_batch:
                # Batch is full — flush in a task of its own. Never await the
                # dispatch here: the caller's task must not own, or be able
                # to cancel, a batch its peers are waiting on.
                self._start_flush_locked()
            elif self._timer_task is None or self._timer_task.done():
                self._timer_task = asyncio.create_task(self._delayed_flush())

        try:
            return await req.future
        except asyncio.CancelledError:
            async with self._lock:
                if req in self._pending:
                    self._pending.remove(req)
                    self._stats["cancelled_requests"] += 1
                if not any(not r.future.done() for r in self._pending):
                    self._pending.clear()
                    if self._timer_task is not None and not self._timer_task.done():
                        self._timer_task.cancel()
                    self._timer_task = None
            raise

    def _start_flush_locked(self) -> None:
        """Schedule a flush as an independent task. Caller must hold the lock."""
        task = asyncio.create_task(self._flush_worker())
        self._flushing.add(task)
        task.add_done_callback(self._flushing.discard)
        if self._timer_task is not None and not self._timer_task.done():
            self._timer_task.cancel()
            self._timer_task = None

    async def _delayed_flush(self) -> None:
        """Wait for window_sec, then flush pending requests."""
        try:
            await asyncio.sleep(self.window_sec)
        except asyncio.CancelledError:
            return
        await self._flush_worker()

    async def _flush_worker(self) -> None:
        """Snapshot pending under the lock, then dispatch outside it.

        Holds the lock only long enough to take ownership of the queue, so a
        slow upstream never blocks new submits, and a cancelled waiter never
        cancels the shared dispatch.
        """
        try:
            async with self._lock:
                snapshot = self._take_pending_locked()
            if not snapshot:
                return

            groups: OrderedDict[str, list[CoalescedRequest]] = OrderedDict()
            for req in snapshot:
                groups.setdefault(_compatibility_key(req.payload, req.scope), []).append(req)

            for group in groups.values():
                await self._flush_compatible_group(group)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive
            logger.exception("coalescer flush failed")

    def _take_pending_locked(self) -> list[CoalescedRequest]:
        """Swap out the pending queue, dropping callers that already left."""
        raw_pending = self._pending
        self._pending = []
        active: list[CoalescedRequest] = []
        for req in raw_pending:
            if req.future.done():
                self._stats["cancelled_requests"] += 1
            else:
                active.append(req)
        return active

    async def _flush(self) -> None:
        """Drain pending now (serialized close path)."""
        await self._flush_worker()

    async def _flush_compatible_group(self, group: list[CoalescedRequest]) -> None:
        """Dispatch a single compatible group and resolve its waiters.

        The group is keyed on complete-input identity, so every member asked
        the same question: the one upstream answer is the correct answer for
        each of them. No merging of distinct conversations happens here.
        """
        live_group = [r for r in group if not r.future.done()]
        if not live_group:
            return

        self._stats["batches_flushed"] += 1
        is_coalesced = len(live_group) > 1
        outbound_payload = _strip_internal_keys(live_group[0].payload)
        if is_coalesced:
            self._stats["requests_coalesced"] += len(live_group)
            self._stats["api_calls_saved"] += len(live_group) - 1
            logger.info(
                "coalesced %d identical requests into 1 batch (saved %d API calls)",
                len(live_group),
                len(live_group) - 1,
            )

        effective_dispatch = live_group[0].dispatch_fn or self.dispatch_fn
        upstream_response: Any = None
        if effective_dispatch is not None:
            try:
                upstream_response = await effective_dispatch(outbound_payload)
            except Exception as exc:
                self._stats["dispatch_errors"] += 1
                for req in live_group:
                    if not req.future.done():
                        req.future.set_exception(exc)
                return

        batch_size = len(live_group)
        for idx, req in enumerate(live_group):
            if req.future.done():
                self._stats["cancelled_requests"] += 1
                continue
            result_envelope: dict[str, Any] = {
                "coalesced": is_coalesced,
                "batch_size": batch_size,
                "request_index": idx,
                "payload": outbound_payload,
            }
            if effective_dispatch is not None:
                result_envelope["response"] = upstream_response
            req.future.set_result(result_envelope)

    async def close(self) -> None:
        """Flush any remaining requests and shut down."""
        async with self._lock:
            if self._timer_task is not None:
                self._timer_task.cancel()
                self._timer_task = None
        await self._flush_worker()
        in_flight = [t for t in self._flushing if not t.done()]
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)


# ── Singleton & Proxy Integration Helpers ────────────────────────────
_default_coalescer: RequestCoalescer | None = None


def coalescing_enabled(headers: dict[str, str] | Any | None = None) -> bool:
    """Check whether request coalescing is enabled via env var or request header."""
    if headers is not None:
        header_val = ""
        if hasattr(headers, "get"):
            header_val = str(
                headers.get("x-headroom-coalesce") or headers.get("X-Headroom-Coalesce") or ""
            ).strip()
        if header_val.lower() in ("1", "true", "yes", "on"):
            return True
        if header_val.lower() in ("0", "false", "no", "off"):
            return False
    env_val = os.environ.get("HEADROOM_COALESCE_REQUESTS", "").strip().lower()
    return env_val in ("1", "true", "yes", "on")


def get_coalescer(
    window_sec: float = 5.0,
    max_batch: int = 5,
    dispatch_fn: DispatchFn | None = None,
) -> RequestCoalescer:
    """Get or create the default coalescer."""
    global _default_coalescer
    if _default_coalescer is None:
        _default_coalescer = RequestCoalescer(
            window_sec=window_sec,
            max_batch=max_batch,
            dispatch_fn=dispatch_fn,
        )
    elif dispatch_fn is not None and _default_coalescer.dispatch_fn is None:
        _default_coalescer.dispatch_fn = dispatch_fn
    return _default_coalescer


def get_coalescer_stats() -> dict[str, int]:
    """Return stats for the default coalescer without instantiating if unused."""
    if _default_coalescer is None:
        return {
            "requests_submitted": 0,
            "requests_coalesced": 0,
            "batches_flushed": 0,
            "api_calls_saved": 0,
            "cancelled_requests": 0,
            "dispatch_errors": 0,
        }
    return _default_coalescer.get_stats()


def reset_coalescer() -> None:
    """Reset the default coalescer singleton and its stats."""
    global _default_coalescer
    if _default_coalescer is not None:
        _default_coalescer.reset_stats()
    _default_coalescer = None
