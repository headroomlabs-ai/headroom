"""Request coalescer — merge rapid short prompts into batched requests.

When an agent sends multiple short tool calls within a configurable window
(default 5s), instead of making N separate API calls (each with system prompt
overhead), buffer them and send one merged request.

Benefits:
- Reduces API call overhead (N calls → 1 call)
- System prompt prefix is sent once instead of N times
- Agents often fire rapid tool calls that can be batched

Usage:
    coalescer = RequestCoalescer(window_sec=5.0, max_batch=5)
    async with coalescer:
        result = await coalescer.submit(payload)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CoalescedRequest:
    """A request waiting to be batched."""
    payload: dict[str, Any]
    future: asyncio.Future = field(default_factory=asyncio.Future)
    arrived_at: float = field(default_factory=time.monotonic)


class RequestCoalescer:
    """Buffer rapid requests and merge them into batches.

    When requests arrive within `window_sec` of each other, they're
    held and merged into a single batch. If no other request arrives
    within the window, the solo request is sent immediately.
    """

    def __init__(self, window_sec: float = 5.0, max_batch: int = 5):
        self.window_sec = window_sec
        self.max_batch = max_batch
        self._pending: list[CoalescedRequest] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None

    async def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit a request. May be batched with other rapid requests."""
        req = CoalescedRequest(payload=payload)
        
        async with self._lock:
            self._pending.append(req)
            
            if len(self._pending) >= self.max_batch:
                # Batch is full — flush immediately
                await self._flush()
            elif self._flush_task is None:
                # Start the flush timer
                self._flush_task = asyncio.create_task(self._delayed_flush())
        
        return await req.future

    async def _delayed_flush(self):
        """Wait for window_sec, then flush pending requests."""
        await asyncio.sleep(self.window_sec)
        async with self._lock:
            await self._flush()

    async def _flush(self):
        """Merge all pending requests into one and resolve futures."""
        if self._flush_task:
            self._flush_task.cancel()
            self._flush_task = None

        if not self._pending:
            return

        pending = self._pending
        self._pending = []

        if len(pending) == 1:
            # Solo request — no batching needed
            # In production: send to upstream API
            pending[0].future.set_result({"coalesced": False, "payload": pending[0].payload})
            return

        # Merge multiple requests
        merged = self._merge_requests([r.payload for r in pending])
        
        logger.info(
            "coalesced %d requests into 1 batch (saved %d API calls)",
            len(pending), len(pending) - 1,
        )

        # In production: send merged payload to upstream API
        # For now: return merged payload to all futures
        for req in pending:
            req.future.set_result({
                "coalesced": True,
                "batch_size": len(pending),
                "payload": merged,
            })

    def _merge_requests(self, payloads: list[dict]) -> dict:
        """Merge multiple Anthropic-format payloads into one."""
        if not payloads:
            return {}

        # Use the first payload as base
        merged = dict(payloads[0])
        
        # Merge messages from all payloads
        all_messages = []
        seen = set()
        for p in payloads:
            for msg in p.get("messages", []):
                # Deduplicate identical messages
                key = str(msg)
                if key not in seen:
                    seen.add(key)
                    all_messages.append(msg)

        merged["messages"] = all_messages
        merged["_coalesced_from"] = len(payloads)
        
        return merged

    async def close(self):
        """Flush any remaining requests and shut down."""
        async with self._lock:
            if self._flush_task:
                self._flush_task.cancel()
            await self._flush()


# ── Singleton ────────────────────────────────────────────────────────
_default_coalescer: RequestCoalescer | None = None


def get_coalescer(window_sec: float = 5.0, max_batch: int = 5) -> RequestCoalescer:
    """Get or create the default coalescer."""
    global _default_coalescer
    if _default_coalescer is None:
        _default_coalescer = RequestCoalescer(window_sec=window_sec, max_batch=max_batch)
    return _default_coalescer
