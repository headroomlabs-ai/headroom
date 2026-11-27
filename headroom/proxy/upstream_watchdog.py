"""A deadline on the upstream waits httpx's own timeouts do not cover.

Why this exists
---------------
httpx's ``connect``/``read``/``write``/``pool`` timeouts bound socket
operations and the wait for a pooled connection. They do not bound the
locks and semaphores httpcore's HTTP/2 connection waits on, and all of a
provider's traffic shares one HTTP/2 connection. In httpcore 1.0.9:

* ``AsyncHTTP2Connection.handle_async_request`` waits for a stream slot on
  ``_max_streams_semaphore`` (``httpcore/_async/http2.py:131``) with no
  timeout. A slot is freed only when a response is closed (``_response_closed``,
  l.409), and the reader shrinking the semaphore after the server lowers
  ``MAX_CONCURRENT_STREAMS`` does so while holding ``_read_lock``
  (l.405-407), so every other stream's read then waits on that lock.
* ``is_available()`` (l.511) ignores stream usage, so the pool keeps routing
  new requests to the jammed connection instead of opening another one.

The proxy then stops answering without crashing: every request parks inside
httpcore, the pre-upstream queue fills, and nothing fails over. Only a
restart recovers it.

What this does
--------------
:class:`UpstreamWatchdogTransport` wraps each transport of the shared upstream
clients and puts an ``asyncio`` deadline on three phases of every request,
tracked through httpcore's ``trace`` extension:

1. **send**: until the request headers are written (pool wait, connect, TLS,
   the HTTP/2 stream slot). Bounded by ``upstream_watchdog_seconds``, raised to
   the request's own ``pool + connect`` timeouts when those are longer. A
   timeout here is unambiguous: nothing upstream can be slow before the request
   reached it, so the client is recycled (see below).
2. **response**: from then until the response headers arrive. A buffered
   request legitimately waits here for the whole generation, so the bound is
   the request's ``write + read`` timeouts plus a margin: httpx's own
   ``ReadTimeout`` fires first whenever it can, and this only catches waits
   httpcore leaves unbounded.
3. **body**: each chunk of the response body, bounded by ``read`` plus the
   margin. This is the streaming first-chunk (and every-chunk) deadline; it
   never bounds the total length of a stream.

Each expiry raises :class:`UpstreamWatchdogTimeout`, an
``httpx.TimeoutException``, so the existing retry and error paths handle it
(it classifies as ``upstream_timeout``), and handlers answer it with 504.

Recycling
---------
A send-phase expiry calls ``on_send_timeout(generation)``. The proxy swaps in
fresh clients once per generation: the first expiry on the current generation
swaps, later ones on the same (now retired) generation are no-ops, so many
requests timing out together cause one swap, not a storm.
:class:`UpstreamClientRetirer` then closes the old clients in the background:
it waits until their in-flight requests and open response streams drain, or
``RETIRE_GRACE_SECONDS`` pass, then calls ``aclose()`` under
``CLOSE_TIMEOUT_SECONDS`` and abandons it if that hangs too. Healthy streams on
the old connection therefore finish; a stream still open after the grace
period is cut, which is the trade-off for not keeping a jammed pool forever.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_UPSTREAM_WATCHDOG_SECONDS = 60
# Added to the request's own timeouts so httpx's native error wins the race
# whenever httpx can detect the stall itself.
NATIVE_TIMEOUT_MARGIN_SECONDS = 5.0
RETIRE_GRACE_SECONDS = 900.0
CLOSE_TIMEOUT_SECONDS = 10.0

PHASE_SEND = "send"
PHASE_RESPONSE = "response"
PHASE_BODY = "body"


class UpstreamWatchdogTimeout(httpx.TimeoutException):
    """An upstream wait outlived the watchdog deadline for its phase."""

    def __init__(self, message: str, *, request: httpx.Request, phase: str) -> None:
        super().__init__(message, request=request)
        self.phase = phase


def is_watchdog_timeout(exc: BaseException) -> bool:
    return isinstance(exc, UpstreamWatchdogTimeout)


def gateway_timeout_status(exc: BaseException, default: int) -> int:
    """504 for a watchdog expiry, else the status the call site already used."""
    return 504 if is_watchdog_timeout(exc) else default


class UpstreamClientGeneration:
    """One generation of the shared upstream clients and its in-flight count.

    A request counts from the moment it enters the transport until its response
    stream is closed (or it fails), which is also when httpcore frees its
    HTTP/2 stream slot.
    """

    def __init__(self, number: int) -> None:
        self.number = number
        self.in_flight = 0
        self._idle: asyncio.Event | None = None

    def enter(self) -> None:
        self.in_flight += 1

    def leave(self) -> None:
        self.in_flight -= 1
        if self.in_flight <= 0 and self._idle is not None:
            self._idle.set()

    async def wait_idle(self) -> None:
        if self.in_flight <= 0:
            return
        if self._idle is None:
            self._idle = asyncio.Event()
        await self._idle.wait()


def _timeout_value(request: httpx.Request, key: str) -> float | None:
    timeout = request.extensions.get("timeout")
    if not isinstance(timeout, dict):
        return None
    value = timeout.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _sum_or_none(*values: float | None) -> float | None:
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


class _WatchedResponseStream(httpx.AsyncByteStream):
    def __init__(
        self,
        inner: httpx.AsyncByteStream,
        *,
        request: httpx.Request,
        chunk_budget: float | None,
        generation: UpstreamClientGeneration,
    ) -> None:
        self._inner = inner
        self._request = request
        self._chunk_budget = chunk_budget
        self._generation = generation
        self._closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        iterator = self._inner.__aiter__()
        while True:
            if self._chunk_budget is None:
                try:
                    chunk = await iterator.__anext__()
                except StopAsyncIteration:
                    return
            else:
                deadline = asyncio.timeout(self._chunk_budget)
                try:
                    async with deadline:
                        chunk = await iterator.__anext__()
                except StopAsyncIteration:
                    return
                except TimeoutError as exc:
                    if not deadline.expired():
                        raise
                    raise UpstreamWatchdogTimeout(
                        f"Upstream response body stalled for {self._chunk_budget:.0f}s",
                        request=self._request,
                        phase=PHASE_BODY,
                    ) from exc
            yield chunk

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._inner.aclose()
        finally:
            self._generation.leave()


class UpstreamWatchdogTransport(httpx.AsyncBaseTransport):
    """Deadline-bounded wrapper around one transport of a shared client."""

    def __init__(
        self,
        inner: httpx.AsyncBaseTransport,
        *,
        send_deadline: float,
        generation: UpstreamClientGeneration,
        on_send_timeout: Callable[[UpstreamClientGeneration], None],
    ) -> None:
        self._inner = inner
        self._send_deadline = send_deadline
        self._generation = generation
        self._on_send_timeout = on_send_timeout

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        loop = asyncio.get_running_loop()
        native_send = _sum_or_none(
            _timeout_value(request, "pool"), _timeout_value(request, "connect")
        )
        send_budget = (
            self._send_deadline
            if native_send is None
            else max(self._send_deadline, native_send + NATIVE_TIMEOUT_MARGIN_SECONDS)
        )
        response_budget = _sum_or_none(
            _timeout_value(request, "write"),
            _timeout_value(request, "read"),
            NATIVE_TIMEOUT_MARGIN_SECONDS,
        )
        chunk_budget = _sum_or_none(_timeout_value(request, "read"), NATIVE_TIMEOUT_MARGIN_SECONDS)

        deadline = asyncio.timeout(send_budget)
        phase = PHASE_SEND
        chained = request.extensions.get("trace")

        async def trace(event: str, info: dict[str, Any]) -> None:
            nonlocal phase
            if phase == PHASE_SEND and event.endswith(".send_request_headers.complete"):
                phase = PHASE_RESPONSE
                deadline.reschedule(
                    None if response_budget is None else loop.time() + response_budget
                )
            if chained is not None:
                await chained(event, info)

        request.extensions = {**request.extensions, "trace": trace}
        generation = self._generation
        generation.enter()
        try:
            async with deadline:
                response = await self._inner.handle_async_request(request)
        except TimeoutError as exc:
            generation.leave()
            if not deadline.expired():
                raise
            if phase == PHASE_SEND:
                logger.warning(
                    "Upstream watchdog: no stream to %s after %.0fs; recycling the "
                    "upstream client (generation %d)",
                    request.url.host,
                    send_budget,
                    generation.number,
                )
                try:
                    self._on_send_timeout(generation)
                except Exception:
                    logger.exception("Upstream watchdog: recycling the upstream client failed")
                message = f"Upstream request could not be sent within {send_budget:.0f}s"
            else:
                message = f"Upstream response headers did not arrive within {response_budget:.0f}s"
            raise UpstreamWatchdogTimeout(message, request=request, phase=phase) from exc
        except BaseException:
            generation.leave()
            raise
        assert isinstance(response.stream, httpx.AsyncByteStream)
        response.stream = _WatchedResponseStream(
            response.stream,
            request=request,
            chunk_budget=chunk_budget,
            generation=generation,
        )
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()

    def __getattr__(self, name: str) -> Any:
        # Keep the wrapped transport's attributes (its ``_pool``, say) reachable,
        # as they were before the wrap.
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


def install_upstream_watchdog(
    client: httpx.AsyncClient,
    *,
    send_deadline: float,
    generation: UpstreamClientGeneration,
    on_send_timeout: Callable[[UpstreamClientGeneration], None],
) -> httpx.AsyncClient:
    """Wrap the client's primary and mounted transports; ``send_deadline <= 0`` is off."""
    if send_deadline <= 0:
        return client

    def wrap(transport: httpx.AsyncBaseTransport) -> UpstreamWatchdogTransport:
        return UpstreamWatchdogTransport(
            transport,
            send_deadline=send_deadline,
            generation=generation,
            on_send_timeout=on_send_timeout,
        )

    client._transport = wrap(client._transport)
    mounts = getattr(client, "_mounts", None)
    if mounts:
        for pattern, transport in list(mounts.items()):
            if transport is not None:  # None defers to the primary transport
                mounts[pattern] = wrap(transport)
    return client


async def close_upstream_client(
    client: httpx.AsyncClient, *, timeout: float = CLOSE_TIMEOUT_SECONDS
) -> None:
    """``aclose()`` that gives up after ``timeout`` instead of hanging with the pool."""
    try:
        await asyncio.wait_for(client.aclose(), timeout)
    except TimeoutError:
        logger.warning("Upstream client did not close within %.0fs; abandoning it", timeout)
    except Exception:
        logger.warning("Upstream client close failed", exc_info=True)


class UpstreamClientRetirer:
    """Closes recycled upstream clients in the background, never inline."""

    def __init__(
        self,
        *,
        grace_seconds: float = RETIRE_GRACE_SECONDS,
        close_timeout: float = CLOSE_TIMEOUT_SECONDS,
    ) -> None:
        self.grace_seconds = grace_seconds
        self.close_timeout = close_timeout
        self._tasks: dict[asyncio.Task[None], list[httpx.AsyncClient]] = {}

    @property
    def pending(self) -> int:
        return len(self._tasks)

    def retire(
        self, clients: list[httpx.AsyncClient], generation: UpstreamClientGeneration
    ) -> asyncio.Task[None]:
        task = asyncio.get_running_loop().create_task(self._retire(clients, generation))
        self._tasks[task] = clients
        task.add_done_callback(lambda done: self._tasks.pop(done, None))
        return task

    async def _retire(
        self, clients: list[httpx.AsyncClient], generation: UpstreamClientGeneration
    ) -> None:
        try:
            await asyncio.wait_for(generation.wait_idle(), self.grace_seconds)
        except TimeoutError:
            logger.warning(
                "Upstream watchdog: closing generation %d with %d request(s) still open "
                "after %.0fs",
                generation.number,
                generation.in_flight,
                self.grace_seconds,
            )
        for client in clients:
            await close_upstream_client(client, timeout=self.close_timeout)

    async def shutdown(self) -> None:
        """Stop waiting for drains and close every retired client, bounded."""
        pending = dict(self._tasks)
        self._tasks.clear()
        for task, clients in pending.items():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            for client in clients:
                await close_upstream_client(client, timeout=self.close_timeout)
