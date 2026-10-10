"""The buffered-CCR grace window: keep status fidelity *and* liveness (#3079).

A buffered CCR turn holds a request open for the whole generation. Two things
have to be true across that window, and the history here is of each being fixed
by breaking the other:

* #2465 → #2479 added a keepalive so the client's *stream-idle* watchdog saw a
  first byte.
* #2997 removed that keepalive, because committing ``200 text/event-stream``
  before the outcome was known destroyed the real status: a later 429 reached
  the client as a 200 with no ``message_start`` and no ``retry-after``, so
  client backoff never fired.
* Removing it put #2465's condition back — zero bytes for 15-25s turns.

The grace window keeps both: full fidelity while there is still time to send a
real status, a heartbeat once there is not, and a *typed* stream error for a
failure that lands after the commit so backoff still works.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from headroom.proxy.buffered_ccr_response import (
    ANTHROPIC_ERROR_FORMAT,
    OPENAI_ERROR_FORMAT,
    buffered_ccr_asgi_call,
)


class _Recorder:
    """Collects ASGI messages, exposing when the response was committed."""

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.committed = asyncio.Event()

    async def send(self, message: dict) -> None:
        self.messages.append(message)
        if message["type"] == "http.response.start":
            self.committed.set()

    @property
    def status(self) -> int | None:
        for m in self.messages:
            if m["type"] == "http.response.start":
                return m["status"]
        return None

    @property
    def headers(self) -> dict[str, str]:
        for m in self.messages:
            if m["type"] == "http.response.start":
                return {k.decode(): v.decode() for k, v in m.get("headers", [])}
        return {}

    @property
    def body(self) -> bytes:
        return b"".join(
            m.get("body", b"") for m in self.messages if m["type"] == "http.response.body"
        )

    @property
    def pings(self) -> int:
        return self.body.count(b"ping")


class _FakeResult:
    """Stands in for a resolved buffered result (a Starlette response)."""

    def __init__(self, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.status_code = status
        self.body = body
        self._headers = headers or {}

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": [(k.encode(), v.encode()) for k, v in self._headers.items()],
            }
        )
        await send({"type": "http.response.body", "body": self.body, "more_body": False})


class _FakeStreamingResult:
    """A resolved result that is already SSE, like the success path."""

    def __init__(self, chunks: list[bytes], status: int = 200) -> None:
        self.status_code = status
        self._chunks = chunks

    @property
    def body_iterator(self):  # noqa: ANN201
        async def _gen():
            for chunk in self._chunks:
                yield chunk

        return _gen()

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        for chunk in self._chunks:
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})


async def _drive(*, produce, grace: float, fmt=ANTHROPIC_ERROR_FORMAT) -> _Recorder:
    """Run the wrapper against a coroutine standing in for the buffered work."""
    recorder = _Recorder()
    failures: list[str] = []

    async def record_failed(provider: str) -> None:
        failures.append(provider)

    operation = asyncio.create_task(produce())
    call = buffered_ccr_asgi_call(
        operation=operation,
        fmt=fmt,
        grace_seconds=grace,
        record_failed=record_failed,
        request_id="req-test",
    )
    await call({"type": "http"}, None, recorder.send)
    recorder.failures = failures  # type: ignore[attr-defined]
    return recorder


# --------------------------------------------------------------------------- #
# Property 1 — fidelity: anything inside the window keeps its real status
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_a_failure_inside_the_window_keeps_its_status_and_headers() -> None:
    """What #2997 bought. A 429 must arrive as a 429, with retry-after."""

    async def produce():
        await asyncio.sleep(0.05)
        return _FakeResult(
            429,
            b'{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}',
            {"content-type": "application/json", "retry-after": "30"},
        )

    rec = await _drive(produce=produce, grace=5.0)

    assert rec.status == 429
    assert rec.headers.get("retry-after") == "30"
    assert rec.pings == 0, "nothing should have been sent before the outcome was known"


@pytest.mark.asyncio
async def test_a_success_inside_the_window_is_relayed_untouched() -> None:
    async def produce():
        await asyncio.sleep(0.05)
        return _FakeStreamingResult([b'event: message_start\ndata: {"x":1}\n\n'])

    rec = await _drive(produce=produce, grace=5.0)

    assert rec.status == 200
    assert b"message_start" in rec.body
    assert rec.pings == 0


# --------------------------------------------------------------------------- #
# Property 2 — liveness: a slow turn gets a first byte before the ceiling
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_a_slow_success_produces_a_first_byte_before_the_ceiling() -> None:
    """What #2479 bought, and what #2997 removed (#3079).

    The client's stream-idle watchdog cannot distinguish "still generating"
    from "dead socket", so silence for the whole generation is what tripped it.
    """
    release = asyncio.Event()

    async def produce():
        await release.wait()
        return _FakeStreamingResult([b'event: message_start\ndata: {"x":1}\n\n'])

    recorder = _Recorder()

    async def record_failed(provider: str) -> None:  # pragma: no cover - not hit
        raise AssertionError("a success must not be recorded as a failure")

    operation = asyncio.create_task(produce())
    call = buffered_ccr_asgi_call(
        operation=operation,
        fmt=ANTHROPIC_ERROR_FORMAT,
        grace_seconds=0.1,
        record_failed=record_failed,
        request_id="req-slow",
    )
    driver = asyncio.create_task(call({"type": "http"}, None, recorder.send))

    # The first byte must arrive from the heartbeat alone, well before the
    # upstream resolves.
    await asyncio.wait_for(recorder.committed.wait(), timeout=2.0)
    assert recorder.status == 200
    assert recorder.headers["content-type"] == "text/event-stream"
    assert recorder.pings >= 1

    release.set()
    await asyncio.wait_for(driver, timeout=2.0)
    assert b"message_start" in recorder.body


# --------------------------------------------------------------------------- #
# Property 3 — a failure past the commit stays actionable
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_a_late_rate_limit_survives_as_a_typed_stream_error() -> None:
    """The status is gone once committed, so the *meaning* must survive.

    Degrading every post-commit failure to a generic ``api_error`` is what made
    early commits harmful: the client cannot tell a rate limit from a bug, so
    it does not back off. A typed error keeps that behaviour reachable.
    """

    async def produce():
        await asyncio.sleep(0.3)
        return _FakeResult(
            429,
            b'{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}',
            {"content-type": "application/json", "retry-after": "30"},
        )

    rec = await _drive(produce=produce, grace=0.05)

    assert rec.status == 200, "already committed; the status can no longer change"
    assert rec.pings >= 1
    payload = json.loads(rec.body.split(b"event: error\ndata: ")[-1].strip())
    assert payload["error"]["type"] == "rate_limit_error"
    assert payload["error"]["message"] == "slow down"


@pytest.mark.asyncio
async def test_a_late_overload_maps_to_the_overloaded_type() -> None:
    async def produce():
        await asyncio.sleep(0.3)
        return _FakeResult(529, b"", {})

    rec = await _drive(produce=produce, grace=0.05)

    payload = json.loads(rec.body.split(b"event: error\ndata: ")[-1].strip())
    assert payload["error"]["type"] == "overloaded_error"


@pytest.mark.asyncio
async def test_a_late_exception_is_reported_on_the_committed_stream() -> None:
    async def produce():
        await asyncio.sleep(0.3)
        raise RuntimeError("upstream exploded")

    rec = await _drive(produce=produce, grace=0.05)

    assert rec.status == 200
    assert b"event: error" in rec.body
    assert rec.failures == ["anthropic"]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_an_early_exception_still_gets_a_real_502() -> None:
    async def produce():
        raise RuntimeError("upstream exploded")

    rec = await _drive(produce=produce, grace=5.0)

    assert rec.status == 502
    assert json.loads(rec.body)["error"]["type"] == "api_error"


# --------------------------------------------------------------------------- #
# The escape hatch
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_a_zero_grace_window_never_commits_early() -> None:
    """Operators who want #2997's behaviour verbatim can still have it."""

    async def produce():
        await asyncio.sleep(0.3)
        return _FakeResult(429, b"{}", {"retry-after": "30"})

    rec = await _drive(produce=produce, grace=0)

    assert rec.status == 429
    assert rec.headers.get("retry-after") == "30"
    assert rec.pings == 0


# --------------------------------------------------------------------------- #
# OpenAI wire format
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_openai_late_failures_use_the_openai_error_shape() -> None:
    async def produce():
        await asyncio.sleep(0.3)
        return _FakeResult(429, b'{"error":{"message":"too many"}}', {})

    rec = await _drive(produce=produce, grace=0.05, fmt=OPENAI_ERROR_FORMAT)

    assert b": ping" in rec.body, "OpenAI keepalives are SSE comments, not ping events"
    payload = json.loads(rec.body.split(b"data: ")[-1].strip())
    assert payload["error"]["type"] == "rate_limit_error"
    assert payload["error"]["message"] == "too many"
