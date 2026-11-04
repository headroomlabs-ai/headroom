"""Request-scoped ML ceiling for the mixed-content path (#3711).

``HEADROOM_KOMPRESS_MAX_TOKENS`` bounds a single block. It cannot bound a
request: ``_compress_mixed`` splits a payload into sections and calls
``_try_ml_compressor`` once per section, so every section can sit under the
per-block ceiling while their sum runs for minutes. The reporter measured ~70s
on a 1.4MB ``tool_result`` of prose blocks separated by small JSON objects,
which blew the 30s compression budget and then quarantined compression for
every following request -- while
``headroom_kompress_size_gate_total`` recorded only ``within``.

A stub stands in for Kompress so these assert the *budget*, not ONNX latency:
the real model is not available in CI, and the bug is about how many times a
slow stage is entered, not how slow it is.
"""

from __future__ import annotations

import concurrent.futures
import json
import random
import threading
import time
from dataclasses import dataclass

import httpx
import pytest

from headroom.transforms import content_router as cr
from headroom.transforms.kompress_remote import RemoteKompressCompressor


class _Tokenizer:
    def count_text(self, content: str) -> int:
        return len(content.split())


@dataclass
class _StubResult:
    compressed: str
    compressed_tokens: int


class _SlowKompress:
    """Stands in for Kompress: ready, and costs `per_call_s` every call."""

    def __init__(self, per_call_s: float) -> None:
        self.per_call_s = per_call_s
        self.calls = 0

    def is_ready(self) -> bool:
        return True

    def ensure_background_load(self) -> None:  # pragma: no cover - never reached
        raise AssertionError("stub is always ready")

    def compress(self, text: str, **_kwargs: object) -> _StubResult:
        self.calls += 1
        time.sleep(self.per_call_s)
        out = text[: max(1, len(text) // 2)]
        return _StubResult(compressed=out, compressed_tokens=cr._estimate_tokens(out))


def _prose(n_chars: int, seed: int = 7) -> str:
    rng = random.Random(seed)
    words = (
        "analysis deployment configuration throughput latency pipeline compression "
        "artifact identifier resolution boundary telemetry inference workspace"
    ).split()
    out: list[str] = []
    total = 0
    while total < n_chars:
        line = " ".join(rng.choice(words) for _ in range(14))
        out.append(line)
        total += len(line) + 1
    return "\n".join(out)


def _reporter_payload(block_chars: int = 40_000, blocks: int = 8) -> str:
    """Prose blocks separated by small JSON objects -- the #3711 shape.

    Each block stays under the 50k-token per-block gate, so the size gate
    cannot fire; only a request-scoped ceiling can stop this.
    """
    parts: list[str] = []
    for i in range(blocks):
        parts.append(_prose(block_chars, seed=i))
        parts.append(json.dumps({"id": i, "status": "ok", "note": f"marker {i}"}))
    return "\n\n".join(parts)


def _tool_result_messages() -> list[dict]:
    """The reported shape: the payload arrives as a tool_result, not user text.

    Plain user text is protected from compression
    (``router:protected:user_message``) and never reaches the ML stage at all.
    """
    return [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": _reporter_payload()}
            ],
        },
    ]


@pytest.fixture
def gate_outcomes(monkeypatch) -> dict[str, int]:
    seen: dict[str, int] = {}
    monkeypatch.setattr(
        cr.ContentRouter,
        "_observe_kompress_size_gate",
        lambda self, outcome: seen.__setitem__(outcome, seen.get(outcome, 0) + 1),
    )
    return seen


@pytest.fixture
def slow_kompress(monkeypatch) -> _SlowKompress:
    stub = _SlowKompress(per_call_s=0.05)
    monkeypatch.setattr(cr.ContentRouter, "_get_kompress", lambda self: stub)
    return stub


def _apply(router: cr.ContentRouter) -> None:
    router.apply(
        _tool_result_messages(),
        _Tokenizer(),
        frozen_message_count=1,
        min_tokens_to_compress=1,
    )


def test_size_gate_alone_cannot_bound_a_request(gate_outcomes, slow_kompress, monkeypatch) -> None:
    """Every section passes the per-block gate -- this is the #3711 precondition.

    Reproduces the reporter's metric exactly: many ``within`` decisions and no
    ``exceeded``. If this ever records ``exceeded`` the payload has stopped
    reproducing the report, and the deadline test below would pass for the
    wrong reason.
    """
    monkeypatch.setenv("HEADROOM_ML_STAGE_DEADLINE_SECONDS", "0")  # isolate the size gate

    _apply(cr.ContentRouter(cr.ContentRouterConfig()))

    assert gate_outcomes.get("exceeded", 0) == 0, (
        f"payload no longer reproduces #3711 (a section exceeded the gate): {gate_outcomes}"
    )
    assert gate_outcomes.get("within", 0) > 1, (
        "expected repeated per-section ML entry, the shape that blows the budget; "
        f"got {gate_outcomes}"
    )
    assert slow_kompress.calls > 1, "the ML stage was entered once or not at all"


def test_ml_deadline_stops_further_ml_work_once_the_budget_is_spent(
    gate_outcomes, slow_kompress, monkeypatch
) -> None:
    """Past the ceiling, remaining sections route off ML instead of compounding."""
    monkeypatch.setenv("HEADROOM_ML_STAGE_DEADLINE_SECONDS", "0.06")

    _apply(cr.ContentRouter(cr.ContentRouterConfig()))

    assert gate_outcomes.get("deadline", 0) > 0, (
        f"request-scoped ceiling never fired: {gate_outcomes}"
    )

    # The saving has to be real, so measure it: the same payload with the
    # ceiling disabled must enter the model strictly more often.
    bounded_calls = slow_kompress.calls
    monkeypatch.setenv("HEADROOM_ML_STAGE_DEADLINE_SECONDS", "0")
    unbounded = _SlowKompress(per_call_s=0.05)
    monkeypatch.setattr(cr.ContentRouter, "_get_kompress", lambda self: unbounded)

    _apply(cr.ContentRouter(cr.ContentRouterConfig()))

    assert bounded_calls < unbounded.calls, (
        f"ceiling skipped no model calls: {bounded_calls} with it, {unbounded.calls} without"
    )


def test_deadline_zero_restores_previous_unbounded_behaviour(
    gate_outcomes, slow_kompress, monkeypatch
) -> None:
    monkeypatch.setenv("HEADROOM_ML_STAGE_DEADLINE_SECONDS", "0")

    _apply(cr.ContentRouter(cr.ContentRouterConfig()))

    assert gate_outcomes.get("deadline", 0) == 0, (
        f"deadline fired although disabled: {gate_outcomes}"
    )


def test_direct_compress_callers_stay_unarmed() -> None:
    """``compress()`` without ``apply()`` keeps the old unbounded behaviour.

    Tests and the ``/v1/compress`` path call ``compress()`` directly; they must
    not inherit a request budget nobody set.
    """
    router = cr.ContentRouter(cr.ContentRouterConfig())
    state = router._runtime_state_var.get()
    assert state.ml_budget_active is False

    # That default is instance-scoped and shared by every direct call for the
    # life of the router, so charging it would accumulate across unrelated
    # calls until ML switched itself off permanently.
    router._charge_ml_time(5.0, "some text to compress")
    assert state.ml_elapsed == 0.0


# --------------------------------------------------------------------------- #
# Review follow-ups: the budget must mean ML time, must refuse a call it cannot
# afford, and must sit under the timeout that quarantines the stage.
#
# The original tests asserted that LATER model calls are skipped. That is not
# the same claim as the request-wide time bound the module documents, and all
# three gaps below passed those tests.
# --------------------------------------------------------------------------- #


class _SlowTokenizer(_Tokenizer):
    """Expensive NON-ML prework: token counting during classification."""

    def __init__(self, per_call_s: float) -> None:
        self.per_call_s = per_call_s
        self.calls = 0

    def count_text(self, content: str) -> int:
        self.calls += 1
        time.sleep(self.per_call_s)
        return super().count_text(content)


@pytest.fixture
def charged(monkeypatch) -> list[float]:
    """Every increment billed to the ML budget, in order."""
    seen: list[float] = []
    real = cr.ContentRouter._charge_ml_time

    def _spy(self, elapsed: float, text: str) -> None:
        seen.append(elapsed)
        real(self, elapsed, text)

    monkeypatch.setattr(cr.ContentRouter, "_charge_ml_time", _spy)
    return seen


def test_non_ml_prework_does_not_consume_the_ml_budget(
    gate_outcomes, slow_kompress, charged, monkeypatch
) -> None:
    """The budget was armed at ``apply()`` entry, so lifecycle work, message
    classification and lossless transforms all ran it down before ML began --
    while the log line claimed to measure time "in ML"."""
    monkeypatch.setenv("HEADROOM_ML_STAGE_DEADLINE_SECONDS", "0.4")
    router = cr.ContentRouter(cr.ContentRouterConfig(enable_kompress=True))

    tokenizer = _SlowTokenizer(per_call_s=0.01)
    started = time.monotonic()
    router.apply(
        _tool_result_messages(),
        tokenizer,
        frozen_message_count=1,
        min_tokens_to_compress=1,
    )
    wall = time.monotonic() - started

    assert tokenizer.calls > 0, "prework must actually have run"
    assert slow_kompress.calls > 0, "prework time must not have spent the ML budget"
    # The bound is on ML time, not on the call.
    assert sum(charged) <= wall
    assert sum(charged) == pytest.approx(slow_kompress.calls * slow_kompress.per_call_s, rel=0.5)


def test_a_slow_call_cannot_be_admitted_against_a_budget_it_will_overrun(
    gate_outcomes, charged, monkeypatch
) -> None:
    """Admission used to be ``now >= deadline`` only, so a call starting at
    14.9s of a 15s budget ran to completion well past it -- the ONNX worker is
    not preemptible, so nothing stops it once it has begun."""
    monkeypatch.setenv("HEADROOM_ML_STAGE_DEADLINE_SECONDS", "0.5")
    stub = _SlowKompress(per_call_s=0.35)
    monkeypatch.setattr(cr.ContentRouter, "_get_kompress", lambda self: stub)
    router = cr.ContentRouter(cr.ContentRouterConfig(enable_kompress=True))

    _apply(router)

    assert stub.calls >= 1, "the first block has no measurement yet and must run"
    # The point of the change: total ML time stays inside the budget, rather
    # than the budget merely being the moment after which no call STARTS.
    assert sum(charged) <= 0.5, (
        f"ML overran its 0.5s budget: {sum(charged):.2f}s across {stub.calls} calls"
    )
    assert gate_outcomes.get("deadline", 0) >= 1, "remaining blocks must route off ML"


def test_budget_is_clamped_below_the_timeout_that_quarantines_the_stage(
    monkeypatch,
) -> None:
    """A 15s guard under a 10s executor timeout can never fire first, which is
    exactly the failure the ceiling exists to prevent."""
    monkeypatch.delenv("HEADROOM_ML_STAGE_DEADLINE_SECONDS", raising=False)

    monkeypatch.setenv("HEADROOM_COMPRESSION_TIMEOUT_SECONDS", "30")
    assert cr._ml_stage_deadline_seconds() == 15.0, "the documented default is unchanged"

    monkeypatch.setenv("HEADROOM_COMPRESSION_TIMEOUT_SECONDS", "10")
    clamped = cr._ml_stage_deadline_seconds()
    assert clamped == 5.0
    assert clamped < 10.0, "must leave room to degrade before the executor gives up"

    # The clamp only ever lowers: an explicit request below it is honoured.
    monkeypatch.setenv("HEADROOM_ML_STAGE_DEADLINE_SECONDS", "2")
    assert cr._ml_stage_deadline_seconds() == 2.0

    # And 0 still disables the ceiling entirely.
    monkeypatch.setenv("HEADROOM_ML_STAGE_DEADLINE_SECONDS", "0")
    assert cr._ml_stage_deadline_seconds() == 0.0


def test_a_failing_ml_call_still_draws_down_the_budget(charged, monkeypatch) -> None:
    """Otherwise a request could retry its way past the ceiling: a compressor
    that raises burns the same non-preemptible wall clock as one that returns."""
    monkeypatch.setenv("HEADROOM_ML_STAGE_DEADLINE_SECONDS", "5")

    class _Exploding(_SlowKompress):
        def compress(self, text: str, **_kwargs: object):
            self.calls += 1
            time.sleep(self.per_call_s)
            raise RuntimeError("onnx said no")

    stub = _Exploding(per_call_s=0.05)
    monkeypatch.setattr(cr.ContentRouter, "_get_kompress", lambda self: stub)
    router = cr.ContentRouter(cr.ContentRouterConfig(enable_kompress=True))

    _apply(router)

    assert stub.calls > 0
    assert charged, "a raising call must still be billed"
    assert sum(charged) > 0.0


# --------------------------------------------------------------------------- #
# Second review: the FIRST ML call of a request was always admitted (the seed
# rate was 0, and there is no measurement yet), and it is non-preemptible, so
# one huge first block could still run past the ML budget and the executor
# timeout. The call is now capped rather than estimated.
# --------------------------------------------------------------------------- #


class _CooperativeKompress(_SlowKompress):
    """Behaves like Kompress under a time cap: stops at a chunk boundary once
    ``_time_budget_cap_seconds`` is spent and returns the block unchanged."""

    shares_request_deadline = True

    def __init__(self, worst_case_s: float, chunk_s: float = 0.02) -> None:
        super().__init__(per_call_s=worst_case_s)
        self.chunk_s = chunk_s
        self.caps: list[float | None] = []

    def compress(self, text: str, **kwargs: object) -> _StubResult:
        self.calls += 1
        cap = kwargs.get("_time_budget_cap_seconds")
        self.caps.append(cap)  # type: ignore[arg-type]
        ends_at = None if cap is None else time.monotonic() + float(cap)  # type: ignore[arg-type]
        spent = 0.0
        while spent < self.per_call_s:
            if ends_at is not None and time.monotonic() >= ends_at:
                return _StubResult(compressed=text, compressed_tokens=cr._estimate_tokens(text))
            time.sleep(self.chunk_s)
            spent += self.chunk_s
        out = text[: max(1, len(text) // 2)]
        return _StubResult(compressed=out, compressed_tokens=cr._estimate_tokens(out))


def test_first_ml_call_is_capped_by_the_budget_it_was_admitted_against(
    charged, monkeypatch
) -> None:
    """A first block whose worst case (3s) exceeds both the outer timeout (1s)
    and the ML budget derived from it (0.5s) must still stop inside the budget."""
    monkeypatch.delenv("HEADROOM_ML_STAGE_DEADLINE_SECONDS", raising=False)
    monkeypatch.setenv("HEADROOM_COMPRESSION_TIMEOUT_SECONDS", "1")
    budget = cr._ml_stage_deadline_seconds()
    assert budget == 0.5

    stub = _CooperativeKompress(worst_case_s=3.0)
    monkeypatch.setattr(cr.ContentRouter, "_get_kompress", lambda self: stub)

    _apply(cr.ContentRouter(cr.ContentRouterConfig(enable_kompress=True)))

    assert stub.caps, "the first block must still reach the model"
    first_cap = stub.caps[0]
    assert first_cap is not None and 0.0 < first_cap <= budget, (
        f"first call was not capped by the ML budget: {stub.caps}"
    )
    # One chunk of slack: the cap is checked at chunk boundaries, as in Kompress.
    assert sum(charged) <= budget + 0.2, (
        f"ML ran {sum(charged):.2f}s against a {budget}s budget under a 1s executor timeout"
    )


def test_blocks_refused_by_the_budget_pass_through_unchanged(monkeypatch) -> None:
    """Not the size gate's lossy fallback: the refusal depends on the request,
    the router caches the result, and only raw bytes stay the same on later
    turns, so the prompt cache holds."""
    router = cr.ContentRouter(cr.ContentRouterConfig(enable_kompress=True))
    router._runtime_state_var.set(
        cr._PerRequestRuntimeState(ml_budget_active=True, ml_elapsed=999.0)
    )

    class _Crusher:
        def compress(self, text: str, context: str = "") -> _StubResult:
            return _StubResult(compressed="crushed", compressed_tokens=1)

    monkeypatch.setattr(router, "_get_text_crusher", lambda: _Crusher())
    stub = _SlowKompress(per_call_s=0.0)
    monkeypatch.setattr(router, "_get_kompress", lambda: stub)

    text = _prose(2_000)
    out, _ = router._try_ml_compressor(text, "")

    assert out == text
    assert stub.calls == 0


# --------------------------------------------------------------------------- #
# Third review: the cap reached only LOCAL Kompress. RemoteKompressCompressor
# got no cap and kept its independent 20s per-phase HTTP timeouts, so a cold
# remote call -- no measured rate, so nothing to refuse it on -- could run past
# both the ML budget and the outer timeout. These drive the real remote class
# through the public apply() with an httpx.MockTransport (no network).
# --------------------------------------------------------------------------- #


def _tool_result_text(messages: list[dict]) -> str:
    return messages[-1]["content"][0]["content"]


def _assert_prose_blocks_unchanged(messages: list[dict]) -> None:
    """Every ML-eligible prose block reaches the provider byte-identical.

    (The small JSON separators go through the lossless JSON path, which
    minifies them; that is not ML and not what is under test.)
    """
    out = _tool_result_text(messages)
    for i in range(8):
        assert _prose(40_000, seed=i) in out, f"prose block {i} was altered"


def _remote_kompress(handler) -> RemoteKompressCompressor:
    return RemoteKompressCompressor(
        endpoint="http://kompress.invalid",
        transport=httpx.MockTransport(handler),
    )


def _compressed_reply(request: httpx.Request) -> httpx.Response:
    content = json.loads(request.content)["content"]
    return httpx.Response(200, json={"compressed": " ".join(content.split()[::2])})


def test_cold_remote_call_cannot_overrun_a_tiny_ml_budget(charged, monkeypatch) -> None:
    """The reviewer's probe: 10 ms ML budget, 40 ms outer timeout, an endpoint
    that takes 100 ms. No HTTP round trip fits in what is left, so the remote
    call is declined and every block passes through unchanged."""
    monkeypatch.setenv("HEADROOM_ML_STAGE_DEADLINE_SECONDS", "0.01")
    monkeypatch.setenv("HEADROOM_COMPRESSION_TIMEOUT_SECONDS", "0.04")
    assert cr._ml_stage_deadline_seconds() == 0.01

    requests: list[httpx.Request] = []

    def _slow_endpoint(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        time.sleep(0.1)
        return _compressed_reply(request)

    remote = _remote_kompress(_slow_endpoint)
    monkeypatch.setattr(cr.ContentRouter, "_get_kompress", lambda self: remote)
    router = cr.ContentRouter(cr.ContentRouterConfig(enable_kompress=True))

    messages = _tool_result_messages()
    result = router.apply(messages, _Tokenizer(), frozen_message_count=1, min_tokens_to_compress=1)

    assert charged, "the remote compressor must still be reached and billed"
    assert sum(charged) < 0.04, f"remote ML ran {sum(charged):.3f}s under a 40 ms outer timeout"
    assert requests == [], "a call that cannot fit the budget must not be sent"
    _assert_prose_blocks_unchanged(result.messages)


def test_cold_remote_call_is_bounded_as_a_whole_by_the_ml_budget(charged, monkeypatch) -> None:
    """With enough budget to try, the first (unmeasured) remote call is capped
    by the remainder -- over the whole request, not per httpx phase -- and an
    endpoint slower than that leaves the block unchanged."""
    budget = 0.2
    monkeypatch.setenv("HEADROOM_ML_STAGE_DEADLINE_SECONDS", str(budget))
    monkeypatch.setenv("HEADROOM_COMPRESSION_TIMEOUT_SECONDS", "30")

    release = threading.Event()
    timeouts: list[dict] = []

    def _hung_endpoint(request: httpx.Request) -> httpx.Response:
        timeouts.append(dict(request.extensions["timeout"]))
        release.wait(5.0)  # far past the budget; MockTransport ignores httpx timeouts
        return _compressed_reply(request)

    remote = _remote_kompress(_hung_endpoint)
    monkeypatch.setattr(cr.ContentRouter, "_get_kompress", lambda self: remote)
    router = cr.ContentRouter(cr.ContentRouterConfig(enable_kompress=True))

    messages = _tool_result_messages()
    try:
        result = router.apply(
            messages, _Tokenizer(), frozen_message_count=1, min_tokens_to_compress=1
        )
    finally:
        release.set()
        remote.close()

    assert timeouts, "the first block must reach the endpoint"
    assert all(0.0 < t <= budget for t in timeouts[0].values()), (
        f"per-phase timeouts were not lowered to the budget: {timeouts[0]}"
    )
    assert sum(charged) <= budget + 0.1, (
        f"remote ML ran {sum(charged):.3f}s against a {budget}s budget"
    )
    _assert_prose_blocks_unchanged(result.messages)


def test_remote_kompress_keeps_its_default_timeout_without_a_budget() -> None:
    """Direct callers pass no cap: same single request, 20s per-phase timeouts."""
    timeouts: list[dict] = []

    def _endpoint(request: httpx.Request) -> httpx.Response:
        timeouts.append(dict(request.extensions["timeout"]))
        return _compressed_reply(request)

    remote = _remote_kompress(_endpoint)
    text = _prose(2_000)
    out = remote.compress(text)

    assert out.compressed != text
    assert timeouts == [{"connect": 20.0, "read": 20.0, "write": 20.0, "pool": 20.0}]
    assert remote._capped_pool is None, "the default path must not hop threads"


# --------------------------------------------------------------------------- #
# Fourth review: the capped worker pool had no admission bound. Its queue is
# unbounded and ``future.cancel()`` does not drop a queued payload, so once
# every worker was held by an abandoned request, each further capped call
# queued its full prompt, timed out, and left it there. A slow-drip body could
# hold a worker forever, since httpx read timeouts reset on every chunk.
# --------------------------------------------------------------------------- #


def _free_permits(remote: RemoteKompressCompressor) -> int:
    return remote._capped_slots._value  # type: ignore[attr-defined]


def _wait_until(predicate, timeout_s: float = 3.0) -> bool:
    ends = time.monotonic() + timeout_s
    while time.monotonic() < ends:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_saturated_remote_calls_fail_open_without_queueing_their_payloads() -> None:
    """The reviewer's repro: hold all 16 workers with requests whose 200 ms
    callers have already timed out, then make five more calls with 60 ms caps.
    Each must fail open within its cap with its payload never enqueued, and
    capacity must come back once the held requests end."""
    from headroom.transforms.kompress_remote import _CAPPED_CALL_WORKERS

    release = threading.Event()
    reached: list[str] = []

    def _stalled_endpoint(request: httpx.Request) -> httpx.Response:
        reached.append(json.loads(request.content)["content"])
        release.wait(5.0)
        return _compressed_reply(request)

    remote = _remote_kompress(_stalled_endpoint)
    text = _prose(2_000)
    try:
        with concurrent.futures.ThreadPoolExecutor(_CAPPED_CALL_WORKERS) as callers:
            held = list(
                callers.map(
                    lambda i: remote.compress(f"held{i} {text}", _time_budget_cap_seconds=0.2),
                    range(_CAPPED_CALL_WORKERS),
                )
            )
        assert all(out.compressed == f"held{i} {text}" for i, out in enumerate(held))
        assert len(reached) == _CAPPED_CALL_WORKERS
        assert _free_permits(remote) == 0, "abandoned requests must keep their permits"

        for i in range(5):
            late = f"late{i} {text}"
            started = time.monotonic()
            out = remote.compress(late, _time_budget_cap_seconds=0.06)
            took = time.monotonic() - started
            assert out.compressed == late, "a saturated call must fail open unchanged"
            assert took < 0.06 + 0.1, f"saturated call took {took:.3f}s against a 60 ms cap"
            assert remote._capped_pool._work_queue.qsize() == 0, "a payload was enqueued"

        assert len(reached) == _CAPPED_CALL_WORKERS, "a saturated call reached the endpoint"
    finally:
        release.set()

    assert _wait_until(lambda: _free_permits(remote) == _CAPPED_CALL_WORKERS)
    recovered = remote.compress(text, _time_budget_cap_seconds=2.0)
    assert recovered.compressed != text, "capacity must return once the held requests end"
    remote.close()


def test_slow_drip_response_is_cut_off_at_an_absolute_deadline() -> None:
    """A body that keeps dripping (a chunk every 50 ms for 2 s) resets httpx's
    read timeout on every chunk. The worker must stop at the cap's absolute
    deadline and give its permit back, not ride the response to the end."""
    from headroom.transforms.kompress_remote import _CAPPED_CALL_WORKERS

    chunks_sent: list[int] = []

    def _dripping_endpoint(request: httpx.Request) -> httpx.Response:
        reply = json.dumps({"compressed": "short"}).encode()

        def _drip():
            for i in range(40):
                chunks_sent.append(i)
                time.sleep(0.05)
                yield b" "  # JSON-legal leading whitespace
            yield reply

        return httpx.Response(200, content=_drip())

    remote = _remote_kompress(_dripping_endpoint)
    text = _prose(2_000)
    cap = 0.2
    started = time.monotonic()
    out = remote.compress(text, _time_budget_cap_seconds=cap)
    assert out.compressed == text
    assert time.monotonic() - started < cap + 0.1

    assert _wait_until(lambda: _free_permits(remote) == _CAPPED_CALL_WORKERS, timeout_s=1.0), (
        "the dripping request still holds its permit"
    )
    assert time.monotonic() - started < 1.0, "the worker rode the drip instead of the deadline"
    assert len(chunks_sent) < 10, f"read {len(chunks_sent)} chunks past a {cap}s deadline"
    remote.close()


def test_permit_is_held_until_the_worker_finishes_not_the_caller() -> None:
    """Releasing on caller timeout would let abandoned requests pile up again;
    the permit belongs to the running request."""
    from headroom.transforms.kompress_remote import _CAPPED_CALL_WORKERS

    release = threading.Event()

    def _stalled_endpoint(request: httpx.Request) -> httpx.Response:
        release.wait(5.0)
        return _compressed_reply(request)

    remote = _remote_kompress(_stalled_endpoint)
    text = _prose(2_000)
    try:
        out = remote.compress(text, _time_budget_cap_seconds=0.1)
        assert out.compressed == text
        assert _free_permits(remote) == _CAPPED_CALL_WORKERS - 1
    finally:
        release.set()
    assert _wait_until(lambda: _free_permits(remote) == _CAPPED_CALL_WORKERS)
    remote.close()
