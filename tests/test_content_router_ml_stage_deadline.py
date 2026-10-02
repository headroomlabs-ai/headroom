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

import json
import random
import time
from dataclasses import dataclass

import pytest

from headroom.transforms import content_router as cr


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
