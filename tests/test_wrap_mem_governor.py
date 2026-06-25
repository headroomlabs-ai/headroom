"""Tests for the ``headroom wrap`` memory-pressure governor.

The wait loop's clock, sleep, and memory read are all injected so these tests
are deterministic and run instantly (no real sleeping).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from headroom.cli.mem_governor import (
    EX_TEMPFAIL,
    available_memory_mb,
    gate_launch,
)


class _FakeClock:
    """Monotonic clock whose ``sleep`` advances the clock, so the wait loop
    converges without wall-clock time."""

    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def _reader(values: list[int | None]) -> Callable[[], int | None]:
    """Return a read_available_mb that yields ``values`` in order, then repeats
    the last value forever (mimics a sustained condition)."""
    state = {"i": 0}

    def read() -> int | None:
        i = state["i"]
        if i < len(values):
            state["i"] = i + 1
            return values[i]
        return values[-1] if values else None

    return read


# --------------------------------------------------------------------------
# Disabled / short-circuit paths
# --------------------------------------------------------------------------


def test_policy_off_short_circuits_without_reading() -> None:
    reads = {"n": 0}

    def read() -> int | None:
        reads["n"] += 1
        return 0

    out = gate_launch(
        env={"HEADROOM_WRAP_MEM_POLICY": "off"},
        read_available_mb=read,
    )
    assert out == "off"
    assert reads["n"] == 0  # never even measured


def test_min_mb_zero_disables() -> None:
    out = gate_launch(
        env={"HEADROOM_WRAP_MEM_MIN_MB": "0"},
        read_available_mb=lambda: 10,
    )
    assert out == "off"


def test_unmeasured_fails_open() -> None:
    out = gate_launch(env={}, read_available_mb=lambda: None)
    assert out == "unmeasured"


def test_enough_headroom_proceeds_immediately() -> None:
    msgs: list[str] = []
    out = gate_launch(
        echo=msgs.append,
        env={"HEADROOM_WRAP_MEM_MIN_MB": "2048"},
        read_available_mb=lambda: 5000,
    )
    assert out == "ok"
    assert msgs == []  # quiet on the happy path


# --------------------------------------------------------------------------
# refuse policy
# --------------------------------------------------------------------------


def test_refuse_below_floor_exits_tempfail() -> None:
    msgs: list[str] = []
    with pytest.raises(SystemExit) as exc:
        gate_launch(
            echo=msgs.append,
            env={"HEADROOM_WRAP_MEM_POLICY": "refuse", "HEADROOM_WRAP_MEM_MIN_MB": "2048"},
            read_available_mb=lambda: 500,
        )
    assert exc.value.code == EX_TEMPFAIL
    assert any("refusing launch" in m for m in msgs)


def test_refuse_with_headroom_proceeds() -> None:
    out = gate_launch(
        env={"HEADROOM_WRAP_MEM_POLICY": "refuse", "HEADROOM_WRAP_MEM_MIN_MB": "2048"},
        read_available_mb=lambda: 4096,
    )
    assert out == "ok"


# --------------------------------------------------------------------------
# wait policy
# --------------------------------------------------------------------------


def test_wait_then_recovers() -> None:
    clock = _FakeClock()
    msgs: list[str] = []
    # 1000 (< floor) up front -> wait; stays low one poll, then frees to 3000.
    read = _reader([1000, 1000, 3000])
    out = gate_launch(
        echo=msgs.append,
        env={
            "HEADROOM_WRAP_MEM_MIN_MB": "2048",
            "HEADROOM_WRAP_MEM_WAIT_SECONDS": "60",
            "HEADROOM_WRAP_MEM_POLL_SECONDS": "3",
        },
        read_available_mb=read,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert out == "recovered"
    assert clock.t == pytest.approx(6.0)  # two polls of 3s
    assert any("waiting up to" in m for m in msgs)
    assert any("proceeding" in m for m in msgs)


def test_wait_times_out_then_fails_open() -> None:
    clock = _FakeClock()
    msgs: list[str] = []
    out = gate_launch(
        echo=msgs.append,
        env={
            "HEADROOM_WRAP_MEM_MIN_MB": "2048",
            "HEADROOM_WRAP_MEM_WAIT_SECONDS": "9",
            "HEADROOM_WRAP_MEM_POLL_SECONDS": "3",
        },
        read_available_mb=_reader([1000]),  # never recovers
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert out == "proceeded"
    assert clock.t == pytest.approx(9.0)  # 3 polls then deadline
    assert any("proceeding anyway (fail-open)" in m for m in msgs)


def test_wait_aborts_if_memory_becomes_unmeasurable() -> None:
    clock = _FakeClock()
    out = gate_launch(
        env={
            "HEADROOM_WRAP_MEM_MIN_MB": "2048",
            "HEADROOM_WRAP_MEM_WAIT_SECONDS": "60",
            "HEADROOM_WRAP_MEM_POLL_SECONDS": "3",
        },
        read_available_mb=_reader([1000, None]),
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert out == "unmeasured"


def test_unknown_policy_defaults_to_wait() -> None:
    clock = _FakeClock()
    out = gate_launch(
        env={
            "HEADROOM_WRAP_MEM_POLICY": "bogus",
            "HEADROOM_WRAP_MEM_MIN_MB": "2048",
            "HEADROOM_WRAP_MEM_WAIT_SECONDS": "3",
            "HEADROOM_WRAP_MEM_POLL_SECONDS": "3",
        },
        read_available_mb=_reader([1000]),
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert out == "proceeded"  # treated as wait, timed out, fail-open


def test_bad_env_values_fall_back_to_defaults() -> None:
    # Non-numeric overrides must not crash; they fall back to defaults and the
    # happy path (plenty of memory) still returns "ok".
    out = gate_launch(
        env={
            "HEADROOM_WRAP_MEM_MIN_MB": "not-a-number",
            "HEADROOM_WRAP_MEM_WAIT_SECONDS": "soon",
        },
        read_available_mb=lambda: 10_000,
    )
    assert out == "ok"


# --------------------------------------------------------------------------
# available_memory_mb integration
# --------------------------------------------------------------------------


def test_available_memory_mb_returns_sane_value() -> None:
    val = available_memory_mb()
    # None on a host with neither /proc/meminfo nor psutil; otherwise positive.
    assert val is None or val > 0
