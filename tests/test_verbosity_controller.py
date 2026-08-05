"""Tests for headroom.proxy.verbosity_controller — the AIMD state machine."""

from __future__ import annotations

from headroom.proxy.verbosity_controller import (
    ControllerState,
    Signal,
    VerbosityController,
    load_state,
    save_state,
)

CTRL = VerbosityController(floor=1, ceil=4, probe_threshold=3, cooldown_turns=5)


def _run(signals, start=2):
    state = ControllerState(level=start)
    for s in signals:
        state = CTRL.observe(state, s)
    return state


class TestAdditiveIncrease:
    def test_steps_up_only_after_threshold(self):
        state = _run([Signal.TOO_MUCH, Signal.TOO_MUCH])
        assert state.level == 2  # not yet at threshold
        assert state.up_streak == 2

    def test_steps_up_at_threshold(self):
        state = _run([Signal.TOO_MUCH] * 3)
        assert state.level == 3
        assert state.up_streak == 0  # reset after stepping

    def test_neutral_breaks_the_streak(self):
        state = _run([Signal.TOO_MUCH, Signal.TOO_MUCH, Signal.NEUTRAL, Signal.TOO_MUCH])
        assert state.level == 2  # streak was broken; only 1 consecutive at end
        assert state.up_streak == 1

    def test_does_not_exceed_ceiling(self):
        state = _run([Signal.TOO_MUCH] * 30, start=4)
        assert state.level == 4


class TestMultiplicativeDecrease:
    def test_too_little_backs_off_immediately(self):
        state = _run([Signal.TOO_LITTLE], start=3)
        assert state.level == 2
        assert state.cooldown == 5

    def test_does_not_go_below_floor(self):
        state = _run([Signal.TOO_LITTLE] * 10, start=2)
        assert state.level == 1

    def test_cooldown_suppresses_reescalation(self):
        # Back off, then immediately get TOO_MUCH pressure — must not re-terse
        # until the cooldown elapses.
        state = ControllerState(level=3)
        state = CTRL.observe(state, Signal.TOO_LITTLE)  # → level 2, cooldown 5
        assert state.level == 2
        for _ in range(3):  # would normally step up at 3, but we're cooling down
            state = CTRL.observe(state, Signal.TOO_MUCH)
        assert state.level == 2  # held

    def test_reescalates_after_cooldown_expires(self):
        state = ControllerState(level=2, cooldown=5)
        # 5 neutral turns drain the cooldown...
        for _ in range(5):
            state = CTRL.observe(state, Signal.NEUTRAL)
        assert state.cooldown == 0
        # ...then sustained pressure can step up again.
        for _ in range(3):
            state = CTRL.observe(state, Signal.TOO_MUCH)
        assert state.level == 3


class TestPersistence:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "ctrl.json"
        save_state(path, ControllerState(level=3, up_streak=2, cooldown=1))
        state = load_state(path, default_level=2, floor=1, ceil=4)
        assert state.level == 3
        assert state.up_streak == 2

    def test_missing_uses_default(self, tmp_path):
        state = load_state(tmp_path / "nope.json", default_level=2, floor=1, ceil=4)
        assert state.level == 2

    def test_corrupt_uses_default(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{broken")
        state = load_state(p, default_level=3, floor=1, ceil=4)
        assert state.level == 3

    def test_loaded_level_clamped(self, tmp_path):
        p = tmp_path / "ctrl.json"
        save_state(p, ControllerState(level=9))
        state = load_state(p, default_level=2, floor=1, ceil=4)
        assert state.level == 4
