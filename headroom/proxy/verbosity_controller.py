"""AIMD controller for live verbosity adjustment.

The offline ``learn --verbosity`` pass sets the *starting* level. This controller
tracks drift during a session and nudges the level from runtime signals, using
the congestion-control intuition:

- **Additive increase** toward terser output: only after *sustained* "the user
  isn't reading this" pressure (a streak of TOO_MUCH signals) do we step the
  level up by one. Probing up is cheap to get wrong only slowly.
- **Multiplicative-style decrease** on a TOO_LITTLE signal (the user asked for
  more): back off immediately by a level and enter a cooldown that suppresses
  re-escalation. Annoying the user is the expensive event — like congestion —
  so we react fast and then hold off.

The controller is a pure state machine over an abstract signal. Detecting the
signals at the proxy (fast-skip from reply timing, interrupt from a cancelled
stream) is the caller's job; keeping detection out of here makes the control
logic deterministic and testable, and lets the live path enable only the
signals it can measure reliably.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path


class Signal(Enum):
    """An abstract feedback signal about the last response's verbosity."""

    TOO_MUCH = "too_much"  # interrupted / fast-skipped → output went unread
    TOO_LITTLE = "too_little"  # user asked to explain / expand
    NEUTRAL = "neutral"  # engaged normally


@dataclass
class ControllerState:
    """Per-conversation (or per-project) controller state."""

    level: int
    up_streak: int = 0
    cooldown: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, int]) -> ControllerState:
        return cls(
            level=int(d.get("level", 2)),
            up_streak=int(d.get("up_streak", 0)),
            cooldown=int(d.get("cooldown", 0)),
        )


@dataclass(frozen=True)
class VerbosityController:
    """Pure AIMD controller. ``observe`` maps (state, signal) → new state."""

    floor: int = 1
    ceil: int = 4
    probe_threshold: int = 3  # consecutive TOO_MUCH before stepping up
    cooldown_turns: int = 5  # turns after a back-off during which we don't re-probe

    def observe(self, state: ControllerState, signal: Signal) -> ControllerState:
        level = state.level
        up_streak = state.up_streak
        cooldown = max(0, state.cooldown - 1)  # one turn elapses per observation

        if signal is Signal.TOO_LITTLE:
            # Fast back-off: drop a level immediately and suppress re-escalation.
            return ControllerState(
                level=max(self.floor, level - 1),
                up_streak=0,
                cooldown=self.cooldown_turns,
            )

        if signal is Signal.TOO_MUCH:
            if cooldown > 0:
                # Recently backed off — don't re-terse yet; keep cooling down.
                return ControllerState(level=level, up_streak=0, cooldown=cooldown)
            up_streak += 1
            if up_streak >= self.probe_threshold and level < self.ceil:
                return ControllerState(level=level + 1, up_streak=0, cooldown=0)
            return ControllerState(level=level, up_streak=up_streak, cooldown=cooldown)

        # NEUTRAL: engagement resets the upward streak (we require *consecutive*
        # pressure) and lets any cooldown tick down.
        return ControllerState(level=level, up_streak=0, cooldown=cooldown)


def load_state(path: Path, default_level: int, floor: int, ceil: int) -> ControllerState:
    """Load controller state, clamped to [floor, ceil]; seed from default."""
    try:
        d = json.loads(Path(path).read_text())
        state = ControllerState.from_dict(d)
    except (OSError, json.JSONDecodeError, ValueError):
        state = ControllerState(level=default_level)
    state.level = max(floor, min(ceil, state.level))
    return state


def save_state(path: Path, state: ControllerState) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state.to_dict(), separators=(",", ":")))
