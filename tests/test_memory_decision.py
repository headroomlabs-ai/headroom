"""Tests for :class:`headroom.proxy.memory_decision.MemoryDecision`.

The point of this file is the *contract* — every behavioural assertion
here is the canonical answer to "should this request have memory
context injected into it?" that today is computed inline across the
proxy's handlers with subtle drift.

Specifically locks (post-PR-this):

* Sites 1/2/3 (Anthropic ``/v1/messages``, Gemini
  ``:generateContent``, OpenAI ``/v1/chat/completions``) MUST gate on
  bypass — pre-this-PR they didn't, so memory injection silently
  mutated requests under ``x-headroom-bypass: true``.
* Site 4 (OpenAI ``/v1/responses``) already gated; locked here.
* Site 6 (OpenAI WS ``/v1/responses``) already gated; locked here.
* ``MemoryMode`` values (``auto_tail`` / ``tool``) and the env-driven
  ``HEADROOM_MEMORY_INJECTION_MODE`` (``disabled`` / ``auto_tail`` /
  ``tool``) all surface as explicit ``skip_reason`` values, not as
  hidden conditional code.

The decision is **input-side only** — it gates whether mutation of the
request bytes happens. It does NOT gate background memory STORAGE
(traffic-learner runs on a separate path and is unaffected by this
decision).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

from headroom.proxy.memory_decision import MemoryDecision


def _memory_handler() -> Any:
    """Minimal stand-in for the proxy's ``self.memory_handler``."""
    return SimpleNamespace(name="local")


# ── Value-type contract ───────────────────────────────────────────────


def test_decision_is_frozen() -> None:
    """Frozen dataclass — mutation would let a handler patch the
    decision after handing it to the funnel."""
    d = MemoryDecision.decide(
        headers={}, memory_handler=_memory_handler(), memory_user_id="u1", mode_name="auto_tail"
    )
    try:
        d.inject = False  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("MemoryDecision must be frozen")


def test_decision_is_value_equal() -> None:
    """Two decisions from identical inputs compare equal."""
    a = MemoryDecision.decide(
        headers={}, memory_handler=_memory_handler(), memory_user_id="u1", mode_name="auto_tail"
    )
    b = MemoryDecision.decide(
        headers={}, memory_handler=_memory_handler(), memory_user_id="u1", mode_name="auto_tail"
    )
    assert a == b


# ── Precedence: bypass > no_handler > no_user_id > mode > INJECT ─────


def test_injects_when_every_gate_open() -> None:
    """Happy path: no bypass, handler wired, user_id set, mode=auto_tail."""
    d = MemoryDecision.decide(
        headers={}, memory_handler=_memory_handler(), memory_user_id="u1", mode_name="auto_tail"
    )
    assert d.inject is True
    assert d.skip_reason is None


def test_bypass_header_wins_over_every_other_gate() -> None:
    """``x-headroom-bypass: true`` is the user's "do not touch my
    bytes" signal — highest priority, even when memory is otherwise
    fully wired. Memory injection mutates the request bytes; bypass
    must skip it. (This was the 3-bug Gemini-class problem pre-PR
    on Anthropic, OpenAI chat, Gemini.)"""
    d = MemoryDecision.decide(
        headers={"x-headroom-bypass": "true"},
        memory_handler=_memory_handler(),
        memory_user_id="u1",
        mode_name="auto_tail",
    )
    assert d.inject is False
    assert d.skip_reason == "bypass_header"


def test_passthrough_mode_header_also_triggers_bypass_skip() -> None:
    """``x-headroom-mode: passthrough`` is the alternate spelling of
    the bypass signal — mirrors _headroom_bypass_enabled semantics."""
    d = MemoryDecision.decide(
        headers={"x-headroom-mode": "passthrough"},
        memory_handler=_memory_handler(),
        memory_user_id="u1",
        mode_name="auto_tail",
    )
    assert d.inject is False
    assert d.skip_reason == "bypass_header"


def test_no_handler_is_skip() -> None:
    """No memory backend configured → ``no_handler`` reason."""
    d = MemoryDecision.decide(
        headers={}, memory_handler=None, memory_user_id="u1", mode_name="auto_tail"
    )
    assert d.inject is False
    assert d.skip_reason == "no_handler"


def test_no_user_id_is_skip() -> None:
    """Memory wired but per-request user_id missing → ``no_user_id``."""
    d = MemoryDecision.decide(
        headers={}, memory_handler=_memory_handler(), memory_user_id=None, mode_name="auto_tail"
    )
    assert d.inject is False
    assert d.skip_reason == "no_user_id"


def test_empty_user_id_string_is_skip() -> None:
    """Empty string for user_id is treated as missing."""
    d = MemoryDecision.decide(
        headers={}, memory_handler=_memory_handler(), memory_user_id="", mode_name="auto_tail"
    )
    assert d.inject is False
    assert d.skip_reason == "no_user_id"


def test_mode_disabled_is_skip() -> None:
    """Operator override via HEADROOM_MEMORY_INJECTION_MODE=disabled."""
    d = MemoryDecision.decide(
        headers={}, memory_handler=_memory_handler(), memory_user_id="u1", mode_name="disabled"
    )
    assert d.inject is False
    assert d.skip_reason == "mode_disabled"


def test_mode_tool_is_skip() -> None:
    """TOOL mode: auto-injection disabled (the agent calls memory
    tools explicitly instead). Distinct from disabled."""
    d = MemoryDecision.decide(
        headers={}, memory_handler=_memory_handler(), memory_user_id="u1", mode_name="tool"
    )
    assert d.inject is False
    assert d.skip_reason == "mode_tool"


# ── Precedence ordering when multiple gates close ────────────────────


def test_bypass_beats_no_handler() -> None:
    """When both bypass AND no_handler would skip, surface bypass —
    user's explicit signal is the more informative dashboard slice."""
    d = MemoryDecision.decide(
        headers={"x-headroom-bypass": "true"},
        memory_handler=None,
        memory_user_id=None,
        mode_name="disabled",
    )
    assert d.skip_reason == "bypass_header"


def test_no_handler_beats_no_user_id() -> None:
    """no_handler is the more fundamental failure (no backend wired);
    no_user_id only matters when a handler exists."""
    d = MemoryDecision.decide(
        headers={}, memory_handler=None, memory_user_id=None, mode_name="auto_tail"
    )
    assert d.skip_reason == "no_handler"


def test_no_user_id_beats_mode() -> None:
    """A request with no user_id can't be served regardless of mode."""
    d = MemoryDecision.decide(
        headers={}, memory_handler=_memory_handler(), memory_user_id=None, mode_name="disabled"
    )
    assert d.skip_reason == "no_user_id"


def test_mode_disabled_beats_mode_tool() -> None:
    """``disabled`` is operator-level kill; ``tool`` is mode pref.
    Disabled is the more emphatic signal."""
    # Construct two separate decisions on identical inputs but mode varying;
    # check that each produces its own reason (no cross-contamination).
    d_dis = MemoryDecision.decide(
        headers={}, memory_handler=_memory_handler(), memory_user_id="u1", mode_name="disabled"
    )
    d_tool = MemoryDecision.decide(
        headers={}, memory_handler=_memory_handler(), memory_user_id="u1", mode_name="tool"
    )
    assert d_dis.skip_reason == "mode_disabled"
    assert d_tool.skip_reason == "mode_tool"


# ── Observability fields ─────────────────────────────────────────────


def test_observability_booleans_populated_when_injecting() -> None:
    """Even on the happy path, every constituent is exposed so debug
    tooling can answer "what did the decision see?" without re-running."""
    d = MemoryDecision.decide(
        headers={}, memory_handler=_memory_handler(), memory_user_id="u1", mode_name="auto_tail"
    )
    assert d.bypass_header_set is False
    assert d.memory_handler_present is True
    assert d.memory_user_id_present is True
    assert d.mode_name == "auto_tail"


def test_observability_booleans_populated_when_skipping() -> None:
    """Same on the skip path — every constituent must be visible."""
    d = MemoryDecision.decide(
        headers={"x-headroom-bypass": "true"},
        memory_handler=None,
        memory_user_id="u1",
        mode_name="auto_tail",
    )
    assert d.bypass_header_set is True
    assert d.memory_handler_present is False
    assert d.memory_user_id_present is True
    assert d.mode_name == "auto_tail"


# ── apply_to_tags — mirror CompressionDecision pattern ───────────────


def test_apply_to_tags_stamps_reason_when_skipping() -> None:
    """Skip decisions surface ``memory_skip_reason`` in tags so the
    dashboard can slice memory-blind traffic by cause."""
    d = MemoryDecision.decide(
        headers={"x-headroom-bypass": "true"},
        memory_handler=_memory_handler(),
        memory_user_id="u1",
        mode_name="auto_tail",
    )
    tags: dict[str, str] = {}
    d.apply_to_tags(tags)
    assert tags == {"memory_skip_reason": "bypass_header"}


def test_apply_to_tags_is_a_noop_when_injecting() -> None:
    """No tag when injecting — absence is the signal for "memory was
    used". Avoids spurious ``memory_skip_reason=None`` strings."""
    d = MemoryDecision.decide(
        headers={}, memory_handler=_memory_handler(), memory_user_id="u1", mode_name="auto_tail"
    )
    tags: dict[str, str] = {"client": "codex"}
    d.apply_to_tags(tags)
    assert tags == {"client": "codex"}
    assert "memory_skip_reason" not in tags


def test_apply_to_tags_preserves_pre_existing_entries() -> None:
    """Existing tags (client, passthrough_reason from CompressionDecision)
    must survive unchanged."""
    d = MemoryDecision.decide(
        headers={}, memory_handler=None, memory_user_id="u1", mode_name="auto_tail"
    )
    tags: dict[str, str] = {"client": "claude-code", "passthrough_reason": "bypass_header"}
    d.apply_to_tags(tags)
    assert tags == {
        "client": "claude-code",
        "passthrough_reason": "bypass_header",
        "memory_skip_reason": "no_handler",
    }


def test_apply_to_tags_for_every_skip_reason() -> None:
    """Every skip reason name must round-trip through apply_to_tags."""
    cases: dict[str, dict[str, Any]] = {
        "bypass_header": {
            "headers": {"x-headroom-bypass": "true"},
            "memory_handler": _memory_handler(),
            "memory_user_id": "u1",
            "mode_name": "auto_tail",
        },
        "no_handler": {
            "headers": {},
            "memory_handler": None,
            "memory_user_id": "u1",
            "mode_name": "auto_tail",
        },
        "no_user_id": {
            "headers": {},
            "memory_handler": _memory_handler(),
            "memory_user_id": None,
            "mode_name": "auto_tail",
        },
        "mode_disabled": {
            "headers": {},
            "memory_handler": _memory_handler(),
            "memory_user_id": "u1",
            "mode_name": "disabled",
        },
        "mode_tool": {
            "headers": {},
            "memory_handler": _memory_handler(),
            "memory_user_id": "u1",
            "mode_name": "tool",
        },
    }
    for expected, kwargs in cases.items():
        d = MemoryDecision.decide(**kwargs)
        tags: dict[str, str] = {}
        d.apply_to_tags(tags)
        assert tags.get("memory_skip_reason") == expected, expected
