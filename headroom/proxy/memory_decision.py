"""``MemoryDecision``: canonical "should we inject memory context?" gate.

Input-side analog of :class:`CompressionDecision`. Pre-this-PR, the
6 memory-injection sites across four handler files computed the gate
inline with subtle drift — most notably, 3 sites (Anthropic chat,
OpenAI chat, Gemini) never gated on ``x-headroom-bypass``, so
memory injection silently mutated requests when the user explicitly
asked for byte-faithful passthrough.

This is **decision-only**. It gates whether the request bytes get
mutated (memory context appended to user-tail). It does NOT gate
background memory STORAGE — traffic-learner runs on a separate path
and continues accumulating signal even under bypass. The user's
"don't touch my bytes" signal is for the INJECTION; the user's
working memory should still grow.

Precedence (highest first):

  1. ``bypass_header`` — user's explicit "do not touch my bytes"
  2. ``no_handler`` — no memory backend configured
  3. ``no_user_id`` — per-request user_id missing
  4. ``mode_disabled`` — operator HEADROOM_MEMORY_INJECTION_MODE=disabled
  5. ``mode_tool`` — operator HEADROOM_MEMORY_INJECTION_MODE=tool
     (auto-inject off; the agent calls memory tools explicitly)
  6. otherwise → ``inject=True``

This module exposes one value type + one factory + one helper. Pure
function; same Rust-port shape as :class:`CompressionDecision`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from headroom.proxy.memory_decision_policy import (
    apply_memory_skip_reason,
    decide_memory_injection,
)


@dataclass(frozen=True)
class MemoryDecision:
    """Immutable, value-equal snapshot of the memory-injection decision.

    Construction policy: use :meth:`decide`. Direct construction is
    legal but unusual — tests use it; handlers always go through
    ``decide``. The constituent observability booleans
    (``bypass_header_set`` etc.) MUST match the inputs ``decide`` saw;
    the factory enforces that invariant, and the dataclass being
    frozen means downstream code can't violate it.
    """

    inject: bool
    # When ``inject`` is False, this is the canonical reason surfaced
    # in logs and in RequestOutcome.tags["memory_skip_reason"] so the
    # dashboard can slice memory-skip traffic by cause. One of:
    #   * "bypass_header"  — user set x-headroom-bypass/x-headroom-mode
    #   * "no_handler"     — no memory backend configured on the proxy
    #   * "no_user_id"     — per-request user_id missing
    #   * "mode_disabled"  — operator HEADROOM_MEMORY_INJECTION_MODE=disabled
    #   * "mode_tool"      — operator HEADROOM_MEMORY_INJECTION_MODE=tool
    # When ``inject`` is True, this is None.
    skip_reason: str | None

    # Observability: every constituent boolean exposed so debug tools
    # answer "what did the decision see?" without re-running.
    bypass_header_set: bool
    memory_handler_present: bool
    memory_user_id_present: bool
    mode_name: str

    @classmethod
    def decide(
        cls,
        *,
        headers: Any,
        memory_handler: Any | None,
        memory_user_id: str | None,
        mode_name: str,
    ) -> MemoryDecision:
        """Compute the canonical memory-injection decision.

        Parameters
        ----------
        headers
            Inbound request headers. Accepts any object with a
            ``.get(key)`` method (dict, starlette Headers, mapping).
            Bypass detected via ``_headroom_bypass_enabled``.
        memory_handler
            The proxy's memory handler instance, or ``None`` if no
            memory backend is configured. Presence only is checked —
            no methods called.
        memory_user_id
            Per-request user_id from ``x-headroom-user-id`` header (or
            env default). ``None`` or empty string treated as missing.
        mode_name
            One of ``"auto_tail"`` / ``"tool"`` / ``"disabled"``.
            Comes from ``get_memory_injection_mode()`` which reads
            ``HEADROOM_MEMORY_INJECTION_MODE``.
        """
        decision = decide_memory_injection(
            headers=headers,
            memory_handler_present=memory_handler is not None,
            memory_user_id_present=bool(memory_user_id),
            mode_name=mode_name,
        )

        return cls(
            inject=decision.inject,
            skip_reason=decision.skip_reason,
            bypass_header_set=decision.bypass_header_set,
            memory_handler_present=decision.memory_handler_present,
            memory_user_id_present=decision.memory_user_id_present,
            mode_name=decision.mode_name,
        )

    def apply_to_tags(self, tags: dict[str, str]) -> None:
        """Stamp the skip reason into a tags dict for dashboard slicing.

        Mutates ``tags`` in place. No-op when ``inject=True`` —
        absence vs presence of ``memory_skip_reason`` is the signal.

        Mirror of :meth:`CompressionDecision.apply_to_tags`. Handlers
        call this immediately after ``decide()`` so the resulting
        ``RequestOutcome.tags`` carries memory observability via the
        same path the funnel already uses for ``client`` and
        ``passthrough_reason``.
        """
        apply_memory_skip_reason(tags, self.skip_reason)
