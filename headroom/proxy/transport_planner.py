"""Decide the upstream wire API for one Copilot-bound request.

The problem
-----------
A Copilot CLI session pins **one** wire API for its whole lifetime, chosen at
launch from the main model. Every later request rides that connection --
including subagent requests for a completely different model. So the inbound
wire API says what the *client* was configured with, not what the *model*
needs, and the proxy has to reconcile the two per request.

Headroom used to reconcile by name (``startswith(("gpt-5", "o1", "o3"))``).
That is wrong in the unsafe direction: ``mai-code-1-flash-picker`` is served
only on ``/responses`` but does not match the prefix, so a name-based downgrade
routes it to ``/chat/completions`` and breaks a model that works today. The
catalog publishes ``supported_endpoints`` directly, so this planner uses that
and keeps the name heuristic strictly as a fallback.

Two rules earn their keep
-------------------------
**Never select an endpoint by list position.** Published order is not
meaningful: ``claude-opus-4.6`` lists ``/v1/messages`` first while
``claude-sonnet-4.6`` lists ``/chat/completions`` first -- same vendor, opposite
order. "First supported" would target the Anthropic wire, which these handlers
cannot produce a body for. Selection intersects with what the proxy can
actually build.

**No published endpoints means no constraint.** 17 of 40 live models omit the
key. Treating that as an empty allow-list would break the entire ``gpt-4*``
family, so it falls through to the heuristic that serves them correctly today.

The planner is pure: no I/O, no globals, fully unit-testable. It never raises;
an unusable catalog degrades to the legacy behaviour.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from headroom.models.copilot_catalog import ModelCard

logger = logging.getLogger(__name__)

RESPONSES_PATH = "/responses"
CHAT_COMPLETIONS_PATH = "/chat/completions"

BRIDGE_RESPONSES_TO_CHAT = "responses->chat"
BRIDGE_CHAT_TO_RESPONSES = "chat->responses"

#: Bridges this proxy has actually implemented. The planner asserts against
#: this set so it can never emit a plan the request path cannot execute --
#: the failure mode that would make catalog routing worse than the heuristic.
IMPLEMENTED_REQUEST_BRIDGES: frozenset[str] = frozenset({BRIDGE_RESPONSES_TO_CHAT})


@dataclass(frozen=True, slots=True)
class TransportPlan:
    """How to send one request upstream."""

    upstream_path: str
    request_bridge: str | None = None
    drop_fields: tuple[str, ...] = ()
    clamp: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    catalog_backed: bool = False

    @property
    def executable(self) -> bool:
        """False when the plan names a bridge that is not implemented.

        Callers must check this and fall back rather than attempting the plan.
        """
        return self.request_bridge is None or self.request_bridge in IMPLEMENTED_REQUEST_BRIDGES


def plan_transport(
    *,
    inbound_path: str,
    card: ModelCard | None,
    heuristic_prefers_responses: bool,
    reasoning_effort: str | None = None,
    requested_max_output_tokens: int | None = None,
    will_buffer: bool = True,
) -> TransportPlan:
    """Return the transport plan for one request.

    ``card`` is ``None`` when the catalog does not know the model (or is
    disabled), in which case ``heuristic_prefers_responses`` -- the existing
    name-based answer -- decides, guaranteeing no regression against today.

    ``will_buffer`` reflects that bridged requests are sent with ``stream:
    false`` and replayed; it enables the non-streaming output clamp, which
    matters because every Claude model caps non-streaming output at 16k against
    a 64k streaming ceiling.
    """
    if card is None:
        target = RESPONSES_PATH if heuristic_prefers_responses else CHAT_COMPLETIONS_PATH
        return _finish(
            inbound_path=inbound_path,
            target=target,
            card=None,
            reasoning_effort=reasoning_effort,
            requested_max_output_tokens=requested_max_output_tokens,
            will_buffer=will_buffer,
            reason=("model unknown to catalog; using name heuristic"),
            catalog_backed=False,
        )

    if not card.constrains_endpoints():
        target = RESPONSES_PATH if heuristic_prefers_responses else CHAT_COMPLETIONS_PATH
        return _finish(
            inbound_path=inbound_path,
            target=target,
            card=card,
            reasoning_effort=reasoning_effort,
            requested_max_output_tokens=requested_max_output_tokens,
            will_buffer=will_buffer,
            reason=(f"{card.id} publishes no supported_endpoints; using name heuristic"),
            catalog_backed=True,
        )

    # Keep the inbound wire whenever the model actually serves it -- the
    # cheapest correct answer, and the one that avoids a bridge entirely.
    if card.supports_endpoint(inbound_path):
        return _finish(
            inbound_path=inbound_path,
            target=inbound_path,
            card=card,
            reasoning_effort=reasoning_effort,
            requested_max_output_tokens=requested_max_output_tokens,
            will_buffer=False,
            reason=f"{card.id} serves {inbound_path} natively",
            catalog_backed=True,
        )

    candidates = card.bridgeable_endpoints()
    if not candidates:
        # The model is served only on wires these handlers cannot build (for
        # example ``/v1/messages`` or ``ws:/responses`` only). Forwarding
        # unchanged at least reproduces today's behaviour and yields a clear
        # upstream error, rather than inventing a body we cannot produce.
        return _finish(
            inbound_path=inbound_path,
            target=inbound_path,
            card=card,
            reasoning_effort=reasoning_effort,
            requested_max_output_tokens=requested_max_output_tokens,
            will_buffer=False,
            reason=(
                f"{card.id} publishes only non-bridgeable endpoints "
                f"{card.endpoints}; forwarding {inbound_path} unchanged"
            ),
            catalog_backed=True,
        )

    target = candidates[0]
    return _finish(
        inbound_path=inbound_path,
        target=target,
        card=card,
        reasoning_effort=reasoning_effort,
        requested_max_output_tokens=requested_max_output_tokens,
        will_buffer=will_buffer,
        reason=(
            f"{card.id} does not serve {inbound_path}; "
            f"routing to {target} (published: {list(card.endpoints)})"
        ),
        catalog_backed=True,
    )


def _finish(
    *,
    inbound_path: str,
    target: str,
    card: ModelCard | None,
    reasoning_effort: str | None,
    requested_max_output_tokens: int | None,
    will_buffer: bool,
    reason: str,
    catalog_backed: bool,
) -> TransportPlan:
    request_bridge: str | None = None
    if target != inbound_path:
        if inbound_path == RESPONSES_PATH and target == CHAT_COMPLETIONS_PATH:
            request_bridge = BRIDGE_RESPONSES_TO_CHAT
        elif inbound_path == CHAT_COMPLETIONS_PATH and target == RESPONSES_PATH:
            request_bridge = BRIDGE_CHAT_TO_RESPONSES

    clamp: dict[str, Any] = {}
    drop: list[str] = []

    # reasoning_effort: clamp to the nearest accepted rung, or drop when the
    # target model publishes no support. A blanket drop would silently
    # downgrade Opus-class models that do support it; forwarding verbatim 400s
    # on a value the model does not accept. Both were observed live.
    if reasoning_effort is not None:
        if card is None:
            # No capability data. The safe legacy behaviour on a bridged
            # request is to drop, since that is what shipped before.
            if request_bridge is not None:
                drop.append("reasoning_effort")
        else:
            clamped = card.clamp_reasoning_effort(reasoning_effort)
            if clamped is None:
                drop.append("reasoning_effort")
            elif clamped != reasoning_effort:
                clamp["reasoning_effort"] = clamped

    # Buffered (bridged) requests cannot exceed the non-streaming output cap.
    if (
        will_buffer
        and card is not None
        and card.max_non_streaming_output_tokens
        and requested_max_output_tokens
        and requested_max_output_tokens > card.max_non_streaming_output_tokens
    ):
        clamp["max_output_tokens"] = card.max_non_streaming_output_tokens

    plan = TransportPlan(
        upstream_path=target,
        request_bridge=request_bridge,
        drop_fields=tuple(drop),
        clamp=clamp,
        reason=reason,
        catalog_backed=catalog_backed,
    )

    if not plan.executable:
        # Refuse to hand back an unexecutable plan: forward on the inbound wire
        # instead, which reproduces today's behaviour rather than attempting a
        # translation that does not exist.
        logger.info(
            "transport planner: %s requires the unimplemented %s bridge; "
            "forwarding on %s unchanged",
            card.id if card else "model",
            plan.request_bridge,
            inbound_path,
        )
        return TransportPlan(
            upstream_path=inbound_path,
            request_bridge=None,
            drop_fields=tuple(drop),
            clamp=clamp,
            reason=f"{reason}; bridge {plan.request_bridge} not implemented, forwarded unchanged",
            catalog_backed=catalog_backed,
        )
    return plan
