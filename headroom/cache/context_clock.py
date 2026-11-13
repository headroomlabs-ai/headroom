"""Conversation-scoped request clocks, separate from retrievable payloads."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ContextTurnSnapshot:
    current_turn: int
    # hash -> (compression event creation timestamp, original conversation turn)
    compression_turns: dict[str, tuple[float, int]]
    compression_event_ids: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class ContextClockBackend(Protocol):
    """Optional capability; implementations atomically advance and retain age."""

    def observe_context_turn(
        self, conversation_key: str, hash_keys: Collection[str], *, namespace_key: str | None = None
    ) -> ContextTurnSnapshot | None: ...


def context_conversation_key(
    session_id: str,
    workspace_key: str,
    messages: list[dict[str, Any]],
    *,
    explicit_session: bool,
) -> str:
    """Use client identity, or include the original conversation's first user turn.

    Model/system fallback IDs can serve independent agents in one workspace.
    The initial user message separates their origins without depending on the
    latest query, process-local tracker IDs, or optimized history. Byte-identical
    origins need explicit client session IDs to be distinguishable.
    """
    origin = None
    if not explicit_session:
        from .prefix_tracker import _canonicalize_for_prefix_compare

        first_user = next((message for message in messages if message.get("role") == "user"), None)
        if first_user is not None:
            origin = _canonicalize_for_prefix_compare([first_user])
    encoded = json.dumps([session_id, workspace_key, origin], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def context_conversation_namespace(session_id: str, workspace_key: str) -> str:
    encoded = json.dumps([session_id, workspace_key], separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def resolve_context_conversation(
    conversation_key: str,
    namespace_key: str | None,
    states: Mapping[str, Any],
    events: Mapping[str, tuple[float, float, str]],
) -> str:
    """Follow retained event lineage when a fallback origin was trimmed.

    Exact origins keep their clocks. A new origin may continue one retained
    clock in the same fallback session/workspace namespace. Multiple matching
    clocks cannot establish identity and must skip proactive expansion.
    """
    if namespace_key is None:
        return conversation_key
    prior = states.get(conversation_key)
    if isinstance(prior, dict) and prior.get("namespace") in (None, "explicit"):
        # Bind an exact older origin to its namespace without restarting age.
        return conversation_key
    candidates = []
    for key, state in states.items():
        if not isinstance(state, dict):
            raise ValueError("Invalid retained CCR conversation clock")
        namespace = state.get("namespace")
        if namespace not in (namespace_key, None):
            continue
        anchors = state.get("events")
        if not isinstance(anchors, dict):
            raise ValueError("Invalid retained CCR conversation anchors")
        for hash_key, (_, _, event_id) in events.items():
            anchor = anchors.get(hash_key)
            if anchor is None:
                continue
            if not isinstance(anchor, dict):
                raise ValueError("Invalid retained CCR compression anchor")
            retained_id = anchor.get("event_id")
            if state.get("version") == 1:
                retained_id = f"legacy:{float(anchor['created_at'])!r}"
            if not isinstance(retained_id, str) or not retained_id:
                raise ValueError("Invalid retained CCR compression identity")
            if retained_id == event_id:
                if namespace is None:
                    raise ValueError("Retained CCR lineage has no verified namespace")
                candidates.append(key)
                break
    if len(candidates) > 1:
        raise ValueError("Ambiguous retained CCR conversation lineage")
    if conversation_key in states:
        if candidates and candidates[0] != conversation_key:
            raise ValueError("Conflicting retained CCR conversation lineage")
        return conversation_key
    return candidates[0] if candidates else conversation_key


def advance_context_state(
    state: dict[str, Any] | None,
    events: Mapping[str, tuple[float, float, str]],
    live_hashes: Collection[str],
    now: float,
) -> tuple[dict[str, Any], ContextTurnSnapshot, float]:
    """Advance one owned-marker request; caller supplies its atomic boundary.

    Clock retention outlives every referenced compression event. An event's
    first turn survives missing markers, restarts, and cache eviction in the
    in-process ContextTracker. A new store event identity resets only that
    event. Invalid durable state must fail closed, never start a fresh clock.
    """
    if state is None:
        state = {"version": 1, "turn": 0, "events": {}}
    if (
        not isinstance(state, dict)
        or state.get("version") not in (1, 2)
        or type(state.get("turn")) is not int
        or state["turn"] < 0
        or not isinstance(state.get("events"), dict)
    ):
        raise ValueError("Invalid retained CCR conversation clock")
    anchors = {}
    for hash_key, anchor in state["events"].items():
        if (
            not isinstance(anchor, dict)
            or type(anchor.get("turn")) is not int
            or not 1 <= anchor["turn"] <= state["turn"]
            or not isinstance(anchor.get("created_at"), (int, float))
            or not isinstance(anchor.get("expires_at"), (int, float))
            or not math.isfinite(anchor["created_at"])
            or not math.isfinite(anchor["expires_at"])
        ):
            raise ValueError("Invalid retained CCR compression turn")
        if hash_key in live_hashes and anchor["expires_at"] >= now:
            retained_anchor = anchor.copy()
            # Version 1 used timestamps as identity; retain legacy payload age
            # without assigning a fresh first turn during upgrade.
            if state["version"] == 1:
                retained_anchor["event_id"] = f"legacy:{float(anchor['created_at'])!r}"
            if (
                not isinstance(retained_anchor.get("event_id"), str)
                or not retained_anchor["event_id"]
            ):
                raise ValueError("Invalid retained CCR compression identity")
            anchors[hash_key] = retained_anchor
    current_turn = state["turn"] + 1
    observed = {}
    event_ids = {}
    for hash_key, (created_at, expires_at, event_id) in events.items():
        anchor = anchors.get(hash_key)
        if anchor is None or anchor["event_id"] != event_id:
            anchor = {
                "created_at": created_at,
                "event_id": event_id,
                "turn": current_turn,
                "expires_at": expires_at,
            }
            anchors[hash_key] = anchor
        observed[hash_key] = (created_at, anchor["turn"])
        event_ids[hash_key] = event_id
    retained = {"version": 2, "turn": current_turn, "events": anchors}
    expires_at = max(anchor["expires_at"] for anchor in anchors.values())
    return retained, ContextTurnSnapshot(current_turn, observed, event_ids), expires_at
