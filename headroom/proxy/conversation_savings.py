"""Which conversation a ``/v1/responses`` request belongs to.

The keys here identify a conversation for attribution (pipeline extensions
read it from ``OutcomeSnapshot``) and for the per-request savings record
(``headroom.proxy.savings.prefix_key``), which compares a request with its
conversation's previous one.

Compression savings are booked per request on every path. Responses re-sends
the whole transcript every turn and the router recompresses all of it, so a
turn's ``tokens_saved`` includes removals from earlier turns, booked again.
Anthropic does the same: the frozen cached prefix is re-sent with its
removals, and a ``router:noop`` turn books the session's removals again. Both
paths count a token every time it is removed from a request that reaches the
provider. The cache-aware price of re-sent removals is the savings record's
job (#4238), the same for both providers.
"""

from __future__ import annotations

import hashlib
from typing import Any

from headroom.proxy.output_savings_policy import _unwrap_response_create_body

__all__ = [
    "savings_conversation_key",
    "turn_conversation_key",
]

_ID_KEYS = ("id", "conversation_id", "session_id", "thread_id")


def _explicit_id(value: Any) -> str:
    if isinstance(value, str):
        return value if value and value.lower() != "auto" else ""
    if isinstance(value, dict):
        for key in _ID_KEYS:
            nested = value.get(key)
            if isinstance(nested, str) and nested:
                return nested
    return ""


def savings_conversation_key(body: Any, *, session_id: str | None = None) -> str | None:
    """Identity under which a Responses request is compared with its conversation's last.

    Returns ``None`` unless BOTH premises hold:

    1. The request names one conversation. Only an explicit id counts: a
       top-level ``conversation_id``/``session_id``/``thread_id``, one of
       those inside ``metadata``/``client_metadata``, or a transport-scoped
       ``session_id`` the caller vouches for (a WebSocket, a session header).
       A ``thread_id`` wins over a ``session_id`` beside it: Codex sends both,
       and its ``session_id`` is the root thread's on every request of the
       session, so a guardian or subagent request would otherwise replace
       the root's previous request.
       Two things are deliberately NOT accepted. The holdout key's fallbacks
       (the instructions prefix, the literal ``responses``): two independent
       conversations with the same instructions would share one previous
       request. And ``prompt_cache_key``: it groups
       prompt-cache routing, and OpenAI documents one key shared across a
       user's sessions and forks, or across independent single-turn judging
       requests, so it aliases conversations the same way. An adapter that
       establishes a client sets it uniquely per conversation may pass it as
       ``session_id``; nothing here does.
    2. The request carries the whole transcript. With ``previous_response_id``
       or a server-side ``conversation`` the provider holds prior context and
       the payload is this turn's increment, not a transcript to compare.

    Chat-completions bodies (no ``input``) return ``None``.
    """
    if not isinstance(body, dict):
        return None
    body = _unwrap_response_create_body(body)
    if "input" not in body:
        return None
    if body.get("previous_response_id") or body.get("conversation"):
        return None
    return turn_conversation_key(body, session_id=session_id)


def turn_conversation_key(body: Any, *, session_id: str | None = None) -> str | None:
    """The conversation a Responses request names, or ``None`` when it names none.

    Premise 1 of :func:`savings_conversation_key` only, with the same ids and
    the same digest: an incremental body (``previous_response_id``) still
    belongs to its conversation, so this is the key turn hooks see as
    ``TurnContext.conversation`` on every turn. So does a body that names a
    server-side ``conversation`` (an id, or an object with one), which
    :func:`savings_conversation_key` excludes before calling this.
    """
    if not isinstance(body, dict):
        return None
    body = _unwrap_response_create_body(body)
    identity = ""
    for key in ("conversation_id", "thread_id", "session_id", "conversation"):
        value = _explicit_id(body.get(key))
        if value:
            identity = f"{key}:{value}"
            break
    if not identity:
        for container_key in ("client_metadata", "metadata"):
            container = body.get(container_key)
            if not isinstance(container, dict):
                continue
            for key in (
                "conversation_id",
                "conversation_key",
                "thread_id",
                "session_id",
                "codex_session_id",
            ):
                value = _explicit_id(container.get(key))
                if value:
                    identity = f"{container_key}.{key}:{value}"
                    break
            if identity:
                break
    if not identity and session_id:
        identity = f"session:{session_id}"
    if not identity:
        return None
    return hashlib.sha256(("savings\x00" + identity).encode("utf-8", "ignore")).hexdigest()
