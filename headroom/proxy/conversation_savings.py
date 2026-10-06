"""Novel-vs-repeat attribution for per-request compression savings.

Providers disagree about what a request's ``tokens_saved`` means, and the
disagreement only shows up once something sums it.

Anthropic's cached prefix is frozen -- the router leaves it alone so the
provider's prefix cache keeps hitting -- so a turn's ``tokens_saved`` covers
only content that newly entered the conversation. Summing across turns counts
each removed token once.

OpenAI's ``/v1/responses`` carries the whole transcript in every request and
the router recompresses all of it, so a turn's ``tokens_saved`` is the running
total of everything removed from that conversation so far. Summing across
turns counts the same removed token once per remaining turn: roughly twenty
times on a twenty-turn conversation.

Measured 2026-09-08 across one machine's retained proxy logs: gpt/codex models
reported 8.10M tokens saved against 3.90M tokens of new input (2.1x, which no
share of new content can be), where Claude reported 4.18M against 43.3M. Two
requests from the same machine and the same minute::

    gpt-6-astra   tok_before=156260 tok_after=80240 tok_saved=76020 cache_read=66304
    claude-opus-5 tok_before=114024 tok_after=113566 tok_saved=458   cache_read=98503

Every cumulative consumer inherits the inflation: lifetime savings totals, the
per-model savings percent, the dollar figure (repeat removals priced at the
full uncached input rate when the alternative was a cache read at ~0.1x), and
any rate dividing savings by new input, which then has no 100% ceiling.

This module converts the cumulative series into the incremental one. Per
request descriptions keep the wire truth -- ``tok_before``/``tok_after``, the
request log and the transformations feed all still say what left this process
-- because those tokens really were removed from this request's payload. Only
the running totals switch to the novel figure, so a removed token is counted
once per conversation instead of once per turn.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any

from headroom.proxy.output_savings_policy import _unwrap_response_create_body

__all__ = [
    "ConversationSavings",
    "get_conversation_savings",
    "reset_conversation_savings",
    "savings_conversation_key",
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
    """Identity under which a Responses request's ``tokens_saved`` is a running total.

    Returns ``None`` unless BOTH premises of the cumulative accounting hold, in
    which case the funnel keeps ordinary per-request accounting:

    1. The request names one conversation. Only an explicit id counts: a
       top-level ``conversation_id``/``session_id``/``thread_id``, one of
       those inside ``metadata``/``client_metadata``, or a transport-scoped
       ``session_id`` the caller vouches for (a WebSocket, a session header).
       Two things are deliberately NOT accepted. The holdout key's fallbacks
       (the instructions prefix, the literal ``responses``): two independent
       conversations with the same instructions would share a running total
       and suppress each other's savings. And ``prompt_cache_key``: it groups
       prompt-cache routing, and OpenAI documents one key shared across a
       user's sessions and forks, or across independent single-turn judging
       requests, so it aliases conversations the same way. An adapter that
       establishes a client sets it uniquely per conversation may pass it as
       ``session_id``; nothing here does.
    2. The request carries the whole transcript. With ``previous_response_id``
       or a server-side ``conversation`` the provider holds prior context and
       the payload is this turn's increment, so ``tokens_saved`` is already
       per-request and must not be differenced.

    Chat-completions bodies (no ``input``) return ``None``: their handlers
    freeze the cached prefix, so their ``tokens_saved`` is novel-only already.
    """
    if not isinstance(body, dict):
        return None
    body = _unwrap_response_create_body(body)
    if "input" not in body:
        return None
    if body.get("previous_response_id") or body.get("conversation"):
        return None

    identity = ""
    for key in ("conversation_id", "session_id", "thread_id"):
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
                "session_id",
                "thread_id",
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


# Conversations tracked before the oldest is forgotten. A forgotten
# conversation that is still live restarts from zero and re-counts its
# transcript once, so the cap trades a bounded, one-off overcount for bounded
# memory. 512 is far above the number of conversations one client keeps warm.
DEFAULT_MAX_CONVERSATIONS = 512


class ConversationSavings:
    """Cumulative-per-turn savings to novel-this-turn savings, bounded."""

    def __init__(self, max_conversations: int = DEFAULT_MAX_CONVERSATIONS) -> None:
        self._max = max(1, int(max_conversations))
        self._seen: OrderedDict[str, int] = OrderedDict()
        self._lock = threading.Lock()

    def novel(self, conversation_key: str | None, cumulative: int | None) -> int | None:
        """Tokens this request removed for the first time in its conversation.

        ``cumulative`` is the conversation's running removed-token total as of
        this request. Returns ``None`` when the caller cannot supply both a
        conversation key and a total, which means "this path does not
        distinguish" -- the funnel then falls back to ``tokens_saved``, which
        is already novel-only on providers that freeze their cached prefix.
        """
        if not conversation_key or cumulative is None:
            return None

        total = max(0, int(cumulative))
        with self._lock:
            previous = self._seen.get(conversation_key, 0)
            self._seen[conversation_key] = total
            self._seen.move_to_end(conversation_key)
            while len(self._seen) > self._max:
                self._seen.popitem(last=False)

        # A total that went DOWN means the transcript shrank under it: a
        # compaction, or a client that dropped history. Nothing was removed
        # for the first time, and the next turn counts from the lower base
        # rather than waiting for the old high-water mark to be re-reached.
        return max(0, total - previous)


_default = ConversationSavings()


def get_conversation_savings() -> ConversationSavings:
    """Process-wide ledger. One conversation spans many requests."""
    return _default


def reset_conversation_savings() -> None:
    """Forget every conversation. Test helper only."""
    global _default
    _default = ConversationSavings()
