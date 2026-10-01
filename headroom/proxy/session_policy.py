"""Session attribution policy for proxy savings.

Claude Code stamps every request with ``x-claude-code-session-id`` (one UUID per interactive session, present on every POST), and ``headroom wrap`` can inject the transport-neutral ``x-headroom-session-id`` via ``ANTHROPIC_CUSTOM_HEADERS`` for harnesses that have no session header of their own. Savings attribution takes the first present of the two; when neither is sent the request simply stays unattributed. A key is never fabricated: the compression session key's md5 fallback identifies a conversation, not a client session, and would silently merge distinct sessions' savings.

The id travels to the outcome funnel on the request's ``tags`` under an internal (underscore-prefixed) key rather than a handler parameter so every outcome-emitting path picks it up from one place, the same way the tool-schema and extension-savings ledgers ride ``tags`` (:mod:`headroom.proxy.savings_attribution`).
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from headroom.proxy.savings_tracker import sanitize_project_name

#: Harness-native session header (Claude Code sends it on every request).
CLAUDE_CODE_SESSION_HEADER = "x-claude-code-session-id"
#: Headroom-injected fallback for harnesses with no session header of their own.
HEADROOM_SESSION_HEADER = "x-headroom-session-id"

#: Internal ``tags`` key carrying the resolved id to the outcome funnel.
#: Listed in ``savings_attribution._INTERNAL_TAGS`` so it never reaches the
#: request log's string-label store.
SESSION_TAG = "_headroom_session_id"


def sanitize_session_id(value: Any) -> str | None:
    """Normalize a client-supplied session id; ``None`` when unusable.

    Same rules as project names (printable characters, length-capped) via the same sanitizer, so a misbehaving client cannot bloat the persisted per-session map or dashboard payloads with arbitrary bytes.
    """
    return sanitize_project_name(value)


def classify_session(headers: Any) -> str | None:
    """Extract the savings-attribution session id from request headers, if present.

    ``x-claude-code-session-id`` wins when both are sent: it is the id the harness itself scopes by, so a wrap-injected ``x-headroom-session-id`` must not override it.
    """
    get = getattr(headers, "get", None)
    if get is None:
        return None
    value = get(CLAUDE_CODE_SESSION_HEADER) or get(HEADROOM_SESSION_HEADER)
    return sanitize_session_id(value)


def bind_session_tag(tags: MutableMapping[str, Any], headers: Any) -> None:
    """Stash the resolved session id on the per-request ``tags``, when there is one.

    Unattributed requests are left untouched rather than stamped with a sentinel, so the funnel can treat absence as "no session attribution".
    """
    session = classify_session(headers)
    if session is not None:
        tags[SESSION_TAG] = session


def session_from_tags(tags: MutableMapping[str, Any] | None) -> str | None:
    """Session id bound to this request's tags, or ``None``.

    Re-sanitized rather than trusted: ``tags`` is a plain dict every handler and extension can write to, so the guarantee has to hold at the read.
    """
    raw = (tags or {}).get(SESSION_TAG)
    return sanitize_session_id(raw)


__all__ = [
    "CLAUDE_CODE_SESSION_HEADER",
    "HEADROOM_SESSION_HEADER",
    "SESSION_TAG",
    "bind_session_tag",
    "classify_session",
    "sanitize_session_id",
    "session_from_tags",
]
