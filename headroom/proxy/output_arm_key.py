"""The output-holdout key for an HTTP ``/v1/responses`` request.

A conversation must keep one holdout arm for its whole life, or the output
ledger compares a conversation against itself. Two things move the body-derived
key (:func:`conversation_key_from_body`) between turns of one conversation:

* ``previous_response_id``: the body carries only this turn's increment, so its
  first user turn is the new prompt, not the opener.
* Nothing else in the body names the conversation, while a header does.

So ``x-headroom-session-id`` wins, then the key of the response a request
continues, then the body. The header is keyed exactly as headroom-kiro keys it,
so the extension's steering decision and core's arm stay the same. Kiro resends
the whole transcript, so it agrees with the body key when the header is absent.
An extension that must agree on every request should call
:func:`output_arm_key` rather than repeat the rule.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

from .output_savings_policy import conversation_key_from_body

# The Headroom-specific header is also used by headroom-kiro for steering.
_SESSION_HEADERS = ("x-headroom-session-id",)

# Response id -> the arm key of the request that produced it. In memory and per
# process: after a restart, or on another replica, an unknown id falls back to
# the body key. Bounded so a long-lived proxy cannot grow without limit.
_MAX_RESPONSES = 4096
_keys_by_response: OrderedDict[str, str] = OrderedDict()
_lock = threading.Lock()


def output_arm_key(body: dict[str, Any], headers: Mapping[str, str]) -> str:
    """The holdout key for a Responses request: session header, response chain, body."""
    for name in _SESSION_HEADERS:
        value = headers.get(name)
        if value:
            return f"header:{name}:{value}"
    previous = body.get("previous_response_id")
    if isinstance(previous, str) and previous:
        with _lock:
            known = _keys_by_response.get(previous)
            if known is not None:
                _keys_by_response.move_to_end(previous)
                return known
    return conversation_key_from_body(body)


def remember_response_arm_key(response_id: Any, arm_key: str | None) -> None:
    """Record that ``response_id`` belongs to the conversation keyed ``arm_key``."""
    if not isinstance(response_id, str) or not response_id or not arm_key:
        return
    with _lock:
        _keys_by_response[response_id] = arm_key
        _keys_by_response.move_to_end(response_id)
        while len(_keys_by_response) > _MAX_RESPONSES:
            _keys_by_response.popitem(last=False)
