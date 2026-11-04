"""Wire-format contract policy for the buffered (non-streaming) reply path.

The problem
-----------

The buffered Anthropic path returns the upstream reply with its headers
copied wholesale::

    response_headers = dict(response.headers)
    response_headers.pop("content-encoding", None)
    response_headers.pop("content-length", None)
    ...
    return Response(content=..., status_code=..., headers=response_headers)

``content-type`` rides along untouched. When the upstream answers a
``stream``-less request with ``text/event-stream``, that body reaches a
caller that asked for JSON, as a ``200`` it cannot parse. Clients report
it as an empty or malformed response and the turn is lost — the reply is
*present and complete*, just wearing the wrong wire format.

The buffered-stream (CCR) path already refuses this shape, logging the
offending ``content-type`` and returning ``upstream_protocol_error``
(#2952). The plain non-streaming path never got the same treatment: an
unparseable body there was assumed to mean "no CCR handling", so it was
logged at DEBUG and passed through.

The contract
------------

A caller that did not set ``stream: true`` must never receive an
event-stream body. Headroom owns both ends of that boundary, so it can
enforce it rather than let the mismatch reach the client.

Behaviour matrix
----------------

============================  ==============  =========  ====================
Client asked for streaming?   Upstream C-T    Status     Result
============================  ==============  =========  ====================
yes                           any             any        untouched
no                            application/…   any        untouched
no                            text/event-…    != 200     untouched (real error)
no                            text/event-…    200        recover, else refuse
============================  ==============  =========  ====================

Recovery reuses ``StreamingMixin._parse_sse_to_response``, the same
reconstruction the streaming path already runs for usage accounting, so
this adds no new parsing surface. Recovering in place — rather than
returning early — keeps the rest of the buffered path (CCR, turn hooks,
security scan, usage accounting) operating on a normal reply.

Non-200 is deliberately excluded: an error status is already actionable
by the client, and passing it through unchanged preserves the upstream's
own error payload.

Public API
----------

* :func:`is_event_stream` — media-type test, parameter- and case-tolerant.
* :func:`should_recover_sse_reply` — the gate above, as one predicate.

Header correction is *not* here — see the note beside the public functions.

Constraints (per project memory)
--------------------------------

* pure: no I/O, no logging, no config — the handler owns those.
* no regexes: media-type parsing is a single ``split``.
* no silent fallbacks: the caller refuses loudly when recovery fails.
"""

from __future__ import annotations

SSE_MEDIA_TYPE = "text/event-stream"
JSON_MEDIA_TYPE = "application/json"


def media_type(content_type: str | None) -> str:
    """Return the bare media type, lower-cased, with parameters dropped.

    ``"text/event-stream; charset=utf-8"`` and ``"Text/Event-Stream"`` both
    yield ``"text/event-stream"``. Returns ``""`` for a missing header.
    """
    if not content_type:
        return ""
    return content_type.split(";", 1)[0].strip().lower()


def is_event_stream(content_type: str | None) -> bool:
    """True when ``content_type`` denotes an SSE body."""
    return media_type(content_type) == SSE_MEDIA_TYPE


def should_recover_sse_reply(
    *,
    client_requested_stream: bool,
    status_code: int,
    content_type: str | None,
    body_is_event_stream: bool = False,
) -> bool:
    """True when a buffered reply violates the caller's non-streaming contract.

    See the behaviour matrix in the module docstring. The three negative
    arms are all deliberate: a streaming caller *wants* SSE, a JSON
    content-type is already correct, and a non-200 carries an upstream
    error the client should see verbatim.

    ``body_is_event_stream`` covers the reply that *is* an event stream while
    saying otherwise — a mislabeled or absent ``content-type``. Trusting the
    declared type alone would let exactly the same unparseable body through,
    so the caller sniffs the payload and passes the answer in. It stays a
    parameter rather than an import because this module is pure: the sniff
    needs the response object, and the handler owns that.
    """
    if client_requested_stream:
        return False
    if status_code != 200:
        return False
    return is_event_stream(content_type) or body_is_event_stream


# Header correction deliberately lives in
# ``helpers.sanitize_forwarded_response_headers`` rather than here. It already
# owns ``FRAMING_RESPONSE_HEADERS`` and strips ``connection``, ``keep-alive``
# and ``server`` alongside the content-* family — a second copy of that list
# would drift, and the ones this module would have missed are load-bearing:
# leaving ``transfer-encoding`` on a rebuilt body is what produced an empty
# HTTP 200 in #3019, and ``server: cloudflare`` is one of the headers the
# client cites as evidence of an intermediary.
