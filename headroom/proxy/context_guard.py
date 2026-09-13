"""Context-limit guard: keep client auto-compaction working under compression.

Compression makes the usage reported back to the client reflect the
*forwarded* (compressed) request, not the client's raw transcript. Clients
whose auto-compaction keys off reported usage (Claude Code) therefore defer
compaction — the selling point — but with no floor: the raw transcript can
grow until even the compressed request exceeds the model's real context
window. From that point every turn fails with a 400 ``prompt is too long``,
the client force-compacts on each error, and the session degrades into a
compact-every-other-prompt loop (field report 2026-08-12).

Three cooperating pieces, all passthrough until the danger zone:

1. ``effective_context_limit`` — the window the forwarded request is really
   subject to: the model's registry limit, raised to 1M when the client sent
   a ``context-1m`` beta token, capped by any limit *learned* from an actual
   ``prompt is too long`` error (piece 2). Also used for compression
   pressure, so 1M sessions stop being treated as 5x over budget.
2. ``note_prompt_too_long`` — parses ``prompt is too long: N tokens > M
   maximum`` from upstream 400 bodies and records M per (model, 1m-beta)
   key. This is how a client that *believes* 1M but whose account is capped
   at 200k stops looping after exactly one error.
3. ``StreamUsageGuard`` — an SSE transformer for the streamed bytes only:
   when the ``message_start`` usage total reaches ``TRIGGER_FRACTION`` of
   the effective limit, it inflates the reported ``input_tokens`` — in both
   ``message_start`` and the final cumulative-usage ``message_delta``, which
   clients merge over the former — so the client's gauge reads
   ``REPORT_FRACTION`` of the window the client believes in, which trips
   the client's own graceful compaction before the 400 wall. Savings
   metrics and telemetry parse the original upstream bytes, never the
   nudged ones.

The nudged number is deliberately unmarked on the wire. It exists to be read
by the client's own compaction gauge, which is a third-party parser we do not
control, so anything added beside it is either ignored or a wire-shape change
against a client we cannot test. Nothing inside headroom needs the marker:
savings, telemetry and billing all read the original upstream bytes, never the
client-bound ones, and each rewrite emits ``event=context_guard_nudge`` with
the forwarded and reported totals for reconciliation.

Kill switch: ``HEADROOM_CONTEXT_GUARD=0``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any

logger = logging.getLogger("headroom.proxy")

# Forwarded-usage fraction of the *effective* (real) limit at which the nudge
# arms. 0.90 leaves one normal turn of headroom before the hard wall.
TRIGGER_FRACTION = 0.90
# Reported-usage fraction of the *believed* limit the nudge inflates to.
# Claude Code auto-compacts around 92%; 0.95 clears that with margin.
REPORT_FRACTION = 0.95

_CONTEXT_1M_PREFIX = "context-1m"

_PROMPT_TOO_LONG_RE = re.compile(
    r"prompt is too long:\s*(\d+)\s*tokens?\s*>\s*(\d+)\s*maximum", re.IGNORECASE
)

# A learned maximum is believed only inside this band. Below the floor it is
# not a context window: gateways return the same "prompt is too long: N > M"
# shape for per-request quota and per-key throttles, and one such number would
# clamp every later request on that model for the life of the process. Above
# the ceiling it cannot cap anything anyway.
_MIN_LEARNED_LIMIT = 8_000
_MAX_LEARNED_LIMIT = 10_000_000

# A learned limit is evidence about one account at one moment, not a fact about
# the model: entitlements get provisioned, gateways get repointed. It expires so
# a stale cap cannot hold a budget down forever -- the cost of expiry is at most
# one more prompt-too-long error, which is what taught it the first time.
_LEARNED_LIMIT_TTL_SECONDS = 3600.0

# (model, has_1m_beta, credential scope) -> (maximum, learned-at monotonic time).
#
# The 1m-beta dimension matters: an account with real 1M access must not inherit
# a 200k limit learned from a request that never sent the beta. The credential
# scope matters for the same reason one step out -- this dict is process-global,
# and on a shared proxy one tenant's capped key would otherwise clamp the budget
# of every other tenant on that model.
_learned_limits: dict[tuple[str, bool, str], tuple[int, float]] = {}

# Give up scanning for message_start beyond this many buffered bytes.
_MAX_GUARD_BUFFER = 256 * 1024


def context_guard_enabled() -> bool:
    """True unless explicitly disabled via HEADROOM_CONTEXT_GUARD=0."""
    from headroom.proxy import runtime_env

    return (runtime_env.getenv("HEADROOM_CONTEXT_GUARD", "1") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def has_context_1m_beta(beta_header: str | None) -> bool:
    """True when the anthropic-beta header carries a context-1m token."""
    if not beta_header:
        return False
    return any(
        token.strip().lower().startswith(_CONTEXT_1M_PREFIX) for token in beta_header.split(",")
    )


def believed_context_limit(model_limit: int, beta_header: str | None) -> int:
    """The context window the *client* thinks it has."""
    if has_context_1m_beta(beta_header):
        return max(model_limit, 1_000_000)
    return model_limit


def credential_scope(credential: str | None) -> str:
    """A stable, non-reversible tag for the upstream credential.

    Only ever a truncated digest: the credential itself is never stored, never
    logged, and never compared outside this function.
    """
    if not credential:
        return ""
    return hashlib.sha256(credential.strip().encode("utf-8", "ignore")).hexdigest()[:16]


def credential_scope_from_headers(headers: Any) -> str:
    """Credential scope for an outbound header mapping, best effort.

    Never raises: an unusable header mapping just means the unscoped bucket,
    which is the behaviour of a single-tenant proxy anyway.
    """
    try:
        for name in ("x-api-key", "authorization"):
            value = headers.get(name) or headers.get(name.title())
            if isinstance(value, str) and value:
                return credential_scope(value)
    except Exception:
        return ""
    return ""


def _learned_key(model: str, beta_header: str | None, scope: str) -> tuple[str, bool, str]:
    return (model, has_context_1m_beta(beta_header), scope)


def _learned_limit(key: tuple[str, bool, str]) -> int | None:
    """The unexpired learned maximum for ``key``, dropping it once stale."""
    entry = _learned_limits.get(key)
    if entry is None:
        return None
    maximum, learned_at = entry
    if time.monotonic() - learned_at >= _LEARNED_LIMIT_TTL_SECONDS:
        _learned_limits.pop(key, None)
        return None
    return maximum


def effective_context_limit(
    model: str,
    model_limit: int,
    beta_header: str | None,
    *,
    scope: str = "",
) -> int:
    """The window the forwarded request is actually subject to.

    Optimistically the believed limit; capped by a limit learned from a real
    ``prompt is too long`` error for the same (model, beta, credential) shape,
    and only while that lesson is still fresh.
    """
    believed = believed_context_limit(model_limit, beta_header)
    learned = _learned_limit(_learned_key(model, beta_header, scope))
    if learned is not None:
        return min(believed, learned)
    return believed


def note_prompt_too_long(
    model: str,
    beta_header: str | None,
    error_text: Any,
    *,
    scope: str = "",
) -> int | None:
    """Learn the real context limit from an upstream prompt-too-long error.

    Returns the learned maximum, or None when there is nothing to learn.

    This runs on the error path, where the client is still waiting for the real
    upstream body, so it accepts whatever the body turned out to be: upstreams
    and gateways send ``"message": null``, nested objects, and lists in the
    slot this reads, and a diagnostic helper must not turn one of those into an
    exception that replaces the error the caller was about to forward.
    """
    if isinstance(error_text, bytes | bytearray):
        error_text = bytes(error_text).decode("utf-8", errors="replace")
    if not isinstance(error_text, str):
        return None
    match = _PROMPT_TOO_LONG_RE.search(error_text)
    if not match:
        return None
    maximum = int(match.group(2))
    if not _MIN_LEARNED_LIMIT <= maximum <= _MAX_LEARNED_LIMIT:
        # Not a context window. Learning it would clamp the budget on a number
        # that never described one.
        logger.debug(
            "event=context_guard_limit_out_of_band model=%s limit=%d (ignored)",
            model,
            maximum,
        )
        return None
    key = _learned_key(model, beta_header, scope)
    previous = _learned_limit(key)
    _learned_limits[key] = (maximum, time.monotonic())
    if previous != maximum:
        logger.warning(
            "event=context_guard_learned_limit model=%s context_1m=%s limit=%d "
            "attempted=%s (client compaction gauge will be nudged near this limit)",
            model,
            key[1],
            maximum,
            match.group(1),
        )
    return maximum


def reset_learned_limits() -> None:
    """Test hook: clear limits learned from prompt-too-long errors."""
    _learned_limits.clear()


def _sse_event_type(event: bytes) -> bytes | None:
    """The event type of one SSE frame, or ``None`` when it carries none.

    Read from the ``event:`` line, not matched as a substring of the frame: a
    substring match reads the *payload* too, so an assistant that writes the
    words ``message_delta`` into its own reply ended the scan and lost the
    final nudge.

    ``None`` means the frame is not part of the message protocol at all --
    an SSE comment, a ``retry:`` field, the keepalives an intermediary shim
    injects. Those pass through without arming or disarming anything; treating
    one as an unknown event disabled the guard for the whole stream.
    """
    saw_data = False
    for line in event.split(b"\n"):
        if line.startswith(b"event:"):
            return line[len(b"event:") :].strip()
        if line.startswith(b"data:"):
            saw_data = True
    if not saw_data:
        return None
    # A data frame with no event: line defaults to the type inside the payload.
    for line in event.split(b"\n"):
        if not line.startswith(b"data:"):
            continue
        try:
            payload = json.loads(line[len(b"data:") :].strip())
        except Exception:
            return b""
        value = payload.get("type") if isinstance(payload, dict) else None
        return value.encode() if isinstance(value, str) else b""
    return b""


def nudged_input_tokens(
    usage: dict[str, Any],
    *,
    believed_limit: int,
    effective_limit: int,
) -> int | None:
    """The inflated ``input_tokens`` for this usage, or ``None`` to leave it be.

    One implementation for every client-bound surface -- ``message_start``, the
    final ``message_delta``, and the non-streaming JSON body -- so a request
    cannot get the raised compression budget on one path and the un-nudged
    gauge on another.
    """
    if believed_limit <= 0 or effective_limit <= 0:
        return None
    input_tokens = int(usage.get("input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
    total = input_tokens + cache_read + cache_creation
    if total < TRIGGER_FRACTION * effective_limit:
        return None
    target_total = int(believed_limit * REPORT_FRACTION)
    if total >= target_total:
        # Already reads as full to the client -- never deflate.
        return None
    return target_total - cache_read - cache_creation


def nudge_response_usage(
    resp_json: Any,
    *,
    believed_limit: int,
    effective_limit: int,
    request_id: str = "",
) -> bool:
    """Nudge the usage of a non-streaming Anthropic message body, in place.

    The buffered path raises the compression budget exactly like the streaming
    one, so it has to move the client's gauge exactly like the streaming one
    too; otherwise a non-streaming client gets the larger forwarded request and
    none of the warning that it is near the wall.

    Returns True when the body was modified.
    """
    if not isinstance(resp_json, dict):
        return False
    usage = resp_json.get("usage")
    if not isinstance(usage, dict):
        return False
    new_input = nudged_input_tokens(
        usage, believed_limit=believed_limit, effective_limit=effective_limit
    )
    if new_input is None:
        return False
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
    forwarded_total = int(usage.get("input_tokens") or 0) + cache_read + cache_creation
    usage["input_tokens"] = new_input
    logger.warning(
        "[%s] event=context_guard_nudge path=non_streaming forwarded_total=%d "
        "effective_limit=%d reported_total=%d believed_limit=%d (nudging client to "
        "compact before the prompt-too-long wall)",
        request_id,
        forwarded_total,
        effective_limit,
        new_input + cache_read + cache_creation,
        believed_limit,
    )
    return True


class StreamUsageGuard:
    """Rewrites near-the-wall usage in the client-bound SSE bytes.

    Feed every chunk headed to the client through :meth:`feed`; call
    :meth:`flush` when the stream ends. The first ``message_start`` decides
    whether the nudge is active. When it is not (the common case), the guard
    goes inert immediately and chunks pass through untouched — steady-state
    overhead is one ``if``. When it is, the guard rewrites the
    ``message_start`` usage AND stays armed for the final ``message_delta``:
    since mid-2025 that event carries cumulative input/cache usage and
    clients (verified live with Claude Code 2026-08-12) let it override the
    ``message_start`` values, so nudging only the first event loses the
    merge. Events are released as soon as their ``\\n\\n`` terminator
    arrives, so no latency is added beyond SSE's own event framing.
    """

    def __init__(
        self,
        *,
        believed_limit: int,
        effective_limit: int,
        request_id: str = "",
    ) -> None:
        self._believed_limit = believed_limit
        self._effective_limit = effective_limit
        self._request_id = request_id
        self._buf = bytearray()
        self._seen_message_start = False
        self._target_total: int | None = None
        self._start_cache_read = 0
        self._start_cache_creation = 0
        # Inert immediately when limits are unusable.
        self._done = believed_limit <= 0 or effective_limit <= 0

    def feed(self, chunk: bytes) -> bytes:
        if self._done:
            return chunk
        self._buf.extend(chunk)
        if len(self._buf) > _MAX_GUARD_BUFFER:
            return self._finish()
        out = bytearray()
        while not self._done:
            boundary = self._buf.find(b"\n\n")
            if boundary == -1:
                break
            event = bytes(self._buf[: boundary + 2])
            del self._buf[: boundary + 2]
            out += self._process_event(event)
        if self._done:
            out += self._finish()
        return bytes(out)

    def flush(self) -> bytes:
        """Return any held-back bytes; the guard goes inert."""
        return self._finish()

    def _finish(self) -> bytes:
        self._done = True
        remaining = bytes(self._buf)
        self._buf = bytearray()
        return remaining

    def _process_event(self, event: bytes) -> bytes:
        event_type = _sse_event_type(event)
        # Pings, comments, retry fields, keepalives: not part of the message
        # protocol, so they neither arm nor disarm the guard.
        if event_type is None or event_type == b"ping":
            return event
        if not self._seen_message_start:
            # Anything data-bearing that isn't message_start (error events,
            # protocol changes) ends the scan.
            self._seen_message_start = True
            if event_type != b"message_start":
                self._done = True
                return event
            try:
                rewritten = self._rewrite_message_start(event)
            except Exception:
                logger.debug("context_guard: message_start rewrite skipped", exc_info=True)
                self._done = True
                return event
            if self._target_total is None:
                # Below trigger (or already reads full): nothing to nudge.
                self._done = True
            return rewritten
        # Armed: pass content deltas through, rewrite the final usage delta.
        if event_type != b"message_delta":
            return event
        self._done = True
        try:
            return self._rewrite_message_delta(event)
        except Exception:
            logger.debug("context_guard: message_delta rewrite skipped", exc_info=True)
            return event

    def _rewrite_message_start(self, event: bytes) -> bytes:
        lines = event.split(b"\n")
        for i, line in enumerate(lines):
            if not line.startswith(b"data:"):
                continue
            payload = json.loads(line[len(b"data:") :].strip())
            if payload.get("type") != "message_start":
                return event
            usage = payload.get("message", {}).get("usage")
            if not isinstance(usage, dict):
                return event
            cache_read = int(usage.get("cache_read_input_tokens") or 0)
            cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
            total = int(usage.get("input_tokens") or 0) + cache_read + cache_creation
            new_input = nudged_input_tokens(
                usage,
                believed_limit=self._believed_limit,
                effective_limit=self._effective_limit,
            )
            if new_input is None:
                return event
            target_total = new_input + cache_read + cache_creation
            self._target_total = target_total
            self._start_cache_read = cache_read
            self._start_cache_creation = cache_creation
            usage["input_tokens"] = new_input
            logger.warning(
                "[%s] event=context_guard_nudge forwarded_total=%d effective_limit=%d "
                "reported_total=%d believed_limit=%d (nudging client to compact "
                "before the prompt-too-long wall)",
                self._request_id,
                total,
                self._effective_limit,
                target_total,
                self._believed_limit,
            )
            lines[i] = b"data: " + json.dumps(payload, separators=(",", ":")).encode()
            return b"\n".join(lines)
        return event

    def _rewrite_message_delta(self, event: bytes) -> bytes:
        assert self._target_total is not None
        lines = event.split(b"\n")
        for i, line in enumerate(lines):
            if not line.startswith(b"data:"):
                continue
            payload = json.loads(line[len(b"data:") :].strip())
            if payload.get("type") != "message_delta":
                return event
            usage = payload.get("usage")
            # A delta without cumulative input usage can't override the
            # message_start values we already rewrote — leave it alone.
            if not isinstance(usage, dict) or "input_tokens" not in usage:
                return event
            cache_read = int(usage.get("cache_read_input_tokens", self._start_cache_read) or 0)
            cache_creation = int(
                usage.get("cache_creation_input_tokens", self._start_cache_creation) or 0
            )
            new_input = self._target_total - cache_read - cache_creation
            if int(usage.get("input_tokens") or 0) >= new_input:
                # Never deflate.
                return event
            usage["input_tokens"] = new_input
            lines[i] = b"data: " + json.dumps(payload, separators=(",", ":")).encode()
            return b"\n".join(lines)
        return event
