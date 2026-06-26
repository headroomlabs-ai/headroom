"""Top-level helper functions and constants for the Headroom proxy.

Contains lazy loaders, file logging setup, request body decompression,
and safety-limit constants.

Extracted from server.py for maintainability.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from headroom import paths as _paths
from headroom._subprocess import run

if TYPE_CHECKING:
    import httpx
    from fastapi import Request

logger = logging.getLogger("headroom.proxy")

_CODEX_WIRE_DEBUG_ENV = "HEADROOM_CODEX_WIRE_DEBUG"
_CODEX_WIRE_DEBUG_DIR_ENV = "HEADROOM_CODEX_WIRE_DEBUG_DIR"
_CODEX_WIRE_REDACTED = "[REDACTED]"
_CODEX_WIRE_SECRET_KEYS = (
    "authorization",
    "cookie",
    "set-cookie",
    "api-key",
    "x-api-key",
    "openai-api-key",
    "anthropic-api-key",
    "access_token",
    "refresh_token",
    "id_token",
    "bearer",
    "password",
    "secret",
    "token",
    "credential",
)


def codex_wire_debug_enabled() -> bool:
    """Return whether opt-in Codex wire capture is enabled."""

    return os.environ.get(_CODEX_WIRE_DEBUG_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _codex_wire_debug_dir() -> Path:
    explicit = os.environ.get(_CODEX_WIRE_DEBUG_DIR_ENV, "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return _paths.codex_wire_debug_dir()


def _should_redact_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    if normalized in {marker.replace("-", "_") for marker in _CODEX_WIRE_SECRET_KEYS}:
        return True
    return (
        normalized.endswith("_api_key")
        or normalized.endswith("_secret")
        or normalized.endswith("_password")
        or normalized.endswith("_access_token")
        or normalized.endswith("_refresh_token")
    )


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: (_CODEX_WIRE_REDACTED if _should_redact_key(str(k)) else _redact_value(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


def redact_for_wire_debug(value: Any) -> Any:
    """Redact obvious secrets while preserving request/response shape."""

    return _redact_value(value)


def _safe_event_name(event: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in event)[:80]


def _wire_debug_preview(value: Any, *, max_chars: int | None = None) -> str:
    """Return the redacted wire payload for proxy.log.

    This is intentionally not truncated. During Codex WS debugging we need the
    proxy log itself to show the complete frame so we can decide later where a
    deliberate trim boundary belongs.
    """

    try:
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="replace")
        elif isinstance(value, str):
            text = value
        elif value is None:
            return ""
        else:
            text = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    except Exception:
        text = repr(value)

    text = " ".join(text.split())
    if max_chars is not None and len(text) > max_chars:
        return text[: max_chars - 1] + "…"
    return text


def capture_codex_wire_debug(
    event: str,
    *,
    request_id: str | None = None,
    session_id: str | None = None,
    transport: str,
    direction: str,
    method: str | None = None,
    url: str | None = None,
    headers: dict[str, Any] | None = None,
    body: Any = None,
    raw_text: str | None = None,
    status_code: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path | None:
    """Write an opt-in redacted Codex wire snapshot to disk.

    This is intentionally file-based rather than log-based: real Codex
    requests can be large, and operators need the exact envelope shape without
    mixing it into normal proxy logs. Header/body secret-looking keys are
    redacted, but request content is otherwise preserved because this mode is
    explicitly for local debugging.
    """

    if not codex_wire_debug_enabled():
        return None

    try:
        out_dir = _codex_wire_debug_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        ts_ns = time.time_ns()
        req = request_id or "no_request"
        safe_req = _safe_event_name(req)
        safe_event = _safe_event_name(event)
        path = out_dir / f"{ts_ns}_{safe_req}_{safe_event}.json"
        payload = {
            "event": event,
            "timestamp_ns": ts_ns,
            "request_id": request_id,
            "session_id": session_id,
            "transport": transport,
            "direction": direction,
            "method": method,
            "url": url,
            "status_code": status_code,
            "headers": redact_for_wire_debug(headers or {}),
            "body": redact_for_wire_debug(body),
            "raw_text": raw_text,
            "metadata": redact_for_wire_debug(metadata or {}),
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        logger.info(
            "event=codex_wire_debug_capture path=%s request_id=%s wire_event=%s",
            path,
            request_id or "",
            event,
        )
        preview_source = redact_for_wire_debug(body) if body is not None else raw_text
        preview = _wire_debug_preview(preview_source)
        meta_keys = ",".join(sorted((metadata or {}).keys()))
        logger.info(
            "event=codex_wire_debug_frame request_id=%s session_id=%s wire_event=%s "
            "transport=%s direction=%s status_code=%s meta_keys=%s preview=%s",
            request_id or "",
            session_id or "",
            event,
            transport,
            direction,
            status_code if status_code is not None else "",
            meta_keys,
            preview,
        )
        return path
    except Exception as exc:  # pragma: no cover - debug path must never break traffic
        logger.warning("event=codex_wire_debug_capture_failed error=%s", exc)
        return None


# Memory injection mode (P0-1 fix in PR-A2).
#
# Values:
#   - "live_zone_tail" (default): Memory context appends to the first text block
#     of the latest non-frozen user message. Cache hot zone (system + frozen
#     prefix) is never mutated.
#   - "disabled": Memory context lookup is skipped entirely; the request
#     forwards untouched.
#
# Configurable via HEADROOM_MEMORY_INJECTION_MODE env var. There is no
# "system_prompt" option — that path is permanently retired by I2 (cache hot
# zone never modified). See REALIGNMENT/02-architecture.md §2.2.
_MEMORY_INJECTION_MODE_ENV = "HEADROOM_MEMORY_INJECTION_MODE"
_MEMORY_INJECTION_MODE_DEFAULT: Literal["live_zone_tail", "disabled"] = "live_zone_tail"
MemoryInjectionMode = Literal["live_zone_tail", "disabled"]


def get_memory_injection_mode() -> MemoryInjectionMode:
    """Return the active memory-injection routing mode.

    Read at request time so the env var can be flipped without restart for
    smoke tests. Unknown values are rejected loudly (no silent fallback).
    """
    raw = os.environ.get(_MEMORY_INJECTION_MODE_ENV, "").strip().lower()
    if not raw:
        return _MEMORY_INJECTION_MODE_DEFAULT
    if raw in ("live_zone_tail", "disabled"):
        return cast(MemoryInjectionMode, raw)
    raise ValueError(
        f"Invalid {_MEMORY_INJECTION_MODE_ENV}={raw!r}; expected 'live_zone_tail' or 'disabled'"
    )


def hash_query_for_log(query: str) -> str:
    """Stable short hash of a memory-context query, safe to log.

    Uses BLAKE2b truncated to 16 hex chars. Never logs the raw query content.
    """
    h = hashlib.blake2b(query.encode("utf-8", errors="replace"), digest_size=8)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Byte-faithful Python forwarder support (PR-A3 — fixes P0-2).
# ---------------------------------------------------------------------------
#
# Every Python forwarder (server.py:_retry_request, streaming.py,
# openai.py:_ws_http_fallback, batch.py) historically used
# ``httpx.AsyncClient.post(url, json=body)``. httpx's default JSON encoder
# uses ``separators=(", ", ": ")`` and ``ensure_ascii=True`` so the bytes
# leaving the proxy never byte-equal the bytes that arrived from a
# well-behaved client (Claude Code, Codex CLI emit compact + UTF-8). Every
# such request collapses Anthropic prefix-cache hit-rate.
#
# PR-A3 switches every forwarder to byte-faithful forwarding:
#   * unmutated body → forward original ``await request.body()`` verbatim;
#   * mutated body  → re-serialize once via ``serialize_body_canonical``.
#
# A ``BodyMutationTracker`` accompanies each request so the forwarder can
# pick the right path. Memory-injection / compression / image-rewrite sites
# call ``tracker.mark_mutated(reason)``.

_PYTHON_FORWARDER_MODE_ENV = "HEADROOM_PROXY_PYTHON_FORWARDER_MODE"
PythonForwarderMode = Literal["byte_faithful", "legacy_json_kwarg"]
_PYTHON_FORWARDER_MODE_DEFAULT: PythonForwarderMode = "byte_faithful"


def get_python_forwarder_mode() -> PythonForwarderMode:
    """Return the active Python-forwarder mode.

    Read at request time. Unknown values raise loudly per the no-silent-
    fallback build constraint. The ``legacy_json_kwarg`` value is an
    explicit operator opt-in for emergency rollback — NOT a fallback.
    """
    raw = os.environ.get(_PYTHON_FORWARDER_MODE_ENV, "").strip().lower()
    if not raw:
        return _PYTHON_FORWARDER_MODE_DEFAULT
    if raw in ("byte_faithful", "legacy_json_kwarg"):
        return cast(PythonForwarderMode, raw)
    raise ValueError(
        f"Invalid {_PYTHON_FORWARDER_MODE_ENV}={raw!r}; "
        "expected 'byte_faithful' or 'legacy_json_kwarg'"
    )


def extract_tags(headers: Any) -> dict[str, str]:
    """Extract ``x-headroom-*`` tags from inbound headers.

    Pure function (no I/O, no state). Used by every handler at request
    entry to capture operator slicing tags into the per-request
    ``RequestOutcome.tags``. Free function rather than a mixin method so
    handler mixins instantiated in isolation (tests using
    ``object.__new__(OpenAIHandlerMixin)``) don't need a shim
    implementation.

    Header name match is case-insensitive; the returned key has the
    ``x-headroom-`` prefix stripped.
    """
    return {
        k.lower().replace("x-headroom-", ""): v
        for k, v in headers.items()
        if k.lower().startswith("x-headroom-")
    }


def _headroom_bypass_enabled(headers: Any) -> bool:
    """Return True when inbound headers request full Headroom passthrough.

    This is transport-neutral policy: HTTP and WebSocket handlers both call
    it on original inbound headers before request-body mutation.
    """

    try:
        bypass = str(headers.get("x-headroom-bypass", "")).strip().lower() == "true"
        passthrough = str(headers.get("x-headroom-mode", "")).strip().lower() == "passthrough"
    except AttributeError:
        return False
    return bypass or passthrough


def serialize_body_canonical(body: dict[str, Any]) -> bytes:
    """Re-serialize a request body deterministically with cache-stable formatting.

    Uses compact separators and preserves UTF-8 (no ``\\uXXXX`` escapes), so
    byte output matches what well-behaved API clients (Claude Code, Codex
    CLI) emit. Python 3.7+ dict insertion order is preserved by
    ``json.dumps`` so message ordering is stable.

    This is the canonical re-serialization for any forwarder path that did
    mutate the body (memory injection, compression, etc.).
    """
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class BodyMutationTracker:
    """Records whether a request body was mutated and why.

    The forwarder reads ``mutated`` to decide between byte-faithful
    passthrough and canonical re-serialization. Reasons are logged with
    each outbound request to make cache-affecting decisions auditable.

    Thread-safety: a single tracker instance is owned by exactly one
    request task. No locking needed.
    """

    __slots__ = ("_mutated", "_reasons")

    def __init__(self) -> None:
        self._mutated: bool = False
        self._reasons: list[str] = []

    def mark_mutated(self, reason: str) -> None:
        """Mark the body as mutated and record the reason.

        ``reason`` should be a stable identifier (snake_case) suitable for
        log aggregation, e.g. ``memory_injection`` or
        ``compression_smart_crusher``.
        """
        if not reason:
            raise ValueError("BodyMutationTracker.mark_mutated: reason must be non-empty")
        self._mutated = True
        if reason not in self._reasons:
            self._reasons.append(reason)

    @property
    def mutated(self) -> bool:
        return self._mutated

    @property
    def reasons(self) -> list[str]:
        return list(self._reasons)


def prepare_outbound_body_bytes(
    *,
    body: dict[str, Any],
    original_body_bytes: bytes | None,
    body_mutated: bool,
    forwarder_mode: PythonForwarderMode | None = None,
) -> tuple[bytes, str]:
    """Pick the outbound body bytes for a forwarder call.

    Returns ``(outbound_bytes, source)`` where ``source`` is one of
    ``passthrough`` (original bytes verbatim), ``canonical`` (re-serialized
    deterministically because body was mutated), or ``legacy`` (rollback
    mode — old ``json=body`` behavior).

    * ``forwarder_mode == "byte_faithful"`` (default): unmutated → passthrough,
      mutated → canonical.
    * ``forwarder_mode == "legacy_json_kwarg"``: always re-encode via the old
      httpx-style separators (operator opt-in, for rollback only).
    """
    mode = forwarder_mode if forwarder_mode is not None else get_python_forwarder_mode()
    if mode == "legacy_json_kwarg":
        # Old httpx default: separators=(", ", ": "), ensure_ascii=True.
        legacy_bytes = json.dumps(body, separators=(", ", ": "), ensure_ascii=True).encode("utf-8")
        return legacy_bytes, "legacy"

    # byte_faithful path
    if body_mutated or original_body_bytes is None:
        return serialize_body_canonical(body), "canonical"
    return original_body_bytes, "passthrough"


def log_outbound_request(
    *,
    forwarder: str,
    method: str,
    path: str,
    body_bytes_count: int,
    body_mutated: bool,
    mutation_reasons: list[str],
    request_id: str | None,
    source: str,
) -> None:
    """Structured log line for every outbound forwarder call.

    Per realignment build constraints: every cache-affecting decision is
    logged. Never includes ``Authorization``/``x-api-key`` content or full
    body bytes.
    """
    logger.info(
        "event=outbound_request forwarder=%s method=%s path=%s body_bytes=%d "
        "body_mutated=%s mutation_reasons=%s source=%s request_id=%s",
        forwarder,
        method,
        path,
        body_bytes_count,
        "true" if body_mutated else "false",
        ",".join(mutation_reasons) if mutation_reasons else "",
        source,
        request_id or "",
    )


def log_memory_injection(
    *,
    request_id: str,
    session_id: str | None,
    decision: str,
    bytes_injected: int,
    query: str | None = None,
) -> None:
    """Emit a structured log line for every memory-context routing decision.

    Per realignment build constraints: log every cache-affecting decision.
    Never log raw query content or Authorization header — only a stable
    hash of the query.
    """
    query_hash = hash_query_for_log(query) if query else ""
    logger.info(
        "event=memory_injection request_id=%s session_id=%s decision=%s "
        "bytes_injected=%d query_hash=%s",
        request_id,
        session_id or "",
        decision,
        bytes_injected,
        query_hash,
    )


def append_text_to_latest_user_chat_message(
    messages: list[dict[str, Any]],
    context_text: str,
) -> tuple[list[dict[str, Any]], int]:
    """Append context text to the first text block of the latest user chat message.

    OpenAI Chat Completions ``body["messages"]`` shape: each message is
    ``{"role": ..., "content": str | list[{"type": "text"|"input_text", "text": ...}]}``.

    This is the OpenAI Chat Completions analog of
    ``_append_context_to_latest_non_frozen_user_turn`` (Anthropic) and
    ``append_text_to_latest_user_input_item`` (OpenAI Responses). Used by
    PR-A3 to retire the legacy system-prepend memory-injection path
    (P0-equivalent for /v1/chat/completions).

    Returns ``(new_messages, bytes_appended)``. ``bytes_appended == 0``
    when no eligible user message was found (no mutation occurred).
    """
    if not messages or not context_text:
        return messages, 0

    new_messages = list(messages)
    for idx in range(len(new_messages) - 1, -1, -1):
        msg = new_messages[idx]
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue

        content = msg.get("content")
        if isinstance(content, str):
            updated_msg = {**msg, "content": content + "\n\n" + context_text}
            new_messages[idx] = updated_msg
            return new_messages, len(context_text)

        if isinstance(content, list) and content:
            new_content: list[dict[str, Any]] = []
            appended = False
            for part in content:
                if (
                    not appended
                    and isinstance(part, dict)
                    and part.get("type") in ("text", "input_text")
                ):
                    existing_text = part.get("text", "")
                    new_part = {**part, "text": existing_text + "\n\n" + context_text}
                    new_content.append(new_part)
                    appended = True
                else:
                    new_content.append(part)
            if appended:
                updated_msg = {**msg, "content": new_content}
                new_messages[idx] = updated_msg
                return new_messages, len(context_text)

        # User message but no eligible text block — leave untouched and stop.
        return messages, 0

    return messages, 0


def append_text_to_latest_user_input_item(
    body_input: list[dict[str, Any]],
    context_text: str,
) -> tuple[list[dict[str, Any]], int]:
    """Append context text to the first text block of the latest user input item.

    Mirrors ``_append_context_to_latest_non_frozen_user_turn`` but for the
    OpenAI Responses API ``body["input"]`` shape, which uses a flat item list
    where each user item's content is a list like
    ``[{"type": "input_text", "text": "..."}]``.

    Returns a tuple ``(new_input, bytes_appended)`` where ``bytes_appended``
    is 0 when the item list was unchanged (no eligible user item).
    """
    if not body_input or not context_text:
        return body_input, 0

    new_input = list(body_input)

    for idx in range(len(new_input) - 1, -1, -1):
        item = new_input[idx]
        if not isinstance(item, dict):
            continue
        if item.get("role") != "user":
            continue

        content = item.get("content")
        if isinstance(content, str):
            updated_item = {**item, "content": content + "\n\n" + context_text}
            new_input[idx] = updated_item
            return new_input, len(context_text)

        if isinstance(content, list) and content:
            new_content: list[dict[str, Any]] = []
            appended = False
            for part in content:
                if (
                    not appended
                    and isinstance(part, dict)
                    and part.get("type") in ("input_text", "text")
                ):
                    existing_text = part.get("text", "")
                    new_part = {**part, "text": existing_text + "\n\n" + context_text}
                    new_content.append(new_part)
                    appended = True
                else:
                    new_content.append(part)
            if appended:
                updated_item = {**item, "content": new_content}
                new_input[idx] = updated_item
                return new_input, len(context_text)

        # User item but no eligible text block — leave untouched and stop.
        return body_input, 0

    return body_input, 0


_CONTEXT_TOOL_ENV = "HEADROOM_CONTEXT_TOOL"
_CONTEXT_TOOL_RTK = "rtk"
_CONTEXT_TOOL_LEAN_CTX = "lean-ctx"
_RTK_GAIN_SCOPE_ENV = "HEADROOM_RTK_GAIN_SCOPE"
_RTK_GAIN_SCOPE_GLOBAL = "global"
_RTK_GAIN_SCOPE_PROJECT = "project"
_RTK_GAIN_SCOPES = {_RTK_GAIN_SCOPE_GLOBAL, _RTK_GAIN_SCOPE_PROJECT}

RTK_STATS_CACHE_TTL_SECONDS = float(os.environ.get("HEADROOM_CONTEXT_TOOL_STATS_TTL_SECONDS", "60"))
CONTEXT_TOOL_STATS_CACHE_TTL_SECONDS = RTK_STATS_CACHE_TTL_SECONDS
_context_tool_stats_cache_lock = threading.Lock()
_context_tool_stats_cache: dict[str, Any] = {
    "expires_at": 0.0,
    "has_value": False,
    "tool": None,
    "value": None,
}
_context_tool_session_baseline: dict[str, Any] = {
    "initialized": False,
    "tool": None,
    "total_commands": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "tokens_saved": 0,
    "total_time_ms": 0,
    "captured_at": 0.0,
}
_rtk_stats_cache_lock = _context_tool_stats_cache_lock
_rtk_stats_cache = _context_tool_stats_cache
_rtk_session_baseline = _context_tool_session_baseline

# Maximum request body size (100MB - increased to support image-heavy requests)
MAX_REQUEST_BODY_SIZE = 100 * 1024 * 1024

# Maximum SSE buffer size (10MB - prevents memory exhaustion from malformed streams)
MAX_SSE_BUFFER_SIZE = 10 * 1024 * 1024

# Per-event SSE size cap (PR-A8 / P1-8). Configurable via
# HEADROOM_SSE_BUFFER_MAX_BYTES. Guards against pathological huge events
# (a single event > 1 MB by default is treated as an upstream protocol bug
# and surfaces loudly rather than silently growing the buffer).
_SSE_EVENT_MAX_BYTES_ENV = "HEADROOM_SSE_BUFFER_MAX_BYTES"
_SSE_EVENT_MAX_BYTES_DEFAULT = 1 * 1024 * 1024  # 1 MB


def get_sse_event_max_bytes() -> int:
    """Return the per-event SSE size cap.

    Read at request time so operators can flip the env var without a
    restart. Negative values are rejected loudly (no silent fallback).
    """
    raw = os.environ.get(_SSE_EVENT_MAX_BYTES_ENV)
    if raw is None or raw == "":
        return _SSE_EVENT_MAX_BYTES_DEFAULT
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_SSE_EVENT_MAX_BYTES_ENV} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{_SSE_EVENT_MAX_BYTES_ENV} must be positive, got {value}")
    return value


# Body-too-large status code (PR-A8 / P5-59). Default 413 (RFC 7231 §6.5.11).
# Configurable via HEADROOM_PROXY_BODY_TOO_LARGE_STATUS for operators who need
# to override (no expected production use; documentation knob).
_BODY_TOO_LARGE_STATUS_ENV = "HEADROOM_PROXY_BODY_TOO_LARGE_STATUS"
_BODY_TOO_LARGE_STATUS_DEFAULT = 413


def get_body_too_large_status() -> int:
    """Return the HTTP status code for body-too-large rejections."""
    raw = os.environ.get(_BODY_TOO_LARGE_STATUS_ENV)
    if raw is None or raw == "":
        return _BODY_TOO_LARGE_STATUS_DEFAULT
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_BODY_TOO_LARGE_STATUS_ENV} must be an integer, got {raw!r}") from exc
    if not 400 <= value < 600:
        raise ValueError(f"{_BODY_TOO_LARGE_STATUS_ENV} must be a 4xx/5xx status, got {value}")
    return value


# SSE byte-buffer helper supports LF and CRLF event separators. Per the SSE
# spec the default event name is "message"; we return ``None`` so callers can
# decide whether to apply that default.
_SSE_EVENT_TERMINATORS = (b"\n\n", b"\r\n\r\n")


def _find_sse_event_terminator(buf: bytearray) -> tuple[int, int] | None:
    """Return the earliest complete SSE event terminator in ``buf``."""
    matches = [
        (idx, len(terminator))
        for terminator in _SSE_EVENT_TERMINATORS
        if (idx := buf.find(terminator)) != -1
    ]
    if not matches:
        return None
    return min(matches, key=lambda match: match[0])


_SSE_EVENT_LINE_PREFIX = b"event:"
_SSE_DATA_LINE_PREFIX = b"data:"


def safe_decode_for_logging(raw: bytes, *, max_bytes: int | None = None) -> str:
    """Decode bytes to a string for **log/diagnostic display only**.

    PR-A8 / P1-8: the SSE wire path forbids ``errors="ignore"`` /
    ``errors="replace"`` because corrupting bytes silently busts cache
    safety. Diagnostic logs (e.g. error response bodies) are fine to
    show with a replacement character because the bytes are already
    discarded; this helper centralizes that single legitimate use of
    the lossy decoder so a project-wide grep stays clean.

    Use ``parse_sse_events_from_byte_buffer`` for SSE parsing instead.
    """
    blob = raw[:max_bytes] if max_bytes is not None else raw
    # Decode incrementally and represent any invalid bytes as the
    # Unicode replacement character (�). Implemented via the
    # `codecs` incremental decoder so we never reach for the
    # forbidden `errors="ignore"`/`errors="replace"` keyword in the
    # SSE-bearing modules.
    import codecs as _codecs

    decoder = _codecs.getincrementaldecoder("utf-8")(errors="replace")
    return decoder.decode(bytes(blob), final=True)


def parse_sse_events_from_byte_buffer(
    buf: bytearray,
) -> list[tuple[str | None, str]]:
    """Drain complete ``event:`` + ``data:`` events from a bytes buffer.

    Returns list of ``(event_name, data_str)`` tuples for complete events.
    Mutates ``buf`` in-place to leave only partial-event tail bytes.

    Operates on bytes; only decodes complete events as UTF-8 (raises if a
    *complete* event has invalid UTF-8 — that's an upstream protocol bug
    we want loud, not silent).

    Per PR-A8 / P1-8: this is the canonical SSE event splitter. NEVER use
    ``decode("utf-8", errors="ignore")`` on a partial buffer; UTF-8
    multi-byte characters split across TCP reads will corrupt content.
    """
    events: list[tuple[str | None, str]] = []
    while True:
        terminator_match = _find_sse_event_terminator(buf)
        if terminator_match is None:
            break
        idx, terminator_len = terminator_match
        event_bytes = bytes(buf[:idx])
        # Drain the event + the trailing terminator from the buffer.
        del buf[: idx + terminator_len]
        # Decoding the COMPLETE event must succeed. If it doesn't, the
        # upstream emitted invalid UTF-8 mid-stream — surface loudly.
        event_text = event_bytes.decode("utf-8")
        event_name: str | None = None
        data_lines: list[str] = []
        for line in event_text.splitlines():
            if not line:
                continue
            # SSE comment line — ignored per spec.
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line[len("event:") :].lstrip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].lstrip())
        # Per SSE spec, multiple `data:` lines join with newline. We
        # preserve that here even though OpenAI/Anthropic emit one
        # `data:` per event.
        if data_lines:
            events.append((event_name, "\n".join(data_lines)))
    return events


# Maximum message array length (prevents DoS from deeply nested payloads)
MAX_MESSAGE_ARRAY_LENGTH = 10000

# Compression pipeline timeout in seconds. Override via the
# HEADROOM_COMPRESSION_TIMEOUT_SECONDS env var for slow CPUs or long Claude Code
# conversations (GH #946). Falls back to 30 on an unparseable value.
try:
    COMPRESSION_TIMEOUT_SECONDS = float(
        os.environ.get("HEADROOM_COMPRESSION_TIMEOUT_SECONDS", "30")
    )
except ValueError:
    COMPRESSION_TIMEOUT_SECONDS = 30.0

# Maximum compression cache sessions (prevents unbounded memory growth)
MAX_COMPRESSION_CACHE_SESSIONS = 500


# ---------------------------------------------------------------------------
# Compression-failure escape hatch
# ---------------------------------------------------------------------------
# When the proxy's compression stage fails (timeout, exception) on a frame
# Headroom thought was large enough to compress, the legacy behaviour was to
# fall through and forward the *original* uncompressed frame to the upstream.
# That fail-open turned a recoverable timeout into a context-window overflow
# downstream: Codex's auto-compaction reads ``total_usage_tokens`` from
# upstream (which Headroom's earlier successful compressions shrunk), then
# the un-compressed retry overflows the model context and the client
# locks up.
#
# Default behaviour is now fail-CLOSED: refuse to forward, close the client
# WS with code 1009 (or return HTTP 413) so the client knows to compact and
# retry. Operators who want the old behaviour can set
# ``HEADROOM_WS_FAIL_OPEN_ON_COMPRESSION_FAILURE=1``. The oversize threshold
# below which transient errors still fall through to passthrough is
# configurable via ``HEADROOM_WS_COMPRESSION_FAIL_THRESHOLD_BYTES``
# (default 256 KiB ≈ 64K tokens).
WS_COMPRESSION_FAIL_OPEN_ENV = "HEADROOM_WS_FAIL_OPEN_ON_COMPRESSION_FAILURE"
WS_COMPRESSION_OVERSIZE_BYTES_ENV = "HEADROOM_WS_COMPRESSION_FAIL_THRESHOLD_BYTES"
WS_COMPRESSION_OVERSIZE_BYTES_DEFAULT = 256 * 1024


@dataclass(frozen=True)
class CompressionFailureAction:
    """Decision returned by :func:`decide_compression_failure_action`."""

    refuse: bool
    """If True, the caller MUST NOT forward the original frame. Close the
    client connection with a clear error code instead."""

    reason: str
    """Short machine-readable label for telemetry. One of:
    ``timeout``, ``oversize:bytes=<n>>threshold=<m>``,
    ``small_frame_transient``, ``client_override:codex``, or
    ``env_override:fail_open``."""

    frame_bytes: int
    """Original frame size in bytes (for logging / metrics)."""


def decide_compression_failure_action(
    exception: BaseException,
    frame_bytes: int,
    *,
    client: str | None = None,
) -> CompressionFailureAction:
    """Decide whether to refuse-and-close vs forward-original after the
    proxy's compression pipeline fails on a Realtime WebSocket frame
    (or analogous HTTP body).

    Decision matrix:

    * env :data:`WS_COMPRESSION_FAIL_OPEN_ENV` truthy → forward (legacy
      behaviour, opt-in for debugging or strict compatibility).
    * Codex client compression timeout → forward. Codex currently treats
      the proxy's 1009/413 refusal path as a hard connection failure, so
      fail-open is safer for Codex sessions even when the proxy is run
      standalone rather than through ``headroom wrap codex``.
    * exception is :class:`asyncio.TimeoutError` → refuse (the compression
      stage hit its own timeout, which only fires on frames Headroom
      thought were big enough to need compression in the first place).
    * ``frame_bytes`` > :data:`WS_COMPRESSION_OVERSIZE_BYTES_ENV`
      (default 256 KiB) → refuse (large + any compression failure is a
      strong signal the upstream will reject the original).
    * otherwise → forward (a transient pipeline error on a small frame
      shouldn't break the request).
    """
    fail_open = os.environ.get(WS_COMPRESSION_FAIL_OPEN_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if fail_open:
        return CompressionFailureAction(
            refuse=False,
            reason="env_override:fail_open",
            frame_bytes=frame_bytes,
        )

    if (client or "").strip().lower() == "codex" and isinstance(exception, asyncio.TimeoutError):
        return CompressionFailureAction(
            refuse=False,
            reason="client_override:codex",
            frame_bytes=frame_bytes,
        )

    threshold = WS_COMPRESSION_OVERSIZE_BYTES_DEFAULT
    raw_threshold = os.environ.get(WS_COMPRESSION_OVERSIZE_BYTES_ENV, "").strip()
    if raw_threshold:
        try:
            parsed = int(raw_threshold)
            if parsed > 0:
                threshold = parsed
        except ValueError:
            # Operator typo'd the env value — keep the default rather than
            # raise on every WS frame. Loud warning instead.
            logger.warning(
                "Ignoring non-integer %s=%r; using default %d",
                WS_COMPRESSION_OVERSIZE_BYTES_ENV,
                raw_threshold,
                WS_COMPRESSION_OVERSIZE_BYTES_DEFAULT,
            )

    if isinstance(exception, asyncio.TimeoutError):
        return CompressionFailureAction(refuse=True, reason="timeout", frame_bytes=frame_bytes)
    if frame_bytes > threshold:
        return CompressionFailureAction(
            refuse=True,
            reason=f"oversize:bytes={frame_bytes}>threshold={threshold}",
            frame_bytes=frame_bytes,
        )
    return CompressionFailureAction(
        refuse=False, reason="small_frame_transient", frame_bytes=frame_bytes
    )


def jitter_delay_ms(base_ms: int, max_ms: int, attempt: int) -> float:
    """Exponential backoff with 50-150% jitter.

    Returns ``min(base_ms * 2**attempt, max_ms) * (0.5 + random())`` — the
    canonical formula used across proxy retry loops. Extracted so every
    retry site shares one implementation.
    """
    capped: float = min(base_ms * (2**attempt), max_ms)
    return capped * (0.5 + random.random())


def retry_after_ms(response: httpx.Response, max_ms: int) -> float | None:
    """Parse an HTTP ``Retry-After`` header into a millisecond delay, capped at ``max_ms``.

    Returns the delay in ms for a numeric ``seconds`` value or an HTTP-date, or
    ``None`` when the header is absent or unparseable so the caller falls back to
    exponential backoff. Anthropic sends integer seconds; the HTTP-date branch
    covers other upstreams. Fails open on any parse error.
    """
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            from datetime import datetime
            from email.utils import parsedate_to_datetime

            retry_at = parsedate_to_datetime(value)
            seconds = (retry_at - datetime.now(retry_at.tzinfo)).total_seconds()
        except (TypeError, ValueError):
            return None
    return min(max(seconds, 0.0) * 1000.0, float(max_ms))


# Image compression availability (do not retain a global compressor instance)
_image_compressor_available: bool | None = None


def _get_image_compressor():
    """Create a short-lived image compressor on demand."""
    global _image_compressor_available
    if _image_compressor_available is False:
        return None

    try:
        from headroom.image import ImageCompressor

        # Callers own closing the compressor; this helper only memoizes whether
        # the optional image stack is importable.
        compressor = ImageCompressor()
        if _image_compressor_available is None:
            logger.info("Image compression enabled (model: chopratejas/technique-router)")
        _image_compressor_available = True
        return compressor
    except ImportError as e:
        if _image_compressor_available is not False:
            logger.warning(f"Image compression not available: {e}")
        _image_compressor_available = False
        return None


# Always-on file logging to the workspace logs directory for `headroom perf` analysis.
# Resolved lazily so HEADROOM_WORKSPACE_DIR env-var changes are honored.


def _headroom_log_dir() -> Path:
    return _paths.log_dir()


def _setup_file_logging() -> None:
    """Add a RotatingFileHandler to the headroom root logger.

    Writes to ~/.headroom/logs/proxy.log with automatic rotation:
    - Rotates at 10 MB
    - Keeps 5 backups (~50 MB max)
    """
    from logging.handlers import RotatingFileHandler

    try:
        log_dir = _headroom_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "proxy.log"
        handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        # Attach to the headroom root logger so all sub-loggers are captured.
        # Disable propagation to root to avoid duplicate writes when
        # wrap.py redirects stderr to the same log file.
        headroom_logger = logging.getLogger("headroom")
        headroom_logger.setLevel(logging.INFO)
        if not any(isinstance(h, RotatingFileHandler) for h in headroom_logger.handlers):
            headroom_logger.addHandler(handler)
        headroom_logger.propagate = False
    except OSError:
        # Non-fatal: can't write logs (read-only fs, permissions, etc.)
        pass


def _selected_context_tool() -> str:
    raw = os.environ.get(_CONTEXT_TOOL_ENV, _CONTEXT_TOOL_RTK).strip().lower()
    normalized = raw.replace("_", "-")
    if normalized in ("leanctx", _CONTEXT_TOOL_LEAN_CTX):
        return _CONTEXT_TOOL_LEAN_CTX
    return _CONTEXT_TOOL_RTK


def _context_tool_label(tool: str) -> str:
    if tool == _CONTEXT_TOOL_LEAN_CTX:
        return "lean-ctx"
    return "RTK"


def _context_tool_default_scope(tool: str) -> str:
    if tool == _CONTEXT_TOOL_LEAN_CTX:
        return "local"
    return _RTK_GAIN_SCOPE_GLOBAL


def _rtk_gain_scope() -> str:
    raw = os.environ.get(_RTK_GAIN_SCOPE_ENV, "").strip().lower()
    if not raw:
        return _RTK_GAIN_SCOPE_GLOBAL
    if raw in _RTK_GAIN_SCOPES:
        return raw

    logger.warning(
        "event=rtk_gain_scope_invalid env=%s value=%r default=%s",
        _RTK_GAIN_SCOPE_ENV,
        raw,
        _RTK_GAIN_SCOPE_GLOBAL,
    )
    return _RTK_GAIN_SCOPE_GLOBAL


def _rtk_gain_command(rtk_path: Any, scope: str) -> list[str]:
    command = [str(rtk_path), "gain"]
    if scope == _RTK_GAIN_SCOPE_PROJECT:
        command.append("--project")
    command.extend(["--format", "json"])
    return command


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return default


def _first_value(mapping: dict[str, Any], keys: tuple[str, ...], default: Any = 0) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _context_tool_summary_payload(
    *,
    tool: str,
    installed: bool,
    scope: str | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize RTK/lean-ctx lifetime gain output into one schema.

    Both tools expose cumulative counters, but field names vary slightly.
    Headroom computes session values by subtracting a startup baseline, so
    keeping raw input/output counters is necessary for a truthful session
    savings percentage.
    """

    summary = summary or {}
    input_tokens = _coerce_int(
        _first_value(
            summary,
            (
                "total_input",
                "total_input_tokens",
                "input_tokens",
                "tokens_input",
                "totalBefore",
            ),
        )
    )
    output_tokens = _coerce_int(
        _first_value(
            summary,
            (
                "total_output",
                "total_output_tokens",
                "output_tokens",
                "tokens_output",
                "totalAfter",
            ),
        )
    )
    tokens_saved = _coerce_int(
        _first_value(
            summary,
            (
                "total_saved",
                "tokens_saved",
                "total_tokens_saved",
                "saved_tokens",
                "totalSaved",
            ),
        )
    )
    if tokens_saved <= 0 and input_tokens > 0 and output_tokens >= 0:
        tokens_saved = max(input_tokens - output_tokens, 0)
    if input_tokens <= 0 and tokens_saved > 0 and output_tokens >= 0:
        input_tokens = tokens_saved + output_tokens

    lifetime_savings_pct = _coerce_float(
        _first_value(
            summary,
            (
                "avg_savings_pct",
                "average_savings_pct",
                "savings_pct",
                "savings_percent",
                "avgSavingsPct",
            ),
            0.0,
        )
    )
    if lifetime_savings_pct <= 0 and input_tokens > 0:
        lifetime_savings_pct = (tokens_saved / input_tokens) * 100.0

    return {
        "tool": tool,
        "label": _context_tool_label(tool),
        "installed": installed,
        "scope": scope or _context_tool_default_scope(tool),
        "total_commands": _coerce_int(
            _first_value(
                summary,
                (
                    "total_commands",
                    "commands",
                    "command_count",
                    "totalCommandCount",
                ),
            )
        ),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_saved": tokens_saved,
        # Backward-compatible name. See `lifetime_avg_savings_pct` and
        # `session_savings_pct` below for explicit scopes.
        "avg_savings_pct": lifetime_savings_pct,
        "lifetime_avg_savings_pct": lifetime_savings_pct,
        "total_time_ms": _coerce_int(
            _first_value(summary, ("total_time_ms", "time_ms", "totalTimeMs"))
        ),
    }


def _context_tool_zero_payload(
    *,
    tool: str,
    installed: bool,
    scope: str | None = None,
) -> dict[str, Any]:
    return _context_tool_summary_payload(
        tool=tool,
        installed=installed,
        scope=scope,
        summary={},
    )


def _read_rtk_lifetime_stats() -> dict[str, Any] | None:
    """Read rtk's lifetime stats using the configured gain scope."""

    from headroom.rtk import get_rtk_path

    scope = _rtk_gain_scope()
    rtk_path = get_rtk_path()
    if not rtk_path:
        return _context_tool_zero_payload(
            tool=_CONTEXT_TOOL_RTK,
            installed=False,
            scope=scope,
        )

    try:
        result = run(
            _rtk_gain_command(rtk_path, scope),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            summary = data.get("summary", {})
            payload = _context_tool_summary_payload(
                tool=_CONTEXT_TOOL_RTK,
                installed=True,
                scope=scope,
                summary=summary if isinstance(summary, dict) else {},
            )
        else:
            # PR-G2 remediation (H2): structured log the synthetic-zero path
            # so downstream consumers (subscription tracker, dashboards) can
            # distinguish a healthy "RTK ran and saved nothing" from a broken
            # "RTK failed and we faked zero".
            stderr_excerpt = (result.stderr or "")[:200]
            logger.warning(
                "event=rtk_stats_subprocess_failed reason=non_zero_exit rc=%s stderr=%r",
                result.returncode,
                stderr_excerpt,
            )
            return _context_tool_zero_payload(
                tool=_CONTEXT_TOOL_RTK,
                installed=True,
                scope=scope,
            )
    except Exception as exc:
        # PR-G2 remediation (H2): log the exception path too. Reason is the
        # exception class name (without payload — RTK exceptions can carry
        # filesystem paths).
        logger.warning(
            "event=rtk_stats_subprocess_failed reason=%s error=%s",
            type(exc).__name__,
            exc,
        )
        return _context_tool_zero_payload(
            tool=_CONTEXT_TOOL_RTK,
            installed=True,
            scope=scope,
        )

    return payload


def _read_lean_ctx_lifetime_stats() -> dict[str, Any] | None:
    """Read lean-ctx's current project-level lifetime stats."""

    from headroom.lean_ctx import get_lean_ctx_path

    lean_ctx_path = get_lean_ctx_path()
    if not lean_ctx_path:
        return _context_tool_zero_payload(tool=_CONTEXT_TOOL_LEAN_CTX, installed=False)

    base_payload = _context_tool_zero_payload(tool=_CONTEXT_TOOL_LEAN_CTX, installed=True)

    try:
        result = run(
            [str(lean_ctx_path), "gain", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return dict(base_payload)

        data = json.loads(result.stdout)
        summary = data.get("summary", data) if isinstance(data, dict) else {}
        if not isinstance(summary, dict):
            return dict(base_payload)

        return _context_tool_summary_payload(
            tool=_CONTEXT_TOOL_LEAN_CTX,
            installed=True,
            summary=summary,
        )
    except Exception:
        return dict(base_payload)


def _read_context_tool_lifetime_stats(tool: str) -> dict[str, Any] | None:
    if tool == _CONTEXT_TOOL_LEAN_CTX:
        return _read_lean_ctx_lifetime_stats()
    return _read_rtk_lifetime_stats()


async def initialize_context_tool_session_baseline() -> None:
    """Pin the current context-tool counters as the proxy-session baseline."""

    tool = _selected_context_tool()
    payload = await asyncio.to_thread(_read_context_tool_lifetime_stats, tool)
    with _context_tool_stats_cache_lock:
        _context_tool_session_baseline.update(
            {
                "initialized": True,
                "tool": tool,
                "total_commands": int((payload or {}).get("total_commands", 0) or 0),
                "input_tokens": int((payload or {}).get("input_tokens", 0) or 0),
                "output_tokens": int((payload or {}).get("output_tokens", 0) or 0),
                "tokens_saved": int((payload or {}).get("tokens_saved", 0) or 0),
                "total_time_ms": int((payload or {}).get("total_time_ms", 0) or 0),
                "captured_at": time.time(),
            }
        )
        _context_tool_stats_cache.update(
            {
                "expires_at": 0.0,
                "has_value": False,
                "tool": None,
                "value": None,
            }
        )


async def initialize_rtk_session_baseline() -> None:
    """Backward-compatible alias for initialize_context_tool_session_baseline."""

    await initialize_context_tool_session_baseline()


def _get_context_tool_stats() -> dict[str, Any] | None:
    """Get context-tool savings for the current Headroom proxy session.

    RTK and lean-ctx persist project-level lifetime counters. Dashboard stats
    should be session-local, so we subtract the counter snapshot captured at
    proxy startup instead of resetting the tool's own history.
    """

    tool = _selected_context_tool()
    now = time.monotonic()
    with _context_tool_stats_cache_lock:
        cached_value = cast(dict[str, Any] | None, _context_tool_stats_cache["value"])
        if (
            _context_tool_stats_cache["has_value"]
            and now < float(_context_tool_stats_cache["expires_at"])
            and _context_tool_stats_cache.get("tool") == tool
        ):
            return cached_value

    payload = _read_context_tool_lifetime_stats(tool)
    with _context_tool_stats_cache_lock:
        if (
            not _context_tool_session_baseline["initialized"]
            or _context_tool_session_baseline.get("tool") != tool
        ):
            _context_tool_session_baseline.update(
                {
                    "initialized": True,
                    "tool": tool,
                    "total_commands": int((payload or {}).get("total_commands", 0) or 0),
                    "input_tokens": int((payload or {}).get("input_tokens", 0) or 0),
                    "output_tokens": int((payload or {}).get("output_tokens", 0) or 0),
                    "tokens_saved": int((payload or {}).get("tokens_saved", 0) or 0),
                    "total_time_ms": int((payload or {}).get("total_time_ms", 0) or 0),
                    "captured_at": time.time(),
                }
            )

        if payload is not None:
            lifetime_total_commands = int(payload.get("total_commands", 0) or 0)
            lifetime_input_tokens = int(payload.get("input_tokens", 0) or 0)
            lifetime_output_tokens = int(payload.get("output_tokens", 0) or 0)
            lifetime_tokens_saved = int(payload.get("tokens_saved", 0) or 0)
            lifetime_total_time_ms = int(payload.get("total_time_ms", 0) or 0)
            baseline_total_commands = int(_context_tool_session_baseline["total_commands"])
            baseline_input_tokens = int(_context_tool_session_baseline["input_tokens"])
            baseline_output_tokens = int(_context_tool_session_baseline["output_tokens"])
            baseline_tokens_saved = int(_context_tool_session_baseline["tokens_saved"])
            baseline_total_time_ms = int(_context_tool_session_baseline["total_time_ms"])
            counter_reset_detected = (
                lifetime_total_commands < baseline_total_commands
                or lifetime_input_tokens < baseline_input_tokens
                or lifetime_output_tokens < baseline_output_tokens
                or lifetime_tokens_saved < baseline_tokens_saved
                or lifetime_total_time_ms < baseline_total_time_ms
            )
            if counter_reset_detected:
                baseline_total_commands = lifetime_total_commands
                baseline_input_tokens = lifetime_input_tokens
                baseline_output_tokens = lifetime_output_tokens
                baseline_tokens_saved = lifetime_tokens_saved
                baseline_total_time_ms = lifetime_total_time_ms
                _context_tool_session_baseline.update(
                    {
                        "total_commands": baseline_total_commands,
                        "input_tokens": baseline_input_tokens,
                        "output_tokens": baseline_output_tokens,
                        "tokens_saved": baseline_tokens_saved,
                        "total_time_ms": baseline_total_time_ms,
                        "captured_at": time.time(),
                    }
                )

            session_total_commands = max(lifetime_total_commands - baseline_total_commands, 0)
            session_input_tokens = max(lifetime_input_tokens - baseline_input_tokens, 0)
            session_output_tokens = max(lifetime_output_tokens - baseline_output_tokens, 0)
            session_tokens_saved = max(lifetime_tokens_saved - baseline_tokens_saved, 0)
            session_total_time_ms = max(lifetime_total_time_ms - baseline_total_time_ms, 0)
            session_savings_pct = (
                round(session_tokens_saved / session_input_tokens * 100.0, 4)
                if session_input_tokens > 0
                else None
            )
            session_avg_time_ms = (
                round(session_total_time_ms / session_total_commands, 2)
                if session_total_commands > 0 and session_total_time_ms > 0
                else None
            )
            lifetime_savings_pct = float(payload.get("lifetime_avg_savings_pct", 0.0) or 0.0)

            payload = {
                **payload,
                "tool": tool,
                "label": _context_tool_label(tool),
                # Backward-compatible session-delta fields.
                "total_commands": session_total_commands,
                "input_tokens": session_input_tokens,
                "output_tokens": session_output_tokens,
                "tokens_saved": session_tokens_saved,
                "total_time_ms": session_total_time_ms,
                "session_savings_pct": session_savings_pct,
                "session_avg_time_ms": session_avg_time_ms,
                # Keep old field for compatibility, but declare its scope.
                "avg_savings_pct": lifetime_savings_pct,
                "avg_savings_pct_scope": "lifetime",
                "lifetime_avg_savings_pct": lifetime_savings_pct,
                "lifetime_total_commands": lifetime_total_commands,
                "lifetime_input_tokens": lifetime_input_tokens,
                "lifetime_output_tokens": lifetime_output_tokens,
                "lifetime_tokens_saved": lifetime_tokens_saved,
                "lifetime_total_time_ms": lifetime_total_time_ms,
                "session_baseline_total_commands": baseline_total_commands,
                "session_baseline_input_tokens": baseline_input_tokens,
                "session_baseline_output_tokens": baseline_output_tokens,
                "session_baseline_tokens_saved": baseline_tokens_saved,
                "session_baseline_total_time_ms": baseline_total_time_ms,
                "session_baseline_captured_at": _context_tool_session_baseline.get(
                    "captured_at", 0.0
                ),
                "session": {
                    "commands": session_total_commands,
                    "input_tokens": session_input_tokens,
                    "output_tokens": session_output_tokens,
                    "tokens_saved": session_tokens_saved,
                    "savings_pct": session_savings_pct,
                    "total_time_ms": session_total_time_ms,
                    "avg_time_ms": session_avg_time_ms,
                },
                "lifetime": {
                    "commands": lifetime_total_commands,
                    "input_tokens": lifetime_input_tokens,
                    "output_tokens": lifetime_output_tokens,
                    "tokens_saved": lifetime_tokens_saved,
                    "savings_pct": lifetime_savings_pct,
                    "total_time_ms": lifetime_total_time_ms,
                },
                "baseline": {
                    "commands": baseline_total_commands,
                    "input_tokens": baseline_input_tokens,
                    "output_tokens": baseline_output_tokens,
                    "tokens_saved": baseline_tokens_saved,
                    "total_time_ms": baseline_total_time_ms,
                    "captured_at": _context_tool_session_baseline.get("captured_at", 0.0),
                },
                "sampled_at": time.time(),
                "sample_ttl_seconds": CONTEXT_TOOL_STATS_CACHE_TTL_SECONDS,
                "refresh_interval_seconds": CONTEXT_TOOL_STATS_CACHE_TTL_SECONDS,
                "counter_reset_detected": counter_reset_detected,
            }

        _context_tool_stats_cache.update(
            {
                "expires_at": time.monotonic() + CONTEXT_TOOL_STATS_CACHE_TTL_SECONDS,
                "has_value": True,
                "tool": tool,
                "value": payload,
            }
        )
    return payload


def _get_rtk_stats() -> dict[str, Any] | None:
    """Backward-compatible alias for selected context-tool stats."""

    return _get_context_tool_stats()


def is_anthropic_auth(headers: dict[str, str]) -> bool:
    """Detect Anthropic auth signals in request headers."""
    if headers.get("x-api-key") or headers.get("anthropic-version"):
        return True
    auth = headers.get("authorization", "")
    if auth.startswith("Bearer sk-ant-"):
        return True
    return False


# ---------------------------------------------------------------------------
# Internal-header stripping (PR-A5 — fixes P5-49).
# ---------------------------------------------------------------------------
#
# `x-headroom-*` request headers (e.g. ``x-headroom-bypass``,
# ``x-headroom-mode``, ``x-headroom-user-id``, ``x-headroom-stack``,
# ``x-headroom-base-url``) are internal control flags consumed by the
# proxy itself. They MUST NOT leak upstream — leaking them would (a)
# fingerprint the proxy to subscription enforcers and (b) expose the
# user-id/stack/base-url internals to whichever vendor terminates the
# request.
#
# Inbound read paths (bypass gating, ``_extract_tags`` reading
# ``x-headroom-*``, memory ``x-headroom-user-id`` lookup) keep using
# the original dict / ``request.headers``. The stripped copy is what
# every upstream-bound forwarder receives.
#
# Note: response-side ``X-Headroom-*`` injection (e.g.
# ``x-headroom-tokens-saved``) is unrelated — the proxy is allowed to
# tell its client about its own work. This helper only filters
# request-side headers.

_INTERNAL_HEADER_PREFIX = "x-headroom-"

# Operator opt-in env var. ``enabled`` (default) strips internal
# ``x-headroom-*`` headers from every upstream-bound forwarder.
# ``disabled`` is an explicit operator opt-in for diagnostic shadow
# tracing — NOT a fallback. Per realignment build constraint #4 the
# behaviour is loud, configurable, and never silent.
_STRIP_INTERNAL_HEADERS_ENV = "HEADROOM_STRIP_INTERNAL_HEADERS"
StripInternalHeadersMode = Literal["enabled", "disabled"]
_STRIP_INTERNAL_HEADERS_DEFAULT: StripInternalHeadersMode = "enabled"


def get_strip_internal_headers_mode() -> StripInternalHeadersMode:
    """Return the active internal-header strip mode.

    Read at request time so operators can flip behaviour without a
    restart. Unknown values raise loudly per the no-silent-fallback
    build constraint.
    """
    raw = os.environ.get(_STRIP_INTERNAL_HEADERS_ENV, "").strip().lower()
    if not raw:
        return _STRIP_INTERNAL_HEADERS_DEFAULT
    if raw in ("enabled", "disabled"):
        return cast(StripInternalHeadersMode, raw)
    raise ValueError(
        f"Invalid {_STRIP_INTERNAL_HEADERS_ENV}={raw!r}; expected 'enabled' or 'disabled'"
    )


def _strip_internal_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return a copy of ``headers`` with internal ``x-headroom-*`` keys stripped.

    Used at every upstream call site to prevent fingerprinting / leakage of
    internal flags like ``x-headroom-bypass``, ``x-headroom-mode``,
    ``x-headroom-user-id``, ``x-headroom-stack``, ``x-headroom-base-url``.
    Case-insensitive on the prefix. Returns a NEW dict; never mutates the
    caller's mapping. Pure function. No regex.

    When the operator opt-in ``HEADROOM_STRIP_INTERNAL_HEADERS=disabled``
    is set, returns a shallow copy unchanged. That mode is for diagnostic
    shadow tracing only and is documented as a per-deploy choice.
    """
    mode = get_strip_internal_headers_mode()
    if mode == "disabled":
        # Always return a copy so callers can mutate without surprise.
        return dict(headers)
    return {k: v for k, v in headers.items() if not k.lower().startswith(_INTERNAL_HEADER_PREFIX)}


def log_outbound_headers(
    *,
    forwarder: str,
    stripped_count: int,
    request_id: str | None,
) -> None:
    """Structured log line for every upstream forwarder header strip.

    Emitted once per outbound request (paired with ``log_outbound_request``).
    Per realignment build constraint #8 we log every cache-affecting
    decision; per #8/#11 we never log header values, only the count of
    stripped internal headers.
    """
    logger.info(
        "event=outbound_headers forwarder=%s stripped_count=%d request_id=%s",
        forwarder,
        stripped_count,
        request_id or "",
    )


# ---------------------------------------------------------------------------
# Beta-header merge + per-session stickiness (PR-A6 — fixes P5-50; preps P0-6).
# ---------------------------------------------------------------------------
#
# Anthropic's `anthropic-beta` and OpenAI's `OpenAI-Beta` request headers
# carry a comma-separated list of opt-in beta tokens. Two cache-killer
# patterns motivated PR-A6:
#
#   1. Mid-session mutation: when memory is enabled the proxy historically
#      did an ad-hoc concat of `context-management-2025-06-27` onto the
#      client value (anthropic.py:1244-1248) — every variant produced a
#      different byte sequence and the order was undefined when the same
#      client value already contained a Headroom-required token.
#
#   2. Token drop-out across turns: clients (Claude Code, Codex CLI) MAY
#      drop a beta token between turn N and turn N+1 even when the proxy
#      mutated turn N to add it. The cache hot zone is positional, so the
#      next turn's prefix bytes hash differently and the prefix-cache
#      read misses.
#
# PR-A6 introduces:
#   * `merge_anthropic_beta` / `merge_openai_beta`: deterministic, pure,
#     order-preserving merge. Client tokens first (in their original order),
#     then Headroom-required tokens (in the order passed). Dedupe is
#     case-insensitive but preserves original casing of first occurrence.
#     Per Anthropic guide §6.3 #6: sticky-on means we add but never reorder.
#
#   * `SessionBetaTracker`: bounded LRU cache keyed by `(provider,
#     session_id)` tracking every beta token observed for that session.
#     On every request we union the client value with previously-seen
#     tokens and update the seen set — so a beta seen in turn N is
#     present in turn N+1 even if the client drops it. LRU bound (default
#     1000 sessions) prevents unbounded growth. Reentrant lock so future
#     callers from inside another locked method don't self-deadlock.
#
# Operator opt-in `HEADROOM_BETA_HEADER_STICKY=disabled` short-circuits
# the tracker (returns the client value verbatim). That mode is loud and
# explicit per realignment build constraint #4 — NOT a silent fallback.

_BETA_HEADER_STICKY_ENV = "HEADROOM_BETA_HEADER_STICKY"
BetaHeaderStickyMode = Literal["enabled", "disabled"]
_BETA_HEADER_STICKY_DEFAULT: BetaHeaderStickyMode = "enabled"

_BETA_TRACKER_MAX_SESSIONS_ENV = "HEADROOM_BETA_TRACKER_MAX_SESSIONS"
_BETA_TRACKER_MAX_SESSIONS_DEFAULT = 1000


def get_beta_header_sticky_mode() -> BetaHeaderStickyMode:
    """Return the active beta-header stickiness mode.

    Read at request time so operators can flip behaviour without a
    restart. Unknown values raise loudly per the no-silent-fallback
    build constraint.
    """
    raw = os.environ.get(_BETA_HEADER_STICKY_ENV, "").strip().lower()
    if not raw:
        return _BETA_HEADER_STICKY_DEFAULT
    if raw in ("enabled", "disabled"):
        return cast(BetaHeaderStickyMode, raw)
    raise ValueError(f"Invalid {_BETA_HEADER_STICKY_ENV}={raw!r}; expected 'enabled' or 'disabled'")


def get_beta_tracker_max_sessions() -> int:
    """Return the LRU bound for `SessionBetaTracker` (sessions cap)."""
    raw = os.environ.get(_BETA_TRACKER_MAX_SESSIONS_ENV, "").strip()
    if not raw:
        return _BETA_TRACKER_MAX_SESSIONS_DEFAULT
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {_BETA_TRACKER_MAX_SESSIONS_ENV}={raw!r}; expected positive int"
        ) from exc
    if value <= 0:
        raise ValueError(f"Invalid {_BETA_TRACKER_MAX_SESSIONS_ENV}={raw!r}; expected positive int")
    return value


def _split_beta_tokens(value: str | None) -> list[str]:
    """Split a comma-separated beta-header value into trimmed tokens.

    Empty/whitespace-only entries are dropped. Pure function, no regex.
    """
    if not value:
        return []
    out: list[str] = []
    for raw in value.split(","):
        token = raw.strip()
        if token:
            out.append(token)
    return out


def _merge_beta_tokens(client_value: str | None, headroom_required: list[str]) -> str:
    """Shared deterministic merge for `anthropic-beta` / `OpenAI-Beta` tokens.

    Rules (per Anthropic guide §6.3 #6 "sticky-on; add but never reorder"):

    * Client tokens come first, in their original order.
    * Headroom-required tokens append in the order given, skipping any
      token already present (case-insensitive).
    * Dedupe is case-insensitive but the FIRST occurrence's casing wins
      (prevents drift when client uses one casing across turns).
    * Returns ``""`` when both inputs are empty.

    Pure function. No regex. No global state.
    """
    seen_lower: set[str] = set()
    out: list[str] = []
    for token in _split_beta_tokens(client_value):
        lower = token.lower()
        if lower in seen_lower:
            continue
        seen_lower.add(lower)
        out.append(token)
    for token in headroom_required:
        if not token:
            continue
        token = token.strip()
        if not token:
            continue
        lower = token.lower()
        if lower in seen_lower:
            continue
        seen_lower.add(lower)
        out.append(token)
    return ",".join(out)


def merge_anthropic_beta(client_value: str | None, headroom_required: list[str]) -> str:
    """Merge client `anthropic-beta` value with Headroom-required tokens.

    See `_merge_beta_tokens` for full semantics. Order is deterministic:
    client tokens first (in their original order), then headroom tokens
    (in the order passed). No sorting — sticky-on per Anthropic guide
    §6.3 #6 means we add but never reorder. Dedupe is case-insensitive
    but preserves the original casing of the first occurrence.

    Returns ``""`` when both inputs are empty.
    """
    return _merge_beta_tokens(client_value, headroom_required)


def merge_openai_beta(client_value: str | None, headroom_required: list[str]) -> str:
    """Merge client `OpenAI-Beta` value with Headroom-required tokens.

    Mirror of `merge_anthropic_beta`. Same semantics — the OpenAI header
    follows the same comma-separated convention and the same cache-stable
    rules apply.
    """
    return _merge_beta_tokens(client_value, headroom_required)


class SessionBetaTracker:
    """Bounded LRU tracker of beta-header tokens observed per (provider, session).

    On every request:
      * Read the client's beta-header value.
      * Union with previously-seen tokens for this session (sticky-on).
      * Update the session's seen set.
      * Return the union (preserving first-seen order).

    Bounded by `max_sessions` (default 1000) via `OrderedDict` LRU
    eviction: hits move-to-end; overflow pops oldest. Reentrant lock so
    future callers from inside another locked method don't self-deadlock
    (mirrors `CompressionCache` pattern).

    The tracker is provider-aware: the same `session_id` for Anthropic
    and OpenAI keeps independent token sets (clients/upstreams differ on
    which tokens are valid).
    """

    def __init__(self, max_sessions: int | None = None) -> None:
        if max_sessions is None:
            max_sessions = get_beta_tracker_max_sessions()
        if max_sessions <= 0:
            raise ValueError("max_sessions must be > 0")
        self._max_sessions: int = max_sessions
        # OrderedDict per `compression_cache.py` LRU pattern. Entries
        # store the per-session ordered token list (preserving first-seen
        # order). RLock allows future callers from inside another locked
        # method to enter without self-deadlock.
        self._lock = threading.RLock()
        self._sessions: OrderedDict[tuple[str, str], list[str]] = OrderedDict()

    @property
    def active_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _key(self, provider: str, session_id: str) -> tuple[str, str]:
        return (provider, session_id)

    def record_and_get_sticky_betas(
        self,
        provider: str,
        session_id: str,
        client_value: str | None,
    ) -> str:
        """Union client tokens with session-seen tokens; update; return.

        ``provider`` is the upstream identifier (``anthropic`` /
        ``openai``). ``session_id`` is the proxy's per-conversation ID
        (e.g. `SessionTrackerStore.compute_session_id` output for the
        HTTP path; the WS handler's per-connection UUID for the WS
        path — note WS sessions are short-lived and won't accumulate
        cross-turn).

        When `HEADROOM_BETA_HEADER_STICKY=disabled` returns the client
        value verbatim (operator diagnostic opt-in; documented as a
        per-deploy choice, NOT a silent fallback).

        Returns the merged comma-separated value (possibly empty).
        """
        if not provider:
            raise ValueError("provider must be non-empty")
        if not session_id:
            raise ValueError("session_id must be non-empty")

        if get_beta_header_sticky_mode() == "disabled":
            # Diagnostic mode — return the client value verbatim, do not
            # touch tracker state. This is loud (operators read the env
            # var) and per-deploy.
            return (client_value or "").strip()

        client_tokens = _split_beta_tokens(client_value)
        key = self._key(provider, session_id)

        with self._lock:
            previous = self._sessions.get(key)
            if previous is None:
                merged_list: list[str] = []
                seen_lower: set[str] = set()
            else:
                # Move-to-end on hit (LRU touch).
                self._sessions.move_to_end(key)
                merged_list = list(previous)
                seen_lower = {t.lower() for t in merged_list}

            # Append client tokens preserving order; first-seen casing wins.
            for token in client_tokens:
                lower = token.lower()
                if lower in seen_lower:
                    continue
                seen_lower.add(lower)
                merged_list.append(token)

            self._sessions[key] = merged_list
            self._sessions.move_to_end(key)

            # Bound: evict oldest until at-or-below cap.
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)

            return ",".join(merged_list)

    def reset(self) -> None:
        """Clear all session state (test helper)."""
        with self._lock:
            self._sessions.clear()


# Process-wide singleton. Lazily replaced by tests via `reset` /
# `_reset_session_beta_tracker_for_test`. One tracker for both providers
# — the (provider, session_id) key keeps namespaces independent.
_session_beta_tracker_lock = threading.Lock()
_session_beta_tracker: SessionBetaTracker | None = None


def get_session_beta_tracker() -> SessionBetaTracker:
    """Return the process-wide `SessionBetaTracker` singleton.

    Lazily constructed so the env-var bound (`HEADROOM_BETA_TRACKER_MAX_SESSIONS`)
    is honored at first use. Tests use `_reset_session_beta_tracker_for_test`.
    """
    global _session_beta_tracker
    with _session_beta_tracker_lock:
        if _session_beta_tracker is None:
            _session_beta_tracker = SessionBetaTracker()
        return _session_beta_tracker


def _reset_session_beta_tracker_for_test() -> None:
    """Clear the process-wide tracker (test-only)."""
    global _session_beta_tracker
    with _session_beta_tracker_lock:
        _session_beta_tracker = None


def log_beta_header_merge(
    *,
    provider: str,
    session_id: str | None,
    client_betas_count: int,
    sticky_betas_count: int,
    headroom_added: list[str],
    request_id: str | None,
) -> None:
    """Structured log for every cache-affecting beta-header merge.

    `headroom_added` is a list of public, documented beta tokens
    (e.g. ``context-management-2025-06-27``,
    ``responses_websockets=2026-02-06``) — safe to log. We intentionally
    do NOT log the raw client value because beta tokens, while public,
    can carry experiment IDs the user has not opted to share with
    Headroom logs. Emitting counts only makes the decision auditable.
    """
    logger.info(
        "event=beta_header_merge provider=%s session_id=%s "
        "client_betas=%d sticky_betas=%d headroom_added=%s request_id=%s",
        provider,
        session_id or "",
        client_betas_count,
        sticky_betas_count,
        ",".join(headroom_added) if headroom_added else "",
        request_id or "",
    )


# ---------------------------------------------------------------------------
# Memory-tool injection session-stickiness (PR-A7 — closes P0-6).
# ---------------------------------------------------------------------------
#
# Memory adds `memory_save` / `memory_search` tool definitions to
# `body["tools"]` when memory is enabled for a request. The cache-killer
# pattern motivated by guide §6.3 #2 ("tool list change → cache bust"):
#
#   * Mid-session toggle: memory is enabled in turn N (tool definitions
#     injected) and disabled in turn N+1 (tool list shrinks). The next
#     turn's prefix bytes hash differently, prefix-cache misses, and the
#     full prompt re-runs at provider cost.
#
#   * Tool definition drift: memory adds the SAME logical tool but the
#     bytes differ across turns (insertion order, dict key order, schema
#     drift between deploys, etc.). Even with the tool list intact the
#     prefix bytes change.
#
# PR-A7 introduces:
#
#   * `SessionToolTracker`: bounded LRU keyed by (provider, session_id)
#     storing the GOLDEN tool-definition bytes injected on the first
#     turn. Subsequent turns of that session always inject the same
#     bytes — even if memory is disabled mid-session (sticky-on per
#     guide §6.3 #2). Provider-aware so the same `session_id` under
#     two providers keeps independent state.
#
# The golden bytes are produced by `serialize_body_canonical` of the
# tool definition object so they are deterministic across deploys
# regardless of dict insertion ordering quirks.
#
# Operator opt-in `HEADROOM_TOOL_INJECTION_STICKY=disabled` short-
# circuits the tracker; per-turn decision flows through unchanged. That
# mode is loud and explicit per realignment build constraint #4 — NOT a
# silent fallback. It exists for diagnostic shadow tracing / emergency
# rollback only.

_TOOL_INJECTION_STICKY_ENV = "HEADROOM_TOOL_INJECTION_STICKY"
ToolInjectionStickyMode = Literal["enabled", "disabled"]
_TOOL_INJECTION_STICKY_DEFAULT: ToolInjectionStickyMode = "enabled"

_TOOL_TRACKER_MAX_SESSIONS_ENV = "HEADROOM_TOOL_TRACKER_MAX_SESSIONS"
_TOOL_TRACKER_MAX_SESSIONS_DEFAULT = 1000


def get_tool_injection_sticky_mode() -> ToolInjectionStickyMode:
    """Return the active memory-tool stickiness mode.

    Read at request time so operators can flip behaviour without a
    restart. Unknown values raise loudly per the no-silent-fallback
    build constraint.
    """
    raw = os.environ.get(_TOOL_INJECTION_STICKY_ENV, "").strip().lower()
    if not raw:
        return _TOOL_INJECTION_STICKY_DEFAULT
    if raw in ("enabled", "disabled"):
        return cast(ToolInjectionStickyMode, raw)
    raise ValueError(
        f"Invalid {_TOOL_INJECTION_STICKY_ENV}={raw!r}; expected 'enabled' or 'disabled'"
    )


def get_tool_tracker_max_sessions() -> int:
    """Return the LRU bound for `SessionToolTracker` (sessions cap)."""
    raw = os.environ.get(_TOOL_TRACKER_MAX_SESSIONS_ENV, "").strip()
    if not raw:
        return _TOOL_TRACKER_MAX_SESSIONS_DEFAULT
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {_TOOL_TRACKER_MAX_SESSIONS_ENV}={raw!r}; expected positive int"
        ) from exc
    if value <= 0:
        raise ValueError(f"Invalid {_TOOL_TRACKER_MAX_SESSIONS_ENV}={raw!r}; expected positive int")
    return value


def serialize_tool_definition_canonical(tool_definition: dict[str, Any]) -> bytes:
    """Deterministic byte serialization of a single memory tool definition.

    Uses ``serialize_body_canonical`` semantics (compact separators, UTF-8,
    no ASCII escaping). Python 3.7+ dict insertion order is preserved by
    ``json.dumps`` so callers must construct the tool definition with a
    stable key order — which the static schemas in
    ``headroom/proxy/memory_handler.py`` and
    ``headroom/proxy/memory_tool_adapter.py`` already do.

    Returned bytes pin the golden tool definition for a session: every
    follow-up turn must inject byte-equal output to keep the prefix
    cache hot.
    """
    return serialize_body_canonical(tool_definition)


class SessionToolTracker:
    """Bounded LRU tracker recording per-session memory-tool injection state.

    Once memory injects tool definitions into a session, future requests
    in that session always inject the byte-equal same definitions —
    never toggling on/off mid-session (guide §6.3 #2). The first turn's
    canonical bytes are stored as the golden definition; subsequent
    turns reuse those bytes verbatim.

    State per session: ordered list of (tool_name → golden_bytes)
    pairs. Order is preserved so the rebuilt tool list matches the
    original injection order.

    Bounded by ``max_sessions`` (default 1000) via ``OrderedDict`` LRU
    eviction: hits move-to-end; overflow pops oldest. Reentrant lock so
    future callers from inside another locked method don't self-deadlock
    (mirrors `SessionBetaTracker` / `CompressionCache` pattern).

    The tracker is provider-aware: the same ``session_id`` for Anthropic
    and OpenAI keeps independent state (the tool schemas differ in
    format).
    """

    def __init__(self, max_sessions: int | None = None) -> None:
        if max_sessions is None:
            max_sessions = get_tool_tracker_max_sessions()
        if max_sessions <= 0:
            raise ValueError("max_sessions must be > 0")
        self._max_sessions: int = max_sessions
        self._lock = threading.RLock()
        # Value is an OrderedDict[tool_name -> golden_definition_bytes].
        # Storing per-tool bytes (not the entire tools list) keeps the
        # tracker resilient to non-memory tool list changes by the client
        # (which are the client's responsibility, not ours to gate).
        self._sessions: OrderedDict[tuple[str, str], OrderedDict[str, bytes]] = OrderedDict()

    @property
    def active_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _key(self, provider: str, session_id: str) -> tuple[str, str]:
        return (provider, session_id)

    def should_inject(self, provider: str, session_id: str) -> bool:
        """Return True iff this session has previously injected memory tools.

        Used by the sticky-on path: when memory is disabled this turn but
        the session previously injected, we still inject the golden bytes.
        """
        if not provider:
            raise ValueError("provider must be non-empty")
        if not session_id:
            raise ValueError("session_id must be non-empty")
        with self._lock:
            entry = self._sessions.get(self._key(provider, session_id))
            if entry is None:
                return False
            # LRU touch on read so the carry-over decision keeps the
            # session in the hot set.
            self._sessions.move_to_end(self._key(provider, session_id))
            return len(entry) > 0

    def get_golden_definitions(
        self, provider: str, session_id: str
    ) -> list[tuple[str, bytes]] | None:
        """Return the previously-recorded (name, bytes) pairs for the session.

        Returns ``None`` when the session has never injected memory tools.
        Callers replay the bytes verbatim into ``body["tools"]``.
        """
        if not provider:
            raise ValueError("provider must be non-empty")
        if not session_id:
            raise ValueError("session_id must be non-empty")
        with self._lock:
            entry = self._sessions.get(self._key(provider, session_id))
            if entry is None:
                return None
            self._sessions.move_to_end(self._key(provider, session_id))
            # Snapshot — never expose internal storage directly.
            return [(name, golden_bytes) for name, golden_bytes in entry.items()]

    def record_injection(
        self,
        provider: str,
        session_id: str,
        tool_name: str,
        tool_definition_bytes: bytes,
    ) -> None:
        """Record the golden bytes for a single memory tool in this session.

        First-write wins: re-recording the same ``tool_name`` for an
        existing session is a no-op (prevents drift if the canonical
        serialization output changed between deploys mid-session). For
        a *new* session, record fresh. LRU bound applies on every write.
        """
        if not provider:
            raise ValueError("provider must be non-empty")
        if not session_id:
            raise ValueError("session_id must be non-empty")
        if not tool_name:
            raise ValueError("tool_name must be non-empty")
        if not tool_definition_bytes:
            raise ValueError("tool_definition_bytes must be non-empty")

        key = self._key(provider, session_id)

        with self._lock:
            entry = self._sessions.get(key)
            if entry is None:
                entry = OrderedDict()
                self._sessions[key] = entry

            # First-write wins: only record if not already pinned.
            if tool_name not in entry:
                entry[tool_name] = tool_definition_bytes

            # LRU touch + bound enforcement.
            self._sessions.move_to_end(key)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)

    def reset(self) -> None:
        """Clear all session state (test helper)."""
        with self._lock:
            self._sessions.clear()


# Process-wide singleton. Lazily replaced by tests via
# `_reset_session_tool_tracker_for_test`.
_session_tool_tracker_lock = threading.Lock()
_session_tool_tracker: SessionToolTracker | None = None


def get_session_tool_tracker() -> SessionToolTracker:
    """Return the process-wide `SessionToolTracker` singleton.

    Lazily constructed so the env-var bound
    (`HEADROOM_TOOL_TRACKER_MAX_SESSIONS`) is honored at first use.
    Tests use ``_reset_session_tool_tracker_for_test``.
    """
    global _session_tool_tracker
    with _session_tool_tracker_lock:
        if _session_tool_tracker is None:
            _session_tool_tracker = SessionToolTracker()
        return _session_tool_tracker


def _reset_session_tool_tracker_for_test() -> None:
    """Clear the process-wide tracker (test-only)."""
    global _session_tool_tracker
    with _session_tool_tracker_lock:
        _session_tool_tracker = None


def log_tool_injection_decision(
    *,
    provider: str,
    session_id: str | None,
    decision: Literal[
        "inject_first_time",
        "inject_sticky_replay",
        "skip",
        "skip_disabled_via_env",
    ],
    tool_definition_bytes_count: int,
    request_id: str | None,
) -> None:
    """Structured log for every cache-affecting tool-injection decision.

    Per realignment build constraint #8 we log every cache-affecting
    decision. ``tool_definition_bytes_count`` is the per-tool byte count
    summed across all memory tools injected this turn. We do NOT log the
    tool definition contents (might contain user-specific schemas) per
    constraint #11.
    """
    logger.info(
        "event=tool_injection_decision provider=%s session_id=%s "
        "decision=%s tool_definition_bytes_count=%d request_id=%s",
        provider,
        session_id or "",
        decision,
        tool_definition_bytes_count,
        request_id or "",
    )


def _extract_tool_name(tool_definition: dict[str, Any]) -> str | None:
    """Extract a stable tool name from a memory tool definition.

    Handles three formats:
      * Anthropic custom: ``{"name": "memory_save", ...}``
      * Anthropic native: ``{"type": "memory_20250818", "name": "memory"}``
      * OpenAI function: ``{"type": "function", "function": {"name": "memory_save", ...}}``
    """
    name = tool_definition.get("name")
    if isinstance(name, str) and name:
        return name
    fn = tool_definition.get("function")
    if isinstance(fn, dict):
        fn_name = fn.get("name")
        if isinstance(fn_name, str) and fn_name:
            return fn_name
    # Native memory tool with no explicit name uses ``type`` as its identifier.
    type_val = tool_definition.get("type")
    if isinstance(type_val, str) and type_val:
        return type_val
    return None


def apply_session_sticky_memory_tools(
    *,
    provider: Literal["anthropic", "openai"],
    session_id: str | None,
    request_id: str | None,
    existing_tools: list[dict[str, Any]] | None,
    memory_tools_to_inject: list[dict[str, Any]],
    inject_this_turn: bool,
) -> tuple[list[dict[str, Any]], bool]:
    """Apply sticky-on memory tool injection per `SessionToolTracker`.

    The single coordination point for all memory-tool injection sites
    (Anthropic custom tools, Anthropic native tool, OpenAI function tools).

    Logic (guide §6.3 #2):

      * If ``HEADROOM_TOOL_INJECTION_STICKY=disabled``: bypass tracker,
        inject only when ``inject_this_turn`` is True. Diagnostic mode.

      * If session previously injected and tracker has golden bytes:
        ALWAYS inject the golden bytes verbatim (sticky-on). Memory-this-
        turn flag is irrelevant — once injected, always injected.

      * If session has NOT previously injected:
          - ``inject_this_turn=True``: serialize ``memory_tools_to_inject``,
            record golden bytes, append to tools list.
          - ``inject_this_turn=False``: skip; no future replay obligation.

    Memory tools whose names already appear in ``existing_tools`` are
    NOT re-appended (the client owns the canonical definition then).

    ``session_id`` may be ``None`` (e.g. WS path with no per-turn
    session); in that case the tracker is bypassed and the caller's
    ``inject_this_turn`` flag drives the decision verbatim. We log the
    bypass once so operators can see it.

    Returns ``(updated_tools, was_injected)``. The returned list is a
    fresh list (caller-safe). ``was_injected`` is True iff at least one
    memory tool was added to the list.
    """
    if provider not in ("anthropic", "openai"):
        raise ValueError(f"unsupported provider: {provider!r}")

    tools_out: list[dict[str, Any]] = list(existing_tools) if existing_tools else []
    existing_names: set[str] = set()
    for t in tools_out:
        n = _extract_tool_name(t)
        if n:
            existing_names.add(n)

    # Diagnostic / rollback path.
    if get_tool_injection_sticky_mode() == "disabled":
        if not inject_this_turn:
            log_tool_injection_decision(
                provider=provider,
                session_id=session_id,
                decision="skip_disabled_via_env",
                tool_definition_bytes_count=0,
                request_id=request_id,
            )
            return tools_out, False
        # Disabled mode + inject_this_turn=True: append the definitions
        # verbatim without recording golden bytes (per-turn decision
        # passes through as the broken behavior — explicit operator
        # opt-in only). Skip names already in the list.
        added_bytes = 0
        for tool_def in memory_tools_to_inject:
            tn = _extract_tool_name(tool_def)
            if tn is None or tn in existing_names:
                continue
            tools_out.append(tool_def)
            existing_names.add(tn)
            added_bytes += len(serialize_tool_definition_canonical(tool_def))
        log_tool_injection_decision(
            provider=provider,
            session_id=session_id,
            decision="skip_disabled_via_env",
            tool_definition_bytes_count=added_bytes,
            request_id=request_id,
        )
        return tools_out, added_bytes > 0

    # Sticky path requires a session_id. None means we cannot track —
    # fall back to the caller's per-turn decision (loud, single log line)
    # so WS handlers / pre-session paths remain functional.
    if not session_id:
        if not inject_this_turn:
            log_tool_injection_decision(
                provider=provider,
                session_id=None,
                decision="skip",
                tool_definition_bytes_count=0,
                request_id=request_id,
            )
            return tools_out, False
        added_bytes = 0
        for tool_def in memory_tools_to_inject:
            tn = _extract_tool_name(tool_def)
            if tn is None or tn in existing_names:
                continue
            tools_out.append(tool_def)
            existing_names.add(tn)
            added_bytes += len(serialize_tool_definition_canonical(tool_def))
        log_tool_injection_decision(
            provider=provider,
            session_id=None,
            decision="inject_first_time",
            tool_definition_bytes_count=added_bytes,
            request_id=request_id,
        )
        return tools_out, added_bytes > 0

    tracker = get_session_tool_tracker()
    previously_injected = tracker.should_inject(provider, session_id)

    if previously_injected:
        # Sticky replay: always inject the golden bytes. inject_this_turn
        # flag is intentionally ignored (memory may be disabled this turn
        # but the cache prefix demands the same tool list as before).
        golden = tracker.get_golden_definitions(provider, session_id) or []
        replay_bytes = 0
        for tool_name, golden_bytes in golden:
            if tool_name in existing_names:
                # Client also has a tool by this name — don't double up.
                # Their bytes win (the client's choice, not ours to gate).
                continue
            try:
                tool_def = json.loads(golden_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                logger.error(
                    "corrupt golden tool bytes for session %s tool %s: %s — skipping tool injection",
                    session_id,
                    tool_name,
                    exc,
                    exc_info=True,
                )
                continue
            tools_out.append(tool_def)
            existing_names.add(tool_name)
            replay_bytes += len(golden_bytes)
        log_tool_injection_decision(
            provider=provider,
            session_id=session_id,
            decision="inject_sticky_replay",
            tool_definition_bytes_count=replay_bytes,
            request_id=request_id,
        )
        return tools_out, replay_bytes > 0

    # Fresh session.
    if not inject_this_turn:
        log_tool_injection_decision(
            provider=provider,
            session_id=session_id,
            decision="skip",
            tool_definition_bytes_count=0,
            request_id=request_id,
        )
        return tools_out, False

    # First-time inject: serialize, record, append.
    added_bytes = 0
    for tool_def in memory_tools_to_inject:
        tn = _extract_tool_name(tool_def)
        if tn is None or tn in existing_names:
            continue
        golden_bytes = serialize_tool_definition_canonical(tool_def)
        tracker.record_injection(
            provider=provider,
            session_id=session_id,
            tool_name=tn,
            tool_definition_bytes=golden_bytes,
        )
        tools_out.append(tool_def)
        existing_names.add(tn)
        added_bytes += len(golden_bytes)
    log_tool_injection_decision(
        provider=provider,
        session_id=session_id,
        decision="inject_first_time",
        tool_definition_bytes_count=added_bytes,
        request_id=request_id,
    )
    return tools_out, added_bytes > 0


# ─── Session-sticky CCR tool injection (PR-B7) ─────────────────────────
#
# Per realignment plan PR-B7 (`REALIGNMENT/04-phase-B-live-zone.md`):
# once a session has performed any CCR compression, the
# `headroom_retrieve` tool stays registered in `body["tools"]` for every
# subsequent request in that session — never toggled off.
#
# The legacy `CCRToolInjector.has_compressed_content` flips on/off based
# on whether the *latest request* contained compression markers, which
# bust the prompt cache every time the flag flips. Sticky-on means the
# tool list bytes stay byte-stable across turns once injected.


class SessionCcrTracker:
    """Bounded LRU tracker recording per-(provider, session_id) CCR state.

    Two pieces of state per session:

      * ``has_done_ccr``: True once the proxy observed any CCR
        compression marker in the messages of a request. Once True, it
        never flips back to False (the prompt cache anchored on the
        previous turn's tool list demands the tool stays present).
      * ``golden_tool_bytes``: canonical serialization of the
        ``headroom_retrieve`` tool definition recorded the first time
        the tracker injected it. Subsequent turns replay these bytes
        verbatim.

    Bounded by ``max_sessions`` via ``OrderedDict`` LRU. Mirrors
    :class:`SessionToolTracker` semantics so the operator's mental model
    is one tracker pattern, not two.
    """

    def __init__(self, max_sessions: int | None = None) -> None:
        if max_sessions is None:
            max_sessions = get_tool_tracker_max_sessions()
        if max_sessions <= 0:
            raise ValueError("max_sessions must be > 0")
        self._max_sessions = max_sessions
        self._lock = threading.RLock()
        # Value is (has_done_ccr, golden_tool_bytes_or_none).
        self._sessions: OrderedDict[tuple[str, str], tuple[bool, bytes | None]] = OrderedDict()

    @property
    def active_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _key(self, provider: str, session_id: str) -> tuple[str, str]:
        return (provider, session_id)

    def has_done_ccr(self, provider: str, session_id: str) -> bool:
        """Return True iff this session has previously performed CCR."""
        if not provider:
            raise ValueError("provider must be non-empty")
        if not session_id:
            raise ValueError("session_id must be non-empty")
        with self._lock:
            entry = self._sessions.get(self._key(provider, session_id))
            if entry is None:
                return False
            self._sessions.move_to_end(self._key(provider, session_id))
            return entry[0]

    def get_golden_tool_bytes(self, provider: str, session_id: str) -> bytes | None:
        """Return the recorded golden tool-definition bytes, or None."""
        if not provider:
            raise ValueError("provider must be non-empty")
        if not session_id:
            raise ValueError("session_id must be non-empty")
        with self._lock:
            entry = self._sessions.get(self._key(provider, session_id))
            if entry is None:
                return None
            self._sessions.move_to_end(self._key(provider, session_id))
            return entry[1]

    def record_ccr_done(
        self,
        provider: str,
        session_id: str,
        golden_tool_bytes: bytes,
    ) -> None:
        """Mark the session as having performed CCR and pin the golden bytes.

        First-write wins for ``golden_tool_bytes`` (subsequent calls
        with the same session keep the original bytes — prevents drift
        if the canonical serialization changed mid-session). The
        ``has_done_ccr`` flag is monotonic: once True, never False.
        """
        if not provider:
            raise ValueError("provider must be non-empty")
        if not session_id:
            raise ValueError("session_id must be non-empty")
        if not golden_tool_bytes:
            raise ValueError("golden_tool_bytes must be non-empty")
        key = self._key(provider, session_id)
        with self._lock:
            existing = self._sessions.get(key)
            if existing is None:
                self._sessions[key] = (True, golden_tool_bytes)
            else:
                # Preserve original golden bytes; just promote the flag.
                pinned = existing[1] if existing[1] is not None else golden_tool_bytes
                self._sessions[key] = (True, pinned)
            self._sessions.move_to_end(key)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)

    def reset(self) -> None:
        """Clear all session state (test helper)."""
        with self._lock:
            self._sessions.clear()


# Process-wide singleton.
_session_ccr_tracker_lock = threading.Lock()
_session_ccr_tracker: SessionCcrTracker | None = None


def get_session_ccr_tracker() -> SessionCcrTracker:
    """Return the process-wide :class:`SessionCcrTracker` singleton."""
    global _session_ccr_tracker
    with _session_ccr_tracker_lock:
        if _session_ccr_tracker is None:
            _session_ccr_tracker = SessionCcrTracker()
        return _session_ccr_tracker


def _reset_session_ccr_tracker_for_test() -> None:
    """Clear the process-wide CCR tracker (test-only)."""
    global _session_ccr_tracker
    with _session_ccr_tracker_lock:
        _session_ccr_tracker = None


def should_inject_ccr_tool(
    *,
    configured_inject_tool: bool,
    frozen_message_count: int,
    has_compressed_content: bool,
) -> tuple[bool, bool]:
    """Decide whether the ``headroom_retrieve`` tool must be injected this turn.

    This is the decision the Anthropic handler used to inline. It is extracted
    so the #1006 regression can be pinned at the decision point itself.

    Tool injection is normally deferred when there is a frozen message prefix
    (``frozen_message_count > 0``) to preserve the prompt cache. But if
    compression emitted fresh markers this turn, deferring would hand the agent
    a ``<<ccr:hash>>`` marker with no tool to redeem it — silent data loss. In
    that case we override the deferral and inject anyway (one cache miss is
    cheaper than dropped content).

    Returns ``(should_inject, is_marker_override)``. ``is_marker_override`` is
    True only when injection happens *because* of new markers despite a deferral,
    so the caller can log the override distinctly.
    """
    inject_tool = configured_inject_tool
    if inject_tool and frozen_message_count > 0:
        inject_tool = False  # defer to preserve cache
    is_marker_override = not inject_tool and has_compressed_content
    return (inject_tool or is_marker_override), is_marker_override


def apply_session_sticky_ccr_tool(
    *,
    provider: Literal["anthropic", "openai", "google"],
    session_id: str | None,
    request_id: str | None,
    existing_tools: list[dict[str, Any]] | None,
    has_compressed_content_this_turn: bool,
) -> tuple[list[dict[str, Any]], bool]:
    """Apply sticky-on CCR retrieval-tool injection per :class:`SessionCcrTracker`.

    Coordination point for both Anthropic and OpenAI handlers — replaces
    the legacy ``CCRToolInjector.inject_tool_definition`` "flip on, flip
    off" behaviour.

    Logic:

      * If ``session_id`` is None: tracker is bypassed and the per-turn
        ``has_compressed_content_this_turn`` flag drives the decision
        verbatim (matching legacy behaviour for WS / pre-session paths).
      * If the session has previously done CCR (``has_done_ccr``):
        ALWAYS inject the recorded golden bytes — even if this turn has
        no fresh compression. That is the load-bearing PR-B7 fix.
      * Otherwise, inject only when this turn produced compressed content.
        The first injection records the golden bytes for future turns.

    Tools whose name already equals ``CCR_TOOL_NAME`` (e.g. the client
    pre-registered it via MCP) are not re-appended; the client's bytes
    win.

    Returns ``(updated_tools, was_injected)``. ``updated_tools`` is a
    fresh list (caller-safe).
    """
    from headroom.ccr.tool_injection import CCR_TOOL_NAME, create_ccr_tool_definition

    if provider not in ("anthropic", "openai", "google"):
        raise ValueError(f"unsupported provider: {provider!r}")

    tools_out: list[dict[str, Any]] = list(existing_tools) if existing_tools else []
    existing_names: set[str] = set()
    for t in tools_out:
        n = _extract_tool_name(t)
        if n:
            existing_names.add(n)

    # Client (or MCP) already provided a tool by this name — don't double up.
    if CCR_TOOL_NAME in existing_names:
        log_tool_injection_decision(
            provider=provider,
            session_id=session_id,
            decision="skip",
            tool_definition_bytes_count=0,
            request_id=request_id,
        )
        return tools_out, False

    # No session_id (e.g. WS path): per-turn decision drives directly.
    if not session_id:
        if not has_compressed_content_this_turn:
            log_tool_injection_decision(
                provider=provider,
                session_id=None,
                decision="skip",
                tool_definition_bytes_count=0,
                request_id=request_id,
            )
            return tools_out, False
        tool_def = create_ccr_tool_definition(provider)
        canonical = serialize_tool_definition_canonical(tool_def)
        tools_out.append(tool_def)
        log_tool_injection_decision(
            provider=provider,
            session_id=None,
            decision="inject_first_time",
            tool_definition_bytes_count=len(canonical),
            request_id=request_id,
        )
        return tools_out, True

    tracker = get_session_ccr_tracker()
    previously_done = tracker.has_done_ccr(provider, session_id)

    if previously_done:
        # Sticky replay path. Always inject — even if this turn had no
        # fresh CCR compression. Prefer the recorded golden bytes; fall
        # back to a freshly serialized definition if (somehow) the
        # tracker lost them. Loud per build constraint #4: we log the
        # path taken either way.
        golden = tracker.get_golden_tool_bytes(provider, session_id)
        if golden is not None:
            try:
                tool_def = json.loads(golden.decode("utf-8"))
                tools_out.append(tool_def)
                log_tool_injection_decision(
                    provider=provider,
                    session_id=session_id,
                    decision="inject_sticky_replay",
                    tool_definition_bytes_count=len(golden),
                    request_id=request_id,
                )
                return tools_out, True
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                logger.error(
                    "corrupt golden CCR tool bytes for session %s: %s — regenerating fresh definition",
                    session_id,
                    exc,
                    exc_info=True,
                )
                # Fall through to fresh creation below
        # Tracker says "done CCR" but has no golden bytes (or they were corrupt). Pin
        # them now so future turns are stable.
        tool_def = create_ccr_tool_definition(provider)
        canonical = serialize_tool_definition_canonical(tool_def)
        tracker.record_ccr_done(provider, session_id, canonical)
        tools_out.append(tool_def)
        log_tool_injection_decision(
            provider=provider,
            session_id=session_id,
            decision="inject_sticky_replay",
            tool_definition_bytes_count=len(canonical),
            request_id=request_id,
        )
        return tools_out, True

    # Fresh session — only inject when this turn produced compressed content.
    if not has_compressed_content_this_turn:
        log_tool_injection_decision(
            provider=provider,
            session_id=session_id,
            decision="skip",
            tool_definition_bytes_count=0,
            request_id=request_id,
        )
        return tools_out, False

    tool_def = create_ccr_tool_definition(provider)
    canonical = serialize_tool_definition_canonical(tool_def)
    tracker.record_ccr_done(provider, session_id, canonical)
    tools_out.append(tool_def)
    log_tool_injection_decision(
        provider=provider,
        session_id=session_id,
        decision="inject_first_time",
        tool_definition_bytes_count=len(canonical),
        request_id=request_id,
    )
    return tools_out, True


async def _read_request_body_bytes(request: Request) -> bytes:
    """Read and (if needed) decompress the request body, returning raw UTF-8 bytes.

    Mirrors ``_read_request_json`` but returns the bytes pre-parse so
    forwarders can implement byte-faithful passthrough (PR-A3, fixes P0-2).
    Raises ``ValueError`` on any decompression failure.
    """
    encoding = (request.headers.get("content-encoding") or "").lower().strip()
    raw = await request.body()

    if encoding in ("zstd", "zstandard"):
        try:
            import zstandard

            dctx = zstandard.ZstdDecompressor()
            reader = dctx.stream_reader(raw)
            raw = reader.read()
            reader.close()
        except ImportError:
            raise ValueError(
                "Request body is zstd-compressed but the 'zstandard' package is not installed. "
                "Install it with: pip install zstandard"
            ) from None
        except Exception as exc:
            raise ValueError(f"Failed to decompress zstd request body: {exc}") from exc
    elif encoding == "gzip":
        import gzip as _gzip

        try:
            raw = _gzip.decompress(raw)
        except Exception as exc:
            raise ValueError(f"Failed to decompress gzip request body: {exc}") from exc
    elif encoding == "deflate":
        import zlib

        try:
            raw = zlib.decompress(raw)
        except Exception as exc:
            raise ValueError(f"Failed to decompress deflate request body: {exc}") from exc
    elif encoding == "br":
        try:
            import brotli

            raw = brotli.decompress(raw)
        except ImportError:
            raise ValueError(
                "Request body is brotli-compressed but the 'brotli' package is not installed."
            ) from None
        except Exception as exc:
            raise ValueError(f"Failed to decompress brotli request body: {exc}") from exc
    elif encoding and encoding != "identity":
        raise ValueError(f"Unsupported Content-Encoding: {encoding}")

    return cast(bytes, raw)


async def _read_request_json(request: Request) -> dict[str, Any]:
    """Read and parse JSON from a request, handling compressed bodies.

    Clients like OpenAI Codex may send zstd, gzip, or deflate-compressed
    request bodies.  Starlette's ``request.json()`` does not decompress
    automatically, causing a UnicodeDecodeError on compressed bytes.

    This helper inspects ``Content-Encoding``, decompresses if needed,
    then JSON-decodes the result.  It raises ``ValueError`` on any
    decompression or parse failure so callers can return a clean 400.
    """
    raw = await _read_request_body_bytes(request)

    # Decode and parse JSON
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Request body is not valid UTF-8 (possibly compressed?): {exc}") from exc

    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Request body must be a JSON object, not " + type(result).__name__)
    return result


async def read_request_json_with_bytes(request: Request) -> tuple[dict[str, Any], bytes]:
    """Read JSON body AND return the original (decompressed) bytes.

    Returned bytes are post-content-decoding (zstd/gzip/deflate/br are
    decompressed) so they represent the body as the upstream API will
    receive it. Forwarders pair this with a ``BodyMutationTracker`` to
    decide between passthrough and canonical re-serialization.
    """
    raw = await _read_request_body_bytes(request)

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Request body is not valid UTF-8 (possibly compressed?): {exc}") from exc

    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Request body must be a JSON object, not " + type(result).__name__)
    return result, raw


def _strip_per_call_annotations(obj: Any) -> Any:
    """Remove annotations that clients mutate between calls in one agent loop.

    ``cache_control`` is the main offender: clients (notably Claude Code)
    move the cache breakpoint to the newest message on each call, which
    means the exact same user-text message carries ``cache_control`` on
    call 1 and not on call 2. Hashing the raw message dicts therefore
    produces a different turn_id for every iteration of a single agent
    loop, collapsing ``turn_id`` to effectively ``request_id`` and
    breaking prompt-level aggregation downstream.
    """
    if isinstance(obj, dict):
        return {k: _strip_per_call_annotations(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip_per_call_annotations(item) for item in obj]
    return obj


def compute_turn_id(
    model: str,
    system: Any,
    messages: list[dict[str, Any]] | None,
) -> str | None:
    """Group all agent-loop API calls triggered by a single user prompt.

    A turn spans the user's text prompt plus every assistant tool-use and
    user tool-result message the agent appends while executing that prompt.
    Hashing the prefix up to and including the last user *text* message yields
    an id that is stable across the turn but rolls over when the user sends a
    new prompt.

    Returns None when no user-text message is present (nothing to identify).
    """
    if not messages:
        return None

    last_text_user_idx: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content:
            last_text_user_idx = i
            break
        if isinstance(content, list):
            has_text = any(
                isinstance(block, dict) and block.get("type") == "text" for block in content
            )
            has_tool_result = any(
                isinstance(block, dict) and block.get("type") == "tool_result" for block in content
            )
            # An agent-loop continuation carries tool_result blocks; only a
            # fresh user turn is text-only.
            if has_text and not has_tool_result:
                last_text_user_idx = i
                break

    if last_text_user_idx is None:
        return None

    prefix = _strip_per_call_annotations(messages[: last_text_user_idx + 1])
    try:
        prefix_json = json.dumps(prefix, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None

    h = hashlib.sha256()
    h.update(model.encode("utf-8", errors="replace"))
    h.update(b"\0")
    if isinstance(system, str):
        h.update(system.encode("utf-8", errors="replace"))
    elif system is not None:
        try:
            normalized_system = _strip_per_call_annotations(system)
            h.update(json.dumps(normalized_system, sort_keys=True, default=str).encode("utf-8"))
        except (TypeError, ValueError):
            pass
    h.update(b"\0")
    h.update(prefix_json.encode("utf-8", errors="replace"))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Issue #746: Claude Code on-demand tool loading (deferral) detection
#
# When Claude Code points at a custom ``ANTHROPIC_BASE_URL`` (the proxy) with
# ``ENABLE_TOOL_SEARCH`` unset, it stops deferring MCP/system tool schemas
# behind the server-side Tool Search Tool and materializes them all into its
# local context window — tens of K tokens. That decision is made client-side
# before the request reaches us, so the proxy cannot reverse it; the only
# remedy is the ``ENABLE_TOOL_SEARCH`` env var (set automatically by
# ``headroom wrap claude``). For users who run ``claude`` manually we cannot
# touch their environment, so the proxy emits a single actionable hint.
# ---------------------------------------------------------------------------

_TOOL_SEARCH_TOOL_TYPE_PREFIX = "tool_search_tool_"
# Substrings of the ``anthropic-beta`` tokens that gate tool search:
# ``advanced-tool-use-2025-11-20`` (firstParty/foundry) and
# ``tool-search-tool-2025-10-19`` (vertex/bedrock/mantle/gateway).
_TOOL_SEARCH_BETA_MARKERS = ("advanced-tool-use", "tool-search-tool")

_tool_search_hint_lock = threading.Lock()
_tool_search_hint_emitted = False


def claude_code_tool_search_inactive(
    *,
    client: str | None,
    tools: Any,
    anthropic_beta: str | None,
) -> bool:
    """Return ``True`` when a Claude Code request is *not* deferring tools.

    Detected from request shape alone — no token thresholds, so it scales to
    any tool surface:

    * the request is from Claude Code (``client == "claude-code"``),
    * it carries one or more tool definitions, yet
    * it includes neither a ``tool_search_tool_*`` tool nor a tool-search
      ``anthropic-beta`` token.

    In that combination Claude Code has eagerly materialized every tool schema
    into its local context window (issue #746).
    """
    if client != "claude-code":
        return False
    if not isinstance(tools, list) or not tools:
        return False
    for tool in tools:
        if isinstance(tool, dict) and str(tool.get("type", "")).startswith(
            _TOOL_SEARCH_TOOL_TYPE_PREFIX
        ):
            return False
    beta = (anthropic_beta or "").lower()
    return not any(marker in beta for marker in _TOOL_SEARCH_BETA_MARKERS)


def format_tool_search_disabled_hint(tools: list[Any]) -> str:
    """Build the one-time, actionable hint for issue #746.

    Reports factual, directional numbers (tool count and serialized schema
    size) rather than a derived token estimate, which avoids implying a
    precision the proxy cannot measure for the client's tokenizer.
    """
    try:
        schema_kb = len(json.dumps(tools, separators=(",", ":"), default=str)) / 1024
    except (TypeError, ValueError):
        schema_kb = 0.0
    return (
        f"Claude Code is sending all {len(tools)} tool definitions eagerly "
        f"(~{schema_kb:.0f} KB of tool schema in local context) because "
        "ENABLE_TOOL_SEARCH is unset with a custom ANTHROPIC_BASE_URL. Set "
        "ENABLE_TOOL_SEARCH=true (or auto) to keep on-demand tool loading active, "
        "or launch via `headroom wrap claude` (which sets it automatically). "
        "See https://github.com/chopratejas/headroom/issues/746"
    )


def tool_search_hint_pending() -> bool:
    """Cheap, lock-free check of whether the one-time hint may still fire.

    Lets the request hot path skip the (O(number-of-tools)) detection scan on
    every request once the hint has already been emitted. A benign race here
    only costs one extra detection scan, never a duplicate warning — the
    actual one-shot guarantee lives in :func:`take_tool_search_hint_slot`.
    """
    return not _tool_search_hint_emitted


def take_tool_search_hint_slot() -> bool:
    """Return ``True`` exactly once per process, gating the one-time hint.

    Thread-safe so concurrent requests cannot each emit the warning.
    """
    global _tool_search_hint_emitted
    if _tool_search_hint_emitted:
        return False
    with _tool_search_hint_lock:
        if _tool_search_hint_emitted:
            return False
        _tool_search_hint_emitted = True
        return True


def reset_tool_search_hint_state() -> None:
    """Reset the one-time hint guard. Test helper only."""
    global _tool_search_hint_emitted
    with _tool_search_hint_lock:
        _tool_search_hint_emitted = False
