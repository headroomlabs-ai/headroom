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
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from headroom import paths as _paths
from headroom.proxy import (
    diagnostic_decode_policy,
    memory_injection_mode_policy,
    query_log_policy,
    request_limit_policy,
    sse_byte_buffer_policy,
    wire_debug_format_policy,
    wire_debug_redaction_policy,
)
from headroom.proxy.beta_header_merge import (
    merge_anthropic_beta as merge_anthropic_beta,
)
from headroom.proxy.beta_header_merge import (
    merge_beta_tokens,
    split_beta_tokens,
)
from headroom.proxy.beta_header_merge import (
    merge_openai_beta as merge_openai_beta,
)
from headroom.proxy.beta_header_policy import (
    BETA_HEADER_STICKY_DEFAULT,
    BETA_HEADER_STICKY_ENV,
    BETA_TRACKER_MAX_SESSIONS_DEFAULT,
    BETA_TRACKER_MAX_SESSIONS_ENV,
    BetaHeaderStickyMode,
    resolve_beta_header_sticky_mode,
    resolve_beta_tracker_max_sessions,
)
from headroom.proxy.body_forwarding import (
    BodyMutationTracker as BodyMutationTracker,  # noqa: F401 - compatibility export
)
from headroom.proxy.body_forwarding import (
    PythonForwarderMode as PythonForwarderMode,  # noqa: F401 - compatibility export
)
from headroom.proxy.body_forwarding import (
    get_python_forwarder_mode as get_python_forwarder_mode,  # noqa: F401 - compatibility export
)
from headroom.proxy.body_forwarding import (
    prepare_outbound_body_bytes as prepare_outbound_body_bytes,  # noqa: F401 - compatibility export
)
from headroom.proxy.body_forwarding import (
    serialize_body_canonical as serialize_body_canonical,  # noqa: F401 - compatibility export
)
from headroom.proxy.ccr_golden_policy import (
    create_fresh_ccr_tool_definition,
    replay_golden_ccr_tool_definition,
)
from headroom.proxy.ccr_marker_policy import (
    has_new_ccr_markers as _has_new_ccr_markers,
)
from headroom.proxy.ccr_session_tracker import SessionCcrTracker as _SessionCcrTracker
from headroom.proxy.internal_header_policy import (
    INTERNAL_HEADER_PREFIX,
    STRIP_INTERNAL_HEADERS_DEFAULT,
    STRIP_INTERNAL_HEADERS_ENV,
    StripInternalHeadersMode,
    resolve_strip_internal_headers_mode,
    strip_internal_headers,
)
from headroom.proxy.memory_golden_policy import (
    replay_golden_memory_tool_definition,
    serialize_memory_tool_definition_canonical,
)
from headroom.proxy.tool_definition_serialization import (
    serialize_tool_definition_canonical as _serialize_tool_definition_canonical,
)
from headroom.proxy.tool_injection_config import (
    ToolInjectionStickyMode,
)
from headroom.proxy.tool_injection_config import (
    get_tool_injection_sticky_mode as _get_tool_injection_sticky_mode,
)
from headroom.proxy.tool_injection_config import (
    get_tool_tracker_max_sessions as _get_tool_tracker_max_sessions,
)
from headroom.proxy.tool_injection_logging import (
    ToolInjectionDecision,
)
from headroom.proxy.tool_injection_logging import (
    log_tool_injection_decision as _log_tool_injection_decision,
)
from headroom.proxy.tool_injection_tracker import SessionToolTracker as _SessionToolTracker
from headroom.proxy.tool_name_policy import extract_tool_name

if TYPE_CHECKING:
    import httpx
    from fastapi import Request

logger = logging.getLogger("headroom.proxy")

_CODEX_WIRE_DEBUG_ENV = "HEADROOM_CODEX_WIRE_DEBUG"
_CODEX_WIRE_DEBUG_DIR_ENV = "HEADROOM_CODEX_WIRE_DEBUG_DIR"
_CODEX_WIRE_REDACTED = wire_debug_redaction_policy.WIRE_DEBUG_REDACTED
_CODEX_WIRE_SECRET_KEYS = wire_debug_redaction_policy.WIRE_DEBUG_SECRET_KEYS


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
    return wire_debug_redaction_policy.should_redact_key(key)


def _redact_value(value: Any) -> Any:
    return wire_debug_redaction_policy.redact_for_wire_debug(value)


def redact_for_wire_debug(value: Any) -> Any:
    """Redact obvious secrets while preserving request/response shape."""
    return wire_debug_redaction_policy.redact_for_wire_debug(value)


def _safe_event_name(event: str) -> str:
    return wire_debug_format_policy.safe_wire_debug_name(event)


def _wire_debug_preview(value: Any, *, max_chars: int | None = None) -> str:
    """Return the redacted wire payload for proxy.log.

    This is intentionally not truncated. During Codex WS debugging we need the
    proxy log itself to show the complete frame so we can decide later where a
    deliberate trim boundary belongs.
    """

    return wire_debug_format_policy.wire_debug_preview(value, max_chars=max_chars)


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
_MEMORY_INJECTION_MODE_ENV = memory_injection_mode_policy.MEMORY_INJECTION_MODE_ENV
_MEMORY_INJECTION_MODE_DEFAULT = memory_injection_mode_policy.MEMORY_INJECTION_MODE_DEFAULT
MemoryInjectionMode = memory_injection_mode_policy.MemoryInjectionMode


def get_memory_injection_mode() -> MemoryInjectionMode:
    """Return the active memory-injection routing mode.

    Read at request time so the env var can be flipped without restart for
    smoke tests. Unknown values are rejected loudly (no silent fallback).
    """
    return memory_injection_mode_policy.resolve_memory_injection_mode(
        os.environ.get(_MEMORY_INJECTION_MODE_ENV)
    )


def hash_query_for_log(query: str) -> str:
    """Stable short hash of a memory-context query, safe to log.

    Uses BLAKE2b truncated to 16 hex chars. Never logs the raw query content.
    """
    return query_log_policy.hash_query_for_log(query)


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
    tags: dict[str, str] | None = None,
) -> None:
    """Emit a structured log line for every memory-context routing decision.

    Per realignment build constraints: log every cache-affecting decision.
    Never log raw query content or Authorization header — only a stable
    hash of the query.
    """
    if tags is not None and bytes_injected > 0:
        tags["memory_injected"] = "true"
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


# Maximum request body size (100MB - increased to support image-heavy requests)
MAX_REQUEST_BODY_SIZE = 100 * 1024 * 1024

# Maximum SSE buffer size (10MB - prevents memory exhaustion from malformed streams)
MAX_SSE_BUFFER_SIZE = 10 * 1024 * 1024

# Per-event SSE size cap (PR-A8 / P1-8). Configurable via
# HEADROOM_SSE_BUFFER_MAX_BYTES. Guards against pathological huge events
# (a single event > 1 MB by default is treated as an upstream protocol bug
# and surfaces loudly rather than silently growing the buffer).
_SSE_EVENT_MAX_BYTES_ENV = request_limit_policy.SSE_EVENT_MAX_BYTES_ENV
_SSE_EVENT_MAX_BYTES_DEFAULT = request_limit_policy.SSE_EVENT_MAX_BYTES_DEFAULT


def get_sse_event_max_bytes() -> int:
    """Return the per-event SSE size cap.

    Read at request time so operators can flip the env var without a
    restart. Negative values are rejected loudly (no silent fallback).
    """
    return request_limit_policy.resolve_sse_event_max_bytes(
        os.environ.get(_SSE_EVENT_MAX_BYTES_ENV)
    )


# Well-known OpenAI-compatible upstreams, matched by host against the
# configured ``--openai-api-url``. Used only to label the dashboard/stats
# display provider — the internal provider key stays ``openai`` so pricing
# and request formatting are unaffected (issue #1533).
_OPENAI_COMPATIBLE_HOSTS: tuple[tuple[str, str], ...] = (
    ("openrouter.ai", "OpenRouter"),
    ("api.groq.com", "Groq"),
    ("api.together.xyz", "Together AI"),
    ("api.fireworks.ai", "Fireworks AI"),
    ("api.deepseek.com", "DeepSeek"),
    ("api.mistral.ai", "Mistral"),
    ("api.perplexity.ai", "Perplexity"),
    ("openai.azure.com", "Azure OpenAI"),
    ("api.openai.com", "OpenAI"),
)


def classify_openai_upstream(url: str | None) -> str | None:
    """Map a custom ``--openai-api-url`` to a well-known provider display name.

    Matches the URL host against :data:`_OPENAI_COMPATIBLE_HOSTS` (exact or
    subdomain). Returns ``None`` when no URL is set or the host is unrecognized
    (callers then fall back to an explicit ``--provider-name`` or the raw
    ``openai`` label).
    """
    if not url:
        return None
    from urllib.parse import urlparse

    try:
        host = (urlparse(url).hostname or "").lower()
    except (ValueError, TypeError):
        return None
    if not host:
        return None
    for needle, name in _OPENAI_COMPATIBLE_HOSTS:
        if host == needle or host.endswith("." + needle):
            return name
    return None


def resolve_display_provider(
    raw_provider: str | None,
    *,
    openai_api_url: str | None = None,
    provider_name: str | None = None,
) -> str:
    """Resolve the dashboard display provider for a logged request.

    Only requests whose internal provider is ``openai`` are reclassified;
    Anthropic/Bedrock/Gemini keep their own labels. This affects the display
    label only — pricing and request formatting still key on ``openai``.
    Precedence: explicit ``--provider-name`` > host detection > raw provider.
    """
    raw = (raw_provider or "").strip()
    if raw.lower() != "openai":
        return raw or "unknown"
    if provider_name:
        return provider_name
    return classify_openai_upstream(openai_api_url) or raw


# Body-too-large status code (PR-A8 / P5-59). Default 413 (RFC 7231 §6.5.11).
# Configurable via HEADROOM_PROXY_BODY_TOO_LARGE_STATUS for operators who need
# to override (no expected production use; documentation knob).
_BODY_TOO_LARGE_STATUS_ENV = request_limit_policy.BODY_TOO_LARGE_STATUS_ENV
_BODY_TOO_LARGE_STATUS_DEFAULT = request_limit_policy.BODY_TOO_LARGE_STATUS_DEFAULT


def get_body_too_large_status() -> int:
    """Return the HTTP status code for body-too-large rejections."""
    return request_limit_policy.resolve_body_too_large_status(
        os.environ.get(_BODY_TOO_LARGE_STATUS_ENV)
    )


_SSE_EVENT_TERMINATORS = sse_byte_buffer_policy.SSE_EVENT_TERMINATORS


def _find_sse_event_terminator(buf: bytearray) -> tuple[int, int] | None:
    """Return the earliest complete SSE event terminator in ``buf``."""
    return sse_byte_buffer_policy.find_sse_event_terminator(buf)


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
    return diagnostic_decode_policy.safe_decode_for_logging(raw, max_bytes=max_bytes)


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
    return sse_byte_buffer_policy.parse_sse_events_from_byte_buffer(buf)


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

# Cold-start fast-pass timeout in seconds. When background compression defers
# a cold-start-large request, the handler still runs the pipeline synchronously
# with skip_kompress=True (everything except the ML stage) under this budget so
# the FORWARDED — and therefore provider-cached and byte-identically frozen —
# form carries the cheap savings instead of the raw transcript. Without the ML
# stage the pass is bounded by routing + statistical crushers (seconds, not the
# 30s Kompress budget). Fail-open: on timeout the request forwards as before.
try:
    COLD_START_FAST_PASS_TIMEOUT_SECONDS = float(
        os.environ.get("HEADROOM_COLD_START_FAST_PASS_TIMEOUT_SECONDS", "10")
    )
except ValueError:
    COLD_START_FAST_PASS_TIMEOUT_SECONDS = 10.0

# Eager startup preload timeout in seconds. The preload (compressor/parser models,
# cache-only, allow_download=False) runs off the event loop during startup; this
# bound only fires on a true hang or an uncatchable native stall so the proxy still
# binds its port instead of never opening (GH #790). Override via
# HEADROOM_EAGER_PRELOAD_TIMEOUT_SECONDS. Falls back to 120 on an unparseable value.
try:
    EAGER_PRELOAD_TIMEOUT_SECONDS = float(
        os.environ.get("HEADROOM_EAGER_PRELOAD_TIMEOUT_SECONDS", "120")
    )
except ValueError:
    EAGER_PRELOAD_TIMEOUT_SECONDS = 120.0

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


# Transient upstream statuses worth retrying with backoff: 429 (rate limit) and
# 529 (Anthropic ``overloaded_error``). Both mean "the server is temporarily
# limiting/overloaded — try again shortly", unlike other 4xx which signal a
# problem with the request itself. Single source of truth so the streaming and
# non-streaming forwarders agree on what is retriable.
RETRYABLE_OVERLOAD_STATUSES: frozenset[int] = frozenset({429, 529})


async def request_with_transient_retry(
    client: httpx.AsyncClient,
    *,
    request_id: str | None = None,
    max_retries: int = 1,
    **request_kwargs: Any,
) -> httpx.Response:
    """Issue a buffered httpx request, retrying once on a transient close.

    ``httpx.RemoteProtocolError`` ("peer closed connection without sending
    complete message body (incomplete chunked read)") is raised when an
    upstream closes a pooled keep-alive connection that httpx then reuses for
    the next request. A direct ``curl`` never hits this because it opens a
    fresh connection per call; Headroom reuses pooled connections, so the
    first request issued on a stale connection fails even though the upstream
    is healthy (it answers a fresh connection with 200). Retrying opens a new
    connection and succeeds, mirroring curl's behaviour. See GH #1112.

    Only ``httpx.RemoteProtocolError`` is retried — the specific stale
    keep-alive symptom; every other exception (``ConnectError``, timeouts,
    HTTP status errors) propagates immediately so existing handling is
    unchanged. Use this for buffered (non-streaming) requests only: a streamed
    response cannot be safely replayed once bytes have reached the client.
    """
    import httpx

    attempt = 0
    while True:
        try:
            return await client.request(**request_kwargs)
        except httpx.RemoteProtocolError as exc:
            if attempt >= max_retries:
                raise
            attempt += 1
            logger.warning(
                "Upstream closed connection mid-response (%s); retrying on a "
                "fresh connection (attempt %d/%d)%s",
                exc,
                attempt,
                max_retries,
                f" [{request_id}]" if request_id else "",
            )


# Image compression availability (do not retain a global compressor instance)
_image_compressor_available: bool | None = None
_image_compressor_instance: Any = None


def _get_image_compressor():
    """Return the process-wide image compressor, or None if unavailable.

    The compressor caches heavyweight models; creating a new one per request
    (and a new ONNX router per image) accumulated native memory and grew RSS
    unboundedly (#2513). Reuse a single shared instance. It is marked a
    singleton so a caller's per-request ``close()`` is a no-op and the models
    stay loaded. The main-process handlers only call ``has_images()`` on it (the
    heavy compression runs in the isolation worker), but sharing still avoids a
    fresh object per request.
    """
    global _image_compressor_available, _image_compressor_instance
    if _image_compressor_available is False:
        return None
    if _image_compressor_instance is not None:
        return _image_compressor_instance

    try:
        from headroom.image import ImageCompressor

        instance = ImageCompressor()
        instance._is_singleton = True
        if _image_compressor_available is None:
            logger.info("Image compression enabled (model: chopratejas/technique-router)")
        _image_compressor_available = True
        _image_compressor_instance = instance
        return instance
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

_INTERNAL_HEADER_PREFIX = INTERNAL_HEADER_PREFIX
_STRIP_INTERNAL_HEADERS_ENV = STRIP_INTERNAL_HEADERS_ENV
_STRIP_INTERNAL_HEADERS_DEFAULT = STRIP_INTERNAL_HEADERS_DEFAULT


def get_strip_internal_headers_mode() -> StripInternalHeadersMode:
    """Return the active internal-header strip mode.

    Read at request time so operators can flip behaviour without a
    restart. Unknown values raise loudly per the no-silent-fallback
    build constraint.
    """
    return resolve_strip_internal_headers_mode(os.environ.get(_STRIP_INTERNAL_HEADERS_ENV))


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
    return strip_internal_headers(headers, mode=get_strip_internal_headers_mode())


def merge_extra_headers(headers: dict[str, str], extra: dict[str, str] | None) -> dict[str, str]:
    """Merge configured extra headers into ``headers``, overriding same-named keys.

    ``extra`` comes from ``ProxyConfig.anthropic_extra_headers``/``openai_extra_headers``
    (settings-panel/CLI-configured, for gateways that need one extra header alongside the
    client's own auth). Returns ``headers`` unchanged (no copy) when nothing is configured.
    """
    if not extra:
        return headers
    # HTTP header names are case-insensitive: drop any existing key that
    # case-insensitively collides with a configured extra so the extra wins.
    # A plain {**headers, **extra} would emit both casings upstream.
    lowered = {k.lower() for k in extra}
    merged = {k: v for k, v in headers.items() if k.lower() not in lowered}
    merged.update(extra)
    return merged


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

_BETA_HEADER_STICKY_ENV = BETA_HEADER_STICKY_ENV
_BETA_HEADER_STICKY_DEFAULT = BETA_HEADER_STICKY_DEFAULT

_BETA_TRACKER_MAX_SESSIONS_ENV = BETA_TRACKER_MAX_SESSIONS_ENV
_BETA_TRACKER_MAX_SESSIONS_DEFAULT = BETA_TRACKER_MAX_SESSIONS_DEFAULT


def get_beta_header_sticky_mode() -> BetaHeaderStickyMode:
    """Return the active beta-header stickiness mode.

    Read at request time so operators can flip behaviour without a
    restart. Unknown values raise loudly per the no-silent-fallback
    build constraint.
    """
    return resolve_beta_header_sticky_mode(os.environ.get(_BETA_HEADER_STICKY_ENV))


def get_beta_tracker_max_sessions() -> int:
    """Return the LRU bound for `SessionBetaTracker` (sessions cap)."""
    return resolve_beta_tracker_max_sessions(os.environ.get(_BETA_TRACKER_MAX_SESSIONS_ENV))


_split_beta_tokens = split_beta_tokens


_merge_beta_tokens = merge_beta_tokens


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


def get_tool_injection_sticky_mode() -> ToolInjectionStickyMode:
    """Return the active memory-tool stickiness mode.

    Read at request time so operators can flip behaviour without a
    restart. Unknown values raise loudly per the no-silent-fallback
    build constraint.
    """
    return _get_tool_injection_sticky_mode()


def get_tool_tracker_max_sessions() -> int:
    """Return the LRU bound for `SessionToolTracker` (sessions cap)."""
    return _get_tool_tracker_max_sessions()


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
    return _serialize_tool_definition_canonical(tool_definition)


class SessionToolTracker(_SessionToolTracker):
    """Env-aware compatibility wrapper for the pure session tool tracker."""

    def __init__(self, max_sessions: int | None = None) -> None:
        if max_sessions is None:
            max_sessions = get_tool_tracker_max_sessions()
        super().__init__(max_sessions=max_sessions)


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
    decision: ToolInjectionDecision,
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
    _log_tool_injection_decision(
        logger=logger,
        provider=provider,
        session_id=session_id,
        decision=decision,
        tool_definition_bytes_count=tool_definition_bytes_count,
        request_id=request_id,
    )


def _extract_tool_name(tool_definition: dict[str, Any]) -> str | None:
    """Extract a stable tool name from a memory tool definition.

    Handles three formats:
      * Anthropic custom: ``{"name": "memory_save", ...}``
      * Anthropic native: ``{"type": "memory_20250818", "name": "memory"}``
      * OpenAI function: ``{"type": "function", "function": {"name": "memory_save", ...}}``
    """
    return extract_tool_name(tool_definition)


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
            added_bytes += len(serialize_memory_tool_definition_canonical(tool_def))
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
            added_bytes += len(serialize_memory_tool_definition_canonical(tool_def))
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
                replay = replay_golden_memory_tool_definition(
                    tool_name=tool_name,
                    golden_tool_bytes=golden_bytes,
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                logger.error(
                    "corrupt golden tool bytes for session %s tool %s: %s — skipping tool injection",
                    session_id,
                    tool_name,
                    exc,
                    exc_info=True,
                )
                continue
            tools_out.append(replay.tool_definition)
            existing_names.add(replay.tool_name)
            replay_bytes += len(replay.canonical_bytes)
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
        golden_bytes = serialize_memory_tool_definition_canonical(tool_def)
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


class SessionCcrTracker(_SessionCcrTracker):
    """Env-aware compatibility wrapper for the pure CCR session tracker."""

    def __init__(self, max_sessions: int | None = None) -> None:
        if max_sessions is None:
            max_sessions = get_tool_tracker_max_sessions()
        super().__init__(max_sessions=max_sessions)


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


def has_new_ccr_markers(
    *,
    current_detected_hashes: list[str],
    previous_forwarded_messages: list[dict[str, Any]] | None,
    provider: Literal["anthropic", "openai", "google"],
) -> bool:
    """Whether the about-to-forward content carries CCR markers NOT already forwarded.

    ``overlay_cached_prefix`` (#1850) replays the previously-forwarded (compressed)
    prefix byte-identical to keep the prompt cache warm — which reintroduces the
    ``hash=…`` markers that prefix already carried. Those markers are *historical*:
    the agent saw them last turn and the retrieve-tool state was already settled
    for them. Only markers that are genuinely NEW this turn justify overriding the
    tool-injection deferral (#1006); counting the replayed ones would re-inject the
    tool on every frozen turn and bust the *tools* cache segment (undoing the very
    cache-safety the overlay provides).

    Returns True iff ``current_detected_hashes`` contains a hash that is not present
    in ``previous_forwarded_messages``.
    """
    return _has_new_ccr_markers(
        current_detected_hashes=current_detected_hashes,
        previous_forwarded_messages=previous_forwarded_messages,
        provider=provider,
    )


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
    from headroom.ccr.tool_injection import CCR_TOOL_NAME

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
        replay = create_fresh_ccr_tool_definition(provider)
        tools_out.append(replay.tool_definition)
        log_tool_injection_decision(
            provider=provider,
            session_id=None,
            decision="inject_first_time",
            tool_definition_bytes_count=len(replay.canonical_bytes),
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
                replay = replay_golden_ccr_tool_definition(golden)
                tools_out.append(replay.tool_definition)
                log_tool_injection_decision(
                    provider=provider,
                    session_id=session_id,
                    decision="inject_sticky_replay",
                    tool_definition_bytes_count=len(replay.canonical_bytes),
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
        replay = create_fresh_ccr_tool_definition(provider)
        tracker.record_ccr_done(provider, session_id, replay.canonical_bytes)
        tools_out.append(replay.tool_definition)
        log_tool_injection_decision(
            provider=provider,
            session_id=session_id,
            decision="inject_sticky_replay",
            tool_definition_bytes_count=len(replay.canonical_bytes),
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

    replay = create_fresh_ccr_tool_definition(provider)
    tracker.record_ccr_done(provider, session_id, replay.canonical_bytes)
    tools_out.append(replay.tool_definition)
    log_tool_injection_decision(
        provider=provider,
        session_id=session_id,
        decision="inject_first_time",
        tool_definition_bytes_count=len(replay.canonical_bytes),
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


# ---------------------------------------------------------------------------
# Output-only content blocks
# ---------------------------------------------------------------------------
# The Anthropic *response* schema can emit signaling blocks that the *request*
# schema (messages[].content[]) does not accept. The primary case is the
# server-side refusal fallback notification introduced with the
# ``server-side-fallback-2026-06-01`` beta::
#
#     {"type": "fallback",
#      "from": {"model": "claude-fable-5"},
#      "to":   {"model": "claude-opus-4-8"}}
#
# The API returns it inside an assistant turn to signal that a refused request
# was transparently re-served by the fallback model. When a client replays that
# assistant turn on the next call, the request validator rejects it::
#
#     400 invalid_request_error: messages.N.content.0: Input tag 'fallback'
#     found using 'type' does not match any of the expected tags
#
# These blocks are output-only and carry no state the model needs on input, so
# they are safe to drop before forwarding.
OUTPUT_ONLY_REQUEST_BLOCK_TYPES: frozenset[str] = frozenset({"fallback"})


def strip_output_only_request_blocks(messages: Any) -> bool:
    """Remove output-only content blocks from request ``messages`` in place.

    Returns ``True`` if any block was removed. If stripping empties a message's
    ``content`` list it is backfilled with a single benign text block, because
    the API also rejects an empty ``content`` array. Idempotent.
    """
    if not isinstance(messages, list):
        return False
    changed = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        kept = [
            block
            for block in content
            if not (
                isinstance(block, dict) and block.get("type") in OUTPUT_ONLY_REQUEST_BLOCK_TYPES
            )
        ]
        if len(kept) != len(content):
            changed = True
            if not kept:
                kept = [{"type": "text", "text": "(model fallback)"}]
            msg["content"] = kept
    return changed


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

    # Drop output-only blocks the request schema rejects (see
    # ``strip_output_only_request_blocks``). Callers of this bytes-less reader
    # (e.g. the Gemini path) re-serialize ``result`` themselves.
    if strip_output_only_request_blocks(result.get("messages")):
        logger.warning(
            "removed output-only content block(s) (%s) from request messages "
            "before forwarding (not valid on the request path)",
            ",".join(sorted(OUTPUT_ONLY_REQUEST_BLOCK_TYPES)),
        )

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

    # Drop output-only blocks (see ``strip_output_only_request_blocks``) before
    # any downstream deepcopy / compression / 400-retry path. This is the shared
    # reader for the Anthropic, OpenAI, and Bedrock handlers, so one guard here
    # covers every client that routes through the proxy. When a block is removed
    # we re-encode ``raw`` so a byte-faithful passthrough forwarder cannot leak
    # the pre-strip body; unchanged requests keep their exact original bytes.
    if strip_output_only_request_blocks(result.get("messages")):
        raw = json.dumps(result, ensure_ascii=False).encode("utf-8")
        logger.warning(
            "removed output-only content block(s) (%s) from request messages "
            "before forwarding (not valid on the request path)",
            ",".join(sorted(OUTPUT_ONLY_REQUEST_BLOCK_TYPES)),
        )

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


# ---------------------------------------------------------------------------
# Server-side Tool Search injection (opencode / non-Claude-Code clients).
#
# Clients that eagerly materialize every tool schema (opencode ships ~135 tool
# defs ≈ 28k tokens on EVERY request) never opt into Anthropic's Tool Search
# Tool themselves. Unlike the Claude Code case above — where the schemas are
# already in the client's own context and the proxy can't reverse it — a plain
# API client's tools live only in the request body, so the proxy CAN defer them:
# mark the non-core tools ``defer_loading: true`` and inject a tool_search tool.
# Anthropic then excludes deferred tools from the context window (they stop
# counting as input tokens until the model searches for one), while every tool
# stays callable. Deterministic output → the tools prefix still prompt-caches.
# ---------------------------------------------------------------------------

# Core coding tools kept non-deferred so routine edit/read/run loops never pay a
# search round-trip. Everything else (Slack/Linear/Sentry/Notion/Snowflake/…) is
# deferred and loaded on demand. Anthropic recommends keeping the 3–5 (here a few
# more) most frequent tools resident.
_TOOL_SEARCH_CORE_TOOLS = frozenset(
    {
        "bash",
        "bash_background",
        "bash_background_output",
        "bash_background_wait",
        "bash_background_kill",
        "read",
        "write",
        "edit",
        "multiedit",
        "apply_patch",
        "glob",
        "grep",
        "task",
        "todowrite",
        "todoread",
        "webfetch",
        "question",
        "skill",
        # A client's own tool-search/schema-fetch tool (Claude Code's ``ToolSearch``).
        # It resolves tools the client keeps in its local registry and never puts in
        # the request body (TaskCreate, WebFetch, …), so deferring it hides the only
        # tool that can load them and they become permanently unreachable.
        "toolsearch",
    }
)
_TOOL_SEARCH_DEFAULT_TYPE = "tool_search_tool_regex_20251119"
_TOOL_SEARCH_DEFAULT_NAME = "tool_search_tool_regex"
# Below this many tools the ~search round-trip isn't worth it (Anthropic's own
# guidance: standard calling is better under ~10 tools).
_TOOL_SEARCH_MIN_TOOLS = 12


def inject_tool_search_deferral(
    tools: Any,
    *,
    core_tools: frozenset[str] = _TOOL_SEARCH_CORE_TOOLS,
    search_type: str = _TOOL_SEARCH_DEFAULT_TYPE,
    search_name: str = _TOOL_SEARCH_DEFAULT_NAME,
) -> Any:
    """Return a new ``tools`` list with non-core tools deferred + a search tool
    injected, or the original list unchanged when injection doesn't apply.

    No-op when: not a list, fewer than ``_TOOL_SEARCH_MIN_TOOLS``, a tool_search
    tool is already present (client already defers), or nothing would be deferred.

    Invariants enforced (else Anthropic 400s): the search tool is never deferred;
    at least one tool stays non-deferred; a deferred tool never carries
    ``cache_control`` — if the client's tools cache breakpoint sat on a now-deferred
    tool, it is moved to the last non-deferred real tool so the (smaller) tools
    prefix still caches.
    """
    if not isinstance(tools, list) or len(tools) < _TOOL_SEARCH_MIN_TOOLS:
        return tools
    for tool in tools:
        if isinstance(tool, dict) and str(tool.get("type", "")).startswith(
            _TOOL_SEARCH_TOOL_TYPE_PREFIX
        ):
            return tools  # client already uses tool search — leave it alone

    search_tool = {"type": search_type, "name": search_name}
    out: list[Any] = [search_tool]
    deferred = 0
    dropped_cache_control = False
    dropped_marker: dict[str, Any] | None = None
    last_resident_real: dict[str, Any] | None = None
    resident_has_cache_control = False

    # Clients disagree on casing for the same tool: Claude Code sends ``Bash`` /
    # ``ToolSearch`` where opencode sends ``bash``. Compare case-insensitively so
    # the exemption applies to both — an exact match silently deferred *every*
    # tool for PascalCase clients, including their own tool-search tool.
    core_lower = {name.lower() for name in core_tools}

    for tool in tools:
        if (
            not isinstance(tool, dict)
            or tool.get("type")
            or str(tool.get("name") or "").lower() in core_lower
        ):
            # Non-dict, server/typed tools (web_search, computer, …), and core
            # tools stay resident and unchanged.
            out.append(tool)
            if isinstance(tool, dict) and not tool.get("type"):
                last_resident_real = tool
                resident_has_cache_control = resident_has_cache_control or bool(
                    tool.get("cache_control")
                )
            continue
        new_tool = dict(tool)
        new_tool["defer_loading"] = True
        _dropped = new_tool.pop("cache_control", None)
        if _dropped is not None:
            dropped_cache_control = True
            # Keep the marker itself, not just the fact of it: re-placing a bare
            # ephemeral would downgrade a 1h breakpoint to the 5m default.
            if isinstance(_dropped, dict):
                dropped_marker = _dropped
        out.append(new_tool)
        deferred += 1

    if deferred == 0:
        return tools  # nothing to defer → don't perturb the cache prefix
    # Preserve a tools cache breakpoint: if we stripped cache_control off a
    # deferred tool and no resident tool carries one, move it to the last
    # resident real tool (never the search tool, to keep its shape canonical).
    if dropped_cache_control and not resident_has_cache_control and last_resident_real is not None:
        last_resident_real["cache_control"] = (
            dict(dropped_marker) if dropped_marker else {"type": "ephemeral"}
        )
    return out


# ---------------------------------------------------------------------------
# Server-side Tool Search injection — OpenAI Responses API (gpt-5.4+).
#
# The OpenAI-side analogue of inject_tool_search_deferral above. OpenAI shipped
# the same idea for the Responses API on gpt-5.4+: mark a function/MCP tool
# ``defer_loading: true`` and add a ``{"type": "tool_search"}`` tool, and OpenAI
# keeps the deferred tools' heavy parameter schemas OUT of the model's context
# (only name+description remain) until the model searches for one — while every
# tool stays callable and the prompt cache is preserved. Same win as Anthropic
# (~15-25k tool-schema tokens -> ~200) for clients that ship a big tool surface
# and never opt into tool search themselves (opencode, plain API clients).
#
# Differences from the Anthropic path that require a separate function:
#   * Responses function tools carry ``type: "function"`` (Anthropic real tools
#     have no ``type``), so the resident/defer test is inverted — we defer
#     ``function`` (non-core) and ``mcp`` tools and keep OTHER typed/hosted tools
#     (web_search, file_search, code_interpreter, computer, image_generation, and
#     the search tool itself) resident.
#   * Model-gated: only gpt-5.4+ support it; older models 400 on the fields.
#   * No ``cache_control`` (OpenAI caches automatically), so no breakpoint move.
# ---------------------------------------------------------------------------

_OPENAI_TOOL_SEARCH_TYPE = "tool_search"
_OPENAI_TOOL_SEARCH_MIN_TOOLS = 12
_OPENAI_TOOL_SEARCH_RESIDENT_NAMES = frozenset({"terminal"})
_OPENAI_TOOL_SEARCH_UNSUPPORTED_CLIENTS = frozenset({"codex"})
# gpt-5.4 is the first model with Responses tool_search (OpenAI docs). Version-
# gated by default; overridable per deployment via a regex in
# HEADROOM_OPENAI_TOOL_SEARCH_MODELS (matched against the model name) so new
# model families can be enabled without a code edit + release.
_OPENAI_TOOL_SEARCH_MIN_VERSION = (5, 4)


def _model_supports_openai_tool_search(model: str | None) -> bool:
    """True when an OpenAI model supports the Responses ``tool_search`` feature.

    Default gate: ``gpt-<major>.<minor>`` >= 5.4. A regex in
    ``HEADROOM_OPENAI_TOOL_SEARCH_MODELS`` (matched against the model name) wins
    when set; a malformed pattern falls back to the version gate rather than
    crashing.
    """
    if not model:
        return False
    override = os.environ.get("HEADROOM_OPENAI_TOOL_SEARCH_MODELS", "").strip()
    if override:
        try:
            return re.search(override, model) is not None
        except re.error:
            pass  # malformed override → fall back to the version gate
    match = re.match(r"gpt-(\d+)(?:\.(\d+))?", model.strip().lower())
    if not match:
        return False
    major, minor = int(match.group(1)), int(match.group(2) or 0)
    return (major, minor) >= _OPENAI_TOOL_SEARCH_MIN_VERSION


def openai_tool_search_client_supported(client: str | None) -> bool:
    """Return whether OpenAI tool search deferral is safe for this client."""
    normalized = client.strip().lower() if client else ""
    return normalized not in _OPENAI_TOOL_SEARCH_UNSUPPORTED_CLIENTS


def inject_tool_search_deferral_openai(
    tools: Any,
    model: str | None,
    *,
    client: str | None = None,
    core_tools: frozenset[str] = _TOOL_SEARCH_CORE_TOOLS,
) -> Any:
    """Return a new Responses ``tools`` list with non-core function/MCP tools
    deferred + a ``{"type": "tool_search"}`` tool injected, or the original list
    unchanged when injection doesn't apply.

    No-op for Codex, whose round-trip structs drop deferred-call namespaces. Also
    no-op when: the model doesn't support tool search (gpt-5.4+ only), ``tools``
    is not a list, there are fewer than ``_OPENAI_TOOL_SEARCH_MIN_TOOLS``, a
    tool_search tool is already present (client already defers), or nothing would
    be deferred. Core coding tools and hosted/typed tools (web_search,
    file_search, code_interpreter, computer, ...) stay resident and unchanged,
    so routine edit/read/run loops never pay a search round-trip and the request
    stays valid; the injected search tool is itself resident.
    """
    if not openai_tool_search_client_supported(client):
        return tools
    if not _model_supports_openai_tool_search(model):
        return tools
    if not isinstance(tools, list) or len(tools) < _OPENAI_TOOL_SEARCH_MIN_TOOLS:
        return tools
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") == _OPENAI_TOOL_SEARCH_TYPE:
            return tools  # client already uses tool search — leave it alone

    out: list[Any] = [{"type": _OPENAI_TOOL_SEARCH_TYPE}]
    deferred = 0
    # Case-insensitive for the same reason as the Anthropic path above: the
    # resident-name sets are lowercase, clients are not required to be.
    resident_lower = {name.lower() for name in core_tools} | {
        name.lower() for name in _OPENAI_TOOL_SEARCH_RESIDENT_NAMES
    }
    for tool in tools:
        if not isinstance(tool, dict):
            out.append(tool)
            continue
        ttype = tool.get("type")
        # Deferrable: a non-core function, or an MCP server (OpenAI models are
        # trained to search namespaces / MCP servers). Everything else — core
        # coding tools and other hosted tools — stays resident.
        deferrable = (
            ttype == "function" and str(tool.get("name") or "").lower() not in resident_lower
        ) or ttype == "mcp"
        if deferrable and not tool.get("defer_loading"):
            new_tool = dict(tool)
            new_tool["defer_loading"] = True
            out.append(new_tool)
            deferred += 1
        else:
            out.append(tool)

    if deferred == 0:
        return tools  # nothing to defer → don't perturb the request / cache prefix
    return out
