"""Top-level helper functions and constants for the Headroom proxy.

Contains lazy loaders, file logging setup, request body decompression,
and safety-limit constants.

Extracted from server.py for maintainability.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from headroom import paths as _paths

if TYPE_CHECKING:
    from fastapi import Request

logger = logging.getLogger("headroom.proxy")

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


RTK_STATS_CACHE_TTL_SECONDS = 5.0
_rtk_stats_cache_lock = threading.Lock()
_rtk_stats_cache: dict[str, Any] = {
    "expires_at": 0.0,
    "has_value": False,
    "value": None,
}

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


# Sentinel used by the SSE byte-buffer helper to mark events that have no
# `event:` line. Per the SSE spec the default event name is "message"; we
# return ``None`` so callers can decide whether to apply that default.
_SSE_EVENT_TERMINATOR = b"\n\n"
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
    terminator = _SSE_EVENT_TERMINATOR
    while True:
        idx = buf.find(terminator)
        if idx == -1:
            break
        event_bytes = bytes(buf[:idx])
        # Drain the event + the trailing terminator from the buffer.
        del buf[: idx + len(terminator)]
        # Decoding the COMPLETE event must succeed. If it doesn't, the
        # upstream emitted invalid UTF-8 mid-stream — surface loudly.
        event_text = event_bytes.decode("utf-8")
        event_name: str | None = None
        data_lines: list[str] = []
        for line in event_text.split("\n"):
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

# Compression pipeline timeout in seconds
COMPRESSION_TIMEOUT_SECONDS = 30

# Maximum compression cache sessions (prevents unbounded memory growth)
MAX_COMPRESSION_CACHE_SESSIONS = 500


def jitter_delay_ms(base_ms: int, max_ms: int, attempt: int) -> float:
    """Exponential backoff with 50-150% jitter.

    Returns ``min(base_ms * 2**attempt, max_ms) * (0.5 + random())`` — the
    canonical formula used across proxy retry loops. Extracted so every
    retry site shares one implementation.
    """
    capped: float = min(base_ms * (2**attempt), max_ms)
    return capped * (0.5 + random.random())


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
        if not any(isinstance(h, RotatingFileHandler) for h in headroom_logger.handlers):
            headroom_logger.addHandler(handler)
        headroom_logger.propagate = False
    except OSError:
        # Non-fatal: can't write logs (read-only fs, permissions, etc.)
        pass


def _get_rtk_stats() -> dict[str, Any] | None:
    """Get rtk (Rust Token Killer) savings stats if rtk is installed.

    Reads from rtk's tracking database via `rtk gain --format json`.
    Results are memoized briefly so dashboard polling does not spawn a new
    subprocess on every refresh.
    """
    import shutil
    import subprocess as _sp

    now = time.monotonic()
    with _rtk_stats_cache_lock:
        if _rtk_stats_cache["has_value"] and now < float(_rtk_stats_cache["expires_at"]):
            return cast(dict[str, Any] | None, _rtk_stats_cache["value"])

    payload: dict[str, Any] | None
    rtk_bin = shutil.which("rtk")
    if not rtk_bin:
        # Check headroom-managed install. Preserve the historical Unix-name
        # behavior here (bin_dir()/"rtk") rather than switching to
        # paths.rtk_path() which would become rtk.exe on Windows.
        rtk_managed = _paths.bin_dir() / "rtk"
        if rtk_managed.exists():
            rtk_bin = str(rtk_managed)
        else:
            payload = None
            with _rtk_stats_cache_lock:
                _rtk_stats_cache.update(
                    {
                        "expires_at": time.monotonic() + RTK_STATS_CACHE_TTL_SECONDS,
                        "has_value": True,
                        "value": payload,
                    }
                )
            return payload

    try:
        result = _sp.run(
            [rtk_bin, "gain", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            summary = data.get("summary", {})
            payload = {
                "installed": True,
                "total_commands": summary.get("total_commands", 0),
                "tokens_saved": summary.get("total_saved", 0),
                "avg_savings_pct": summary.get("avg_savings_pct", 0.0),
            }
        else:
            payload = {
                "installed": True,
                "total_commands": 0,
                "tokens_saved": 0,
                "avg_savings_pct": 0.0,
            }
    except Exception:
        payload = {
            "installed": True,
            "total_commands": 0,
            "tokens_saved": 0,
            "avg_savings_pct": 0.0,
        }

    with _rtk_stats_cache_lock:
        _rtk_stats_cache.update(
            {
                "expires_at": time.monotonic() + RTK_STATS_CACHE_TTL_SECONDS,
                "has_value": True,
                "value": payload,
            }
        )
    return payload


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
                # Should never happen — golden bytes were produced by us.
                # Loud failure per build constraint #4.
                raise RuntimeError(
                    f"corrupt golden tool bytes for session {session_id} tool {tool_name}: {exc}"
                ) from exc
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
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                # Should never happen — golden bytes were produced by us.
                raise RuntimeError(
                    f"corrupt golden CCR tool bytes for session {session_id}: {exc}"
                ) from exc
            tools_out.append(tool_def)
            log_tool_injection_decision(
                provider=provider,
                session_id=session_id,
                decision="inject_sticky_replay",
                tool_definition_bytes_count=len(golden),
                request_id=request_id,
            )
            return tools_out, True
        # Tracker says "done CCR" but somehow has no golden bytes. Pin
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
