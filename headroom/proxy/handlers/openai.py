"""OpenAI handler mixin for HeadroomProxy.

Contains all OpenAI Chat Completions, Responses API, and passthrough handlers.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, unquote, urlparse

from headroom.proxy.helpers import (
    COMPRESSION_TIMEOUT_SECONDS,
    _headroom_bypass_enabled,
    extract_tags,
    jitter_delay_ms,
)
from headroom.proxy.loopback_guard import is_loopback_host
from headroom.proxy.stage_timer import StageTimer, emit_stage_timings_log
from headroom.proxy.ws_headers import WS_HOP_BY_HOP_HEADERS
from headroom.proxy.ws_session_registry import (
    TerminationCause,
    WebSocketSessionRegistry,
    WSSessionHandle,
)

if TYPE_CHECKING:
    from fastapi import Request, WebSocket
    from fastapi.responses import JSONResponse, Response, StreamingResponse

import httpx

from headroom.agent_savings import proxy_pipeline_kwargs
from headroom.config import unwrap_tool_call_name
from headroom.copilot_auth import (
    apply_copilot_api_auth,
    build_copilot_upstream_url,
    is_copilot_api_url,
)
from headroom.pipeline import PipelineStage, summarize_routing_markers
from headroom.providers.codex.responses import (
    codex_responses_http_url,
    codex_responses_websocket_url,
    has_chatgpt_account_header,
)
from headroom.providers.codex.runtime import (
    decode_openai_bearer_payload,
)
from headroom.providers.codex.runtime import (
    resolve_codex_routing_headers as _resolve_codex_routing_headers,
)
from headroom.providers.copilot import model_prefers_responses_api
from headroom.proxy.auth_mode import (
    classify_auth_mode,
    classify_client,
    should_stamp_codex_client,
)
from headroom.proxy.compression_decision import CompressionDecision
from headroom.proxy.cost import header_safe_transforms
from headroom.proxy.handlers._debug_dump import _debug_dump_mode, _redact_debug_value
from headroom.proxy.image_isolation import run_image_compression_isolated
from headroom.proxy.outcome import RequestOutcome
from headroom.proxy.passthrough import (
    custom_base_passthrough_telemetry as _custom_base_passthrough_telemetry,
)
from headroom.proxy.project_context import (
    classify_project,
    get_current_project,
    set_current_project,
)
from headroom.proxy.token_counting import gemini_output_tokens

logger = logging.getLogger("headroom.proxy")

_OPENAI_RESPONSES_UNIT_CACHE_MAX_ENTRIES = 10_000
_OPENAI_RESPONSES_UNIT_CACHE_VERSION = "openai_responses_unit_v1"
_OPENAI_RESPONSES_UNIT_PARALLELISM_ENV = "HEADROOM_TOOL_OUTPUT_COMPRESSION_PARALLELISM"
_OPENAI_RESPONSES_UNIT_PARALLELISM_DEFAULT = 4
_OPENAI_RESPONSES_UNIT_PARALLELISM_MAX = 16
_OPENAI_RESPONSES_UNIT_CACHE_INIT_LOCK = threading.RLock()
_OPENAI_RESPONSES_UNIT_EXECUTOR_LOCK = threading.RLock()
_OPENAI_RESPONSES_UNIT_EXECUTOR: ThreadPoolExecutor | None = None
_CODEX_WS_COMPRESSION_TIMEOUT_SECONDS = 5.0


def _codex_ws_compression_timeout_seconds() -> float:
    return min(COMPRESSION_TIMEOUT_SECONDS, _CODEX_WS_COMPRESSION_TIMEOUT_SECONDS)


_WS_ALLOWED_ORIGINS_ENV = "HEADROOM_WS_ORIGINS"
_CORS_ALLOWED_ORIGINS_ENV = "HEADROOM_CORS_ORIGINS"
_CODEX_RESPONSES_LITE_HEADER = "x-openai-internal-codex-responses-lite"
# Codex mirrors the responses-lite request header into the response.create
# frame body under client_metadata; upstream rejects gpt-5.x when it is
# truthy. Stripping the WS handshake header alone is insufficient
# (headroomlabs-ai/headroom#1523) — the frame-body mirror must be removed too.
_CODEX_LITE_METADATA_KEY = "ws_request_header_x_openai_internal_codex_responses_lite"


def _strip_codex_lite_metadata(raw_msg: str) -> str:
    """Remove the Codex responses-lite marker mirrored into a response.create
    frame's client_metadata. Fail-safe: returns raw_msg unchanged on any
    parse issue or when the marker is absent."""
    try:
        frame = json.loads(raw_msg)
    except (json.JSONDecodeError, TypeError, ValueError):
        return raw_msg
    if not isinstance(frame, dict):
        return raw_msg
    changed = False
    for container in (frame, frame.get("response")):
        if isinstance(container, dict):
            cm = container.get("client_metadata")
            if isinstance(cm, dict) and _CODEX_LITE_METADATA_KEY in cm:
                del cm[_CODEX_LITE_METADATA_KEY]
                changed = True
    return json.dumps(frame) if changed else raw_msg


_OPENAI_CHAT_COMPLETIONS_PATH = "/chat/completions"
_OPENAI_RESPONSES_PATH = "/responses"
_OPENAI_ORIGINAL_PATH_HEADER = "x-headroom-original-path"
_OPENAI_BASE_URL_HEADER = "x-headroom-base-url"
_decode_openai_bearer_payload = decode_openai_bearer_payload


def _normalize_openai_max_tokens(
    body: dict[str, Any], *, backend_owns_translation: bool = False
) -> None:
    """Rename the legacy ``max_tokens`` to ``max_completion_tokens`` in-place.

    This direct-OpenAI compatibility shim leaves provider-specific translation
    to backend-routed requests.

    GPT-5 / o-series chat models reject ``max_tokens`` and require
    ``max_completion_tokens``; gpt-4o/4.1 accept the latter too. So translating
    is a safe, one-way shim for current OpenAI models that lets openai-compatible
    clients (opencode, older SDKs) which still send ``max_tokens`` work unchanged.
    No-op when there is no ``max_tokens``; keeps an already-set
    ``max_completion_tokens`` and just drops the rejected legacy key.
    """
    if backend_owns_translation or not isinstance(body, dict) or "max_tokens" not in body:
        return
    legacy = body.get("max_tokens")
    if legacy is not None and body.get("max_completion_tokens") is None:
        body["max_completion_tokens"] = legacy
    body.pop("max_tokens", None)


def _apply_stream_usage_option(body: dict[str, Any]) -> None:
    """Ask the upstream for a usage chunk (for token counting) in-place, without
    overriding an explicit client ``stream_options.include_usage``.

    Forcing ``include_usage`` on a client that passed ``false`` (or that set the
    dict without the key) makes the upstream append a trailing usage-only chunk
    (``choices: []``) the client never asked for; the common
    ``chunk.choices[0].delta`` pattern then raises ``IndexError``. So only inject
    the option when the client left the choice open: no ``stream_options`` at all,
    or a ``stream_options`` dict that does not mention ``include_usage``.
    """
    stream_options = body.get("stream_options")
    if stream_options is None:
        body["stream_options"] = {"include_usage": True}
    elif isinstance(stream_options, dict) and "include_usage" not in stream_options:
        stream_options["include_usage"] = True


def _header_get(headers: dict[str, str], name: str) -> str | None:
    """Case-insensitive header lookup for plain dicts."""
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return value
    return None


def _sanitize_forwarded_response_headers(
    headers: httpx.Headers | dict[str, str],
    *extra_names: str,
) -> dict[str, str]:
    cleaned = dict(headers)
    for name in ("content-encoding", "content-length", "server", *extra_names):
        cleaned.pop(name, None)
    return cleaned


def _resolve_openai_handler_path(
    request_headers: dict[str, str],
    *,
    handler_path: str,
) -> str:
    raw_path = _header_get(request_headers, _OPENAI_ORIGINAL_PATH_HEADER)
    upstream_path = raw_path.strip() if raw_path is not None else None

    default_path = f"/v1{handler_path}"
    if upstream_path is None:
        return default_path

    if not upstream_path.startswith("/") or upstream_path.startswith("//"):
        return default_path

    parsed = urlparse(upstream_path)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        return default_path

    if not parsed.path.endswith(handler_path):
        return f"/v1{handler_path}"

    return parsed.path


def _resolve_openai_upstream_base(request_headers: dict[str, str]) -> str | None:
    raw_base_url = _header_get(request_headers, _OPENAI_BASE_URL_HEADER)
    if raw_base_url is None:
        return None

    normalized = _normalize_origin(raw_base_url)
    if normalized is None:
        return None

    # _normalize_origin drops the path; re-attach it so a custom upstream served
    # from a sub-path (e.g. https://host/api/v1) is preserved rather than routed
    # to the bare origin.
    path = (urlparse(raw_base_url.strip()).path or "").rstrip("/")
    if path:
        normalized = f"{normalized}{path}"

    if urlparse(normalized).scheme not in {"http", "https"}:
        return None
    return normalized


def _resolve_openai_chat_handler_path(base_url: str, model: str | None) -> str:
    """Return the upstream path suffix for an OpenAI chat-completions request."""

    if is_copilot_api_url(base_url) and model_prefers_responses_api(model):
        return _OPENAI_RESPONSES_PATH
    return _OPENAI_CHAT_COMPLETIONS_PATH


def _append_request_query(url: str, query: str) -> str:
    if not query:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{query}"


def _normalize_origin(origin: str) -> str | None:
    parsed = urlparse(origin.strip())
    if not parsed.scheme or not parsed.hostname:
        return None
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname.lower()
    if scheme not in {"http", "https", "ws", "wss"}:
        return None
    port = parsed.port
    default_port = (scheme in {"http", "ws"} and port == 80) or (
        scheme in {"https", "wss"} and port == 443
    )
    port_part = "" if port is None or default_port else f":{port}"
    return f"{scheme}://{hostname}{port_part}"


def _allowed_ws_origins_from_env() -> list[str] | None:
    raw = os.environ.get(_WS_ALLOWED_ORIGINS_ENV)
    if raw is None or not raw.strip():
        raw = os.environ.get(_CORS_ALLOWED_ORIGINS_ENV)
    if raw is None or not raw.strip():
        return None
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _is_loopback_ws_origin(origin: str) -> bool:
    parsed = urlparse(origin.strip())
    if parsed.scheme.lower() not in {"http", "https", "ws", "wss"}:
        return False
    if parsed.hostname is None:
        return False
    return is_loopback_host(parsed.hostname)


def _is_allowed_websocket_origin(headers: dict[str, str]) -> bool:
    """Return True when the WebSocket Origin matches the configured policy.

    Native clients commonly omit Origin, so absence is allowed. When Origin is
    present, default to loopback-only and allow explicit configured origins via
    HEADROOM_WS_ORIGINS or HEADROOM_CORS_ORIGINS.
    """
    origin = _header_get(headers, "origin")
    if not origin:
        return True

    allowed_origins = _allowed_ws_origins_from_env()
    if allowed_origins is None:
        return _is_loopback_ws_origin(origin)
    if "*" in allowed_origins:
        return True

    normalized_origin = _normalize_origin(origin)
    if normalized_origin is None:
        return False

    normalized_allowed = {
        normalized
        for allowed in allowed_origins
        for normalized in (_normalize_origin(allowed),)
        if normalized is not None
    }
    return normalized_origin in normalized_allowed


def _usage_int(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def _passthrough_usage_from_json(payload: Any) -> dict[str, int]:
    """Normalize usage from pass-through provider response shapes."""
    if not isinstance(payload, dict):
        return {}

    usage_meta = payload.get("usageMetadata")
    if isinstance(usage_meta, dict):
        return {
            "input_tokens": _usage_int(usage_meta.get("promptTokenCount")),
            "output_tokens": gemini_output_tokens(usage_meta),
            "cache_read_input_tokens": _usage_int(usage_meta.get("cachedContentTokenCount")),
        }

    usage = payload.get("usage")
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens")
        if input_tokens is None:
            input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("output_tokens")
        if output_tokens is None:
            output_tokens = usage.get("completion_tokens")
        details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
        cache_read = details.get("cached_tokens") if isinstance(details, dict) else None
        return {
            "input_tokens": _usage_int(input_tokens),
            "output_tokens": _usage_int(output_tokens),
            "cache_read_input_tokens": _usage_int(usage.get("cache_read_input_tokens", cache_read)),
            "cache_creation_input_tokens": _usage_int(usage.get("cache_creation_input_tokens")),
        }

    return {}


def _passthrough_model_from_path(path: str, endpoint_name: str) -> str:
    marker = "/models/"
    if marker in path:
        model_part = path.split(marker, 1)[1].split("/", 1)[0]
        model = model_part.split(":", 1)[0]
        if model:
            return model
    return f"passthrough:{endpoint_name}"


def _openai_responses_unit_parallelism() -> int:
    raw = os.getenv(_OPENAI_RESPONSES_UNIT_PARALLELISM_ENV)
    if raw is None or raw.strip() == "":
        return _OPENAI_RESPONSES_UNIT_PARALLELISM_DEFAULT
    try:
        requested = int(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; using default %d",
            _OPENAI_RESPONSES_UNIT_PARALLELISM_ENV,
            raw,
            _OPENAI_RESPONSES_UNIT_PARALLELISM_DEFAULT,
        )
        return _OPENAI_RESPONSES_UNIT_PARALLELISM_DEFAULT
    return max(1, min(_OPENAI_RESPONSES_UNIT_PARALLELISM_MAX, requested))


def _openai_responses_unit_executor() -> ThreadPoolExecutor:
    global _OPENAI_RESPONSES_UNIT_EXECUTOR
    with _OPENAI_RESPONSES_UNIT_EXECUTOR_LOCK:
        if _OPENAI_RESPONSES_UNIT_EXECUTOR is None:
            _OPENAI_RESPONSES_UNIT_EXECUTOR = ThreadPoolExecutor(
                max_workers=_OPENAI_RESPONSES_UNIT_PARALLELISM_MAX,
                thread_name_prefix="headroom-openai-unit",
            )
        return _OPENAI_RESPONSES_UNIT_EXECUTOR


def _openai_responses_unit_cache_key(
    unit: Any,
    *,
    model: str,
    target_ratio: float | None = None,
) -> str:
    text_hash = hashlib.sha256(unit.text.encode("utf-8", errors="replace")).hexdigest()
    key_payload = {
        "version": _OPENAI_RESPONSES_UNIT_CACHE_VERSION,
        "model": model,
        "provider": unit.provider,
        "endpoint": unit.endpoint,
        "role": unit.role,
        "item_type": unit.item_type,
        "cache_zone": unit.cache_zone,
        "mutable": unit.mutable,
        "min_bytes": unit.min_bytes,
        "context": unit.context,
        "question": unit.question,
        "bias": unit.bias,
        "metadata": unit.metadata,
        "target_ratio": target_ratio,
        "text_sha256": text_hash,
    }
    serialized = json.dumps(key_payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _openai_responses_result_with_cache_hit(result: Any) -> Any:
    router_result = getattr(result, "router_result", None)
    if router_result is None:
        return result
    return replace(result, router_result=replace(router_result, cache_hit=True))


def _codex_ws_text_shape(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "empty"
    if stripped.startswith("```"):
        return "code_fence"
    if stripped.startswith("<") and stripped.endswith(">"):
        return "xml_or_html"
    if stripped.startswith("["):
        return "json_array_like"
    if stripped.startswith("{"):
        lines = [line for line in stripped.splitlines() if line.strip()]
        if len(lines) > 1 and all(line.lstrip().startswith("{") for line in lines[:20]):
            return "jsonl_like"
        return "json_object_like"
    if stripped.startswith("Traceback (most recent call last)"):
        return "traceback"
    lines = stripped.splitlines()
    sample = lines[:50]
    if sample:
        timestamp_lines = sum(
            1
            for line in sample
            if len(line) >= 10 and line[:4].isdigit() and line[4:5] == "-" and line[7:8] == "-"
        )
        level_lines = sum(
            1
            for line in sample
            if any(level in line for level in (" ERROR ", " WARN ", " WARNING ", " INFO "))
        )
        search_lines = sum(
            1
            for line in sample
            if ":" in line and line.split(":", 2)[1:2] and line.split(":", 2)[1].isdigit()
        )
        if timestamp_lines >= max(2, len(sample) // 5) or level_lines >= max(2, len(sample) // 5):
            return "log_like"
        if search_lines >= max(2, len(sample) // 3):
            return "search_result_like"
    return "plain_text_like"


def _json_debug_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def _log_codex_compression_debug(_event: str, **_payload: Any) -> None:
    return


_CODEX_COMPRESSION_DEBUG_NOOP = _log_codex_compression_debug


def _codex_compression_debug_enabled() -> bool:
    return _log_codex_compression_debug is not _CODEX_COMPRESSION_DEBUG_NOOP


def _json_shape(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except Exception as exc:
        return {"is_json": False, "error": type(exc).__name__}
    if isinstance(parsed, dict):
        return {
            "is_json": True,
            "kind": "object",
            "keys": list(parsed.keys()),
            "length": len(parsed),
        }
    if isinstance(parsed, list):
        return {"is_json": True, "kind": "array", "length": len(parsed)}
    return {"is_json": True, "kind": type(parsed).__name__}


def _routing_log_debug(_router_result: Any) -> list[dict[str, Any]]:
    return []


_OPENAI_TOOL_SCHEMA_DROP_KEYS = {
    "$id",
    "$schema",
    "$comment",
    "deprecated",
    "examples",
    "example",
    "markdownDescription",
    "readOnly",
    "title",
    "writeOnly",
}
# Kept for backward compatibility with evals that import this symbol.
# The canonical set lives in headroom.proxy.tool_schema_compaction.


def _json_byte_len(value: Any) -> int:
    return len(_json_debug_dumps(value).encode("utf-8", errors="replace"))


def _shape_openai_responses_payload(
    payload: dict[str, Any],
    *,
    model: str,
    request_id: str,
) -> tuple[list[str], bool]:
    """Output shaping for a Responses payload (opt-in, HEADROOM_OUTPUT_SHAPER).

    The Responses counterpart of the Anthropic handler's shaping block:
    conversation-stable holdout assignment, stratum labelling for the
    output-savings ledger, then verbosity steering on the ``instructions``
    tail and ``reasoning.effort`` routing on mechanical continuations.

    Returns ``(labels, mutated)``. ``labels`` are carried on the transforms
    channel for both arms (the control arm's stratum label is what feeds the
    ledger's live baseline); ``mutated`` is True only when the payload bytes
    actually changed, so an unshaped control-arm request never forces a
    re-serialization of byte-faithful client bytes. Never raises — shaping
    must not be able to break request forwarding.
    """
    try:
        from headroom.proxy import runtime_env
        from headroom.proxy.output_savings import (
            assign_arm,
            conversation_key_from_responses_body,
            stratum_key,
            stratum_label,
        )
        from headroom.proxy.output_shaper import (
            OutputShaperSettings,
            classify_responses_turn,
            resolve_verbosity_level,
            shape_responses_request,
        )

        settings = OutputShaperSettings.from_env()
        if not settings.enabled:
            return [], False

        try:
            holdout = float(runtime_env.getenv("HEADROOM_OUTPUT_HOLDOUT", "0") or "0")
        except ValueError:
            holdout = 0.0
        arm = assign_arm(conversation_key_from_responses_body(payload), holdout)

        turn_kind = classify_responses_turn(payload.get("input")).value
        approx_input_tokens = len(json.dumps(payload)) // 4
        stratum = stratum_key(
            turn_kind=turn_kind,
            input_tokens=approx_input_tokens,
            model=model or str(payload.get("model", "")),
            has_tools=bool(payload.get("tools")),
        )
        labels = [stratum_label(arm, stratum)]

        if arm != "treatment":
            return labels, False

        level, src = resolve_verbosity_level(settings)
        shape_result = shape_responses_request(payload, settings, level_override=level)
        if shape_result.changed:
            labels.extend(shape_result.labels or [])
            logger.info(
                "[%s] OutputShaper(responses, L%s/%s): %s",
                request_id,
                level,
                src,
                shape_result.labels,
            )
        return labels, shape_result.changed
    except Exception:  # pragma: no cover - defensive; never break forwarding
        logger.warning("[%s] OutputShaper(responses) failed; skipping", request_id, exc_info=True)
        return [], False


def _compact_openai_tool_schema_value(
    value: Any,
    _parent_key: str | None = None,
) -> Any:
    # Delegate to shared compaction logic.
    from headroom.proxy.tool_schema_compaction import compact_tool_schema_value

    return compact_tool_schema_value(value, _parent_key)


def _compact_openai_responses_tools(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], bool, int, int]:
    from headroom.proxy.tool_schema_compaction import compact_tools

    return compact_tools(payload)


def _responses_request_allows_memory_tool_continuation(payload: dict[str, Any]) -> bool:
    """Return whether Responses memory tools may rely on stored continuations.

    Headroom memory tools use ``previous_response_id`` continuations after a
    tool call. Those continuations require the originating response to be
    stored. When a client explicitly sends ``store=false``, preserve that
    contract and skip the Responses memory-tool injection path instead of
    mutating the request.
    """

    return payload.get("store") is not False


def _ensure_responses_store_for_memory_tools(
    payload: dict[str, Any],
    *,
    memory_tools_injected: bool,
) -> bool:
    """Return True when memory-tool injection requires and receives store=true."""

    if memory_tools_injected and payload.get("store") is not True:
        payload["store"] = True
        return True
    return False


def _allow_responses_memory_tools(is_chatgpt_auth: bool) -> bool:
    # ChatGPT Codex rejects Responses payloads unless store=false. The
    # transparent memory-tool continuation flow needs stored responses, so keep
    # it on the regular API path only.
    return not is_chatgpt_auth


def _ensure_chatgpt_responses_store_false(
    payload: dict[str, Any],
    *,
    is_chatgpt_auth: bool,
) -> bool:
    """Return True when ChatGPT auth requires and receives a store rewrite."""

    if is_chatgpt_auth and payload.get("store") is not False:
        payload["store"] = False
        return True
    return False


def _responses_input_item_text_bytes(item: Any) -> int:
    if not isinstance(item, dict):
        return _json_byte_len(item)

    output = item.get("output")
    if isinstance(output, str):
        return len(output.encode("utf-8", errors="replace"))
    if isinstance(output, list):
        total = 0
        for part in output:
            if isinstance(part, str):
                total += len(part.encode("utf-8", errors="replace"))
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                total += len(part["text"].encode("utf-8", errors="replace"))
        if total > 0:
            return total

    content = item.get("content")
    if isinstance(content, str):
        return len(content.encode("utf-8", errors="replace"))
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, str):
                total += len(part.encode("utf-8", errors="replace"))
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                total += len(part["text"].encode("utf-8", errors="replace"))
        return total

    return _json_byte_len(item)


_RESPONSES_OUTPUT_ITEM_TYPES = frozenset(
    {
        "custom_tool_call_output",
        "function_call_output",
        "local_shell_call_output",
        "apply_patch_call_output",
    }
)


def _responses_part_text(value: Any) -> str:
    """Best-effort text from a Responses item field (string or part list)."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        texts = []
        for part in value:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
        return "\n".join(t for t in texts if t)
    return ""


def _responses_input_to_waste_messages(instructions: Any, input_data: Any) -> list[dict[str, Any]]:
    """Convert a Responses payload to OpenAI-style messages for waste parsing (#820).

    Telemetry-only — never used as a compression input. Tool output items
    become ``role="tool"`` messages so tool results (where most waste lives)
    reach ``parse_messages``; ``message`` items keep their role and joined
    part text.
    """
    messages: list[dict[str, Any]] = []
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})
    if isinstance(input_data, str):
        if input_data:
            messages.append({"role": "user", "content": input_data})
        return messages
    if not isinstance(input_data, list):
        return messages
    for item in input_data:
        if not isinstance(item, dict):
            continue
        if item.get("type") in _RESPONSES_OUTPUT_ITEM_TYPES:
            text = _responses_part_text(item.get("output"))
            if text:
                message: dict[str, Any] = {"role": "tool", "content": text}
                call_id = item.get("call_id")
                if isinstance(call_id, str) and call_id:
                    message["tool_call_id"] = call_id
                messages.append(message)
            continue
        text = _responses_part_text(item.get("content"))
        if text:
            role = item.get("role")
            messages.append(
                {"role": role if isinstance(role, str) and role else "user", "content": text}
            )
    return messages


def _responses_input_to_learner_messages(
    instructions: Any,
    input_data: Any,
) -> list[dict[str, Any]]:
    """Normalize Responses input for ``TrafficLearner``.

    The learner already understands Anthropic-style ``tool_use`` / ``tool_result``
    blocks. Converting at the provider boundary keeps that extraction logic shared
    without teaching the learner about every OpenAI transport shape.
    """
    messages: list[dict[str, Any]] = []
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})
    if isinstance(input_data, str):
        if input_data:
            messages.append({"role": "user", "content": input_data})
        return messages
    if not isinstance(input_data, list):
        return messages

    for item in input_data:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call":
            arguments = item.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    parsed_arguments = json.loads(arguments)
                except (json.JSONDecodeError, TypeError):
                    parsed_arguments = {}
                arguments = parsed_arguments if isinstance(parsed_arguments, dict) else {}
            if not isinstance(arguments, dict):
                arguments = {}
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": item.get("call_id", ""),
                            "name": item.get("name", "unknown"),
                            "input": arguments,
                        }
                    ],
                }
            )
            continue
        if item_type in _RESPONSES_OUTPUT_ITEM_TYPES:
            output = item.get("output", "")
            output_text = _responses_part_text(output)
            if not output_text and output not in (None, ""):
                output_text = json.dumps(output, ensure_ascii=False, default=str)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": item.get("call_id", ""),
                            "content": output_text,
                            "is_error": bool(item.get("is_error"))
                            or item.get("status") in {"failed", "error", "incomplete"},
                        }
                    ],
                }
            )
            continue
        text = _responses_part_text(item.get("content"))
        if text:
            role = item.get("role")
            messages.append(
                {"role": role if isinstance(role, str) and role else "unknown", "content": text}
            )
    return messages


def _has_headroom_retrieve_tool_responses(tools: Any) -> bool:
    """Return True when the Responses API tool list includes CCR retrieve.

    Responses API tool defs are flat (``{"type": "function", "name": ...}``)
    rather than nested under a "function" key like chat-completions
    tool_calls, so this can't reuse the chat-completions tool-list check.
    Mirrors ``AnthropicHandler._has_headroom_retrieve_tool``.
    """
    from headroom.ccr import CCR_TOOL_NAME

    if not isinstance(tools, list):
        return False
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("name") == CCR_TOOL_NAME:
            return True
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name") == CCR_TOOL_NAME:
            return True
    return False


def _should_buffer_openai_responses_stream_ccr(
    *,
    stream: bool,
    ccr_response_handler_enabled: bool,
    tools: Any,
    is_chatgpt_auth: bool,
) -> bool:
    """Return whether streaming Responses CCR should use buffered JSON mode."""

    return bool(
        stream
        and ccr_response_handler_enabled
        and not is_chatgpt_auth
        and _has_headroom_retrieve_tool_responses(tools)
    )


def _responses_input_to_items(input_data: Any) -> list[dict[str, Any]]:
    """Normalize a Responses ``input`` field into an item list for CCR continuation.

    ``input`` is either a plain string or an already-item-shaped list; the
    CCR continuation loop needs a list it can append output/tool-result
    items onto.
    """
    if isinstance(input_data, list):
        return list(input_data)
    if isinstance(input_data, str) and input_data:
        return [{"role": "user", "content": input_data}]
    return []


def _dedup_responses_output_items(
    items: list[dict[str, Any]],
    output_types: frozenset[str],
    count_tokens: Any = None,
    protected_call_ids: set[str] | None = None,
) -> tuple[int, int]:
    """Cross-turn verbatim de-dup over Responses tool-output items (mutates in place).

    A file re-read on the Responses path (e.g. Codex) comes back as a repeated
    ``function_call_output`` whose ``output`` string carries the same file body
    under per-call-varying ``Chunk ID`` / ``Wall time`` headers. Per-unit
    compression cannot fold that — it is inherently cross-item. This collects the
    tool-output ``.output`` strings in request order and runs the SAME
    prefix-monotonic, keep-earliest ``dedup_blocks`` the chat path uses: the
    earliest copy stays verbatim (it is the in-context reference and lives in the
    cached prefix), later duplicates fold to a one-line pointer. Longest-span
    matching folds the identical body and leaves the varying header intact.

    Returns ``(spans_folded, tokens_saved)`` — ``tokens_saved`` is 0 when no
    ``count_tokens`` callable is given. Never raises on malformed input.
    """
    try:
        from headroom.transforms.cross_turn_dedup import DedupBlock, dedup_blocks
    except Exception:
        return 0, 0

    locs: list[int] = []
    blocks: list[DedupBlock] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict) or item.get("type") not in output_types:
            continue
        out = item.get("output")
        if isinstance(out, str) and out:
            locs.append(i)
            blocks.append(
                DedupBlock(
                    text=out,
                    turn=i,
                    protected=bool(
                        isinstance(item.get("call_id"), str)
                        and protected_call_ids
                        and item.get("call_id") in protected_call_ids
                    ),
                )
            )
    if len(blocks) < 2:
        return 0, 0

    deduped, stats = dedup_blocks(blocks)
    if not stats.get("spans_folded"):
        return 0, 0

    tokens_saved = 0
    for idx, orig, new in zip(locs, blocks, deduped):
        if new.text == orig.text:
            continue
        if count_tokens is not None:
            try:
                tokens_saved += max(0, int(count_tokens(orig.text)) - int(count_tokens(new.text)))
            except Exception:
                pass
        items[idx]["output"] = new.text
    return int(stats["spans_folded"]), tokens_saved


def _openai_responses_to_sse(response: dict[str, Any]) -> list[bytes]:
    """Convert a complete Responses API JSON body into an SSE stream.

    Used only for the buffered-CCR path: the client asked for ``stream: true``
    but we forced a non-streaming upstream call so CCR retrieval could be
    resolved server-side. We then have to replay the response as SSE.

    Some Responses clients read the whole answer off the terminal
    ``response.completed`` event, but others (OpenCode / the Vercel AI SDK)
    render output only from the *incremental* item/text events and show nothing
    when they are absent (#2410). So reconstruct the real event sequence:
    ``response.created`` -> ``response.in_progress`` -> per output item
    (``response.output_item.added``, and for message items the
    ``response.content_part.added`` / ``response.output_text.delta`` /
    ``response.output_text.done`` / ``response.content_part.done`` sequence) ->
    ``response.output_item.done`` -> ``response.completed`` -> ``[DONE]``.
    """
    events: list[bytes] = []
    seq = 0

    def _emit(event_type: str, extra: dict[str, Any]) -> None:
        nonlocal seq
        payload = {"type": event_type, "sequence_number": seq, **extra}
        events.append(f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode())
        seq += 1

    output_items = response.get("output") or []
    created_response = {**response, "status": "in_progress", "output": []}
    _emit("response.created", {"response": created_response})
    _emit("response.in_progress", {"response": created_response})

    for out_idx, item in enumerate(output_items):
        if not isinstance(item, dict):
            continue
        item_id = item.get("id", f"item_{out_idx}")

        # ``output_item.added`` carries the item shell; message content streams
        # via the content-part events below, so start it empty there.
        if item.get("type") == "message":
            added_item = {k: v for k, v in item.items() if k != "content"}
            added_item["content"] = []
        else:
            added_item = item
        _emit("response.output_item.added", {"output_index": out_idx, "item": added_item})

        content = item.get("content")
        if item.get("type") == "message" and isinstance(content, list):
            for c_idx, part in enumerate(content):
                if not isinstance(part, dict):
                    continue
                loc = {"item_id": item_id, "output_index": out_idx, "content_index": c_idx}
                if part.get("type") in ("output_text", "text"):
                    text = part.get("text", "") or ""
                    _emit("response.content_part.added", {**loc, "part": {**part, "text": ""}})
                    if text:
                        _emit("response.output_text.delta", {**loc, "delta": text})
                    _emit("response.output_text.done", {**loc, "text": text})
                    _emit("response.content_part.done", {**loc, "part": part})
                else:
                    # Non-text part (e.g. refusal): add + done with the full part.
                    _emit("response.content_part.added", {**loc, "part": part})
                    _emit("response.content_part.done", {**loc, "part": part})

        _emit("response.output_item.done", {"output_index": out_idx, "item": item})

    _emit("response.completed", {"response": response})
    events.append(b"data: [DONE]\n\n")
    return events


def _output_shaping_holdout_fraction() -> float:
    from headroom.proxy import runtime_env

    try:
        return float(runtime_env.getenv("HEADROOM_OUTPUT_HOLDOUT", "0") or "0")
    except ValueError:
        return 0.0


def _shape_openai_responses_for_output(
    payload: dict[str, Any],
    *,
    input_tokens: int,
    model: str,
    conversation_key: str | None = None,
) -> Any:
    """Apply OpenAI Responses output shaping and attach holdout labels."""
    from headroom.proxy.output_savings import (
        assign_arm,
        conversation_key_from_body,
        stratum_key,
        stratum_label,
    )
    from headroom.proxy.output_shaper import (
        OutputShaperSettings,
        ShapeResult,
        classify_openai_responses_input,
        resolve_verbosity_level,
        shape_openai_responses_request,
    )

    settings = OutputShaperSettings.from_env()
    result = ShapeResult()
    if not settings.enabled:
        return result

    assert result.labels is not None
    key = conversation_key or conversation_key_from_body(payload)
    arm = assign_arm(key, _output_shaping_holdout_fraction())
    turn_kind = classify_openai_responses_input(payload.get("input")).value
    stratum = stratum_key(
        turn_kind=turn_kind,
        input_tokens=input_tokens,
        model=model or str(payload.get("model") or ""),
        has_tools=bool(payload.get("tools")),
    )
    result.labels.append(stratum_label(arm, stratum))
    if arm == "control":
        return result

    level, _source = resolve_verbosity_level(settings)
    shaped = shape_openai_responses_request(
        payload,
        settings=settings,
        level_override=level,
    )
    shaped.labels = [*result.labels, *(shaped.labels or [])]
    return shaped


def _append_unique_transforms(transforms: list[str], labels: list[str] | None) -> None:
    for label in labels or []:
        if label not in transforms:
            transforms.append(label)


def _openai_responses_payload_input_tokens(
    payload: dict[str, Any],
    token_provider: Any,
) -> int:
    try:
        tokenizer = token_provider.get_token_counter(str(payload.get("model") or ""))
        return max(0, int(tokenizer.count_text(_json_debug_dumps(payload))))
    except Exception:
        return max(0, _json_byte_len(payload) // 4)


def _openai_response_create_frame_input_tokens(
    raw_msg: str,
    token_provider: Any,
) -> int:
    try:
        parsed = json.loads(raw_msg)
    except json.JSONDecodeError:
        return 0
    if not isinstance(parsed, dict) or parsed.get("type") != "response.create":
        return 0
    payload = parsed.get("response") if isinstance(parsed.get("response"), dict) else parsed
    if not isinstance(payload, dict):
        return 0
    return _openai_responses_payload_input_tokens(payload, token_provider)


def _shape_openai_response_create_frame(
    raw_msg: str,
    *,
    input_tokens: int,
    conversation_key: str | None = None,
) -> tuple[str, bool, list[str], str | None]:
    try:
        parsed = json.loads(raw_msg)
    except json.JSONDecodeError:
        return raw_msg, False, [], "non_json"
    if not isinstance(parsed, dict) or parsed.get("type") != "response.create":
        return raw_msg, False, [], "not_response_create"

    wrapped = isinstance(parsed.get("response"), dict)
    payload = parsed["response"] if wrapped else parsed
    if not isinstance(payload, dict):
        return raw_msg, False, [], "invalid_inner_payload"

    result = _shape_openai_responses_for_output(
        payload,
        input_tokens=input_tokens,
        model=str(payload.get("model") or ""),
        conversation_key=conversation_key,
    )
    labels = list(result.labels or [])
    if not result.changed:
        return raw_msg, False, labels, None

    if wrapped:
        parsed["response"] = payload
        return json.dumps(parsed), True, labels, None
    return json.dumps(payload), True, labels, None


def _openai_responses_context_budget(payload: dict[str, Any]) -> dict[str, Any]:
    payload_bytes = _json_byte_len(payload)
    buckets: dict[str, int] = {}
    for key in ("instructions", "tools", "input", "messages", "client_metadata"):
        if key in payload:
            buckets[key] = _json_byte_len(payload.get(key))

    other_bytes = max(payload_bytes - sum(buckets.values()), 0)
    if other_bytes:
        buckets["other"] = other_bytes

    input_breakdown: dict[str, dict[str, int]] = {}
    items = payload.get("input") or payload.get("messages")
    if isinstance(items, list):
        for item in items:
            item_type = item.get("type", "unknown") if isinstance(item, dict) else "non_dict"
            row = input_breakdown.setdefault(
                str(item_type),
                {"items": 0, "bytes": 0, "text_bytes": 0},
            )
            row["items"] += 1
            row["bytes"] += _json_byte_len(item)
            row["text_bytes"] += _responses_input_item_text_bytes(item)

    return {
        "payload_bytes": payload_bytes,
        "buckets": {
            key: {
                "bytes": value,
                "pct": (value / payload_bytes * 100.0) if payload_bytes else 0.0,
            }
            for key, value in sorted(
                buckets.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        },
        "input_breakdown": input_breakdown,
    }


# Interactive Responses turns are latency-sensitive. Fail open quickly rather
# than holding the session hostage on memory lookup.
RESPONSES_CONTEXT_SEARCH_TIMEOUT_SECONDS = 2.0

# Cap the wait for the first client frame after the WS handshake completes.
# A zombie or malicious client that accepts the upgrade but never sends the
# first response.create frame would otherwise hold a slot indefinitely and
# starve the session registry. 60 s is generous for real clients (Codex
# typically sends the first frame within a few hundred milliseconds of the
# accept) but short enough to bound the damage from a hung peer.
WS_FIRST_FRAME_TIMEOUT_SECONDS = 60.0

# Accepted values for ``config.mode`` on POST /v1/compress. Unset/None means
# the default marker-free pipeline. Anything else is a 400 rather than a
# silent fall-through to the default (a typo or an unsupported mode like
# "lossless" would otherwise look like it worked).
COMPRESS_MODES = ("ccr", "lossy_inline", "lossless_then_lossy")


def _extract_codex_handshake_headers(upstream: Any) -> list[tuple[str, str]]:
    """Return the ``x-codex-*`` headers from an upstream WS handshake response.

    OpenAI delivers the Codex subscription/rate-limit window only on the
    WebSocket handshake response headers (not in data frames). We forward
    that subset onto the client-facing 101 so Codex, ``/stats``, and the
    headroom-desktop gauge can all read the live window. Filtered strictly
    to ``x-codex-*`` -- never ``set-cookie``/``authorization``/etc.
    """
    resp = getattr(upstream, "response", None)
    headers = getattr(resp, "headers", None)
    if headers is None:
        return []
    raw_items = getattr(headers, "raw_items", None)
    try:
        items = list(raw_items()) if callable(raw_items) else list(headers.items())
    except Exception:
        return []
    out: list[tuple[str, str]] = []
    for name, value in items:
        name_str = name.decode("latin-1") if isinstance(name, bytes | bytearray) else str(name)
        if name_str.lower().startswith("x-codex-"):
            value_str = (
                value.decode("latin-1") if isinstance(value, bytes | bytearray) else str(value)
            )
            out.append((name_str, value_str))
    return out


def _infer_openai_cache_write_tokens(input_tokens: int, cache_read_tokens: int) -> int:
    """Infer OpenAI automatic prompt-cache writes from uncached input tokens.

    OpenAI reports prompt-cache reads as ``cached_tokens`` but does not expose a
    separate write counter. For dashboard observability, the uncached portion of
    a Codex/OpenAI request is the best available write-volume proxy. OpenAI has
    no write premium in our cache economics, so this affects cache-write
    counters, not dollar savings.
    """

    return max(input_tokens - cache_read_tokens, 0)


def _deferrable_savings_delta(input_delta: int, saved_delta: int) -> int:
    """Gate a WS Responses turn's compression-savings delta on input accounting.

    ``tokens_saved`` accumulates at compression time (our own token count),
    while input tokens only arrive with a usage frame on
    ``response.completed``. A turn that was compressed but never completed
    (cancelled mid-response, upstream error) therefore shows
    ``saved_delta > 0`` with ``input_delta == 0`` — and recording that pair
    writes a savings-with-zero-spend checkpoint into the savings tracker,
    desyncing every downstream funnel (dashboards flag "savings but no
    spend"). Return 0 for that case so the caller defers the savings until a
    usage-carrying turn; non-positive deltas pass through unchanged so the
    recorded-total bookkeeping keeps its normal resync behaviour.
    """

    if saved_delta > 0 and input_delta <= 0:
        return 0
    return saved_delta


def _extract_responses_usage(event: dict[str, Any]) -> tuple[int, int, int, int, int]:
    """Return input/output/cache usage from a Responses event.

    Codex WebSocket streams include usage on ``response.completed`` events.
    The shape mirrors HTTP Responses usage:
    ``response.usage.input_tokens`` plus
    ``response.usage.input_tokens_details.cached_tokens``.
    """

    if event.get("type") != "response.completed":
        return 0, 0, 0, 0, 0

    response = event.get("response")
    if not isinstance(response, dict):
        response = {}
    usage = response.get("usage") or event.get("usage")
    if not isinstance(usage, dict):
        return 0, 0, 0, 0, 0

    def _int(value: Any) -> int:
        try:
            return max(int(value), 0)
        except (TypeError, ValueError):
            return 0

    input_tokens = _int(usage.get("input_tokens"))
    output_tokens = _int(usage.get("output_tokens"))
    details = usage.get("input_tokens_details")
    cached_tokens = _int(details.get("cached_tokens")) if isinstance(details, dict) else 0
    cache_write_tokens = _infer_openai_cache_write_tokens(input_tokens, cached_tokens)
    uncached_tokens = max(input_tokens - cached_tokens, 0)
    return input_tokens, output_tokens, cached_tokens, cache_write_tokens, uncached_tokens


def _prefers_http1_passthrough(base_url: str) -> bool:
    """Whether passthrough to this host must use HTTP/1.1.

    ChatGPT's Cloudflare edge issues a managed challenge to our HTTP/2
    fingerprint on sensitive account endpoints; HTTP/1.1 is accepted.
    """
    host = (urlparse(base_url).hostname or "").lower()
    return host == "chatgpt.com" or host.endswith(".chatgpt.com")


class OpenAIHandlerMixin:
    """Mixin providing OpenAI API handler methods for HeadroomProxy."""

    async def _count_tokens_offloaded(self, model, messages):  # noqa: ANN001, ANN201
        from headroom.proxy.token_counting import count_tokens_offloaded

        return await count_tokens_offloaded(self, model, messages)

    OPENAI_RESPONSES_ROUTER_MIN_BYTES = 512
    OPENAI_RESPONSES_OUTPUT_TYPES = _RESPONSES_OUTPUT_ITEM_TYPES

    def _openai_responses_unit_cache(self) -> tuple[Any, OrderedDict[str, Any]]:
        with _OPENAI_RESPONSES_UNIT_CACHE_INIT_LOCK:
            lock = getattr(self, "_openai_responses_unit_cache_lock", None)
            if lock is None:
                lock = threading.RLock()
                self._openai_responses_unit_cache_lock = lock
            cache = getattr(self, "_openai_responses_unit_result_cache", None)
            if cache is None:
                cache = OrderedDict()
                self._openai_responses_unit_result_cache = cache
            return lock, cache

    def _get_openai_responses_cached_unit(self, key: str) -> Any | None:
        lock, cache = self._openai_responses_unit_cache()
        with lock:
            result = cache.get(key)
            if result is None:
                return None
            cache.move_to_end(key)
        return _openai_responses_result_with_cache_hit(result)

    def _store_openai_responses_cached_unit(self, key: str, result: Any) -> None:
        lock, cache = self._openai_responses_unit_cache()
        with lock:
            cache[key] = result
            cache.move_to_end(key)
            while len(cache) > _OPENAI_RESPONSES_UNIT_CACHE_MAX_ENTRIES:
                cache.popitem(last=False)

    async def _observe_openai_responses_traffic(
        self,
        body: dict[str, Any],
        *,
        request_id: str,
    ) -> None:
        """Feed one Responses HTTP request into the live traffic learner."""
        traffic_learner = getattr(self, "traffic_learner", None)
        if traffic_learner is None:
            return
        try:
            memory_handler = getattr(self, "memory_handler", None)
            if (
                traffic_learner._backend is None
                and memory_handler
                and memory_handler.initialized
                and memory_handler.backend
            ):
                traffic_learner.set_backend(memory_handler.backend)

            learner_messages = _responses_input_to_learner_messages(
                body.get("instructions"),
                body.get("input", ""),
            )
            tool_results = traffic_learner.extract_tool_results_from_messages(learner_messages)
            for tool_result in tool_results[-5:]:
                await traffic_learner.on_tool_result(
                    tool_name=tool_result["tool_name"],
                    tool_input=tool_result["input"],
                    tool_output=tool_result["output"],
                    is_error=tool_result["is_error"],
                )
            await traffic_learner.on_messages(learner_messages)
        except Exception as exc:
            logger.debug("[%s] Traffic learner (responses): %s", request_id, exc)

    async def _observe_openai_chat_traffic(
        self,
        messages: list[dict[str, Any]],
        *,
        request_id: str,
    ) -> None:
        """Feed one chat/completions request into the live traffic learner.

        The chat counterpart of :meth:`_observe_openai_responses_traffic`.
        Chat/completions clients (GitHub Copilot CLI, opencode, OpenAI SDKs)
        route here rather than through ``/v1/responses``, so without this call
        their tool results and user preferences never reached the learner even
        with Learn enabled (part of #2060). Chat messages are already
        ``role``/``content`` shaped, so ``on_messages`` consumes them directly;
        tool results use the OpenAI-format extractor.
        """
        traffic_learner = getattr(self, "traffic_learner", None)
        if traffic_learner is None:
            return
        try:
            memory_handler = getattr(self, "memory_handler", None)
            if (
                traffic_learner._backend is None
                and memory_handler
                and memory_handler.initialized
                and memory_handler.backend
            ):
                traffic_learner.set_backend(memory_handler.backend)

            tool_results = traffic_learner.extract_tool_results_from_openai_messages(messages)
            for tool_result in tool_results[-5:]:
                await traffic_learner.on_tool_result(
                    tool_name=tool_result["tool_name"],
                    tool_input=tool_result["input"],
                    tool_output=tool_result["output"],
                    is_error=tool_result["is_error"],
                )
            await traffic_learner.on_messages(messages)
        except Exception as exc:
            logger.debug("[%s] Traffic learner (chat): %s", request_id, exc)

    @staticmethod
    def _headroom_bypass_enabled(headers: Any) -> bool:
        """Return True when inbound headers request full passthrough."""
        return _headroom_bypass_enabled(headers)

    def _resolve_openai_upstream(self, request: Request) -> str:
        """Return the OpenAI upstream base URL for ``request``.

        Honors the ``x-headroom-base-url`` request header so OpenAI-compatible
        gateways (LiteLLM, CPA, self-hosted vLLM, Azure OpenAI) route through
        the dedicated ``/v1/chat/completions`` and ``/v1/responses`` handlers,
        not just the generic passthrough route that already honors it. Falls
        back to the configured ``OPENAI_API_URL`` (``OPENAI_TARGET_API_URL``).
        """
        return _resolve_openai_upstream_base(request.headers) or self.OPENAI_API_URL

    @staticmethod
    def _strict_previous_turn_frozen_count(
        messages: list[dict[str, Any]],
        base_frozen_count: int,
    ) -> int:
        """Freeze all prior turns in cache mode; only the final OBSERVATION turn
        is mutable (the newest delta we compress-once-then-freeze).

        The newest observation is the compressible delta. Its role depends on the
        harness: text/back-tick harnesses (mini-swe-agent, Codex) append it as
        ``role:"user"``, but OpenAI function-calling harnesses (Kimi / any
        fireworks/OpenAI-compatible tool-based model) append it as ``role:"tool"``
        (and legacy function-calling as ``role:"function"``). Gating solely on
        ``role == "user"`` froze the ENTIRE conversation on every OpenAI
        tool-based turn — so NOTHING was ever compressed on those models (the
        delta was frozen before the content router saw it). Treat tool/function
        observations as the mutable tail too; assistant/system endings still
        freeze everything (they are not observations).
        """
        if not messages:
            return base_frozen_count
        final_idx = len(messages) - 1
        if messages[final_idx].get("role") in ("user", "tool", "function"):
            return final_idx
        return len(messages)

    @staticmethod
    def _restore_frozen_prefix(
        original_messages: list[dict[str, Any]],
        candidate_messages: list[dict[str, Any]],
        *,
        frozen_message_count: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Force frozen prefix bytes to match original request exactly."""
        if frozen_message_count <= 0 or not original_messages:
            return candidate_messages, 0

        frozen = min(frozen_message_count, len(original_messages))
        restored = list(candidate_messages)

        if len(restored) < frozen:
            return list(original_messages[:frozen]) + restored, frozen

        changed = 0
        for idx in range(frozen):
            if restored[idx] != original_messages[idx]:
                restored[idx] = original_messages[idx]
                changed += 1
        return restored, changed

    def _compress_openai_responses_live_text_units_with_router(
        self,
        payload: dict[str, Any],
        *,
        model: str,
        request_id: str,
        pass_id: str | None = None,
        timing: dict[str, float] | None = None,
    ) -> tuple[dict[str, Any], bool, int, list[str], dict[str, int], list[str], int]:
        """Run ContentRouter on OpenAI Responses text units.

        This is the Responses provider scaffold: it extracts text-bearing
        request slots into provider-neutral ``CompressionUnit`` objects, lets
        the shared router enforce role/type policy and choose compressors, then
        splices accepted replacements back into the Responses payload. Opaque
        items such as reasoning, compaction, tool calls, and non-string outputs
        are intentionally not exposed as text units.
        """

        debug_enabled = _codex_compression_debug_enabled()

        def _log(_event: str, **_fields: Any) -> None:
            if debug_enabled:
                _log_codex_compression_debug(
                    _event,
                    request_id=request_id,
                    pass_id=pass_id,
                    model=model,
                    **_fields,
                )

        input_items = payload.get("input")
        messages_items = payload.get("messages")
        items = input_items if isinstance(input_items, list) else messages_items
        if not isinstance(items, list):
            return payload, False, 0, [], {}, [], 0
        try:
            from headroom.transforms.compression_batches import (
                CompressionBatchEntry,
                build_compression_batches,
                compress_batch_with_router,
            )
            from headroom.transforms.compression_units import (
                CompressionUnit,
                RoutedCompressionUnit,
                compress_unit_with_router,
                find_content_router,
            )
        except Exception as exc:
            logger.debug(
                "[%s] CompressionUnit adapter unavailable: %s",
                request_id,
                exc,
            )
            return payload, False, 0, [], {}, [], 0

        router = find_content_router(self.openai_pipeline)
        if router is None:
            logger.debug("[%s] OpenAI Responses ContentRouter unavailable", request_id)
            return payload, False, 0, [], {}, [], 0
        profile_kwargs = proxy_pipeline_kwargs(getattr(self, "config", None))
        unit_target_ratio = profile_kwargs.get("target_ratio")
        if unit_target_ratio is not None:
            unit_target_ratio = float(unit_target_ratio)

        try:
            tokenizer = self.openai_provider.get_token_counter(model)
        except Exception as exc:
            logger.debug(
                "[%s] OpenAI Responses ContentRouter tokenizer unavailable: %s",
                request_id,
                exc,
            )
            return payload, False, 0, [], {}, [], 0

        def _slot_texts(item: dict[str, Any]) -> list[tuple[str, tuple[str, int | None]]]:
            # Only tool-output items are eligible for in-place compression.
            # Message items (user/system/assistant) sit inside the request's
            # cacheable prefix; mutating them busts prefix caching on every
            # subsequent turn. Role-level guards in compression_units.py
            # remain as defense-in-depth.
            type_tag = item.get("type")
            if type_tag not in self.OPENAI_RESPONSES_OUTPUT_TYPES:
                return []
            output = item.get("output")
            if isinstance(output, str):
                return [(output, ("output", None))]
            if not isinstance(output, list):
                return []
            return [
                (part["text"], ("output_part", index))
                for index, part in enumerate(output)
                if isinstance(part, dict)
                and part.get("type") in {"input_text", "output_text"}
                and isinstance(part.get("text"), str)
            ]

        def _set_slot_text(
            item: dict[str, Any],
            slot: tuple[str, int | None],
            replacement: str,
        ) -> bool:
            kind, index = slot
            if kind == "output":
                item["output"] = replacement
                return True
            if kind == "output_part" and isinstance(index, int):
                output = item.get("output")
                if isinstance(output, list) and 0 <= index < len(output):
                    part = output[index]
                    if isinstance(part, dict) and part.get("type") in {"input_text", "output_text"}:
                        part["text"] = replacement
                        return True
            return False

        headroom_retrieve_call_ids: set[str] = set()
        # Map each Responses tool call to its name so that outputs belonging to
        # excluded tools (HEADROOM_EXCLUDE_TOOLS) can be protected from
        # compression. The chat/Anthropic paths get this via
        # ContentRouter._build_tool_name_map; the Responses payload carries the
        # name on the `function_call` item and the originating call_id on the
        # matching `function_call_output`, so we correlate them here.
        function_name_by_call_id: dict[str, str] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "function_call":
                continue
            name = item.get("name")
            call_id = item.get("call_id")
            if name:
                # Hermes deferred tools arrive wrapped as `tool_call` with
                # the real name inside the arguments/input payload.
                name = unwrap_tool_call_name(name, item.get("arguments") or item.get("input"))
            if isinstance(name, str) and isinstance(call_id, str) and call_id:
                function_name_by_call_id[call_id] = name
            if isinstance(name, str) and (
                name == "headroom_retrieve" or name.endswith("__headroom_retrieve")
            ):
                if isinstance(call_id, str) and call_id:
                    headroom_retrieve_call_ids.add(call_id)

        # Resolve the effective exclude set once (None -> built-in defaults),
        # mirroring ContentRouter's policy. exclude_tools already contains both
        # original and lowercased name variants (see _parse_exclude_tools), but
        # we also test the lowercased name defensively for case-insensitivity.
        from headroom.config import (
            DEFAULT_EXCLUDE_TOOLS,
            DEFAULT_VERBATIM_EXCLUDE_TOOLS,
            is_tool_excluded,
        )

        router_exclude_tools = getattr(router.config, "exclude_tools", None)
        effective_exclude_tools = (
            router_exclude_tools if router_exclude_tools is not None else DEFAULT_EXCLUDE_TOOLS
        )
        excluded_call_ids: set[str] = {
            call_id
            for call_id, fn_name in function_name_by_call_id.items()
            if is_tool_excluded(fn_name, effective_exclude_tools)
        }
        verbatim_excluded_call_ids: set[str] = {
            call_id
            for call_id, fn_name in function_name_by_call_id.items()
            if is_tool_excluded(fn_name, DEFAULT_VERBATIM_EXCLUDE_TOOLS)
        }

        timing_sink: dict[str, float] = timing if timing is not None else {}

        def _add_timing(name: str, started_at: float) -> None:
            timing_sink[name] = (
                timing_sink.get(name, 0.0) + (time.perf_counter() - started_at) * 1000.0
            )

        extraction_started = time.perf_counter()
        candidates: list[tuple[int, tuple[str, int | None], str]] = []
        # Excluded-tool outputs that are losslessly foldable (grep/log/json):
        # (item_index, slot_ref, folded_text, original_text). Spliced after the
        # normal candidate compression — no ML, byte/data-lossless only.
        lossless_excluded: list[tuple[int, tuple[str, int | None], str, str]] = []
        extraction_debug: list[dict[str, Any]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                if debug_enabled:
                    extraction_debug.append(
                        {
                            "index": idx,
                            "eligible": False,
                            "reason": "item_not_dict",
                            "item_type": type(item).__name__,
                            "item": item,
                        }
                    )
                continue
            item_type = item.get("type")
            if item_type in self.OPENAI_RESPONSES_OUTPUT_TYPES:
                call_id = item.get("call_id")
                if isinstance(call_id, str) and call_id in headroom_retrieve_call_ids:
                    if debug_enabled:
                        extraction_debug.append(
                            {
                                "index": idx,
                                "eligible": False,
                                "reason": "headroom_retrieve_output_protected",
                                "item_type": item_type,
                                "call_id": call_id,
                                "item": item,
                            }
                        )
                    continue
                if isinstance(call_id, str) and call_id in excluded_call_ids:
                    if call_id in verbatim_excluded_call_ids:
                        if debug_enabled:
                            extraction_debug.append(
                                {
                                    "index": idx,
                                    "eligible": False,
                                    "reason": "exclude_tools_verbatim",
                                    "item_type": item_type,
                                    "call_id": call_id,
                                    "tool_name": function_name_by_call_id.get(call_id),
                                    "item": item,
                                }
                            )
                        continue
                    # Protected from lossy compression — but grep/log/json output
                    # can still be losslessly compacted. Reuse the router helper
                    # so the Responses path matches the chat/Anthropic behavior.
                    # Note: when output is a content-part array, fold each text part
                    # individually using ("output_part", index) slots to preserve the
                    # array structure (non-text parts like images are left untouched).
                    raw_output = item.get("output")
                    if isinstance(raw_output, list):
                        for pidx, part in enumerate(raw_output):
                            if (
                                isinstance(part, dict)
                                and part.get("type") in {"input_text", "output_text"}
                                and isinstance(part.get("text"), str)
                            ):
                                part_text = part["text"]
                                pf = router._lossless_compact_excluded(part_text)
                                if pf is not None:
                                    lossless_excluded.append(
                                        (idx, ("output_part", pidx), pf[0], part_text)
                                    )
                    else:
                        excl_out = _responses_part_text(raw_output)
                        fold = router._lossless_compact_excluded(excl_out) if excl_out else None
                        if fold is not None:
                            lossless_excluded.append((idx, ("output", None), fold[0], excl_out))
                    if debug_enabled:
                        extraction_debug.append(
                            {
                                "index": idx,
                                "eligible": False,
                                "reason": (
                                    "exclude_tools_lossless_fold"
                                    if fold is not None
                                    else "exclude_tools_protected"
                                ),
                                "item_type": item_type,
                                "call_id": call_id,
                                "tool_name": function_name_by_call_id.get(call_id),
                                "item": item,
                            }
                        )
                    continue
                slots = _slot_texts(item)
                if slots:
                    for text, slot_ref in slots:
                        candidates.append((idx, slot_ref, text))
                        if debug_enabled:
                            extraction_debug.append(
                                {
                                    "index": idx,
                                    "eligible": True,
                                    "item_type": item_type,
                                    "role": item.get("role"),
                                    "slot": slot_ref,
                                    "text_chars": len(text),
                                    "text_bytes": len(text.encode("utf-8", errors="replace")),
                                    "text_json_shape": _json_shape(text),
                                    "item": item,
                                    "text": text,
                                }
                            )
                else:
                    if debug_enabled:
                        extraction_debug.append(
                            {
                                "index": idx,
                                "eligible": False,
                                "reason": "output_type_without_text_slot",
                                "item_type": item_type,
                                "item": item,
                            }
                        )
            else:
                if debug_enabled:
                    extraction_debug.append(
                        {
                            "index": idx,
                            "eligible": False,
                            "reason": "unsupported_item_type",
                            "item_type": item_type,
                            "role": item.get("role"),
                            "item": item,
                        }
                    )

        _add_timing("compression_live_unit_extraction", extraction_started)
        _log(
            "codex_compression_extraction",
            item_count=len(items),
            candidate_count=len(candidates),
            payload=payload,
            extraction=extraction_debug,
        )
        if not candidates and not lossless_excluded:
            _log(
                "codex_compression_payload_result",
                modified=False,
                reason="no_candidates",
                tokens_saved_total=0,
                transforms=[],
                input_payload=payload,
                output_payload=payload,
            )
            return payload, False, 0, [], {}, [], 0

        deepcopy_started = time.perf_counter()
        updated = copy.deepcopy(payload)
        _add_timing("compression_payload_deepcopy", deepcopy_started)
        updated_input_items = updated.get("input")
        updated_messages_items = updated.get("messages")
        updated_items = (
            updated_input_items if isinstance(updated_input_items, list) else updated_messages_items
        )
        if not isinstance(updated_items, list):
            return payload, False, 0, [], {}, [], 0

        modified = False
        tokens_saved_total = 0
        # `attempted_input_tokens` is the *compressible* portion of the
        # request — only the tokens we actually fed to the router (i.e.
        # extracted units that passed the floor + role + cache_zone
        # gates). It excludes user messages, system prompts, prior-turn
        # assistant content, and other frozen prefix bytes. This is the
        # right denominator for the dashboard savings ratio: comparing
        # tokens_saved against tokens we ATTEMPTED to compress, not
        # against everything in the request.
        attempted_input_tokens = 0
        transforms: list[str] = []
        routed_units: list[RoutedCompressionUnit] = []

        unit_build_started = time.perf_counter()
        unit_debug: list[dict[str, Any]] = []
        for item_idx, slot_ref, original_text in candidates:
            item = items[item_idx] if item_idx < len(items) else {}
            item_type = item.get("type", "unknown") if isinstance(item, dict) else "unknown"
            role = str(item.get("role") or "tool") if isinstance(item, dict) else "tool"
            unit = CompressionUnit(
                text=original_text,
                provider="openai",
                endpoint="responses",
                role=role,
                item_type=str(item_type),
                cache_zone="live",
                mutable=True,
                min_bytes=self.OPENAI_RESPONSES_ROUTER_MIN_BYTES,
            )
            routed_units.append(RoutedCompressionUnit(unit=unit, slot=(item_idx, slot_ref)))
            if debug_enabled:
                unit_debug.append(
                    {
                        "item_index": item_idx,
                        "slot": slot_ref,
                        "provider": unit.provider,
                        "endpoint": unit.endpoint,
                        "role": unit.role,
                        "item_type": unit.item_type,
                        "cache_zone": unit.cache_zone,
                        "mutable": unit.mutable,
                        "min_bytes": unit.min_bytes,
                        "text_chars": len(unit.text),
                        "text_bytes": len(unit.text.encode("utf-8", errors="replace")),
                        "text_json_shape": _json_shape(unit.text),
                        "text": unit.text,
                    }
                )
        _add_timing("compression_unit_build", unit_build_started)

        _log(
            "codex_compression_units",
            units=unit_debug,
        )

        # Tally per-category counts as units stream in so the pass_summary
        # event below can emit a one-line breakdown — log readers shouldn't
        # have to re-aggregate from scattered unit_result events.
        units_by_category: dict[str, int] = {}
        strategy_chain_union: list[str] = []

        def _compress_routed_unit(
            routed: RoutedCompressionUnit,
        ) -> tuple[object, Any, float]:
            # `elapsed_ms` is pure compute time. Prior to the P2 scheduler
            # fix this was wall-clock-from-submit, which conflated
            # semaphore wait with real work — passthrough units showed
            # `elapsed_ms=60000+` in production logs even though they did
            # no work. With the semaphore deleted, this timer is honest.
            unit_started = time.perf_counter()
            result = compress_unit_with_router(
                routed.unit,
                router=router,
                tokenizer=tokenizer,
                target_ratio=unit_target_ratio,
            )
            elapsed_ms = (time.perf_counter() - unit_started) * 1000.0
            return routed.slot, result, elapsed_ms

        router_total_started = time.perf_counter()
        routed_results: list[tuple[object, Any, float] | None] = [None] * len(routed_units)
        unit_index_by_slot = {routed.slot: unit_idx for unit_idx, routed in enumerate(routed_units)}
        small_batch_entries: list[CompressionBatchEntry] = []
        large_unit_indexes: list[int] = []
        for unit_idx, routed in enumerate(routed_units):
            text_bytes = len(routed.unit.text.encode("utf-8", errors="replace"))
            if text_bytes < routed.unit.min_bytes:
                small_batch_entries.append(
                    CompressionBatchEntry(entry_id=f"u{unit_idx}", routed=routed)
                )
            else:
                large_unit_indexes.append(unit_idx)
        small_batches, small_batch_skipped = build_compression_batches(
            small_batch_entries,
            min_batch_bytes=self.OPENAI_RESPONSES_ROUTER_MIN_BYTES,
        )
        cache_misses: list[tuple[int, str, RoutedCompressionUnit]] = []
        cache_miss_followers: dict[str, list[int]] = {}
        for unit_idx in large_unit_indexes:
            routed = routed_units[unit_idx]
            cache_key = _openai_responses_unit_cache_key(
                routed.unit,
                model=model,
                target_ratio=unit_target_ratio,
            )
            cached = self._get_openai_responses_cached_unit(cache_key)
            if cached is not None:
                routed_results[unit_idx] = (routed.slot, cached, 0.0)
                continue
            if cache_key in cache_miss_followers:
                cache_miss_followers[cache_key].append(unit_idx)
                continue
            cache_miss_followers[cache_key] = []
            cache_misses.append((unit_idx, cache_key, routed))

        def _compress_and_store(
            unit_idx: int,
            cache_key: str,
            routed: RoutedCompressionUnit,
        ) -> tuple[int, str, tuple[object, Any, float]]:
            slot, result, elapsed_ms = _compress_routed_unit(routed)
            self._store_openai_responses_cached_unit(cache_key, result)
            return unit_idx, cache_key, (slot, result, elapsed_ms)

        def _record_routed_result(
            unit_idx: int,
            cache_key: str,
            routed_result: tuple[object, Any, float],
        ) -> None:
            routed_results[unit_idx] = routed_result
            _slot, result, _elapsed_ms = routed_result
            for follower_idx in cache_miss_followers.get(cache_key, []):
                routed_results[follower_idx] = (
                    routed_units[follower_idx].slot,
                    _openai_responses_result_with_cache_hit(result),
                    0.0,
                )

        parallelism = _openai_responses_unit_parallelism()
        if len(cache_misses) > 1 and parallelism > 1:
            executor = _openai_responses_unit_executor()
            for start in range(0, len(cache_misses), parallelism):
                batch = cache_misses[start : start + parallelism]
                futures = [executor.submit(_compress_and_store, *item) for item in batch]
                for future in as_completed(futures):
                    unit_idx, cache_key, routed_result = future.result()
                    _record_routed_result(unit_idx, cache_key, routed_result)
        else:
            for unit_idx, cache_key, routed in cache_misses:
                _record_routed_result(
                    unit_idx,
                    cache_key,
                    _compress_and_store(unit_idx, cache_key, routed)[2],
                )

        # Tail batches below the shared 512B floor keep the existing size-floor
        # result without entering the router or the unit-result cache.
        for entry in small_batch_skipped:
            unit_idx = unit_index_by_slot[entry.routed.slot]
            routed_results[unit_idx] = _compress_routed_unit(entry.routed)

        def _compress_batch(batch: Any) -> tuple[list[tuple[object, Any]], float]:
            batch_started = time.perf_counter()
            results = compress_batch_with_router(
                batch,
                router=router,
                tokenizer=tokenizer,
                target_ratio=unit_target_ratio,
            )
            return results, (time.perf_counter() - batch_started) * 1000.0

        def _record_batch_result(batch_result: tuple[list[tuple[object, Any]], float]) -> None:
            batch_results, elapsed_ms = batch_result
            elapsed_per_unit = elapsed_ms / len(batch_results) if batch_results else 0.0
            for slot, result in batch_results:
                routed_results[unit_index_by_slot[slot]] = (slot, result, elapsed_per_unit)

        if len(small_batches) > 1 and parallelism > 1:
            executor = _openai_responses_unit_executor()
            for start in range(0, len(small_batches), parallelism):
                batch_group = small_batches[start : start + parallelism]
                futures = [executor.submit(_compress_batch, batch) for batch in batch_group]
                for future in as_completed(futures):
                    _record_batch_result(future.result())
        else:
            for batch in small_batches:
                _record_batch_result(_compress_batch(batch))

        ordered_routed_results = [result for result in routed_results if result is not None]

        for _, result, elapsed_ms in ordered_routed_results:
            router_chain = list(result.router_result.strategy_chain) if result.router_result else []
            router_content_type = (
                result.router_result.routing_log[0].content_type.value
                if result.router_result and result.router_result.routing_log
                else "unknown"
            )
            timing_sink["compression_unit_router_total"] = (
                timing_sink.get("compression_unit_router_total", 0.0) + elapsed_ms
            )
            timing_sink[f"compression_unit_router_strategy_{result.strategy}"] = (
                timing_sink.get(f"compression_unit_router_strategy_{result.strategy}", 0.0)
                + elapsed_ms
            )
            timing_sink[f"compression_unit_router_category_{result.reason_category}"] = (
                timing_sink.get(f"compression_unit_router_category_{result.reason_category}", 0.0)
                + elapsed_ms
            )
            record_unit = getattr(getattr(self, "metrics", None), "record_codex_ws_unit", None)
            if record_unit is not None:
                record_unit(
                    strategy=result.strategy,
                    reason_category=result.reason_category,
                    elapsed_ms=elapsed_ms,
                    text_bytes=result.text_bytes,
                    tokens_before=result.tokens_before,
                    tokens_after=result.tokens_after,
                    tokens_saved=result.tokens_saved,
                    modified=result.modified,
                    strategy_chain=router_chain,
                    content_type=router_content_type,
                    text_shape=_codex_ws_text_shape(result.original),
                )
            if elapsed_ms >= 1000.0:
                logger.info(
                    "[%s] WS /v1/responses slow compression unit "
                    "elapsed_ms=%.0f strategy=%s category=%s modified=%s "
                    "content_type=%s text_shape=%s bytes=%d min_bytes=%d "
                    "tokens_before=%d tokens_after=%d tokens_saved=%d "
                    "strategy_chain=%s",
                    request_id,
                    elapsed_ms,
                    result.strategy,
                    result.reason_category,
                    result.modified,
                    router_content_type,
                    _codex_ws_text_shape(result.original),
                    result.text_bytes,
                    result.min_bytes,
                    result.tokens_before,
                    result.tokens_after,
                    result.tokens_saved,
                    router_chain,
                )
        _add_timing("compression_units_router_loop", router_total_started)

        apply_started = time.perf_counter()
        for slot, result, _elapsed_ms in ordered_routed_results:
            item_idx, slot_ref = slot
            router_chain = list(result.router_result.strategy_chain) if result.router_result else []
            for s in router_chain:
                if s not in strategy_chain_union:
                    strategy_chain_union.append(s)
            cat = result.reason_category or "applied"
            units_by_category[cat] = units_by_category.get(cat, 0) + 1
            # A unit "reached the router" iff the result carries a
            # router_result OR was modified — both indicate we got
            # past the early gates. Units that were size-floored,
            # role-protected, or in a frozen cache_zone don't count.
            if result.router_result is not None or result.modified:
                attempted_input_tokens += result.tokens_before
            if debug_enabled:
                _log(
                    "codex_compression_unit_result",
                    item_index=item_idx,
                    slot=slot_ref,
                    modified=result.modified,
                    reason=result.reason,
                    reason_category=cat,
                    text_bytes=result.text_bytes,
                    min_bytes=result.min_bytes,
                    strategy=result.strategy,
                    strategy_chain=router_chain,
                    tokens_before=result.tokens_before,
                    tokens_after=result.tokens_after,
                    tokens_saved=result.tokens_saved,
                    transforms_applied=result.transforms_applied,
                    router_strategy=(
                        result.router_result.strategy_used.value if result.router_result else None
                    ),
                    router_summary=result.router_result.summary() if result.router_result else None,
                    router_routing_log=_routing_log_debug(result.router_result),
                    router_cache_hit=(
                        result.router_result.cache_hit if result.router_result else False
                    ),
                    original=result.original,
                    compressed=result.compressed,
                )
            if not result.modified:
                continue

            target_item = updated_items[item_idx]
            if not isinstance(target_item, dict):
                continue
            if _set_slot_text(target_item, slot_ref, result.compressed):
                modified = True
                tokens_saved_total += result.tokens_saved
                for transform in result.transforms_applied:
                    if transform not in transforms:
                        transforms.append(transform)
        _add_timing("compression_unit_apply_results", apply_started)

        # Splice byte/data-lossless folds of excluded tool outputs (grep/log/
        # json). These skip the ML compressor entirely — the fold is already
        # information-preserving — so "excluded = no lossy" still holds.
        for e_idx, e_slot, e_folded, e_orig in lossless_excluded:
            e_target = updated_items[e_idx] if e_idx < len(updated_items) else None
            if not isinstance(e_target, dict):
                continue
            if _set_slot_text(e_target, e_slot, e_folded):
                modified = True
            e_before = tokenizer.count_text(e_orig)
            e_saved = e_before - tokenizer.count_text(e_folded)
            if e_saved > 0:
                tokens_saved_total += e_saved
            attempted_input_tokens += e_before
            if "router:excluded:lossless" not in transforms:
                transforms.append("router:excluded:lossless")

        # Cross-turn dedup over tool-output slots (Codex/Responses re-reads): a
        # repeated function_call_output.output folds to a one-line pointer to the
        # earlier copy. Per-unit compression above can't do this — it is
        # inherently cross-item. Keep-earliest + prefix-monotonic: the earlier
        # (cached-prefix) copy is never rewritten, so appending a turn does not
        # change cached bytes. Runs on the post-per-unit forms, mirroring the
        # chat path (ContentRouter._cross_turn_dedup_messages runs last there too).
        if getattr(router, "_cross_turn_dedup_enabled", False):
            dd_folded, dd_saved = _dedup_responses_output_items(
                updated_items,
                self.OPENAI_RESPONSES_OUTPUT_TYPES,
                tokenizer.count_text,
                protected_call_ids=verbatim_excluded_call_ids,
            )
            if dd_folded:
                modified = True
                tokens_saved_total += dd_saved
                if "router:responses_cross_turn_dedup" not in transforms:
                    transforms.append("router:responses_cross_turn_dedup")

        _log(
            "codex_compression_payload_result",
            modified=modified,
            tokens_saved_total=tokens_saved_total,
            attempted_input_tokens=attempted_input_tokens,
            transforms=transforms,
            units_by_category=units_by_category,
            strategy_chain=strategy_chain_union,
            input_payload=payload,
            output_payload=updated if modified else payload,
        )
        return (
            updated,
            modified,
            tokens_saved_total,
            transforms,
            units_by_category,
            strategy_chain_union,
            attempted_input_tokens,
        )

    def _compress_openai_responses_payload(
        self,
        payload: dict[str, Any],
        *,
        model: str,
        request_id: str,
        timing: dict[str, float] | None = None,
        client: str | None = None,
    ) -> tuple[dict[str, Any], bool, int, list[str], str | None, int, int, int]:
        """Compress an OpenAI Responses payload through the shared router.

        Provider adapters pass only the inner Responses payload here. This
        function is envelope-agnostic: it extracts Responses text slots into
        provider-neutral compression units, lets ContentRouter choose the
        compressor, then splices accepted replacements back into the payload.
        """

        timing_sink: dict[str, float] = timing if timing is not None else {}

        def _add_timing(name: str, started_at: float) -> None:
            timing_sink[name] = (
                timing_sink.get(name, 0.0) + (time.perf_counter() - started_at) * 1000.0
            )

        input_serialization_started = time.perf_counter()
        input_bytes = json.dumps(payload).encode("utf-8")
        _add_timing("compression_input_json_dump", input_serialization_started)
        # Codex/Responses requests can re-enter this method many times per
        # request_id (one per turn over the same websocket). Tag every
        # event in this single pass with a content-derived id so dashboards
        # can attribute each unit_result to its originating pass.
        # Aggregation note: per-pass `tokens_saved` SHOULD sum across
        # passes — every pass independently avoided sending those tokens
        # upstream, regardless of any prefix cache the upstream applies.
        # Identical pass_ids within one request_id indicate idempotent
        # retries on the same input bytes and are the only thing that
        # should be deduped.
        debug_enabled = _codex_compression_debug_enabled()
        pass_id = hashlib.sha256(input_bytes).hexdigest()[:12] if debug_enabled else None
        input_context_budget: dict[str, Any] | None = None
        if debug_enabled:
            input_context_budget = _openai_responses_context_budget(payload)
            _log_codex_compression_debug(
                "codex_compression_payload_input",
                request_id=request_id,
                pass_id=pass_id,
                model=model,
                input_bytes=len(input_bytes),
                context_budget=input_context_budget,
                input_top_level_keys=list(payload.keys()),
                input_field_type=type(payload.get("input")).__name__,
                messages_field_type=type(payload.get("messages")).__name__,
                payload=payload,
            )
        working = payload
        modified = False
        tokens_saved = 0
        transforms: list[str] = []
        reason: str | None = None

        tool_compaction_started = time.perf_counter()
        compacted_payload, tools_modified, tools_before_bytes, tools_after_bytes = (
            _compact_openai_responses_tools(working)
        )
        _add_timing("compression_tool_schema_compaction", tool_compaction_started)
        if tools_modified:
            working = compacted_payload
            modified = True
            reason = None
            transforms.append("openai:responses:tool_schema_compaction")
            try:
                tool_token_started = time.perf_counter()
                tokenizer = self.openai_provider.get_token_counter(model)
                tokens_saved += max(
                    0,
                    tokenizer.count_text(_json_debug_dumps(payload.get("tools")))
                    - tokenizer.count_text(_json_debug_dumps(working.get("tools"))),
                )
                _add_timing("compression_tool_schema_token_count", tool_token_started)
            except Exception:
                pass
            if debug_enabled:
                _log_codex_compression_debug(
                    "codex_tool_schema_compaction",
                    request_id=request_id,
                    pass_id=pass_id,
                    model=model,
                    modified=True,
                    tools_bytes_before=tools_before_bytes,
                    tools_bytes_after=tools_after_bytes,
                    tools_bytes_saved=tools_before_bytes - tools_after_bytes,
                )

        # Layer 2: Tool description truncation (opt-in via
        # HEADROOM_TOOL_DESC_MAX_CHARS).
        try:
            from headroom.proxy.tool_schema_compaction import (
                compact_tool_descriptions,
                tool_desc_max_chars,
            )

            _desc_max = tool_desc_max_chars()
            if _desc_max > 0:
                _desc_compact_started = time.perf_counter()
                desc_payload, desc_modified, desc_before, desc_after = compact_tool_descriptions(
                    working, _desc_max
                )
                _add_timing("compression_tool_desc_compaction", _desc_compact_started)
                if desc_modified:
                    working = desc_payload
                    modified = True
                    transforms.append("openai:responses:tool_desc_compaction")
                    try:
                        tokenizer = self.openai_provider.get_token_counter(model)
                        tokens_saved += max(
                            0,
                            tokenizer.count_text(_json_debug_dumps(payload.get("tools")))
                            - tokenizer.count_text(_json_debug_dumps(working.get("tools"))),
                        )
                    except Exception:
                        pass
                    if debug_enabled:
                        _log_codex_compression_debug(
                            "codex_tool_desc_compaction",
                            request_id=request_id,
                            pass_id=pass_id,
                            model=model,
                            modified=True,
                            tools_bytes_before=desc_before,
                            tools_bytes_after=desc_after,
                            tools_bytes_saved=desc_before - desc_after,
                        )
        except Exception:
            pass

        # Server-side Tool Search deferral (OpenAI Responses, gpt-5.4+): mark
        # non-core function/MCP tools defer_loading + inject {"type": "tool_search"}
        # so OpenAI keeps their heavy parameter schemas out of the model's context
        # until searched (every tool stays callable, cache preserved). No-op for
        # older models / small tool sets / clients already using tool search. The
        # deferred defs still ride in the request body (OpenAI needs them to load
        # on demand), so this is a provider-side context saving, not a request-byte
        # one — hence a transform tag but no tokens_saved claim.
        from headroom.proxy.helpers import inject_tool_search_deferral_openai

        _deferred_tools = inject_tool_search_deferral_openai(
            working.get("tools"), model, client=client
        )
        if _deferred_tools is not working.get("tools"):
            if working is payload:
                working = copy.deepcopy(payload)
            working["tools"] = _deferred_tools
            modified = True
            transforms.append("openai:responses:tool_search_deferral")

        # Turn hooks (opt-in extensions): a registered hook may inspect or rewrite
        # the outbound tools before we send — the extensible counterpart to the
        # built-in deferral above. Gated on the registry so it is a no-op (no copy,
        # no context construction) when no hook is registered.
        from headroom.proxy.turn_hooks import (
            TurnContext,
            registered_turn_hooks,
            run_request_hooks,
        )

        if registered_turn_hooks():
            if working is payload:
                working = copy.deepcopy(payload)
            # Match the ctx's `input or messages or []` truthy fallback so we write
            # back / re-count the SAME list the hook was handed.
            _msg_key = (
                "input"
                if working.get("input")
                else ("messages" if working.get("messages") else None)
            )
            _msgs_before = (working.get(_msg_key) if _msg_key else None) or []
            # Snapshot the pre-hook count BEFORE run_request_hooks — a hook may fold
            # in place, which would corrupt a post-hook "before" measurement.
            _mt_before = 0
            if _msg_key:
                try:
                    _mt_before = self.openai_provider.get_token_counter(model).count_text(
                        _json_debug_dumps(_msgs_before)
                    )
                except Exception:
                    _mt_before = 0
            _req_ctx = TurnContext(
                provider="openai",
                model=str(model),
                messages=_msgs_before,
                tools=working.get("tools"),
                config=getattr(self, "config", None),
            )
            run_request_hooks(_req_ctx)
            if _req_ctx.tools is not working.get("tools"):
                working["tools"] = _req_ctx.tools
            # A hook may also fold the messages (replace or in-place). Write back a
            # replaced list — previously dropped on this path — then re-count so the
            # message-fold saving is both applied AND recorded in tokens_saved.
            if _msg_key and _req_ctx.messages is not _msgs_before:
                working[_msg_key] = _req_ctx.messages
            if _msg_key and _mt_before:
                try:
                    _mt_after = self.openai_provider.get_token_counter(model).count_text(
                        _json_debug_dumps(working.get(_msg_key) or [])
                    )
                    if _mt_after < _mt_before:
                        tokens_saved += _mt_before - _mt_after
                except Exception:
                    pass
            modified = True
            transforms.append("openai:responses:turn_hook")

        live_units_started = time.perf_counter()
        (
            router_payload,
            router_modified,
            router_saved,
            router_transforms,
            units_by_category,
            strategy_chain,
            router_attempted_tokens,
        ) = self._compress_openai_responses_live_text_units_with_router(
            working,
            model=model,
            request_id=request_id,
            pass_id=pass_id,
            timing=timing_sink,
        )
        _add_timing("compression_live_units_total", live_units_started)
        if router_modified:
            working = router_payload
            modified = True
            reason = None
            tokens_saved += int(router_saved)
            transforms.extend(router_transforms)
        elif not modified:
            reason = "router_no_compression"

        # Total tokens we *attempted* to compress on this pass:
        # router-fed unit tokens + the original (pre-compaction) tool
        # schema tokens we ran schema_compaction against. Excludes
        # instructions, user messages, prior assistant turns, and
        # other prefix bytes we never tried to touch — those belong
        # to the prefix-cache denominator, not the active-compression
        # one.
        attempted_input_tokens = int(router_attempted_tokens)
        if tools_modified:
            try:
                attempted_token_started = time.perf_counter()
                tokenizer = self.openai_provider.get_token_counter(model)
                attempted_input_tokens += tokenizer.count_text(
                    _json_debug_dumps(payload.get("tools"))
                )
                _add_timing(
                    "compression_tool_schema_attempted_token_count",
                    attempted_token_started,
                )
            except Exception:
                pass

        dedupe_started = time.perf_counter()
        deduped: list[str] = []
        for transform in transforms:
            if transform not in deduped:
                deduped.append(transform)
        _add_timing("compression_transform_dedupe", dedupe_started)

        output_serialization_started = time.perf_counter()
        output_bytes = json.dumps(working).encode("utf-8")
        _add_timing("compression_output_json_dump", output_serialization_started)
        output_context_budget = _openai_responses_context_budget(working) if debug_enabled else None
        # One-line summary at INFO — the single event a human reading
        # logs should scan first to understand "what happened on this
        # pass". All the verbose per-event debug data stays available
        # but at DEBUG level. Contains: byte totals, savings, the
        # strategy chain we walked, unit-outcome counts by category,
        # and the transforms applied.
        savings_pct = (
            (1.0 - len(output_bytes) / len(input_bytes)) * 100.0 if len(input_bytes) else 0.0
        )
        # Active-compression ratio: savings as a fraction of what we
        # *attempted* to compress, not of the whole request. The whole-
        # request ratio is in `savings_pct`; this one is the metric the
        # dashboard should display (otherwise frozen prefix bytes drown
        # the wins from the compressible tail).
        #
        # Math note: `attempted_input_tokens` is the pre-compression
        # size of the eligible content (sum of unit.tokens_before +
        # original tool schema). `tokens_saved` is what we removed
        # from it. So the savings rate is plain `saved / attempted` —
        # NOT `saved / (attempted + saved)`, which would double-count.
        attempted_pct = (
            (tokens_saved / attempted_input_tokens) * 100.0 if attempted_input_tokens > 0 else 0.0
        )
        if debug_enabled:
            _log_codex_compression_debug(
                "codex_compression_pass_summary",
                request_id=request_id,
                pass_id=pass_id,
                model=model,
                modified=modified,
                reason=reason,
                input_bytes=len(input_bytes),
                output_bytes=len(output_bytes),
                bytes_saved=len(input_bytes) - len(output_bytes),
                savings_pct=round(savings_pct, 2),
                tokens_saved=tokens_saved,
                attempted_input_tokens=attempted_input_tokens,
                attempted_pct=round(attempted_pct, 2),
                strategy_chain=strategy_chain,
                units_by_category=units_by_category,
                transforms=deduped,
            )
            _log_codex_compression_debug(
                "codex_compression_payload_output",
                request_id=request_id,
                pass_id=pass_id,
                model=model,
                modified=modified,
                reason=reason,
                tokens_saved=tokens_saved,
                attempted_input_tokens=attempted_input_tokens,
                transforms=deduped,
                input_bytes=len(input_bytes),
                output_bytes=len(output_bytes),
                context_budget_before=input_context_budget,
                context_budget_after=output_context_budget,
                input_payload=payload,
                output_payload=working,
            )
        return (
            working,
            modified,
            tokens_saved,
            deduped,
            reason,
            len(input_bytes),
            len(output_bytes),
            attempted_input_tokens,
        )

    async def _compress_openai_responses_payload_in_executor(
        self,
        payload: dict[str, Any],
        *,
        model: str,
        request_id: str,
        timeout: float = COMPRESSION_TIMEOUT_SECONDS,
        client: str | None = None,
    ) -> tuple[dict[str, Any], bool, int, list[str], str | None, int, int, int, dict[str, float]]:
        timing: dict[str, float] = {}

        def _compress():  # noqa: ANN202
            # Output shaping (opt-in via HEADROOM_OUTPUT_SHAPER) runs before
            # compression so the turn classifier sees the client's input as
            # sent, and the steering/effort mutations ride the same rewrite
            # path as compression at every call site (HTTP /v1/responses, WS
            # first frame, WS subsequent frames). Runs inside the executor
            # closure so the extra payload serialization stays off the event
            # loop.
            shape_labels, shape_mutated = _shape_openai_responses_payload(
                payload, model=model, request_id=request_id
            )
            compression_kwargs: dict[str, Any] = {
                "model": model,
                "request_id": request_id,
                "timing": timing,
                "client": client,
            }
            while True:
                try:
                    result = self._compress_openai_responses_payload(
                        payload,
                        **compression_kwargs,
                    )
                    break
                except TypeError as exc:
                    unsupported_kwarg = next(
                        (
                            name
                            for name in ("client", "timing")
                            if f"unexpected keyword argument '{name}'" in str(exc)
                            and name in compression_kwargs
                        ),
                        None,
                    )
                    if unsupported_kwarg is None:
                        raise
                    compression_kwargs.pop(unsupported_kwarg)
            if shape_labels:
                # Carry the shaper labels on the transforms channel so the
                # outcome funnel feeds the output-savings ledger
                # (outcome.py record_from_labels). The modified flag is
                # forced only when shaping actually mutated the payload, so
                # every call site's rewrite path serializes the shaped bytes.
                result = (
                    result[0],
                    result[1] or shape_mutated,
                    result[2],
                    [*shape_labels, *result[3]],
                    *result[4:],
                )
            return result

        result = await self._run_compression_in_executor(
            _compress,
            timeout=timeout,
        )
        if len(result) == 8:
            return (*result, timing)
        return result

    async def handle_openai_chat(
        self,
        request: Request,
    ) -> Response | StreamingResponse:
        """Handle OpenAI /v1/chat/completions endpoint."""
        if not hasattr(self, "pipeline_extensions"):
            from headroom.pipeline import PipelineExtensionManager

            self.pipeline_extensions = PipelineExtensionManager(discover=False)

        from fastapi import HTTPException
        from fastapi.responses import JSONResponse, Response

        from headroom.ccr import CCRToolInjector
        from headroom.proxy.helpers import (
            COMPRESSION_TIMEOUT_SECONDS,
            MAX_MESSAGE_ARRAY_LENGTH,
            MAX_REQUEST_BODY_SIZE,
            _read_request_json,
        )
        from headroom.proxy.modes import is_cache_mode, is_token_mode
        from headroom.utils import extract_user_query

        start_time = time.time()
        request_id = await self._next_request_id()

        # Phase F PR-F1: classify auth mode at request entry. The result
        # is stored on `request.state` so downstream handlers (cache
        # gates, header injection, lossy-compressor gates) read it
        # without re-classifying. Pure function, well under 10us.
        auth_mode = classify_auth_mode(request.headers)
        request.state.auth_mode = auth_mode
        logger.debug(f"[{request_id}] auth_mode_classified mode={auth_mode.value}")

        # Check request body size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": f"Request body too large. Maximum size is {MAX_REQUEST_BODY_SIZE // (1024 * 1024)}MB",
                        "type": "invalid_request_error",
                        "code": "request_too_large",
                    }
                },
            )

        # Parse request
        try:
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid request body: {e!s}",
                        "type": "invalid_request_error",
                        "code": "invalid_json",
                    }
                },
            )
        model = body.get("model", "unknown")
        messages = body.get("messages", [])
        original_client_messages = copy.deepcopy(messages)
        custom_upstream_base_url = _resolve_openai_upstream_base(request.headers)
        upstream_base_url = self._resolve_openai_upstream(request)
        handler_path_suffix = _resolve_openai_chat_handler_path(
            upstream_base_url,
            model,
        )
        handler_path = (
            _resolve_openai_handler_path(request.headers, handler_path=handler_path_suffix)
            if custom_upstream_base_url is not None
            else f"/v1{handler_path_suffix}"
        )
        input_event = self.pipeline_extensions.emit(
            PipelineStage.INPUT_RECEIVED,
            operation="proxy.request",
            request_id=request_id,
            provider="openai",
            model=model,
            messages=messages,
            tools=body.get("tools"),
            metadata={"path": handler_path, "stream": body.get("stream", False)},
        )
        if input_event.messages is not None:
            messages = input_event.messages
            original_client_messages = copy.deepcopy(messages)
        if input_event.tools is not None:
            body["tools"] = input_event.tools

        # Validate message array size
        if len(messages) > MAX_MESSAGE_ARRAY_LENGTH:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Message array too large ({len(messages)} messages). "
                        f"Maximum is {MAX_MESSAGE_ARRAY_LENGTH}.",
                        "type": "invalid_request_error",
                        "code": "invalid_request",
                    }
                },
            )

        stream = body.get("stream", False)

        # Learn from the original client payload before memory context or
        # compression mutates it, mirroring the Responses and Anthropic
        # ingestion paths. Without this, chat/completions traffic (Copilot CLI,
        # opencode, OpenAI SDKs) fed nothing to the learner (part of #2060).
        await self._observe_openai_chat_traffic(original_client_messages, request_id=request_id)

        # Bypass: skip ALL compression for explicit opt-out
        _bypass = self._headroom_bypass_enabled(request.headers)
        if _bypass:
            logger.info(f"[{request_id}] Bypass: skipping compression (header)")

        # Image compression: tile alignment + ML-based technique routing.
        # Gated on ImageCompressionDecision — same value-type pattern
        # as CompressionDecision + MemoryDecision; locks bypass-respect
        # in tests so a future site can't drift.
        from headroom.proxy.image_compression_decision import ImageCompressionDecision

        _image_decision = ImageCompressionDecision.decide(
            headers=request.headers, config=self.config, messages=messages
        )
        # tags is populated downstream at L1229 — defer apply_to_tags
        # to where the tags dict exists. The decision is captured here
        # so the conditional is uniform with the other gates.
        if _image_decision.should_compress:
            from headroom.proxy.helpers import _get_image_compressor

            compressor = None
            try:
                compressor = _get_image_compressor()
                if compressor and compressor.has_images(messages):
                    messages, image_result = await run_image_compression_isolated(
                        messages,
                        provider="openai",
                        timeout=COMPRESSION_TIMEOUT_SECONDS,
                    )
                    if image_result is not None:
                        logger.info(
                            f"[{request_id}] Image: {image_result['technique']} "
                            f"({image_result['savings_percent']:.0f}% saved, "
                            f"{image_result['original_tokens']} → "
                            f"{image_result['compressed_tokens']} tokens)"
                        )
            except Exception as e:
                # Image compression is best-effort — fail open on timeout/error and
                # forward the original messages, matching the text path.
                logger.warning(f"[{request_id}] Image compression failed: {type(e).__name__}: {e}")
            finally:
                if compressor and hasattr(compressor, "close"):
                    compressor.close()

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        # The parsed body was already content-decoded upstream, so the bytes we
        # forward are plain JSON. A stale content-encoding makes OpenAI try to
        # decompress already-decoded JSON and reject it with HTTP 400 (#1542 —
        # same fix the /v1/responses path already carries).
        headers.pop("content-encoding", None)
        headers.pop("transfer-encoding", None)
        # Strip accept-encoding so httpx negotiates its own encoding.
        # Cloudflare Workers forward "br, zstd" which OpenAI may honor;
        # if httpx lacks brotli support the response body is undecipherable → 502.
        headers.pop("accept-encoding", None)
        tags = extract_tags(headers)
        client = classify_client(headers)
        # Surface the image-compression decision (computed earlier) into
        # tags now that the tags dict exists. Same observability pattern
        # the funnel uses for passthrough_reason + memory_skip_reason.
        _image_decision.apply_to_tags(tags)
        # PR-A5 (P5-49): strip internal x-headroom-* from upstream-bound
        # headers AFTER `_extract_tags` reads them. Inbound bypass gating
        # uses `request.headers.get(...)` above; memory user-id reads
        # `request.headers` below. From this point on, `headers` is the
        # upstream-bound copy.
        from headroom.proxy.helpers import (
            _strip_internal_headers,
            log_outbound_headers,
            merge_extra_headers,
        )

        _pre_strip_count_chat = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        headers = merge_extra_headers(headers, self.config.openai_extra_headers)
        log_outbound_headers(
            forwarder="openai_chat_completions",
            stripped_count=_pre_strip_count_chat,
            request_id=request_id,
        )
        upstream_base_url = _resolve_openai_upstream_base(request.headers)
        handler_path = (
            _resolve_openai_handler_path(
                request.headers,
                handler_path=_OPENAI_CHAT_COMPLETIONS_PATH,
            )
            if upstream_base_url is not None
            else "/v1/chat/completions"
        )
        _, custom_chat_provider = _custom_base_passthrough_telemetry(
            request.method,
            handler_path,
            upstream_base_url or "",
        )
        openai_chat_outcome_provider = custom_chat_provider or "openai"

        # Memory: Get user ID when memory is enabled. Reads `request.headers`
        # directly because `headers` was stripped of `x-headroom-*` for the
        # upstream-bound copy (PR-A5).
        memory_user_id: str | None = None
        memory_request_ctx = None
        if self.memory_handler:
            memory_user_id = request.headers.get(
                "x-headroom-user-id",
                os.environ.get("USER", os.environ.get("USERNAME", "default")),
            )
            # Per-project memory routing (GH #462). Built once per request
            # so every save/search/inject resolves to the same workspace.
            from headroom.memory.storage_router import (
                RequestContext as _MemRequestContext,
            )
            from headroom.memory.storage_router import (
                extract_system_prompt as _extract_sys_prompt,
            )

            memory_request_ctx = _MemRequestContext(
                headers=dict(request.headers),
                system_prompt=_extract_sys_prompt(body),
                base_user_id=memory_user_id,
                project_root_override=(
                    getattr(self.memory_handler.config, "project_root_override", "") or None
                ),
            )

        # Canonical memory-injection gate (parallels Anthropic). Pre-
        # PR-this the inline conjunction at the memory site silently
        # ignored `x-headroom-bypass: true`, mutating request bytes
        # under the user's "don't touch my bytes" signal.
        from headroom.proxy.helpers import get_memory_injection_mode
        from headroom.proxy.memory_decision import MemoryDecision
        from headroom.proxy.memory_query import MemoryQuery

        memory_decision = MemoryDecision.decide(
            headers=request.headers,
            memory_handler=self.memory_handler,
            memory_user_id=memory_user_id,
            mode_name=get_memory_injection_mode(),
        )
        memory_decision.apply_to_tags(tags)

        # Rate limiting
        if self.rate_limiter:
            rate_key = headers.get("authorization", "default")[:20]
            allowed, wait_seconds = await self.rate_limiter.check_request(rate_key)
            if not allowed:
                await self.metrics.record_rate_limited(provider=openai_chat_outcome_provider)
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limited. Retry after {wait_seconds:.1f}s",
                )

        # Snapshot cache-key fields ONCE here (pre-upstream), reused verbatim
        # at the cache.set site below — re-reading body at set risks a mutated
        # body (e.g. tools reassigned) and a key mismatch (#327). OpenAI's
        # system prompt lives inside `messages` (already in the key), so it is
        # not folded separately. Fold in the response-shaping fields the request
        # forwards — else two requests with identical messages but a different
        # reasoning_effort / response_format / sampling config collide and the
        # second caller is served a response made under other semantics (#1473
        # review). Transport/metadata fields (stream, store, user, service_tier)
        # and the deprecated functions API are intentionally excluded.
        cache_key_fields = {
            "tools": body.get("tools"),
            "tool_choice": body.get("tool_choice"),
            "response_format": body.get("response_format"),
            "parallel_tool_calls": body.get("parallel_tool_calls"),
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
            "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens"),
            "stop": body.get("stop"),
            "seed": body.get("seed"),
            "presence_penalty": body.get("presence_penalty"),
            "frequency_penalty": body.get("frequency_penalty"),
            "logit_bias": body.get("logit_bias"),
            "n": body.get("n"),
            "logprobs": body.get("logprobs"),
            "top_logprobs": body.get("top_logprobs"),
            "reasoning_effort": body.get("reasoning_effort"),
            "verbosity": body.get("verbosity"),
            "modalities": body.get("modalities"),
        }
        # Snapshot the lookup messages too. `messages` is the primary cache
        # key component, but the pre_compress hook below reassigns it, so
        # caching the response under the live `messages` would store it under a
        # different key than it was looked up by — the cache would never hit
        # and would fill with unreachable entries. Reuse this raw snapshot
        # verbatim at cache.set (the same reason cache_key_fields is
        # snapshotted here, #327). Image compression above also rebinds
        # `messages`, but it runs before this snapshot, so its output is already
        # captured — keep this snapshot after image compression, or a reorder
        # silently reintroduces the drift.
        cache_lookup_messages = messages
        # Check cache
        if self.cache and not stream:
            cached = await self.cache.get(messages, model, **cache_key_fields)
            if cached:
                self.pipeline_extensions.emit(
                    PipelineStage.INPUT_CACHED,
                    operation="proxy.request",
                    request_id=request_id,
                    provider="openai",
                    model=model,
                    messages=messages,
                    metadata={"cache_hit": True, "path": handler_path},
                )
                # Response-cache hit: same pattern as the anthropic
                # cache-hit site. ``from_response_cache=True`` is the
                # distinct signal that the proxy served from its own
                # semantic cache (not upstream prompt cache).
                _cache_hit_latency = (time.time() - start_time) * 1000
                await self._record_request_outcome(
                    RequestOutcome(
                        request_id=request_id,
                        provider=openai_chat_outcome_provider,
                        model=model,
                        original_tokens=0,
                        optimized_tokens=0,
                        output_tokens=0,
                        tokens_saved=0,
                        attempted_input_tokens=0,
                        from_response_cache=True,
                        total_latency_ms=_cache_hit_latency,
                        num_messages=len(messages),
                        tags=tags,
                        client=client,
                    )
                )

                # Remove compression headers from cached response
                response_headers = _sanitize_forwarded_response_headers(cached.response_headers)

                return Response(content=cached.response_body, headers=response_headers)

        # Token counting (offloaded off the event loop — GH #1701)
        tokenizer, original_tokens = await self._count_tokens_offloaded(model, messages)

        # Hook: pre_compress
        _hook_biases = None
        if self.config.hooks:
            from headroom.hooks import CompressContext

            _hook_ctx = CompressContext(model=model, provider="openai")
            try:
                messages = self.config.hooks.pre_compress(messages, _hook_ctx)
                _hook_biases = self.config.hooks.compute_biases(messages, _hook_ctx)
            except Exception as e:
                logger.debug(f"[{request_id}] Hook error: {e}")

        # Optimization
        transforms_applied = []
        pipeline_timing: dict[str, float] = {}
        waste_signals_dict: dict[str, int] | None = None
        optimized_messages = messages
        optimized_tokens = original_tokens

        # Get prefix cache tracker for this session, resolved by conversation
        # lineage within the session id (#2085) so concurrent conversations
        # sharing a fallback id (same model + system prompt) do not thrash one
        # tracker's frozen-prefix state. Both the id and the lineage derive
        # from the SAME original client bytes — pre image compression and
        # pre_compress hook, both of which may rewrite history bytes
        # differently turn-to-turn (a turn-dependent rewrite of the leading
        # system text would rotate the id mid-conversation).
        openai_session_id = self.session_tracker_store.compute_session_id(
            request, model, original_client_messages
        )
        openai_prefix_tracker = self.session_tracker_store.resolve_tracker(
            openai_session_id, "openai", messages=original_client_messages
        )

        # PR-A6 (P5-50, preps P0-6): session-sticky `OpenAI-Beta` merge.
        # Same pattern as anthropic.py — read client value, union with
        # session-seen tokens, update tracker. WS auto-injection of
        # `responses_websockets=2026-02-06` lives on the WS handler;
        # chat-completions has no Headroom-required tokens today, so the
        # merge effectively just makes the client value byte-stable
        # across turns.
        from headroom.proxy.helpers import (
            get_session_beta_tracker as _get_session_beta_tracker_chat,
        )
        from headroom.proxy.helpers import (
            log_beta_header_merge as _log_beta_header_merge_chat,
        )

        _client_openai_beta = headers.get("openai-beta")
        _client_openai_beta_count = (
            len([t for t in (_client_openai_beta or "").split(",") if t.strip()])
            if _client_openai_beta
            else 0
        )
        _sticky_openai_beta = _get_session_beta_tracker_chat().record_and_get_sticky_betas(
            provider="openai",
            session_id=openai_session_id,
            client_value=_client_openai_beta,
        )
        _sticky_openai_beta_count = (
            len([t for t in _sticky_openai_beta.split(",") if t.strip()])
            if _sticky_openai_beta
            else 0
        )
        if _sticky_openai_beta and _sticky_openai_beta != (_client_openai_beta or ""):
            headers["openai-beta"] = _sticky_openai_beta
        _log_beta_header_merge_chat(
            provider="openai",
            session_id=openai_session_id,
            client_betas_count=_client_openai_beta_count,
            sticky_betas_count=_sticky_openai_beta_count,
            headroom_added=[],
            request_id=request_id,
        )

        openai_frozen_count = openai_prefix_tracker.get_frozen_message_count()
        if is_cache_mode(self.config.mode):
            openai_frozen_count = OpenAIHandlerMixin._strict_previous_turn_frozen_count(
                messages,
                openai_frozen_count,
            )
        # Cold-prefix cache-miss hook, branch 2 (HEADROOM_COLD_RECOMPACT). When the
        # prompt cache has lapsed (idle past TTL → dead, nothing to bust), unfreeze the
        # whole prefix so the router recompacts it (cross-turn dedupe + superseded-read
        # drop + lossless folds) — the lever for encrypted-reasoning models (Codex)
        # whose reasoning we can't touch. Composes with branch 1 (plain-text reasoning
        # drop at PRE_SEND for Kimi/GLM). Token mode only; deterministic → cache-stable.
        if is_token_mode(self.config.mode) and os.environ.get(
            "HEADROOM_COLD_RECOMPACT", ""
        ).strip().lower() in ("1", "true", "yes"):
            from headroom.cache.ttl_observations import resolve_learned_ttl
            from headroom.transforms.cold_prefix import is_cold_prefix

            # OpenAI/Kimi don't expose their cache TTL, so use the offline-learned
            # value if available (else is_cold_prefix falls back to the static guess).
            _learned_ttl = resolve_learned_ttl("openai", model)
            if is_cold_prefix(openai_prefix_tracker, ttl_seconds=_learned_ttl):
                logger.info(
                    "[%s] cold-prefix recompaction: unfreezing prefix for "
                    "dedupe/superseded-read compaction",
                    request_id,
                )
                openai_frozen_count = 0

        _compression_failed = False
        original_messages = messages  # Preserve for 400-retry fallback
        _decision = CompressionDecision.decide(
            headers=request.headers,
            config=self.config,
            usage_reporter=self.usage_reporter,
            messages=messages,
        )
        _decision.apply_to_tags(tags)
        if not _decision.should_compress:
            logger.info(
                f"[{request_id}] Compression skipped: reason={_decision.passthrough_reason}"
            )
        if _decision.should_compress:
            try:
                context_limit = self.openai_provider.get_context_limit(model)

                # F2.1 c5/5: per-request CompressionPolicy. Hoisted out of
                # the is_token_mode branch so the else (non-token) branch
                # below can pass it through too. See the equivalent block
                # in handlers/anthropic.py.
                from headroom.transforms.compression_policy import resolve_policy

                compression_policy = resolve_policy(getattr(request.state, "auth_mode", None))

                if is_token_mode(self.config.mode):
                    comp_cache = self._get_compression_cache(openai_session_id)

                    # Zone 1: Swap cached compressed versions
                    working_messages = comp_cache.apply_cached(messages)

                    # Re-freeze boundary. Token mode can use the compression
                    # cache's positional frozen count. Cache mode must keep the
                    # latest observation mutable even when the compression
                    # cache has no compressible entry for it yet; otherwise
                    # OpenAI-compatible tool-call clients freeze the entire
                    # conversation and report near-zero savings.
                    if not is_cache_mode(self.config.mode):
                        openai_frozen_count = comp_cache.compute_frozen_count(messages)

                    result = await self._run_compression_in_executor(
                        lambda: self.openai_pipeline.apply(
                            messages=working_messages,
                            model=model,
                            model_limit=context_limit,
                            context=extract_user_query(working_messages),
                            frozen_message_count=(
                                OpenAIHandlerMixin._strict_previous_turn_frozen_count(
                                    working_messages,
                                    openai_frozen_count,
                                )
                                if is_cache_mode(self.config.mode)
                                else openai_frozen_count
                            ),
                            biases=_hook_biases,
                            compression_policy=compression_policy,
                            # Thread the savings-profile knobs (e.g.
                            # HEADROOM_SAVINGS_PROFILE=agent-90) onto the live
                            # chat-completions path, matching handlers/
                            # anthropic.py and the dedicated OpenAI compress
                            # endpoint. Without this the profile's
                            # compress_user_messages/target_ratio/etc. were
                            # silently dropped here (#1534).
                            **proxy_pipeline_kwargs(self.config),
                        ),
                        timeout=COMPRESSION_TIMEOUT_SECONDS,
                    )

                    if result.messages != working_messages:
                        comp_cache.update_from_result(messages, result.messages)

                    # Always use pipeline result in token mode
                    optimized_messages = result.messages
                    transforms_applied = result.transforms_applied
                    pipeline_timing = result.timing
                    # Keep original_tokens as the REAL original (pre-Zone-1-swap)
                    # so tokens_saved captures both Zone 1 + Zone 2 savings.
                    optimized_tokens = result.tokens_after
                else:
                    apply_frozen_count = (
                        OpenAIHandlerMixin._strict_previous_turn_frozen_count(
                            messages,
                            openai_frozen_count,
                        )
                        if is_cache_mode(self.config.mode)
                        else openai_frozen_count
                    )
                    result = await self._run_compression_in_executor(
                        lambda: self.openai_pipeline.apply(
                            messages=messages,
                            model=model,
                            model_limit=context_limit,
                            context=extract_user_query(messages),
                            frozen_message_count=apply_frozen_count,
                            biases=_hook_biases,
                            compression_policy=compression_policy,
                            # Same savings-profile threading as the token-mode
                            # branch above — the non-token chat path must honor
                            # the configured profile too (#1534).
                            **proxy_pipeline_kwargs(self.config),
                        ),
                        timeout=COMPRESSION_TIMEOUT_SECONDS,
                    )

                    if result.messages != messages:
                        optimized_messages = result.messages
                        transforms_applied = result.transforms_applied
                        pipeline_timing = result.timing
                        original_tokens = result.tokens_before
                        optimized_tokens = result.tokens_after

                if result.waste_signals:
                    waste_signals_dict = result.waste_signals.to_dict()
            except Exception as e:
                # Include type so TimeoutError vs other failures is distinguishable
                # in bug reports — str(asyncio.TimeoutError()) is empty otherwise.
                logger.warning(
                    f"[{request_id}] Optimization failed: {type(e).__name__}: {e} "
                    f"(request forwarded unoptimized)",
                    exc_info=True,
                )
                # Flag compression failure for observability
                _compression_failed = True

        # Cache-safety (ALL modes): forward the previously-cached (compressed)
        # prefix byte-identical, so freezing can't bust the prompt cache. See the
        # matching guard in the Anthropic handler for the full rationale. Append-
        # only-guarded and idempotent (cache mode already replays).
        from headroom.cache.prefix_tracker import overlay_cached_prefix

        _ov = overlay_cached_prefix(
            optimized_messages,
            original_client_messages,
            openai_prefix_tracker.get_last_original_messages(),
            openai_prefix_tracker.get_last_forwarded_messages(),
        )
        if _ov != optimized_messages:
            optimized_messages = _ov
            optimized_tokens = tokenizer.count_messages(optimized_messages)

        # Guard: if "optimization" inflated tokens, revert to originals
        if optimized_tokens > original_tokens:
            logger.warning(
                f"[{request_id}] Optimization inflated tokens "
                f"({original_tokens} -> {optimized_tokens}), reverting to original messages"
            )
            optimized_messages = original_messages
            optimized_tokens = original_tokens
            transforms_applied = []

        tokens_saved = original_tokens - optimized_tokens
        optimization_latency = (time.time() - start_time) * 1000

        routing_markers = summarize_routing_markers(transforms_applied)
        if routing_markers:
            routed_event = self.pipeline_extensions.emit(
                PipelineStage.INPUT_ROUTED,
                operation="proxy.request",
                request_id=request_id,
                provider="openai",
                model=model,
                messages=optimized_messages,
                metadata={
                    "routing_markers": routing_markers,
                    "transforms_applied": transforms_applied,
                },
            )
            if routed_event.messages is not None:
                optimized_messages = routed_event.messages
                optimized_tokens = tokenizer.count_messages(optimized_messages)
                tokens_saved = original_tokens - optimized_tokens

        compressed_event = self.pipeline_extensions.emit(
            PipelineStage.INPUT_COMPRESSED,
            operation="proxy.request",
            request_id=request_id,
            provider="openai",
            model=model,
            messages=optimized_messages,
            metadata={
                "tokens_before": original_tokens,
                "tokens_after": optimized_tokens,
                "transforms_applied": transforms_applied,
                # Read-only reference for recording extensions (probe
                # recorder); extensions must not mutate it.
                "original_messages": original_messages,
            },
        )
        if compressed_event.messages is not None:
            optimized_messages = compressed_event.messages
            optimized_tokens = tokenizer.count_messages(optimized_messages)
            tokens_saved = original_tokens - optimized_tokens

        # Hook: post_compress
        if self.config.hooks and tokens_saved > 0:
            from headroom.hooks import CompressEvent

            try:
                self.config.hooks.post_compress(
                    CompressEvent(
                        tokens_before=original_tokens,
                        tokens_after=optimized_tokens,
                        tokens_saved=tokens_saved,
                        compression_ratio=tokens_saved / original_tokens
                        if original_tokens > 0
                        else 0,
                        transforms_applied=transforms_applied,
                        model=model,
                        provider="openai",
                    )
                )
            except Exception as e:
                logger.debug(f"[{request_id}] post_compress hook error: {e}")

        # CCR Tool Injection: Inject retrieval tool if compression occurred
        # OR if this session has previously done CCR (PR-B7 sticky-on).
        # See `headroom/proxy/handlers/anthropic.py` and PR-B7 plan
        # `REALIGNMENT/04-phase-B-live-zone.md` for the rationale: once a
        # session has done CCR, the `headroom_retrieve` tool stays
        # registered for every subsequent turn so the prompt cache
        # anchored on the previous turn's tool list never busts.
        tools = body.get("tools")
        _original_tools = tools  # Preserve for diagnostic / future retry
        if (
            self.config.ccr_inject_tool or self.config.ccr_inject_system_instructions
        ) and not _bypass:
            injector = CCRToolInjector(
                provider="openai",
                inject_tool=False,  # routed through sticky helper below
                inject_system_instructions=self.config.ccr_inject_system_instructions,
            )
            injector.scan_for_markers(optimized_messages)
            if self.config.ccr_inject_system_instructions and injector.has_compressed_content:
                optimized_messages = injector.inject_into_system_message(optimized_messages)

            if self.config.ccr_inject_tool:
                from headroom.proxy.helpers import (
                    apply_session_sticky_ccr_tool,
                    has_new_ccr_markers,
                )

                # #1850: markers replayed from overlay_cached_prefix are
                # historical; only markers NEW this turn should drive injection,
                # else we re-inject the tool every frozen turn and bust the
                # *tools* cache segment (undoing the overlay's messages-prefix
                # cache-safety).
                has_new_compressed_content = has_new_ccr_markers(
                    current_detected_hashes=injector.detected_hashes,
                    previous_forwarded_messages=openai_prefix_tracker.get_last_forwarded_messages(),
                    provider="openai",
                )
                tools, ccr_tool_injected = apply_session_sticky_ccr_tool(
                    provider="openai",
                    session_id=openai_session_id,
                    request_id=request_id,
                    existing_tools=tools,
                    has_compressed_content_this_turn=has_new_compressed_content,
                )
                if ccr_tool_injected:
                    logger.debug(
                        f"[{request_id}] CCR: tool registered (session={openai_session_id}, "
                        f"compressed_this_turn={injector.has_compressed_content}, "
                        f"hashes_seen={len(injector.detected_hashes)})"
                    )

        if is_cache_mode(self.config.mode):
            optimized_messages, restored_count = self._restore_frozen_prefix(
                original_client_messages,
                optimized_messages,
                frozen_message_count=openai_frozen_count,
            )
            if restored_count > 0:
                logger.warning(
                    f"[{request_id}] Restored {restored_count} frozen prefix message(s) "
                    "to preserve cache stability (openai)"
                )

        # Memory: inject context and tools for OpenAI requests.
        #
        # PR-A3 follow-up to A2: memory context now routes exclusively to
        # the live-zone tail (latest user message), never via a system-level
        # prepend. The cache hot zone (system messages) is sacrosanct —
        # invariant I2. See REALIGNMENT/03-phase-A-lockdown.md PR-A3.
        memory_context_injected = False
        memory_tools_injected = False
        if memory_decision.inject:
            # Memory-handler is guaranteed present when inject=True.
            # Timeout-wrap (matches Anthropic /v1/messages and
            # /v1/responses) — pre-PR-this site was the only chat
            # path without one.
            try:
                if self.memory_handler.config.inject_context:
                    memory_context = await asyncio.wait_for(
                        self.memory_handler.search_and_format_context(
                            memory_user_id,
                            optimized_messages,
                            request_context=memory_request_ctx,
                            query=MemoryQuery.from_messages(optimized_messages),
                        ),
                        timeout=(self.config.anthropic_pre_upstream_memory_context_timeout_seconds),
                    )
                    if memory_context:
                        from headroom.proxy.helpers import (
                            append_text_to_latest_user_chat_message,
                            get_memory_injection_mode,
                            log_memory_injection,
                        )

                        injection_mode = get_memory_injection_mode()
                        if injection_mode == "disabled":
                            log_memory_injection(
                                request_id=request_id,
                                session_id=None,
                                decision="skipped_disabled",
                                bytes_injected=0,
                                query=None,
                            )
                        else:
                            new_messages, bytes_appended = append_text_to_latest_user_chat_message(
                                optimized_messages, memory_context
                            )
                            if bytes_appended > 0:
                                optimized_messages = new_messages
                                memory_context_injected = True
                                log_memory_injection(
                                    request_id=request_id,
                                    session_id=None,
                                    decision="injected_live_zone_tail_chat",
                                    bytes_injected=bytes_appended,
                                    query=None,
                                    tags=tags,
                                )
                                logger.info(
                                    f"[{request_id}] Memory: Injected {bytes_appended} chars "
                                    f"into latest user message tail for user {memory_user_id}"
                                )
                            else:
                                log_memory_injection(
                                    request_id=request_id,
                                    session_id=None,
                                    decision="no_eligible_user_message",
                                    bytes_injected=0,
                                    query=None,
                                )

                # Inject memory tools — PR-A7 (P0-6) routes through
                # `apply_session_sticky_memory_tools` so byte-stable across turns.
                from headroom.proxy.helpers import (
                    apply_session_sticky_memory_tools as _apply_sticky_mem_tools,
                )

                memory_tool_defs = (
                    self.memory_handler.compute_memory_tool_definitions("openai")
                    if self.memory_handler.config.inject_tools
                    else []
                )
                tools, mem_tools_injected = _apply_sticky_mem_tools(
                    provider="openai",
                    session_id=openai_session_id,
                    request_id=request_id,
                    existing_tools=tools,
                    memory_tools_to_inject=memory_tool_defs,
                    inject_this_turn=bool(self.memory_handler.config.inject_tools),
                )
                if mem_tools_injected:
                    memory_tools_injected = True
                    logger.info(f"[{request_id}] Memory: Injected memory tools (openai)")
            except Exception as e:
                logger.warning(f"[{request_id}] Memory injection failed: {e}")

        if memory_context_injected or memory_tools_injected:
            remembered_event = self.pipeline_extensions.emit(
                PipelineStage.INPUT_REMEMBERED,
                operation="proxy.request",
                request_id=request_id,
                provider="openai",
                model=model,
                messages=optimized_messages,
                tools=tools,
                metadata={
                    "memory_context_injected": memory_context_injected,
                    "memory_tools_injected": memory_tools_injected,
                },
            )
            if remembered_event.messages is not None:
                optimized_messages = remembered_event.messages
            if remembered_event.tools is not None:
                tools = remembered_event.tools

        tool_tokens_before_compaction = 0
        tool_tokens_after_compaction = 0
        if _decision.should_compress and not _bypass and tools is not None:
            try:
                compacted_tool_payload, tools_modified, _, _ = _compact_openai_responses_tools(
                    {"tools": tools}
                )
                if tools_modified and compacted_tool_payload.get("tools") is not None:
                    tool_tokens_before_compaction = tokenizer.count_text(_json_debug_dumps(tools))
                    tools = compacted_tool_payload["tools"]
                    if "openai:chat:tool_schema_compaction" not in transforms_applied:
                        transforms_applied.append("openai:chat:tool_schema_compaction")
            except Exception as e:
                logger.debug(f"[{request_id}] tool schema compaction failed: {e}")

            # Layer 2: tool description truncation (opt-in via
            # HEADROOM_TOOL_DESC_MAX_CHARS). The Anthropic and Responses handlers
            # both ran this pass; chat-completions never did, so the env var was a
            # silent no-op for every chat client (opencode, Cline, Aider, Roo,
            # LiteLLM-routed). Tool descriptions live on the tools array, which the
            # message pipeline never sees, so nothing else was covering them.
            # `compact_tool_descriptions` walks both the nested chat shape
            # ({"function": {"description": ...}}) and the flat Responses shape.
            try:
                from headroom.proxy.tool_schema_compaction import (
                    compact_tool_descriptions,
                    tool_desc_max_chars,
                )

                _desc_max = tool_desc_max_chars()
                if _desc_max > 0:
                    _desc_payload, _desc_modified, _desc_before, _desc_after = (
                        compact_tool_descriptions({"tools": tools}, _desc_max)
                    )
                    if _desc_modified and _desc_payload.get("tools") is not None:
                        # Seed "before" only if schema compaction above didn't; the
                        # two passes chain, so "after" must track the latest tools.
                        if not tool_tokens_before_compaction:
                            tool_tokens_before_compaction = tokenizer.count_text(
                                _json_debug_dumps(tools)
                            )
                        tools = _desc_payload["tools"]
                        transforms_applied.append("openai:chat:tool_desc_compaction")
                        logger.debug(
                            "[%s] tool description compaction: %d -> %d bytes "
                            "(%.0f%% saved, max_chars=%d)",
                            request_id,
                            _desc_before,
                            _desc_after,
                            (1 - _desc_after / max(_desc_before, 1)) * 100,
                            _desc_max,
                        )
            except Exception as e:
                logger.debug(f"[{request_id}] tool desc compaction failed: {e}")

        body["messages"] = optimized_messages
        if tools or _original_tools is not None:
            body["tools"] = tools

        # Reasoning compaction (HEADROOM_THINKING_COMPACT, off by default). Unlike
        # Anthropic/OpenAI (encrypted reasoning handles, nothing to compress),
        # Kimi/GLM/DeepSeek-R1 resend reasoning as PLAIN TEXT billed as input (Kimi
        # K2.7: ~1,558 tok/block, kept every turn) — so Kompress can actually shrink
        # it. Shape-driven: compacts the `reasoning_content` field (Kimi) and inline
        # `<think>…</think>` (GLM/DeepSeek); no-ops when neither is present (OpenAI's
        # own encrypted models). Deterministic → forwarded prefix stays cache-stable;
        # keep_last_turns protects the active reasoning. Never breaks the request.
        if os.environ.get("HEADROOM_THINKING_COMPACT", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            try:
                from headroom.cache.ttl_observations import resolve_learned_ttl
                from headroom.transforms.cold_prefix import is_cold_prefix
                from headroom.transforms.compression_units import find_content_router
                from headroom.transforms.thinking_compactor import (
                    compact_reasoning_openai_chat,
                )

                # Cold-prefix cache-miss hook: on a warm turn Kompress the reasoning
                # (deterministic → cache-stable); on a COLD turn (idle past provider
                # TTL, cache dead) DROP it outright — the full block, safe because the
                # cold turn re-caches from scratch. See cold_prefix / thinking_compactor.
                # Uses the offline-learned TTL (Kimi/OpenAI don't expose it).
                _rc_cold = is_cold_prefix(
                    openai_prefix_tracker, ttl_seconds=resolve_learned_ttl("openai", model)
                )
                _rc_router = find_content_router(self.openai_pipeline)
                _rc_kompress = (
                    (_rc_router._get_remote_kompress() or _rc_router._get_kompress())
                    if _rc_router is not None
                    else None
                )
                # Drop needs no compressor; Kompress (warm) does.
                if _rc_kompress is not None or _rc_cold:
                    _rc_keep = int(os.environ.get("HEADROOM_THINKING_COMPACT_KEEP_LAST", "1"))
                    optimized_messages, _rc_stats = compact_reasoning_openai_chat(
                        optimized_messages,
                        kompress=_rc_kompress,
                        keep_last_turns=_rc_keep,
                        drop=_rc_cold,
                    )
                    body["messages"] = optimized_messages
                    if _rc_stats["turns_compacted"]:
                        transforms_applied.append(
                            f"openai:reasoning_{'drop' if _rc_cold else 'compact'}:{_rc_stats['turns_compacted']}"
                        )
                        logger.info(
                            "[%s] reasoning %s: %d turns, %d blocks, %d->%d words (cold=%s)",
                            request_id,
                            "drop" if _rc_cold else "compact",
                            _rc_stats["turns_compacted"],
                            _rc_stats["blocks"],
                            _rc_stats["words_before"],
                            _rc_stats["words_after"],
                            _rc_cold,
                        )
            except Exception as _rc_exc:  # never break the request on compaction
                logger.warning("[%s] reasoning compaction skipped: %s", request_id, _rc_exc)

        presend_event = self.pipeline_extensions.emit(
            PipelineStage.PRE_SEND,
            operation="proxy.request",
            request_id=request_id,
            provider="openai",
            model=model,
            messages=optimized_messages,
            tools=tools,
            headers=headers,
            metadata={"path": handler_path, "stream": stream},
        )
        if presend_event.messages is not None:
            optimized_messages = presend_event.messages
            body["messages"] = optimized_messages
        if presend_event.tools or _original_tools is not None:
            tools = presend_event.tools
            body["tools"] = tools
        if presend_event.headers is not None:
            headers = presend_event.headers
        # Consistency: recount BOTH endpoints with the provider tokenizer. An upstream
        # branch may have left original_tokens in the pipeline's char-estimator scale
        # (result.tokens_before), which mismatches optimized_tokens (provider tokenizer)
        # and yields impossible tok_after>tok_before. Recount original from the
        # pre-compression snapshot so the message delta is on one scale.
        original_tokens = tokenizer.count_messages(original_client_messages)
        optimized_tokens = tokenizer.count_messages(body["messages"])
        if tool_tokens_before_compaction > 0:
            try:
                tool_tokens_after_compaction = tokenizer.count_text(_json_debug_dumps(tools))
            except Exception:
                tool_tokens_after_compaction = tool_tokens_before_compaction
        if 0 < tool_tokens_after_compaction < tool_tokens_before_compaction:
            original_tokens += tool_tokens_before_compaction
            optimized_tokens += tool_tokens_after_compaction
        tokens_saved = max(0, original_tokens - optimized_tokens)

        # Turn hooks (opt-in extensions): a registered hook may rewrite the
        # outbound tools/messages before we send. on_response re-drive (below, in
        # the buffered response path) can't run on a stream, but on_request folds
        # can — so on streaming we run only stream-safe (fold-only) hooks, and
        # buffered runs all of them. Gated on the registry so it's a no-op when
        # none are registered; the net tool-schema token delta is recorded as a
        # saving.
        from headroom.proxy.turn_hooks import (
            TurnContext,
            registered_turn_hooks,
            run_request_hooks,
        )

        if registered_turn_hooks():
            _th_tools_before = body.get("tools")
            _th_tok_before = (
                tokenizer.count_text(json.dumps(_th_tools_before, default=str))
                if _th_tools_before
                else 0
            )
            _th_ctx = TurnContext(
                provider="openai",
                model=str(model),
                messages=body["messages"],
                tools=_th_tools_before,
                config=self.config,
            )
            # Snapshot messages BEFORE the hook (same tokenizer) so we can tell whether
            # the hook itself folded — comparing against optimized_tokens instead
            # conflates a real fold with a cross-estimator scale delta.
            try:
                _th_msg_before: int | None = tokenizer.count_messages(body["messages"])
            except Exception:
                _th_msg_before = None
            run_request_hooks(_th_ctx, stream_safe_only=stream)
            # A hook may either replace ctx.messages/ctx.tools or mutate them in
            # place (the contract allows both). Use object identity only to decide
            # whether body needs reassignment; measure the saving from the FINAL
            # tools object regardless, so an in-place shrink is still counted.
            if _th_ctx.messages is not body["messages"]:
                optimized_messages = _th_ctx.messages
                body["messages"] = optimized_messages
            if _th_ctx.tools is not _th_tools_before:
                tools = _th_ctx.tools
                body["tools"] = tools
            # Recount messages after the hook (it may fold in place), preserving the
            # tool-schema delta already folded into the headline above and keeping the
            # scale consistent with original_tokens (both provider tokenizer).
            try:
                _th_msg_after = tokenizer.count_messages(body["messages"])
                optimized_tokens = _th_msg_after
                if 0 < tool_tokens_after_compaction < tool_tokens_before_compaction:
                    optimized_tokens += tool_tokens_after_compaction
                tokens_saved = max(0, original_tokens - optimized_tokens)
                # Attribute to the hook ONLY when the hook itself reduced tokens.
                if _th_msg_before is not None and _th_msg_after < _th_msg_before:
                    transforms_applied.append("turn_hook")
            except Exception:
                logger.debug("turn-hook token re-count skipped", exc_info=True)
            _th_tok_after = (
                tokenizer.count_text(json.dumps(_th_ctx.tools, default=str)) if _th_ctx.tools else 0
            )
            _th_saved = max(0, _th_tok_before - _th_tok_after)
            if _th_saved > 0:
                tags["turn_hook_tools_saved_tokens"] = (
                    int(tags.get("turn_hook_tools_saved_tokens", 0) or 0) + _th_saved
                )
                transforms_applied.append(f"turn_hook:tools:{_th_saved}tok")

        # Compatibility shim: GPT-5 / o-series chat models REJECT the legacy
        # `max_tokens` ("Unsupported parameter … Use 'max_completion_tokens'
        # instead"); gpt-4o/4.1 accept `max_completion_tokens` too. openai-
        # compatible clients (opencode, older SDKs) still send `max_tokens`, so
        # translate it here — the proxy already owns the outbound body — and
        # those requests work unchanged. No-op when the caller already set
        # `max_completion_tokens`.
        _normalize_openai_max_tokens(
            body, backend_owns_translation=self.anthropic_backend is not None
        )

        # Output shaping (opt-in via HEADROOM_OUTPUT_SHAPER): verbosity steering
        # on the chat system message. Runs after every other body mutation so the
        # turn classifier sees the final messages, and respects the same bypass
        # as compression. OpenAI-compatible clients that route through
        # /v1/chat/completions (GitHub Copilot CLI, opencode, older SDKs) never
        # reached the shaper before, so they saw zero output savings (#2302).
        # Mutating `body` in place is sufficient here — the outbound request
        # serializes `body` fresh, so no body-mutation tracker is needed.
        if not _bypass:
            from headroom.proxy import runtime_env
            from headroom.proxy.output_savings import (
                assign_arm,
                conversation_key_from_body,
                stratum_key,
                stratum_label,
            )
            from headroom.proxy.output_shaper import (
                OutputShaperSettings,
                classify_turn,
                resolve_verbosity_level,
                shape_openai_chat_request,
            )

            _shaper_settings = OutputShaperSettings.from_env()
            if _shaper_settings.enabled:
                # Conversation-stable holdout: a whole conversation is treatment
                # or control, which keeps the A/B comparison clean and the
                # provider prefix cache stable (the steering block never flips
                # mid-conversation).
                _holdout = 0.0
                try:
                    _holdout = float(runtime_env.getenv("HEADROOM_OUTPUT_HOLDOUT", "0") or "0")
                except ValueError:
                    _holdout = 0.0
                _arm = assign_arm(conversation_key_from_body(body), _holdout)
                _turn_kind = classify_turn(body.get("messages", [])).value
                _stratum = stratum_key(
                    turn_kind=_turn_kind,
                    input_tokens=original_tokens,
                    model=model,
                    has_tools=bool(body.get("tools")),
                )
                # Carry (arm, stratum) on the transforms channel so the outcome
                # funnel feeds the output-savings ledger from the chat path too.
                transforms_applied.append(stratum_label(_arm, _stratum))
                if _arm == "treatment":
                    _level, _src = resolve_verbosity_level(_shaper_settings)
                    _shape_result = shape_openai_chat_request(
                        body, _shaper_settings, level_override=_level
                    )
                    if _shape_result.changed:
                        transforms_applied.extend(_shape_result.labels or [])
                        logger.info(
                            f"[{request_id}] OutputShaper(chat, L{_level}/{_src}): "
                            f"{_shape_result.labels}"
                        )

        # Route through LiteLLM/any-llm backend if configured
        if self.anthropic_backend is not None:
            try:
                if stream:
                    self.pipeline_extensions.emit(
                        PipelineStage.POST_SEND,
                        operation="proxy.request",
                        request_id=request_id,
                        provider="openai",
                        model=model,
                        messages=body["messages"],
                        tools=tools,
                        metadata={"path": handler_path, "stream": True},
                    )
                    # Streaming: use stream_openai_message() → SSE events
                    return await self._stream_openai_via_backend(
                        body,
                        headers,
                        model,
                        request_id,
                        start_time,
                        original_tokens,
                        optimized_tokens,
                        tokens_saved,
                        transforms_applied,
                        tags,
                        optimization_latency,
                        pipeline_timing=pipeline_timing,
                        waste_signals=waste_signals_dict,
                        prefix_tracker=openai_prefix_tracker,
                        optimized_messages=optimized_messages,
                    )
                else:
                    # Non-streaming: use send_openai_message() → JSON
                    backend_response = await self.anthropic_backend.send_openai_message(
                        body, headers
                    )
                    self.pipeline_extensions.emit(
                        PipelineStage.POST_SEND,
                        operation="proxy.request",
                        request_id=request_id,
                        provider="openai",
                        model=model,
                        messages=body["messages"],
                        tools=tools,
                        response=backend_response.body,
                        metadata={
                            "path": handler_path,
                            "stream": False,
                            "status_code": backend_response.status_code,
                        },
                    )
                    self.pipeline_extensions.emit(
                        PipelineStage.RESPONSE_RECEIVED,
                        operation="proxy.request",
                        request_id=request_id,
                        provider="openai",
                        model=model,
                        response=backend_response.body,
                        metadata={
                            "path": handler_path,
                            "stream": False,
                            "status_code": backend_response.status_code,
                        },
                    )

                    if backend_response.error:
                        return JSONResponse(
                            status_code=backend_response.status_code,
                            content=backend_response.body,
                        )

                    # CCR Response Handling: intercept headroom_retrieve
                    # tool calls server-side so a Bedrock/LiteLLM
                    # OpenAI-shape response doesn't propagate a tool_call
                    # the downstream caller (e.g. Strands) can't resolve.
                    # Mirrors the Anthropic handler block (anthropic.py
                    # ~1893-2034) but on the OpenAI provider shape.
                    #
                    # NO SILENT FALLBACK: per feedback_no_silent_fallbacks
                    # we re-raise on CCR errors instead of swallowing
                    # them. The Anthropic version still swallows for
                    # legacy reasons; align it in a follow-up.
                    # TODO(#realignment): align anthropic.py CCR block to
                    # re-raise on exception so both providers fail loud.
                    if (
                        self.ccr_response_handler
                        and backend_response.body
                        and backend_response.status_code == 200
                        and self.ccr_response_handler.has_ccr_tool_calls(
                            backend_response.body, "openai"
                        )
                    ):
                        logger.info(
                            f"[{request_id}] CCR: Detected retrieval tool call "
                            f"on backend path, handling via {self.anthropic_backend.name}"
                        )

                        # Continuation closure — delegates transport to
                        # the backend abstraction. We strip encoding
                        # headers for safety even though the backend
                        # owns transport (mirrors the Anthropic block).
                        async def api_call_fn(
                            msgs: list[dict[str, Any]],
                            tls: list[dict[str, Any]] | None,
                        ) -> dict[str, Any]:
                            continuation_body = {**body, "messages": msgs}
                            if tls is not None:
                                continuation_body["tools"] = tls

                            continuation_headers = {
                                k: v
                                for k, v in headers.items()
                                if k.lower()
                                not in (
                                    "content-encoding",
                                    "transfer-encoding",
                                    "accept-encoding",
                                    "content-length",
                                )
                            }

                            assert self.anthropic_backend is not None
                            logger.info(
                                f"[{request_id}] CCR: Issuing continuation via "
                                f"{self.anthropic_backend.name} backend "
                                f"({len(msgs)} messages)"
                            )
                            cont_resp = await self.anthropic_backend.send_openai_message(
                                continuation_body, continuation_headers
                            )
                            return cont_resp.body

                        try:
                            final_resp_json = await self.ccr_response_handler.handle_response(
                                backend_response.body,
                                optimized_messages,
                                tools,
                                api_call_fn,
                                provider="openai",
                            )
                            # Turn hooks (opt-in extensions) may inspect the turn
                            # or re-drive the model before we hand back the
                            # response. Inert when no hook is registered.
                            from headroom.proxy.turn_hooks import (
                                TurnContext,
                                run_response_hooks,
                            )

                            final_resp_json = await run_response_hooks(
                                TurnContext(
                                    provider="openai",
                                    model=str(model),
                                    messages=optimized_messages,
                                    tools=tools,
                                    config=self.config,
                                ),
                                final_resp_json,
                                api_call_fn,
                            )
                            backend_response.body = final_resp_json
                            logger.info(
                                f"[{request_id}] CCR: Retrieval handled "
                                "successfully on backend path"
                            )
                        except Exception as e:
                            import traceback

                            logger.error(
                                f"[{request_id}] CCR: Response handling failed on "
                                f"backend path: {e}\n"
                                f"Traceback: {traceback.format_exc()}"
                            )
                            # No silent fallback — fail loud per
                            # feedback_no_silent_fallbacks.md.
                            raise

                    # Extract usage from the FINAL backend body (after
                    # any CCR resolution) so the prefix tracker counts
                    # cache stats from the LAST upstream call.
                    total_latency = (time.time() - start_time) * 1000
                    usage = backend_response.body.get("usage", {})
                    # `.get(key, default)` only falls back when the key is
                    # absent; a present-but-null count (some OpenAI-compatible
                    # backends emit these on a stopped/empty turn) would return
                    # None and crash the downstream `max(...)` arithmetic and the
                    # int-typed outcome/metrics. `_usage_int` coerces both cases,
                    # matching the streaming path and the guarded cache keys below
                    # (same class as the gemini fix in #2347).
                    output_tokens = _usage_int(usage.get("completion_tokens"))
                    total_input_tokens = _usage_int(usage.get("prompt_tokens")) or optimized_tokens

                    # Cache stats: prefer the Anthropic/Bedrock top-level
                    # keys when present (authoritative). Fall back to
                    # OpenAI's `prompt_tokens_details.cached_tokens` only
                    # if the top-level keys are absent/zero.
                    cache_read_tokens = usage.get("cache_read_input_tokens", 0) or 0
                    cache_creation_input_tokens = usage.get("cache_creation_input_tokens", 0) or 0
                    if cache_read_tokens == 0:
                        prompt_details = usage.get("prompt_tokens_details") or {}
                        cache_read_tokens = prompt_details.get("cached_tokens", 0) or 0

                    # Bedrock reports cache creation directly. Only infer
                    # when no explicit count is available. Skip inference
                    # entirely when upstream omitted prompt_tokens.
                    if cache_creation_input_tokens > 0:
                        cache_write_tokens = cache_creation_input_tokens
                    elif "prompt_tokens" in usage:
                        cache_write_tokens = _infer_openai_cache_write_tokens(
                            total_input_tokens,
                            cache_read_tokens,
                        )
                    else:
                        cache_write_tokens = 0

                    uncached_input_tokens = max(
                        0, total_input_tokens - cache_read_tokens - cache_write_tokens
                    )

                    # Cache-TTL learning seam: attribute this turn's cache outcome
                    # (hit / ttl_expiry / prefix_change) and record it BEFORE
                    # update_from_response overwrites the prior-turn state. Feeds the
                    # offline TTL learner (HEADROOM_CACHE_TTL_LEARN); best-effort.
                    try:
                        from headroom.cache.ttl_observations import (
                            observations_enabled,
                            record_cache_observation,
                        )

                        # Only run the extra attribution when learning is enabled —
                        # otherwise this path adds nothing over the base proxy.
                        if observations_enabled() and hasattr(
                            openai_prefix_tracker, "classify_cache_miss"
                        ):
                            record_cache_observation(
                                provider="openai",
                                model=model,
                                attribution=openai_prefix_tracker.classify_cache_miss(
                                    cache_read_tokens=cache_read_tokens,
                                    current_forwarded_messages=optimized_messages,
                                ),
                            )
                    except Exception:
                        pass

                    openai_prefix_tracker.update_from_response(
                        cache_read_tokens=cache_read_tokens,
                        cache_write_tokens=cache_write_tokens,
                        messages=optimized_messages,
                    )

                    await self._record_request_outcome(
                        RequestOutcome(
                            request_id=request_id,
                            provider=self.anthropic_backend.name,
                            model=model,
                            original_tokens=original_tokens,
                            # Local count, same tokenizer as original_tokens, so
                            # every delta built from the pair is coherent. The
                            # provider's own number rides along separately for
                            # billing — see RequestOutcome's field docs.
                            optimized_tokens=optimized_tokens,
                            provider_input_tokens=total_input_tokens,
                            output_tokens=output_tokens,
                            tokens_saved=tokens_saved,
                            attempted_input_tokens=optimized_tokens + tokens_saved,
                            cache_read_tokens=cache_read_tokens,
                            cache_write_tokens=cache_write_tokens,
                            uncached_input_tokens=uncached_input_tokens,
                            total_latency_ms=total_latency,
                            overhead_ms=optimization_latency,
                            pipeline_timing=pipeline_timing,
                            waste_signals=waste_signals_dict,
                            transforms_applied=tuple(transforms_applied),
                            num_messages=len(body.get("messages", [])),
                            tags=tags or {},
                            request_messages=body.get("messages")
                            if getattr(self.config, "log_full_messages", False)
                            else None,
                            client=client,
                        )
                    )

                    if tokens_saved > 0:
                        logger.info(
                            f"[{request_id}] {model}: {original_tokens:,} → {optimized_tokens:,} "
                            f"(saved {tokens_saved:,} tokens) via {self.anthropic_backend.name}"
                        )

                    return JSONResponse(
                        status_code=backend_response.status_code,
                        content=backend_response.body,
                    )
            except Exception as e:
                logger.error(f"[{request_id}] Backend error: {e}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "message": str(e),
                            "type": "api_error",
                            "code": "backend_error",
                        }
                    },
                )

        # Direct OpenAI API (no backend configured)
        url = build_copilot_upstream_url(
            upstream_base_url or self.OPENAI_API_URL,
            handler_path,
        )
        url = _append_request_query(url, request.url.query)

        try:
            if stream:
                # Request usage stats for accurate token counting instead of
                # byte-based estimation — without overriding an explicit client
                # choice (see _apply_stream_usage_option).
                _apply_stream_usage_option(body)

                self.pipeline_extensions.emit(
                    PipelineStage.POST_SEND,
                    operation="proxy.request",
                    request_id=request_id,
                    provider="openai",
                    model=model,
                    messages=body["messages"],
                    tools=tools,
                    metadata={"path": handler_path, "stream": True},
                )
                return await self._stream_response(
                    url,
                    headers,
                    body,
                    "openai",
                    model,
                    request_id,
                    original_tokens,
                    optimized_tokens,
                    tokens_saved,
                    transforms_applied,
                    tags,
                    optimization_latency,
                    pipeline_timing=pipeline_timing,
                    prefix_tracker=openai_prefix_tracker,
                    outcome_provider=openai_chat_outcome_provider,
                )
            else:
                headers = await apply_copilot_api_auth(headers, url=url)
                response = await self._retry_request("POST", url, headers, body)

                # Turn hooks: a registered extension may re-drive this turn
                # (e.g. resolve a tool the model asked to load) before we treat
                # the response as final. Buffered path only; no-op when no hook
                # is registered.
                from headroom.proxy.turn_hooks import (
                    TurnContext as _TurnContext,
                )
                from headroom.proxy.turn_hooks import (
                    registered_turn_hooks as _registered_turn_hooks,
                )
                from headroom.proxy.turn_hooks import (
                    run_response_hooks,
                )

                if _registered_turn_hooks() and response.status_code == 200:
                    try:
                        _hook_resp_json = response.json()
                    except (ValueError, json.JSONDecodeError):
                        _hook_resp_json = None
                    if isinstance(_hook_resp_json, dict):
                        _hook_ctx = _TurnContext(
                            provider="openai",
                            model=str(model),
                            messages=body["messages"],
                            tools=body.get("tools"),
                            config=self.config,
                        )

                        async def _hook_call_model(_msgs):
                            body["messages"] = _msgs
                            _r = await self._retry_request("POST", url, headers, body)
                            return _r.json()

                        _hook_final = await run_response_hooks(
                            _hook_ctx, _hook_resp_json, _hook_call_model
                        )
                        if _hook_final is not _hook_resp_json:
                            response = httpx.Response(
                                status_code=200,
                                headers={
                                    k: v
                                    for k, v in response.headers.items()
                                    if k.lower() not in ("content-encoding", "content-length")
                                },
                                content=json.dumps(_hook_final).encode(),
                            )

                self.pipeline_extensions.emit(
                    PipelineStage.POST_SEND,
                    operation="proxy.request",
                    request_id=request_id,
                    provider="openai",
                    model=model,
                    messages=body["messages"],
                    tools=tools,
                    response=response,
                    metadata={
                        "path": handler_path,
                        "stream": False,
                        "status_code": response.status_code,
                    },
                )
                self.pipeline_extensions.emit(
                    PipelineStage.RESPONSE_RECEIVED,
                    operation="proxy.request",
                    request_id=request_id,
                    provider="openai",
                    model=model,
                    response=response,
                    metadata={
                        "path": handler_path,
                        "stream": False,
                        "status_code": response.status_code,
                    },
                )

                # Full diagnostic dump on upstream errors (OpenAI handler)
                if response.status_code >= 400:
                    try:
                        err_body = response.json()
                        err_msg = err_body.get("error", {}).get("message", "")
                        err_type = err_body.get("error", {}).get("type", "")
                    except Exception:
                        err_body = {"raw": response.text[:2000]}
                        err_msg = str(response.text[:500])
                        err_type = "parse_error"

                    logger.warning(
                        f"[{request_id}] UPSTREAM_ERROR "
                        f"status={response.status_code} "
                        f"error_type={err_type} "
                        f"error_msg={err_msg!r} "
                        f"model={model} "
                        f"compressed={'yes' if transforms_applied else 'no'} "
                        f"transforms={transforms_applied} "
                        f"original_tokens={original_tokens} "
                        f"optimized_tokens={optimized_tokens} "
                        f"message_count={len(body.get('messages', []))} "
                        f"stream={stream}"
                    )

                    # Diagnostic dump — OFF by default (can contain cleartext
                    # prompt/tool/system content). Opt in via HEADROOM_DEBUG_DUMP
                    # (=1 redacted, =full with content); never in stateless mode.
                    dump_mode = _debug_dump_mode(self.config)
                    if dump_mode != "off":
                        try:
                            from headroom import paths as _hr_paths

                            debug_dir = _hr_paths.debug_400_dir()
                            debug_dir.mkdir(parents=True, exist_ok=True)
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            debug_file = debug_dir / f"{ts}_{request_id}.json"

                            safe_headers = {}
                            for k, v in headers.items():
                                if k.lower() in ("x-api-key", "authorization"):
                                    safe_headers[k] = v[:12] + "..." if v else ""
                                else:
                                    safe_headers[k] = v

                            redact = dump_mode == "redacted"
                            messages_sent = body.get("messages")
                            original_dump: Any = (
                                original_messages
                                if original_messages is not body.get("messages")
                                else "__same_as_sent__"
                            )
                            tools_sent = body.get("tools")
                            system_prompt = body.get("system")
                            if redact:
                                messages_sent = _redact_debug_value(messages_sent)
                                if original_dump != "__same_as_sent__":
                                    original_dump = _redact_debug_value(original_dump)
                                tools_sent = _redact_debug_value(tools_sent)
                                system_prompt = _redact_debug_value(system_prompt)

                            debug_payload = {
                                "request_id": request_id,
                                "timestamp": datetime.now().isoformat(),
                                "dump_mode": dump_mode,
                                "status_code": response.status_code,
                                "error_response": err_body,
                                "model": model,
                                "stream": stream,
                                "headers": safe_headers,
                                "compression": {
                                    "was_compressed": bool(transforms_applied),
                                    "transforms": transforms_applied,
                                    "original_tokens": original_tokens,
                                    "optimized_tokens": optimized_tokens,
                                    "tokens_saved": tokens_saved,
                                    "compression_failed": _compression_failed,
                                },
                                "tools_sent": tools_sent,
                                "tool_count": len(body.get("tools") or []),
                                "original_tool_count": len(_original_tools or []),
                                "messages_sent": messages_sent,
                                "message_count": len(body.get("messages", [])),
                                "original_messages": original_dump,
                                "original_message_count": len(original_messages),
                                "system_prompt": system_prompt,
                            }

                            with open(debug_file, "w") as f:
                                json.dump(debug_payload, f, indent=2, default=str)

                            logger.warning(f"[{request_id}] Debug dump ({dump_mode}): {debug_file}")
                        except Exception as dump_err:
                            logger.error(f"[{request_id}] Failed to write debug dump: {dump_err}")

                total_latency = (time.time() - start_time) * 1000

                total_input_tokens = optimized_tokens  # fallback
                output_tokens = 0
                cache_read_tokens = 0
                resp_json = None
                try:
                    resp_json = response.json()
                    usage = resp_json.get("usage", {})
                    # Coerce present-but-null counts: the arithmetic below
                    # (`_infer_openai_cache_write_tokens`, `max(...)`) runs
                    # outside this try, so a null `prompt_tokens`/`cached_tokens`
                    # would otherwise raise an uncaught TypeError and 500 the
                    # request (same class as the gemini fix in #2347).
                    total_input_tokens = _usage_int(usage.get("prompt_tokens")) or optimized_tokens
                    output_tokens = _usage_int(usage.get("completion_tokens"))
                    # OpenAI returns cached_tokens in prompt_tokens_details
                    # These are charged at 50% of the input price
                    prompt_details = usage.get("prompt_tokens_details") or {}
                    cache_read_tokens = _usage_int(prompt_details.get("cached_tokens"))
                except (KeyError, TypeError, AttributeError) as e:
                    logger.debug(
                        f"[{request_id}] Failed to extract cached tokens from OpenAI response: {e}"
                    )

                # Update prefix cache tracker for next turn
                cache_write_tokens = _infer_openai_cache_write_tokens(
                    total_input_tokens,
                    cache_read_tokens,
                )
                # Cache-TTL learning seam (see the /v1/chat path above): record the
                # cache-outcome attribution before update_from_response. Best-effort.
                try:
                    from headroom.cache.ttl_observations import (
                        observations_enabled,
                        record_cache_observation,
                    )

                    if observations_enabled() and hasattr(
                        openai_prefix_tracker, "classify_cache_miss"
                    ):
                        record_cache_observation(
                            provider="openai",
                            model=model,
                            attribution=openai_prefix_tracker.classify_cache_miss(
                                cache_read_tokens=cache_read_tokens,
                                current_forwarded_messages=optimized_messages,
                            ),
                        )
                except Exception:
                    pass

                openai_prefix_tracker.update_from_response(
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                    messages=optimized_messages,
                )

                # OpenAI has no write penalty — uncached = total - cached
                uncached_input_tokens = max(0, total_input_tokens - cache_read_tokens)

                # Cost is recorded exactly once by the outcome funnel below
                # (_record_request_outcome -> emit_request_outcome -> cost_tracker.
                # record_tokens); recording here too double-counted spend + budget.

                # Memory: handle memory tool calls in OpenAI Chat Completions response.
                # After executing tools, send a continuation request so the model
                # can produce a final user-facing response (not just tool_calls).
                if (
                    self.memory_handler
                    and memory_user_id
                    and resp_json
                    and response.status_code == 200
                    and self.memory_handler.has_memory_tool_calls(resp_json, "openai")
                ):
                    try:
                        tool_results = await self.memory_handler.handle_memory_tool_calls(
                            resp_json,
                            memory_user_id,
                            "openai",
                            request_context=memory_request_ctx,
                        )
                        if tool_results:
                            # Build continuation: original messages + assistant tool_calls + tool results
                            assistant_msg = resp_json.get("choices", [{}])[0].get("message", {})
                            continuation_messages = list(optimized_messages)
                            continuation_messages.append(assistant_msg)
                            continuation_messages.extend(tool_results)

                            continuation_body = {
                                **body,
                                "messages": continuation_messages,
                            }

                            cont_response = await self._retry_request(
                                "POST", url, headers, continuation_body
                            )
                            if cont_response.status_code == 200:
                                resp_json = cont_response.json()
                                response = cont_response

                            logger.info(
                                f"[{request_id}] Memory: Handled {len(tool_results)} "
                                f"tool call(s) with continuation for user {memory_user_id}"
                            )
                    except Exception as e:
                        logger.warning(f"[{request_id}] Memory tool handling failed: {e}")

                # Cache response under the SAME key it was looked up by:
                # cache_lookup_messages is the raw pre-mutation snapshot, not
                # the live (hooked) `messages` (#327).
                if self.cache and response.status_code == 200:
                    await self.cache.set(
                        cache_lookup_messages,
                        model,
                        response.content,
                        dict(response.headers),
                        tokens_saved,
                        **cache_key_fields,
                    )

                # Capture Codex rate-limit window data from response headers
                from headroom.subscription.codex_rate_limits import (
                    get_codex_rate_limit_state,
                )

                get_codex_rate_limit_state().update_from_headers(dict(response.headers))

                # Tag the metric/log with auth_mode + endpoint so the
                # dashboard can break down by client class (PAYG vs
                # subscription vs OAuth) without re-classifying.
                _auth_mode_chat = getattr(request.state, "auth_mode", None)
                _chat_log_tags = {
                    **(tags or {}),
                    "auth_mode": _auth_mode_chat.value if _auth_mode_chat else "payg",
                    "endpoint": "chat_completions",
                }

                # OpenAI Chat direct (non-backend) non-streaming.
                # Fallback denominator: full pre-comp size — see
                # equivalent note at the backend-routed sibling.
                from headroom.proxy.helpers import compute_turn_id

                await self._record_request_outcome(
                    RequestOutcome(
                        request_id=request_id,
                        provider=openai_chat_outcome_provider,
                        model=model,
                        status_code=response.status_code,
                        original_tokens=original_tokens,
                        # Same-tokenizer pair; provider count carried separately.
                        optimized_tokens=optimized_tokens,
                        provider_input_tokens=total_input_tokens,
                        output_tokens=output_tokens,
                        tokens_saved=tokens_saved,
                        attempted_input_tokens=optimized_tokens + tokens_saved,
                        cache_read_tokens=cache_read_tokens,
                        cache_write_tokens=cache_write_tokens,
                        uncached_input_tokens=uncached_input_tokens,
                        cache_inferred=True,
                        total_latency_ms=total_latency,
                        overhead_ms=optimization_latency,
                        pipeline_timing=pipeline_timing,
                        waste_signals=waste_signals_dict,
                        transforms_applied=tuple(transforms_applied),
                        num_messages=len(body.get("messages", [])),
                        tags=_chat_log_tags,
                        turn_id=compute_turn_id(model, body.get("system"), body.get("messages")),
                        request_messages=body.get("messages")
                        if getattr(self.config, "log_full_messages", False)
                        else None,
                        client=client,
                    )
                )

                if tokens_saved > 0:
                    logger.info(
                        f"[{request_id}] {model}: {original_tokens:,} → {optimized_tokens:,} "
                        f"(saved {tokens_saved:,} tokens)"
                    )

                # Remove compression headers since httpx already decompressed the response
                response_headers = _sanitize_forwarded_response_headers(response.headers)

                # Inject Headroom compression metrics (for SaaS metering)
                response_headers["x-headroom-tokens-before"] = str(original_tokens)
                response_headers["x-headroom-tokens-after"] = str(optimized_tokens)
                response_headers["x-headroom-tokens-saved"] = str(tokens_saved)
                response_headers["x-headroom-model"] = model
                if transforms_applied:
                    response_headers["x-headroom-transforms"] = ",".join(
                        header_safe_transforms(transforms_applied)
                    )
                if cache_read_tokens > 0:
                    response_headers["x-headroom-cached"] = "true"
                if _compression_failed:
                    response_headers["x-headroom-compression-failed"] = "true"

                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=response_headers,
                )
        except Exception as e:
            await self.metrics.record_failed(provider=openai_chat_outcome_provider)
            # Log full error details internally for debugging
            logger.error(f"[{request_id}] OpenAI request failed: {type(e).__name__}: {e}")
            # Return sanitized error message to client (don't expose internal details)
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": "An error occurred while processing your request. Please try again.",
                        "type": "server_error",
                        "code": "proxy_error",
                    }
                },
            )

    async def handle_openai_responses(
        self,
        request: Request,
    ) -> Response | StreamingResponse:
        """Handle OpenAI /v1/responses endpoint (new Responses API).

        The Responses API differs from /v1/chat/completions:
        - Input: `input` (string or array) instead of `messages`
        - System: `instructions` instead of system message
        - Output: `output[]` array instead of `choices[].message`
        - State: `previous_response_id` for multi-turn
        - Built-in tools: web_search, file_search, code_interpreter
        """
        from fastapi import HTTPException
        from fastapi.responses import JSONResponse, Response, StreamingResponse

        from headroom.proxy.body_forwarding import BodyMutationTracker
        from headroom.proxy.helpers import (
            MAX_REQUEST_BODY_SIZE,
            read_request_json_with_bytes,
        )
        from headroom.utils import extract_user_query

        start_time = time.time()
        request_id = await self._next_request_id()

        # Phase F PR-F1: classify auth mode at request entry. The result
        # is stored on `request.state` so downstream handlers (cache
        # gates, header injection, lossy-compressor gates) read it
        # without re-classifying. Pure function, well under 10us.
        auth_mode = classify_auth_mode(request.headers)
        request.state.auth_mode = auth_mode
        logger.debug(f"[{request_id}] auth_mode_classified mode={auth_mode.value}")

        # Check request body size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": f"Request body too large. Maximum size is {MAX_REQUEST_BODY_SIZE // (1024 * 1024)}MB",
                        "type": "invalid_request_error",
                        "code": "request_too_large",
                    }
                },
            )

        # Parse request. Keep the original (post-content-decoding) bytes so a
        # request we never mutate is forwarded byte-for-byte instead of being
        # canonically re-serialized. Codex Desktop posts whose body differs
        # from our re-serialization are rejected upstream with HTTP 400
        # (#1542); byte-faithful passthrough avoids that.
        try:
            body, original_body_bytes = await read_request_json_with_bytes(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Invalid request body: {e!s}",
                        "type": "invalid_request_error",
                        "code": "invalid_json",
                    }
                },
            )

        model = body.get("model", "unknown")
        stream = body.get("stream", False)
        body_mutation_tracker = BodyMutationTracker()
        _bypass = self._headroom_bypass_enabled(request.headers)
        if _bypass:
            logger.info(
                "[%s] Responses passthrough reason=bypass_header mutation=disabled",
                request_id,
            )

        from headroom.proxy.helpers import capture_codex_wire_debug

        capture_codex_wire_debug(
            "http_inbound_request",
            request_id=request_id,
            transport="http",
            direction="client_to_headroom",
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers.items()),
            body=body,
            metadata={"path": request.url.path, "stream": stream},
        )

        # /v1/responses uses provider-specific CompressionUnit extraction
        # below, then routes mutable text through ContentRouter. The
        # standalone Rust proxy has native item-aware handling, but the
        # Python CLI runtime does not run that proxy today. We synthesise a
        # minimal `messages` list purely for downstream memory injection and
        # telemetry; list-typed `input` is consulted directly by the unit
        # extraction helpers.
        input_data = body.get("input", "")
        instructions = body.get("instructions")

        messages: list[dict[str, Any]] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        if isinstance(input_data, str):
            messages.append({"role": "user", "content": input_data})

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        # The parsed request body has already been content-decoded. Remove
        # entity headers that described the client-to-proxy wire body.
        headers.pop("content-encoding", None)
        headers.pop("transfer-encoding", None)
        # Strip accept-encoding so httpx negotiates its own encoding.
        # Cloudflare Workers forward "br, zstd" which OpenAI may honor;
        # if httpx lacks brotli support the response body is undecipherable → 502.
        headers.pop("accept-encoding", None)
        # Strip content-encoding: read_request_json_with_bytes already decoded
        # the inbound body (zstd/gzip/deflate/br), so the bytes we forward are
        # plain. Leaving a stale content-encoding header makes the upstream try
        # to decompress already-decoded JSON and reject it with HTTP 400 (#1542).
        headers.pop("content-encoding", None)
        tags = extract_tags(headers)
        client = classify_client(headers)

        # Learn from the original client payload before memory context or
        # compression mutates it. This mirrors the Anthropic ingestion path.
        await self._observe_openai_responses_traffic(body, request_id=request_id)

        # PR-A5 (P5-49): strip internal x-headroom-* from upstream-bound
        # headers AFTER `_extract_tags` reads them. Memory user-id reads
        # `request.headers` below.
        from headroom.proxy.helpers import (
            _strip_internal_headers,
            log_outbound_headers,
            merge_extra_headers,
        )

        _pre_strip_count_resp = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        headers = merge_extra_headers(headers, self.config.openai_extra_headers)
        # Mirror the WS handler: never forward Codex's client-only lite header
        # upstream. OpenAI rejects newer Codex models when it leaks, and the HTTP
        # POST path (unlike the WS path) otherwise forwards request headers verbatim.
        headers = {
            key: value
            for key, value in headers.items()
            if key.lower() != _CODEX_RESPONSES_LITE_HEADER
        }
        log_outbound_headers(
            forwarder="openai_responses",
            stripped_count=_pre_strip_count_resp,
            request_id=request_id,
        )
        headers, is_chatgpt_auth = _resolve_codex_routing_headers(headers)
        if is_chatgpt_auth:
            client = "codex"
        if _ensure_chatgpt_responses_store_false(body, is_chatgpt_auth=is_chatgpt_auth):
            logger.info(f"[{request_id}] Responses: forced store=false for ChatGPT auth")
        responses_memory_tools_allowed = _allow_responses_memory_tools(is_chatgpt_auth)

        # PR-A6 (P5-50, preps P0-6): session-sticky `OpenAI-Beta` merge
        # for /v1/responses. Compute a session_id off the same store the
        # chat handler uses so multi-endpoint clients within one
        # conversation share the sticky-token set.
        _responses_session_id = self.session_tracker_store.compute_session_id(
            request, model, messages
        )
        from headroom.proxy.helpers import (
            get_session_beta_tracker as _get_session_beta_tracker_resp,
        )
        from headroom.proxy.helpers import (
            log_beta_header_merge as _log_beta_header_merge_resp,
        )

        _client_resp_beta = headers.get("openai-beta")
        _client_resp_beta_count = (
            len([t for t in (_client_resp_beta or "").split(",") if t.strip()])
            if _client_resp_beta
            else 0
        )
        _sticky_resp_beta = _get_session_beta_tracker_resp().record_and_get_sticky_betas(
            provider="openai",
            session_id=_responses_session_id,
            client_value=_client_resp_beta,
        )
        _sticky_resp_beta_count = (
            len([t for t in _sticky_resp_beta.split(",") if t.strip()]) if _sticky_resp_beta else 0
        )
        if _sticky_resp_beta and _sticky_resp_beta != (_client_resp_beta or ""):
            headers["openai-beta"] = _sticky_resp_beta
        _log_beta_header_merge_resp(
            provider="openai",
            session_id=_responses_session_id,
            client_betas_count=_client_resp_beta_count,
            sticky_betas_count=_sticky_resp_beta_count,
            headroom_added=[],
            request_id=request_id,
        )

        # Memory: Get user ID when memory is enabled. Reads `request.headers`
        # directly because `headers` was stripped of `x-headroom-*` (PR-A5).
        memory_user_id: str | None = None
        memory_request_ctx = None
        if self.memory_handler:
            memory_user_id = request.headers.get(
                "x-headroom-user-id",
                os.environ.get("USER", os.environ.get("USERNAME", "default")),
            )
            from headroom.memory.storage_router import (
                RequestContext as _MemRequestContext,
            )
            from headroom.memory.storage_router import (
                extract_system_prompt as _extract_sys_prompt,
            )

            memory_request_ctx = _MemRequestContext(
                headers=dict(request.headers),
                system_prompt=_extract_sys_prompt(body),
                base_user_id=memory_user_id,
                project_root_override=(
                    getattr(self.memory_handler.config, "project_root_override", "") or None
                ),
            )

        # Rate limiting
        if self.rate_limiter:
            rate_key = headers.get("authorization", "default")[:20]
            allowed, wait_seconds = await self.rate_limiter.check_request(rate_key)
            if not allowed:
                await self.metrics.record_rate_limited(provider="openai")
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limited. Retry after {wait_seconds:.1f}s",
                )

        # Token counting on converted messages (offloaded off the event loop — GH #1701)
        tokenizer, original_tokens = await self._count_tokens_offloaded(model, messages)

        # Defaults below feed downstream telemetry and memory injection.
        # If optimization remains enabled, the Responses payload is compressed
        # later through `_compress_openai_responses_payload`.
        optimized_messages = messages
        optimized_tokens = original_tokens
        tokens_saved = 0
        # Eligible-only denominator for the active compression ratio.
        # Populated by `_compress_openai_responses_payload` if it runs;
        # stays 0 on bypass / passthrough paths so we don't fabricate a
        # denominator we haven't earned.
        attempted_input_tokens = 0
        transforms_applied: list[str] = []
        optimization_latency = (time.time() - start_time) * 1000

        # Memory: inject context and tools for Responses API requests.
        # Gated on MemoryDecision — uniformly respects bypass across all
        # five injection sites. The Responses path is the only one that
        # injects BEFORE compression today (sites 1/2/3 inject after);
        # bringing this into alignment is queued as a follow-up
        # (FUTURE: move context injection to post-compression for
        # uniform "memory text rides uncompressed across all
        # handlers" semantics — separate PR with cache-stability tests).
        from headroom.proxy.helpers import get_memory_injection_mode
        from headroom.proxy.memory_decision import MemoryDecision
        from headroom.proxy.memory_query import MemoryQuery

        responses_memory_decision = MemoryDecision.decide(
            headers=request.headers,
            memory_handler=self.memory_handler,
            memory_user_id=memory_user_id,
            mode_name=get_memory_injection_mode(),
        )
        responses_memory_decision.apply_to_tags(tags)
        if responses_memory_decision.inject:
            try:
                # Memory context now routes exclusively to the live-zone tail
                # (latest non-frozen user item). Instructions are part of the
                # cache hot zone and must never be mutated — invariant I2.
                # See REALIGNMENT/03-phase-A-lockdown.md PR-A2.
                if self.memory_handler.config.inject_context:
                    try:
                        memory_context = await asyncio.wait_for(
                            self.memory_handler.search_and_format_context(
                                memory_user_id,
                                optimized_messages,
                                request_context=memory_request_ctx,
                                query=MemoryQuery.from_messages(optimized_messages),
                            ),
                            timeout=RESPONSES_CONTEXT_SEARCH_TIMEOUT_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        memory_context = None
                        logger.info(
                            f"[{request_id}] Memory context lookup exceeded "
                            f"{RESPONSES_CONTEXT_SEARCH_TIMEOUT_SECONDS:.1f}s; continuing without it"
                        )
                    if memory_context:
                        from headroom.proxy.helpers import (
                            append_text_to_latest_user_input_item,
                            get_memory_injection_mode,
                            log_memory_injection,
                        )

                        injection_mode = get_memory_injection_mode()
                        user_query = extract_user_query(optimized_messages) or ""
                        if injection_mode == "disabled":
                            log_memory_injection(
                                request_id=request_id,
                                session_id=None,
                                decision="skipped_disabled",
                                bytes_injected=0,
                                query=user_query,
                            )
                        else:
                            # Route into body["input"] (the canonical Responses API
                            # field) targeting the latest user item's first text
                            # block. body["instructions"] (cache hot zone) is left
                            # untouched.
                            current_input = body.get("input")
                            if isinstance(current_input, str):
                                # String input: append to it. The string IS the
                                # latest user content; appending here is the
                                # equivalent of the live-zone tail.
                                body["input"] = (
                                    current_input + "\n\n" + memory_context
                                    if current_input
                                    else memory_context
                                )
                                body_mutation_tracker.mark_mutated("responses_memory_context")
                                log_memory_injection(
                                    request_id=request_id,
                                    session_id=None,
                                    decision="injected_live_zone_tail_string",
                                    bytes_injected=len(memory_context),
                                    query=user_query,
                                    tags=tags,
                                )
                            elif isinstance(current_input, list):
                                new_input, bytes_appended = append_text_to_latest_user_input_item(
                                    current_input, memory_context
                                )
                                if bytes_appended > 0:
                                    body["input"] = new_input
                                    body_mutation_tracker.mark_mutated("responses_memory_context")
                                    log_memory_injection(
                                        request_id=request_id,
                                        session_id=None,
                                        decision="injected_live_zone_tail",
                                        bytes_injected=bytes_appended,
                                        query=user_query,
                                        tags=tags,
                                    )
                                else:
                                    log_memory_injection(
                                        request_id=request_id,
                                        session_id=None,
                                        decision="no_eligible_user_item",
                                        bytes_injected=0,
                                        query=user_query,
                                    )
                            else:
                                log_memory_injection(
                                    request_id=request_id,
                                    session_id=None,
                                    decision="no_input_field",
                                    bytes_injected=0,
                                    query=user_query,
                                )

                # Inject memory tools (Responses API format) — PR-A7 (P0-6).
                # Pre-convert the Chat-Completions schema to Responses API
                # format BEFORE handing to the sticky tracker so the
                # canonical bytes pinned in turn 1 already reflect the
                # exact bytes that will hit the wire.
                from headroom.proxy.helpers import (
                    apply_session_sticky_memory_tools as _apply_sticky_mem_tools_resp,
                )

                memory_tool_defs_chat = (
                    self.memory_handler.compute_memory_tool_definitions("openai")
                    if self.memory_handler.config.inject_tools and responses_memory_tools_allowed
                    else []
                )
                memory_tool_defs_responses: list[dict[str, Any]] = []
                for t in memory_tool_defs_chat:
                    if t.get("type") == "function" and "function" in t:
                        fn = t["function"]
                        memory_tool_defs_responses.append(
                            {
                                "type": "function",
                                "name": fn.get("name"),
                                "description": fn.get("description", ""),
                                "parameters": fn.get("parameters", {}),
                            }
                        )
                    else:
                        memory_tool_defs_responses.append(t)

                if (
                    responses_memory_tools_allowed
                    and _responses_request_allows_memory_tool_continuation(body)
                ):
                    resp_tools = body.get("tools") or []
                    resp_tools, mem_tools_injected = _apply_sticky_mem_tools_resp(
                        provider="openai",
                        session_id=_responses_session_id,
                        request_id=request_id,
                        existing_tools=resp_tools,
                        memory_tools_to_inject=memory_tool_defs_responses,
                        inject_this_turn=bool(self.memory_handler.config.inject_tools),
                    )
                    if mem_tools_injected:
                        body["tools"] = resp_tools
                        body_mutation_tracker.mark_mutated("responses_memory_tools")
                        logger.info(
                            f"[{request_id}] Memory: Injected memory tools (openai/responses)"
                        )
                        if _ensure_responses_store_for_memory_tools(
                            body,
                            memory_tools_injected=True,
                        ):
                            body_mutation_tracker.mark_mutated("responses_memory_store")
                            logger.info(
                                f"[{request_id}] Memory: forced store=true for Responses memory tool continuation"
                            )
                elif self.memory_handler.config.inject_tools:
                    logger.info(
                        "[%s] Memory: skipped Responses memory tools because client set store=false",
                        request_id,
                    )
            except Exception as e:
                logger.warning(f"[{request_id}] Memory injection failed (responses): {e}")
        elif self.memory_handler and memory_user_id and _bypass:
            logger.info(
                "[%s] Responses memory passthrough reason=bypass_header",
                request_id,
            )

        # /v1/responses is OpenAI-specific (Codex) — always routes direct.
        # LiteLLM/AnyLLM backends use /v1/chat/completions or /v1/messages.
        if self.anthropic_backend is not None:
            logger.debug(
                f"[{request_id}] /v1/responses always routes to OpenAI direct "
                f"(backend '{self.anthropic_backend.name}' not used for Responses API)"
            )

        # Route to correct endpoint based on auth mode.
        # ChatGPT session auth (codex login) uses chatgpt.com, not api.openai.com.
        if is_chatgpt_auth:
            url = codex_responses_http_url()
        else:
            upstream_base_url = _resolve_openai_upstream_base(request.headers)
            handler_path = (
                _resolve_openai_handler_path(request.headers, handler_path=_OPENAI_RESPONSES_PATH)
                if upstream_base_url is not None
                else "/v1/responses"
            )
            url = build_copilot_upstream_url(
                upstream_base_url or self.OPENAI_API_URL,
                handler_path,
            )
            url = _append_request_query(url, request.url.query)

        # The standalone Rust proxy has native /v1/responses item handling,
        # but the default CLI runtime is this Python proxy. Compress the
        # Python runtime path here by extracting mutable Responses text into
        # CompressionUnits and routing them through ContentRouter. Policy
        # gating already happened upstream (auth_mode classify,
        # CompressionPolicy resolve at request entry).
        if self.config.optimize and not _bypass:
            try:
                (
                    body,
                    _modified,
                    _tokens_saved,
                    _transforms,
                    _reason,
                    _bytes_before,
                    _bytes_after,
                    _attempted_tokens,
                    _compression_timing,
                ) = await self._compress_openai_responses_payload_in_executor(
                    body,
                    model=model,
                    request_id=request_id,
                    client=client,
                )
                attempted_input_tokens = int(_attempted_tokens)
                if _transforms:
                    # Record transform labels even when the payload bytes are
                    # unchanged: control-arm output-shaper labels
                    # (output_shaper:control:*) must reach the outcome ledger
                    # for A/B baseline accounting.
                    transforms_applied = [*_transforms, *list(transforms_applied)]
                if _modified:
                    body_mutation_tracker.mark_mutated("responses_compression")
                    tokens_saved = int(_tokens_saved)
                    optimized_tokens = max(0, original_tokens - tokens_saved)
                    logger.info(
                        "[%s] /v1/responses compressed %d→%d bytes "
                        "(%d tokens saved, auth_mode=%s, transforms=%s)",
                        request_id,
                        _bytes_before,
                        _bytes_after,
                        tokens_saved,
                        auth_mode.value,
                        transforms_applied,
                    )
                else:
                    logger.info(
                        "[%s] /v1/responses compression passthrough "
                        "reason=%s bytes=%d auth_mode=%s model=%s",
                        request_id,
                        _reason or "no_compression",
                        _bytes_before,
                        auth_mode.value,
                        model or "unknown",
                    )
            except Exception as _e:
                _http_body_bytes = len(json.dumps(body).encode("utf-8", errors="replace"))
                logger.warning(
                    f"[{request_id}] /v1/responses compression failed "
                    f"(bytes={_http_body_bytes}): {type(_e).__name__}: {_e}"
                )
                # Fail-closed protection (default): refuse to forward
                # oversized requests after compression failure. Same
                # decision matrix and override env var as the WS path
                # (HEADROOM_WS_FAIL_OPEN_ON_COMPRESSION_FAILURE) — see
                # helpers.decide_compression_failure_action.
                from headroom.proxy.helpers import (
                    decide_compression_failure_action,
                )

                _http_action = decide_compression_failure_action(
                    _e,
                    _http_body_bytes,
                    client=client,
                )
                if _http_action.refuse:
                    logger.error(
                        "[%s] /v1/responses REFUSING to forward request "
                        "after compression failure (reason=%s, bytes=%d); "
                        "returning HTTP 413 so the client can compact "
                        "context and retry. To restore legacy passthrough "
                        "behaviour set "
                        "HEADROOM_WS_FAIL_OPEN_ON_COMPRESSION_FAILURE=1.",
                        request_id,
                        _http_action.reason,
                        _http_action.frame_bytes,
                    )
                    raise HTTPException(
                        status_code=413,
                        detail={
                            "error": {
                                "type": "compression_refused",
                                "message": (
                                    f"headroom: compression "
                                    f"{_http_action.reason} on a "
                                    f"{_http_body_bytes}-byte request "
                                    "— please compact context and retry."
                                ),
                            }
                        },
                    ) from _e

        if not _bypass:
            _http_conversation_key = request.headers.get("x-headroom-session-id")
            _shape_result = _shape_openai_responses_for_output(
                body,
                input_tokens=original_tokens,
                model=str(model or ""),
                conversation_key=(
                    f"header:x-headroom-session-id:{_http_conversation_key}"
                    if _http_conversation_key
                    else None
                ),
            )
            _append_unique_transforms(transforms_applied, _shape_result.labels)
            if _shape_result.changed:
                body_mutation_tracker.mark_mutated("responses_output_shaping")
                logger.info(
                    "[%s] /v1/responses output shaping labels=%s",
                    request_id,
                    _shape_result.labels,
                )

            capture_codex_wire_debug(
                "http_upstream_request",
                request_id=request_id,
                transport="http",
                direction="headroom_to_upstream",
                method="POST",
                url=url,
                headers=headers,
                body=body,
                metadata={
                    "path": request.url.path,
                    "stream": stream,
                    "auth_mode": auth_mode.value,
                    "is_chatgpt_auth": is_chatgpt_auth,
                    "tokens_saved": tokens_saved,
                    "transforms_applied": transforms_applied,
                },
            )

        # Waste-signal detection for the Responses path (#820). The transform
        # pipeline never runs here (compression goes through CompressionUnits),
        # so parse a telemetry-only message conversion directly, behind the
        # same >100 saved-token gate as TransformPipeline.apply.
        waste_signals_dict: dict[str, int] | None = None
        if tokens_saved > 100:
            try:
                from headroom.parser import parse_messages

                _, _, _waste = parse_messages(
                    _responses_input_to_waste_messages(instructions, input_data),
                    tokenizer,
                )
                if _waste.total() > 0:
                    waste_signals_dict = _waste.to_dict()
            except Exception:
                pass

        # CCR: a stream:true request whose tool list carries headroom_retrieve
        # can't be intercepted mid-SSE-stream without full event-level
        # splicing (#1877 proposals B/C, out of scope here). Instead, force
        # a buffered stream:false upstream call so retrieval can be resolved
        # server-side, then reconstruct a minimal SSE stream for the client.
        # Mirrors AnthropicHandler's buffered_stream_ccr decision.
        _ccr_response_handler = getattr(self, "ccr_response_handler", None)
        _ccr_handler_config = getattr(_ccr_response_handler, "config", None)
        _ccr_response_handler_enabled = bool(
            _ccr_response_handler and getattr(_ccr_handler_config, "enabled", True)
        )
        buffered_stream_ccr = _should_buffer_openai_responses_stream_ccr(
            stream=stream,
            ccr_response_handler_enabled=_ccr_response_handler_enabled,
            tools=body.get("tools"),
            is_chatgpt_auth=is_chatgpt_auth,
        )
        if buffered_stream_ccr:
            if body.get("stream") is not False:
                body["stream"] = False
                body_mutation_tracker.mark_mutated("ccr_streaming_retrieve_buffered_non_stream")
            logger.info(
                f"[{request_id}] CCR: stream:true /v1/responses request has "
                "headroom_retrieve available; using buffered stream:false "
                "upstream request for server-side retrieval handling"
            )

        try:
            if stream and not buffered_stream_ccr:
                # Streaming for Responses API uses semantic events
                return await self._stream_response(
                    url,
                    headers,
                    body,
                    "openai",
                    model,
                    request_id,
                    original_tokens,
                    optimized_tokens,
                    tokens_saved,
                    transforms_applied,
                    tags,
                    optimization_latency,
                    memory_user_id=memory_user_id,
                    memory_request_ctx=memory_request_ctx,
                    original_body_bytes=original_body_bytes,
                    body_mutated=body_mutation_tracker.mutated,
                    mutation_reasons=body_mutation_tracker.reasons,
                    waste_signals=waste_signals_dict,
                )
            else:

                async def _buffered_ccr_operation():
                    nonlocal headers
                    headers = await apply_copilot_api_auth(headers, url=url)
                    response = await self._retry_request(
                        "POST",
                        url,
                        headers,
                        body,
                        original_body_bytes=original_body_bytes,
                        body_mutated=body_mutation_tracker.mutated,
                        mutation_reasons=body_mutation_tracker.reasons,
                        request_id=request_id,
                        forwarder_name="openai_responses",
                        path_for_log=url,
                    )
                    _response_body_for_debug: Any = None
                    _response_raw_for_debug: str | None = None
                    try:
                        _response_body_for_debug = response.json()
                    except Exception:
                        try:
                            _response_raw_for_debug = response.text[:200_000]
                        except Exception:
                            _response_raw_for_debug = None
                    capture_codex_wire_debug(
                        "http_upstream_response",
                        request_id=request_id,
                        transport="http",
                        direction="upstream_to_headroom",
                        method="POST",
                        url=url,
                        headers=dict(response.headers),
                        body=_response_body_for_debug,
                        raw_text=_response_raw_for_debug,
                        status_code=response.status_code,
                        metadata={"stream": stream, "auth_mode": auth_mode.value},
                    )
                    total_latency = (time.time() - start_time) * 1000

                    total_input_tokens = original_tokens  # fallback
                    output_tokens = 0
                    cache_read_tokens = 0
                    try:
                        resp_json = response.json()
                        usage = resp_json.get("usage", {})

                        def _usage_int(value: Any, default: int = 0) -> int:
                            try:
                                return max(int(value), 0)
                            except (TypeError, ValueError):
                                return default

                        total_input_tokens = _usage_int(
                            usage.get("input_tokens"),
                            original_tokens,
                        )
                        output_tokens = _usage_int(usage.get("output_tokens"))
                        details = usage.get("input_tokens_details")
                        if isinstance(details, dict):
                            cache_read_tokens = _usage_int(details.get("cached_tokens"))
                    except (KeyError, TypeError, AttributeError) as e:
                        logger.debug(
                            f"[{request_id}] Failed to extract cached tokens from OpenAI passthrough response: {e}"
                        )

                    # CCR Response Handling: intercept headroom_retrieve tool
                    # calls server-side so a Responses API function_call the
                    # downstream caller can't resolve (e.g. Strands, or a
                    # buffered-stream request) never reaches the client. Mirrors
                    # the chat-completions backend-path block (handle_openai_chat
                    # ~2775-2848), adapted for the Responses API's flat
                    # function_call / output[] shape instead of Messages API
                    # tool_calls. Runs before memory tool handling below so a
                    # retrieve call never gets treated as an unresolved tool_call
                    # by the memory-tool branch.
                    if (
                        _ccr_response_handler
                        and resp_json
                        and response.status_code == 200
                        and _ccr_response_handler.has_ccr_tool_calls(resp_json, "openai_responses")
                    ):
                        logger.info(
                            f"[{request_id}] CCR: Detected retrieval tool call (responses), handling..."
                        )

                        async def api_call_fn(
                            items: list[dict[str, Any]],
                            tls: list[dict[str, Any]] | None,
                        ) -> dict[str, Any]:
                            continuation_body = {**body, "input": items}
                            if tls is not None:
                                continuation_body["tools"] = tls
                            # Fresh stateless continuation: resend the full
                            # item history rather than chaining through
                            # previous_response_id, matching how
                            # CCRResponseHandler accumulates `current_messages`
                            # for every other provider. `body["stream"]` is
                            # left as-is: for a buffered_stream_ccr request it
                            # was already forced False above, and continuations
                            # must stay non-streaming so this handler (not
                            # `_stream_response`) can parse the JSON reply.
                            continuation_body.pop("previous_response_id", None)
                            continuation_body["stream"] = False

                            continuation_headers = {
                                k: v
                                for k, v in headers.items()
                                if k.lower()
                                not in (
                                    "content-encoding",
                                    "transfer-encoding",
                                    "accept-encoding",
                                    "content-length",
                                )
                            }
                            logger.info(
                                f"[{request_id}] CCR: Issuing Responses continuation "
                                f"({len(items)} input items)"
                            )
                            cont_response = await self._retry_request(
                                "POST",
                                url,
                                continuation_headers,
                                continuation_body,
                                request_id=request_id,
                                forwarder_name="openai_responses_ccr_continuation",
                                path_for_log=url,
                            )
                            return cont_response.json()

                        try:
                            final_resp_json = await _ccr_response_handler.handle_response(
                                resp_json,
                                _responses_input_to_items(body.get("input")),
                                body.get("tools"),
                                api_call_fn,
                                provider="openai_responses",
                            )
                            resp_json = final_resp_json
                            # Remove encoding headers since content is now
                            # uncompressed JSON we synthesized.
                            ccr_response_headers = {
                                k: v
                                for k, v in response.headers.items()
                                if k.lower() not in ("content-encoding", "content-length")
                            }
                            response = httpx.Response(
                                status_code=200,
                                content=json.dumps(final_resp_json).encode(),
                                headers=ccr_response_headers,
                            )
                            logger.info(
                                f"[{request_id}] CCR: Retrieval handled successfully (responses)"
                            )
                        except Exception as e:
                            logger.error(
                                f"[{request_id}] CCR: Response handling failed (responses): {e}"
                            )
                            # NO SILENT FALLBACK: re-raise so the client sees a
                            # clear failure instead of an unresolved tool_call
                            # it can't act on. Matches the OpenAI backend-path
                            # block in handle_openai_chat; see
                            # feedback_no_silent_fallbacks.
                            raise

                    # Memory: handle memory tool calls in Responses API response
                    if (
                        self.memory_handler
                        and memory_user_id
                        and responses_memory_tools_allowed
                        and resp_json
                        and response.status_code == 200
                        and self.memory_handler.has_memory_tool_calls(resp_json, "openai")
                    ):
                        try:
                            # Extract function_call items from output
                            from headroom.proxy.memory_handler import MEMORY_TOOL_NAMES

                            output_items = resp_json.get("output", [])
                            memory_fc_items = [
                                item
                                for item in output_items
                                if isinstance(item, dict)
                                and item.get("type") == "function_call"
                                and item.get("name") in MEMORY_TOOL_NAMES
                            ]

                            # Execute memory tool calls
                            tool_outputs: list[dict[str, Any]] = []
                            for fc in memory_fc_items:
                                call_id = fc.get("call_id", fc.get("id", ""))
                                name = fc.get("name", "")
                                args_str = fc.get("arguments", "{}")
                                try:
                                    args = json.loads(args_str)
                                except json.JSONDecodeError:
                                    args = {}

                                await self.memory_handler._ensure_initialized()
                                if self.memory_handler._backend:
                                    result = await self.memory_handler._execute_memory_tool(
                                        name, args, memory_user_id, "openai"
                                    )
                                else:
                                    result = json.dumps({"error": "Memory backend not initialized"})

                                tool_outputs.append(
                                    {
                                        "type": "function_call_output",
                                        "call_id": call_id,
                                        "output": result,
                                    }
                                )

                            if tool_outputs:
                                # Make continuation request with tool results
                                response_id = resp_json.get("id")
                                continuation_body = {
                                    "model": model,
                                    "input": tool_outputs,
                                }
                                if response_id:
                                    continuation_body["previous_response_id"] = response_id
                                existing_tools = body.get("tools")
                                if existing_tools:
                                    continuation_body["tools"] = existing_tools

                                cont_response = await self._retry_request(
                                    "POST", url, headers, continuation_body
                                )
                                resp_json = cont_response.json()
                                response = cont_response
                                logger.info(
                                    f"[{request_id}] Memory: Handled {len(tool_outputs)} "
                                    f"tool call(s) with continuation for user {memory_user_id} (responses)"
                                )
                        except Exception as e:
                            logger.warning(
                                f"[{request_id}] Memory tool handling failed (responses): {e}"
                            )

                    # Cost is recorded once by the outcome funnel below; here we only
                    # compute the cache-write / uncached split the funnel needs.
                    # (Recording here too double-counted spend + budget on this path.)
                    cache_write_tokens = _infer_openai_cache_write_tokens(
                        total_input_tokens,
                        cache_read_tokens,
                    )
                    uncached_input_tokens = max(0, total_input_tokens - cache_read_tokens)

                    # Was: optimized := provider count, then original := max(original,
                    # optimized + saved) to stop `attempted` exceeding `original`.
                    # That paper over the symptom by inflating `original` upward,
                    # which is why the beacon still shipped `eligible_pct > 100`
                    # from the other emit sites that lacked the same fudge. The
                    # real cause was mixing tokenizer scales; keep the local pair
                    # coherent and hand the provider count over separately.
                    effective_optimized_tokens = optimized_tokens
                    effective_original_tokens = original_tokens

                    _resp_log_tags = {
                        **(tags or {}),
                        "auth_mode": auth_mode.value if auth_mode else "payg",
                        "endpoint": "responses_http",
                    }

                    # OpenAI Responses HTTP (non-WS, non-streaming). Codex
                    # uses this path when configured for HTTP transport.
                    # Pre-refactor `cache_hit` was hardcoded False on
                    # RequestLog even when cache_read>0 — funnel derives
                    # it correctly.
                    from headroom.proxy.helpers import compute_turn_id

                    await self._record_request_outcome(
                        RequestOutcome(
                            request_id=request_id,
                            provider="openai",
                            model=model,
                            status_code=response.status_code,
                            original_tokens=effective_original_tokens,
                            optimized_tokens=effective_optimized_tokens,
                            provider_input_tokens=total_input_tokens,
                            output_tokens=output_tokens,
                            tokens_saved=tokens_saved,
                            attempted_input_tokens=attempted_input_tokens,
                            cache_read_tokens=cache_read_tokens,
                            cache_write_tokens=cache_write_tokens,
                            uncached_input_tokens=uncached_input_tokens,
                            cache_inferred=True,
                            total_latency_ms=total_latency,
                            overhead_ms=optimization_latency,
                            transforms_applied=tuple(transforms_applied),
                            waste_signals=waste_signals_dict,
                            num_messages=len(messages) if isinstance(messages, list) else 0,
                            tags=_resp_log_tags,
                            turn_id=compute_turn_id(model, body.get("instructions"), messages),
                            request_messages=messages
                            if getattr(self.config, "log_full_messages", False)
                            else None,
                            client=client,
                        )
                    )

                    logger.info(
                        f"[{request_id}] /v1/responses {model}: {total_input_tokens:,} tokens"
                    )

                    # Capture Codex rate-limit window data from response headers
                    from headroom.subscription.codex_rate_limits import (
                        get_codex_rate_limit_state,
                    )

                    get_codex_rate_limit_state().update_from_headers(dict(response.headers))

                    # Remove compression headers
                    response_headers = _sanitize_forwarded_response_headers(response.headers)

                    if buffered_stream_ccr and response.status_code == 200 and resp_json:
                        sse_headers = {
                            k: v
                            for k, v in response_headers.items()
                            if k.lower() not in ("content-length", "content-type")
                        }
                        if _ccr_response_handler and _ccr_response_handler.has_ccr_tool_calls(
                            resp_json, "openai_responses"
                        ):
                            # Handling above didn't fully resolve the retrieve
                            # call (e.g. max rounds hit, or it was mixed with a
                            # non-CCR tool call). Fail closed rather than stream
                            # a response the client can't act on — matches the
                            # Anthropic buffered path's residual-CCR guard.
                            logger.warning(
                                f"[{request_id}] CCR: Buffered streaming Responses "
                                "reply still contains headroom_retrieve after "
                                "handling; failing closed"
                            )

                            async def _residual_ccr_error_sse():
                                error_event = {
                                    "type": "error",
                                    "error": {
                                        "message": "Unable to safely complete streamed CCR retrieval.",
                                    },
                                }
                                yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode()

                            return StreamingResponse(
                                _residual_ccr_error_sse(),
                                media_type="text/event-stream",
                                headers=sse_headers,
                                status_code=502,
                            )

                        async def _buffered_ccr_sse():
                            for event in _openai_responses_to_sse(resp_json):
                                yield event

                        return StreamingResponse(
                            _buffered_ccr_sse(),
                            media_type="text/event-stream",
                            headers=sse_headers,
                        )

                    return Response(
                        content=response.content,
                        status_code=response.status_code,
                        headers=response_headers,
                    )

            if buffered_stream_ccr:
                operation = asyncio.create_task(_buffered_ccr_operation())
                record_failed = self.metrics.record_failed

                class _BufferedCCRResponse(Response):
                    async def __call__(self, scope, receive, send):  # noqa: ANN001
                        await asyncio.sleep(0)
                        loop = asyncio.get_running_loop()
                        keepalive_deadline = loop.time() + 1.0
                        started = False
                        try:
                            while True:
                                timeout = (
                                    0.25 if started else max(0.0, keepalive_deadline - loop.time())
                                )
                                done, _ = await asyncio.wait({operation}, timeout=timeout)
                                if done:
                                    try:
                                        result = operation.result()
                                    except Exception as e:
                                        await record_failed(provider="openai")
                                        logger.error(
                                            f"[{request_id}] OpenAI responses request failed: {type(e).__name__}: {e}"
                                        )
                                        if not started:
                                            await send(
                                                {
                                                    "type": "http.response.start",
                                                    "status": 502,
                                                    "headers": [
                                                        (b"content-type", b"application/json")
                                                    ],
                                                }
                                            )
                                            await send(
                                                {
                                                    "type": "http.response.body",
                                                    "body": json.dumps(
                                                        {
                                                            "error": {
                                                                "message": "An error occurred while processing your request. Please try again.",
                                                                "type": "server_error",
                                                                "code": "proxy_error",
                                                            }
                                                        }
                                                    ).encode(),
                                                    "more_body": False,
                                                }
                                            )
                                            return
                                        await send(
                                            {
                                                "type": "http.response.body",
                                                "body": b'event: error\ndata: {"type":"error","error":{"message":"An error occurred while processing the request."}}\n\n',
                                                "more_body": False,
                                            }
                                        )
                                        return

                                    if not started:
                                        await result(scope, receive, send)
                                        return

                                    body_iterator = getattr(result, "body_iterator", None)
                                    if body_iterator is not None:
                                        async for chunk in body_iterator:
                                            await send(
                                                {
                                                    "type": "http.response.body",
                                                    "body": chunk,
                                                    "more_body": True,
                                                }
                                            )
                                        await send(
                                            {
                                                "type": "http.response.body",
                                                "body": b"",
                                                "more_body": False,
                                            }
                                        )
                                        return

                                    await send(
                                        {
                                            "type": "http.response.body",
                                            "body": b'event: error\ndata: {"type":"error","error":{"message":"An error occurred while processing the request."}}\n\n',
                                            "more_body": False,
                                        }
                                    )
                                    return

                                if not started:
                                    await send(
                                        {
                                            "type": "http.response.start",
                                            "status": 200,
                                            "headers": [(b"content-type", b"text/event-stream")],
                                        }
                                    )
                                    started = True
                                await send(
                                    {
                                        "type": "http.response.body",
                                        "body": b'event: ping\ndata: {"type":"ping"}\n\n',
                                        "more_body": True,
                                    }
                                )
                        except asyncio.CancelledError:
                            raise
                        finally:
                            if not operation.done():
                                operation.cancel()
                            try:
                                await operation
                            except asyncio.CancelledError:
                                pass
                            except Exception:
                                pass

                return _BufferedCCRResponse(media_type="text/event-stream")
            return await _buffered_ccr_operation()
        except Exception as e:
            await self.metrics.record_failed(provider="openai")
            logger.error(f"[{request_id}] OpenAI responses request failed: {type(e).__name__}: {e}")
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": "An error occurred while processing your request. Please try again.",
                        "type": "server_error",
                        "code": "proxy_error",
                    }
                },
            )

    async def handle_openai_responses_ws(self, websocket: WebSocket) -> None:
        """WebSocket proxy for /v1/responses (Codex gpt-5.4+).

        Newer Codex versions use WebSocket instead of HTTP POST for the
        Responses API.  This handler:
        1. Validates origin and routing, then accepts ChatGPT-auth sessions
           immediately or API-key sessions after the upstream handshake
        2. Receives the first message (``response.create`` request)
        3. Opens an upstream WebSocket to OpenAI
        4. Compresses eligible `response.create` text through the Python
           ContentRouter path, then sends the request upstream
        5. Relays all subsequent messages bidirectionally
        """
        try:
            import websockets
        except ImportError:
            await websocket.accept()
            await websocket.close(
                code=1011,
                reason="websockets package not installed. pip install websockets",
            )
            return

        request_id = await self._next_request_id()
        session_id = uuid.uuid4().hex

        # Stage-timer — captures per-stage durations for the structured
        # log emitted on session close. Unit 2 instrumentation.
        stage_timer = StageTimer()
        session_started_at = time.perf_counter()

        # Unit 3: initialize registry variables *before* accept so the
        # outermost ``finally`` can rely on them existing even if
        # registration itself fails for some reason.
        ws_sessions: WebSocketSessionRegistry | None = getattr(self, "ws_sessions", None)
        session_handle: WSSessionHandle | None = None
        termination_cause: TerminationCause = "unknown"

        # Forward client headers to upstream, adding required OpenAI-Beta header
        ws_headers = dict(websocket.headers)
        _ws_url_obj = getattr(websocket, "url", None)
        _ws_url = str(_ws_url_obj) if _ws_url_obj is not None else ""
        _ws_path = getattr(_ws_url_obj, "path", "") if _ws_url_obj is not None else ""
        if not _ws_path:
            _ws_path = "/v1/responses"
        if not _is_allowed_websocket_origin(ws_headers):
            logger.warning(
                "event=websocket_origin_not_allowed request_id=%s session_id=%s path=%s origin=%r",
                request_id,
                session_id,
                _ws_path,
                _header_get(ws_headers, "origin"),
            )
            await websocket.close(code=1008, reason="origin not allowed")
            return
        # WS sessions bypass the HTTP middleware that stamps X-Client: codex on
        # the Responses endpoint, so apply the same path-based stamp here before
        # classify_client runs (parallels server.py / should_stamp_codex_client).
        if should_stamp_codex_client(_ws_path, ws_headers):
            ws_headers["x-client"] = "codex"
        # Identify the WS harness before downstream auth/header rewrites.
        # Captured in closure so per-turn RequestOutcome can stamp it.
        client = classify_client(ws_headers)
        # WS sessions bypass the HTTP middleware, so bind the project here;
        # per-turn outcome emission inside this task inherits the context. An
        # explicit X-Headroom-Project header wins; otherwise fall back to the
        # /p/<name> path prefix already bound by WebSocketProjectPrefixMiddleware
        # so prefix-only clients (aider, Copilot BYOK, Cursor) stay attributed.
        set_current_project(classify_project(ws_headers) or get_current_project())
        metrics_for_inbound_ws = getattr(self, "metrics", None)
        if metrics_for_inbound_ws is not None and hasattr(
            metrics_for_inbound_ws, "record_inbound_request"
        ):
            with contextlib.suppress(Exception):
                metrics_for_inbound_ws.record_inbound_request(method="WS", path=_ws_path)
        logger.info(
            "event=proxy_inbound_websocket request_id=%s session_id=%s path=%s "
            "client=%s header_count=%d",
            request_id,
            session_id,
            _ws_path,
            getattr(websocket, "client", ""),
            len(ws_headers),
        )
        from headroom.proxy.helpers import capture_codex_wire_debug

        capture_codex_wire_debug(
            "ws_inbound_handshake",
            request_id=request_id,
            session_id=session_id,
            transport="websocket",
            direction="client_to_headroom",
            url=_ws_url,
            headers=ws_headers,
            metadata={"path": _ws_path},
        )
        # Extract per-request tags from headers up front so the
        # session-end RequestLog can attach them. `_extract_tags` is
        # the same helper the HTTP handlers use; on a WebSocket the
        # tags come from `x-headroom-tag-*` headers in the upgrade
        # handshake. Returns `{}` when no tags are present.
        _extract_ws_tags = getattr(self, "_extract_tags", None)
        ws_tags = _extract_ws_tags(ws_headers) if callable(_extract_ws_tags) else {}

        # Extract subprotocol from client — this is an application-level negotiation
        # that MUST be forwarded end-to-end (unlike sec-websocket-key which is per-connection).
        # Codex and OpenAI negotiate a subprotocol; stripping it causes OpenAI to return 500.
        client_subprotocols: list[str] = []
        raw_protocol = ws_headers.get("sec-websocket-protocol", "")
        if raw_protocol:
            client_subprotocols = [p.strip() for p in raw_protocol.split(",") if p.strip()]

        # Forward all client headers except hop-by-hop / per-connection headers.
        # These are WebSocket handshake mechanics that the `websockets` library
        # generates fresh for the upstream connection — forwarding them would conflict.
        # Everything else (auth, org, beta, user-agent, custom headers) is forwarded as-is.
        _skip_headers = WS_HOP_BY_HOP_HEADERS
        # PR-A5 (P5-49): also drop internal x-headroom-* from the upstream
        # WebSocket handshake. Inbound reads on `ws_headers` (memory user-id
        # below) keep working because we filter only when building
        # `upstream_headers`, not when reading from `ws_headers`.
        from headroom.proxy.helpers import (
            _strip_internal_headers as _strip_internal,
        )
        from headroom.proxy.helpers import (
            log_outbound_headers as _log_outbound_headers,
        )

        _ws_pre_strip_filtered: dict[str, str] = {}
        for k, v in ws_headers.items():
            if k.lower() not in _skip_headers:
                _ws_pre_strip_filtered[k] = v
        _ws_pre_strip_count = sum(
            1 for k in _ws_pre_strip_filtered if k.lower().startswith("x-headroom-")
        )
        upstream_headers = _strip_internal(_ws_pre_strip_filtered)
        _log_outbound_headers(
            forwarder="openai_responses_ws",
            stripped_count=_ws_pre_strip_count,
            request_id=request_id,
        )

        upstream_headers, is_chatgpt_auth = _resolve_codex_routing_headers(upstream_headers)
        # OpenAI rejects newer Codex models when this client-only lite header leaks upstream.
        upstream_headers = {
            key: value
            for key, value in upstream_headers.items()
            if key.lower() != _CODEX_RESPONSES_LITE_HEADER
        }
        ws_memory_tools_allowed = _allow_responses_memory_tools(is_chatgpt_auth)
        _lower_headers = {k.lower(): v for k, v in upstream_headers.items()}

        # Build upstream WebSocket URL based on auth mode
        if is_chatgpt_auth:
            # ChatGPT session auth → route to chatgpt.com backend
            upstream_url = codex_responses_websocket_url()
            logger.debug(
                f"[{request_id}] WS: ChatGPT session auth detected, routing to chatgpt.com"
            )
        else:
            # API key auth → route to configured OpenAI API URL
            base = self.OPENAI_API_URL
            ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
            upstream_url = build_copilot_upstream_url(ws_base, "/v1/responses")

        # Resolve Copilot auth for this upstream FIRST, before any generic
        # OpenAI-key fallback below -- ensures a real client-supplied Copilot
        # credential is used/preserved, and Headroom's own credential fetch
        # only kicks in when the client truly sent none.
        upstream_headers = await apply_copilot_api_auth(upstream_headers, url=upstream_url)

        capture_codex_wire_debug(
            "ws_upstream_handshake",
            request_id=request_id,
            session_id=session_id,
            transport="websocket",
            direction="headroom_to_upstream",
            url=upstream_url,
            headers=upstream_headers,
            metadata={
                "is_chatgpt_auth": is_chatgpt_auth,
                "subprotocols": client_subprotocols,
            },
        )

        logger.info(
            "[%s] WS /v1/responses accepted (route=%s, auth_mode=%s, subprotocols=%s)",
            request_id,
            "chatgpt_subscription" if is_chatgpt_auth else "openai_api",
            classify_auth_mode(ws_headers).value,
            client_subprotocols,
        )

        # Ensure Authorization header is present — fall back to OPENAI_API_KEY env var.
        # Safety net for clients that don't forward auth headers via WebSocket upgrade.
        # Resolved AFTER apply_copilot_api_auth (moved earlier, see below) so a
        # real client-supplied Copilot credential is never mistaken for, or
        # clobbered by, this unrelated OpenAI-key fallback.
        if not any(k.lower() == "authorization" for k in upstream_headers):
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                upstream_headers["Authorization"] = f"Bearer {api_key}"
                logger.debug(f"[{request_id}] WS: injected Authorization from OPENAI_API_KEY env")
            else:
                logger.warning(
                    f"[{request_id}] WS: no Authorization header from client and "
                    f"OPENAI_API_KEY not set — upstream will likely reject"
                )

        # Ensure the required beta header is present — OpenAI returns 500 without it.
        # PR-A6 (P5-50): use the deterministic `merge_openai_beta` helper
        # so the auto-injected `responses_websockets=2026-02-06` is
        # appended to the client's value (preserving order, deduping
        # case-insensitively) rather than overwriting it. The
        # SessionBetaTracker also records the merge so a future cross-
        # connection sticky model can replay tokens by session_id.
        from headroom.proxy.helpers import (
            get_session_beta_tracker as _get_session_beta_tracker_ws,
        )
        from headroom.proxy.helpers import (
            log_beta_header_merge as _log_beta_header_merge_ws,
        )
        from headroom.proxy.helpers import merge_openai_beta as _merge_openai_beta_ws

        _ws_required_tokens = ["responses_websockets=2026-02-06"]
        # Read the original (pre-merge) client value from the WS headers
        # to preserve casing and ordering.
        _ws_client_beta_value: str | None = None
        for _k, _v in upstream_headers.items():
            if _k.lower() == "openai-beta":
                _ws_client_beta_value = _v
                break
        # Record session-stickiness BEFORE adding required tokens so the
        # tracker stores the canonical client baseline.
        _ws_sticky_beta = _get_session_beta_tracker_ws().record_and_get_sticky_betas(
            provider="openai",
            session_id=session_id,
            client_value=_ws_client_beta_value,
        )
        _ws_merged_beta = _merge_openai_beta_ws(_ws_sticky_beta, _ws_required_tokens)
        # Replace any existing case-variants of openai-beta with the
        # canonical "OpenAI-Beta" key carrying the merged value.
        _ws_existing_keys = [_k for _k in upstream_headers if _k.lower() == "openai-beta"]
        for _k in _ws_existing_keys:
            del upstream_headers[_k]
        if _ws_merged_beta:
            upstream_headers["OpenAI-Beta"] = _ws_merged_beta
        _ws_client_beta_count = (
            len([t for t in (_ws_client_beta_value or "").split(",") if t.strip()])
            if _ws_client_beta_value
            else 0
        )
        _ws_merged_beta_count = (
            len([t for t in _ws_merged_beta.split(",") if t.strip()]) if _ws_merged_beta else 0
        )
        _log_beta_header_merge_ws(
            provider="openai",
            session_id=session_id,
            client_betas_count=_ws_client_beta_count,
            sticky_betas_count=_ws_merged_beta_count,
            headroom_added=_ws_required_tokens,
            request_id=request_id,
        )

        capture_codex_wire_debug(
            "ws_upstream_handshake_final",
            request_id=request_id,
            session_id=session_id,
            transport="websocket",
            direction="headroom_to_upstream",
            url=upstream_url,
            headers=upstream_headers,
            metadata={
                "is_chatgpt_auth": is_chatgpt_auth,
                "subprotocols": client_subprotocols,
            },
        )

        logger.debug(
            f"[{request_id}] WS upstream headers: "
            f"{[k for k in upstream_headers if k.lower() != 'authorization']}, "
            f"subprotocols={client_subprotocols}"
        )
        accept_subprotocol = client_subprotocols[0] if client_subprotocols else None
        accepted_client_ws = False

        def _register_accepted_session() -> None:
            nonlocal session_handle
            if session_handle is not None or ws_sessions is None:
                return
            client_addr: str | None = None
            client_info = getattr(websocket, "client", None)
            if client_info is not None:
                host = getattr(client_info, "host", None)
                port = getattr(client_info, "port", None)
                if host is not None and port is not None:
                    client_addr = f"{host}:{port}"
                elif host is not None:
                    client_addr = str(host)
            session_handle = WSSessionHandle(
                session_id=session_id,
                request_id=request_id,
                client_addr=client_addr,
                upstream_url=upstream_url,
            )
            ws_sessions.register(session_handle)
            metrics = getattr(self, "metrics", None)
            if metrics is not None and hasattr(metrics, "inc_active_ws_sessions"):
                try:
                    metrics.inc_active_ws_sessions()
                except Exception:  # pragma: no cover - defensive
                    pass

        def _schedule_usage_poll() -> None:
            with contextlib.suppress(Exception):
                from headroom.subscription.codex_rate_limits import (
                    maybe_schedule_usage_poll,
                )

                maybe_schedule_usage_poll(ws_headers)

        try:
            # ChatGPT-auth sessions no longer need upstream x-codex-* headers on
            # the client-facing 101, so accept them before the upstream retry
            # loop. API-key sessions still connect first so the allowlisted
            # handshake headers remain attachable there.
            if is_chatgpt_auth:
                async with stage_timer.measure("accept"):
                    await websocket.accept(subprotocol=accept_subprotocol)
                accepted_client_ws = True
                _register_accepted_session()
                _schedule_usage_poll()

            # --- Connect to upstream OpenAI WebSocket ---
            logger.info(f"[{request_id}] WS /v1/responses connecting to {upstream_url}")

            # Use ssl=True to let the websockets library handle SSL natively.
            # Manual ssl.create_default_context() + certifi doesn't load the
            # Windows system cert store, causing HTTP 500 on wss:// connections.
            use_ssl: bool | None = True if upstream_url.startswith("wss://") else None

            ws_connected = False
            ws_connect_attempts = max(1, getattr(self.config, "retry_max_attempts", 3))
            ws_last_err: Exception | None = None
            _upstream_connect_started = time.perf_counter()
            _upstream_connect_recorded = False
            _upstream_first_event_started: float | None = None
            upstream: Any = None
            from headroom.proxy.helpers import merge_extra_headers

            upstream_headers = merge_extra_headers(
                upstream_headers, self.config.openai_extra_headers
            )

            for ws_attempt in range(ws_connect_attempts):
                try:
                    upstream = await websockets.connect(
                        upstream_url,
                        additional_headers=upstream_headers,
                        subprotocols=(
                            [websockets.Subprotocol(p) for p in client_subprotocols]
                            if client_subprotocols and hasattr(websockets, "Subprotocol")
                            else client_subprotocols or None
                        ),
                        ssl=use_ssl,
                        open_timeout=max(30, self.config.connect_timeout_seconds * 3),
                        close_timeout=10,
                        ping_interval=20,
                        # Image-generation turns go silent for 20-60s while the
                        # model renders (a single ``image_generation_call`` event,
                        # then a long quiet gap with no data frames). A 20s pong
                        # deadline false-kills the still-healthy upstream
                        # mid-render with ``upstream_error`` before the image
                        # lands. Keep ``ping_interval`` for NAT keepalive but do
                        # not tear the session down on a missing pong.
                        ping_timeout=None,
                        # The finished image arrives inline as a single base64
                        # frame that exceeds the websockets default 1 MiB cap,
                        # raising ``PayloadTooBig`` exactly as the image lands.
                        # The relay must accept frames as large as the endpoints
                        # do, so do not cap the upstream payload size.
                        max_size=None,
                    )
                    ws_connected = True
                    if not _upstream_connect_recorded:
                        stage_timer.record(
                            "upstream_connect",
                            (time.perf_counter() - _upstream_connect_started) * 1000.0,
                        )
                        _upstream_connect_recorded = True
                        _upstream_first_event_started = time.perf_counter()
                    break
                except Exception as ws_err:
                    ws_last_err = ws_err
                    if ws_attempt >= ws_connect_attempts - 1:
                        break
                    delay_with_jitter = jitter_delay_ms(
                        self.config.retry_base_delay_ms,
                        self.config.retry_max_delay_ms,
                        ws_attempt,
                    )
                    logger.warning(
                        f"[{request_id}] WS upstream connect failed "
                        f"(attempt {ws_attempt + 1}/{ws_connect_attempts}): {ws_err}; "
                        f"retrying in {delay_with_jitter:.0f}ms"
                    )
                    await asyncio.sleep(delay_with_jitter / 1000)

            # Preserve any upstream x-codex-* handshake headers for internal
            # rate-limit state, and forward them onto the client-facing 101 only
            # for API-key sessions that have not been accepted yet.
            accept_headers: list[tuple[bytes, bytes]] = []
            if ws_connected:
                _codex_handshake = _extract_codex_handshake_headers(upstream)
                if _codex_handshake:
                    accept_headers = [
                        (name.encode("latin-1"), value.encode("latin-1"))
                        for name, value in _codex_handshake
                    ]
                    # Parity with the HTTP path: also refresh Python /stats state.
                    from headroom.subscription.codex_rate_limits import (
                        get_codex_rate_limit_state,
                    )

                    with contextlib.suppress(Exception):
                        get_codex_rate_limit_state().update_from_headers(dict(_codex_handshake))

            if not accepted_client_ws:
                _schedule_usage_poll()
                async with stage_timer.measure("accept"):
                    await websocket.accept(
                        subprotocol=accept_subprotocol,
                        headers=accept_headers or None,
                    )
                accepted_client_ws = True
                _register_accepted_session()
            # Receive the first message from client (the response.create request).
            # Bound the wait with WS_FIRST_FRAME_TIMEOUT_SECONDS so a zombie
            # client that opens the WS but never sends a frame cannot hold a
            # session slot indefinitely. The StageTimer measurement still
            # captures the elapsed time up to the timeout so operators can
            # see the slow-client pattern in the stage-timings log.
            try:
                async with stage_timer.measure("first_client_frame"):
                    first_msg_raw = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=WS_FIRST_FRAME_TIMEOUT_SECONDS,
                    )
            except asyncio.TimeoutError:
                logger.info(
                    f"[{request_id}] WS first-frame timeout after "
                    f"{WS_FIRST_FRAME_TIMEOUT_SECONDS:.0f}s; closing session "
                    f"{session_id} (no client data)"
                )
                termination_cause = "client_timeout"
                with contextlib.suppress(Exception):
                    # 1001 (going away): server is cleanly terminating a slow
                    # client, not an internal error.
                    await websocket.close(code=1001, reason="first-frame timeout")
                # Exit the outer try so the session-lifecycle ``finally`` runs
                # deregister / metrics / stage-timings emission as usual.
                return

            # The standalone Rust proxy has a native Responses path, but the
            # CLI runtime runs this Python proxy. Compress eligible
            # `response.create` frames through the shared Python
            # CompressionUnit + ContentRouter path before upstream send.
            # Subsequent client→upstream frames are now ALSO compressed
            # via `_maybe_compress_response_create_frame` in
            # `_client_to_upstream` so long-lived subscription Codex
            # sessions get savings on every turn, not just the first.

            def _log_ws_passthrough(
                reason: str,
                *,
                frame_index: int,
                raw_bytes: int,
                frame_type: str = "",
                model: str = "",
            ) -> None:
                logger.info(
                    "[%s] WS /v1/responses frame passthrough "
                    "reason=%s frame=%d bytes=%d type=%s auth_mode=%s model=%s",
                    request_id,
                    reason,
                    frame_index,
                    raw_bytes,
                    frame_type or "unknown",
                    classify_auth_mode(ws_headers).value,
                    model or "unknown",
                )

            def _normalize_ws_response_create_for_upstream(raw_msg: str) -> str:
                """Optionally flatten Codex WS response.create frames for gateways.

                Codex sends {"type":"response.create","response":{...}}. Some
                OpenAI-compatible gateways accept the WebSocket upgrade but expect
                model/input/tools at the top level of response.create frames.
                Preserve default upstream behavior unless explicitly enabled.
                """
                if os.environ.get(
                    "HEADROOM_OPENAI_WS_FLATTEN_RESPONSE_CREATE", ""
                ).strip().lower() not in (
                    "1",
                    "true",
                    "yes",
                    "on",
                ):
                    return raw_msg
                try:
                    parsed = json.loads(raw_msg)
                except (json.JSONDecodeError, TypeError):
                    return raw_msg
                if not isinstance(parsed, dict) or parsed.get("type") != "response.create":
                    return raw_msg
                inner = parsed.get("response")
                if not isinstance(inner, dict):
                    return raw_msg

                flattened = dict(inner)
                flattened["type"] = "response.create"
                for key, value in parsed.items():
                    if key not in {"type", "response"} and key not in flattened:
                        flattened[key] = value
                return json.dumps(flattened, ensure_ascii=False)

            body: dict[str, Any] = {}
            tokens_saved = 0
            # Session-scoped accumulator for tokens we *attempted* to
            # compress (extracted units + schema). Drives the active-
            # compression ratio surfaced to the dashboard.
            attempted_input_tokens_total = 0
            transforms_applied: list[str] = []
            ws_frames_compressed = 0
            try:
                body = json.loads(first_msg_raw)
            except json.JSONDecodeError:
                # Not JSON — pass through as-is
                pass
            if isinstance(body, dict) and body:
                ws_response_body_for_store = (
                    body["response"] if isinstance(body.get("response"), dict) else body
                )
                if _ensure_chatgpt_responses_store_false(
                    ws_response_body_for_store,
                    is_chatgpt_auth=is_chatgpt_auth,
                ):
                    first_msg_raw = json.dumps(body)
                    logger.info(f"[{request_id}] WS Responses: forced store=false for ChatGPT auth")
            ws_input_tokens_total = 0
            ws_output_tokens_total = 0
            ws_cache_read_tokens_total = 0
            ws_cache_write_tokens_total = 0
            ws_uncached_input_tokens_total = 0
            ws_recorded_input_tokens_total = 0
            ws_recorded_output_tokens_total = 0
            ws_recorded_cache_read_tokens_total = 0
            ws_recorded_cache_write_tokens_total = 0
            ws_recorded_uncached_input_tokens_total = 0
            ws_recorded_tokens_saved_total = 0
            ws_recorded_attempted_input_tokens_total = 0
            ws_response_create_frames = 1
            ws_client_frames_total = 1
            ws_upstream_frames_total = 0
            ws_cancel_frames = 0
            ws_last_client_frame_type = str(body.get("type") or "unknown") if body else "unknown"
            ws_last_upstream_frame_type = "unknown"
            ws_client_disconnect_seen = False
            ws_overhead_ms_total = 0.0
            ws_recorded_overhead_ms_total = 0.0
            ws_compression_timing_totals: dict[str, float] = {}
            ws_recorded_compression_timing_totals: dict[str, float] = {}
            ws_ttfb_ms: float | None = None
            ws_recorded_ttfb_ms = False
            _ws_bypass = self._headroom_bypass_enabled(ws_headers)
            if _ws_bypass:
                logger.info(
                    "[%s] WS /v1/responses passthrough reason=bypass_header mutation=disabled",
                    request_id,
                )

            capture_codex_wire_debug(
                "ws_inbound_first_frame",
                request_id=request_id,
                session_id=session_id,
                transport="websocket",
                direction="client_to_headroom",
                url=_ws_url,
                body=body if body else None,
                raw_text=None if body else first_msg_raw,
                metadata={"frame": 1},
            )

            def _record_ws_compression_overhead(duration_ms: float) -> None:
                nonlocal ws_overhead_ms_total
                ws_overhead_ms_total += max(0.0, float(duration_ms))
                if ws_overhead_ms_total > 0:
                    stage_timer.record("compression", ws_overhead_ms_total)

            def _record_ws_compression_timing(name: str, duration_ms: float) -> None:
                ws_compression_timing_totals[name] = ws_compression_timing_totals.get(
                    name, 0.0
                ) + max(0.0, float(duration_ms))

            def _codex_ws_final_strategies(timing: dict[str, float]) -> list[str]:
                prefix = "compression_unit_router_strategy_"
                return [
                    name.removeprefix(prefix)
                    for name, ms in timing.items()
                    if name.startswith(prefix) and ms > 0
                ]

            def _codex_ws_strategy_chain(transforms: list[str]) -> list[str]:
                chain: list[str] = []
                for transform in transforms:
                    if ":" in transform:
                        continue
                    if transform not in chain:
                        chain.append(transform)
                return chain

            def _current_ws_overhead_ms() -> float:
                summary = stage_timer.summary()
                return ws_overhead_ms_total + max(0.0, float(summary.get("memory_context") or 0.0))

            def _ws_dashboard_pipeline_timing(
                *,
                overhead_ms: float,
                ttfb_ms: float,
            ) -> dict[str, float]:
                timing: dict[str, float] = {}
                if overhead_ms > 0:
                    timing["codex_ws.compression"] = overhead_ms
                if ttfb_ms > 0:
                    timing["codex_ws.ttfb"] = ttfb_ms

                for stage_name, total_ms in ws_compression_timing_totals.items():
                    recorded_ms = ws_recorded_compression_timing_totals.get(stage_name, 0.0)
                    delta_ms = max(0.0, total_ms - recorded_ms)
                    if delta_ms > 0:
                        timing[f"codex_ws.{stage_name}"] = delta_ms

                summary = stage_timer.summary()
                for stage_name in (
                    "memory_context",
                    "upstream_connect",
                    "upstream_first_event",
                ):
                    value = summary.get(stage_name)
                    if value is not None and value > 0:
                        timing[f"codex_ws.{stage_name}"] = float(value)
                return timing

            def _prepare_ws_performance_metrics() -> tuple[float, float, dict[str, float]]:
                current_overhead_ms = _current_ws_overhead_ms()
                overhead_delta_ms = max(
                    0.0,
                    current_overhead_ms - ws_recorded_overhead_ms_total,
                )
                ttfb_for_record_ms = (
                    max(0.0, float(ws_ttfb_ms))
                    if ws_ttfb_ms is not None and not ws_recorded_ttfb_ms
                    else 0.0
                )
                return (
                    overhead_delta_ms,
                    ttfb_for_record_ms,
                    _ws_dashboard_pipeline_timing(
                        overhead_ms=overhead_delta_ms,
                        ttfb_ms=ttfb_for_record_ms,
                    ),
                )

            memory_user_id: str | None = None
            memory_request_ctx = None
            from headroom.proxy.helpers import get_memory_injection_mode, log_memory_injection
            from headroom.proxy.memory_decision import MemoryDecision
            from headroom.proxy.memory_query import MemoryQuery

            async def _prepare_memory_frame(frame_body: dict[str, Any], frame_raw: str) -> str:
                nonlocal memory_user_id, memory_request_ctx

                memory_user_id_candidate = (
                    ws_headers.get(
                        "x-headroom-user-id",
                        os.environ.get("USER", os.environ.get("USERNAME", "default")),
                    )
                    if self.memory_handler
                    else None
                )
                memory_decision = MemoryDecision.decide(
                    headers=ws_headers,
                    memory_handler=self.memory_handler,
                    memory_user_id=memory_user_id_candidate,
                    mode_name=get_memory_injection_mode(),
                )
                memory_decision.apply_to_tags(ws_tags)
                if not memory_decision.inject:
                    return frame_raw

                memory_user_id = memory_user_id_candidate
                try:
                    # Unwrap response.create envelope to access the response body
                    ws_response_body = frame_body.get("response", frame_body)

                    # Per-project memory routing (GH #462). For WS,
                    # ``ws_response_body`` carries ``instructions`` —
                    # that's the system-prompt-equivalent we feed to the
                    # resolver.
                    from headroom.memory.storage_router import (
                        RequestContext as _MemRequestContext,
                    )

                    memory_request_ctx = _MemRequestContext(
                        headers=dict(ws_headers),
                        system_prompt=str(ws_response_body.get("instructions") or ""),
                        base_user_id=memory_user_id,
                        project_root_override=(
                            getattr(self.memory_handler.config, "project_root_override", "") or None
                        ),
                    )

                    # Debug: log what Codex sends so we can see the full tool list
                    existing_tool_names = [
                        t.get("name") or t.get("function", {}).get("name", "?")
                        for t in (ws_response_body.get("tools") or [])
                    ]
                    instr_preview = (ws_response_body.get("instructions") or "")[:200]
                    logger.info(
                        f"[{request_id}] WS Memory: Codex tools={existing_tool_names}, "
                        f"instructions_len={len(ws_response_body.get('instructions') or '')}, "
                        f"instructions_preview={instr_preview!r}"
                    )

                    # Inject memory context into instructions
                    if self.memory_handler.config.inject_context:
                        ws_input = ws_response_body.get("input", "")
                        ws_instructions = ws_response_body.get("instructions")
                        ws_msgs: list[dict[str, Any]] = []
                        if ws_instructions:
                            ws_msgs.append({"role": "system", "content": ws_instructions})
                        if isinstance(ws_input, str) and ws_input:
                            ws_msgs.append({"role": "user", "content": ws_input})
                        # PR-C5: list-typed `input` no longer feeds memory
                        # search via the Python converter — the Rust handler
                        # owns native item-aware processing. Memory context
                        # for list-input WS sessions falls back to the
                        # `instructions` system message only.

                        try:
                            async with stage_timer.measure("memory_context"):
                                memory_context = await asyncio.wait_for(
                                    self.memory_handler.search_and_format_context(
                                        memory_user_id,
                                        ws_msgs,
                                        request_context=memory_request_ctx,
                                        query=MemoryQuery.from_messages(ws_msgs),
                                    ),
                                    timeout=RESPONSES_CONTEXT_SEARCH_TIMEOUT_SECONDS,
                                )
                        except asyncio.TimeoutError:
                            memory_context = None
                            logger.info(
                                f"[{request_id}] WS Memory: Context lookup exceeded "
                                f"{RESPONSES_CONTEXT_SEARCH_TIMEOUT_SECONDS:.1f}s; "
                                f"continuing without it"
                            )
                        if memory_context:
                            # Route memory into ws_response_body["input"]
                            # (the user-input field) rather than
                            # ws_response_body["instructions"] (the
                            # system/cache-hot-zone field). All other
                            # handlers inject at the user-message tail
                            # so the cache prefix bytes stay byte-
                            # stable across turns — invariant I2. The
                            # WS path was the lone outlier writing to
                            # instructions (system); fixed here for
                            # uniformity with sites 1/2/3/4.
                            ws_input_for_inject = ws_response_body.get("input", "")
                            if isinstance(ws_input_for_inject, str):
                                if ws_input_for_inject:
                                    ws_response_body["input"] = (
                                        ws_input_for_inject + "\n\n" + memory_context
                                    )
                                else:
                                    ws_response_body["input"] = memory_context
                                logger.info(
                                    f"[{request_id}] WS Memory: Injected {len(memory_context)} chars "
                                    f"into input tail (string-shaped input)"
                                )
                                log_memory_injection(
                                    request_id=request_id,
                                    session_id=session_id,
                                    decision="injected_live_zone_tail_ws",
                                    bytes_injected=len(memory_context),
                                    query=None,
                                    tags=ws_tags,
                                )
                            else:
                                # List-shaped WS input is owned by the
                                # Rust handler (per PR-C5 comment). The
                                # Python path leaves memory un-injected
                                # for list inputs rather than touching
                                # instructions.
                                logger.info(
                                    f"[{request_id}] WS Memory: list-shaped input — "
                                    f"injection deferred to Rust handler"
                                )

                    # Inject memory tools (Responses API format) — PR-A7 (P0-6).
                    # WS path uses a per-connection UUID; tracker scope is
                    # the WS session (short-lived). Pre-convert to Responses
                    # API format so canonical bytes match the wire format.
                    from headroom.proxy.helpers import (
                        apply_session_sticky_memory_tools as _apply_sticky_mem_tools_ws,
                    )

                    ws_mem_defs_chat = (
                        self.memory_handler.compute_memory_tool_definitions("openai")
                        if self.memory_handler.config.inject_tools and ws_memory_tools_allowed
                        else []
                    )
                    ws_mem_defs_responses: list[dict[str, Any]] = []
                    for t in ws_mem_defs_chat:
                        if t.get("type") == "function" and "function" in t:
                            fn = t["function"]
                            ws_mem_defs_responses.append(
                                {
                                    "type": "function",
                                    "name": fn.get("name"),
                                    "description": fn.get("description", ""),
                                    "parameters": fn.get("parameters", {}),
                                }
                            )
                        else:
                            ws_mem_defs_responses.append(t)

                    ws_tools = ws_response_body.get("tools") or []
                    ws_tools, mem_injected = _apply_sticky_mem_tools_ws(
                        provider="openai",
                        session_id=session_id if ws_memory_tools_allowed else None,
                        request_id=request_id,
                        existing_tools=ws_tools,
                        memory_tools_to_inject=ws_mem_defs_responses,
                        inject_this_turn=bool(
                            self.memory_handler.config.inject_tools and ws_memory_tools_allowed
                        ),
                    )
                    if mem_injected:
                        ws_response_body["tools"] = ws_tools

                        # Add memory instruction so the model uses
                        # memory tools as persistent cross-session knowledge.
                        mem_instruction = (
                            "\n\n## Memory\n"
                            "You have persistent memory via memory_search and "
                            "memory_save tools. Memory stores knowledge across "
                            "sessions — user info, project details, org context, "
                            "decisions, architecture, conventions, anything worth "
                            "remembering.\n\n"
                            "- ALWAYS call memory_search BEFORE searching files "
                            "when the user asks a question that could be answered "
                            "from prior knowledge.\n"
                            "- Call memory_save to store important facts, decisions, "
                            "or context that would be useful in future sessions.\n"
                            "- Memory is your first source of truth for anything "
                            "not visible in the current conversation."
                        )
                        existing_instr = ws_response_body.get("instructions") or ""
                        ws_response_body["instructions"] = existing_instr + mem_instruction
                        logger.info(
                            f"[{request_id}] WS Memory: Injected memory tools + instruction"
                        )

                    # Write back into envelope if it was wrapped
                    if "response" in frame_body and isinstance(frame_body["response"], dict):
                        frame_body["response"] = ws_response_body
                    else:
                        frame_body = ws_response_body

                    return json.dumps(frame_body)
                except Exception as e:
                    logger.warning(f"[{request_id}] WS Memory injection failed: {e}")
                    return frame_raw

            if isinstance(body, dict) and (
                body.get("type") == "response.create" or ("type" not in body and "input" in body)
            ):
                first_msg_raw = await _prepare_memory_frame(body, first_msg_raw)

            # Hot-fix follow-up to PR #406 — inline Rust compression on the
            # WS first frame before forwarding upstream. PR #406 enabled
            # the same call for HTTP /v1/responses; PR-C5's "WS-side
            # compression is a follow-up" note is closed here. Codex
            # subscription users default to WebSocket transport for
            # /v1/responses (proxy-confirmed via #409 reviewer testing),
            # so without this call subscription traffic flows through
            # Headroom uncompressed.
            #
            # The first frame may be either:
            #   • {"type": "response.create", "response": {...payload...}}
            #   • the payload directly (older shapes)
            # We unwrap, compress the inner payload via the PyO3 dispatcher,
            # and re-wrap so both shapes work.
            #
            # Re-parses from `first_msg_raw` rather than reusing `body`
            # because `body` may be partially mutated if memory injection
            # raised an exception above (in which case `first_msg_raw` is
            # the canonical pre-memory bytes that will actually be sent
            # upstream). The PyO3 binding never raises (passthrough on
            # internal errors), but we wrap the call site in try/except
            # anyway so a JSON-shape edge case can never break the WS
            # session.
            first_frame_rewritten = False
            if self.config.optimize and not _ws_bypass:
                _first_frame_compression_elapsed_ms = 0.0
                try:
                    _preflight_started = time.perf_counter()
                    _ws_auth_mode = classify_auth_mode(ws_headers)
                    try:
                        _send_body = json.loads(first_msg_raw)
                    except json.JSONDecodeError:
                        _send_body = None

                    if isinstance(_send_body, dict):
                        _wrapped = "response" in _send_body and isinstance(
                            _send_body["response"], dict
                        )
                        _inner = _send_body["response"] if _wrapped else _send_body
                        _model = (_inner.get("model") if isinstance(_inner, dict) else None) or ""

                        _preflight_ms = (time.perf_counter() - _preflight_started) * 1000.0
                        _record_ws_compression_timing(
                            "compression_preflight_serialization",
                            _preflight_ms,
                        )
                        _record_ws_compression_overhead(_preflight_ms)
                        _compression_started = time.perf_counter()
                        try:
                            (
                                _new_inner,
                                _modified,
                                _ws_saved,
                                _ws_transforms,
                                _ws_reason,
                                _bytes_before,
                                _bytes_after,
                                _ws_attempted_tokens,
                                _ws_compression_timing,
                            ) = await self._compress_openai_responses_payload_in_executor(
                                _inner,
                                model=_model,
                                request_id=request_id,
                                timeout=_codex_ws_compression_timeout_seconds()
                                if client == "codex"
                                else COMPRESSION_TIMEOUT_SECONDS,
                                client=client,
                            )
                            for _timing_name, _timing_ms in _ws_compression_timing.items():
                                _record_ws_compression_timing(_timing_name, _timing_ms)
                        finally:
                            _first_frame_compression_elapsed_ms = (
                                time.perf_counter() - _compression_started
                            ) * 1000.0
                            _record_ws_compression_timing(
                                "compression_executor_wait_run",
                                _first_frame_compression_elapsed_ms,
                            )
                            _record_ws_compression_overhead(_first_frame_compression_elapsed_ms)
                        record_frame = getattr(
                            getattr(self, "metrics", None), "record_codex_ws_frame", None
                        )
                        if record_frame is not None:
                            record_frame(
                                elapsed_ms=_first_frame_compression_elapsed_ms,
                                bytes_before=_bytes_before,
                                bytes_after=_bytes_after,
                                attempted_tokens=_ws_attempted_tokens,
                                tokens_saved=_ws_saved,
                                modified=_modified,
                                strategy_chain=_codex_ws_strategy_chain(_ws_transforms),
                                final_strategies=_codex_ws_final_strategies(_ws_compression_timing),
                            )
                        # Record transform labels even when the frame bytes are
                        # unchanged: control-arm output-shaper labels
                        # (output_shaper:control:*) must reach the outcome
                        # ledger for A/B baseline accounting.
                        for _t in _ws_transforms:
                            if _t not in transforms_applied:
                                transforms_applied.append(_t)
                        if _modified:
                            if isinstance(_new_inner, dict):
                                _rewrite_started = time.perf_counter()
                                if _wrapped:
                                    _send_body["response"] = _new_inner
                                else:
                                    _send_body = _new_inner
                                first_msg_raw = json.dumps(_send_body)
                                _rewrite_ms = (time.perf_counter() - _rewrite_started) * 1000.0
                                _record_ws_compression_timing(
                                    "compression_payload_rewrite_json_dump",
                                    _rewrite_ms,
                                )
                                _record_ws_compression_overhead(_rewrite_ms)
                                tokens_saved += int(_ws_saved)
                                attempted_input_tokens_total += int(_ws_attempted_tokens)
                                logger.info(
                                    "[%s] WS /v1/responses compressed "
                                    "%d→%d bytes (%d tokens saved, "
                                    "auth_mode=%s, transforms=%s)",
                                    request_id,
                                    _bytes_before,
                                    _bytes_after,
                                    int(_ws_saved),
                                    _ws_auth_mode.value,
                                    transforms_applied,
                                )
                                ws_frames_compressed += 1
                                first_frame_rewritten = True
                        else:
                            _log_ws_passthrough(
                                _ws_reason or "no_compression",
                                frame_index=1,
                                raw_bytes=_bytes_before,
                                frame_type=str(_send_body.get("type") or "response.create"),
                                model=_model or "unknown",
                            )
                    else:
                        _log_ws_passthrough(
                            "first_frame_non_json",
                            frame_index=1,
                            raw_bytes=len(first_msg_raw.encode("utf-8", errors="replace")),
                            frame_type="unknown",
                        )
                except Exception as _ce:
                    _ws_frame_bytes = len(first_msg_raw.encode("utf-8", errors="replace"))
                    if _first_frame_compression_elapsed_ms > 0:
                        record_frame = getattr(
                            getattr(self, "metrics", None), "record_codex_ws_frame", None
                        )
                        if record_frame is not None:
                            record_frame(
                                elapsed_ms=_first_frame_compression_elapsed_ms,
                                bytes_before=_ws_frame_bytes,
                                failed=True,
                            )
                    _timeout_failure = isinstance(_ce, asyncio.TimeoutError)
                    logger.warning(
                        f"[{request_id}] WS /v1/responses compression "
                        f"{'timed out' if _timeout_failure else 'failed'} "
                        f"(bytes={_ws_frame_bytes}): {type(_ce).__name__}: {_ce}"
                    )
                    _log_ws_passthrough(
                        "compression_timeout" if _timeout_failure else "compression_exception",
                        frame_index=1,
                        raw_bytes=_ws_frame_bytes,
                        frame_type="response.create" if body else "unknown",
                        model=str(body.get("model") or "unknown")
                        if isinstance(body, dict)
                        else "unknown",
                    )
                    # Fail-closed protection (default): refuse to forward
                    # oversized frames after a compression failure. Forwarding
                    # the original to the upstream would cause a
                    # context-window-exceeded response that the client
                    # (e.g. Codex) cannot recover from, because Headroom's
                    # earlier successful compressions hid the cumulative
                    # context pressure from the client's auto-compaction
                    # heuristic. Close the client WS with 1009 instead so the
                    # client gets a clear "compact and retry" signal.
                    # See helpers.decide_compression_failure_action for the
                    # decision matrix and env-var overrides.
                    from headroom.proxy.helpers import (
                        decide_compression_failure_action,
                    )

                    _ws_action = decide_compression_failure_action(
                        _ce,
                        _ws_frame_bytes,
                        client=client,
                    )
                    if _ws_action.refuse:
                        logger.error(
                            "[%s] WS /v1/responses REFUSING to forward "
                            "frame after compression failure "
                            "(reason=%s, bytes=%d); closing client "
                            "websocket with 1009 so client can compact "
                            "context and retry. To restore legacy "
                            "passthrough behaviour set "
                            "HEADROOM_WS_FAIL_OPEN_ON_COMPRESSION_FAILURE=1.",
                            request_id,
                            _ws_action.reason,
                            _ws_action.frame_bytes,
                        )
                        termination_cause = "compression_refused"
                        with contextlib.suppress(Exception):
                            await websocket.close(
                                code=1009,
                                reason=(
                                    "headroom: compression "
                                    f"{_ws_action.reason} — please "
                                    "compact context and retry"
                                ),
                            )
                        return
            else:
                _log_ws_passthrough(
                    "bypass_header" if _ws_bypass else "optimize_disabled",
                    frame_index=1,
                    raw_bytes=len(first_msg_raw.encode("utf-8", errors="replace")),
                    frame_type="response.create" if body else "unknown",
                    model=str(body.get("model") or "unknown")
                    if isinstance(body, dict)
                    else "unknown",
                )

            if not _ws_bypass:
                (
                    first_msg_raw,
                    _shape_modified,
                    _shape_labels,
                    _shape_reason,
                ) = _shape_openai_response_create_frame(
                    first_msg_raw,
                    input_tokens=_openai_response_create_frame_input_tokens(
                        first_msg_raw,
                        self.openai_provider,
                    ),
                    conversation_key=f"ws:{session_id}",
                )
                _append_unique_transforms(transforms_applied, _shape_labels)
                if _shape_modified:
                    if not first_frame_rewritten:
                        ws_frames_compressed += 1
                        first_frame_rewritten = True
                    logger.info(
                        "[%s] WS /v1/responses output shaping frame=%d labels=%s",
                        request_id,
                        1,
                        _shape_labels,
                    )

            first_msg_raw = _normalize_ws_response_create_for_upstream(first_msg_raw)
            _first_upstream_body: Any = None
            try:
                _first_upstream_body = json.loads(first_msg_raw)
            except json.JSONDecodeError:
                _first_upstream_body = None
            capture_codex_wire_debug(
                "ws_upstream_first_frame",
                request_id=request_id,
                session_id=session_id,
                transport="websocket",
                direction="headroom_to_upstream",
                url=upstream_url,
                body=_first_upstream_body,
                raw_text=None if _first_upstream_body is not None else first_msg_raw,
                metadata={
                    "frame": 1,
                    "tokens_saved": tokens_saved,
                    "transforms_applied": transforms_applied,
                },
            )

            if ws_connected:
                async with upstream:
                    await upstream.send(_strip_codex_lite_metadata(first_msg_raw))

                    # Unit 3: flag the upstream side flips on seeing
                    # ``response.completed`` so the outer cause
                    # classifier can prefer it over the raw
                    # "upstream iterator ended" default.
                    response_completed_seen = False
                    # Captures the first exception surfaced by the
                    # inner relay ``except`` blocks so the outer
                    # classifier can still tell ``upstream_error``
                    # from ``upstream_disconnect`` / ``response_completed``
                    # even though the halves swallow and log.
                    upstream_relay_error: BaseException | None = None
                    client_relay_error: BaseException | None = None

                    async def _maybe_compress_response_create_frame(
                        raw_msg: str,
                        *,
                        frame_index: int,
                    ) -> tuple[str, bool, str | None]:
                        """Compress a single client→upstream frame
                        when its `type` is `response.create`. Other
                        event types (response.cancel, session.update,
                        etc.) pass through unchanged. Errors are
                        warned and the safest frame available is returned —
                        fail loud in logs, fail safe on the wire.
                        Updates outer-scope ``tokens_saved``,
                        ``transforms_applied``, and
                        ``ws_frames_compressed`` so the session-end
                        log reports cumulative savings across all
                        frames in the WS session.
                        """
                        nonlocal tokens_saved, transforms_applied, attempted_input_tokens_total
                        nonlocal ws_frames_compressed
                        _preflight_started = time.perf_counter()
                        try:
                            parsed_frame = json.loads(raw_msg)
                        except json.JSONDecodeError:
                            _log_ws_passthrough(
                                "non_json",
                                frame_index=frame_index,
                                raw_bytes=len(raw_msg.encode("utf-8", errors="replace")),
                            )
                            return raw_msg, False, "non_json"
                        if (
                            not isinstance(parsed_frame, dict)
                            or parsed_frame.get("type") != "response.create"
                        ):
                            _log_ws_passthrough(
                                "not_response_create",
                                frame_index=frame_index,
                                raw_bytes=len(raw_msg.encode("utf-8", errors="replace")),
                                frame_type=(
                                    parsed_frame.get("type")
                                    if isinstance(parsed_frame, dict)
                                    else type(parsed_frame).__name__
                                ),
                            )
                            return raw_msg, False, "not_response_create"
                        wrapped_frame = isinstance(parsed_frame.get("response"), dict)
                        inner_payload = parsed_frame["response"] if wrapped_frame else parsed_frame
                        if not isinstance(inner_payload, dict):
                            _log_ws_passthrough(
                                "invalid_inner_payload",
                                frame_index=frame_index,
                                raw_bytes=len(raw_msg.encode("utf-8", errors="replace")),
                                frame_type="response.create",
                            )
                            return raw_msg, False, "invalid_inner_payload"
                        store_forced = _ensure_chatgpt_responses_store_false(
                            inner_payload,
                            is_chatgpt_auth=is_chatgpt_auth,
                        )
                        raw_after_store = raw_msg
                        if store_forced:
                            raw_after_store = json.dumps(parsed_frame)
                            logger.info(
                                "[%s] WS Responses: forced store=false for ChatGPT auth frame=%d",
                                request_id,
                                frame_index,
                            )
                        if _ws_bypass:
                            _log_ws_passthrough(
                                "bypass_header",
                                frame_index=frame_index,
                                raw_bytes=len(raw_after_store.encode("utf-8", errors="replace")),
                                frame_type="response.create",
                                model=str(inner_payload.get("model") or "unknown"),
                            )
                            return (
                                raw_after_store,
                                store_forced,
                                "chatgpt_store_false" if store_forced else "bypass_header",
                            )
                        if not self.config.optimize:
                            _log_ws_passthrough(
                                "optimize_disabled",
                                frame_index=frame_index,
                                raw_bytes=len(raw_after_store.encode("utf-8", errors="replace")),
                                frame_type="response.create",
                                model=str(inner_payload.get("model") or "unknown"),
                            )
                            return (
                                raw_after_store,
                                store_forced,
                                "chatgpt_store_false" if store_forced else "optimize_disabled",
                            )
                        frame_compression_elapsed_ms = 0.0
                        try:
                            model_for_frame = inner_payload.get("model") or ""
                            _frame_auth_mode = classify_auth_mode(ws_headers)
                            _preflight_ms = (time.perf_counter() - _preflight_started) * 1000.0
                            _record_ws_compression_timing(
                                "compression_preflight_serialization",
                                _preflight_ms,
                            )
                            _record_ws_compression_overhead(_preflight_ms)
                            _compression_started = time.perf_counter()
                            try:
                                (
                                    new_inner,
                                    modified,
                                    frame_saved,
                                    frame_transforms,
                                    frame_reason,
                                    bytes_before,
                                    bytes_after,
                                    frame_attempted_tokens,
                                    frame_compression_timing,
                                ) = await self._compress_openai_responses_payload_in_executor(
                                    inner_payload,
                                    model=model_for_frame,
                                    request_id=request_id,
                                    timeout=_codex_ws_compression_timeout_seconds()
                                    if client == "codex"
                                    else COMPRESSION_TIMEOUT_SECONDS,
                                    client=client,
                                )
                                for _timing_name, _timing_ms in frame_compression_timing.items():
                                    _record_ws_compression_timing(_timing_name, _timing_ms)
                            except asyncio.TimeoutError as _frame_err:
                                frame_compression_elapsed_ms = (
                                    time.perf_counter() - _compression_started
                                ) * 1000.0
                                if frame_compression_elapsed_ms > 0:
                                    record_frame = getattr(
                                        getattr(self, "metrics", None),
                                        "record_codex_ws_frame",
                                        None,
                                    )
                                    if record_frame is not None:
                                        record_frame(
                                            elapsed_ms=frame_compression_elapsed_ms,
                                            bytes_before=len(
                                                raw_msg.encode("utf-8", errors="replace")
                                            ),
                                            failed=True,
                                        )
                                logger.warning(
                                    "[%s] WS /v1/responses frame compression "
                                    "timed out; forwarding original: %s: %s",
                                    request_id,
                                    type(_frame_err).__name__,
                                    _frame_err,
                                )
                                _log_ws_passthrough(
                                    "compression_timeout",
                                    frame_index=frame_index,
                                    raw_bytes=len(raw_msg.encode("utf-8", errors="replace")),
                                    frame_type="response.create",
                                    model=str(inner_payload.get("model") or "unknown"),
                                )
                                return raw_msg, False, "compression_timeout"
                            finally:
                                frame_compression_elapsed_ms = (
                                    time.perf_counter() - _compression_started
                                ) * 1000.0
                                _record_ws_compression_timing(
                                    "compression_executor_wait_run",
                                    frame_compression_elapsed_ms,
                                )
                                _record_ws_compression_overhead(frame_compression_elapsed_ms)
                            record_frame = getattr(
                                getattr(self, "metrics", None),
                                "record_codex_ws_frame",
                                None,
                            )
                            if record_frame is not None:
                                record_frame(
                                    elapsed_ms=frame_compression_elapsed_ms,
                                    bytes_before=bytes_before,
                                    bytes_after=bytes_after,
                                    attempted_tokens=frame_attempted_tokens,
                                    tokens_saved=frame_saved,
                                    modified=modified,
                                    strategy_chain=_codex_ws_strategy_chain(frame_transforms),
                                    final_strategies=_codex_ws_final_strategies(
                                        frame_compression_timing
                                    ),
                                )
                        except Exception as _frame_err:
                            if frame_compression_elapsed_ms > 0:
                                record_frame = getattr(
                                    getattr(self, "metrics", None),
                                    "record_codex_ws_frame",
                                    None,
                                )
                                if record_frame is not None:
                                    record_frame(
                                        elapsed_ms=frame_compression_elapsed_ms,
                                        bytes_before=len(raw_msg.encode("utf-8", errors="replace")),
                                        failed=True,
                                    )
                            logger.warning(
                                "[%s] WS /v1/responses frame compression "
                                "failed; forwarding original: %s: %s",
                                request_id,
                                type(_frame_err).__name__,
                                _frame_err,
                            )
                            _log_ws_passthrough(
                                "compression_exception",
                                frame_index=frame_index,
                                raw_bytes=len(raw_after_store.encode("utf-8", errors="replace")),
                                frame_type="response.create",
                                model=str(inner_payload.get("model") or "unknown"),
                            )
                        # Record transform labels even when the frame bytes are
                        # unchanged: control-arm output-shaper labels
                        # (output_shaper:control:*) must reach the outcome
                        # ledger for A/B baseline accounting.
                        for t in frame_transforms:
                            if t not in transforms_applied:
                                transforms_applied.append(t)
                        return (
                            raw_after_store,
                            store_forced,
                            "chatgpt_store_false" if store_forced else "compression_exception",
                        )
                        if not modified:
                            reason = frame_reason or "no_compression"
                            _log_ws_passthrough(
                                reason,
                                frame_index=frame_index,
                                raw_bytes=len(raw_after_store.encode("utf-8", errors="replace")),
                                frame_type="response.create",
                                model=str(inner_payload.get("model") or "unknown"),
                            )
                            return (
                                raw_after_store,
                                store_forced,
                                "chatgpt_store_false" if store_forced else reason,
                            )
                        if not isinstance(new_inner, dict):
                            _log_ws_passthrough(
                                "compressed_payload_not_dict",
                                frame_index=frame_index,
                                raw_bytes=len(raw_after_store.encode("utf-8", errors="replace")),
                                frame_type="response.create",
                                model=str(inner_payload.get("model") or "unknown"),
                            )
                            return (
                                raw_after_store,
                                store_forced,
                                "chatgpt_store_false"
                                if store_forced
                                else "compressed_payload_not_dict",
                            )
                        _ensure_chatgpt_responses_store_false(
                            new_inner,
                            is_chatgpt_auth=is_chatgpt_auth,
                        )
                        if wrapped_frame:
                            _rewrite_started = time.perf_counter()
                            parsed_frame["response"] = new_inner
                            rewritten = json.dumps(parsed_frame)
                        else:
                            _rewrite_started = time.perf_counter()
                            rewritten = json.dumps(new_inner)
                        _rewrite_ms = (time.perf_counter() - _rewrite_started) * 1000.0
                        _record_ws_compression_timing(
                            "compression_payload_rewrite_json_dump",
                            _rewrite_ms,
                        )
                        _record_ws_compression_overhead(_rewrite_ms)
                        tokens_saved += int(frame_saved)
                        attempted_input_tokens_total += int(frame_attempted_tokens)
                        ws_frames_compressed += 1
                        logger.info(
                            "[%s] WS /v1/responses frame compressed "
                            "%d→%d bytes (%d tokens saved, "
                            "auth_mode=%s, frame=%d)",
                            request_id,
                            bytes_before,
                            bytes_after,
                            int(frame_saved),
                            _frame_auth_mode.value,
                            ws_frames_compressed,
                        )
                        return rewritten, True, frame_reason or "compressed"

                    async def _client_to_upstream() -> None:
                        nonlocal client_relay_error, ws_response_create_frames
                        nonlocal ws_client_frames_total, ws_cancel_frames
                        nonlocal ws_frames_compressed
                        nonlocal ws_last_client_frame_type, ws_client_disconnect_seen
                        client_frame_index = 1
                        try:
                            while True:
                                msg = await websocket.receive_text()
                                client_frame_index += 1
                                ws_client_frames_total += 1
                                if session_handle is not None:
                                    session_handle.mark_activity()
                                _inbound_frame_body: Any = None
                                try:
                                    _inbound_frame_body = json.loads(msg)
                                except json.JSONDecodeError:
                                    _inbound_frame_body = None
                                ws_last_client_frame_type = (
                                    str(_inbound_frame_body.get("type") or "unknown")
                                    if isinstance(_inbound_frame_body, dict)
                                    else "non_json"
                                )
                                if ws_last_client_frame_type == "response.cancel":
                                    ws_cancel_frames += 1
                                    logger.info(
                                        "[%s] WS client sent response.cancel "
                                        "session_id=%s frame=%d cancels=%d",
                                        request_id,
                                        session_id,
                                        client_frame_index,
                                        ws_cancel_frames,
                                    )
                                else:
                                    logger.debug(
                                        "[%s] WS client frame session_id=%s frame=%d type=%s",
                                        request_id,
                                        session_id,
                                        client_frame_index,
                                        ws_last_client_frame_type,
                                    )
                                capture_codex_wire_debug(
                                    "ws_inbound_client_frame",
                                    request_id=request_id,
                                    session_id=session_id,
                                    transport="websocket",
                                    direction="client_to_headroom",
                                    url=_ws_url,
                                    body=_inbound_frame_body,
                                    raw_text=None if _inbound_frame_body is not None else msg,
                                    metadata={"frame": client_frame_index},
                                )
                                if (
                                    isinstance(_inbound_frame_body, dict)
                                    and _inbound_frame_body.get("type") == "response.create"
                                ):
                                    ws_response_create_frames += 1
                                    msg = await _prepare_memory_frame(_inbound_frame_body, msg)
                                (
                                    msg,
                                    _frame_modified,
                                    _frame_reason,
                                ) = await _maybe_compress_response_create_frame(
                                    msg,
                                    frame_index=client_frame_index,
                                )
                                if not _ws_bypass:
                                    (
                                        msg,
                                        _shape_modified,
                                        _shape_labels,
                                        _shape_reason,
                                    ) = _shape_openai_response_create_frame(
                                        msg,
                                        input_tokens=_openai_response_create_frame_input_tokens(
                                            msg,
                                            self.openai_provider,
                                        ),
                                        conversation_key=f"ws:{session_id}",
                                    )
                                    _append_unique_transforms(
                                        transforms_applied,
                                        _shape_labels,
                                    )
                                    if _shape_modified:
                                        if not _frame_modified:
                                            ws_frames_compressed += 1
                                        _frame_modified = True
                                        logger.info(
                                            "[%s] WS /v1/responses output shaping frame=%d labels=%s",
                                            request_id,
                                            client_frame_index,
                                            _shape_labels,
                                        )

                                msg = _normalize_ws_response_create_for_upstream(msg)
                                _outbound_frame_body: Any = None
                                try:
                                    _outbound_frame_body = json.loads(msg)
                                except json.JSONDecodeError:
                                    _outbound_frame_body = None
                                capture_codex_wire_debug(
                                    "ws_upstream_client_frame",
                                    request_id=request_id,
                                    session_id=session_id,
                                    transport="websocket",
                                    direction="headroom_to_upstream",
                                    url=upstream_url,
                                    body=_outbound_frame_body,
                                    raw_text=None if _outbound_frame_body is not None else msg,
                                    metadata={
                                        "frame": client_frame_index,
                                        "tokens_saved_total": tokens_saved,
                                        "transforms_applied": transforms_applied,
                                    },
                                )
                                await upstream.send(_strip_codex_lite_metadata(msg))
                        except asyncio.CancelledError:
                            # Explicit cancel from the outer
                            # orchestrator — re-raise so
                            # ``t.cancelled()`` and ``t.exception()``
                            # behave correctly in the caller.
                            raise
                        except Exception as relay_err:
                            # Surface real errors to the classifier
                            # without re-raising (existing fork
                            # behavior: log and return so the
                            # partner task can be cancelled
                            # deterministically).
                            if "WebSocketDisconnect" not in type(relay_err).__name__:
                                client_relay_error = relay_err
                                logger.debug(
                                    f"[{request_id}] WS client→upstream relay ended: {relay_err}"
                                )
                            else:
                                ws_client_disconnect_seen = True
                                logger.info(
                                    "[%s] WS client disconnected session_id=%s "
                                    "frames=%d cancels=%d last_type=%s",
                                    request_id,
                                    session_id,
                                    ws_client_frames_total,
                                    ws_cancel_frames,
                                    ws_last_client_frame_type,
                                )
                            with contextlib.suppress(Exception):
                                await upstream.close()

                    async def _upstream_to_client() -> None:
                        """Relay upstream→client with transparent memory tool handling.

                        Uses a buffer-then-decide approach:
                        1. Buffer events until first output item arrives
                        2. If first output is a memory tool → suppress entire response,
                           execute tools silently, send continuation upstream
                        3. If first output is non-memory → flush buffer, stream normally
                        4. Continuation response events are relayed to Codex seamlessly

                        This prevents orphaned response.created events from confusing Codex.
                        """
                        from headroom.proxy.memory_handler import MEMORY_TOOL_NAMES

                        # Unit 3: surface response.completed observation
                        # to the outer scope so the termination-cause
                        # classifier can prefer ``response_completed``
                        # over ``upstream_disconnect``.
                        nonlocal response_completed_seen
                        nonlocal upstream_relay_error
                        nonlocal ws_input_tokens_total, ws_output_tokens_total
                        nonlocal ws_cache_read_tokens_total, ws_cache_write_tokens_total
                        nonlocal ws_uncached_input_tokens_total
                        nonlocal ws_recorded_input_tokens_total
                        nonlocal ws_recorded_output_tokens_total
                        nonlocal ws_recorded_cache_read_tokens_total
                        nonlocal ws_recorded_cache_write_tokens_total
                        nonlocal ws_recorded_uncached_input_tokens_total
                        nonlocal ws_recorded_tokens_saved_total
                        nonlocal ws_recorded_overhead_ms_total, ws_recorded_ttfb_ms
                        nonlocal ws_upstream_frames_total, ws_last_upstream_frame_type
                        nonlocal ws_ttfb_ms

                        memory_enabled = bool(
                            self.memory_handler and memory_user_id and ws_memory_tools_allowed
                        )

                        # Per-response state (reset after each response.completed)
                        event_buffer: list[str] = []
                        decided = False
                        suppress_response = False
                        pending_fcs: list[dict[str, Any]] = []
                        resp_id: str | None = None

                        def _reset() -> None:
                            nonlocal decided, suppress_response, resp_id
                            event_buffer.clear()
                            decided = False
                            suppress_response = False
                            pending_fcs.clear()
                            resp_id = None

                        response_started_ms: float | None = None

                        async def _record_ws_response_metrics() -> None:
                            """Record one completed Responses turn on long-lived WS sessions."""
                            nonlocal ws_recorded_input_tokens_total
                            nonlocal ws_recorded_output_tokens_total
                            nonlocal ws_recorded_cache_read_tokens_total
                            nonlocal ws_recorded_cache_write_tokens_total
                            nonlocal ws_recorded_uncached_input_tokens_total
                            nonlocal ws_recorded_tokens_saved_total
                            nonlocal ws_recorded_attempted_input_tokens_total
                            nonlocal ws_recorded_overhead_ms_total, ws_recorded_ttfb_ms

                            input_delta = ws_input_tokens_total - ws_recorded_input_tokens_total
                            output_delta = ws_output_tokens_total - ws_recorded_output_tokens_total
                            cache_read_delta = (
                                ws_cache_read_tokens_total - ws_recorded_cache_read_tokens_total
                            )
                            cache_write_delta = (
                                ws_cache_write_tokens_total - ws_recorded_cache_write_tokens_total
                            )
                            uncached_delta = (
                                ws_uncached_input_tokens_total
                                - ws_recorded_uncached_input_tokens_total
                            )
                            # Usage-less turn (cancelled/failed before
                            # response.completed): defer its savings — see
                            # _deferrable_savings_delta. The recorded total
                            # advances by the recorded delta below, so
                            # deferred savings stay pending and land with the
                            # next usage-carrying turn.
                            saved_delta = _deferrable_savings_delta(
                                input_delta,
                                tokens_saved - ws_recorded_tokens_saved_total,
                            )
                            attempted_delta = (
                                attempted_input_tokens_total
                                - ws_recorded_attempted_input_tokens_total
                            )
                            (
                                overhead_delta_ms,
                                ttfb_for_record_ms,
                                dashboard_pipeline_timing,
                            ) = _prepare_ws_performance_metrics()
                            if (
                                input_delta <= 0
                                and output_delta <= 0
                                and cache_read_delta <= 0
                                and cache_write_delta <= 0
                                and uncached_delta <= 0
                                and saved_delta <= 0
                                and attempted_delta <= 0
                                and overhead_delta_ms <= 0
                                and ttfb_for_record_ms <= 0
                            ):
                                return

                            model_for_metrics = str(body.get("model") or "unknown")
                            latency_ms = (
                                (time.perf_counter() * 1000.0 - response_started_ms)
                                if response_started_ms is not None
                                else 0.0
                            )
                            # Per-turn record: delta values capture
                            # this turn's contribution since the
                            # Codex WS handler accumulates session
                            # totals. Pre-refactor this site
                            # emitted only metrics + cost_tracker
                            # — no RequestLog, no PERF — so Codex
                            # traffic was invisible to
                            # ``headroom perf`` and the recent-
                            # requests feed. Funnel restores all
                            # four effects uniformly per turn. Per-
                            # turn outcomes carry ``ws_tags`` (the
                            # `x-headroom-tag-*` headers extracted
                            # at the WS upgrade) so dashboards can
                            # slice WS turns by tag — same surface
                            # as HTTP turns.
                            await self._record_request_outcome(
                                RequestOutcome(
                                    request_id=request_id,
                                    provider="openai",
                                    model=model_for_metrics,
                                    original_tokens=max(0, input_delta) + max(0, saved_delta),
                                    optimized_tokens=max(0, input_delta),
                                    output_tokens=max(0, output_delta),
                                    tokens_saved=max(0, saved_delta),
                                    attempted_input_tokens=max(0, attempted_delta),
                                    cache_read_tokens=max(0, cache_read_delta),
                                    cache_write_tokens=max(0, cache_write_delta),
                                    uncached_input_tokens=max(0, uncached_delta),
                                    total_latency_ms=latency_ms,
                                    overhead_ms=overhead_delta_ms,
                                    ttfb_ms=ttfb_for_record_ms,
                                    pipeline_timing=dashboard_pipeline_timing,
                                    transforms_applied=tuple(transforms_applied),
                                    num_messages=len(
                                        body.get("messages") or body.get("input") or []
                                    )
                                    if isinstance(body, dict)
                                    else 0,
                                    tags=ws_tags,
                                    client=client,
                                )
                            )

                            # The PERF line for this Codex turn is emitted once by
                            # the outcome funnel above (emit_request_outcome), using
                            # the same per-turn deltas — a second emit here duplicated
                            # every WS turn in `headroom perf` (double saved + requests).

                            ws_recorded_input_tokens_total = ws_input_tokens_total
                            ws_recorded_output_tokens_total = ws_output_tokens_total
                            ws_recorded_cache_read_tokens_total = ws_cache_read_tokens_total
                            ws_recorded_cache_write_tokens_total = ws_cache_write_tokens_total
                            ws_recorded_uncached_input_tokens_total = ws_uncached_input_tokens_total
                            # Advance by the recorded delta, not to the live
                            # total: when the usage-less guard above zeroed
                            # saved_delta, the un-recorded savings must stay
                            # pending for the next usage-carrying turn.
                            # Equivalent to `= tokens_saved` whenever the
                            # delta was recorded as computed.
                            ws_recorded_tokens_saved_total += saved_delta
                            ws_recorded_attempted_input_tokens_total = attempted_input_tokens_total
                            ws_recorded_overhead_ms_total = _current_ws_overhead_ms()
                            ws_recorded_compression_timing_totals.update(
                                ws_compression_timing_totals
                            )
                            if ttfb_for_record_ms > 0:
                                ws_recorded_ttfb_ms = True

                        # The retry-loop variable is safe to close over here:
                        # ``_upstream_to_client`` is defined and awaited within
                        # a single iteration and never escapes.
                        _first_event_started_at = _upstream_first_event_started  # noqa: B023

                        try:
                            upstream_frame_index = 0
                            async for msg in upstream:
                                upstream_frame_index += 1
                                ws_upstream_frames_total += 1
                                if session_handle is not None:
                                    session_handle.mark_activity()
                                if (
                                    _first_event_started_at is not None
                                    and "upstream_first_event" not in stage_timer
                                ):
                                    if ws_ttfb_ms is None:
                                        ws_ttfb_ms = (
                                            time.perf_counter() - session_started_at
                                        ) * 1000.0
                                    stage_timer.record(
                                        "upstream_first_event",
                                        (time.perf_counter() - _first_event_started_at) * 1000.0,
                                    )
                                if isinstance(msg, bytes):
                                    ws_last_upstream_frame_type = "binary"
                                    capture_codex_wire_debug(
                                        "ws_upstream_binary_frame",
                                        request_id=request_id,
                                        session_id=session_id,
                                        transport="websocket",
                                        direction="upstream_to_headroom",
                                        url=upstream_url,
                                        metadata={
                                            "frame": upstream_frame_index,
                                            "byte_count": len(msg),
                                        },
                                    )
                                    await websocket.send_bytes(msg)
                                    continue
                                msg_str = msg if isinstance(msg, str) else str(msg)
                                _upstream_frame_body: Any = None
                                try:
                                    _upstream_frame_body = json.loads(msg_str)
                                except json.JSONDecodeError:
                                    _upstream_frame_body = None
                                capture_codex_wire_debug(
                                    "ws_upstream_text_frame",
                                    request_id=request_id,
                                    session_id=session_id,
                                    transport="websocket",
                                    direction="upstream_to_headroom",
                                    url=upstream_url,
                                    body=_upstream_frame_body,
                                    raw_text=None if _upstream_frame_body is not None else msg_str,
                                    metadata={"frame": upstream_frame_index},
                                )

                                # Parse event
                                try:
                                    event = json.loads(msg_str)
                                except (json.JSONDecodeError, TypeError):
                                    ws_last_upstream_frame_type = "non_json"
                                    await websocket.send_text(msg_str)
                                    continue

                                event_type = event.get("type", "")
                                ws_last_upstream_frame_type = str(event_type or "unknown")
                                logger.debug(
                                    "[%s] WS upstream frame session_id=%s frame=%d type=%s",
                                    request_id,
                                    session_id,
                                    upstream_frame_index,
                                    ws_last_upstream_frame_type,
                                )
                                if event_type == "response.created":
                                    response_started_ms = time.perf_counter() * 1000.0
                                (
                                    usage_input_tokens,
                                    usage_output_tokens,
                                    usage_cache_read_tokens,
                                    usage_cache_write_tokens,
                                    usage_uncached_tokens,
                                ) = _extract_responses_usage(event)
                                if usage_input_tokens or usage_output_tokens:
                                    ws_input_tokens_total += usage_input_tokens
                                    ws_output_tokens_total += usage_output_tokens
                                    ws_cache_read_tokens_total += usage_cache_read_tokens
                                    ws_cache_write_tokens_total += usage_cache_write_tokens
                                    ws_uncached_input_tokens_total += usage_uncached_tokens

                                if not memory_enabled:
                                    if event_type == "response.completed":
                                        response_completed_seen = True
                                        await _record_ws_response_metrics()
                                    await websocket.send_text(msg_str)
                                    continue

                                # --- Phase 1: Buffer until first output item ---
                                if not decided:
                                    event_buffer.append(msg_str)

                                    if event_type == "response.output_item.added":
                                        item = event.get("item", {})
                                        if (
                                            item.get("type") == "function_call"
                                            and item.get("name") in MEMORY_TOOL_NAMES
                                        ):
                                            # Memory tool first → suppress entire response
                                            suppress_response = True
                                            decided = True
                                            event_buffer.clear()
                                            logger.info(
                                                f"[{request_id}] WS Memory: Detected "
                                                f"{item.get('name')} — suppressing response"
                                            )
                                        else:
                                            # Non-memory first → flush buffer, pass through
                                            decided = True
                                            for buf in event_buffer:
                                                await websocket.send_text(buf)
                                            event_buffer.clear()

                                    elif event_type == "response.completed":
                                        # No output items at all — flush
                                        decided = True
                                for buf in event_buffer:
                                    await websocket.send_text(buf)
                                event_buffer.clear()
                                await _record_ws_response_metrics()
                                _reset()
                                response_completed_seen = True

                                continue

                                # --- Phase 2a: Suppress mode (memory response) ---
                                if suppress_response:
                                    if event_type == "response.output_item.done":
                                        item = event.get("item", {})
                                        if (
                                            item.get("type") == "function_call"
                                            and item.get("name") in MEMORY_TOOL_NAMES
                                        ):
                                            pending_fcs.append(item)

                                    elif event_type == "response.completed":
                                        response_completed_seen = True
                                        await _record_ws_response_metrics()
                                        resp = event.get("response", {})
                                        resp_id = resp.get("id")

                                        if pending_fcs:
                                            logger.info(
                                                f"[{request_id}] WS Memory: Executing "
                                                f"{len(pending_fcs)} tool(s) transparently"
                                            )

                                            # Execute memory tool calls
                                            tool_outputs: list[dict[str, Any]] = []
                                            for fc in pending_fcs:
                                                call_id = fc.get("call_id", fc.get("id", ""))
                                                fc_name = fc.get("name", "")
                                                args_str = fc.get("arguments", "{}")
                                                try:
                                                    fc_args = json.loads(args_str)
                                                except json.JSONDecodeError:
                                                    fc_args = {}

                                                await self.memory_handler._ensure_initialized()
                                                if self.memory_handler._backend:
                                                    result = await self.memory_handler._execute_memory_tool(
                                                        fc_name,
                                                        fc_args,
                                                        memory_user_id,
                                                        "openai",
                                                    )
                                                else:
                                                    result = json.dumps(
                                                        {"error": "backend not ready"}
                                                    )

                                                tool_outputs.append(
                                                    {
                                                        "type": "function_call_output",
                                                        "call_id": call_id,
                                                        "output": result,
                                                    }
                                                )
                                                logger.info(
                                                    f"[{request_id}] WS Memory: Executed "
                                                    f"{fc_name} for user {memory_user_id}"
                                                )

                                            # Send continuation upstream
                                            cont: dict[str, Any] = {
                                                "type": "response.create",
                                                "response": {"input": tool_outputs},
                                            }
                                            if resp_id:
                                                cont["response"]["previous_response_id"] = resp_id
                                            await upstream.send(
                                                _normalize_ws_response_create_for_upstream(
                                                    json.dumps(cont)
                                                )
                                            )
                                            logger.info(
                                                f"[{request_id}] WS Memory: Sent continuation "
                                                f"with {len(tool_outputs)} result(s)"
                                            )

                                        _reset()
                                    # All events suppressed in this mode
                                    continue

                                # --- Phase 2b: Pass-through mode ---
                                await websocket.send_text(msg_str)

                        except asyncio.CancelledError:
                            raise
                        except Exception as relay_err:
                            if "WebSocketDisconnect" not in type(relay_err).__name__:
                                # Capture for the outer classifier
                                # so ``upstream_error`` can be
                                # distinguished from a clean
                                # upstream disconnect.
                                upstream_relay_error = relay_err
                                _upstream_close_code = getattr(relay_err, "code", None) or getattr(
                                    getattr(relay_err, "rcvd", None), "code", None
                                )
                                _upstream_close_reason = (
                                    getattr(relay_err, "reason", None)
                                    or getattr(getattr(relay_err, "rcvd", None), "reason", None)
                                    or str(relay_err)
                                )
                                logger.warning(
                                    "[%s] WS upstream→client relay ended: %s code=%s reason=%s",
                                    request_id,
                                    type(relay_err).__name__,
                                    _upstream_close_code,
                                    _upstream_close_reason,
                                )
                                if os.environ.get(
                                    "HEADROOM_OPENAI_WS_PROPAGATE_UPSTREAM_CLOSE", ""
                                ).strip().lower() in (
                                    "1",
                                    "true",
                                    "yes",
                                    "on",
                                ):
                                    _client_close_code = (
                                        int(_upstream_close_code)
                                        if isinstance(_upstream_close_code, int)
                                        and 1000 <= int(_upstream_close_code) <= 4999
                                        else 1011
                                    )
                                    with contextlib.suppress(Exception):
                                        await websocket.close(
                                            code=_client_close_code,
                                            reason=str(_upstream_close_reason)[:120],
                                        )
                        finally:
                            with contextlib.suppress(Exception):
                                await websocket.close()

                    # --- Unit 3: deterministic relay-task cancellation ---
                    # Spawn each half as a named task so we can:
                    #   (a) attach them to the session registry for
                    #       ``/debug/ws-sessions``,
                    #   (b) cancel the survivor explicitly when the
                    #       first one exits, and
                    #   (c) classify the termination cause for the
                    #       duration histogram.
                    client_task = asyncio.create_task(
                        _client_to_upstream(),
                        name=f"codex-ws-c2u-{session_id}",
                    )
                    upstream_task = asyncio.create_task(
                        _upstream_to_client(),
                        name=f"codex-ws-u2c-{session_id}",
                    )
                    relay_tasks = [client_task, upstream_task]
                    if ws_sessions is not None:
                        ws_sessions.attach_tasks(session_id, relay_tasks)
                        metrics_for_tasks = getattr(self, "metrics", None)
                        if metrics_for_tasks is not None and hasattr(
                            metrics_for_tasks, "inc_active_relay_tasks"
                        ):
                            try:
                                metrics_for_tasks.inc_active_relay_tasks(len(relay_tasks))
                            except Exception:  # pragma: no cover - defensive
                                pass

                    try:
                        done, pending = await asyncio.wait(
                            {client_task, upstream_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        # Cancel the survivor so we don't leak the
                        # partner task. Suppress the CancelledError
                        # we just raised ourselves — any *other*
                        # exception from the cancelled task is
                        # already logged inside its own try/except.
                        for t in pending:
                            t.cancel()
                        if pending:
                            with contextlib.suppress(asyncio.CancelledError):
                                await asyncio.gather(*pending, return_exceptions=True)

                        # Classify termination cause from whichever
                        # task completed first. ``CancelledError``
                        # can show up on the "done" side if the
                        # handler itself was cancelled from outside
                        # (e.g. server shutdown).
                        for t in done:
                            exc = None
                            # Cancelled tasks raise CancelledError from
                            # .exception(); surface it explicitly so the
                            # downstream ``isinstance(exc, CancelledError)``
                            # branches actually run. For any other
                            # unexpected state (``InvalidStateError`` if
                            # the task somehow isn't done — shouldn't
                            # happen post-gather but defensive), we
                            # suppress and leave ``exc`` as ``None``.
                            if t.cancelled():
                                exc = asyncio.CancelledError()
                            else:
                                with contextlib.suppress(asyncio.InvalidStateError):
                                    exc = t.exception()
                            task_name = t.get_name() or ""
                            if t is client_task:
                                if client_relay_error is not None:
                                    termination_cause = "client_error"
                                elif exc is None:
                                    termination_cause = "client_disconnect"
                                elif isinstance(exc, asyncio.CancelledError):
                                    termination_cause = "client_disconnect"
                                else:
                                    # Distinguish legitimate client
                                    # disconnect exceptions from
                                    # real errors: WebSocketDisconnect
                                    # is a normal client exit.
                                    if "WebSocketDisconnect" in type(exc).__name__:
                                        termination_cause = "client_disconnect"
                                    else:
                                        termination_cause = "client_error"
                            elif t is upstream_task:
                                if upstream_relay_error is not None:
                                    termination_cause = "upstream_error"
                                    logger.debug(
                                        f"[{request_id}] WS relay {task_name} "
                                        f"raised: {upstream_relay_error!r}"
                                    )
                                elif exc is None:
                                    termination_cause = (
                                        "response_completed"
                                        if response_completed_seen
                                        else "upstream_disconnect"
                                    )
                                elif isinstance(exc, asyncio.CancelledError):
                                    termination_cause = "upstream_disconnect"
                                else:
                                    termination_cause = "upstream_error"
                                    logger.debug(
                                        f"[{request_id}] WS relay {task_name} raised: {exc!r}"
                                    )
                        if (
                            ws_cancel_frames > 0
                            and not response_completed_seen
                            and termination_cause
                            in {"upstream_disconnect", "client_disconnect", "unknown"}
                        ):
                            termination_cause = "client_cancel"
                    finally:
                        # In case anything above raised before the
                        # cancel-and-await loop ran.
                        for t in relay_tasks:
                            if not t.done():
                                t.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await asyncio.gather(*relay_tasks, return_exceptions=True)

                    logger.info(
                        "[%s] WS /v1/responses completed "
                        "(tokens_saved=%d, cause=%s, client_frames=%d, upstream_frames=%d, "
                        "cancel_frames=%d, client_disconnect=%s, last_client_type=%s, "
                        "last_upstream_type=%s)",
                        request_id,
                        tokens_saved,
                        termination_cause,
                        ws_client_frames_total,
                        ws_upstream_frames_total,
                        ws_cancel_frames,
                        ws_client_disconnect_seen,
                        ws_last_client_frame_type,
                        ws_last_upstream_frame_type,
                    )
            else:
                # WS upgrade failed (HTTP 500 from OpenAI is common).
                # Fall back to HTTP POST streaming and relay SSE events
                # back over the client WebSocket transparently.
                ws_err = ws_last_err or RuntimeError("unknown websocket connect failure")
                _ws_detail = str(ws_err)
                if hasattr(ws_err, "response"):
                    resp_body = getattr(getattr(ws_err, "response", None), "body", b"")
                    if resp_body:
                        from headroom.proxy.helpers import safe_decode_for_logging

                        _ws_detail += f" | {safe_decode_for_logging(resp_body, max_bytes=300)}"
                logger.warning(
                    f"[{request_id}] WS upstream failed ({_ws_detail}), "
                    f"falling back to HTTP POST streaming"
                )
                await self._ws_http_fallback(
                    websocket, body, first_msg_raw, upstream_headers, request_id
                )

            # ── WS session-end metric + RequestLog ──────────────────
            #
            # Unconditional (was previously gated on `tokens_saved>0`,
            # which made first-frame no-changes invisible). We record
            # one entry per WS session that aggregates `tokens_saved`
            # across every `response.create` frame compressed by the
            # first-frame block + `_maybe_compress_response_create_frame`.
            # The RequestLog entry mirrors the streaming.py /
            # anthropic.py shape so /transformations/feed surfaces
            # Codex WS turns.
            ws_session_duration_ms = (time.perf_counter() - session_started_at) * 1000.0
            ws_inner_for_telemetry: dict[str, Any] = (
                body.get("response", body) if isinstance(body, dict) else {}
            )
            if not isinstance(ws_inner_for_telemetry, dict):
                ws_inner_for_telemetry = {}
            model_name = (
                ws_inner_for_telemetry.get("model")
                or (body.get("model") if isinstance(body, dict) else None)
                or "unknown"
            )
            _final_auth_mode = classify_auth_mode(ws_headers)
            residual_input_tokens = max(0, ws_input_tokens_total - ws_recorded_input_tokens_total)
            residual_output_tokens = max(
                0, ws_output_tokens_total - ws_recorded_output_tokens_total
            )
            residual_cache_read_tokens = max(
                0, ws_cache_read_tokens_total - ws_recorded_cache_read_tokens_total
            )
            residual_cache_write_tokens = max(
                0, ws_cache_write_tokens_total - ws_recorded_cache_write_tokens_total
            )
            residual_uncached_input_tokens = max(
                0,
                ws_uncached_input_tokens_total - ws_recorded_uncached_input_tokens_total,
            )
            # Savings deferred from usage-less turns (per-turn guard in
            # _record_ws_response_metrics) land here when the session closes
            # before another usage frame arrives. With no residual input there
            # is no spend to pair them with — drop them rather than write the
            # savings-with-zero-spend checkpoint the per-turn guard prevents.
            residual_tokens_saved = _deferrable_savings_delta(
                residual_input_tokens,
                max(0, tokens_saved - ws_recorded_tokens_saved_total),
            )
            residual_attempted_input_tokens = max(
                0,
                attempted_input_tokens_total - ws_recorded_attempted_input_tokens_total,
            )
            (
                final_overhead_delta_ms,
                final_ttfb_ms,
                final_pipeline_timing,
            ) = _prepare_ws_performance_metrics()
            ws_session_tags = {
                **(ws_tags or {}),
                "auth_mode": _final_auth_mode.value,
                "endpoint": "responses_ws",
                "compression_scope": "live_zone",
                "cache_policy": "prefix_safe",
                "transport": "websocket",
                "route": "chatgpt_subscription" if is_chatgpt_auth else "openai_api",
                "ws_response_create_frames": str(ws_response_create_frames),
                "ws_frames_compressed": str(ws_frames_compressed),
                "ws_client_frames_total": str(ws_client_frames_total),
                "ws_upstream_frames_total": str(ws_upstream_frames_total),
                "ws_cancel_frames": str(ws_cancel_frames),
                "ws_last_client_frame_type": ws_last_client_frame_type,
                "ws_last_upstream_frame_type": ws_last_upstream_frame_type,
                "ws_client_disconnect_seen": str(ws_client_disconnect_seen),
                "ws_termination_cause": termination_cause,
                "cache_read_tokens": str(ws_cache_read_tokens_total),
                "cache_write_tokens": str(ws_cache_write_tokens_total),
                "uncached_input_tokens": str(ws_uncached_input_tokens_total),
            }
            if (
                residual_input_tokens > 0
                or residual_output_tokens > 0
                or residual_tokens_saved > 0
                or residual_cache_read_tokens > 0
                or residual_cache_write_tokens > 0
                or residual_uncached_input_tokens > 0
                or residual_attempted_input_tokens > 0
                or final_overhead_delta_ms > 0
                or final_ttfb_ms > 0
            ):
                # Session-end residual: tokens not captured by any
                # per-turn record (e.g. signaling frames after the
                # last response.completed). The funnel emits the full
                # bookkeeping quartet for the residual. This is the
                # session's ONLY end-of-session RequestLog row: per-turn
                # rows already carry the session's tokens as deltas, so a
                # separate cumulative session-summary row would double-count
                # Codex WS traffic in the recent-requests feed and every
                # log-derived stat (savings, throughput).
                from headroom.proxy.helpers import compute_turn_id

                ws_messages_for_log: list[dict[str, Any]] = []
                ws_input_for_log = ws_inner_for_telemetry.get("input")
                ws_instructions_for_log = ws_inner_for_telemetry.get("instructions")
                if isinstance(ws_instructions_for_log, str) and ws_instructions_for_log:
                    ws_messages_for_log.append(
                        {"role": "system", "content": ws_instructions_for_log}
                    )
                if isinstance(ws_input_for_log, str) and ws_input_for_log:
                    ws_messages_for_log.append({"role": "user", "content": ws_input_for_log})
                await self._record_request_outcome(
                    RequestOutcome(
                        request_id=request_id,
                        provider="openai",
                        model=model_name,
                        original_tokens=residual_input_tokens + residual_tokens_saved,
                        optimized_tokens=residual_input_tokens,
                        output_tokens=residual_output_tokens,
                        tokens_saved=residual_tokens_saved,
                        attempted_input_tokens=residual_attempted_input_tokens,
                        cache_read_tokens=residual_cache_read_tokens,
                        cache_write_tokens=residual_cache_write_tokens,
                        uncached_input_tokens=residual_uncached_input_tokens,
                        total_latency_ms=ws_session_duration_ms,
                        overhead_ms=final_overhead_delta_ms,
                        ttfb_ms=final_ttfb_ms,
                        pipeline_timing=final_pipeline_timing,
                        transforms_applied=tuple(transforms_applied),
                        tags=ws_session_tags,
                        client=client,
                        request_messages=ws_messages_for_log
                        if getattr(self.config, "log_full_messages", False)
                        else None,
                        turn_id=compute_turn_id(
                            model_name,
                            ws_instructions_for_log,
                            ws_messages_for_log,
                        ),
                    )
                )
                ws_recorded_overhead_ms_total = _current_ws_overhead_ms()
                if final_ttfb_ms > 0:
                    ws_recorded_ttfb_ms = True

        except Exception as e:
            if "WebSocketDisconnect" in type(e).__name__:
                # Unit 3: client dropped the socket before or during
                # relay. The registry classifier may already have set
                # ``client_disconnect`` via the relay task exit path;
                # preserve that, otherwise set it here.
                if termination_cause == "unknown":
                    termination_cause = "client_disconnect"
            else:
                # Extract response body from websockets InvalidStatus for better debugging
                error_detail = str(e)
                if hasattr(e, "response"):
                    try:
                        resp = e.response
                        body_bytes = getattr(resp, "body", None) or b""
                        if body_bytes:
                            from headroom.proxy.helpers import safe_decode_for_logging

                            error_detail += (
                                f" | body: {safe_decode_for_logging(body_bytes, max_bytes=500)}"
                            )
                    except Exception:
                        pass
                logger.error(f"[{request_id}] WS proxy error: {error_detail}")
                if termination_cause == "unknown":
                    termination_cause = "client_error"
            with contextlib.suppress(Exception):
                await websocket.close(code=1011, reason=str(e)[:120])
        finally:
            # Unit 2: emit structured per-session stage timings.
            stage_timer.record(
                "total_session",
                (time.perf_counter() - session_started_at) * 1000.0,
            )
            # Close the upstream WS on early-return paths (e.g. first-frame
            # timeout after we connected). The relay path closes it via
            # `async with upstream`; this idempotent backstop covers the rest.
            with contextlib.suppress(Exception):
                if upstream is not None:
                    await upstream.close()
            # Unit 3: deregister the session before (or independently
            # of) the stage-timings log so a failure there cannot leak
            # the registry entry. ``deregister`` is idempotent, so a
            # session that never registered is a no-op.
            if ws_sessions is not None and session_handle is not None:
                # Use deregister_and_count so the handle pop and the
                # relay-task count are read atomically inside the
                # registry. Capturing ``len(session_handle.relay_tasks)``
                # separately before ``deregister`` would risk drift if
                # the registry's bookkeeping ever changes.
                _deregistered, released_tasks = ws_sessions.deregister_and_count(
                    session_id, cause=termination_cause
                )
                session_duration_ms = (time.perf_counter() - session_started_at) * 1000.0
                metrics_for_close = getattr(self, "metrics", None)
                if metrics_for_close is not None:
                    with contextlib.suppress(Exception):
                        if hasattr(metrics_for_close, "dec_active_ws_sessions"):
                            metrics_for_close.dec_active_ws_sessions()
                        if released_tasks and hasattr(metrics_for_close, "dec_active_relay_tasks"):
                            metrics_for_close.dec_active_relay_tasks(released_tasks)
                        if hasattr(metrics_for_close, "record_ws_session_duration"):
                            metrics_for_close.record_ws_session_duration(
                                session_duration_ms, termination_cause
                            )
            metrics_for_ws_inbound_close = getattr(self, "metrics", None)
            if metrics_for_ws_inbound_close is not None and hasattr(
                metrics_for_ws_inbound_close, "record_inbound_response"
            ):
                with contextlib.suppress(Exception):
                    metrics_for_ws_inbound_close.record_inbound_response(
                        status_code=f"ws:{termination_cause}"
                    )
            logger.info(
                "event=proxy_inbound_websocket_closed request_id=%s session_id=%s "
                "path=%s cause=%s duration_ms=%.2f",
                request_id,
                session_id,
                _ws_path,
                termination_cause,
                (time.perf_counter() - session_started_at) * 1000.0,
            )
            await emit_stage_timings_log(
                path="openai_responses_ws",
                request_id=request_id,
                session_id=session_id,
                stage_timer=stage_timer,
                expected_stages=(
                    "accept",
                    "first_client_frame",
                    "upstream_connect",
                    "upstream_first_event",
                    "memory_context",
                    "compression",
                    "total_session",
                ),
                metrics=getattr(self, "metrics", None),
            )

    async def _ws_http_fallback(
        self,
        websocket: WebSocket,
        body: dict[str, Any],
        first_msg_raw: str,
        upstream_headers: dict[str, str],
        request_id: str,
    ) -> None:
        """Fall back to HTTP POST streaming when upstream WS fails.

        Converts the WS ``response.create`` message to an HTTP POST to
        ``/v1/responses?stream=true``, reads SSE events, and relays each
        ``data:`` line as a WS text message to the client.  This makes
        Codex work immediately instead of exhausting its WS retry budget.
        """
        # Route to correct endpoint based on auth mode
        is_chatgpt_fallback = has_chatgpt_account_header(upstream_headers)
        if is_chatgpt_fallback:
            http_url = codex_responses_http_url()
        else:
            http_url = build_copilot_upstream_url(self.OPENAI_API_URL, "/v1/responses")

        # Build HTTP body from the WS response.create payload.
        # WS messages use {"type": "response.create", "response": {...}} wrapper.
        # The HTTP POST endpoint expects the inner response object directly.
        http_body: dict[str, Any]
        try:
            parsed = json.loads(first_msg_raw) if isinstance(first_msg_raw, str) else body
        except (json.JSONDecodeError, TypeError):
            parsed = body

        # Normalize WebSocket response.create payload into the HTTP request body.
        # Codex may send either:
        # 1. {"type":"response.create","response":{...}}
        # 2. {"type":"response.create", ...response fields...}
        if isinstance(parsed, dict) and isinstance(parsed.get("response"), dict):
            http_body = dict(parsed["response"])
        elif isinstance(parsed, dict):
            http_body = dict(parsed)
            if http_body.get("type") == "response.create":
                http_body.pop("type", None)
        else:
            http_body = body if isinstance(body, dict) else {}

        # Some clients include response-ish metadata that the HTTP endpoint rejects.
        if http_body.get("type") in {"response.create", "response"}:
            http_body.pop("type", None)

        # Ensure streaming is enabled so we get SSE events
        http_body["stream"] = True
        mutation_reasons = ["ws_http_fallback_resynthesized"]
        if _ensure_chatgpt_responses_store_false(
            http_body,
            is_chatgpt_auth=is_chatgpt_fallback,
        ):
            mutation_reasons.append("chatgpt_store_false")

        # Build HTTP headers from the upstream headers (already stripped of WS
        # hop-by-hop headers by the caller).
        http_headers = dict(upstream_headers)
        http_headers["content-type"] = "application/json"

        # Byte-faithful re-serialization (PR-A3, fixes P0-2). The WS payload
        # is always synthesized from the WebSocket frame so the body is
        # treated as mutated; we still go through the canonical path so
        # numeric precision and UTF-8 are preserved.
        from headroom.proxy.body_forwarding import prepare_outbound_body_bytes
        from headroom.proxy.helpers import log_outbound_request

        outbound_bytes, outbound_source = prepare_outbound_body_bytes(
            body=http_body,
            original_body_bytes=None,
            body_mutated=True,
        )
        log_outbound_request(
            forwarder="openai_ws",
            method="POST",
            path=http_url,
            body_bytes_count=len(outbound_bytes),
            body_mutated=True,
            mutation_reasons=mutation_reasons,
            request_id=request_id,
            source=outbound_source,
        )

        logger.debug(f"[{request_id}] WS→HTTP fallback POST to {http_url}")

        try:
            retry_attempts = max(1, getattr(self.config, "retry_max_attempts", 3))
            for http_attempt in range(retry_attempts):
                try:
                    async with self.http_client.stream(
                        "POST",
                        http_url,
                        headers=http_headers,
                        content=outbound_bytes,
                        timeout=120.0,
                    ) as response:
                        if response.status_code != 200:
                            error_body = b""
                            async for chunk in response.aiter_bytes():
                                error_body += chunk
                                if len(error_body) > 2000:
                                    break
                            from headroom.proxy.helpers import safe_decode_for_logging

                            error_text = safe_decode_for_logging(error_body)
                            logger.error(
                                f"[{request_id}] WS→HTTP fallback got {response.status_code}: "
                                f"{error_text[:500]}"
                            )
                            # Send error as WS message so client sees it
                            error_event = {
                                "type": "error",
                                "error": {
                                    "type": "server_error",
                                    "message": f"Upstream returned {response.status_code}",
                                },
                            }
                            await websocket.send_text(json.dumps(error_event))
                            return

                        # Refresh Codex /stats from the fallback response
                        # headers. We can't forward them onto the client 101
                        # (already accepted headerless on this arm), but /stats
                        # parity is still worth keeping on a WS->HTTP fallback.
                        with contextlib.suppress(Exception):
                            from headroom.subscription.codex_rate_limits import (
                                get_codex_rate_limit_state,
                            )

                            get_codex_rate_limit_state().update_from_headers(dict(response.headers))

                        # Relay SSE events as WS text messages
                        buffer = ""
                        async for chunk in response.aiter_text():
                            buffer += chunk
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                line = line.strip()
                                if not line:
                                    continue
                                if line.startswith("data: "):
                                    data = line[6:]
                                    if data == "[DONE]":
                                        continue
                                    try:
                                        await websocket.send_text(data)
                                    except Exception:
                                        return
                                elif line.startswith("event: "):
                                    # SSE event type — skip, the data line contains the type
                                    continue

                        # Flush any remaining data in buffer
                        for line in buffer.strip().splitlines():
                            line = line.strip()
                            if line.startswith("data: ") and line[6:] != "[DONE]":
                                with contextlib.suppress(Exception):
                                    await websocket.send_text(line[6:])
                        return
                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as http_err:
                    if http_attempt >= retry_attempts - 1:
                        raise

                    delay_with_jitter = jitter_delay_ms(
                        self.config.retry_base_delay_ms,
                        self.config.retry_max_delay_ms,
                        http_attempt,
                    )
                    logger.warning(
                        f"[{request_id}] WS→HTTP fallback connect failed "
                        f"(attempt {http_attempt + 1}/{retry_attempts}): {http_err}; "
                        f"retrying in {delay_with_jitter:.0f}ms"
                    )
                    await asyncio.sleep(delay_with_jitter / 1000)

        except Exception as http_err:
            logger.error(f"[{request_id}] WS→HTTP fallback failed: {http_err}")
            error_event = {
                "type": "error",
                "error": {
                    "type": "server_error",
                    "message": f"HTTP fallback failed: {http_err!s}"[:200],
                },
            }
            with contextlib.suppress(Exception):
                await websocket.send_text(json.dumps(error_event))
        finally:
            with contextlib.suppress(Exception):
                await websocket.close()

    def _derived_compress_pipeline(self, key: str, **overrides: Any) -> Any:
        """Cached ``/v1/compress`` pipeline derived from the live OpenAI router.

        ``overrides`` are applied to the live ContentRouter's config, so the
        derived pipeline inherits every operator setting (Kompress on/off,
        exclusions, profile knobs) and differs only in what the caller needs.

        The derived router deliberately does NOT share the base router's
        compression cache. ``ContentRouter._cache`` is per-instance, and its
        keys do not encode the CCR-marker mode, so a shared cache would serve
        marker-laden entries (written by the OpenAI request path) to this
        marker-free path and vice versa. Sharing would be a correctness bug,
        not an optimisation — the duplicate cache is the intended trade.

        Built with ``provider=None`` on purpose: ``TransformPipeline`` then
        resolves the tokenizer from the per-model registry
        (``headroom.tokenizers.get_tokenizer``) on every call instead of pinning
        one provider's counter for the whole route.

        That matters because this route does no format conversion — callers send
        whichever wire shape they already use. The provider counters walk only the
        block types they own (``OpenAITokenCounter`` handles ``text`` and
        ``image_url``; everything else falls through with NO else branch, so an
        Anthropic ``tool_result`` contributed literally zero and ``tokens_saved``
        reported 0 on requests where compression really ran). Every registry
        tokenizer derives from ``BaseTokenizer``, whose ``_count_content_parts``
        ends in a serialize-and-count catch-all, so no block type counts as zero
        and there is no per-provider type list to keep in sync. It also stops
        defaulting Gemini/Mistral/DeepSeek/Kimi traffic to a tiktoken count when
        the registry already has a calibrated counter for them.

        Note the pipeline's own ``_provider_name()`` is therefore ``None`` here.
        That is honest for a provider-agnostic route — it used to report
        ``"openai"`` even for Claude payloads — and the request outcome still
        records ``provider="compress"`` for /stats and /metrics.

        ponytail: a first-request race just builds it twice — both are
        equivalent and Kompress weights are cached at module level, so no lock.
        """
        cache = getattr(self, "_compress_pipeline_cache", None)
        if cache is None:
            cache = self._compress_pipeline_cache = {}
        cached = cache.get(key)
        if cached is not None:
            return cached

        from headroom.transforms.compression_units import find_content_router
        from headroom.transforms.content_router import ContentRouter
        from headroom.transforms.pipeline import TransformPipeline

        base = find_content_router(self.openai_pipeline)
        if base is None:  # ponytail: nothing to derive from — use default pipeline
            return self.openai_pipeline
        pipeline = TransformPipeline(
            transforms=[ContentRouter(replace(base.config, **overrides), observer=self.metrics)],
            provider=None,  # per-model registry tokenizer — see docstring
        )
        cache[key] = pipeline
        return pipeline

    def _no_ccr_pipeline(self) -> Any:
        """Default ``/v1/compress`` pipeline: compress, but emit no CCR markers.

        Every caller of this route is a gateway/sidecar or SDK client that
        forwards the returned messages straight to a provider. A
        ``Retrieve more: hash=`` marker is only useful to a caller that also
        injects the ``headroom_retrieve`` tool AND can reach ``/v1/retrieve``
        (loopback-only). LiteLLM's `headroom` guardrail — the main consumer —
        does neither: it swaps ``messages`` and forwards. So markers here are a
        dangling pointer for the model plus a pointless CCR store write.
        ``config.mode="ccr"`` opts back in for callers that do run the loop
        (e.g. the TypeScript SDK's ``retrieve``/``handleToolCall``).

        Nothing else changes: same compressors, same aggressiveness. Measured
        identical token savings to the marker-on default on OpenAI-shaped
        gateway traffic.
        """
        return self._derived_compress_pipeline(
            "no_ccr",
            ccr_inject_marker=False,  # no markers in returned content
            ccr_enabled=False,  # no CCR store writes
        )

    def _ccr_pipeline(self) -> Any:
        """Pipeline for ``/v1/compress`` ``config.mode="ccr"``.

        No config overrides: markers and CCR store writes are exactly what this
        mode asks for, so it inherits the live router's settings verbatim. It is
        still a DERIVED pipeline rather than ``openai_pipeline`` so the tokenizer
        comes from the per-model registry instead of a pinned provider counter.
        """
        return self._derived_compress_pipeline("ccr")

    def _lossy_inline_pipeline(self) -> Any:
        """Pipeline for ``/v1/compress`` ``config.mode="lossy_inline"``.

        Runs the lossless byte/data fold first, then Kompresses the folded
        remainder (``lossless_then_lossy``), marker-free like
        :meth:`_no_ccr_pipeline`. Kept as a distinct mode because the
        fold-then-Kompress ordering is a different compression posture, not a
        different CCR setting.
        """
        return self._derived_compress_pipeline(
            "lossy_inline",
            lossless=False,  # lossy mode (not lossless-only)
            lossless_then_lossy=True,  # fold first, then Kompress the remainder
            ccr_inject_marker=False,  # inline, marker-free everywhere
            ccr_enabled=False,  # no CCR store writes
            smart_crusher_lossless_only=False,  # keep SmartCrusher lossy
        )

    async def handle_compress(self, request: Request) -> JSONResponse:
        """Compress messages without calling an LLM.

        POST /v1/compress
        Body: {"messages": [...], "model": "...", "config": {}}

        ``config.mode`` selects the pipeline:

        * unset (default) — marker-free; forward the result straight to a
          provider with no CCR retrieval round-trip (see _no_ccr_pipeline).
        * ``"ccr"`` — CCR markers + store writes, for a caller that injects the
          ``headroom_retrieve`` tool and can reach loopback ``/v1/retrieve``.
        * ``"lossy_inline"`` (alias ``"lossless_then_lossy"``) — marker-free,
          lossless fold first then Kompress the remainder.

        Any other ``config.mode`` value is a 400 (see ``COMPRESS_MODES``).

        ``config.frozen_message_count`` pins a prefix: the first N messages are
        returned byte-for-byte unchanged, while still being visible to cross-message
        transforms such as dedup. Callers that resend a growing conversation each turn
        should set it to the number of messages the provider has already cached, so
        compression does not rewrite the prefix and bust that cache. Must be a
        non-negative integer; anything else is a 400.

        Returns compressed messages + metrics.
        """
        from fastapi.responses import JSONResponse

        from headroom.proxy.helpers import _read_request_json

        # Check bypass header
        if request.headers.get("x-headroom-bypass", "").lower() == "true":
            try:
                body = await _read_request_json(request)
            except (json.JSONDecodeError, ValueError) as e:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid request body: {e!s}"},
                )
            messages = body.get("messages", [])
            return JSONResponse(
                {
                    "messages": messages,
                    "tokens_before": 0,
                    "tokens_after": 0,
                    "tokens_saved": 0,
                    "compression_ratio": 1.0,
                    "transforms_applied": [],
                    "ccr_hashes": [],
                }
            )

        try:
            body = await _read_request_json(request)
        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_request",
                        "message": "Invalid JSON in request body.",
                    }
                },
            )

        messages = body.get("messages")
        model = body.get("model")

        if messages is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_request",
                        "message": "Missing required field: messages",
                    }
                },
            )

        if model is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_request",
                        "message": "Missing required field: model",
                    }
                },
            )

        if not messages:
            return JSONResponse(
                {
                    "messages": [],
                    "tokens_before": 0,
                    "tokens_after": 0,
                    "tokens_saved": 0,
                    "compression_ratio": 1.0,
                    "transforms_applied": [],
                    "ccr_hashes": [],
                }
            )

        start_time = time.time()
        headers = dict(request.headers)
        tags = extract_tags(headers)
        client = classify_client(headers)

        try:
            # Use OpenAI pipeline (messages are in OpenAI format from TS SDK)
            # Allow optional token_budget to override model's context limit
            # (used by OpenClaw compact() and other callers that need tighter budgets)
            token_budget = body.get("token_budget")
            # Resolve the CONTEXT LIMIT against the model's own provider. This
            # route accepts either wire shape, and LiteLLM's `headroom` guardrail
            # passes Anthropic model names straight through
            # (claude-sonnet-4-5-..., bedrock/anthropic.claude-3-5-sonnet,
            # anthropic/claude-opus-4), where the OpenAI provider answers with its
            # 128K default instead of 200K+. Substring match is enough — no model
            # registry.
            #
            # Deliberately separate from tokenizer selection, which the derived
            # pipelines resolve per model from the tokenizer registry (see
            # _derived_compress_pipeline). Welding the two together would let a
            # tokenizer decision move `model_limit`, and `model_limit` feeds
            # context_pressure -> min_ratio: e.g. gpt-4-32k answered by the
            # Anthropic table is 8,192 instead of 32,768, a 4x under-estimate that
            # silently changes compression aggressiveness.
            model_name = model if isinstance(model, str) else str(model)
            limit_provider = (
                self.anthropic_provider
                if ("claude" in model_name.lower() or "anthropic" in model_name.lower())
                else self.openai_provider
            )
            context_limit = (
                token_budget
                if token_budget and isinstance(token_budget, int)
                else limit_provider.get_context_limit(model)
            )
            # Extract CompressConfig options from request body
            compress_config = body.get("config", {})
            if not isinstance(compress_config, dict):
                compress_config = {}
            compress_user_messages = compress_config.get("compress_user_messages", False)
            target_ratio = compress_config.get("target_ratio")
            protect_recent = compress_config.get("protect_recent")
            protect_analysis_context = compress_config.get("protect_analysis_context")
            # Leading messages already in the provider's prompt cache. Callers that
            # resend a growing conversation every turn (agent loops) need to pin the
            # prefix they have already paid for: without it the router compresses old
            # messages harder as the conversation grows, so their bytes change and the
            # cache misses from that point on. `protect_recent` guards the other end of
            # the list and cannot express this.
            frozen_message_count = compress_config.get("frozen_message_count")
            if frozen_message_count is not None and (
                isinstance(frozen_message_count, bool)
                or not isinstance(frozen_message_count, int)
                or frozen_message_count < 0
            ):
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "type": "invalid_request",
                            "message": (
                                f"Invalid config.frozen_message_count: {frozen_message_count!r}. "
                                "Expected a non-negative integer."
                            ),
                        }
                    },
                )
            # Mode selection. Default is marker-free (see _no_ccr_pipeline):
            # no caller of this route can resolve a CCR marker unless it opts in
            # with mode="ccr", which restores the full marker + store behaviour.
            mode = compress_config.get("mode")
            if mode is not None and mode not in COMPRESS_MODES:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "type": "invalid_request",
                            "message": (
                                f"Invalid config.mode: {mode!r}. "
                                f"Valid values are: {', '.join(COMPRESS_MODES)} "
                                "(or omit config.mode for the default marker-free mode)."
                            ),
                        }
                    },
                )
            if mode in ("lossy_inline", "lossless_then_lossy"):
                pipeline = self._lossy_inline_pipeline()
            elif mode == "ccr":
                # Markers + store writes wanted, so inherit the live router config
                # unchanged — but as a DERIVED pipeline, not `openai_pipeline`
                # itself, so the per-model registry tokenizer applies here too.
                # Sharing the request path's instance pinned the OpenAI counter,
                # which reports zero for Anthropic content blocks. Costs this mode
                # its own (cold) compression cache; correct metrics win.
                pipeline = self._ccr_pipeline()
            else:
                pipeline = self._no_ccr_pipeline()

            pipeline_kwargs: dict = {
                "model_limit": context_limit,
                **proxy_pipeline_kwargs(self.config),
            }
            if compress_user_messages:
                pipeline_kwargs["compress_user_messages"] = True
            if target_ratio is not None:
                pipeline_kwargs["target_ratio"] = float(target_ratio)
            if protect_recent is not None:
                pipeline_kwargs["protect_recent"] = int(protect_recent)
            if protect_analysis_context is not None:
                pipeline_kwargs["protect_analysis_context"] = bool(protect_analysis_context)
            if frozen_message_count is not None:
                pipeline_kwargs["frozen_message_count"] = frozen_message_count

            # Offload the CPU-bound pipeline to the bounded compression executor
            # (mirrors the request handlers above). Running apply() inline blocked
            # the single event loop on a large payload, so even GET /health stalled
            # until it finished (#718). The executor also enforces a timeout so a
            # too-large body fails fast instead of hanging forever.
            result = await self._run_compression_in_executor(
                lambda: pipeline.apply(
                    messages=messages,
                    model=model,
                    **pipeline_kwargs,
                ),
                timeout=COMPRESSION_TIMEOUT_SECONDS,
            )

            tokens_before = result.tokens_before
            tokens_after = result.tokens_after
            tokens_saved = max(0, tokens_before - tokens_after)
            latency_ms = (time.time() - start_time) * 1000
            await self._record_request_outcome(
                RequestOutcome(
                    request_id=(
                        await self._next_request_id()
                        if hasattr(self, "_next_request_id")
                        else f"compress_{int(time.time())}"
                    ),
                    provider="compress",
                    model=model if isinstance(model, str) else str(model),
                    original_tokens=tokens_before,
                    optimized_tokens=tokens_after,
                    output_tokens=0,
                    tokens_saved=tokens_saved,
                    attempted_input_tokens=tokens_before,
                    total_latency_ms=latency_ms,
                    overhead_ms=latency_ms,
                    num_messages=len(messages) if isinstance(messages, list) else 0,
                    transforms_applied=tuple(result.transforms_applied or ()),
                    waste_signals=(
                        result.waste_signals.to_dict()
                        if getattr(result, "waste_signals", None) is not None
                        else None
                    ),
                    pipeline_timing=getattr(result, "timing", None) or None,
                    tags=tags,
                    client=client,
                )
            )

            return JSONResponse(
                {
                    "messages": result.messages,
                    "tokens_before": result.tokens_before,
                    "tokens_after": result.tokens_after,
                    "tokens_saved": result.tokens_before - result.tokens_after,
                    "compression_ratio": (
                        result.tokens_after / result.tokens_before
                        if result.tokens_before > 0
                        else 1.0
                    ),
                    "transforms_applied": result.transforms_applied,
                    "transforms_summary": result.transforms_summary,
                    "ccr_hashes": result.markers_inserted,
                }
            )
        except TimeoutError:
            logger.warning(
                "Compression timed out after %.0fs; failing open with original messages",
                COMPRESSION_TIMEOUT_SECONDS,
            )
            self.metrics.record_compression_failed("timeout")
            latency_ms = (time.time() - start_time) * 1000
            await self._record_request_outcome(
                RequestOutcome(
                    request_id=(
                        await self._next_request_id()
                        if hasattr(self, "_next_request_id")
                        else f"compress_{int(time.time())}"
                    ),
                    provider="compress",
                    model=model if isinstance(model, str) else str(model),
                    original_tokens=0,
                    optimized_tokens=0,
                    output_tokens=0,
                    tokens_saved=0,
                    attempted_input_tokens=0,
                    total_latency_ms=latency_ms,
                    overhead_ms=latency_ms,
                    num_messages=len(messages) if isinstance(messages, list) else 0,
                    tags=tags,
                    client=client,
                )
            )
            return JSONResponse(
                content={
                    "messages": messages,
                    "tokens_before": 0,
                    "tokens_after": 0,
                    "tokens_saved": 0,
                    "compression_ratio": 1.0,
                    "transforms_applied": [],
                    "transforms_summary": {},
                    "ccr_hashes": [],
                    "compression_skipped": True,
                    "skip_reason": "compression_timeout",
                },
            )
        except Exception as e:
            logger.exception("Compression failed: %s", e)
            await self.metrics.record_failed(provider="compress")
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "type": "compression_error",
                        "message": str(e),
                    }
                },
            )

    async def _maybe_compress_passthrough_responses(
        self, body: bytes, *, client: str | None = None
    ) -> bytes:
        """Compress an OpenAI Responses-shaped passthrough body, fail-open.

        Reuses the native `/v1/responses` compression path so custom
        wrapper-proxy routes get the same ContentRouter/Kompress treatment.
        Any parse/compression failure returns the original body unchanged so a
        catch-all request is never dropped by opting into compression.
        """
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, ValueError, TypeError):
            return body
        if not isinstance(payload, dict) or "input" not in payload:
            # Not a Responses payload (no `input` array) — leave it alone.
            return body

        model = str(payload.get("model") or "passthrough")
        request_id = await self._next_request_id()
        try:
            try:
                result = await self._compress_openai_responses_payload_in_executor(
                    payload,
                    model=model,
                    request_id=request_id,
                    client=client,
                )
            except TypeError as exc:
                if "unexpected keyword argument 'client'" not in str(exc):
                    raise
                result = await self._compress_openai_responses_payload_in_executor(
                    payload,
                    model=model,
                    request_id=request_id,
                )
            compressed_payload, modified, *_rest = result
        except Exception as exc:  # noqa: BLE001 — fail-open on any compressor error
            logger.warning(
                "[%s] passthrough Responses compression failed, forwarding verbatim: %s",
                request_id,
                exc,
            )
            return body
        if not modified:
            return body
        try:
            return json.dumps(
                compressed_payload,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        except (TypeError, ValueError):
            return body

    async def handle_passthrough(
        self,
        request: Request,
        base_url: str,
        endpoint_name: str | None = None,
        provider: str | None = None,
    ) -> Response:
        """Pass through request unchanged.

        Args:
            request: The incoming request
            base_url: The upstream API base URL
            endpoint_name: Optional name for stats tracking (e.g., "models", "embeddings")
            provider: Optional provider name for stats (e.g., "openai", "anthropic", "gemini")
        """
        from fastapi.responses import Response

        if endpoint_name in {"streamGenerateContent", "streamRawPredict"} and provider:
            return await self._handle_streaming_passthrough(
                request=request,
                base_url=base_url,
                endpoint_name=endpoint_name,
                provider=provider,
            )

        start_time = time.time()
        path = request.url.path
        if provider == "anthropic" and endpoint_name == "models" and path.startswith("/v1/models/"):
            from headroom.providers.anthropic import sanitize_anthropic_model_id

            raw_model_id = path[len("/v1/models/") :]
            clean_model_id = sanitize_anthropic_model_id(unquote(raw_model_id))
            if clean_model_id != unquote(raw_model_id):
                path = "/v1/models/" + quote(clean_model_id, safe="")
        url = build_copilot_upstream_url(base_url, path)

        # Preserve query string parameters
        if request.url.query:
            url = f"{url}?{request.url.query}"

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("accept-encoding", None)
        client = classify_client(headers)
        tags = extract_tags(headers)
        # PR-A5 (P5-49): strip internal x-headroom-* before forwarding upstream.
        from headroom.proxy.helpers import (
            _strip_internal_headers,
            log_outbound_headers,
            request_with_transient_retry,
        )

        _pre_strip_count_pt = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        log_outbound_headers(
            forwarder="openai_passthrough",
            stripped_count=_pre_strip_count_pt,
            request_id=None,
        )

        from starlette.requests import ClientDisconnect

        try:
            body = await request.body()
        except ClientDisconnect:
            logger.debug("Client disconnected during body read for passthrough")
            return Response(status_code=204)

        # Hermes owns its scoped coding-agent protocols; keep that adapter out of
        # the generic OpenAI passthrough handler.
        from headroom.providers.hermes import compress_scoped_passthrough_body

        optimize_enabled = bool(getattr(getattr(self, "config", None), "optimize", False))
        original_body = body
        body = compress_scoped_passthrough_body(
            path,
            body,
            optimize=optimize_enabled,
            bypass=_headroom_bypass_enabled(request.headers),
        )
        if body is not original_body:
            headers["content-length"] = str(len(body))

        # Opt-in: compress requests that fall through here because their path
        # doesn't match a built-in API route (custom wrapper-proxy paths like
        # `/api/codex-proxy/<key>/v1/responses`). Off by default; only touches
        # OpenAI Responses-shaped bodies (path ends in `/responses`) so we reuse
        # the exact same ContentRouter/Kompress path the native handler runs.
        _pt_config = getattr(self, "config", None)
        if (
            getattr(_pt_config, "compress_passthrough", False)
            and getattr(_pt_config, "optimize", False)
            and request.method == "POST"
            and path.rstrip("/").endswith("/responses")
            and body
        ):
            compressed = await self._maybe_compress_passthrough_responses(body, client=client)
            if compressed != body:
                body = compressed
                # Body size changed — let httpx recompute Content-Length.
                for _hk in [k for k in headers if k.lower() == "content-length"]:
                    headers.pop(_hk, None)

        headers = await apply_copilot_api_auth(headers, url=url)
        # Cloudflare bot-management challenges our HTTP/2 fingerprint on
        # ChatGPT's sensitive account endpoints (/backend-api/me,
        # /backend-api/accounts/check), returning a 403 challenge page instead
        # of JSON and collapsing the Codex account menu to just "Settings".
        # Those endpoints answer fine over HTTP/1.1, so forward ChatGPT
        # passthrough on the HTTP/1.1 client. Other hosts keep HTTP/2.
        passthrough_client = self.http_client
        if _prefers_http1_passthrough(base_url):
            passthrough_client = self.http_client_h1 or self.http_client
        try:
            # Retry once on a transient keep-alive close (httpx
            # RemoteProtocolError / "incomplete chunked read"): the upstream
            # closed a pooled connection httpx then reused. A fresh connection
            # succeeds, mirroring what a direct curl call does. See GH #1112.
            response = await request_with_transient_retry(
                passthrough_client,  # type: ignore[arg-type]
                method=request.method,
                url=url,
                headers=headers,
                content=body,
            )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            logger.warning(
                "Passthrough request failed before upstream response: %s %s -> %s: %s",
                request.method,
                path,
                url,
                e,
            )
            return Response(
                content=json.dumps(
                    {
                        "error": {
                            "type": "connection_error",
                            "message": f"Failed to connect to upstream API: {e}",
                        }
                    }
                ),
                status_code=502,
                media_type="application/json",
            )
        except httpx.RemoteProtocolError as e:
            # Persisted across the retry: the upstream really is sending an
            # incomplete response. Return a clear 502 instead of letting the
            # raw protocol error surface as an opaque/unhandled 502.
            logger.warning(
                "Passthrough upstream closed connection without a complete "
                "response after retry: %s %s -> %s: %s",
                request.method,
                path,
                url,
                e,
            )
            return Response(
                content=json.dumps(
                    {
                        "error": {
                            "type": "upstream_protocol_error",
                            # Full exception detail is logged server-side above;
                            # keep the client-facing message generic so upstream
                            # exception/stack-trace text is not exposed (CodeQL
                            # py/stack-trace-exposure).
                            "message": (
                                "Upstream closed the connection without sending "
                                "a complete response."
                            ),
                        }
                    }
                ),
                status_code=502,
                media_type="application/json",
            )

        # Remove compression headers since httpx already decompressed the response
        response_headers = _sanitize_forwarded_response_headers(response.headers)
        response_content = response.content

        if provider == "anthropic" and endpoint_name == "models":
            from headroom.providers.anthropic import sanitize_anthropic_model_metadata

            try:
                payload = response.json()
                sanitized_payload = sanitize_anthropic_model_metadata(payload)
            except (TypeError, ValueError):
                sanitized_payload = None
            if sanitized_payload is not None and sanitized_payload != payload:
                response_content = json.dumps(
                    sanitized_payload,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
                response_headers["content-type"] = "application/json"

        # Passthrough request: forwarded upstream with no transforms.
        # Still recorded so dashboards see traffic on the passthrough
        # endpoints. When the upstream exposes provider-native usage
        # fields, normalize them so dashboard totals do not collapse to
        # zero for Vertex/Gemini and other pass-through endpoints.
        if endpoint_name and provider:
            latency_ms = (time.time() - start_time) * 1000
            request_id = await self._next_request_id()
            usage: dict[str, int] = {}
            if response.headers.get("content-type", "").lower().startswith("application/json"):
                try:
                    usage = _passthrough_usage_from_json(response.json())
                except (json.JSONDecodeError, ValueError, TypeError):
                    usage = {}
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cache_read_tokens = usage.get("cache_read_input_tokens", 0)
            cache_write_tokens = usage.get("cache_creation_input_tokens", 0)
            uncached_input_tokens = max(0, input_tokens - cache_read_tokens - cache_write_tokens)
            await self._record_request_outcome(
                RequestOutcome(
                    request_id=request_id,
                    provider=provider,
                    model=_passthrough_model_from_path(path, endpoint_name),
                    status_code=response.status_code,
                    original_tokens=input_tokens,
                    optimized_tokens=input_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=0,
                    attempted_input_tokens=input_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                    uncached_input_tokens=uncached_input_tokens,
                    total_latency_ms=latency_ms,
                    tags=tags,
                    client=client,
                )
            )

        return Response(
            content=response_content,
            status_code=response.status_code,
            headers=response_headers,
        )

    async def _handle_streaming_passthrough(
        self,
        request: Request,
        base_url: str,
        endpoint_name: str,
        provider: str,
    ) -> Response:
        """Stream pass-through responses without buffering the upstream body."""
        from fastapi.responses import Response, StreamingResponse

        from headroom.proxy.helpers import MAX_SSE_BUFFER_SIZE

        start_time = time.time()
        path = request.url.path
        url = build_copilot_upstream_url(base_url, path)
        if request.url.query:
            url = f"{url}?{request.url.query}"

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("accept-encoding", None)
        client = classify_client(headers)
        tags = extract_tags(headers)

        from headroom.proxy.helpers import _strip_internal_headers, log_outbound_headers

        _pre_strip_count_pt = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        log_outbound_headers(
            forwarder="streaming_passthrough",
            stripped_count=_pre_strip_count_pt,
            request_id=None,
        )

        from starlette.requests import ClientDisconnect

        try:
            body = await request.body()
        except ClientDisconnect:
            logger.debug("Client disconnected during body read for streaming passthrough")
            return Response(status_code=204)
        headers = await apply_copilot_api_auth(headers, url=url)
        request_id = await self._next_request_id()
        stream_provider = "gemini" if provider == "vertex:google" else "anthropic"
        stream_state: dict[str, Any] = {
            "input_tokens": None,
            "output_tokens": None,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_creation_ephemeral_5m_input_tokens": 0,
            "cache_creation_ephemeral_1h_input_tokens": 0,
            "total_bytes": 0,
            "sse_buffer": bytearray(),
            "ttfb_ms": None,
        }

        assert self.http_client is not None, "http_client must be initialized before streaming"
        try:
            upstream_request = self.http_client.build_request(
                request.method,
                url,
                headers=headers,
                content=body,
            )
            upstream_response = await self.http_client.send(upstream_request, stream=True)
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            logger.warning(
                "Streaming passthrough failed before upstream response: %s %s -> %s: %s",
                request.method,
                path,
                url,
                e,
            )
            return Response(
                content=json.dumps(
                    {
                        "error": {
                            "type": "connection_error",
                            "message": f"Failed to connect to upstream API: {e}",
                        }
                    }
                ),
                status_code=502,
                media_type="application/json",
            )

        response_headers = _sanitize_forwarded_response_headers(
            upstream_response.headers,
            "transfer-encoding",
            "connection",
        )

        if upstream_response.status_code >= 400:
            try:
                error_content = await upstream_response.aread()
            finally:
                await upstream_response.aclose()
            return Response(
                content=error_content,
                status_code=upstream_response.status_code,
                headers=response_headers,
            )

        def _absorb_usage(usage: dict[str, int] | None) -> None:
            if not usage:
                return
            if "input_tokens" in usage:
                stream_state["input_tokens"] = usage["input_tokens"]
            if "output_tokens" in usage:
                stream_state["output_tokens"] = usage["output_tokens"]
            if "cache_read_input_tokens" in usage:
                stream_state["cache_read_input_tokens"] = usage["cache_read_input_tokens"]
            if "cache_creation_input_tokens" in usage:
                stream_state["cache_creation_input_tokens"] = usage["cache_creation_input_tokens"]
            if "cache_creation_ephemeral_5m_input_tokens" in usage:
                stream_state["cache_creation_ephemeral_5m_input_tokens"] = usage[
                    "cache_creation_ephemeral_5m_input_tokens"
                ]
            if "cache_creation_ephemeral_1h_input_tokens" in usage:
                stream_state["cache_creation_ephemeral_1h_input_tokens"] = usage[
                    "cache_creation_ephemeral_1h_input_tokens"
                ]

        async def generate():
            try:
                async with contextlib.aclosing(upstream_response) as response:
                    async for chunk in response.aiter_bytes():
                        if stream_state["ttfb_ms"] is None:
                            stream_state["ttfb_ms"] = (time.time() - start_time) * 1000
                        stream_state["total_bytes"] += len(chunk)
                        stream_state["sse_buffer"].extend(chunk)
                        if len(stream_state["sse_buffer"]) > MAX_SSE_BUFFER_SIZE:
                            tail = bytes(stream_state["sse_buffer"][-MAX_SSE_BUFFER_SIZE // 2 :])
                            stream_state["sse_buffer"] = bytearray(tail)

                        _absorb_usage(
                            self._parse_sse_usage_from_buffer(stream_state, stream_provider)
                        )
                        yield chunk
            finally:
                buf = stream_state["sse_buffer"]
                if len(buf) > 0:
                    buf.extend(b"\n\n")
                    _absorb_usage(self._parse_sse_usage_from_buffer(stream_state, stream_provider))

                input_tokens = stream_state["input_tokens"] or 0
                output_tokens = stream_state["output_tokens"] or 0
                cache_read_tokens = stream_state["cache_read_input_tokens"] or 0
                cache_write_tokens = stream_state["cache_creation_input_tokens"] or 0
                uncached_input_tokens = max(
                    0,
                    input_tokens - cache_read_tokens - cache_write_tokens,
                )
                await self._record_request_outcome(
                    RequestOutcome(
                        request_id=request_id,
                        provider=provider,
                        model=_passthrough_model_from_path(path, endpoint_name),
                        original_tokens=input_tokens,
                        optimized_tokens=input_tokens,
                        output_tokens=output_tokens,
                        tokens_saved=0,
                        attempted_input_tokens=input_tokens,
                        cache_read_tokens=cache_read_tokens,
                        cache_write_tokens=cache_write_tokens,
                        cache_write_5m_tokens=stream_state[
                            "cache_creation_ephemeral_5m_input_tokens"
                        ],
                        cache_write_1h_tokens=stream_state[
                            "cache_creation_ephemeral_1h_input_tokens"
                        ],
                        uncached_input_tokens=uncached_input_tokens,
                        total_latency_ms=(time.time() - start_time) * 1000,
                        ttfb_ms=stream_state["ttfb_ms"] or 0,
                        tags=tags,
                        client=client,
                    )
                )

        media_type = upstream_response.headers.get("content-type") or "text/event-stream"
        return StreamingResponse(
            generate(),
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=media_type,
        )
