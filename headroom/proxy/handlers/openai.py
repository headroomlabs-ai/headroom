"""OpenAI handler mixin for HeadroomProxy.

Contains all OpenAI Chat Completions, Responses API, and passthrough handlers.
"""

from __future__ import annotations

import asyncio
import base64
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

from headroom.proxy.helpers import (
    COMPRESSION_TIMEOUT_SECONDS,
    _headroom_bypass_enabled,
    extract_tags,
    jitter_delay_ms,
)
from headroom.proxy.stage_timer import StageTimer, emit_stage_timings_log
from headroom.proxy.ws_session_registry import (
    TerminationCause,
    WebSocketSessionRegistry,
    WSSessionHandle,
)

if TYPE_CHECKING:
    from fastapi import Request, WebSocket
    from fastapi.responses import JSONResponse, Response, StreamingResponse

import httpx

from headroom.copilot_auth import apply_copilot_api_auth, build_copilot_upstream_url
from headroom.pipeline import PipelineStage, summarize_routing_markers
from headroom.proxy.auth_mode import classify_auth_mode, classify_client
from headroom.proxy.compression_decision import CompressionDecision
from headroom.proxy.cost import _summarize_transforms
from headroom.proxy.outcome import RequestOutcome

logger = logging.getLogger("headroom.proxy")

_OPENAI_RESPONSES_UNIT_CACHE_MAX_ENTRIES = 10_000
_OPENAI_RESPONSES_UNIT_CACHE_VERSION = "openai_responses_unit_v1"
_OPENAI_RESPONSES_UNIT_PARALLELISM_ENV = "HEADROOM_TOOL_OUTPUT_COMPRESSION_PARALLELISM"
_OPENAI_RESPONSES_UNIT_PARALLELISM_DEFAULT = 4
_OPENAI_RESPONSES_UNIT_PARALLELISM_MAX = 16
_OPENAI_RESPONSES_UNIT_CACHE_INIT_LOCK = threading.RLock()
_OPENAI_RESPONSES_UNIT_EXECUTOR_LOCK = threading.RLock()
_OPENAI_RESPONSES_UNIT_EXECUTOR: ThreadPoolExecutor | None = None


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


def _openai_responses_unit_cache_key(unit: Any, *, model: str) -> str:
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


def _json_byte_len(value: Any) -> int:
    return len(_json_debug_dumps(value).encode("utf-8", errors="replace"))


def _compact_openai_tool_schema_value(
    value: Any,
) -> Any:
    if isinstance(value, list):
        return [_compact_openai_tool_schema_value(item) for item in value]

    if not isinstance(value, dict):
        return value

    compacted: dict[str, Any] = {}
    for key, child in value.items():
        if key in _OPENAI_TOOL_SCHEMA_DROP_KEYS:
            continue

        if key == "description" and isinstance(child, str):
            compacted[key] = " ".join(child.split())
            continue

        compacted[key] = _compact_openai_tool_schema_value(child)

    return compacted


def _compact_openai_responses_tools(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], bool, int, int]:
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return payload, False, 0, 0

    compacted_tools = _compact_openai_tool_schema_value(tools)
    before = _json_byte_len(tools)
    after = _json_byte_len(compacted_tools)
    if after >= before:
        return payload, False, before, after

    updated = copy.deepcopy(payload)
    updated["tools"] = compacted_tools
    return updated, True, before, after


def _responses_input_item_text_bytes(item: Any) -> int:
    if not isinstance(item, dict):
        return _json_byte_len(item)

    output = item.get("output")
    if isinstance(output, str):
        return len(output.encode("utf-8", errors="replace"))

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
        name_str = name.decode("latin-1") if isinstance(name, (bytes, bytearray)) else str(name)
        if name_str.lower().startswith("x-codex-"):
            value_str = (
                value.decode("latin-1") if isinstance(value, (bytes, bytearray)) else str(value)
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


def _decode_openai_bearer_payload(headers: dict[str, str]) -> dict[str, Any] | None:
    """Best-effort decode of an OpenAI OAuth bearer token payload.

    OpenClaw's Codex OAuth flow may forward only the bearer token after the
    provider base URL is overridden. In that case the explicit
    ``ChatGPT-Account-ID`` header can be missing even though the JWT still
    carries the account id we need to route to the ChatGPT Codex backend.
    """
    auth = headers.get("authorization") or headers.get("Authorization")
    if not auth:
        return None

    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or token.count(".") < 2:
        return None

    payload = token.split(".", 2)[1]
    payload += "=" * (-len(payload) % 4)
    # Intentionally no signature verification here: this is only a best-effort
    # routing hint extractor. Upstream still performs the actual auth/authz checks.
    try:
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        data = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None

    return data if isinstance(data, dict) else None


def _resolve_codex_routing_headers(headers: dict[str, str]) -> tuple[dict[str, str], bool]:
    """Resolve ChatGPT Codex routing hints from explicit headers or OAuth JWT."""
    resolved = dict(headers)
    lower_lookup = {k.lower(): k for k in resolved}

    if "chatgpt-account-id" in lower_lookup:
        return resolved, True

    payload = _decode_openai_bearer_payload(resolved)
    auth_claims = payload.get("https://api.openai.com/auth") if isinstance(payload, dict) else None
    account_id = auth_claims.get("chatgpt_account_id") if isinstance(auth_claims, dict) else None
    if isinstance(account_id, str) and account_id.strip():
        resolved["ChatGPT-Account-ID"] = account_id.strip()
        return resolved, True

    return resolved, False


class OpenAIHandlerMixin:
    """Mixin providing OpenAI API handler methods for HeadroomProxy."""

    OPENAI_RESPONSES_ROUTER_MIN_BYTES = 512
    OPENAI_RESPONSES_OUTPUT_TYPES = {
        "custom_tool_call_output",
        "function_call_output",
        "local_shell_call_output",
        "apply_patch_call_output",
    }

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

    @staticmethod
    def _headroom_bypass_enabled(headers: Any) -> bool:
        """Return True when inbound headers request full passthrough."""
        return _headroom_bypass_enabled(headers)

    @staticmethod
    def _strict_previous_turn_frozen_count(
        messages: list[dict[str, Any]],
        base_frozen_count: int,
    ) -> int:
        """Freeze all prior turns in cache mode; only final user turn is mutable."""
        if not messages:
            return base_frozen_count
        final_idx = len(messages) - 1
        if messages[final_idx].get("role") == "user":
            return max(base_frozen_count, final_idx)
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

        try:
            tokenizer = self.openai_provider.get_token_counter(model)
        except Exception as exc:
            logger.debug(
                "[%s] OpenAI Responses ContentRouter tokenizer unavailable: %s",
                request_id,
                exc,
            )
            return payload, False, 0, [], {}, [], 0

        def _slot_text(item: dict[str, Any]) -> tuple[str, tuple[str, int | None]] | None:
            # Only tool-output items are eligible for in-place compression.
            # Message items (user/system/assistant) sit inside the request's
            # cacheable prefix; mutating them busts prefix caching on every
            # subsequent turn. Role-level guards in compression_units.py
            # remain as defense-in-depth.
            type_tag = item.get("type")
            if type_tag in self.OPENAI_RESPONSES_OUTPUT_TYPES:
                output = item.get("output")
                if isinstance(output, str):
                    return output, ("output", None)
            return None

        def _set_slot_text(
            item: dict[str, Any],
            slot: tuple[str, int | None],
            replacement: str,
        ) -> None:
            kind, _ = slot
            if kind == "output":
                item["output"] = replacement

        headroom_retrieve_call_ids: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "function_call":
                continue
            name = item.get("name")
            if isinstance(name, str) and (
                name == "headroom_retrieve" or name.endswith("__headroom_retrieve")
            ):
                call_id = item.get("call_id")
                if isinstance(call_id, str) and call_id:
                    headroom_retrieve_call_ids.add(call_id)

        timing_sink: dict[str, float] = timing if timing is not None else {}

        def _add_timing(name: str, started_at: float) -> None:
            timing_sink[name] = (
                timing_sink.get(name, 0.0) + (time.perf_counter() - started_at) * 1000.0
            )

        extraction_started = time.perf_counter()
        candidates: list[tuple[int, tuple[str, int | None], str]] = []
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
                slot = _slot_text(item)
                if slot is not None:
                    text, slot_ref = slot
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
        if not candidates:
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
            result = compress_unit_with_router(routed.unit, router=router, tokenizer=tokenizer)
            elapsed_ms = (time.perf_counter() - unit_started) * 1000.0
            return routed.slot, result, elapsed_ms

        router_total_started = time.perf_counter()
        routed_results: list[tuple[object, Any, float] | None] = [None] * len(routed_units)
        cache_misses: list[tuple[int, str, RoutedCompressionUnit]] = []
        cache_miss_followers: dict[str, list[int]] = {}
        for unit_idx, routed in enumerate(routed_units):
            cache_key = _openai_responses_unit_cache_key(routed.unit, model=model)
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
            _set_slot_text(target_item, slot_ref, result.compressed)
            modified = True
            tokens_saved_total += result.tokens_saved
            for transform in result.transforms_applied:
                if transform not in transforms:
                    transforms.append(transform)
        _add_timing("compression_unit_apply_results", apply_started)

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
    ) -> tuple[dict[str, Any], bool, int, list[str], str | None, int, int, int, dict[str, float]]:
        timing: dict[str, float] = {}

        def _compress():  # noqa: ANN202
            try:
                return self._compress_openai_responses_payload(
                    payload,
                    model=model,
                    request_id=request_id,
                    timing=timing,
                )
            except TypeError as exc:
                if "unexpected keyword argument 'timing'" not in str(exc):
                    raise
                return self._compress_openai_responses_payload(
                    payload,
                    model=model,
                    request_id=request_id,
                )

        result = await self._run_compression_in_executor(
            _compress,
            timeout=COMPRESSION_TIMEOUT_SECONDS,
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
        from headroom.tokenizers import get_tokenizer
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
        input_event = self.pipeline_extensions.emit(
            PipelineStage.INPUT_RECEIVED,
            operation="proxy.request",
            request_id=request_id,
            provider="openai",
            model=model,
            messages=messages,
            tools=body.get("tools"),
            metadata={"path": "/v1/chat/completions", "stream": body.get("stream", False)},
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
                    messages = compressor.compress(messages, provider="openai")
                    if compressor.last_result:
                        logger.info(
                            f"[{request_id}] Image: {compressor.last_result.technique.value} "
                            f"({compressor.last_result.savings_percent:.0f}% saved, "
                            f"{compressor.last_result.original_tokens} → "
                            f"{compressor.last_result.compressed_tokens} tokens)"
                        )
            finally:
                if compressor and hasattr(compressor, "close"):
                    compressor.close()

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
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
        from headroom.proxy.helpers import _strip_internal_headers, log_outbound_headers

        _pre_strip_count_chat = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        log_outbound_headers(
            forwarder="openai_chat_completions",
            stripped_count=_pre_strip_count_chat,
            request_id=request_id,
        )

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
                await self.metrics.record_rate_limited(provider="openai")
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limited. Retry after {wait_seconds:.1f}s",
                )

        # Check cache
        if self.cache and not stream:
            cached = await self.cache.get(messages, model)
            if cached:
                self.pipeline_extensions.emit(
                    PipelineStage.INPUT_CACHED,
                    operation="proxy.request",
                    request_id=request_id,
                    provider="openai",
                    model=model,
                    messages=messages,
                    metadata={"cache_hit": True, "path": "/v1/chat/completions"},
                )
                # Response-cache hit: same pattern as the anthropic
                # cache-hit site. ``from_response_cache=True`` is the
                # distinct signal that the proxy served from its own
                # semantic cache (not upstream prompt cache).
                _cache_hit_latency = (time.time() - start_time) * 1000
                await self._record_request_outcome(
                    RequestOutcome(
                        request_id=request_id,
                        provider="openai",
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
                response_headers = dict(cached.response_headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)

                return Response(content=cached.response_body, headers=response_headers)

        # Token counting
        tokenizer = get_tokenizer(model)
        original_tokens = tokenizer.count_messages(messages)

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

        # Get prefix cache tracker for this session
        openai_session_id = self.session_tracker_store.compute_session_id(request, model, messages)
        openai_prefix_tracker = self.session_tracker_store.get_or_create(
            openai_session_id, "openai"
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
            openai_frozen_count = self._strict_previous_turn_frozen_count(
                original_client_messages,
                openai_frozen_count,
            )

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

                    # Re-freeze boundary
                    openai_frozen_count = comp_cache.compute_frozen_count(messages)

                    result = await self._run_compression_in_executor(
                        lambda: self.openai_pipeline.apply(
                            messages=working_messages,
                            model=model,
                            model_limit=context_limit,
                            context=extract_user_query(working_messages),
                            frozen_message_count=openai_frozen_count,
                            biases=_hook_biases,
                            compression_policy=compression_policy,
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
                    result = await self._run_compression_in_executor(
                        lambda: self.openai_pipeline.apply(
                            messages=messages,
                            model=model,
                            model_limit=context_limit,
                            context=extract_user_query(messages),
                            frozen_message_count=openai_frozen_count,
                            biases=_hook_biases,
                            compression_policy=compression_policy,
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
                logger.warning(f"Optimization failed: {e}")
                # Flag compression failure for observability
                _compression_failed = True

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
                from headroom.proxy.helpers import apply_session_sticky_ccr_tool

                tools, ccr_tool_injected = apply_session_sticky_ccr_tool(
                    provider="openai",
                    session_id=openai_session_id,
                    request_id=request_id,
                    existing_tools=tools,
                    has_compressed_content_this_turn=injector.has_compressed_content,
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

        body["messages"] = optimized_messages
        if tools or _original_tools is not None:
            body["tools"] = tools

        presend_event = self.pipeline_extensions.emit(
            PipelineStage.PRE_SEND,
            operation="proxy.request",
            request_id=request_id,
            provider="openai",
            model=model,
            messages=optimized_messages,
            tools=tools,
            headers=headers,
            metadata={"path": "/v1/chat/completions", "stream": stream},
        )
        if presend_event.messages is not None:
            optimized_messages = presend_event.messages
            body["messages"] = optimized_messages
        if presend_event.tools is not None:
            tools = presend_event.tools
            body["tools"] = tools
        if presend_event.headers is not None:
            headers = presend_event.headers
        optimized_tokens = tokenizer.count_messages(body["messages"])
        tokens_saved = original_tokens - optimized_tokens

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
                        metadata={"path": "/v1/chat/completions", "stream": True},
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
                            "path": "/v1/chat/completions",
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
                            "path": "/v1/chat/completions",
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
                    output_tokens = usage.get("completion_tokens", 0)
                    total_input_tokens = usage.get("prompt_tokens", optimized_tokens)

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
                    # when no explicit count is available.
                    if cache_creation_input_tokens > 0:
                        cache_write_tokens = cache_creation_input_tokens
                    else:
                        cache_write_tokens = _infer_openai_cache_write_tokens(
                            total_input_tokens,
                            cache_read_tokens,
                        )

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
                            optimized_tokens=total_input_tokens,
                            output_tokens=output_tokens,
                            tokens_saved=tokens_saved,
                            attempted_input_tokens=total_input_tokens + tokens_saved,
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
        url = build_copilot_upstream_url(self.OPENAI_API_URL, "/v1/chat/completions")

        try:
            if stream:
                # Inject stream_options to get usage stats in streaming response
                # This allows accurate token counting instead of byte-based estimation
                if "stream_options" not in body:
                    body["stream_options"] = {"include_usage": True}
                elif isinstance(body.get("stream_options"), dict):
                    body["stream_options"]["include_usage"] = True

                self.pipeline_extensions.emit(
                    PipelineStage.POST_SEND,
                    operation="proxy.request",
                    request_id=request_id,
                    provider="openai",
                    model=model,
                    messages=body["messages"],
                    tools=tools,
                    metadata={"path": "/v1/chat/completions", "stream": True},
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
                )
            else:
                headers = await apply_copilot_api_auth(headers, url=url)
                response = await self._retry_request("POST", url, headers, body)
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
                        "path": "/v1/chat/completions",
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
                        "path": "/v1/chat/completions",
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

                        debug_payload = {
                            "request_id": request_id,
                            "timestamp": datetime.now().isoformat(),
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
                            "tools_sent": body.get("tools"),
                            "tool_count": len(body.get("tools") or []),
                            "original_tool_count": len(_original_tools or []),
                            "messages_sent": body.get("messages"),
                            "message_count": len(body.get("messages", [])),
                            "original_messages": (
                                original_messages
                                if original_messages is not body.get("messages")
                                else "__same_as_sent__"
                            ),
                            "original_message_count": len(original_messages),
                            "system_prompt": body.get("system"),
                        }

                        with open(debug_file, "w") as f:
                            json.dump(debug_payload, f, indent=2, default=str)

                        logger.warning(f"[{request_id}] Full debug dump: {debug_file}")
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
                    total_input_tokens = usage.get("prompt_tokens", optimized_tokens)
                    output_tokens = usage.get("completion_tokens", 0)
                    # OpenAI returns cached_tokens in prompt_tokens_details
                    # These are charged at 50% of the input price
                    prompt_details = usage.get("prompt_tokens_details") or {}
                    cache_read_tokens = prompt_details.get("cached_tokens", 0)
                except (KeyError, TypeError, AttributeError) as e:
                    logger.debug(
                        f"[{request_id}] Failed to extract cached tokens from OpenAI response: {e}"
                    )

                # Update prefix cache tracker for next turn
                cache_write_tokens = _infer_openai_cache_write_tokens(
                    total_input_tokens,
                    cache_read_tokens,
                )
                openai_prefix_tracker.update_from_response(
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                    messages=optimized_messages,
                )

                # OpenAI has no write penalty — uncached = total - cached
                uncached_input_tokens = max(0, total_input_tokens - cache_read_tokens)

                if self.cost_tracker:
                    self.cost_tracker.record_tokens(
                        model,
                        tokens_saved,
                        optimized_tokens,
                        cache_read_tokens=cache_read_tokens,
                        cache_write_tokens=cache_write_tokens,
                        uncached_tokens=uncached_input_tokens,
                    )

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

                # Cache
                if self.cache and response.status_code == 200:
                    await self.cache.set(
                        messages,
                        model,
                        response.content,
                        dict(response.headers),
                        tokens_saved,
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
                        provider="openai",
                        model=model,
                        original_tokens=original_tokens,
                        optimized_tokens=total_input_tokens,
                        output_tokens=output_tokens,
                        tokens_saved=tokens_saved,
                        attempted_input_tokens=total_input_tokens + tokens_saved,
                        cache_read_tokens=cache_read_tokens,
                        cache_write_tokens=cache_write_tokens,
                        uncached_input_tokens=uncached_input_tokens,
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
                response_headers = dict(response.headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)  # Length changed after decompression

                # Inject Headroom compression metrics (for SaaS metering)
                response_headers["x-headroom-tokens-before"] = str(original_tokens)
                response_headers["x-headroom-tokens-after"] = str(optimized_tokens)
                response_headers["x-headroom-tokens-saved"] = str(tokens_saved)
                response_headers["x-headroom-model"] = model
                if transforms_applied:
                    response_headers["x-headroom-transforms"] = ",".join(transforms_applied)
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
            await self.metrics.record_failed(provider="openai")
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
        from fastapi.responses import JSONResponse, Response

        from headroom.proxy.helpers import (
            MAX_REQUEST_BODY_SIZE,
            _read_request_json,
        )
        from headroom.tokenizers import get_tokenizer
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
        stream = body.get("stream", False)
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
        # Strip accept-encoding so httpx negotiates its own encoding.
        # Cloudflare Workers forward "br, zstd" which OpenAI may honor;
        # if httpx lacks brotli support the response body is undecipherable → 502.
        headers.pop("accept-encoding", None)
        tags = extract_tags(headers)
        client = classify_client(headers)
        # PR-A5 (P5-49): strip internal x-headroom-* from upstream-bound
        # headers AFTER `_extract_tags` reads them. Memory user-id reads
        # `request.headers` below.
        from headroom.proxy.helpers import _strip_internal_headers, log_outbound_headers

        _pre_strip_count_resp = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        log_outbound_headers(
            forwarder="openai_responses",
            stripped_count=_pre_strip_count_resp,
            request_id=request_id,
        )

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

        # Token counting on converted messages
        tokenizer = get_tokenizer(model)
        original_tokens = tokenizer.count_messages(messages)

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
                                log_memory_injection(
                                    request_id=request_id,
                                    session_id=None,
                                    decision="injected_live_zone_tail_string",
                                    bytes_injected=len(memory_context),
                                    query=user_query,
                                )
                            elif isinstance(current_input, list):
                                new_input, bytes_appended = append_text_to_latest_user_input_item(
                                    current_input, memory_context
                                )
                                if bytes_appended > 0:
                                    body["input"] = new_input
                                    log_memory_injection(
                                        request_id=request_id,
                                        session_id=None,
                                        decision="injected_live_zone_tail",
                                        bytes_injected=bytes_appended,
                                        query=user_query,
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
                    if self.memory_handler.config.inject_tools
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
                    logger.info(f"[{request_id}] Memory: Injected memory tools (openai/responses)")
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

        headers, is_chatgpt_auth = _resolve_codex_routing_headers(headers)

        # Route to correct endpoint based on auth mode.
        # ChatGPT session auth (codex login) uses chatgpt.com, not api.openai.com.
        if is_chatgpt_auth:
            url = "https://chatgpt.com/backend-api/codex/responses"
        else:
            url = build_copilot_upstream_url(self.OPENAI_API_URL, "/v1/responses")

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
                )
                attempted_input_tokens = int(_attempted_tokens)
                if _modified:
                    tokens_saved = int(_tokens_saved)
                    optimized_tokens = max(0, original_tokens - tokens_saved)
                    transforms_applied = [*_transforms, *list(transforms_applied)]
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

        try:
            if stream:
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
                )
            else:
                headers = await apply_copilot_api_auth(headers, url=url)
                response = await self._retry_request("POST", url, headers, body)
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

                # Memory: handle memory tool calls in Responses API response
                if (
                    self.memory_handler
                    and memory_user_id
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

                if self.cost_tracker:
                    cache_write_tokens = _infer_openai_cache_write_tokens(
                        total_input_tokens,
                        cache_read_tokens,
                    )
                    uncached_input_tokens = max(0, total_input_tokens - cache_read_tokens)
                    self.cost_tracker.record_tokens(
                        model,
                        tokens_saved,
                        total_input_tokens,
                        cache_read_tokens=cache_read_tokens,
                        cache_write_tokens=cache_write_tokens,
                        uncached_tokens=uncached_input_tokens,
                    )
                else:
                    cache_write_tokens = _infer_openai_cache_write_tokens(
                        total_input_tokens,
                        cache_read_tokens,
                    )
                    uncached_input_tokens = max(0, total_input_tokens - cache_read_tokens)

                effective_optimized_tokens = (
                    total_input_tokens if total_input_tokens > 0 else optimized_tokens
                )
                effective_original_tokens = max(
                    original_tokens,
                    effective_optimized_tokens + tokens_saved,
                )

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
                        original_tokens=effective_original_tokens,
                        optimized_tokens=effective_optimized_tokens,
                        output_tokens=output_tokens,
                        tokens_saved=tokens_saved,
                        attempted_input_tokens=attempted_input_tokens,
                        cache_read_tokens=cache_read_tokens,
                        cache_write_tokens=cache_write_tokens,
                        uncached_input_tokens=uncached_input_tokens,
                        total_latency_ms=total_latency,
                        overhead_ms=optimization_latency,
                        transforms_applied=tuple(transforms_applied),
                        num_messages=len(messages) if isinstance(messages, list) else 0,
                        tags=_resp_log_tags,
                        turn_id=compute_turn_id(model, body.get("instructions"), messages),
                        request_messages=messages
                        if getattr(self.config, "log_full_messages", False)
                        else None,
                        client=client,
                    )
                )

                logger.info(f"[{request_id}] /v1/responses {model}: {total_input_tokens:,} tokens")

                # Capture Codex rate-limit window data from response headers
                from headroom.subscription.codex_rate_limits import (
                    get_codex_rate_limit_state,
                )

                get_codex_rate_limit_state().update_from_headers(dict(response.headers))

                # Remove compression headers
                response_headers = dict(response.headers)
                response_headers.pop("content-encoding", None)
                response_headers.pop("content-length", None)

                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=response_headers,
                )
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
        1. Accepts the client WebSocket
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
        # Identify the WS harness before downstream auth/header rewrites.
        # Captured in closure so per-turn RequestOutcome can stamp it.
        client = classify_client(ws_headers)
        _ws_url_obj = getattr(websocket, "url", None)
        _ws_url = str(_ws_url_obj) if _ws_url_obj is not None else ""
        _ws_path = getattr(_ws_url_obj, "path", "") if _ws_url_obj is not None else ""
        if not _ws_path:
            _ws_path = "/v1/responses"
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
        _skip_headers = frozenset(
            {
                "host",  # must match upstream, not local proxy
                "connection",  # hop-by-hop
                "upgrade",  # hop-by-hop
                "sec-websocket-key",  # per-connection cryptographic nonce
                "sec-websocket-version",  # protocol version (websockets lib sets this)
                "sec-websocket-extensions",  # per-connection negotiation
                "sec-websocket-accept",  # server-side only
                "sec-websocket-protocol",  # handled via subprotocols param below
                "content-length",  # hop-by-hop
                "transfer-encoding",  # hop-by-hop
            }
        )
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
        _lower_headers = {k.lower(): v for k, v in upstream_headers.items()}

        # Build upstream WebSocket URL based on auth mode
        if is_chatgpt_auth:
            # ChatGPT session auth → route to chatgpt.com backend
            upstream_url = "wss://chatgpt.com/backend-api/codex/responses"
            logger.debug(
                f"[{request_id}] WS: ChatGPT session auth detected, routing to chatgpt.com"
            )
        else:
            # API key auth → route to configured OpenAI API URL
            base = self.OPENAI_API_URL
            ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
            upstream_url = build_copilot_upstream_url(ws_base, "/v1/responses")

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
        if "authorization" not in _lower_headers:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                upstream_headers["Authorization"] = f"Bearer {api_key}"
                logger.debug(f"[{request_id}] WS: injected Authorization from OPENAI_API_KEY env")
            else:
                logger.warning(
                    f"[{request_id}] WS: no Authorization header from client and "
                    f"OPENAI_API_KEY not set — upstream will likely reject"
                )

        upstream_headers = await apply_copilot_api_auth(upstream_headers, url=upstream_url)

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

        try:
            # --- Connect to upstream OpenAI WebSocket ---
            # NOTE: we connect *before* accepting the client. OpenAI delivers the
            # Codex subscription/rate-limit window only on the upstream WS
            # handshake response headers, so we must read them here and attach
            # the x-codex-* subset to the client-facing 101 (below). Once accept()
            # sends the 101 the headers can no longer be added.
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
                        ping_timeout=20,
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

            # Accept the client WS, forwarding OpenAI's x-codex-* subscription
            # window from the upstream handshake onto the client-facing 101 so
            # Codex, /stats, and the headroom-desktop gauge can read the live
            # window. In API-key mode the handshake carries no x-codex-* headers,
            # so accept_headers stays empty and this behaves exactly as before.
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
            async with stage_timer.measure("accept"):
                await websocket.accept(
                    subprotocol=client_subprotocols[0] if client_subprotocols else None,
                    headers=accept_headers or None,
                )

            # --- Unit 3: register the session as soon as accept succeeds ---
            client_addr: str | None = None
            client_info = getattr(websocket, "client", None)
            if client_info is not None:
                host = getattr(client_info, "host", None)
                port = getattr(client_info, "port", None)
                if host is not None and port is not None:
                    client_addr = f"{host}:{port}"
                elif host is not None:
                    client_addr = str(host)
            if ws_sessions is not None:
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

            # --- Memory: inject context, tools, and instructions ---
            # Gated on MemoryDecision — uniform bypass-respect across
            # all five sites. WS sets memory_user_id only on the inject
            # path (matches pre-PR behaviour); MemoryDecision is the
            # canonical gate.
            memory_user_id: str | None = None
            memory_request_ctx = None
            if self.memory_handler and body:
                _ws_memory_user_id_candidate = ws_headers.get(
                    "x-headroom-user-id",
                    os.environ.get("USER", os.environ.get("USERNAME", "default")),
                )
            else:
                _ws_memory_user_id_candidate = None
            from headroom.proxy.helpers import get_memory_injection_mode
            from headroom.proxy.memory_decision import MemoryDecision
            from headroom.proxy.memory_query import MemoryQuery

            ws_memory_decision = MemoryDecision.decide(
                headers=ws_headers,
                memory_handler=self.memory_handler if body else None,
                memory_user_id=_ws_memory_user_id_candidate,
                mode_name=get_memory_injection_mode(),
            )
            # ws_tags was extracted at handler entry (L3028); applying
            # the memory skip reason here so per-turn RequestOutcomes
            # carry it for dashboard slicing.
            ws_memory_decision.apply_to_tags(ws_tags)
            if ws_memory_decision.inject:
                memory_user_id = _ws_memory_user_id_candidate
                try:
                    # Unwrap response.create envelope to access the response body
                    ws_response_body = body.get("response", body)

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
                        if self.memory_handler.config.inject_tools
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
                        session_id=session_id,
                        request_id=request_id,
                        existing_tools=ws_tools,
                        memory_tools_to_inject=ws_mem_defs_responses,
                        inject_this_turn=bool(self.memory_handler.config.inject_tools),
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
                    if "response" in body and isinstance(body["response"], dict):
                        body["response"] = ws_response_body
                    else:
                        body = ws_response_body

                    first_msg_raw = json.dumps(body)
                except Exception as e:
                    logger.warning(f"[{request_id}] WS Memory injection failed: {e}")
            elif self.memory_handler and body and _ws_bypass:
                logger.info(
                    "[%s] WS memory passthrough reason=bypass_header",
                    request_id,
                )

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
                                for _t in _ws_transforms:
                                    if _t not in transforms_applied:
                                        transforms_applied.append(_t)
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
                    logger.warning(
                        f"[{request_id}] WS /v1/responses compression failed "
                        f"(bytes={_ws_frame_bytes}): {type(_ce).__name__}: {_ce}"
                    )
                    _log_ws_passthrough(
                        "compression_exception",
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
                    await upstream.send(first_msg_raw)

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
                        warned and the original frame is returned —
                        fail loud in logs, fail safe on the wire.
                        Updates outer-scope ``tokens_saved``,
                        ``transforms_applied``, and
                        ``ws_frames_compressed`` so the session-end
                        log reports cumulative savings across all
                        frames in the WS session.
                        """
                        nonlocal tokens_saved, transforms_applied, attempted_input_tokens_total
                        nonlocal ws_frames_compressed
                        if _ws_bypass:
                            _log_ws_passthrough(
                                "bypass_header",
                                frame_index=frame_index,
                                raw_bytes=len(raw_msg.encode("utf-8", errors="replace")),
                            )
                            return raw_msg, False, "bypass_header"
                        if not self.config.optimize:
                            _log_ws_passthrough(
                                "optimize_disabled",
                                frame_index=frame_index,
                                raw_bytes=len(raw_msg.encode("utf-8", errors="replace")),
                            )
                            return raw_msg, False, "optimize_disabled"
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
                                )
                                for (
                                    _timing_name,
                                    _timing_ms,
                                ) in frame_compression_timing.items():
                                    _record_ws_compression_timing(_timing_name, _timing_ms)
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
                                raw_bytes=len(raw_msg.encode("utf-8", errors="replace")),
                                frame_type="response.create",
                                model=str(inner_payload.get("model") or "unknown"),
                            )
                            return raw_msg, False, "compression_exception"
                        if not modified:
                            reason = frame_reason or "no_compression"
                            _log_ws_passthrough(
                                reason,
                                frame_index=frame_index,
                                raw_bytes=bytes_before,
                                frame_type="response.create",
                                model=str(inner_payload.get("model") or "unknown"),
                            )
                            return raw_msg, False, reason
                        if not isinstance(new_inner, dict):
                            _log_ws_passthrough(
                                "compressed_payload_not_dict",
                                frame_index=frame_index,
                                raw_bytes=len(raw_msg.encode("utf-8", errors="replace")),
                                frame_type="response.create",
                                model=str(inner_payload.get("model") or "unknown"),
                            )
                            return raw_msg, False, "compressed_payload_not_dict"
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
                        for t in frame_transforms:
                            if t not in transforms_applied:
                                transforms_applied.append(t)
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
                                (
                                    msg,
                                    _frame_modified,
                                    _frame_reason,
                                ) = await _maybe_compress_response_create_frame(
                                    msg,
                                    frame_index=client_frame_index,
                                )
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
                                await upstream.send(msg)
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

                        memory_enabled = bool(self.memory_handler and memory_user_id)

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
                            saved_delta = tokens_saved - ws_recorded_tokens_saved_total
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

                            # Structured PERF log line so ``headroom perf``
                            # counts this Codex turn. Pre-P2 this emit was
                            # missing, which is why Codex traffic showed up
                            # as ``Requests: 0`` in the perf report even
                            # under heavy load — the same visibility bug
                            # class as #327's "Cache write: 0" report.
                            _perf_input_tokens = max(0, input_delta)
                            _perf_cache_read = max(0, cache_read_delta)
                            _perf_cache_write = max(0, cache_write_delta)
                            _perf_cache_hit_pct = (
                                round(
                                    _perf_cache_read / (_perf_cache_read + _perf_cache_write) * 100
                                )
                                if (_perf_cache_read + _perf_cache_write) > 0
                                else 0
                            )
                            _perf_tok_before = _perf_input_tokens + max(0, saved_delta)
                            _perf_num_msgs = (
                                len(body.get("messages") or body.get("input") or [])
                                if isinstance(body, dict)
                                else 0
                            )
                            logger.info(
                                f"[{request_id}] PERF "
                                f"model={model_for_metrics} msgs={_perf_num_msgs} "
                                f"tok_before={_perf_tok_before} "
                                f"tok_after={_perf_input_tokens} "
                                f"tok_saved={max(0, saved_delta)} "
                                f"cache_read={_perf_cache_read} "
                                f"cache_write={_perf_cache_write} "
                                f"cache_hit_pct={_perf_cache_hit_pct} "
                                f"opt_ms={overhead_delta_ms:.0f} "
                                f"transforms={_summarize_transforms(transforms_applied)}"
                            )

                            ws_recorded_input_tokens_total = ws_input_tokens_total
                            ws_recorded_output_tokens_total = ws_output_tokens_total
                            ws_recorded_cache_read_tokens_total = ws_cache_read_tokens_total
                            ws_recorded_cache_write_tokens_total = ws_cache_write_tokens_total
                            ws_recorded_uncached_input_tokens_total = ws_uncached_input_tokens_total
                            ws_recorded_tokens_saved_total = tokens_saved
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
                                            await upstream.send(json.dumps(cont))
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
                                logger.debug(
                                    f"[{request_id}] WS upstream→client relay ended: {relay_err}"
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
            residual_tokens_saved = max(0, tokens_saved - ws_recorded_tokens_saved_total)
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
                # bookkeeping quartet for the residual; the explicit
                # session-summary RequestLog below remains a separate
                # entry (different semantics — cumulative session
                # totals vs delta residual).
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
                    )
                )
                ws_recorded_overhead_ms_total = _current_ws_overhead_ms()
                if final_ttfb_ms > 0:
                    ws_recorded_ttfb_ms = True
            if getattr(self, "logger", None) is not None:
                from headroom.proxy.helpers import compute_turn_id
                from headroom.proxy.models import RequestLog

                ws_messages_for_log: list[dict[str, Any]] = []
                ws_input_for_log = ws_inner_for_telemetry.get("input")
                ws_instructions_for_log = ws_inner_for_telemetry.get("instructions")
                if isinstance(ws_instructions_for_log, str) and ws_instructions_for_log:
                    ws_messages_for_log.append(
                        {"role": "system", "content": ws_instructions_for_log}
                    )
                if isinstance(ws_input_for_log, str) and ws_input_for_log:
                    ws_messages_for_log.append({"role": "user", "content": ws_input_for_log})
                self.logger.log(
                    RequestLog(
                        request_id=request_id,
                        timestamp=datetime.now().isoformat(),
                        provider="openai",
                        model=model_name,
                        input_tokens_original=ws_input_tokens_total + tokens_saved,
                        input_tokens_optimized=ws_input_tokens_total,
                        output_tokens=ws_output_tokens_total,
                        tokens_saved=tokens_saved,
                        savings_percent=(
                            tokens_saved / (ws_input_tokens_total + tokens_saved) * 100
                        )
                        if ws_input_tokens_total + tokens_saved > 0
                        else 0.0,
                        optimization_latency_ms=_current_ws_overhead_ms(),
                        total_latency_ms=ws_session_duration_ms,
                        tags=ws_session_tags,
                        cache_hit=False,
                        transforms_applied=transforms_applied,
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
        _lower = {k.lower() for k in upstream_headers}
        if "chatgpt-account-id" in _lower:
            http_url = "https://chatgpt.com/backend-api/codex/responses"
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

        # Build HTTP headers from the upstream headers (already stripped of WS
        # hop-by-hop headers by the caller).
        http_headers = dict(upstream_headers)
        http_headers["content-type"] = "application/json"

        # Byte-faithful re-serialization (PR-A3, fixes P0-2). The WS payload
        # is always synthesized from the WebSocket frame so the body is
        # treated as mutated; we still go through the canonical path so
        # numeric precision and UTF-8 are preserved.
        from headroom.proxy.helpers import (
            log_outbound_request,
            prepare_outbound_body_bytes,
        )

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
            mutation_reasons=["ws_http_fallback_resynthesized"],
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

    async def handle_compress(self, request: Request) -> JSONResponse:
        """Compress messages without calling an LLM.

        POST /v1/compress
        Body: {"messages": [...], "model": "...", "config": {}}
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

        try:
            # Use OpenAI pipeline (messages are in OpenAI format from TS SDK)
            # Allow optional token_budget to override model's context limit
            # (used by OpenClaw compact() and other callers that need tighter budgets)
            token_budget = body.get("token_budget")
            context_limit = (
                token_budget
                if token_budget and isinstance(token_budget, int)
                else self.openai_provider.get_context_limit(model)
            )
            # Extract CompressConfig options from request body
            compress_config = body.get("config", {})
            compress_user_messages = compress_config.get("compress_user_messages", False)
            target_ratio = compress_config.get("target_ratio")
            protect_recent = compress_config.get("protect_recent")
            protect_analysis_context = compress_config.get("protect_analysis_context")

            pipeline_kwargs: dict = {"model_limit": context_limit}
            if compress_user_messages:
                pipeline_kwargs["compress_user_messages"] = True
            if target_ratio is not None:
                pipeline_kwargs["target_ratio"] = float(target_ratio)
            if protect_recent is not None:
                pipeline_kwargs["protect_recent"] = int(protect_recent)
            if protect_analysis_context is not None:
                pipeline_kwargs["protect_analysis_context"] = bool(protect_analysis_context)

            result = self.openai_pipeline.apply(
                messages=messages,
                model=model,
                **pipeline_kwargs,
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
        except Exception as e:
            logger.exception("Compression failed: %s", e)
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "type": "compression_error",
                        "message": str(e),
                    }
                },
            )

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

        start_time = time.time()
        path = request.url.path
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
        from headroom.proxy.helpers import _strip_internal_headers, log_outbound_headers

        _pre_strip_count_pt = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        log_outbound_headers(
            forwarder="openai_passthrough",
            stripped_count=_pre_strip_count_pt,
            request_id=None,
        )

        body = await request.body()

        headers = await apply_copilot_api_auth(headers, url=url)
        try:
            response = await self.http_client.request(  # type: ignore[union-attr]
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

        # Remove compression headers since httpx already decompressed the response
        response_headers = dict(response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("content-length", None)  # Length changed after decompression

        # Passthrough request: forwarded upstream with no transforms.
        # Still recorded so dashboards see traffic on the passthrough
        # endpoints. Funnel handles the "no tokens, no cache" shape
        # via zero defaults.
        if endpoint_name and provider:
            latency_ms = (time.time() - start_time) * 1000
            request_id = await self._next_request_id()
            await self._record_request_outcome(
                RequestOutcome(
                    request_id=request_id,
                    provider=provider,
                    model=f"passthrough:{endpoint_name}",
                    original_tokens=0,
                    optimized_tokens=0,
                    output_tokens=0,
                    tokens_saved=0,
                    attempted_input_tokens=0,
                    total_latency_ms=latency_ms,
                    tags=tags,
                    client=client,
                )
            )

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
        )
