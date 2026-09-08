"""Anthropic handler mixin for HeadroomProxy.

Contains all Anthropic Messages API handlers including batch operations.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from headroom.proxy.stage_timer import StageTimer, emit_stage_timings_log

if TYPE_CHECKING:
    from fastapi import Request
    from fastapi.responses import Response, StreamingResponse

import httpx

from headroom.agent_savings import proxy_pipeline_kwargs
from headroom.ccr.context_tracker import looks_like_claude_code_compact_summary
from headroom.ccr.marker_resolution import resolve_markers_in_response
from headroom.copilot_auth import apply_copilot_api_auth, build_copilot_upstream_url
from headroom.pipeline import PipelineStage, summarize_routing_markers
from headroom.providers.vertex import annotate_backend_error_body, backend_error_hint
from headroom.proxy.auth_mode import (
    classify_auth_mode,
    classify_client,
    supports_mid_turn_coalescing,
)
from headroom.proxy.buffered_ccr_response import (
    ANTHROPIC_ERROR_FORMAT,
    DEFAULT_BUFFERED_CCR_GRACE_SECONDS,
    buffered_ccr_asgi_call,
)
from headroom.proxy.compression_decision import CompressionDecision
from headroom.proxy.forwarded_headers import resolve_client_ip
from headroom.proxy.handlers._debug_dump import _debug_dump_mode, _redact_debug_value
from headroom.proxy.helpers import (
    extract_tags,
    relocate_system_messages_to_top_level,
    sanitize_forwarded_response_headers,
)
from headroom.proxy.identity import resolve_memory_identity
from headroom.proxy.image_isolation import run_image_compression_isolated
from headroom.proxy.memory_decision import MemoryDecision
from headroom.proxy.memory_query import MemoryQuery
from headroom.proxy.model_router import estimate_input_tokens
from headroom.proxy.nonstream_sse_policy import should_recover_sse_reply
from headroom.proxy.outcome import RequestOutcome
from headroom.proxy.output_shaper import shaper_enabled_for, steering_allowed_for
from headroom.proxy.thinking_tokens import ThinkingTokens, extract_thinking_tokens

logger = logging.getLogger("headroom.proxy")


@lru_cache(maxsize=1)
def _thinking_estimator() -> Callable[[str], int] | None:
    """A process-wide local token counter for thinking-block estimation.

    Cached because ``AnthropicTokenCounter.__init__`` loads a tiktoken encoding;
    building one per request would put a vocab load in the response path.

    ``client=None`` is deliberate and load-bearing: with a client the counter
    calls Anthropic's /v1/messages/count_tokens endpoint, which would turn a
    bookkeeping field into a network round trip on every response. The local
    tiktoken approximation is the right trade here — the result is reported as
    ``inferred`` precisely because it is an approximation. ``warn=False``
    suppresses the "no client" UserWarning, which is advice for SDK callers and
    noise for an intentional internal estimate.

    Returns ``None`` when no counter can be built, so the caller reports
    "unknown" rather than a fabricated zero.
    """
    try:
        from headroom.providers.anthropic import AnthropicTokenCounter

        return AnthropicTokenCounter(
            "claude-3-5-sonnet-20241022", client=None, warn=False
        ).count_text
    except Exception:  # noqa: BLE001 - estimation is optional, never fatal
        logger.debug("thinking-token estimator unavailable", exc_info=True)
        return None


def _thinking_tokens_for(payload: object) -> ThinkingTokens:
    """Split thinking from visible output for one Anthropic response.

    Anthropic reports no thinking count in ``usage``, so the count is derived
    from the response's own thinking blocks and marked inferred. Without the
    split, reasoning-effort routing and verbosity steering move the same
    counter and neither can be attributed.

    Never raises: token accounting is bookkeeping, and a failure here must
    degrade to "unknown" rather than cost the caller their response.
    """
    try:
        return extract_thinking_tokens(payload, estimator=_thinking_estimator())
    except Exception:  # noqa: BLE001 - accounting must never break a response
        return ThinkingTokens()


def _is_googleapis_endpoint(value: object) -> bool:
    """Return whether *value* targets Google APIs by parsed hostname.

    A substring check would also trust attacker-controlled hosts such as
    ``googleapis.com.example.test``. URL parsing plus a label-boundary suffix
    check accepts Google API subdomains without widening the route gate.
    """
    raw = str(value).strip()
    if not raw:
        return False
    try:
        hostname = (urlsplit(raw).hostname or "").rstrip(".").lower()
    except ValueError:
        return False
    return hostname == "googleapis.com" or hostname.endswith(".googleapis.com")


class _AnthropicTurnHookUsage:
    """Usage from hook-triggered Anthropic calls the main response omits.

    The handler accounts for the response ultimately returned to the client.
    A turn hook may make additional, billed calls before selecting that final
    response, so retain every response and settle by object identity. This is
    the Anthropic counterpart of OpenAI's ``TurnHookUsage`` and includes the
    provider's disjoint uncached/read/write input buckets.
    """

    __slots__ = (
        "_seen",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "cache_write_5m_tokens",
        "cache_write_1h_tokens",
        "extra_calls",
    )

    def __init__(self) -> None:
        self._seen: list[tuple[Any, int, int, int, int, int, int]] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.cache_write_5m_tokens = 0
        self.cache_write_1h_tokens = 0
        self.extra_calls = 0

    @staticmethod
    def _int(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    def record(self, payload: Any) -> None:
        usage = payload.get("usage") if isinstance(payload, dict) else None
        if not isinstance(usage, dict):
            self._seen.append((payload, 0, 0, 0, 0, 0, 0))
            return
        cache_read = self._int(usage.get("cache_read_input_tokens"))
        cache_write = self._int(usage.get("cache_creation_input_tokens"))
        creation = usage.get("cache_creation")
        cache_write_5m = (
            self._int(creation.get("ephemeral_5m_input_tokens"))
            if isinstance(creation, dict)
            else 0
        )
        cache_write_1h = (
            self._int(creation.get("ephemeral_1h_input_tokens"))
            if isinstance(creation, dict)
            else 0
        )
        self._seen.append(
            (
                payload,
                _anthropic_provider_input_tokens(usage),
                self._int(usage.get("output_tokens")),
                cache_read,
                cache_write,
                cache_write_5m,
                cache_write_1h,
            )
        )

    def settle(self, final: Any) -> None:
        self.input_tokens = self.output_tokens = 0
        self.cache_read_tokens = self.cache_write_tokens = 0
        self.cache_write_5m_tokens = self.cache_write_1h_tokens = 0
        self.extra_calls = 0
        dropped = False
        for (
            payload,
            input_tokens,
            output_tokens,
            cache_read,
            cache_write,
            cache_write_5m,
            cache_write_1h,
        ) in self._seen:
            if not dropped and payload is final:
                dropped = True
                continue
            self.extra_calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.cache_read_tokens += cache_read
            self.cache_write_tokens += cache_write
            self.cache_write_5m_tokens += cache_write_5m
            self.cache_write_1h_tokens += cache_write_1h


def _anthropic_provider_input_tokens(usage: Any) -> int:
    """Provider-scaled prompt total across Anthropic's disjoint buckets."""
    if not isinstance(usage, dict):
        return 0
    return sum(
        _AnthropicTurnHookUsage._int(usage.get(key))
        for key in (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
    )


def _strip_streaming_only_content_fields(messages: Any) -> None:
    """Remove streaming-only ``index`` keys from request content blocks, in place.

    ``index`` is a field Anthropic emits on streaming RESPONSE content-block deltas
    (see proxy/handlers/streaming.py). It is not part of the request-message schema, so
    forwarding it upstream triggers a 400 ("...content.N.text.index: Extra inputs are
    not permitted") that aborts multi-turn sessions once a client echoes a reconstructed
    assistant turn back. Strip it (including nested tool_result content) so requests are
    always schema-valid.
    """
    if not isinstance(messages, list):
        return
    for message in messages:
        if isinstance(message, dict):
            _strip_index_from_content_blocks(message.get("content"))


def _strip_index_from_content_blocks(content: Any) -> None:
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, dict):
            block.pop("index", None)
            # tool_result blocks nest their own content list of blocks.
            _strip_index_from_content_blocks(block.get("content"))


def _looks_like_sse_response(response: httpx.Response) -> bool:
    """Return whether an upstream reply is a Server-Sent Events stream.

    Trusts the declared content-type first and falls back to sniffing the
    leading bytes for an SSE field, because a gateway in front of Anthropic may
    relay the stream under a vaguer type.
    """
    content_type = (response.headers.get("content-type") or "").lower()
    if "text/event-stream" in content_type:
        return True
    head = response.content[:64].lstrip()
    return head.startswith(b"event:") or head.startswith(b"data:")


class AnthropicHandlerMixin:
    """Mixin providing Anthropic API handler methods for HeadroomProxy."""

    def _adapt_event_stream_to_json(
        self,
        response: httpx.Response,
        request_id: str,
    ) -> httpx.Response:
        """Rebuild an SSE reply as the JSON a non-streaming caller asked for.

        A caller that sent ``stream: false`` cannot parse ``text/event-stream``,
        so relaying it verbatim loses a turn the upstream already charged for
        (#3130). Reconstruction is strict: a truncated stream, or one carrying
        an ``error`` event, becomes an explicit 502 rather than a successful
        HTTP 200 whose message is silently short.
        """
        headers = {
            k: v
            for k, v in sanitize_forwarded_response_headers(
                response.headers,
                "content-type",
            ).items()
            if not k.lower().startswith("cf-")
        }

        parsed = None
        try:
            parsed = self._parse_sse_to_response(
                response.content.decode("utf-8", "replace"),
                "anthropic",
                require_complete=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"[{request_id}] SSE->JSON reconstruction raised: {exc}")

        if parsed is None:
            logger.error(
                f"[{request_id}] Upstream answered a non-streaming request with an "
                f"event stream that could not be faithfully reconstructed "
                f"(body_bytes={len(response.content)}); returning 502 rather than "
                f"a wire format the client cannot parse"
            )
            return httpx.Response(
                502,
                json={
                    "type": "error",
                    "error": {
                        "type": "upstream_protocol_error",
                        "message": (
                            "Upstream answered a non-streaming request with an "
                            "incomplete event stream."
                        ),
                    },
                },
                headers=headers,
            )

        logger.info(
            f"[{request_id}] Upstream answered a non-streaming request with an "
            f"event stream; adapted {len(response.content)} bytes of SSE to JSON"
        )
        return httpx.Response(200, json=parsed, headers=headers)

    async def _count_tokens_offloaded(self, model, messages):  # noqa: ANN001, ANN201
        from headroom.proxy.token_counting import count_tokens_offloaded

        return await count_tokens_offloaded(self, model, messages)

    @staticmethod
    def _resolve_ccr_workspace(
        request: Any,
        body: Any,
    ) -> tuple[str, str | None]:
        """Resolve (workspace_key, workspace_label) for CCR scoping.

        Uses the same ``ProjectResolver`` the memory subsystem uses
        (``headroom/memory/storage_router.py``) so CCR and memory always
        agree on which project a request belongs to. Tier order matches:
        ``x-headroom-project-id`` → ``x-headroom-cwd`` → CLI override →
        ``cwd:`` line in the system prompt.

        Returns:
            ``(workspace_key, workspace_label)``. If no signal yields a
            project, returns ``("", None)`` — the empty key is the
            fail-closed signal that callers gate on (skipping
            ``track_compression`` and ``analyze_query`` entirely
            rather than tracking under an empty workspace which would
            create un-matchable entries).

        See also: the 2026-05-26 cross-project leak report which
        motivated this scoping (Python content from project ``tamag0``
        surfaced inside a Ruby ``daphni-rails`` session).
        """
        from headroom.memory.storage_router import (
            ProjectResolver,
        )
        from headroom.memory.storage_router import (
            RequestContext as _CtxFor,
        )
        from headroom.memory.storage_router import (
            extract_system_prompt as _extract_sys_prompt,
        )

        try:
            ctx = _CtxFor(
                headers=dict(request.headers),
                system_prompt=_extract_sys_prompt(body),
                base_user_id=resolve_memory_identity(request, default=""),
                project_root_override=None,
            )
            ident = ProjectResolver().resolve(ctx)
        except Exception as exc:  # noqa: BLE001
            # ProjectResolver is best-effort — log loudly and fail
            # closed so a malformed request doesn't crash the proxy
            # AND doesn't accidentally bypass the workspace filter.
            logger.warning(
                "event=ccr_workspace_resolve_failed error=%s; "
                "CCR proactive expansion disabled for this request",
                exc,
            )
            return "", None

        if ident is None:
            return "", None
        return ident[0], ident[1]

    @staticmethod
    def _tool_sort_key(tool: dict[str, Any]) -> tuple[str, str]:
        """Deterministic sort key for Anthropic/OpenAI-style tool definitions."""
        name = (
            str(tool.get("name", ""))
            or str(tool.get("function", {}).get("name", ""))
            or str(tool.get("type", ""))
        )
        try:
            canonical = json.dumps(tool, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except Exception:
            canonical = str(tool)
        return (name, canonical)

    @staticmethod
    def _has_headroom_retrieve_tool(tools: Any) -> bool:
        """Return True when the final Anthropic tool list includes CCR retrieve."""
        if not isinstance(tools, list):
            return False
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("name") == "headroom_retrieve":
                return True
            function = tool.get("function")
            if isinstance(function, dict) and function.get("name") == "headroom_retrieve":
                return True
        return False

    def _can_salvage_buffered_upstream(self, resp_json: Any) -> bool:
        """May this upstream response be relayed when post-processing failed?

        Only when the client can actually consume it. The buffered path exists
        because a ``headroom_retrieve`` call has to be resolved server-side, so
        a response still carrying one is exactly the case the handler already
        fails closed on: relaying it would hand the client a tool call naming an
        endpoint it is not expected to reach, and a marker nobody expanded.

        Anything else — an ordinary answer, a client tool call, a turn whose
        retrieval already resolved — is a complete provider turn and is safer in
        the client's hands than a synthesized error (#3088).
        """
        if not isinstance(resp_json, dict):
            return False
        handler = getattr(self, "ccr_response_handler", None)
        if handler is None:
            return True
        try:
            from headroom.ccr.response_handler import RESIDUAL_CCR_ERROR

            if handler.residual_ccr_status(resp_json, "anthropic") == RESIDUAL_CCR_ERROR:
                return False
            return not handler.has_ccr_tool_calls(resp_json, "anthropic")
        except Exception:  # pragma: no cover - defensive
            logger.debug("CCR: salvage check failed; not salvaging", exc_info=True)
            return False

    @staticmethod
    def _outgoing_body_has_redeemable_marker(body: Any) -> bool:
        """Does the body about to be sent carry a marker retrieval could expand?

        ``headroom_retrieve`` exists only to expand a ``<<ccr:...>>`` marker, so
        a request carrying none cannot benefit from the buffered path (#3071).

        Ownership is verified rather than shape-matched: the marker shape is not
        unique to Headroom, and adopting another context tool's hash would send
        the model to an endpoint that is guaranteed to miss (#2836). A hash that
        survives ``verify_ownership`` is redeemable right now.

        Errors are swallowed deliberately and answered ``True``. This gates a
        wire-format decision, and the safe direction on an unexpected message
        shape is the long-standing buffered behavior, not a silent change.
        """
        if not isinstance(body, dict):
            return True
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return False
        try:
            from headroom.ccr.tool_injection import CCRToolInjector

            probe = CCRToolInjector(
                provider="anthropic",
                inject_tool=False,
                inject_system_instructions=False,
            )
            probe.scan_for_markers(messages)
            if not probe.detected_hashes:
                return False
            probe.verify_ownership()
            return bool(probe.detected_hashes)
        except Exception:  # pragma: no cover - defensive
            logger.debug("CCR: marker probe failed; keeping the buffered path", exc_info=True)
            return True

    @staticmethod
    def _extract_anthropic_cache_ttl_metrics(usage: dict[str, Any] | None) -> tuple[int, int]:
        """Extract observed Anthropic cache-write TTL bucket usage.

        HeadroomProxy also inherits StreamingMixin, which exposes the same
        helper for SSE usage parsing. Keep this local copy so the Anthropic
        handler remains safe when tested or embedded without StreamingMixin.
        """
        if not isinstance(usage, dict):
            return (0, 0)
        cache_creation = usage.get("cache_creation")
        if not isinstance(cache_creation, dict):
            return (0, 0)
        return (
            int(cache_creation.get("ephemeral_5m_input_tokens", 0) or 0),
            int(cache_creation.get("ephemeral_1h_input_tokens", 0) or 0),
        )

    def _anthropic_buffered_request_timeout(self) -> httpx.Timeout:
        """Timeout for buffered Anthropic reads."""
        return httpx.Timeout(
            connect=self.config.connect_timeout_seconds,
            read=self.config.anthropic_buffered_request_timeout_seconds,
            # A buffered turn waits longer for the *answer*, but sending the
            # request is the same operation it always was — it gets the same
            # write bound as every other path (#3259).
            write=self.config.write_timeout_seconds,
            pool=self.config.connect_timeout_seconds,
        )

    @classmethod
    def _sort_tools_deterministically(
        cls, tools: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        """Return tools in deterministic order to preserve prompt-cache stability.

        Skipped entirely when any tool carries ``cache_control``. A breakpoint on
        a tool means "cache everything up to and including this one", so
        reordering the array changes which tools are inside that prefix -- and
        with two markers of different TTLs it can put the 1h one behind the 5m
        one, which Anthropic rejects outright (#2939). The Rust proxy already
        refuses for the same reason (``any_tool_has_cache_control`` in
        ``crates/headroom-proxy/src/compression/live_zone_anthropic.rs``); this
        is the Python side of that guard.

        Clients that mark no tools -- the common case, and the one the sort was
        written for -- are unaffected, so this costs nobody a cache bust.
        """
        if not tools:
            return tools
        marked = sum(1 for t in tools if isinstance(t, dict) and t.get("cache_control"))
        if marked:
            logger.info(
                "event=tool_sort_skipped reason=marker_present tool_count=%d marked=%d",
                len(tools),
                marked,
            )
            return tools
        return sorted(tools, key=cls._tool_sort_key)

    @classmethod
    def _tools_for_forwarding(
        cls,
        tools: list[dict[str, Any]] | None,
        *,
        preserve_order: bool,
    ) -> list[dict[str, Any]] | None:
        """Return upstream tools, preserving client order for passthrough requests."""
        if preserve_order:
            return tools
        return cls._sort_tools_deterministically(tools)

    @staticmethod
    def _compress_latest_user_turn_images_cache_safe(
        messages: list[dict[str, Any]],
        *,
        frozen_message_count: int,
        compressor: Any,
    ) -> list[dict[str, Any]]:
        """Compress images only in the latest non-frozen user turn.

        This avoids rewriting historical image bytes that may already be in the
        provider prefix cache.
        """
        if not messages:
            return messages

        target_idx = len(messages) - 1
        if target_idx < frozen_message_count:
            return messages
        target_msg = messages[target_idx]
        if target_msg.get("role") != "user":
            return messages
        content = target_msg.get("content")
        if not isinstance(content, list):
            return messages
        if not any(isinstance(block, dict) and block.get("type") == "image" for block in content):
            return messages

        compressed_one = compressor.compress([target_msg], provider="anthropic")
        if not compressed_one:
            return messages

        if compressed_one[0] == target_msg:
            return messages

        updated = list(messages)
        updated[target_idx] = compressed_one[0]
        return updated

    @staticmethod
    def _append_context_to_latest_non_frozen_user_turn(
        messages: list[dict[str, Any]],
        context_text: str,
        *,
        frozen_message_count: int,
    ) -> list[dict[str, Any]]:
        """Append context to the first text block of the latest non-frozen user turn.

        This is the canonical memory-injection path (P0-1 fix in PR-A2). The
        cache hot zone (system + frozen prefix) is never touched. Only the
        first text block of the latest user message is mutated, which is by
        definition the live zone.

        Returns the input list unchanged if no eligible user text block
        exists (e.g., the last message is an assistant turn or a tool
        result, or the user message has no text block).
        """
        if not messages or not context_text:
            return messages

        i = len(messages) - 1
        if i < frozen_message_count:
            return messages
        msg = messages[i]
        if msg.get("role") != "user":
            return messages

        content = msg.get("content", "")
        if isinstance(content, str):
            updated = list(messages)
            updated[i] = {**msg, "content": content + "\n\n" + context_text}
            return updated

        if isinstance(content, list) and content:
            # Append to the first text block of the latest user message.
            # Anthropic content blocks are dicts with a "type" field; the
            # text block has type "text" and a "text" field.
            new_content: list[dict[str, Any]] = []
            appended = False
            for block in content:
                if (
                    not appended
                    and isinstance(block, dict)
                    and block.get("type") == "text"
                    # Never mutate a block carrying the client's cache
                    # breakpoint: the client re-sends the original bytes
                    # next turn, so any injection here busts the prefix
                    # cache at this message from then on.
                    and "cache_control" not in block
                ):
                    existing = block.get("text", "")
                    new_content.append({**block, "text": existing + "\n\n" + context_text})
                    appended = True
                else:
                    new_content.append(block)
            if appended:
                updated = list(messages)
                updated[i] = {**msg, "content": new_content}
                return updated

        return messages

    @staticmethod
    def _strict_previous_turn_frozen_count(
        messages: list[dict[str, Any]],
        base_frozen_count: int,
    ) -> int:
        """Freeze all prior turns; only the final turn is mutable.

        If the final message is not a user turn, freeze everything.
        """
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
        """Force frozen prefix bytes to match the original request exactly."""
        if frozen_message_count <= 0 or not original_messages:
            return candidate_messages, 0

        frozen = min(frozen_message_count, len(original_messages))
        restored = list(candidate_messages)

        # Defensive: if a transform dropped prefix messages, restore them.
        if len(restored) < frozen:
            return list(original_messages[:frozen]) + restored, frozen

        changed = 0
        for idx in range(frozen):
            if restored[idx] != original_messages[idx]:
                restored[idx] = original_messages[idx]
                changed += 1
        return restored, changed

    @staticmethod
    def _extract_cache_stable_delta(
        current_messages: list[dict[str, Any]],
        previous_original_messages: list[dict[str, Any]] | None,
        previous_forwarded_messages: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
        """Return (stable_forwarded_prefix, appended_delta_messages) when safe.

        Safe means the prior original request is an exact message-prefix of the
        current original request. This lets us replay the exact forwarded bytes
        for historical context and only transform newly appended message suffixes.

        The append-only check ignores per-turn transport / cache-directive / client
        annotation noise (cache_control moved to the newest block, litellm caller,
        provider_specific_fields, streaming index, string<->block content shape, …) via
        the shared canonicalizer, so that churn doesn't spuriously drop cache mode to raw
        forwarding. Delegates to the provider-agnostic engine in prefix_tracker so
        OpenAI / Bedrock share one implementation.
        """
        from headroom.cache.prefix_tracker import extract_cache_stable_delta

        return extract_cache_stable_delta(
            current_messages,
            previous_original_messages,
            previous_forwarded_messages,
        )

    @staticmethod
    def _extract_cache_stable_last_message_suffix(
        current_messages: list[dict[str, Any]],
        previous_original_messages: list[dict[str, Any]] | None,
        previous_forwarded_messages: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]] | None:
        """Return append-only delta when only the latest message grew in place."""
        if not previous_original_messages or previous_forwarded_messages is None:
            return None
        if (
            len(current_messages) != len(previous_original_messages)
            or len(previous_forwarded_messages) != len(previous_original_messages)
            or not current_messages
        ):
            return None

        prefix_len = len(current_messages) - 1
        if (
            prefix_len > 0
            and current_messages[:prefix_len] != previous_original_messages[:prefix_len]
        ):
            return None

        current_last = current_messages[-1]
        previous_original_last = previous_original_messages[-1]
        previous_forwarded_last = previous_forwarded_messages[-1]
        if current_last.get("role") != previous_original_last.get("role") or current_last.get(
            "role"
        ) != previous_forwarded_last.get("role"):
            return None

        current_content = current_last.get("content")
        previous_original_content = previous_original_last.get("content")
        previous_forwarded_content = previous_forwarded_last.get("content")

        if (
            isinstance(current_content, str)
            and isinstance(previous_original_content, str)
            and isinstance(previous_forwarded_content, str)
            and current_content.startswith(previous_original_content)
        ):
            suffix = current_content[len(previous_original_content) :]
            delta_messages = []
            if suffix:
                delta_messages = [{**copy.deepcopy(current_last), "content": suffix}]
            return (
                copy.deepcopy(previous_forwarded_messages[:-1]),
                copy.deepcopy(previous_forwarded_last),
                delta_messages,
            )

        if (
            isinstance(current_content, list)
            and isinstance(previous_original_content, list)
            and isinstance(previous_forwarded_content, list)
            and len(current_content) >= len(previous_original_content)
            and current_content[: len(previous_original_content)] == previous_original_content
        ):
            delta_blocks = copy.deepcopy(current_content[len(previous_original_content) :])
            delta_messages = []
            if delta_blocks:
                delta_messages = [{**copy.deepcopy(current_last), "content": delta_blocks}]
            return (
                copy.deepcopy(previous_forwarded_messages[:-1]),
                copy.deepcopy(previous_forwarded_last),
                delta_messages,
            )
        return None

    @staticmethod
    def _merge_appended_message_delta(
        previous_forwarded_message: dict[str, Any],
        delta_forwarded_message: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Merge a compressed suffix back into the prior forwarded message."""
        if delta_forwarded_message is None:
            return copy.deepcopy(previous_forwarded_message)
        if previous_forwarded_message.get("role") != delta_forwarded_message.get("role"):
            return None

        previous_content = previous_forwarded_message.get("content")
        delta_content = delta_forwarded_message.get("content")
        if isinstance(previous_content, str) and isinstance(delta_content, str):
            return {
                **copy.deepcopy(previous_forwarded_message),
                "content": previous_content + delta_content,
            }
        if isinstance(previous_content, list) and isinstance(delta_content, list):
            return {
                **copy.deepcopy(previous_forwarded_message),
                "content": copy.deepcopy(previous_content) + copy.deepcopy(delta_content),
            }
        return None

    @staticmethod
    def _assistant_message_from_response_json(
        resp_json: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not resp_json:
            return None
        if resp_json.get("role") != "assistant":
            return None
        return {
            "role": "assistant",
            "content": copy.deepcopy(resp_json.get("content", "")),
        }

    def _maybe_route_model(
        self,
        model: str,
        messages: object,
        body: dict[str, Any],
        body_mutation_tracker: Any,
        bypass: bool,
    ) -> str:
        """Apply cost-aware model routing (#1706), returning the model to forward.

        Fails closed to disabled when no ``model_router`` is present: alternate
        mixin hosts and test doubles that do not run ``HeadroomProxy.__init__``
        never set the attribute, and reading it unconditionally would crash them
        even when routing is off. Also skipped under bypass/passthrough so a
        byte-faithful request is never model-rewritten.
        """
        router = getattr(self, "model_router", None)
        if router is None or not router.enabled or bypass:
            return model
        decision = router.select(
            model=model,
            input_tokens=estimate_input_tokens(messages, body.get("tools"), body.get("system")),
            has_tools=bool(body.get("tools")),
        )
        logger.info("model routing decision: %s", decision.reason)
        if not decision.changed:
            return model
        body["model"] = decision.routed_model
        body_mutation_tracker.mark_mutated("model_router")
        return decision.routed_model

    async def handle_anthropic_messages(
        self,
        request: Request,
        upstream_base_url: str | None = None,
        provider_name: str = "anthropic",
        model_override: str | None = None,
        force_stream: bool = False,
    ) -> Response | StreamingResponse:
        """Handle Anthropic /v1/messages endpoint."""
        if not hasattr(self, "pipeline_extensions"):
            from headroom.pipeline import PipelineExtensionManager

            self.pipeline_extensions = PipelineExtensionManager(discover=False)

        from fastapi import HTTPException
        from fastapi.responses import JSONResponse, Response, StreamingResponse

        from headroom.cache.compression_store import get_compression_store
        from headroom.ccr import CCRToolInjector
        from headroom.providers.anthropic import sanitize_anthropic_model_id
        from headroom.proxy.body_forwarding import BodyMutationTracker
        from headroom.proxy.helpers import (
            MAX_MESSAGE_ARRAY_LENGTH,
            MAX_REQUEST_BODY_SIZE,
            _get_image_compressor,
            compute_turn_id,
            read_request_json_with_bytes,
        )
        from headroom.proxy.modes import is_cache_mode, is_token_mode
        from headroom.utils import extract_user_query

        start_time = time.time()
        request_id = await self._next_request_id()
        trace_session_id = uuid.uuid4().hex

        # Phase F PR-F1: classify auth mode at request entry. The result
        # is stored on `request.state` so downstream handlers (cache
        # gates, header injection, lossy-compressor gates) read it
        # without re-classifying. Pure function, well under 10us.
        auth_mode = classify_auth_mode(request.headers)
        request.state.auth_mode = auth_mode
        logger.debug(f"[{request_id}] auth_mode_classified mode={auth_mode.value}")

        # Unit 2: per-stage timings for the pre-upstream phase. The
        # finalizer emits one structured log line + Prometheus
        # observations even if the handler raises.
        stage_timer = StageTimer()
        pre_upstream_started_at = time.perf_counter()
        _stage_timings_emitted = False

        # Unit 4: bounded pre-upstream concurrency. When the proxy is
        # configured with a semaphore, acquire it before reading the
        # request body so a replay storm cannot starve ``/livez``, new
        # Codex WS opens, or the HTTP thread pool. The release happens
        # BEFORE we start streaming response bytes back to the client —
        # the semaphore must not be held for the whole response
        # lifetime. ``_release_pre_upstream_sem`` is idempotent and
        # called on every exit path (early 4xx return, upstream error,
        # exception, and just before each streaming handoff).
        pre_upstream_sem = getattr(self, "anthropic_pre_upstream_sem", None)
        _pre_upstream_sem_acquired = False

        def _release_pre_upstream_sem() -> None:
            nonlocal _pre_upstream_sem_acquired
            if _pre_upstream_sem_acquired and pre_upstream_sem is not None:
                _pre_upstream_sem_acquired = False
                pre_upstream_sem.release()

        async def _finalize_pre_upstream() -> None:
            """Release the pre-upstream semaphore and emit stage-timing metrics.

            Idempotent: safe to call multiple times. The PRIMARY action is
            releasing the Unit 4 pre-upstream semaphore via
            ``_release_pre_upstream_sem()`` (itself idempotent); emitting
            stage timings is secondary bookkeeping guarded by
            ``_stage_timings_emitted``. Doing both here (rather than only
            at explicit handoff sites) guarantees semaphore release on
            every exit path — early 4xx returns, security blocks, cache
            hits, upstream errors, streaming handoff.
            """
            nonlocal _stage_timings_emitted
            _release_pre_upstream_sem()
            if _stage_timings_emitted:
                return
            _stage_timings_emitted = True
            if "total_pre_upstream" not in stage_timer:
                stage_timer.record(
                    "total_pre_upstream",
                    (time.perf_counter() - pre_upstream_started_at) * 1000.0,
                )
            await emit_stage_timings_log(
                path="anthropic_messages",
                request_id=request_id,
                session_id=trace_session_id,
                stage_timer=stage_timer,
                expected_stages=(
                    "pre_upstream_wait",
                    "read_request_json",
                    "deep_copy",
                    "compression_first_stage",
                    "memory_context",
                    "upstream_connect",
                    "upstream_first_byte",
                    "total_pre_upstream",
                ),
                metrics=getattr(self, "metrics", None),
            )

        if pre_upstream_sem is not None:
            _pre_upstream_saturated = False
            _wait_started_at = time.perf_counter()
            _acquire_timeout_seconds = self.config.anthropic_pre_upstream_acquire_timeout_seconds
            try:
                await asyncio.wait_for(
                    pre_upstream_sem.acquire(),
                    timeout=_acquire_timeout_seconds,
                )
            except asyncio.TimeoutError:
                _wait_ms = (time.perf_counter() - _wait_started_at) * 1000.0
                _pre_upstream_saturated = True
                logger.warning(
                    "[%s] Anthropic pre-upstream queue saturated after %.2f ms "
                    "(timeout=%.1fs, session_id=%s)",
                    request_id,
                    _wait_ms,
                    _acquire_timeout_seconds,
                    trace_session_id,
                )
                logger.info(
                    "[%s] pre-upstream saturation fail-open; continuing without compression path",
                    request_id,
                )
            else:
                _pre_upstream_sem_acquired = True
                _wait_ms = (time.perf_counter() - _wait_started_at) * 1000.0
            stage_timer.record("pre_upstream_wait", _wait_ms)
            if _wait_ms > 100.0:
                logger.info(
                    "[%s] pre_upstream_wait_ms=%.2f session_id=%s "
                    "(anthropic pre-upstream semaphore contention)",
                    request_id,
                    _wait_ms,
                    trace_session_id,
                )
        else:
            stage_timer.record("pre_upstream_wait", 0.0)
            _pre_upstream_saturated = False

        try:
            # Check request body size
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > MAX_REQUEST_BODY_SIZE:
                await _finalize_pre_upstream()
                return JSONResponse(
                    status_code=413,
                    content={
                        "type": "error",
                        "error": {
                            "type": "request_too_large",
                            "message": f"Request body too large. Maximum size is {MAX_REQUEST_BODY_SIZE // (1024 * 1024)}MB",
                        },
                    },
                )

            # Parse request — capture both the parsed dict AND the original
            # bytes so the forwarder can pick byte-faithful passthrough when
            # nothing mutated the body (PR-A3, fixes P0-2). The mutation
            # tracker is updated by every transform site that touches the
            # body (image compression, memory injection, message rewriting,
            # tool sorting, etc.).
            body_mutation_tracker = BodyMutationTracker()
            try:
                async with stage_timer.measure("read_request_json"):
                    body, original_body_bytes = await read_request_json_with_bytes(request)
            except (json.JSONDecodeError, ValueError) as e:
                await _finalize_pre_upstream()
                return JSONResponse(
                    status_code=400,
                    content={
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": f"Invalid request body: {e!s}",
                        },
                    },
                )
            raw_model = body.get("model") or model_override or "unknown"
            model = (
                sanitize_anthropic_model_id(raw_model) if isinstance(raw_model, str) else raw_model
            )
            body_model = body.get("model")
            if isinstance(body_model, str) and model != body_model:
                body["model"] = model
                body_mutation_tracker.mark_mutated("sanitize_model_id")
            messages = body.get("messages", [])
            # Strip streaming-only "index" keys from request content blocks BEFORE any
            # prefix-cache tracking or compression. The proxy's streaming reconstruction
            # tags assistant blocks with an "index" for SSE re-emission; clients (e.g.
            # opencode) persist that assistant message and echo it back next turn, but
            # "index" is a response-delta field that Anthropic REJECTS in a request
            # ("messages.N.content.0.text.index: Extra inputs are not permitted", 400),
            # aborting multi-turn sessions. Canonicalizing here (in place, so body,
            # original, forwarded, and the recorded/replayed prefix are all identical)
            # keeps it cache-safe: overlay_cached_prefix replays the same stripped bytes.
            _strip_streaming_only_content_fields(messages)
            pipeline_provider = provider_name
            pipeline_path = request.url.path if upstream_base_url else "/v1/messages"
            pipeline_stream = bool(body.get("stream", False) or force_stream)
            with stage_timer.measure("deep_copy"):
                original_client_messages = copy.deepcopy(messages)
            input_event = self.pipeline_extensions.emit(
                PipelineStage.INPUT_RECEIVED,
                operation="proxy.request",
                request_id=request_id,
                provider=pipeline_provider,
                model=model,
                messages=messages,
                tools=body.get("tools"),
                metadata={"path": pipeline_path, "stream": pipeline_stream},
            )
            if input_event.messages is not None:
                messages = input_event.messages
                with stage_timer.measure("deep_copy"):
                    original_client_messages = copy.deepcopy(messages)
            if input_event.tools is not None:
                body["tools"] = input_event.tools

            # Snapshot the client's cache_control breakpoints before any
            # transform runs; paired with the outbound count right before
            # forwarding (event=cache_breakpoints) so a dropped or moved
            # final breakpoint is self-diagnosing from proxy.log alone.
            from headroom.proxy.helpers import (
                CACHE_TTL_1H,
                cache_control_ttl_lanes,
                count_cache_breakpoints,
            )

            inbound_breakpoints = count_cache_breakpoints(
                body.get("system"), messages, body.get("tools")
            )
            # Which TTL lane did the CLIENT ask for on THIS turn? Claude Code
            # picks 5m or 1h per request (a `/btw` side question drops to 5m and
            # omits the extended-cache-ttl beta header even mid-1h-session), so
            # this cannot be inferred from the session or from config. The
            # pre-forward guard needs it to tell a breakpoint the client asked
            # for from one an earlier turn's replayed bytes dragged in (#2939).
            client_uses_1h = CACHE_TTL_1H in cache_control_ttl_lanes(
                body.get("system"), messages, body.get("tools")
            )

            # Validate message array size
            if len(messages) > MAX_MESSAGE_ARRAY_LENGTH:
                await _finalize_pre_upstream()
                return JSONResponse(
                    status_code=400,
                    content={
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": f"Message array too large ({len(messages)} messages). "
                            f"Maximum is {MAX_MESSAGE_ARRAY_LENGTH}.",
                        },
                    },
                )

            stream = pipeline_stream

            # Bypass: skip ALL compression, TOIN learning, and CCR injection
            # when the caller explicitly opts out via header.
            # Prevents Headroom from corrupting sub-agent API calls
            # (e.g., Claude Code sub-agents that inherit ANTHROPIC_BASE_URL).
            _bypass = (
                request.headers.get("x-headroom-bypass", "").lower() == "true"
                or request.headers.get("x-headroom-mode", "").lower() == "passthrough"
            )
            preserve_tool_order = _bypass or not self.config.optimize
            if _bypass:
                logger.info(f"[{request_id}] Bypass: skipping compression (header)")

            # Cost-aware model routing (#1706). Opt-in and disabled by default;
            # fail-closed and bypass handling live in the helper. A model override
            # comes from a provider URL (for example Vertex rawPredict), where
            # rewriting body["model"] would not change the upstream model.
            if model_override is None:
                model = self._maybe_route_model(
                    model, messages, body, body_mutation_tracker, _bypass
                )

            # NOTE: Upstream temporarily disabled broad image compression due to
            # token-counting inaccuracies. We only compress the latest non-frozen
            # user turn later in this handler to preserve Anthropic prefix caching.
            # Extract headers and tags
            headers = dict(request.headers.items())
            headers.pop("host", None)
            headers.pop("content-length", None)
            # read_request_json_with_bytes already content-decoded the inbound
            # body (zstd/gzip/deflate/br), so the bytes we forward are plain.
            # A stale content-encoding makes the upstream try to decompress
            # already-decoded JSON and reject it with HTTP 400 (#1542 — same
            # fix the /v1/responses path already carries).
            headers.pop("content-encoding", None)
            headers.pop("transfer-encoding", None)
            # Strip accept-encoding so httpx negotiates its own encoding.
            # Edge proxies (Cloudflare Workers, etc.) may forward "br, zstd" which
            # the upstream can honor; if httpx lacks brotli support the response
            # body is undecipherable → 502.
            headers.pop("accept-encoding", None)
            tags = extract_tags(headers)
            from headroom.proxy.savings_attribution import bind_scope

            bind_scope(tags, request.scope)
            # Identify the harness (codex / claude-code / aider / etc.)
            # from User-Agent or X-Client. Surfaced via the funnel into
            # PERF logs and RequestLog.tags — see RequestOutcome.client.
            client = classify_client(headers, default="claude")
            # PR-A5 (P5-49): strip internal x-headroom-* from upstream-bound
            # headers AFTER `_extract_tags` reads them. Inbound bypass gating
            # uses `request.headers.get(...)` directly above; memory user-id
            # is read from `request.headers` below if needed. From this
            # point on, `headers` is the upstream-bound copy.
            from headroom.proxy.helpers import (
                _strip_internal_headers,
                log_outbound_headers,
                merge_extra_headers,
            )

            _pre_strip_count = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
            headers = _strip_internal_headers(headers)
            # `upstream_base_url` is the per-request `x-headroom-base-url`
            # override when the client sent one. These headers are secrets, so
            # they only travel to a host the operator designated.
            headers = merge_extra_headers(
                headers,
                self.config.anthropic_extra_headers,
                upstream_url=upstream_base_url,
                config=self.config,
            )
            log_outbound_headers(
                forwarder="anthropic_messages",
                stripped_count=_pre_strip_count
                - sum(1 for k in headers if k.lower().startswith("x-headroom-")),
                request_id=request_id,
            )

            # Subscription tracker: notify on OAuth requests (not API-key requests)
            _auth_header = headers.get("authorization", "")
            if _auth_header.startswith("Bearer ") and not _auth_header.startswith(
                "Bearer sk-ant-api"
            ):
                from headroom.subscription.tracker import (
                    get_subscription_tracker as _get_sub_tracker,
                )

                _sub_tracker = _get_sub_tracker()
                if _sub_tracker is not None:
                    _sub_tracker.notify_active(_auth_header)

            # Rate limiting
            if self.rate_limiter:
                api_key = headers.get("x-api-key", "")
                if not api_key:
                    auth = headers.get("authorization", "")
                    if auth.startswith("Bearer "):
                        api_key = auth[7:]
                # Phase F PR-F4: trust ``X-Forwarded-For`` for the rate-limit
                # key only when the connecting peer is in
                # ``HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS``; otherwise we use
                # the direct peer IP and a malicious client cannot rotate
                # rate-limit buckets by forging headers.
                client_ip = resolve_client_ip(request) or "unknown"
                rate_key = f"{api_key[:16]}:{client_ip}" if api_key else client_ip
                allowed, wait_seconds = await self.rate_limiter.check_request(rate_key)
                if not allowed:
                    await self.metrics.record_rate_limited(provider=provider_name)
                    # Unit 4: release the pre-upstream semaphore before we
                    # bail out of the handler via HTTPException — FastAPI's
                    # exception handler will NOT run our ``finally``.
                    await _finalize_pre_upstream()
                    raise HTTPException(
                        status_code=429,
                        detail=f"Rate limited. Retry after {wait_seconds:.1f}s",
                        headers={"Retry-After": str(int(wait_seconds) + 1)},
                    )

            # Budget check
            if self.cost_tracker:
                allowed, remaining = self.cost_tracker.check_budget()
                if not allowed:
                    # Unit 4: release the pre-upstream semaphore before we
                    # bail out of the handler via HTTPException.
                    await _finalize_pre_upstream()
                    raise HTTPException(
                        status_code=429,
                        detail=self.cost_tracker.budget_denial_detail(),
                    )

            # Memory: Get user ID when memory is enabled (fallback to "default" for simple DevEx).
            # Reads `request.headers` directly because the local `headers` dict was
            # stripped of `x-headroom-*` above for the upstream-bound copy (PR-A5).
            memory_user_id: str | None = None
            memory_request_ctx = None
            if self.memory_handler:
                memory_user_id = resolve_memory_identity(request)
                # Per-project memory routing (GH #462). Build the context
                # once here so save / search / inject all resolve against
                # the same workspace. Tier order: explicit project-id /
                # cwd headers → CLI override → system prompt env block.
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

            # Canonical memory-injection gate. Reads `request.headers`
            # so bypass detection sees the original inbound (the local
            # `headers` dict was stripped of x-headroom-* above).
            # Replaces the pre-PR-this raw `if self.memory_handler and
            # memory_user_id:` conjunction that silently ignored
            # `x-headroom-bypass: true` and mutated request bytes
            # under the user's "don't touch my bytes" signal.
            from headroom.proxy.helpers import get_memory_injection_mode

            memory_decision = MemoryDecision.decide(
                headers=request.headers,
                memory_handler=self.memory_handler,
                memory_user_id=memory_user_id,
                mode_name=get_memory_injection_mode(),
            )
            memory_decision.apply_to_tags(tags)

            # Snapshot cache-key fields from the request body ONCE here
            # (pre-upstream) and reuse them verbatim at the cache.set site
            # below. The pipeline may mutate body before the response is
            # cached, so re-reading there would compute a different key and the
            # cache would never hit (#327). Anthropic system/stop_sequences are
            # top-level fields, never inside messages. Fold in the response-shaping
            # fields the request forwards — else two requests with identical
            # messages but a different tool_choice / thinking / output shape
            # collide and the second caller is served a response made under other
            # semantics (#1473 review). Non-generation metadata (metadata,
            # service_tier) is intentionally excluded.
            cache_key_fields = {
                "system": body.get("system"),
                "tools": body.get("tools"),
                "tool_choice": body.get("tool_choice"),
                "temperature": body.get("temperature"),
                "top_p": body.get("top_p"),
                "top_k": body.get("top_k"),
                "max_tokens": body.get("max_tokens"),
                "stop": body.get("stop_sequences"),
                "thinking": body.get("thinking"),
                "output_config": body.get("output_config"),
            }
            # Snapshot the lookup messages too. `messages` is the primary cache
            # key component, but it is reassigned below by the security scan, the
            # pre_compress hook, and image compression, so caching the response
            # under the live `messages` would store it under a different key than
            # it was looked up by — the cache would never hit and would fill with
            # unreachable entries. Reuse this raw snapshot verbatim at cache.set
            # (the same reason cache_key_fields is snapshotted here, #327).
            cache_lookup_messages = messages
            # Check cache (non-streaming only)
            cache_hit = False
            if self.cache and not stream:
                cached = await self.cache.get(messages, model, **cache_key_fields)
                if cached:
                    cache_hit = True
                    self.pipeline_extensions.emit(
                        PipelineStage.INPUT_CACHED,
                        operation="proxy.request",
                        request_id=request_id,
                        provider=pipeline_provider,
                        model=model,
                        messages=messages,
                        metadata={"cache_hit": True, "path": pipeline_path},
                    )
                    optimization_latency = (time.time() - start_time) * 1000

                    # Response-cache hit: response body came from
                    # Headroom's semantic cache, not the upstream
                    # provider. ``from_response_cache=True`` is a
                    # distinct signal from `cache_read_tokens > 0`
                    # (which means upstream-prompt-cache hit). Dashboards
                    # can split the two; the funnel collapses them into
                    # the single `cached` boolean for Prometheus.
                    await self._record_request_outcome(
                        RequestOutcome(
                            request_id=request_id,
                            provider=provider_name,
                            model=model,
                            original_tokens=0,
                            optimized_tokens=0,
                            output_tokens=0,
                            tokens_saved=0,
                            attempted_input_tokens=0,
                            from_response_cache=True,
                            total_latency_ms=optimization_latency,
                            overhead_ms=optimization_latency,
                            num_messages=len(messages),
                            tags=tags,
                            client=client,
                        )
                    )

                    # Strip the stored response's wire-framing headers. The
                    # entry carries whatever the *producing* upstream sent,
                    # and replaying that framing over a different connection
                    # breaks the body: a stale ``transfer-encoding: chunked``
                    # makes the client parse plain JSON as chunked frames and
                    # read nothing out of an HTTP 200 (#3019). ``content-type``
                    # goes too, because Starlette lets an explicit header win
                    # over ``media_type`` and the producing request's type
                    # would hand this caller a wire format it never asked
                    # for (#2952).
                    response_headers = sanitize_forwarded_response_headers(
                        cached.response_headers,
                        "content-type",
                    )

                    # A cache hit answers the client without touching the
                    # upstream, so it emits no outbound_request line and no
                    # upstream stage timings. Without this log a served-from-
                    # cache turn is indistinguishable from a turn that died
                    # silently, which is exactly how #3019 stayed invisible.
                    logger.info(
                        f"[{request_id}] RESPONSE-CACHE-HIT: model={model} "
                        f"bytes={len(cached.response_body)} "
                        f"age_s={(datetime.now() - cached.created_at).total_seconds():.0f} "
                        f"hits={cached.hit_count}"
                    )

                    # Unit 4: release the pre-upstream semaphore on cache
                    # hit — no upstream call will happen.
                    await _finalize_pre_upstream()
                    return Response(
                        content=cached.response_body,
                        headers=response_headers,
                        media_type="application/json",
                    )

            # Count original tokens off the event loop: first-use tokenizer
            # resolution may hit the network (HF download) and counting a full
            # conversation is CPU-bound — on-loop it froze the server (#1701).
            tokenizer, original_tokens = await self._count_tokens_offloaded(model, messages)

            # Enterprise Security: scan request before compression
            _security_ctx = None
            if self.security:
                try:
                    messages, _security_ctx = self.security.scan_request(
                        messages,
                        {
                            "provider": provider_name,
                            "model": model,
                            "request_id": str(request_id),
                            "user_id": headers.get("x-api-key", "")[:16],
                        },
                    )
                except Exception as e:
                    if hasattr(e, "reason"):
                        from fastapi.responses import JSONResponse as _JSONResp

                        # Unit 4: release the pre-upstream semaphore on
                        # security block — no upstream call will happen.
                        await _finalize_pre_upstream()
                        return _JSONResp(
                            status_code=403,
                            content={
                                "type": "error",
                                "error": {
                                    "type": "security_block",
                                    "message": str(e),
                                },
                            },
                        )
                    logger.warning(f"[{request_id}] Security scan error: {e}")

            # Hook: pre_compress — let hooks modify messages before compression

            if self.config.hooks and not is_cache_mode(self.config.mode):
                from headroom.hooks import CompressContext

                _hook_ctx = CompressContext(
                    model=model,
                    user_query=extract_user_query(messages),
                    provider=provider_name,
                )
                try:
                    messages = self.config.hooks.pre_compress(messages, _hook_ctx)
                except Exception as e:
                    logger.debug(f"[{request_id}] pre_compress hook error: {e}")
            else:
                _hook_ctx = None

            # Apply optimization
            transforms_applied = []
            pipeline_timing: dict[str, float] = {}
            waste_signals_dict: dict[str, int] | None = None
            optimized_messages = messages
            optimized_tokens = original_tokens

            # Get prefix cache tracker for this session. Anthropic carries the
            # system prompt as a top-level field, not a role:"system" message, so
            # fold it into the session-id inputs as a synthetic system message —
            # otherwise every conversation on the same model would share one
            # session id (and its sticky CCR/memory tools, beta headers, and
            # frozen-prefix state). This synthetic message only derives the id; it
            # is never forwarded.
            # Both the session id and the conversation lineage derive from the
            # SAME original client bytes (pre security-scan/hook mutation,
            # which may rewrite messages differently turn-to-turn — a
            # turn-dependent rewrite of the leading system text would rotate
            # the id mid-conversation, orphaning all session-sticky state).
            # The synthetic system message folds the top-level system prompt
            # into both derivations, so conversations sharing an explicit
            # header id but differing in system prompt keep separate lineages.
            system_prompt = body.get("system")
            session_messages = (
                [{"role": "system", "content": system_prompt}, *original_client_messages]
                if system_prompt is not None
                else original_client_messages
            )
            session_id = self.session_tracker_store.compute_session_id(
                request, model, session_messages
            )
            # Prefix trackers must follow the provider's cache key, not just
            # message history.  Anthropic renders tools before system/messages;
            # parallel sub-calls commonly share model+system+history while
            # carrying different tool sets.  Sharing one tracker across those
            # requests pins cache reads at the early tools segment (#2671).
            from headroom.cache.prefix_tracker import segment_fingerprint

            affinity_tools = self._tools_for_forwarding(
                body.get("tools"), preserve_order=preserve_tool_order
            )
            cache_affinity = segment_fingerprint(
                {
                    "model": model,
                    "tools": affinity_tools,
                    "tool_choice": body.get("tool_choice"),
                    "thinking": body.get("thinking"),
                    "output_config": body.get("output_config"),
                }
            )
            # Resolve the tracker by conversation lineage within the session id
            # (#2085): one model + system prompt spans a Claude Code session and
            # all its parallel subagents, so concurrent conversations share this
            # fallback id — on one shared tracker their interleaved histories
            # thrash the frozen-prefix state and the provider prompt cache is
            # re-written on nearly every call.
            prefix_tracker = self.session_tracker_store.resolve_tracker(
                session_id,
                "anthropic",
                messages=session_messages,
                cache_affinity=cache_affinity,
            )
            # Snapshot lineage state once.  Reusing the same pair for delta
            # extraction, byte-stable replay, and breakpoint placement keeps
            # all three decisions tied to one previous request (and avoids
            # repeatedly deep-copying a multi-megabyte agent transcript).
            previous_original_messages = prefix_tracker.get_last_original_messages()
            previous_forwarded_messages = prefix_tracker.get_last_forwarded_messages()
            frozen_message_count = prefix_tracker.get_frozen_message_count()
            # Pre-strict-override tracker truth: >0 only when a provider cache
            # prefix was actually confirmed (or restored) for this session.
            tracker_frozen_count = frozen_message_count
            # Idle gap since the previous turn's response, snapshotted at fetch
            # (before get_or_create bumped the access clock). Forwarded to the
            # pipeline so the net-cost/TTL gate (HEADROOM_NET_COST_POLICY=1) can
            # decay P_alive as the ~300s prompt-cache lapse nears and admit
            # otherwise-frozen compaction as a free rewrite. Harmless when the
            # policy is off (router ignores idle_seconds unless enabled) and near
            # 0 for back-to-back agent turns (cache still warm → no decay).
            idle_seconds = getattr(prefix_tracker, "_idle_seconds_at_fetch", 0.0)
            if is_cache_mode(self.config.mode):
                frozen_message_count = self._strict_previous_turn_frozen_count(
                    original_client_messages,
                    frozen_message_count,
                )
            # Cold-prefix cache-miss hook (HEADROOM_COLD_RECOMPACT). Claude's thinking is
            # an encrypted handle we can't shrink, so when the prompt cache has lapsed
            # (idle past TTL → dead, nothing to bust) we instead recompact the whole
            # prefix — cross-turn dedupe (+HEADROOM_DEDUPE) + superseded-read drop +
            # lossless folds. Decided once here, then applied per mode below: TOKEN mode
            # sets frozen_message_count=0 (reuses the frozen==0 path); CACHE mode runs a
            # lossless whole-prefix recompaction instead of the byte-identical splice
            # (the splice preserves a dead cache) and skips the overlay replay. Both are
            # deterministic → the recompacted prefix re-caches byte-stable on warm turns.
            from headroom.transforms.cold_prefix import (
                anthropic_cache_ttl_seconds,
                is_cold_prefix,
            )

            # Resolve the authoritative request-level prompt-cache tier once.
            # The same value drives cold-prefix handling and net-cost pricing.
            _cc_ttl = anthropic_cache_ttl_seconds(model, original_client_messages, system_prompt)
            _cold_recompact_active = False
            if os.environ.get("HEADROOM_COLD_RECOMPACT", "").strip().lower() in (
                "1",
                "true",
                "yes",
            ):
                _cold_recompact_active = _cc_ttl is None or is_cold_prefix(
                    prefix_tracker, ttl_seconds=_cc_ttl
                )
                if _cold_recompact_active:
                    logger.info(
                        "[%s] cold-prefix recompaction: cc_cache_ttl=%s idle=%.0fs — recompacting "
                        "whole prefix (dedupe/superseded-read/lossless)",
                        request_id,
                        "disabled" if _cc_ttl is None else f"{_cc_ttl}s",
                        idle_seconds,
                    )
                    if is_token_mode(self.config.mode):
                        frozen_message_count = 0

            # PR-A6 (P5-50, preps P0-6): session-sticky `anthropic-beta` merge.
            # Read the client's beta value (note: anthropic-beta is NOT
            # an x-headroom-* header so it survived the A5 strip), union
            # with previously-seen tokens for this session, and update
            # the tracker. Memory-injection (below at line ~1244) uses
            # `merge_anthropic_beta` to add `context-management-2025-06-27`
            # on top of the sticky baseline. Order matters: session-sticky
            # FIRST so we have the canonical baseline; memory injection
            # adds Headroom-required tokens AFTER.
            from headroom.proxy.helpers import (
                get_session_beta_tracker,
                log_beta_header_merge,
            )

            _client_beta_value = headers.get("anthropic-beta")
            _client_beta_count = (
                len([t for t in (_client_beta_value or "").split(",") if t.strip()])
                if _client_beta_value
                else 0
            )
            _sticky_beta_value = get_session_beta_tracker().record_and_get_sticky_betas(
                provider="anthropic",
                session_id=session_id,
                client_value=_client_beta_value,
            )
            _sticky_beta_count = (
                len([t for t in _sticky_beta_value.split(",") if t.strip()])
                if _sticky_beta_value
                else 0
            )
            if _sticky_beta_value and _sticky_beta_value != (_client_beta_value or ""):
                headers["anthropic-beta"] = _sticky_beta_value
            elif not _sticky_beta_value and "anthropic-beta" in headers:
                # Sticky value can only equal "" when both client and
                # session are empty; preserve the (absent) client state.
                pass
            log_beta_header_merge(
                provider="anthropic",
                session_id=session_id,
                client_betas_count=_client_beta_count,
                sticky_betas_count=_sticky_beta_count,
                headroom_added=[],
                request_id=request_id,
            )
            _headroom_beta_added = False

            # In cache mode, avoid rewriting any message body bytes. The latest user
            # turn becomes historical on the next request, so even "latest turn only"
            # rewrites can invalidate the next cache read when the client resends the
            # original transcript.
            #
            # Bypass / image_optimize / messages gating routes through
            # ImageCompressionDecision for uniformity with CompressionDecision +
            # MemoryDecision. The cache_mode check stays inline because it's
            # Anthropic-specific (sites in openai.py / gemini.py don't have it).
            from headroom.proxy.image_compression_decision import ImageCompressionDecision

            _image_decision = ImageCompressionDecision.decide(
                headers=request.headers, config=self.config, messages=messages
            )
            _image_decision.apply_to_tags(tags)
            if _image_decision.should_compress and not is_cache_mode(self.config.mode):
                from headroom.proxy.helpers import COMPRESSION_TIMEOUT_SECONDS

                compressor = None
                try:
                    compressor = _get_image_compressor()
                    if compressor and compressor.has_images(messages):
                        messages, image_result = await run_image_compression_isolated(
                            messages,
                            provider="anthropic",
                            timeout=COMPRESSION_TIMEOUT_SECONDS,
                        )
                        if image_result is not None:
                            body_mutation_tracker.mark_mutated("image_compression")
                            logger.info(
                                f"Image compression: {image_result['technique']} "
                                f"({image_result['savings_percent']:.0f}% saved, "
                                f"{image_result['original_tokens']} -> "
                                f"{image_result['compressed_tokens']} tokens)"
                            )
                except Exception as e:
                    # Image compression is best-effort — fail open on timeout/error and
                    # forward the original messages, matching the text path.
                    logger.warning(f"Image compression failed: {type(e).__name__}: {e}")
                finally:
                    if compressor and hasattr(compressor, "close"):
                        compressor.close()

            _compression_failed = False
            # Provider-confirmed frozen count, stashed BEFORE prepare_turn
            # clamps it against the local byte-replay cache: the overlay's
            # unconditional-replay floor must cover everything the provider
            # has cached, not just what local state can byte-replay (the
            # replay source is the positional prev_fwd snapshot, which does
            # not depend on the CompressionCache). Stays 0 on paths that
            # never compute a tracker count (backpressure, cache mode).
            _confirmed_frozen = 0
            original_messages = messages  # Preserve for 400-retry fallback
            _decision = CompressionDecision.decide(
                headers=request.headers,
                config=self.config,
                usage_reporter=self.usage_reporter,
                messages=messages,
            )
            _decision.apply_to_tags(tags)
            _skip_compression_for_backpressure = (
                _pre_upstream_saturated and _decision.should_compress
            )
            if _skip_compression_for_backpressure:
                tags["passthrough_reason"] = "pre_upstream_backpressure"
                logger.info(
                    "[%s] Compression skipped: reason=pre_upstream_backpressure",
                    request_id,
                )
            if not _decision.should_compress:
                logger.info(
                    f"[{request_id}] Compression skipped: reason={_decision.passthrough_reason}"
                )
            if _decision.should_compress and not _skip_compression_for_backpressure:
                try:
                    from headroom.proxy.helpers import COMPRESSION_TIMEOUT_SECONDS

                    context_limit = self.anthropic_provider.get_context_limit(model)
                    result = None
                    biases = (
                        self.config.hooks.compute_biases(messages, _hook_ctx)
                        if self.config.hooks and _hook_ctx is not None
                        else None
                    )

                    # F2.1 c5/5: derive the per-request CompressionPolicy
                    # from the auth_mode classified at request entry. The
                    # policy short-circuits CacheAligner for subscription
                    # users (closes the cache-instability complaints in
                    # #327/#388). When the enforcement env var is off, the
                    # policy collapses to PAYG so behaviour is unchanged.
                    # Hoisted here so all three pipeline.apply call sites
                    # (token / non-cache / cache-delta) see the same policy.
                    from headroom.transforms.compression_policy import resolve_policy

                    compression_policy = resolve_policy(getattr(request.state, "auth_mode", None))
                    if is_token_mode(self.config.mode):
                        comp_cache = self._get_compression_cache(session_id)

                        # Freeze + stable marking + Zone-1 swap now live in the
                        # shared session engine (PROXY policy: clamp by BOTH the
                        # provider-confirmed count and the locally-replayable
                        # bound — see session_engine.py's module docstring).
                        # `frozen_message_count` here has already been through
                        # tracker + strict-override logic above, so it is the
                        # AUTHORITATIVE positional truth derived from Anthropic's
                        # `cache_read_input_tokens` response.
                        #
                        # Issue #327 (history kept at the call site): a previous
                        # version walked past `prefix_tracker.frozen_message_count`
                        # whenever an upcoming tool_result's content-hash matched
                        # `_stable_hashes` or `should_defer_compression` returned
                        # True. That conflated content equality with positional
                        # cache membership: the prefix cache is positional (bytes
                        # 0..K cached, anything past K is fresh), but
                        # `_stable_hashes` is content-keyed and grows unbounded.
                        # On long Claude Code sessions where tool_result content
                        # rhymes across turns, the walker advanced
                        # `frozen_message_count` to `len(messages)` and the
                        # pipeline produced `transforms_applied=[]` on 73% of
                        # requests. The walker has been removed; trust
                        # `prefix_tracker` clamped by `compute_frozen_count`.
                        from headroom.proxy.session_engine import (
                            FREEZE_POLICY_CONFIRMED_CLAMP,
                            prepare_turn,
                        )

                        _confirmed_frozen = max(int(frozen_message_count or 0), 0)
                        _prep = prepare_turn(
                            comp_cache,
                            messages,
                            policy=FREEZE_POLICY_CONFIRMED_CLAMP,
                            tracker_frozen=frozen_message_count,
                        )
                        frozen_message_count = _prep.frozen_message_count
                        working_messages = _prep.pipeline_input
                        if (
                            getattr(self, "_background_compression_enabled", False)
                            and frozen_message_count == 0
                            and original_tokens >= self._background_compression_min_tokens
                        ):
                            accepted = self._background_compressor.enqueue(
                                session_id,
                                lambda: self.anthropic_pipeline.apply(
                                    messages=working_messages,
                                    model=model,
                                    model_limit=context_limit,
                                    context=extract_user_query(working_messages),
                                    frozen_message_count=frozen_message_count,
                                    idle_seconds=idle_seconds,
                                    biases=biases,
                                    request_id=request_id,
                                    compression_policy=compression_policy,
                                    cache_ttl_seconds=_cc_ttl,
                                    **proxy_pipeline_kwargs(self.config),
                                ),
                                lambda bg_result: comp_cache.update_from_result(
                                    messages, bg_result.messages
                                ),
                            )

                            # Cold-start fast pass: run everything EXCEPT the
                            # Kompress ML stage synchronously before forwarding.
                            # The byte-identical freeze (#1850) locks a session
                            # to whatever form its cold start put in the provider
                            # cache; deferring the WHOLE pipeline locks in the raw
                            # transcript and forfeits the session's savings for
                            # its lifetime — including sub-second wins like
                            # read_lifecycle stale-read drops. Only Kompress can
                            # blow the request budget (#1171), so only Kompress
                            # stays deferred. Fail-open: on timeout/error the
                            # request forwards exactly as before this pass
                            # existed. On timeout the worker can't be cancelled
                            # (Python can't preempt a running thread), but
                            # _run_compression_in_executor runs it on the bounded
                            # compression pool and tracks it via the leaked-thread
                            # metric, so stragglers are capped and observable
                            # rather than unbounded. The pass is also bounded by
                            # routing + statistical crushers (observed seconds even
                            # on multi-M-token counts).
                            from headroom.proxy.helpers import (
                                COLD_START_FAST_PASS_TIMEOUT_SECONDS,
                            )

                            _fast_pass = None
                            try:
                                async with stage_timer.measure("compression_first_stage"):
                                    _fast_pass = await self._run_compression_in_executor(
                                        lambda: self.anthropic_pipeline.apply(
                                            messages=working_messages,
                                            model=model,
                                            model_limit=context_limit,
                                            context=extract_user_query(working_messages),
                                            frozen_message_count=frozen_message_count,
                                            idle_seconds=idle_seconds,
                                            biases=biases,
                                            request_id=request_id,
                                            compression_policy=compression_policy,
                                            cache_ttl_seconds=_cc_ttl,
                                            skip_kompress=True,
                                            **proxy_pipeline_kwargs(self.config),
                                        ),
                                        timeout=COLD_START_FAST_PASS_TIMEOUT_SECONDS,
                                    )
                            except Exception as e:
                                logger.info(
                                    "[%s] Cold-start fast pass skipped (%s: %s); "
                                    "deferring full pipeline to background",
                                    request_id,
                                    type(e).__name__,
                                    e,
                                )

                            if _fast_pass is not None:
                                comp_cache.update_from_result(messages, _fast_pass.messages)
                                _fast_pass.transforms_applied = list(
                                    _fast_pass.transforms_applied
                                ) + [
                                    "deferred:kompress_background"
                                    if accepted
                                    else "deferred:dropped"
                                ]
                                result = _fast_pass
                            else:

                                class _DeferredCompressionResult:
                                    messages = working_messages
                                    transforms_applied = [
                                        "deferred:background_compression"
                                        if accepted
                                        else "deferred:dropped"
                                    ]
                                    timing = {}
                                    waste_signals = None

                                result = _DeferredCompressionResult()
                        else:
                            async with stage_timer.measure("compression_first_stage"):
                                result = await self._run_compression_in_executor(
                                    lambda: self.anthropic_pipeline.apply(
                                        messages=working_messages,
                                        model=model,
                                        model_limit=context_limit,
                                        context=extract_user_query(working_messages),
                                        frozen_message_count=frozen_message_count,
                                        idle_seconds=idle_seconds,
                                        biases=biases,
                                        request_id=request_id,
                                        compression_policy=compression_policy,
                                        cache_ttl_seconds=_cc_ttl,
                                        **proxy_pipeline_kwargs(self.config),
                                    ),
                                    timeout=COMPRESSION_TIMEOUT_SECONDS,
                                )

                        # Cache newly compressed messages (index-aligned diff)
                        if result.messages != working_messages:
                            comp_cache.update_from_result(messages, result.messages)

                        # Always use pipeline result — Zone 1 swaps are already applied
                        optimized_messages = result.messages
                        transforms_applied = result.transforms_applied
                        pipeline_timing = result.timing
                        # Issue #327 / Bug 3: pipeline.apply uses the provider-
                        # side tokenizer (AnthropicProvider tiktoken estimator),
                        # which counts ~25% higher than the proxy-side
                        # EstimatingTokenCounter used to set `original_tokens`
                        # at line 634. Reusing `result.tokens_after` here
                        # produced an apples-vs-oranges comparison against
                        # `original_tokens` in the inflation guard below
                        # (line ~901): even after a real 12% compression the
                        # provider-tokenizer figure was higher than the proxy-
                        # tokenizer baseline, triggering a spurious revert.
                        # Recount optimized_messages with the proxy tokenizer
                        # so original_tokens vs optimized_tokens is self-
                        # consistent. The recount cost (~ms on a 50K-token
                        # request) is paid once per request and is dwarfed by
                        # the upstream call latency.
                        optimized_tokens = tokenizer.count_messages(optimized_messages)
                    elif not is_cache_mode(self.config.mode):
                        async with stage_timer.measure("compression_first_stage"):
                            result = await self._run_compression_in_executor(
                                lambda: self.anthropic_pipeline.apply(
                                    messages=messages,
                                    model=model,
                                    model_limit=context_limit,
                                    context=extract_user_query(messages),
                                    frozen_message_count=frozen_message_count,
                                    biases=biases,
                                    request_id=request_id,
                                    compression_policy=compression_policy,
                                    cache_ttl_seconds=_cc_ttl,
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
                    elif _cold_recompact_active:
                        # CACHE mode, cold turn: the prompt cache is dead, so the
                        # byte-identical splice preserves nothing. Recompact the whole
                        # prefix losslessly (dedupe + superseded-read drop + folds) and
                        # forward that — the overlay replay below is skipped on cold so
                        # this survives. Deterministic → re-caches byte-stable warm.
                        from headroom.transforms.cold_prefix import cold_recompact_messages

                        recompacted, _cold_transforms = await self._run_compression_in_executor(
                            lambda: cold_recompact_messages(
                                original_client_messages,
                                tokenizer=tokenizer,
                                context=extract_user_query(original_client_messages),
                            ),
                            timeout=COMPRESSION_TIMEOUT_SECONDS,
                        )
                        optimized_messages = recompacted
                        optimized_tokens = tokenizer.count_messages(optimized_messages)
                        transforms_applied = _cold_transforms
                    else:
                        if not previous_original_messages and tracker_frozen_count == 0:
                            # Session cold start: nothing has been forwarded for
                            # this session yet and no frozen prefix survives from
                            # a prior process (frozen_message_count > 0 means a
                            # provider cache prefix may already exist upstream —
                            # e.g. a resumed session restored from the CCR
                            # compression cache — so it must stay passthrough),
                            # hence there is no provider cache prefix to protect — run the same full-message
                            # compression as non-cache modes (issue #2357: the
                            # previous silent passthrough here meant cache mode
                            # never compressed anything until a stable delta
                            # appeared, which for resumed 100k-token transcripts
                            # was never). The compressed output is recorded as
                            # the forwarded messages below, and later turns
                            # replay that prefix byte-identically via the
                            # stable-delta path, so append-only cache safety is
                            # preserved.
                            delta = None
                            async with stage_timer.measure("compression_first_stage"):
                                result = await self._run_compression_in_executor(
                                    lambda: self.anthropic_pipeline.apply(
                                        messages=messages,
                                        model=model,
                                        model_limit=context_limit,
                                        context=extract_user_query(messages),
                                        frozen_message_count=frozen_message_count,
                                        biases=biases,
                                        request_id=request_id,
                                        compression_policy=compression_policy,
                                        **proxy_pipeline_kwargs(self.config),
                                    ),
                                    timeout=COMPRESSION_TIMEOUT_SECONDS,
                                )

                            if result.messages != messages:
                                optimized_messages = result.messages
                                transforms_applied = list(result.transforms_applied) + [
                                    "cache_mode:cold_start_full"
                                ]
                                pipeline_timing = result.timing
                                original_tokens = result.tokens_before
                                optimized_tokens = result.tokens_after
                        else:
                            delta = self._extract_cache_stable_delta(
                                original_client_messages,
                                previous_original_messages,
                                previous_forwarded_messages,
                            )
                        if delta is not None:
                            stable_forwarded_prefix, delta_messages = delta
                            if delta_messages:
                                # Compress the delta, with two cache-mode adjustments:
                                #
                                # fix-5: strip the client's transient cache_control marker so
                                #   the router's per-block "never compress an explicit cache
                                #   key" guard (content_router.py:4006) doesn't skip the ONLY
                                #   compressible content every turn (route_counts had
                                #   cache_control_protected == the whole delta -> 0%). In cache
                                #   mode that marker is NOT the real forwarded breakpoint: the
                                #   compressed delta is frozen + replayed verbatim next turn and
                                #   normalize_message_cache_control (AFTER compression, below)
                                #   owns the single forwarded breakpoint. Cache-safety is
                                #   enforced post-compression, not by protecting the delta.
                                #
                                # fix-6: the delta is a lone tool_result whose tool_use (tool
                                #   NAME + call args) lives in the frozen prefix. Passing only
                                #   the delta to the router leaves tool_name="" so
                                #   _bash_search_fold (lossless grep/rg folding, no size floor),
                                #   per-tool bias, and relevance-query enrichment all degrade.
                                #   Pass the FULL current messages with frozen_message_count =
                                #   prefix length: _build_tool_name_map scans ALL messages (the
                                #   delta resolves its tool_name from the prefix's tool_use) but
                                #   the compression loop only touches indices >= frozen count,
                                #   so ONLY the delta is compressed. Splice the compressed delta
                                #   onto the byte-stable forwarded prefix.
                                from headroom.cache.prefix_tracker import _strip_cache_control

                                # Compression context = the EXACT forwarded (cached) prefix
                                # + the stripped delta, with the prefix frozen. Using the
                                # forwarded prefix (not the original) keeps _build_tool_name_map
                                # AND cross-turn dedup consistent with what is actually cached:
                                # dedup can only reference bytes that are truly present in the
                                # forwarded context, so no pointer can dangle. The prefix is
                                # frozen (never compressed) and we discard the router's copy of
                                # it below, so the forwarded prefix stays byte-identical to last
                                # turn -> append-only -> no bust.
                                prefix_n = len(stable_forwarded_prefix)
                                compression_input = list(stable_forwarded_prefix) + list(
                                    _strip_cache_control(delta_messages)
                                )
                                result = await self._run_compression_in_executor(
                                    lambda: self.anthropic_pipeline.apply(
                                        messages=compression_input,
                                        model=model,
                                        model_limit=context_limit,
                                        context=extract_user_query(compression_input),
                                        frozen_message_count=prefix_n,
                                        idle_seconds=idle_seconds,
                                        biases=biases,
                                        request_id=request_id,
                                        compression_policy=compression_policy,
                                        cache_ttl_seconds=_cc_ttl,
                                        **proxy_pipeline_kwargs(self.config),
                                    ),
                                    timeout=COMPRESSION_TIMEOUT_SECONDS,
                                )
                                # Only the delta was eligible for compression (prefix frozen);
                                # forward the byte-identical cached prefix + the compressed delta.
                                compressed_delta = result.messages[prefix_n:]
                                optimized_messages = stable_forwarded_prefix + compressed_delta
                                transforms_applied = result.transforms_applied
                                pipeline_timing = result.timing
                                optimized_tokens = tokenizer.count_messages(optimized_messages)
                            else:
                                optimized_messages = stable_forwarded_prefix
                                optimized_tokens = tokenizer.count_messages(optimized_messages)
                        elif previous_original_messages:
                            # Conservative rule for cache mode:
                            # only replay exact stable message-prefix extensions.
                            # In-message append rewriting is deferred until we can
                            # prove it is perfectly replayable across future turns.
                            # Tag the passthrough so the dashboard can explain 0
                            # savings instead of silently reporting "optimization
                            # enabled" with Before == After (issue #2357).
                            tags["passthrough_reason"] = "cache_mode_prefix_mismatch"
                            logger.info(
                                "[%s] Compression skipped: reason=cache_mode_prefix_mismatch",
                                request_id,
                            )
                            optimized_messages = messages
                            optimized_tokens = original_tokens
                        elif tracker_frozen_count > 0:
                            # Cold start with a frozen prefix restored from a
                            # prior process: the provider cache may already hold
                            # that prefix, so forward unmodified.
                            tags["passthrough_reason"] = "cache_mode_frozen_cold_start"
                            optimized_messages = messages
                            optimized_tokens = original_tokens

                    if result and result.waste_signals:
                        waste_signals_dict = result.waste_signals.to_dict()
                except Exception as e:
                    # Include type so TimeoutError vs other failures is distinguishable
                    # in bug reports — str(asyncio.TimeoutError()) is empty otherwise.
                    logger.warning(f"[{request_id}] Optimization failed: {type(e).__name__}: {e}")
                    # Flag compression failure for observability
                    _compression_failed = True
                    # Split timeout from other errors: a timeout means the
                    # compression budget was too tight, not a code bug.
                    reason = (
                        "timeout" if isinstance(e, asyncio.TimeoutError | TimeoutError) else "error"
                    )
                    self.metrics.record_compression_failed(reason)

            # Cache-safety (ALL modes): forward the previously-cached (compressed)
            # prefix byte-identical. The freeze path can emit the agent's ORIGINAL
            # bytes for a frozen message, but the provider cached whatever we
            # FORWARDED last turn (the compressed form); forwarding original then
            # mismatches the cached prefix and busts it (prefix_change was 100% of
            # observed misses, ~56% of all cache-writes). Replaying the exact
            # previously-forwarded prefix keeps it byte-identical → cache hits.
            # Append-only-guarded and idempotent (cache mode already replays), so
            # it is safe to run unconditionally here.
            from headroom.cache.prefix_tracker import normalize_message_cache_control
            from headroom.proxy.session_engine import finalize_turn

            _overlay_replayed = False
            # On a confirmed-cold turn we deliberately do NOT replay the previously
            # forwarded prefix: the cache is dead (nothing to keep byte-identical for)
            # and the replay would clobber the whole-prefix recompaction we just did.
            #
            # Backpressure skips the compression PIPELINE but must NOT skip this
            # replay: on the saturated path `optimized_messages` is the raw
            # originals, which mismatch the compressed prefix the provider cached
            # — so every gated request busted its session's prompt cache exactly
            # when traffic (and the re-write cost) peaked. The overlay itself is
            # O(prefix) comparisons plus one token recount only when it actually
            # replays, which is far cheaper than the whole-prefix cache re-write
            # it prevents, so it stays on even under backpressure.
            if _decision.should_compress:
                if _cold_recompact_active:
                    _overlay_replayed = False
                else:
                    _final = finalize_turn(
                        optimized_messages,
                        original_client_messages,
                        previous_original_messages,
                        previous_forwarded_messages,
                        count_tokens=tokenizer.count_messages,
                        # The provider-confirmed prefix is replayed
                        # unconditionally: those bytes are exactly what the
                        # provider cached, so declining a byte-larger replay
                        # there re-forwards freshly recompressed history at
                        # the full input rate and busts the cache every time
                        # background compression improves on an
                        # already-forwarded message. Beyond the confirmed
                        # floor the size bound still lets improvements
                        # through, and a collapsed floor (cold cache) lets
                        # every accumulated improvement land at once.
                        confirmed_frozen_count=_confirmed_frozen,
                    )
                    _overlay_replayed = _final.replayed
                    if _overlay_replayed:
                        optimized_messages = _final.messages
                        if _final.tokens is not None:
                            optimized_tokens = _final.tokens
            else:
                logger.debug(
                    "[%s] Cached-prefix replay skipped: reason=%s",
                    request_id,
                    _decision.passthrough_reason,
                )

            # Own cache_control placement: the client moves the breakpoint each
            # turn and the overlay replays past markers, so they accumulate ~1/turn
            # and Anthropic hard-errors at >4. Strip message-level markers and
            # re-place them at the CLIENT's current positions: Anthropic resolves
            # each breakpoint with a ~20-content-block lookback, and the client's
            # previous-message marker is the read anchor that lets a big turn's
            # write chain to the prior entry. Collapsing to one newest-block
            # marker breaks that chain on tool-heavy turns (silent full re-write)
            # and can leave the tail billing uncached. Applied last so the
            # forwarded AND recorded (next_forwarded) messages stay bounded.
            _norm = normalize_message_cache_control(
                optimized_messages,
                previous_forwarded_messages,
                client_messages=original_client_messages,
            )
            if _norm is not optimized_messages:
                optimized_messages = _norm

            # Guard: if "optimization" inflated tokens, revert to originals.
            # Skip in cache mode where prefix-stability may legitimately shift counts.
            # ALSO skip when overlay_cached_prefix just replayed a byte-identical
            # cached prefix (token mode): reverting to the raw originals there
            # would re-forward the uncompressed prefix and BUST the live prompt
            # cache — trading a 90% read discount for a full re-write — which is
            # far more expensive than the small tail inflation this guard exists
            # to avoid. The nominal "inflation" is also an artifact of comparing
            # the cached (compressed) forwarding against the raw original count.
            if (
                optimized_tokens > original_tokens
                and not is_cache_mode(self.config.mode)
                and not _overlay_replayed
            ):
                logger.warning(
                    f"[{request_id}] Optimization inflated tokens "
                    f"({original_tokens} -> {optimized_tokens}), reverting to original messages"
                )
                optimized_messages = original_messages
                optimized_tokens = original_tokens
                transforms_applied = []

            tokens_saved = max(0, original_tokens - optimized_tokens)
            optimization_latency = (time.time() - start_time) * 1000

            routing_markers = summarize_routing_markers(transforms_applied)
            if routing_markers:
                routed_event = self.pipeline_extensions.emit(
                    PipelineStage.INPUT_ROUTED,
                    operation="proxy.request",
                    request_id=request_id,
                    provider=pipeline_provider,
                    model=model,
                    messages=optimized_messages,
                    metadata={
                        "routing_markers": routing_markers,
                        "transforms_applied": transforms_applied,
                    },
                )
                if routed_event.messages is not None:
                    previous_optimized_messages = optimized_messages
                    optimized_messages = routed_event.messages
                    if routed_event.messages is not previous_optimized_messages:
                        optimized_tokens = tokenizer.count_messages(optimized_messages)
                        tokens_saved = max(0, original_tokens - optimized_tokens)

            compressed_event = self.pipeline_extensions.emit(
                PipelineStage.INPUT_COMPRESSED,
                operation="proxy.request",
                request_id=request_id,
                provider=pipeline_provider,
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
                previous_optimized_messages = optimized_messages
                optimized_messages = compressed_event.messages
                if compressed_event.messages is not previous_optimized_messages:
                    optimized_tokens = tokenizer.count_messages(optimized_messages)
                    tokens_saved = max(0, original_tokens - optimized_tokens)

            # Mechanism B: activity-based read maturation (flag-gated,
            # default off). Runs after compression so read_lifecycle
            # markers are respected, and before body assembly so the
            # held-Read breakpoint relocation lands in the forwarded
            # request. Session state (matured markers) rides on the
            # prefix tracker — same affinity and TTL cleanup as the
            # freeze state. Advisory: must never fail the request.
            if self.config.read_maturation and not _bypass:
                try:
                    from headroom.config import ReadMaturationConfig
                    from headroom.transforms.read_maturation import (
                        ReadMaturationManager,
                        relocate_cache_breakpoint,
                    )

                    maturation_mgr = prefix_tracker.read_maturation_manager
                    if maturation_mgr is None:
                        maturation_mgr = ReadMaturationManager(
                            ReadMaturationConfig(
                                enabled=True,
                                quiesce_turns=self.config.read_maturation_quiesce_turns,
                                max_hold_turns=self.config.read_maturation_max_hold_turns,
                                min_size_bytes=self.config.read_maturation_min_size_bytes,
                            ),
                            compression_store=get_compression_store(),
                        )
                        prefix_tracker.read_maturation_manager = maturation_mgr
                    maturation = maturation_mgr.apply(
                        optimized_messages,
                        frozen_message_count=frozen_message_count,
                    )
                    if maturation.replacements_applied or maturation.holding_msg_indices:
                        optimized_messages = relocate_cache_breakpoint(
                            maturation.messages,
                            maturation.holding_msg_indices,
                        )
                        optimized_tokens = tokenizer.count_messages(optimized_messages)
                        tokens_saved = max(0, original_tokens - optimized_tokens)
                        if maturation.newly_matured:
                            transforms_applied.append(f"read_maturation:{maturation.newly_matured}")
                        logger.debug(
                            f"[{request_id}] read_maturation: "
                            f"holding={len(maturation.holding_msg_indices)} "
                            f"matured={maturation.newly_matured} "
                            f"replayed={maturation.replacements_applied} "
                            f"bytes_saved={maturation.bytes_saved}"
                        )
                except Exception as e:
                    logger.warning(f"[{request_id}] read maturation failed: {e}")

            # Hook: post_compress — let hooks observe compression results
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
                            user_query=_hook_ctx.user_query if self.config.hooks else "",
                            provider=provider_name,
                        )
                    )
                except Exception as e:
                    logger.debug(f"[{request_id}] post_compress hook error: {e}")

            # CCR Tool Injection: Inject retrieval tool if compression occurred
            # OR if this session has previously done CCR (PR-B7 sticky-on).
            # The legacy `CCRToolInjector` flips on/off based on the *current*
            # request's compressed-content presence, busting cache every flip.
            # We now route the tool-list update through
            # `apply_session_sticky_ccr_tool`, which once-on/always-on per
            # `SessionCcrTracker`. System-instruction injection keeps its
            # existing per-request scan (it lives in the system prompt, which
            # is the cache hot zone — gated separately by the
            # `frozen_message_count > 0` guard below).
            tools = body.get("tools")
            _original_tools = tools  # Preserve for diagnostic / future retry

            # Issue #746: when Claude Code talks to a custom ANTHROPIC_BASE_URL
            # with ENABLE_TOOL_SEARCH unset, it stops deferring tool schemas and
            # loads them all into local context. That is a client-side decision
            # we cannot reverse from here, so emit a single actionable hint for
            # users who launch `claude` manually (the wrap path sets the env var).
            # Gate on the cheap one-time flag first so the detection scan stops
            # running once the hint has fired; never let it break a request.
            from headroom.proxy.helpers import tool_search_hint_pending

            if tool_search_hint_pending():
                try:
                    from headroom.proxy.helpers import (
                        claude_code_tool_search_inactive,
                        format_tool_search_disabled_hint,
                        take_tool_search_hint_slot,
                    )

                    if (
                        claude_code_tool_search_inactive(
                            client=client,
                            tools=tools,
                            anthropic_beta=request.headers.get("anthropic-beta"),
                        )
                        and take_tool_search_hint_slot()
                    ):
                        logger.warning(
                            "[%s] %s", request_id, format_tool_search_disabled_hint(tools)
                        )
                except Exception:  # advisory hint only — must never fail a request
                    pass
            # Initialize before the gated block so the proactive-expansion
            # gate below (which references ``ccr_workspace_key`` regardless of
            # the inject flags) does not raise ``UnboundLocalError`` when the
            # block is skipped — e.g. ``--no-ccr-inject-tool`` with the default
            # ``ccr_inject_system_instructions=False``, or when ``_bypass`` is
            # set. The downstream uses already treat falsy as "unresolved".
            ccr_workspace_key, ccr_workspace_label = None, None
            if (
                self.config.ccr_inject_tool or self.config.ccr_inject_system_instructions
            ) and not _bypass:
                inject_system_instructions = self.config.ccr_inject_system_instructions
                if inject_system_instructions and frozen_message_count > 0:
                    logger.info(
                        f"[{request_id}] CCR: skipping system instruction injection "
                        f"(frozen prefix={frozen_message_count}) to preserve cache"
                    )
                    inject_system_instructions = False
                configured_inject_tool = self.config.ccr_inject_tool
                # Scan for compression markers + maybe inject system instructions.
                # Tool-list injection is handled separately via the sticky helper.
                injector = CCRToolInjector(
                    provider="anthropic",
                    inject_tool=False,  # routed through sticky helper below
                    inject_system_instructions=inject_system_instructions,
                )
                injector.scan_for_markers(optimized_messages)
                # Shape-only scanning also matches markers from other context
                # tools; drop hashes this proxy never actually stored before
                # they can drive tool injection (issue #2836).
                injector.verify_ownership()
                if inject_system_instructions and injector.has_compressed_content:
                    optimized_messages = injector.inject_into_system_message(optimized_messages)

                # Sticky-on tool registration (PR-B7): always inject the
                # retrieval tool once a session has done CCR, regardless
                # of whether THIS turn produced compressed content.
                #
                # Injection is deliberately NOT gated on
                # ``frozen_message_count``. That counter answers "is the
                # prefix warm?", but the decision needs "does the established
                # prefix already contain the tool?" — which only
                # ``SessionCcrTracker`` knows. Gating on the counter dropped a
                # tool that was already inside the provider-cached prefix, and
                # ``tools`` is the head of Anthropic's cache key, so every
                # toggle invalidated the entire prefix in both directions.
                # ``apply_session_sticky_ccr_tool`` carries the correct rule:
                # a session that has never compressed still gets no tool, so
                # dropping the gate cannot start injecting into non-CCR
                # conversations.
                if configured_inject_tool:
                    from headroom.proxy.helpers import (
                        apply_session_sticky_ccr_tool,
                        history_references_ccr_tool,
                    )

                    # Inject whenever the request carries ANY CCR marker, new or
                    # replayed from the frozen prefix. #1850 narrowed the
                    # first-time gate to markers created THIS turn to avoid
                    # arming a session that never compressed, but a replayed
                    # marker is exactly as unredeemable as a fresh one: the agent
                    # redeems hashes it was handed turns ago (project instructions
                    # can even tell it to), and if `headroom_retrieve` is absent
                    # Anthropic rejects the whole request with 400 "Tool reference
                    # 'headroom_retrieve' not found in available tools" (#2766). A
                    # present marker means the session HAS compressed, so this
                    # cannot start injecting into non-CCR conversations. It is also
                    # the cache-stable choice: toggling the tool in and out of the
                    # tools array between turns is what busts the tools cache
                    # segment, whereas injecting consistently whenever markers
                    # exist keeps it stable. The `SessionCcrTracker` is
                    # per-process, so a proxy restart mid-conversation makes live
                    # sessions look fresh again, which is what re-armed the 400.
                    tools, ccr_tool_injected = apply_session_sticky_ccr_tool(
                        provider="anthropic",
                        session_id=session_id,
                        request_id=request_id,
                        existing_tools=tools,
                        has_compressed_content_this_turn=injector.has_compressed_content,
                        history_has_ccr_reference=history_references_ccr_tool(optimized_messages),
                    )
                    if ccr_tool_injected:
                        logger.debug(
                            f"[{request_id}] CCR: tool registered (session={session_id}, "
                            f"compressed_this_turn={injector.has_compressed_content}, "
                            f"hashes_seen={len(injector.detected_hashes)})"
                        )

                # CCR workspace scoping: resolve a stable project identity
                # for the request once and reuse it for both track_compression
                # AND analyze_query. The shared `self.ccr_context_tracker`
                # is process-global across all sessions/projects served by
                # this proxy; without this gate, Project A's compressed
                # sample content keyword-matches Project B's later query
                # and gets surfaced as "relevant" — see
                # `headroom/ccr/context_tracker.py` module docstring for
                # the 2026-05-26 leak report (Python from tamag0
                # injected into a daphni-rails Ruby session).
                ccr_workspace_key, ccr_workspace_label = self._resolve_ccr_workspace(request, body)

                if injector.has_compressed_content:
                    # Track compression in context tracker for multi-turn awareness.
                    # Gated on a resolved workspace: tracking under an empty
                    # workspace would create entries that the workspace-filter
                    # in analyze_query can never match. Fail-closed per
                    # `feedback_no_silent_fallbacks`.
                    if self.ccr_context_tracker and ccr_workspace_key:
                        self._turn_counter += 1
                        for hash_key in injector.detected_hashes:
                            # Get compression metadata from store
                            store = get_compression_store()
                            entry = store.get_metadata(hash_key)
                            if entry:
                                if looks_like_claude_code_compact_summary(
                                    entry.get("query_context"),
                                    entry.get("compressed_content"),
                                    entry.get("original_content_preview"),
                                ):
                                    logger.info(
                                        f"[{request_id}] CCR: skipping proactive "
                                        f"tracking for Claude Code compact summary {hash_key}"
                                    )
                                    continue
                                self.ccr_context_tracker.track_compression(
                                    hash_key=hash_key,
                                    turn_number=self._turn_counter,
                                    tool_name=entry.get("tool_name"),
                                    original_count=entry.get("original_item_count", 0),
                                    compressed_count=entry.get("compressed_item_count", 0),
                                    workspace_key=ccr_workspace_key,
                                    query_context=entry.get("query_context", ""),
                                    sample_content=entry.get("compressed_content", "")[:500],
                                )
                    elif self.ccr_context_tracker and not ccr_workspace_key:
                        logger.info(
                            f"[{request_id}] CCR: workspace unresolved; skipping "
                            "track_compression (fail-closed — no x-headroom-cwd / "
                            "x-headroom-project-id header and no cwd: in system prompt)"
                        )

            # CCR Proactive Expansion: Check if current query needs expanded context.
            # Same workspace gate as track_compression above.
            if (
                self.ccr_context_tracker
                and self.config.ccr_proactive_expansion
                and ccr_workspace_key
            ):
                # Extract user query from messages
                user_query = ""
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            user_query = content
                        elif isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    user_query = block.get("text", "")
                                    break
                        break

                if user_query:
                    recommendations = self.ccr_context_tracker.analyze_query(
                        user_query,
                        self._turn_counter,
                        workspace_key=ccr_workspace_key,
                    )
                    if recommendations:
                        expansions = self.ccr_context_tracker.execute_expansions(recommendations)
                        if expansions:
                            # Add expanded context to the system message or as additional context.
                            # Pass workspace_label so the injected block declares its provenance
                            # — symmetric with the memory-injection block header.
                            expansion_text = self.ccr_context_tracker.format_expansions_for_context(
                                expansions,
                                workspace_label=ccr_workspace_label,
                            )
                            logger.info(
                                f"[{request_id}] CCR: Proactively expanded {len(expansions)} context(s) "
                                f"based on query relevance"
                            )
                            if is_cache_mode(self.config.mode):
                                logger.info(
                                    f"[{request_id}] CCR: skipping proactive expansion append "
                                    "in cache mode to preserve next-turn prefix stability"
                                )
                            else:
                                optimized_messages = (
                                    self._append_context_to_latest_non_frozen_user_turn(
                                        optimized_messages,
                                        expansion_text,
                                        frozen_message_count=frozen_message_count,
                                    )
                                )

            # Traffic Learner: Extract patterns from inbound tool results
            if self.traffic_learner:
                try:
                    # Wire backend on first use (lazy init after memory handler is ready)
                    if (
                        self.traffic_learner._backend is None
                        and self.memory_handler
                        and self.memory_handler.initialized
                        and self.memory_handler.backend
                    ):
                        self.traffic_learner.set_backend(self.memory_handler.backend)

                    # Extract tool results from messages and learn from them
                    tool_results = self.traffic_learner.extract_tool_results_from_messages(
                        optimized_messages
                    )
                    for tr in tool_results[-5:]:  # Only recent results
                        await self.traffic_learner.on_tool_result(
                            tool_name=tr["tool_name"],
                            tool_input=tr["input"],
                            tool_output=tr["output"],
                            is_error=tr["is_error"],
                        )

                    # Also extract preference signals from user messages
                    await self.traffic_learner.on_messages(optimized_messages)
                except Exception as e:
                    logger.debug(f"[{request_id}] Traffic learner: {e}")

            # Memory: Inject context and tools — gated on MemoryDecision.
            # ``inject`` is False under bypass, missing handler, missing
            # user_id, or HEADROOM_MEMORY_INJECTION_MODE in disabled/tool.
            # Pre-PR-this the gate was a raw conjunction that silently
            # ignored bypass; now bypass is honoured here on Anthropic
            # /v1/messages just as on /v1/responses.
            memory_context_injected = False
            memory_tools_injected = False
            if memory_decision.inject:
                # Search and inject memory context
                if self.memory_handler.config.inject_context:
                    try:
                        async with stage_timer.measure("memory_context"):
                            memory_context = await asyncio.wait_for(
                                self.memory_handler.search_and_format_context(
                                    memory_user_id,
                                    optimized_messages,
                                    request_context=memory_request_ctx,
                                    query=MemoryQuery.from_messages(optimized_messages),
                                ),
                                timeout=(
                                    self.config.anthropic_pre_upstream_memory_context_timeout_seconds
                                ),
                            )
                    except asyncio.TimeoutError:
                        memory_context = None
                        logger.info(
                            f"[{request_id}] Memory: Context lookup exceeded "
                            f"{self.config.anthropic_pre_upstream_memory_context_timeout_seconds:.1f}s; "
                            "continuing without it"
                        )
                    try:
                        if memory_context:
                            from headroom.proxy.helpers import (
                                get_memory_injection_mode,
                                log_memory_injection,
                            )

                            injection_mode = get_memory_injection_mode()
                            user_query = extract_user_query(optimized_messages) or ""
                            if injection_mode == "disabled":
                                log_memory_injection(
                                    request_id=request_id,
                                    session_id=session_id,
                                    decision="skipped_disabled",
                                    bytes_injected=0,
                                    query=user_query,
                                )
                            elif is_cache_mode(self.config.mode):
                                # Cache mode: skip injection entirely so the next-turn
                                # prefix bytes remain byte-equal to this turn's bytes.
                                log_memory_injection(
                                    request_id=request_id,
                                    session_id=session_id,
                                    decision="skipped_cache_mode",
                                    bytes_injected=0,
                                    query=user_query,
                                )
                            else:
                                # P0-1 fix: route exclusively to the live zone tail
                                # (latest non-frozen user turn). System prompt + frozen
                                # prefix are never mutated — invariant I2.
                                before = optimized_messages
                                optimized_messages = (
                                    self._append_context_to_latest_non_frozen_user_turn(
                                        optimized_messages,
                                        memory_context,
                                        frozen_message_count=frozen_message_count,
                                    )
                                )
                                if optimized_messages is not before:
                                    memory_context_injected = True
                                    log_memory_injection(
                                        request_id=request_id,
                                        session_id=session_id,
                                        decision="injected_live_zone_tail",
                                        bytes_injected=len(memory_context),
                                        query=user_query,
                                        tags=tags,
                                    )
                                else:
                                    log_memory_injection(
                                        request_id=request_id,
                                        session_id=session_id,
                                        decision="no_eligible_user_turn",
                                        bytes_injected=0,
                                        query=user_query,
                                    )
                    except Exception as e:
                        logger.warning(f"[{request_id}] Memory: Context injection failed: {e}")

                # Inject memory tools — PR-A7 (P0-6) routes through
                # `apply_session_sticky_memory_tools` so tool list bytes
                # stay byte-stable across turns: once a session injects,
                # every subsequent turn replays the same canonical bytes.
                # `inject_this_turn` is True iff memory is enabled this
                # turn (i.e. memory_handler.config.inject_tools and we
                # have a memory_user_id, which the outer guard at line
                # 1192 already enforces).
                from headroom.proxy.helpers import (
                    apply_session_sticky_memory_tools,
                )

                memory_tool_defs = (
                    self.memory_handler.compute_memory_tool_definitions("anthropic")
                    if self.memory_handler.config.inject_tools
                    else []
                )
                tools, mem_tools_injected = apply_session_sticky_memory_tools(
                    provider="anthropic",
                    session_id=session_id,
                    request_id=request_id,
                    existing_tools=tools,
                    memory_tools_to_inject=memory_tool_defs,
                    inject_this_turn=bool(self.memory_handler.config.inject_tools),
                )
                if mem_tools_injected:
                    memory_tools_injected = True
                    tool_names = [
                        t.get("name") or t.get("type", "")
                        for t in tools
                        if t.get("name", "").startswith("memory")
                        or t.get("type", "").startswith("memory")
                    ]
                    logger.info(f"[{request_id}] Memory: Injected tools: {tool_names}")

                    # Add beta headers for native memory tool. PR-A6
                    # (P5-50): use the deterministic `merge_anthropic_beta`
                    # helper instead of ad-hoc string concat. Order:
                    # client tokens first (preserved from session-sticky
                    # baseline above), then Headroom-required tokens.
                    # The session tracker already recorded the client
                    # value; we append Headroom-required tokens here so
                    # the next turn re-applies them deterministically.
                    beta_headers = self.memory_handler.get_beta_headers()
                    if beta_headers:
                        from headroom.proxy.helpers import (
                            log_beta_header_merge as _log_beta_header_merge_mem,
                        )
                        from headroom.proxy.helpers import (
                            merge_anthropic_beta,
                        )

                        for key, value in beta_headers.items():
                            if key.lower() != "anthropic-beta":
                                # Defensive: memory handler currently
                                # only emits anthropic-beta. Any future
                                # provider-specific beta header would
                                # need its own merge helper.
                                headers[key] = value
                                continue
                            existing_value = headers.get(key, "")
                            required_tokens = [t.strip() for t in value.split(",") if t.strip()]
                            _headroom_beta_added = True
                            merged = merge_anthropic_beta(existing_value, required_tokens)
                            _existing_count = (
                                len([t for t in existing_value.split(",") if t.strip()])
                                if existing_value
                                else 0
                            )
                            _merged_count = (
                                len([t for t in merged.split(",") if t.strip()]) if merged else 0
                            )
                            headers[key] = merged
                            _log_beta_header_merge_mem(
                                provider="anthropic",
                                session_id=session_id,
                                client_betas_count=_existing_count,
                                sticky_betas_count=_merged_count,
                                headroom_added=required_tokens,
                                request_id=request_id,
                            )
                            logger.info(f"[{request_id}] Memory: Added beta header: {key}={merged}")

            if memory_context_injected or memory_tools_injected:
                remembered_event = self.pipeline_extensions.emit(
                    PipelineStage.INPUT_REMEMBERED,
                    operation="proxy.request",
                    request_id=request_id,
                    provider=pipeline_provider,
                    model=model,
                    messages=optimized_messages,
                    tools=tools,
                    headers=headers,
                    metadata={
                        "memory_context_injected": memory_context_injected,
                        "memory_tools_injected": memory_tools_injected,
                    },
                )
                if remembered_event.messages is not None:
                    optimized_messages = remembered_event.messages
                if remembered_event.tools is not None:
                    tools = remembered_event.tools
                if remembered_event.headers is not None:
                    headers = remembered_event.headers

            # Final sanitization of the FORWARDED body: strip streaming-only "index"
            # keys from content blocks. In cache mode the forwarded prefix is replayed
            # from Headroom's own recorded/reconstructed messages (streaming.py tags
            # blocks with "index"), so the inbound-side strip above doesn't cover it —
            # this catches both the client-echoed and the cache-replayed forms. Applied
            # deterministically to the exact bytes forwarded (and thus recorded as the
            # next prefix), so Anthropic always caches/matches the same stripped prefix.
            _strip_streaming_only_content_fields(optimized_messages)
            # Update body
            body["messages"] = optimized_messages
            if tools or _original_tools is not None:
                forwarded_tools = self._tools_for_forwarding(
                    tools,
                    preserve_order=preserve_tool_order,
                )
                if forwarded_tools != tools:
                    tools = forwarded_tools
                if tools != _original_tools:
                    body["tools"] = tools

            # Tool schema compaction: strip annotation keys ($schema, title,
            # examples, etc.) and normalise description whitespace.  Runs
            # after tools are finalised (sorting, CCR injection) but before
            # the PRE_SEND pipeline event so extensions see the compacted
            # schema.  Mirrors the same pass that the OpenAI handler applies.
            #
            # Token accounting (see tool_schema_savings_policy): the compaction passes
            # below rewrite the tool array, so both endpoints are countable and the
            # delta is folded into original/optimized_tokens at the final recount.
            # Without this the savings were computed, debug-logged, and discarded —
            # Claude Code reported them nowhere.
            _tool_tokens_before = 0
            _tool_tokens_after = 0

            def _count_tool_tokens(value: object) -> int:
                try:
                    return tokenizer.count_text(json.dumps(value, default=str))
                except Exception:
                    return 0

            _tools_compaction_started = time.time()
            try:
                from headroom.proxy.tool_schema_compaction import compact_tools

                _pre_compaction_tools = body.get("tools")
                _tools_modified = False
                # Auxiliary passes honor the same disable/bypass decision as messages.
                if _decision.should_compress:
                    body, _tools_modified, _tools_before_bytes, _tools_after_bytes = compact_tools(
                        body
                    )
                if _tools_modified:
                    tools = body["tools"]
                    transforms_applied.append("anthropic:tool_schema_compaction")
                    _tool_tokens_before = _count_tool_tokens(_pre_compaction_tools)
                    _tool_tokens_after = _count_tool_tokens(tools)
                    _tools_compaction_ms = (time.time() - _tools_compaction_started) * 1000
                    logger.debug(
                        "[%s] tool schema compaction: %d -> %d bytes (%.0f%% saved) in %.1fms",
                        request_id,
                        _tools_before_bytes,
                        _tools_after_bytes,
                        (1 - _tools_after_bytes / max(_tools_before_bytes, 1)) * 100,
                        _tools_compaction_ms,
                    )
            except Exception as _tools_compaction_exc:
                _tools_modified = False
                logger.warning(
                    "[%s] tool schema compaction FAILED: %s", request_id, _tools_compaction_exc
                )

            # Layer 2: Tool description truncation (opt-in via
            # HEADROOM_TOOL_DESC_MAX_CHARS).  Preserves first sentence
            # of each description so tool selection still works.
            try:
                from headroom.proxy.tool_schema_compaction import (
                    compact_tool_descriptions,
                    tool_desc_max_chars,
                )

                _desc_max = tool_desc_max_chars()
                if _decision.should_compress and _desc_max > 0:
                    _pre_desc_tools = body.get("tools")
                    body, _desc_modified, _desc_before, _desc_after = compact_tool_descriptions(
                        body, _desc_max
                    )
                    if _desc_modified:
                        tools = body["tools"]
                        transforms_applied.append("anthropic:tool_desc_compaction")
                        # Runs after schema compaction, so only seed "before" when that
                        # pass didn't already; "after" always tracks the latest tools.
                        if not _tool_tokens_before:
                            _tool_tokens_before = _count_tool_tokens(_pre_desc_tools)
                        _tool_tokens_after = _count_tool_tokens(tools)
                        logger.debug(
                            "[%s] tool description compaction: %d -> %d bytes (%.0f%% saved, max_chars=%d)",
                            request_id,
                            _desc_before,
                            _desc_after,
                            (1 - _desc_after / max(_desc_before, 1)) * 100,
                            _desc_max,
                        )
            except Exception as _desc_compaction_exc:
                _desc_modified = False
                logger.warning(
                    "[%s] tool desc compaction FAILED: %s", request_id, _desc_compaction_exc
                )

            # Layer 3: System prompt compaction (opt-in via
            # HEADROOM_SYSTEM_COMPACT).  Uses CCR to compress large
            # system content blocks (CLAUDE.md, rules, hooks, etc.)
            # while preserving cache_control and short instruction blocks.
            try:
                from headroom.proxy.system_compaction import (
                    compact_system_prompt,
                    system_compact_enabled,
                )
                from headroom.transforms.compression_units import find_content_router

                if _decision.should_compress and system_compact_enabled():
                    _sys_router = find_content_router(self.anthropic_pipeline)
                    if _sys_router is not None:
                        body, _sys_modified, _sys_before, _sys_after = compact_system_prompt(
                            body,
                            router=_sys_router,
                            model=model,
                            request_id=request_id,
                        )
                    if _sys_modified:
                        transforms_applied.append("anthropic:system_prompt_compaction")
                        logger.debug(
                            "[%s] system prompt compaction: %d -> %d bytes (%.0f%% saved)",
                            request_id,
                            _sys_before,
                            _sys_after,
                            (1 - _sys_after / max(_sys_before, 1)) * 100,
                        )
            except Exception as _sys_compaction_exc:
                _sys_modified = False
                logger.warning(
                    "[%s] system prompt compaction FAILED: %s", request_id, _sys_compaction_exc
                )

            presend_event = self.pipeline_extensions.emit(
                PipelineStage.PRE_SEND,
                operation="proxy.request",
                request_id=request_id,
                provider=pipeline_provider,
                model=model,
                messages=optimized_messages,
                tools=tools,
                headers=headers,
                metadata={"path": pipeline_path, "stream": stream},
            )
            previous_presend_messages = optimized_messages
            if presend_event.messages is not None:
                optimized_messages = presend_event.messages
                body["messages"] = optimized_messages
            if presend_event.tools is not None:
                tools = self._tools_for_forwarding(
                    presend_event.tools,
                    preserve_order=preserve_tool_order,
                )
                if tools or body.get("tools") is not None:
                    if tools != body.get("tools"):
                        body["tools"] = tools
            if presend_event.headers is not None:
                headers = presend_event.headers
            if presend_event.messages is not previous_presend_messages:
                optimized_tokens = tokenizer.count_messages(body["messages"])
                tokens_saved = max(0, original_tokens - optimized_tokens)

            from headroom.proxy.helpers import (
                anthropic_first_party_tool_search_supported,
                strip_first_party_tool_search_tools_for_third_party_upstream,
            )

            _anthropic_target_base_url = upstream_base_url or self.ANTHROPIC_API_URL
            _third_party_anthropic_upstream = provider_name == "anthropic" and (
                not anthropic_first_party_tool_search_supported(_anthropic_target_base_url)
            )
            if _third_party_anthropic_upstream:
                _tools_before_strip = body.get("tools")
                _tools_after_strip = strip_first_party_tool_search_tools_for_third_party_upstream(
                    _tools_before_strip,
                    _anthropic_target_base_url,
                )
                if _tools_after_strip is not _tools_before_strip:
                    body["tools"] = _tools_after_strip
                    tools = _tools_after_strip
                    tags["third_party_tool_search_stripped"] = max(
                        0,
                        len(_tools_before_strip) - len(_tools_after_strip),
                    )

            # Server-side Tool Search (on by default; HEADROOM_TOOL_SEARCH=0 opts
            # out — the `coding` savings profile already seeded it on via
            # seed_proxy_env_defaults, so default-on here just makes the same
            # posture hold for entry points that never seeded): defer the
            # non-core tool schemas behind a tool_search tool so Anthropic excludes
            # them from the context window — they stop counting as input tokens until
            # the model searches for one — while every tool stays callable.
            # Deterministic → the tools prefix still prompt-caches. Safe for opencode:
            # its @ai-sdk/anthropic parses the server_tool_use / tool_search_tool_result
            # round-trip natively, and its cache breakpoints sit on messages (never
            # tools), so defer_loading cannot collide with cache_control. We COUNT the
            # deferred tool-schema tokens as the tool-search input-token saving (those
            # bytes are excluded from context this turn); the response usage confirms it.
            #
            # FIRST-PARTY ANTHROPIC ONLY: the tool_search_tool_* type + defer_loading
            # here use the first-party Claude API shape (GA, no beta header). Custom
            # Anthropic-compatible gateways reject that shape, so third-party routes
            # strip client-originated tool_search_tool_* entries above and skip
            # Headroom's own injector here.
            if (
                provider_name == "anthropic"
                and anthropic_first_party_tool_search_supported(_anthropic_target_base_url)
                and getattr(self, "anthropic_backend", None) is None
                and os.environ.get("HEADROOM_TOOL_SEARCH", "1").strip().lower()
                in ("1", "true", "yes", "on", "auto")
            ):
                from headroom.proxy.helpers import inject_tool_search_deferral

                _ts_before = body.get("tools")
                _ts_after = inject_tool_search_deferral(_ts_before)
                if _ts_after is not _ts_before:
                    _ts_deferred = [
                        t for t in _ts_after if isinstance(t, dict) and t.get("defer_loading")
                    ]
                    try:
                        _ts_saved_tokens = tokenizer.count_text(
                            json.dumps(_ts_deferred, default=str)
                        )
                    except Exception:
                        _ts_saved_tokens = 0
                    body["tools"] = _ts_after
                    tools = _ts_after
                    tags["tool_search_deferred_tools"] = len(_ts_deferred)
                    tags["tool_search_deferred_tokens"] = _ts_saved_tokens
                    from headroom.proxy.savings_attribution import record_savings

                    record_savings(tags, "tool_search", tokens=_ts_saved_tokens)
                    transforms_applied.append(
                        f"router:tool_search_deferral:{len(_ts_deferred)}tools:"
                        f"{_ts_saved_tokens}tok"
                    )

            # Turn hooks (opt-in extensions): a registered hook may inspect or
            # rewrite the outbound tools/messages before we send upstream — the
            # extensible counterpart to the built-in deferral above. A single
            # registry check keeps this a no-op when no hook is registered.
            from headroom.proxy.turn_hooks import (
                TurnContext,
                registered_turn_hooks,
                run_request_hooks,
            )

            _pre_hook_tokens: int | None = None
            _req_ctx: TurnContext | None = None
            if registered_turn_hooks():
                _req_ctx = TurnContext(
                    provider="anthropic",
                    model=str(model),
                    messages=optimized_messages,
                    tools=body.get("tools"),
                    config=self.config,
                    tags=tags,
                    count_messages=tokenizer.count_messages,
                    count_tools=_count_tool_tokens,
                )
                # Snapshot BEFORE the hook (same tokenizer) so we can tell whether the
                # hook itself folded — comparing against the pipeline's optimized_tokens
                # instead conflates a real fold with a cross-estimator delta (see below).
                try:
                    _pre_hook_tokens = tokenizer.count_messages(optimized_messages)
                except Exception:
                    _pre_hook_tokens = None
                _th_tools_before = body.get("tools")
                _th_tok_before = _count_tool_tokens(_th_tools_before) if _th_tools_before else 0
                run_request_hooks(_req_ctx, stream_safe_only=bool(stream))
                if _req_ctx.messages is not optimized_messages:
                    optimized_messages = _req_ctx.messages
                    body["messages"] = optimized_messages
                if _req_ctx.tools is not body.get("tools"):
                    tools = _req_ctx.tools
                    body["tools"] = tools
                # A hook may shrink the tool array either by replacing it or in place,
                # so measure the FINAL tools object. Deferral-shaped (removes schemas
                # count_messages never saw), hence a tag rather than a fold — mirrors
                # the OpenAI chat path so a turn-hook extension is credited on both.
                _th_tok_after = _count_tool_tokens(_req_ctx.tools) if _req_ctx.tools else 0
                _th_saved = max(0, _th_tok_before - _th_tok_after)
                if _th_saved > 0:
                    tags["turn_hook_tools_saved_tokens"] = (
                        int(tags.get("turn_hook_tools_saved_tokens", 0) or 0) + _th_saved
                    )

            # Tool-search history repair (#2805). Once deferral is on, the client
            # stores Anthropic's server_tool_use / tool_search_tool_result blocks in
            # its transcript forever, and upstream validates every tool_reference in
            # that history against THIS request's tools array. Claude Code replays
            # the same transcript on side-requests carrying a different, smaller
            # tools array (the prompt-type Stop hook evaluator, /compact), which the
            # proxy cannot predict — so upstream 400s with "Tool reference 'X' not
            # found in available tools". Drop the blocks such a request cannot
            # support. Unconditional (not gated on the flag) so transcripts poisoned
            # before the flag was turned off still recover.
            #
            # ORDERING (#2888): this must be the LAST stage that can invalidate a
            # tool_reference, so it runs after BOTH the deferral injection above (the
            # tool we just added counts as present, so the main loop strips nothing
            # and the prefix is untouched) AND the turn hooks (a hook may rewrite the
            # tools array, and repairing before it validated against a stale view).
            # Nothing past this point mutates `body["tools"]` on the outbound path.
            from headroom.proxy.helpers import strip_unsupported_tool_search_blocks

            _ts_repaired, _ts_stripped = strip_unsupported_tool_search_blocks(
                body.get("messages"), body.get("tools")
            )
            if _ts_stripped:
                body["messages"] = _ts_repaired
                optimized_messages = _ts_repaired
                body_mutation_tracker.mark_mutated("tool_search_history_repair")
                transforms_applied.append(f"router:tool_search_repair:{_ts_stripped}blocks")
                logger.info(
                    "[%s] Tool search: dropped %d unsupportable history block(s) "
                    "(tools array cannot resolve their tool_reference entries)",
                    request_id,
                    _ts_stripped,
                )

            # CCR retrieve history repair (#2814). Run beside the tool-search
            # repair and after both normal CCR injection and turn hooks, so it
            # validates against the final outbound tools array. A side-request
            # that does not declare headroom_retrieve cannot retain historical
            # tool_use/tool_result references that Anthropic would reject.
            from headroom.proxy.helpers import strip_unsupported_ccr_retrieve_blocks

            _ccr_repaired, _ccr_neutralized = strip_unsupported_ccr_retrieve_blocks(
                body.get("messages"), body.get("tools")
            )
            if _ccr_neutralized:
                body["messages"] = _ccr_repaired
                optimized_messages = _ccr_repaired
                body_mutation_tracker.mark_mutated("ccr_retrieve_history_repair")
                transforms_applied.append(f"router:ccr_retrieve_repair:{_ccr_neutralized}blocks")
                logger.info(
                    "[%s] CCR: neutralized %d headroom_retrieve history block(s) "
                    "(tools array does not declare the tool this turn)",
                    request_id,
                    _ccr_neutralized,
                )

            # Consistency: report tok_before/tok_after with ONE tokenizer. The pipeline
            # and the handler use different token estimators, and cache-mode branches
            # can leave original_tokens (handler, line ~1049) and optimized_tokens
            # (pipeline, result.tokens_after) on different scales — which produced
            # impossible tok_after>tok_before deltas and masked real savings. Recount
            # BOTH endpoints (pre-compression snapshot vs final outbound messages) with
            # the handler tokenizer so the delta is meaningful. This also captures any
            # turn-hook fold (optimized_messages is post-hook). Runs unconditionally.
            try:
                _orig_snapshot = original_client_messages  # noqa: F821 (bound at request start)
                # Off the event loop (#2810): both passes are CPU-bound real BPE
                # since #2543, and running them inline stalled every other
                # in-flight request on the same process (~1s on a 2.3 MB body).
                # Same tokenizer instance, so the reported values are unchanged.
                original_tokens = await asyncio.to_thread(tokenizer.count_messages, _orig_snapshot)
                optimized_tokens = await asyncio.to_thread(
                    tokenizer.count_messages, optimized_messages
                )
                # Fold the tool-schema/desc compaction delta into BOTH endpoints so
                # tok_before - tok_after == tok_saved stays coherent in the PERF line
                # (count_messages never sees tool bytes). Same shape as the OpenAI chat
                # handler; guarded so a no-op or inflating pass contributes nothing.
                if 0 < _tool_tokens_after < _tool_tokens_before:
                    original_tokens += _tool_tokens_before
                    optimized_tokens += _tool_tokens_after
                tokens_saved = max(0, original_tokens - optimized_tokens)
                # Attribute the fold to the hook ONLY when the hook itself reduced
                # tokens (same-tokenizer pre vs post) — not when the recount above
                # merely normalized a cross-estimator scale difference.
                if _pre_hook_tokens is not None and optimized_tokens < _pre_hook_tokens:
                    transforms_applied.append("turn_hook")
            except Exception:
                logger.debug("consistency token re-count skipped", exc_info=True)

            # Output shaping (opt-in via HEADROOM_OUTPUT_SHAPER): verbosity
            # steering appended to the system-prompt tail + effort routing on
            # mechanical tool_result continuations. Runs after every other
            # body mutation so the turn classifier sees the final messages,
            # and respects the same bypass header as compression.
            if not _bypass:
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
                    shape_request,
                )

                _shaper_settings = OutputShaperSettings.from_env(
                    enabled=(shaper_enabled_for(getattr(self, "config", None))),
                    steering_enabled=steering_allowed_for(getattr(self, "config", None)),
                )
                if _shaper_settings.enabled:
                    # Conversation-stable holdout assignment: a whole
                    # conversation is treatment or control. This keeps the A/B
                    # comparison clean AND keeps the prefix cache stable (we
                    # never flip a conversation's system-prompt tail mid-stream).
                    from headroom.proxy import runtime_env

                    _holdout = 0.0
                    try:
                        _holdout = float(runtime_env.getenv("HEADROOM_OUTPUT_HOLDOUT", "0") or "0")
                    except ValueError:
                        _holdout = 0.0
                    _arm = assign_arm(conversation_key_from_body(body), _holdout)

                    # Stratum from request features observable now (mirrors the
                    # offline baseline so live and learned strata line up).
                    _turn_kind = classify_turn(body.get("messages", [])).value
                    _stratum = stratum_key(
                        turn_kind=_turn_kind,
                        input_tokens=original_tokens,
                        model=model,
                        has_tools=bool(body.get("tools")),
                    )
                    # Carry (arm, stratum) on the existing label channel so the
                    # outcome funnel can feed the savings ledger from any path.
                    transforms_applied.append(stratum_label(_arm, _stratum))

                    if _arm == "treatment":
                        _level, _src = resolve_verbosity_level(_shaper_settings)
                        shape_result = shape_request(body, _shaper_settings, level_override=_level)
                        if shape_result.changed:
                            body_mutation_tracker.mark_mutated("output_shaper")
                            transforms_applied.extend(shape_result.labels or [])
                            logger.info(
                                f"[{request_id}] OutputShaper(L{_level}/{_src}): "
                                f"{shape_result.labels}"
                            )

            # Params stage: the last chance to change what the model WRITES.
            # Emitted here, after every message mutation and after the built-in
            # shaper, so an extension sees the request exactly as it will go out
            # and a turn classifier reading ``messages`` sees the final list.
            #
            # Unit 2: mark end of pre-upstream phase. Everything after this
            # point is upstream I/O or post-response bookkeeping.
            stage_timer.record(
                "total_pre_upstream",
                (time.perf_counter() - pre_upstream_started_at) * 1000.0,
            )

            # Anthropic wire-contract guard (issue #765). Any transform or
            # pipeline extension above may have left a ``role="system"`` entry
            # in ``messages`` (e.g. a harness system block relocated during
            # compression). Anthropic rejects that with a 400 ("messages.0: use
            # the top-level 'system' parameter ..."), so relocate it back to the
            # top-level ``system`` parameter as the last step before forwarding.
            relocated_messages, relocated_system, system_relocated = (
                relocate_system_messages_to_top_level(
                    body["messages"],
                    body.get("system"),
                    (
                        str(model)
                        if (
                            not upstream_base_url
                            or getattr(self, "anthropic_backend", None) is not None
                            or _is_googleapis_endpoint(upstream_base_url)
                        )
                        else None
                    ),
                )
            )
            if system_relocated:
                body["messages"] = relocated_messages
                if relocated_system is None:
                    body.pop("system", None)
                else:
                    body["system"] = relocated_system
                body_mutation_tracker.mark_mutated("system_role_relocated")
                logger.warning(
                    "[%s] Relocated role=system message(s) out of messages[] into the "
                    "top-level system parameter (Anthropic wire-contract guard, issue #765)",
                    request_id,
                )

            # Byte-faithful forwarder support (PR-A3, fixes P0-2). At this
            # point body has been through every transform (image, compression,
            # memory, tool sort, pipeline extensions). If a transform reported
            # it touched the body, mark mutated; we additionally compare the
            # final body against the parsed original bytes as a structural
            # safety net so any silent mutation we missed still triggers
            # canonical re-serialization.
            if not body_mutation_tracker.mutated and original_body_bytes is not None:
                try:
                    parsed_original = json.loads(original_body_bytes)
                    if parsed_original != body:
                        body_mutation_tracker.mark_mutated("structural_diff_vs_original")
                # MemoryError is not a ValueError, so a re-parse spike on 1M-context
                # bodies used to escape and abort an otherwise-fine request (#2768).
                # This block is a safety net; marking mutated is already the safe
                # outcome (it forces canonical re-serialization).
                except (json.JSONDecodeError, ValueError, MemoryError, RecursionError):
                    body_mutation_tracker.mark_mutated("original_unparseable")

            if (
                (upstream_base_url or self.ANTHROPIC_API_URL != "https://api.anthropic.com")
                and stream
                and _client_beta_value
                and _sticky_beta_value
                and _sticky_beta_value != _client_beta_value
                and not body_mutation_tracker.mutated
                and not _headroom_beta_added
            ):
                headers["anthropic-beta"] = _client_beta_value

            # Forward request - use Bedrock backend if configured, otherwise direct API
            #
            # An extension may have published a per-request routing decision on
            # `request.state.headroom_route` (see proxy/route_advice.py). When
            # it names a provider we do not already speak, serve this ONE
            # request from a translating backend for it. Absent, unresolvable,
            # or same-protocol -> `self.anthropic_backend`, i.e. exactly the
            # behaviour this line had before.
            from headroom.proxy.route_advice import resolver_for

            request_backend = resolver_for(self).for_request(request, body=body)
            if request_backend is not None:
                # Route through Bedrock backend
                try:
                    if stream:
                        self.pipeline_extensions.emit(
                            PipelineStage.POST_SEND,
                            operation="proxy.request",
                            request_id=request_id,
                            provider=pipeline_provider,
                            model=model,
                            messages=body["messages"],
                            tools=tools,
                            metadata={"path": pipeline_path, "stream": True},
                        )
                        await _finalize_pre_upstream()
                        return await self._stream_response_bedrock(
                            body,
                            headers,
                            "anthropic",
                            model,
                            request_id,
                            original_tokens,
                            optimized_tokens,
                            tokens_saved,
                            transforms_applied,
                            tags,
                            optimization_latency,
                            pipeline_timing=pipeline_timing,
                            original_messages=original_client_messages,
                            prefix_tracker=prefix_tracker,
                            optimized_messages=optimized_messages,
                            backend=request_backend,
                        )
                    else:
                        async with stage_timer.measure("upstream_connect"):
                            backend_response = await request_backend.send_message(body, headers)
                        self.pipeline_extensions.emit(
                            PipelineStage.POST_SEND,
                            operation="proxy.request",
                            request_id=request_id,
                            provider=pipeline_provider,
                            model=model,
                            messages=body["messages"],
                            tools=tools,
                            response=backend_response.body,
                            metadata={
                                "path": pipeline_path,
                                "stream": False,
                                "status_code": backend_response.status_code,
                            },
                        )
                        self.pipeline_extensions.emit(
                            PipelineStage.RESPONSE_RECEIVED,
                            operation="proxy.request",
                            request_id=request_id,
                            provider=pipeline_provider,
                            model=model,
                            response=backend_response.body,
                            metadata={
                                "path": pipeline_path,
                                "stream": False,
                                "status_code": backend_response.status_code,
                            },
                        )
                        # Non-stream: first-byte and connect are effectively
                        # the same horizon — ``send_message`` awaits until
                        # the response body is fully buffered.
                        if (
                            "upstream_first_byte" not in stage_timer
                            and "upstream_connect" in stage_timer
                        ):
                            stage_timer.record(
                                "upstream_first_byte",
                                stage_timer.summary()["upstream_connect"],
                            )
                        await _finalize_pre_upstream()
                        if backend_response.error:
                            return JSONResponse(
                                status_code=backend_response.status_code,
                                content=annotate_backend_error_body(
                                    backend_response.body,
                                    backend_response.status_code,
                                    logger=logger,
                                    request_id=request_id,
                                ),
                            )

                        # Track metrics
                        total_latency = (time.time() - start_time) * 1000
                        usage = backend_response.body.get("usage", {})
                        # A backend may report these counters as JSON null (key
                        # present, value null), for which ``.get(key, 0)`` returns
                        # ``None`` rather than the default. Coerce with ``or 0`` so
                        # the arithmetic below (and ``RequestOutcome``) never sees
                        # ``None`` — matching the direct-Anthropic path.
                        output_tokens = int(usage.get("output_tokens", 0) or 0)
                        _thinking = _thinking_tokens_for(backend_response.body)
                        _stop_reason = (backend_response.body or {}).get("stop_reason")

                        _backend_name = request_backend.name if request_backend else "anthropic"
                        # Eligible-only denominator for the active
                        # compression ratio: tokens in the live zone we
                        # actually attempted to compress. Frozen prefix
                        # (system + prior cached turns) is byte-identical
                        # pre/post — counting it would dilute the metric
                        # with content we deliberately don't touch for
                        # prefix-cache safety. Fall back to the full
                        # pre-comp request if the live-zone count fails
                        # so the aggregate denominator stays coherent.
                        try:
                            attempted_input_tokens = tokenizer.count_messages(
                                original_client_messages[frozen_message_count:]
                            )
                        except Exception:
                            attempted_input_tokens = original_tokens

                        cr_tokens = int(usage.get("cache_read_input_tokens", 0) or 0)
                        cw_tokens = int(usage.get("cache_creation_input_tokens", 0) or 0)
                        cw_5m_tokens, cw_1h_tokens = self._extract_anthropic_cache_ttl_metrics(
                            usage
                        )
                        # Prefer the backend's Anthropic-shaped ``input_tokens``,
                        # which is already the uncached count (prompt tokens minus
                        # cache read/creation; see ``_anthropic_usage_from_litellm``
                        # / #1345), matching the direct-API path below. Re-deriving
                        # it as ``attempted_input_tokens - cache`` is wrong whenever
                        # the backend reports it: ``attempted_input_tokens`` is the
                        # live-zone tokenizer count kept for the compression-ratio
                        # denominator, not the full request size, so on any turn
                        # whose cached prefix is larger than the new turn it
                        # underflows and ``max(0, ...)`` clamps uncached to 0.
                        #
                        # Guard on the backend actually reporting it: a backend that
                        # omits ``input_tokens`` (or sends null) must not silently
                        # report uncached=0 — fall back to the live-zone derivation,
                        # which is never worse than the previous behaviour.
                        _reported_input_tokens = usage.get("input_tokens")
                        if _reported_input_tokens is not None:
                            uncached_input_tokens = int(_reported_input_tokens)
                        else:
                            uncached_input_tokens = max(
                                0, attempted_input_tokens - cr_tokens - cw_tokens
                            )

                        # Update prefix cache tracker for next turn. Mirrors the
                        # direct-Anthropic-API branch below (~line 3011) — without
                        # this, PrefixCacheTracker never sees a turn 2+ update on
                        # the Bedrock path, extract_cache_stable_delta() always
                        # returns None (no previous_original_messages), and cache
                        # mode falls back to full unmodified passthrough every
                        # turn instead of compressing the append-only delta.
                        next_original_messages = copy.deepcopy(original_client_messages)
                        next_forwarded_messages = copy.deepcopy(optimized_messages)
                        assistant_message = self._assistant_message_from_response_json(
                            backend_response.body
                        )
                        if assistant_message is not None:
                            next_original_messages.append(copy.deepcopy(assistant_message))
                            next_forwarded_messages.append(copy.deepcopy(assistant_message))
                        if hasattr(prefix_tracker, "classify_cache_miss"):
                            miss = prefix_tracker.classify_cache_miss(
                                cache_read_tokens=cr_tokens,
                                current_forwarded_messages=optimized_messages,
                            )
                            from headroom.cache.ttl_observations import (
                                record_cache_observation,
                            )

                            record_cache_observation(
                                provider="anthropic", model=model, attribution=miss
                            )
                            if miss.is_miss:
                                logger.info(
                                    f"[{request_id}] CACHE-MISS-ATTRIBUTION: reason={miss.reason} "
                                    f"idle={miss.idle_seconds:.0f}s ttl={miss.cache_ttl_seconds}s "
                                    f"expected_cached={miss.expected_cached_tokens:,} "
                                    f"prefix_changed={miss.prefix_changed} "
                                    f"ttl_exceeded={miss.ttl_exceeded}"
                                )
                                await self.metrics.record_cache_miss_attribution(
                                    provider_name, miss.reason
                                )
                        prefix_tracker.update_from_response(
                            cache_read_tokens=cr_tokens,
                            cache_write_tokens=cw_tokens,
                            messages=next_forwarded_messages,
                            original_messages=next_original_messages,
                        )

                        await self._record_request_outcome(
                            RequestOutcome(
                                request_id=request_id,
                                provider=_backend_name,
                                model=model,
                                original_tokens=original_tokens,
                                optimized_tokens=optimized_tokens,
                                output_tokens=output_tokens,
                                tokens_saved=tokens_saved,
                                attempted_input_tokens=attempted_input_tokens,
                                provider_input_tokens=(
                                    uncached_input_tokens + cr_tokens + cw_tokens
                                ),
                                cache_read_tokens=cr_tokens,
                                cache_write_tokens=cw_tokens,
                                cache_write_5m_tokens=cw_5m_tokens,
                                cache_write_1h_tokens=cw_1h_tokens,
                                uncached_input_tokens=uncached_input_tokens,
                                thinking_tokens=_thinking.tokens,
                                thinking_inferred=_thinking.inferred,
                                stop_reason=_stop_reason,
                                total_latency_ms=total_latency,
                                overhead_ms=optimization_latency,
                                pipeline_timing=pipeline_timing,
                                transforms_applied=tuple(transforms_applied),
                                num_messages=len(body.get("messages", [])),
                                tags=tags,
                                client=client,
                                turn_id=compute_turn_id(
                                    model, body.get("system"), body.get("messages")
                                ),
                                # `original_client_messages` is the deep-copied
                                # pre-compression snapshot; `body["messages"]`
                                # is the compressed list sent upstream. Both
                                # share the `log_full_messages` gate so the two
                                # sides stay symmetric.
                                request_messages=original_client_messages
                                if self.config.log_full_messages
                                else None,
                                compressed_messages=body.get("messages")
                                if self.config.log_full_messages
                                else None,
                            )
                        )

                        return JSONResponse(
                            status_code=backend_response.status_code,
                            content=annotate_backend_error_body(
                                backend_response.body,
                                backend_response.status_code,
                                logger=logger,
                                request_id=request_id,
                            ),
                        )
                except Exception as e:
                    logger.error(f"[{request_id}] Bedrock backend error: {e}")
                    # Unit 4: release the pre-upstream semaphore on error.
                    await _finalize_pre_upstream()
                    # A backend that never initialized (missing optional SDK, no
                    # ADC) otherwise surfaces as an opaque provider string that
                    # tells the user nothing about the fix.
                    message = str(e)
                    hint = backend_error_hint(message)
                    if hint:
                        logger.error(f"[{request_id}] backend setup hint: {hint}")
                        message = f"{message}\n[headroom] hint: {hint}"
                    return JSONResponse(
                        status_code=500,
                        content={
                            "type": "error",
                            "error": {"type": "api_error", "message": message},
                        },
                    )

            # Direct Anthropic API, or a provider-compatible Anthropic
            # Messages endpoint such as Vertex AI publisher rawPredict.
            # Both arms build through `build_copilot_upstream_url` because that
            # is the only place `mark_request_routed_to_copilot` fires, and the
            # outcome funnel relabels `provider` to "copilot" off that flag (see
            # proxy/outcome.py). The resolved target reaches Copilot without any
            # per-request `upstream_base_url` — `wrap vscode` points the
            # Anthropic target at the Copilot host — so this else arm carries the
            # Claude-on-Copilot traffic, and building it by f-string attributed
            # every one of those turns to "anthropic" on the dashboard.
            # For a non-Copilot base the builder only joins base + path, so the
            # URL itself is unchanged.
            url = (
                build_copilot_upstream_url(upstream_base_url, request.url.path)
                if upstream_base_url
                else build_copilot_upstream_url(self.ANTHROPIC_API_URL, "/v1/messages")
            )
            if upstream_base_url and request.url.query:
                url = f"{url}?{request.url.query}"

            try:
                ccr_handler_config = getattr(self.ccr_response_handler, "config", None)
                ccr_response_handler_enabled = bool(
                    self.ccr_response_handler and getattr(ccr_handler_config, "enabled", True)
                )
                # A body carrying signed thinking blocks leaves as the client's
                # original bytes (see ``select_outbound_body``), which throws
                # away every edit made here — including the ``stream`` flip
                # below. Taking the buffered path anyway asks upstream for a
                # stream:true reply and then tries to read it as buffered JSON:
                # the parse fails, SSE resynthesis is skipped, and the client
                # gets a 200 with no usable body (#2952). The retrieve tool is
                # itself an injected (and equally discarded) mutation on these
                # turns, so the plain streaming path is the coherent choice.
                from headroom.proxy.body_forwarding import outbound_body_is_client_bytes

                outbound_locked_to_client_bytes = outbound_body_is_client_bytes(
                    body=body,
                    original_body_bytes=original_body_bytes,
                )
                # ``headroom_retrieve`` stays resident for the session lifetime so
                # the tools array is byte-stable and the prompt cache survives, so
                # its mere presence is a poor reason to buffer. Once a session went
                # sticky, *every* later streaming turn took the buffered path, and
                # buffering replaces incremental delivery with one write at the
                # end: time-to-last-byte is roughly unchanged, but time-to-first-
                # byte becomes the entire generation. In the traffic reported in
                # #3071 that was 8s on average and up to 100s, on 234 requests in
                # a single day.
                #
                # The tool can only expand a marker that is in the outgoing body
                # and redeemable now, so a turn carrying none cannot benefit from
                # server-side retrieval and should keep streaming. This reads
                # ``body`` rather than the earlier scan of ``optimized_messages``
                # because memory hooks, pre-send extensions and the CCR/tool-search
                # repairs can all replace the message list after that scan; the
                # only list that matters is the one about to go on the wire.
                retrieve_tool_is_offered = (
                    stream
                    and ccr_response_handler_enabled
                    and self._has_headroom_retrieve_tool(
                        tools if tools is not None else body.get("tools")
                    )
                )
                buffered_retrieval_can_help = (
                    retrieve_tool_is_offered and self._outgoing_body_has_redeemable_marker(body)
                )
                wants_buffered_stream_ccr = bool(buffered_retrieval_can_help)
                if retrieve_tool_is_offered and not buffered_retrieval_can_help:
                    logger.info(
                        f"[{request_id}] CCR: headroom_retrieve is resident but this "
                        "request carries no redeemable marker, so server-side "
                        "retrieval cannot fire; keeping the streaming path (#3071)"
                    )
                buffered_stream_ccr = (
                    wants_buffered_stream_ccr and not outbound_locked_to_client_bytes
                )
                if wants_buffered_stream_ccr and outbound_locked_to_client_bytes:
                    logger.info(
                        f"[{request_id}] CCR: signed thinking blocks force byte-faithful "
                        "passthrough, so a stream:false flip could not reach upstream; "
                        "using the plain streaming path instead of buffered retrieval"
                    )
                if buffered_stream_ccr:
                    if body.get("stream") is not False:
                        body["stream"] = False
                        body_mutation_tracker.mark_mutated(
                            "ccr_streaming_retrieve_buffered_non_stream"
                        )
                    # The ``Accept`` rewrite this flip used to do lives at the
                    # buffered boundary below, which every non-streaming
                    # request reaches — this one and the client's own (#3130).
                    logger.info(
                        f"[{request_id}] CCR: stream:true request has "
                        "headroom_retrieve available; using buffered stream:false "
                        "upstream request for server-side retrieval handling"
                    )

                # Last stop before the wire. Every transform, the tool sort, the
                # tool-search deferral, CCR injection and the pipeline
                # extensions have run, so this is the only place that can see
                # the cache_control markers Anthropic will actually evaluate --
                # and the ordering rule spans tools/system/messages, which no
                # single transform is in a position to check (#2939).
                from headroom.proxy.helpers import (
                    enforce_cache_control_ttl_order,
                    log_cache_breakpoints,
                )

                (
                    _ttl_system,
                    _ttl_messages,
                    _ttl_tools,
                    _ttl_stats,
                ) = enforce_cache_control_ttl_order(
                    body.get("system"),
                    body.get("messages"),
                    body.get("tools"),
                    client_uses_1h=client_uses_1h,
                    request_id=request_id,
                )
                if _ttl_stats["violation"]:
                    if body.get("system") is not None:
                        body["system"] = _ttl_system
                    body["messages"] = _ttl_messages
                    if body.get("tools") is not None:
                        body["tools"] = _ttl_tools
                    tools = _ttl_tools
                    body_mutation_tracker.mark_mutated("cache_control_ttl_order")

                # Signed thinking locks the request to the client's original
                # bytes. Once all mutation sites have run, make every downstream
                # observer use that same wire body and neutralize savings from
                # edits that will not be sent (#2990). This covers PERF, /stats,
                # durable savings, response headers, pipeline events, and the
                # prefix tracker rather than fixing only one reporting surface.
                #
                # RECOMPUTE rather than reusing the probe from before the CCR
                # branch. That probe answers "will my stream flip survive?" and
                # has to be asked early; this asks "what are we actually about to
                # bill for?" and must be asked last. The two diverge now that the
                # predicate tests thinking-block CONTENT and not merely presence:
                # `enforce_cache_control_ttl_order` immediately above rewrites
                # `body["messages"]`, so a marker moved onto or off a message
                # holding a thinking block changes the answer after the early
                # probe was taken. Evaluating the same predicate on the same
                # final `body` that `select_outbound_body` will see is what keeps
                # the accounting and the wire in agreement.
                final_locked_to_client_bytes = outbound_body_is_client_bytes(
                    body=body,
                    original_body_bytes=original_body_bytes,
                )
                if final_locked_to_client_bytes and body_mutation_tracker.mutated:
                    discarded_reasons = body_mutation_tracker.reasons
                    try:
                        wire_body = json.loads(original_body_bytes or b"")
                    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, RecursionError):
                        wire_body = None
                    if not isinstance(wire_body, dict):
                        raise ValueError(
                            "signed-thinking passthrough could not reconstruct its client wire body"
                        )

                    from headroom.proxy.savings_attribution import SAVINGS_ATTRIBUTION_TAG
                    from headroom.proxy.tool_schema_savings_policy import (
                        TOOL_SCHEMA_SAVINGS_TAGS,
                    )

                    attribution = tags.get(SAVINGS_ATTRIBUTION_TAG)
                    if isinstance(attribution, list):
                        attribution.clear()
                    for savings_tag in TOOL_SCHEMA_SAVINGS_TAGS:
                        tags.pop(savings_tag, None)
                    tags.pop("tool_search_deferred_tools", None)
                    tags["wire_mutations_discarded"] = len(discarded_reasons)
                    tags["wire_mutation_reasons"] = ",".join(discarded_reasons)

                    body = wire_body
                    optimized_messages = body.get("messages", [])
                    tools = body.get("tools")
                    optimized_tokens = original_tokens
                    tokens_saved = 0
                    transforms_applied = []

                log_cache_breakpoints(
                    request_id=request_id,
                    inbound=inbound_breakpoints,
                    outbound=count_cache_breakpoints(
                        body.get("system"), body.get("messages"), body.get("tools")
                    ),
                )

                if stream and not buffered_stream_ccr:
                    self.pipeline_extensions.emit(
                        PipelineStage.POST_SEND,
                        operation="proxy.request",
                        request_id=request_id,
                        provider=pipeline_provider,
                        model=model,
                        messages=body["messages"],
                        tools=tools,
                        metadata={"path": pipeline_path, "stream": True},
                    )
                    await _finalize_pre_upstream()
                    explicit_session_header = request.headers.get("x-headroom-session-id")
                    session_key = self._get_session_key(
                        body,
                        session_header=explicit_session_header,
                    )
                    # Coalesce mid-turn messages only for Claude Code, the sole
                    # client that understands the 202 `headroom_queued` reply and
                    # the `headroom_pending_messages` SSE event. Other harnesses
                    # (e.g. OpenCode subagents sharing a body-derived session key)
                    # would otherwise have their request swallowed and never
                    # answered. (#1608) `_should_queue_mid_turn` further restricts
                    # this to opt-in (header-bearing) callers with an active
                    # stream, so the coarse md5 fallback can't 202 a streaming
                    # caller.
                    if supports_mid_turn_coalescing(
                        classify_client(request.headers)
                    ) and self._should_queue_mid_turn(session_key, explicit_session_header):
                        from fastapi.responses import JSONResponse

                        queued = self._queue_mid_turn_message(session_key, body)
                        return JSONResponse(content=queued, status_code=202)
                    return await self._stream_response(
                        url,
                        headers,
                        body,
                        "anthropic",
                        model,
                        request_id,
                        original_tokens,
                        optimized_tokens,
                        tokens_saved,
                        transforms_applied,
                        tags,
                        optimization_latency,
                        memory_user_id=memory_user_id,
                        pipeline_timing=pipeline_timing,
                        prefix_tracker=prefix_tracker,
                        original_messages=original_client_messages,
                        original_body_bytes=original_body_bytes,
                        body_mutated=body_mutation_tracker.mutated,
                        mutation_reasons=body_mutation_tracker.reasons,
                        memory_request_ctx=memory_request_ctx,
                        outcome_provider=provider_name,
                        session_key=session_key,
                    )
                else:
                    # Whatever set it — the client's own ``stream: false`` or
                    # the CCR flip above — this branch sends a non-streaming
                    # request, so the client's ``Accept: text/event-stream`` no
                    # longer describes what is being asked for. Forwarding it
                    # unchanged puts a self-contradicting request on the wire:
                    # "answer as JSON" in the body, "I only accept SSE" in the
                    # headers.
                    #
                    # Anthropic tolerates the contradiction; stricter
                    # Anthropic-compatible gateways do not — GitHub Copilot's
                    # answers a generic ``api_error`` (#3078). On a
                    # client-originated non-stream turn — Claude Code's retry
                    # after a failed stream — an SSE answer to a JSON request
                    # is the empty/malformed HTTP 200 of #3130.
                    #
                    # Mutated in place: ``headers`` is captured by the
                    # closures defined below, and rebinding it here would
                    # leave them holding the old mapping.
                    if body.get("stream", False) is False:
                        for _accept_key in [k for k in headers if k.lower() == "accept"]:
                            headers.pop(_accept_key, None)
                        headers["accept"] = "application/json"

                    # Copilot auth is applied per-URL, and until now only the
                    # streaming forwarder did it (``_stream_response``). This
                    # buffered arm sends via ``_retry_request``, which forwards
                    # headers untouched — so a Claude turn routed to Copilot
                    # arrived with whatever the client happened to send and none
                    # of Headroom's own credential handling: no minted or
                    # refreshed token (the one ``wrap vscode`` hands the proxy),
                    # no ``Copilot-Integration-Id`` default. A client token that
                    # went stale mid-session therefore 401'd here while the
                    # streaming path recovered.
                    #
                    # A non-Copilot URL returns the headers unchanged, so this is
                    # a no-op everywhere else.
                    #
                    # Mutated in place for the same reason as the accept header
                    # above: the closures below capture ``headers``, and the CCR
                    # continuation rebuilds its own header set from it.
                    _copilot_authed_headers = await apply_copilot_api_auth(dict(headers), url=url)
                    headers.clear()
                    headers.update(_copilot_authed_headers)

                    # Populated once the upstream answers 200 with parseable
                    # JSON, so the guard below can fall back to it (#3088).
                    _salvageable_upstream: dict[str, Any] = {}

                    async def _buffered_ccr_operation():
                        async with stage_timer.measure("upstream_connect"):
                            response = await self._retry_request(
                                "POST",
                                url,
                                headers,
                                body,
                                original_body_bytes=original_body_bytes,
                                body_mutated=body_mutation_tracker.mutated,
                                mutation_reasons=body_mutation_tracker.reasons,
                                request_id=request_id,
                                forwarder_name="anthropic_messages",
                                path_for_log="/v1/messages",
                                timeout=self._anthropic_buffered_request_timeout(),
                            )
                        self.pipeline_extensions.emit(
                            PipelineStage.POST_SEND,
                            operation="proxy.request",
                            request_id=request_id,
                            provider=pipeline_provider,
                            model=model,
                            messages=body["messages"],
                            tools=tools,
                            response=response,
                            metadata={
                                "path": pipeline_path,
                                "stream": False,
                                "client_stream": buffered_stream_ccr,
                                "ccr_stream_buffered": buffered_stream_ccr,
                                "status_code": response.status_code,
                            },
                        )
                        self.pipeline_extensions.emit(
                            PipelineStage.RESPONSE_RECEIVED,
                            operation="proxy.request",
                            request_id=request_id,
                            provider=pipeline_provider,
                            model=model,
                            response=response,
                            metadata={
                                "path": pipeline_path,
                                "stream": False,
                                "client_stream": buffered_stream_ccr,
                                "ccr_stream_buffered": buffered_stream_ccr,
                                "status_code": response.status_code,
                            },
                        )
                        if (
                            "upstream_first_byte" not in stage_timer
                            and "upstream_connect" in stage_timer
                        ):
                            stage_timer.record(
                                "upstream_first_byte",
                                stage_timer.summary()["upstream_connect"],
                            )
                        await _finalize_pre_upstream()
                        # Full diagnostic dump on upstream errors.
                        # Writes pre/post compression messages, tools, and error
                        # to ~/.headroom/logs/debug_400/ for offline analysis.
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

                            # Diagnostic dump of the full upstream-error request.
                            # OFF by default: it can contain cleartext prompt / tool /
                            # system content. Opt in with HEADROOM_DEBUG_DUMP=1
                            # (redacted: structure + lengths only) or =full (content).
                            # Never written in stateless mode.
                            dump_mode = _debug_dump_mode(self.config)
                            if dump_mode != "off":
                                try:
                                    from headroom import paths as _hr_paths

                                    debug_dir = _hr_paths.debug_400_dir()
                                    debug_dir.mkdir(parents=True, exist_ok=True)
                                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    debug_file = debug_dir / f"{ts}_{request_id}.json"

                                    # Sanitize headers (redact API keys)
                                    safe_headers = {}
                                    _sensitive_header_names = {"x-api-key", "authorization"} | {
                                        k.lower()
                                        for k in (self.config.anthropic_extra_headers or {})
                                    }
                                    for k, v in headers.items():
                                        if k.lower() in _sensitive_header_names:
                                            safe_headers[k] = v[:12] + "..." if v else ""
                                        else:
                                            safe_headers[k] = v

                                    # In redacted mode, elide prompt/tool/system
                                    # content but keep structure, roles, and lengths.
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

                                    logger.warning(
                                        f"[{request_id}] Debug dump ({dump_mode}): {debug_file}"
                                    )
                                except Exception as dump_err:
                                    logger.error(
                                        f"[{request_id}] Failed to write debug dump: {dump_err}"
                                    )

                        # A non-streaming request answered with an event
                        # stream (#3130). The turn is complete and already
                        # paid for — it is just wearing the wrong wire
                        # format — so adapt it to the JSON this caller asked
                        # for *here*, ahead of everything that reads the
                        # body: CCR retrieval, memory, turn hooks, usage and
                        # cost accounting, prefix tracking, the response
                        # cache, marker resolution and the security scan.
                        # Adapting at the final return instead would leave
                        # every one of those looking at an unparseable body.
                        #
                        # ``stream`` is what the *client* asked for, not what
                        # went upstream: a buffered CCR turn deliberately
                        # requests JSON on behalf of a streaming client and
                        # re-emits SSE further down, and must keep doing so.
                        if should_recover_sse_reply(
                            client_requested_stream=bool(stream),
                            status_code=response.status_code,
                            content_type=response.headers.get("content-type"),
                            body_is_event_stream=_looks_like_sse_response(response),
                        ):
                            response = self._adapt_event_stream_to_json(response, request_id)

                        # Parse response for CCR handling
                        resp_json = None
                        try:
                            resp_json = response.json()
                            if buffered_stream_ccr and response.status_code == 200 and resp_json:
                                # Remember the upstream's own answer before any
                                # post-processing touches it. Everything from here
                                # to the SSE resynthesis — retrieval, memory tool
                                # calls, turn hooks, usage accounting, caching — is
                                # work layered on top of a turn the provider has
                                # already produced and billed. If any of it raises
                                # unexpectedly, this is what the client should get
                                # instead of a synthesized error (#3088).
                                _salvageable_upstream["resp_json"] = resp_json
                        except (json.JSONDecodeError, ValueError) as e:
                            # DEBUG is right for the buffered non-stream path, where
                            # an unparseable body is just "no CCR handling". On the
                            # buffered-stream path it means the reply came back in a
                            # wire format we did not ask for, and every downstream
                            # step (retrieval, SSE resynthesis, usage accounting)
                            # silently no-ops — that has to be visible (#2952).
                            if buffered_stream_ccr:
                                logger.warning(
                                    f"[{request_id}] CCR: buffered stream:false request got a "
                                    f"non-JSON {response.status_code} reply "
                                    f"(content-type={response.headers.get('content-type')!r}): {e}"
                                )
                            else:
                                logger.debug(
                                    f"[{request_id}] Failed to parse response JSON for CCR "
                                    f"handling: {e}"
                                )

                        # CCR Response Handling: Handle headroom_retrieve tool calls automatically
                        if (
                            self.ccr_response_handler
                            and resp_json
                            and response.status_code == 200
                            and self.ccr_response_handler.has_ccr_tool_calls(resp_json, "anthropic")
                        ):
                            logger.info(
                                f"[{request_id}] CCR: Detected retrieval tool call, handling..."
                            )

                            # Create API call function for continuation
                            # Use a fresh client to avoid potential decompression state issues
                            async def api_call_fn(
                                msgs: list[dict], tls: list[dict] | None
                            ) -> dict[str, Any]:
                                continuation_body = {
                                    **body,
                                    "messages": msgs,
                                }
                                if tls is not None:
                                    continuation_body["tools"] = tls

                                # Use clean headers for continuation
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

                                # Reuse main client for CCR continuations (connection pooling)
                                logger.info(
                                    f"CCR: Making continuation request with {len(msgs)} messages"
                                )
                                assert self.http_client is not None, "HTTP client not initialized"
                                # Byte-faithful (PR-A3, fixes P0-2). The CCR
                                # continuation body is synthesized by Headroom
                                # so it is treated as mutated and goes through
                                # the canonical serializer.
                                from headroom.proxy.body_forwarding import (
                                    prepare_outbound_body_bytes,
                                )
                                from headroom.proxy.helpers import log_outbound_request

                                ccr_outbound_bytes, ccr_outbound_source = (
                                    prepare_outbound_body_bytes(
                                        body=continuation_body,
                                        original_body_bytes=None,
                                        body_mutated=True,
                                    )
                                )
                                # A continuation is a non-streaming call, so it
                                # needs a matching Accept for the same reason the
                                # buffered flip above does (#3078).
                                ccr_outbound_headers = {
                                    k: v
                                    for k, v in continuation_headers.items()
                                    if k.lower() not in ("accept", "content-type")
                                }
                                ccr_outbound_headers["content-type"] = "application/json"
                                ccr_outbound_headers["accept"] = "application/json"
                                log_outbound_request(
                                    forwarder="anthropic_ccr_continuation",
                                    method="POST",
                                    path=url,
                                    body_bytes_count=len(ccr_outbound_bytes),
                                    body_mutated=True,
                                    mutation_reasons=["ccr_continuation"],
                                    request_id=request_id,
                                    source=ccr_outbound_source,
                                )
                                try:
                                    cont_response = await self.http_client.post(
                                        url,
                                        content=ccr_outbound_bytes,
                                        headers=ccr_outbound_headers,
                                        timeout=self._anthropic_buffered_request_timeout(),
                                    )
                                    logger.info(
                                        f"CCR: Got response status={cont_response.status_code}, "
                                        f"content-encoding={cont_response.headers.get('content-encoding')}"
                                    )
                                    result: dict[str, Any] = cont_response.json()
                                    logger.info("CCR: Parsed JSON successfully")
                                    return result
                                except Exception as e:
                                    resp_headers: str | dict[str, str] = "N/A"
                                    try:
                                        resp_headers = dict(cont_response.headers)
                                    except Exception:
                                        pass
                                    logger.error(
                                        f"CCR: API call failed: {e}, response headers: {resp_headers}"
                                    )
                                    raise

                            # Handle CCR tool calls
                            try:
                                final_resp_json = await self.ccr_response_handler.handle_response(
                                    resp_json,
                                    optimized_messages,
                                    tools,
                                    api_call_fn,
                                    provider="anthropic",
                                )
                                # Update response content with final response
                                resp_json = final_resp_json
                                # Remove encoding headers since content is now uncompressed JSON
                                ccr_response_headers = {
                                    k: v
                                    for k, v in response.headers.items()
                                    if k.lower() not in ("content-encoding", "content-length")
                                }
                                try:
                                    ccr_content = json.dumps(final_resp_json).encode()
                                except (TypeError, ValueError) as json_err:
                                    logger.warning(
                                        f"[{request_id}] CCR: JSON serialization failed: {json_err}"
                                    )
                                    ccr_content = json.dumps(resp_json).encode()
                                response = httpx.Response(
                                    status_code=200,
                                    content=ccr_content,
                                    headers=ccr_response_headers,
                                )
                                # Only claim success when no headroom_retrieve remains.
                                # On an intentional mixed-tool skip (#839) the response
                                # still carries headroom_retrieve for the client to
                                # resolve — logging "handled successfully" there is
                                # misleading. Classify via the shared, provider-generic
                                # residual-CCR signal.
                                from headroom.ccr.response_handler import (
                                    RESIDUAL_CCR_SKIPPED_MIXED,
                                )

                                residual_status = self.ccr_response_handler.residual_ccr_status(
                                    final_resp_json, "anthropic"
                                )
                                if residual_status == RESIDUAL_CCR_SKIPPED_MIXED:
                                    logger.info(
                                        f"[{request_id}] CCR: Skipped retrieval — "
                                        "headroom_retrieve returned alongside a client "
                                        "tool for the client to resolve"
                                    )
                                else:
                                    logger.info(
                                        f"[{request_id}] CCR: Retrieval handled successfully"
                                    )
                            except Exception as e:
                                import traceback

                                logger.error(
                                    f"[{request_id}] CCR: Response handling failed: {e}\n"
                                    f"Traceback: {traceback.format_exc()}"
                                )
                                raise

                        # Memory: Handle memory tool calls in response
                        if (
                            self.memory_handler
                            and memory_user_id
                            and resp_json
                            and response.status_code == 200
                            and self.memory_handler.has_memory_tool_calls(resp_json, "anthropic")
                        ):
                            logger.info(
                                f"[{request_id}] Memory: Detected memory tool call, handling..."
                            )

                            try:
                                # Execute memory tool calls
                                tool_results = await self.memory_handler.handle_memory_tool_calls(
                                    resp_json,
                                    memory_user_id,
                                    "anthropic",
                                    request_context=memory_request_ctx,
                                )

                                if tool_results:
                                    # Create continuation messages
                                    assistant_msg = {
                                        "role": "assistant",
                                        "content": resp_json.get("content", []),
                                    }
                                    user_msg = {
                                        "role": "user",
                                        "content": tool_results,
                                    }

                                    continuation_messages = optimized_messages + [
                                        assistant_msg,
                                        user_msg,
                                    ]

                                    # Make continuation API call
                                    continuation_body = {**body, "messages": continuation_messages}
                                    if tools:
                                        continuation_body["tools"] = tools

                                    cont_response = await self._retry_request(
                                        "POST",
                                        url,
                                        headers,
                                        continuation_body,
                                        timeout=self._anthropic_buffered_request_timeout(),
                                    )

                                    # Update response with continuation
                                    resp_json = cont_response.json()
                                    response = cont_response
                                    logger.info(
                                        f"[{request_id}] Memory: Tool calls handled, continuation complete"
                                    )

                            except Exception as e:
                                logger.warning(
                                    f"[{request_id}] Memory: Tool call handling failed: {e}"
                                )
                                # Continue with original response

                        # Buffered response hooks run for every successful turn,
                        # not only the CCR branch. Reuse the request context so
                        # observers close the exact turn they opened and a
                        # search/reload hook can consume its synthetic tool call.
                        _hook_usage = _AnthropicTurnHookUsage()
                        if _req_ctx is not None and resp_json and response.status_code == 200:
                            from headroom.proxy.turn_hooks import run_response_hooks

                            _hook_usage.record(resp_json)

                            async def _turn_hook_call_model(
                                hook_messages: list[dict[str, Any]],
                            ) -> dict[str, Any]:
                                continuation_body = {**body, "messages": hook_messages}
                                continuation_response = await self._retry_request(
                                    "POST",
                                    url,
                                    headers,
                                    continuation_body,
                                    timeout=self._anthropic_buffered_request_timeout(),
                                )
                                continuation_json = continuation_response.json()
                                _hook_usage.record(continuation_json)
                                return continuation_json

                            hooked_json = await run_response_hooks(
                                _req_ctx, resp_json, _turn_hook_call_model
                            )
                            _hook_usage.settle(hooked_json)
                            if hooked_json is not resp_json:
                                resp_json = hooked_json
                                response = httpx.Response(
                                    status_code=200,
                                    content=json.dumps(resp_json).encode(),
                                    headers={
                                        key: value
                                        for key, value in response.headers.items()
                                        if key.lower() not in ("content-encoding", "content-length")
                                    },
                                )

                        total_latency = (time.time() - start_time) * 1000

                        # Parse response for output token count and cache metrics
                        output_tokens = 0
                        cr_tokens = 0
                        cw_tokens = 0
                        cw_5m_tokens = 0
                        cw_1h_tokens = 0
                        uncached_input_tokens = 0
                        # Bound before the block below so the RequestOutcome emit can never
                        # hit an unbound name on an early-exit or except path — an
                        # accounting field must not be able to 500 a live request.
                        _thinking = ThinkingTokens()
                        _stop_reason = None
                        if resp_json:
                            usage = resp_json.get("usage", {})
                            output_tokens = int(usage.get("output_tokens", 0) or 0)
                            _thinking = _thinking_tokens_for(resp_json)
                            _stop_reason = (resp_json or {}).get("stop_reason")
                            # The split is measured against the final response
                            # only. ``output_tokens`` below also absorbs usage
                            # from hook-triggered calls whose bodies are not
                            # ``resp_json``, so on those turns any thinking they
                            # did lands in the visible bucket. Bounded and
                            # documented rather than threaded through the hook
                            # accumulator: it affects only hook paths, and the
                            # Anthropic count is flagged inferred regardless.
                            output_tokens += _hook_usage.output_tokens
                            cr_tokens = int(usage.get("cache_read_input_tokens", 0) or 0)
                            cr_tokens += _hook_usage.cache_read_tokens
                            cw_tokens = int(usage.get("cache_creation_input_tokens", 0) or 0)
                            cw_tokens += _hook_usage.cache_write_tokens
                            cw_5m_tokens, cw_1h_tokens = self._extract_anthropic_cache_ttl_metrics(
                                usage
                            )
                            cw_5m_tokens += _hook_usage.cache_write_5m_tokens
                            cw_1h_tokens += _hook_usage.cache_write_1h_tokens
                            uncached_input_tokens = int(usage.get("input_tokens", 0) or 0)
                            uncached_input_tokens += max(
                                0,
                                _hook_usage.input_tokens
                                - _hook_usage.cache_read_tokens
                                - _hook_usage.cache_write_tokens,
                            )

                        # Track cache bust: tokens that lost their cache discount due to compression.
                        # If we had X tokens cached last turn and only Y hit cache this turn,
                        # then (X - Y) tokens were busted by our modifications.
                        expected_cached = prefix_tracker._cached_token_count
                        if expected_cached > 0 and tokens_saved > 0:
                            bust_tokens = max(0, expected_cached - cr_tokens)
                            if bust_tokens > 0:
                                logger.info(
                                    f"[{request_id}] CACHE-BUST: "
                                    f"expected_cached={expected_cached:,} actual_read={cr_tokens:,} "
                                    f"tokens_lost={bust_tokens:,} tokens_saved={tokens_saved:,}"
                                )
                                await self.metrics.record_cache_bust(bust_tokens)

                        # Update prefix cache tracker for next turn
                        next_original_messages = copy.deepcopy(original_client_messages)
                        next_forwarded_messages = copy.deepcopy(optimized_messages)
                        assistant_message = self._assistant_message_from_response_json(resp_json)
                        if assistant_message is not None:
                            next_original_messages.append(copy.deepcopy(assistant_message))
                            next_forwarded_messages.append(copy.deepcopy(assistant_message))

                        # Cache-miss attribution (#1313): when this turn expected a
                        # prompt-cache hit but got cr_tokens == 0, decide whether the
                        # cache most likely lapsed (idle > provider TTL → suggest a
                        # longer TTL) or the cacheable prefix changed (content shifted).
                        # Classify BEFORE update_from_response, which overwrites the
                        # last-turn state the classifier reads (idle clock, prefix,
                        # cached-token count). `optimized_messages` is the prefix we
                        # forwarded this turn; compare it against last turn's.
                        # `hasattr` guard: some tests inject a SimpleNamespace stub
                        # tracker that only implements the freeze API, not the full
                        # PrefixCacheTracker surface.
                        if hasattr(prefix_tracker, "classify_cache_miss"):
                            miss = prefix_tracker.classify_cache_miss(
                                cache_read_tokens=cr_tokens,
                                current_forwarded_messages=optimized_messages,
                            )
                            from headroom.cache.ttl_observations import (
                                record_cache_observation,
                            )

                            record_cache_observation(
                                provider="anthropic", model=model, attribution=miss
                            )
                            if miss.is_miss:
                                logger.info(
                                    f"[{request_id}] CACHE-MISS-ATTRIBUTION: reason={miss.reason} "
                                    f"idle={miss.idle_seconds:.0f}s ttl={miss.cache_ttl_seconds}s "
                                    f"expected_cached={miss.expected_cached_tokens:,} "
                                    f"prefix_changed={miss.prefix_changed} "
                                    f"ttl_exceeded={miss.ttl_exceeded}"
                                )
                                await self.metrics.record_cache_miss_attribution(
                                    provider_name, miss.reason
                                )

                        prefix_tracker.update_from_response(
                            cache_read_tokens=cr_tokens,
                            cache_write_tokens=cw_tokens,
                            messages=next_forwarded_messages,
                            original_messages=next_original_messages,
                        )

                        # Cache response under the SAME key it was looked up by:
                        # cache_lookup_messages is the raw pre-mutation snapshot, not
                        # the live (compressed/hooked) `messages` (#327).
                        # ``resp_json`` is None when the reply did not parse as
                        # JSON — an SSE stream, most often. Caching those bytes
                        # poisons the entry for every later caller that shares
                        # the key: the cache key has no ``stream`` component, so
                        # a buffered request would be answered with a stream it
                        # cannot read (#2952).
                        #
                        # ``not stream`` mirrors the read gate at the cache
                        # lookup above. ``stream`` still holds the *client's*
                        # original flag here — the buffered-CCR conversion
                        # flips ``body["stream"]``, never this variable — so a
                        # turn the client asked to stream is the one case that
                        # can reach this store site with a buffered body. That
                        # body was shaped by a forced ``stream: false`` flip
                        # plus CCR tool injection, and the key cannot tell it
                        # apart from an ordinary non-streaming reply, so
                        # storing it lets a later caller be answered with a
                        # response built for a request it never made (#3019).
                        if (
                            self.cache
                            and not stream
                            and response.status_code == 200
                            and resp_json is not None
                        ):
                            await self.cache.set(
                                cache_lookup_messages,
                                model,
                                response.content,
                                dict(response.headers),
                                tokens_saved=tokens_saved,
                                **cache_key_fields,
                            )

                        # Subscription tracker: update headroom contribution
                        # counters. Provider-specific OAuth/subscription
                        # accounting — stays outside the funnel (different
                        # concern, only fires for Bearer-not-sk-ant tokens).
                        if _auth_header.startswith("Bearer ") and not _auth_header.startswith(
                            "Bearer sk-ant-api"
                        ):
                            from headroom.subscription.tracker import (
                                get_subscription_tracker as _get_sub_tracker,
                            )

                            _sub_tracker = _get_sub_tracker()
                            if _sub_tracker is not None:
                                _sub_tracker.update_contribution(
                                    tokens_submitted=optimized_tokens,
                                    tokens_saved_compression=tokens_saved,
                                    tokens_saved_cache_reads=cr_tokens,
                                )

                        # The pre-refactor PERF emit (above) read raw usage
                        # off ``resp_usage`` instead of trusting cr_tokens /
                        # cw_tokens. Both paths land on identical numbers
                        # (extraction happens just above the cost_tracker
                        # call), so the funnel uses the already-computed
                        # values for consistency. Pre-refactor's
                        # ``cache_hit`` local was correctly derived from
                        # cache_read>0; the funnel re-derives via the
                        # outcome property — same result.
                        #
                        # ``attempted_input_tokens`` was MISSING from the
                        # pre-refactor record_request call here (one of the
                        # 7-of-18 sites the P0 audit flagged). The funnel
                        # forces it to a value — using
                        # ``optimized_tokens + tokens_saved`` as the
                        # fallback denominator, same as the streaming path
                        # uses (see _finalize_stream_response). Dashboards
                        # that were showing 0% active-savings on non-
                        # streaming Anthropic traffic will now show the
                        # correct ratio.
                        await self._record_request_outcome(
                            RequestOutcome(
                                request_id=request_id,
                                provider=provider_name,
                                model=model,
                                status_code=response.status_code,
                                original_tokens=original_tokens,
                                optimized_tokens=optimized_tokens,
                                provider_input_tokens=(
                                    uncached_input_tokens + cr_tokens + cw_tokens
                                ),
                                output_tokens=output_tokens,
                                tokens_saved=tokens_saved,
                                attempted_input_tokens=optimized_tokens + tokens_saved,
                                cache_read_tokens=cr_tokens,
                                cache_write_tokens=cw_tokens,
                                cache_write_5m_tokens=cw_5m_tokens,
                                cache_write_1h_tokens=cw_1h_tokens,
                                uncached_input_tokens=uncached_input_tokens,
                                thinking_tokens=_thinking.tokens,
                                thinking_inferred=_thinking.inferred,
                                stop_reason=_stop_reason,
                                total_latency_ms=total_latency,
                                overhead_ms=optimization_latency,
                                pipeline_timing=pipeline_timing,
                                waste_signals=waste_signals_dict,
                                transforms_applied=tuple(transforms_applied),
                                num_messages=len(messages),
                                tags=tags,
                                client=client,
                                turn_id=compute_turn_id(
                                    model, body.get("system"), body.get("messages")
                                ),
                                # `original_client_messages` is the deep-copied
                                # pre-compression snapshot; `body["messages"]` is the
                                # compressed list sent upstream. Both gated by
                                # `log_full_messages`.
                                request_messages=original_client_messages
                                if self.config.log_full_messages
                                else None,
                                compressed_messages=body.get("messages")
                                if self.config.log_full_messages
                                else None,
                            )
                        )

                        # Framing headers describe how the *upstream* framed
                        # its body, not what this response is: httpx already
                        # decompressed it, Starlette recomputes the length,
                        # and uvicorn owns the connection. Replaying a stale
                        # ``transfer-encoding: chunked`` over a fixed-length
                        # body is what made an HTTP 200 read as empty in
                        # #3019. ``cf-*`` is CDN provenance the caller has no
                        # use for, and the header set clients cite as
                        # evidence of an intermediary mangling a reply
                        # (#3130).
                        response_headers = {
                            k: v
                            for k, v in sanitize_forwarded_response_headers(
                                response.headers
                            ).items()
                            if not k.lower().startswith("cf-")
                        }

                        # Inject Headroom compression metrics (for SaaS metering)
                        response_headers["x-headroom-tokens-before"] = str(original_tokens)
                        response_headers["x-headroom-tokens-after"] = str(optimized_tokens)
                        response_headers["x-headroom-tokens-saved"] = str(tokens_saved)
                        response_headers["x-headroom-model"] = model
                        if transforms_applied:
                            from headroom.proxy.cost import header_safe_transforms

                            response_headers["x-headroom-transforms"] = ",".join(
                                header_safe_transforms(transforms_applied)
                            )
                        if cache_hit:
                            response_headers["x-headroom-cached"] = "true"
                        if _compression_failed:
                            response_headers["x-headroom-compression-failed"] = "true"

                        # Inline CCR marker resolution, non-streaming only.
                        # Deliberately OUTSIDE the has_ccr_tool_calls gate
                        # above: the callers this flag exists for (#2509,
                        # Headroom behind a LiteLLM guardrail hop) never get
                        # a headroom_retrieve tool-call turn, so gating on
                        # one makes the flag a no-op for its own use case.
                        # Runs before the security scan so the scanner sees
                        # the resolved text, not the marker.
                        if (
                            getattr(self.config, "ccr_resolve_markers_inline", False)
                            and resp_json
                            and response.status_code == 200
                        ):
                            resp_json = resolve_markers_in_response(resp_json)
                            response = httpx.Response(
                                status_code=200,
                                content=json.dumps(resp_json).encode(),
                                headers=response_headers,
                            )

                        # Enterprise Security: scan response + de-anonymize.
                        # Gate on a 200 upstream like the sibling CCR/cache/buffered
                        # blocks below: without this, a non-2xx upstream (rate limit
                        # 429, overloaded 529, 5xx) whose JSON body is scanned was
                        # rebuilt as httpx.Response(status_code=200) and returned as
                        # HTTP 200, so the client's retry/backoff never triggered and
                        # an error looked like success.
                        if (
                            self.security
                            and _security_ctx
                            and resp_json
                            and response.status_code == 200
                        ):
                            try:
                                resp_json = self.security.scan_response(resp_json, _security_ctx)
                                response = httpx.Response(
                                    status_code=200,
                                    content=json.dumps(resp_json).encode(),
                                    headers=response_headers,
                                )
                                if not buffered_stream_ccr:
                                    return Response(
                                        content=response.content,
                                        status_code=response.status_code,
                                        headers=response_headers,
                                    )
                            except Exception as sec_err:
                                logger.warning(
                                    f"[{request_id}] Security response scan error: {sec_err}"
                                )

                        if (
                            buffered_stream_ccr
                            and response.status_code == 200
                            and not resp_json
                            and _looks_like_sse_response(response)
                        ):
                            # Upstream streamed instead of buffering, so there is
                            # nothing to resynthesize — but the client asked for a
                            # stream and this already is one. Relay it verbatim
                            # rather than falling through to a plain Response the
                            # _BufferedCCRResponse wrapper can only turn into a bare
                            # error event (#2952).
                            logger.warning(
                                f"[{request_id}] CCR: relaying the upstream SSE reply verbatim; "
                                "server-side retrieval was skipped for this turn"
                            )
                            relay_headers = {
                                k: v
                                for k, v in response_headers.items()
                                if k.lower()
                                not in (
                                    "content-encoding",
                                    "content-length",
                                    "transfer-encoding",
                                    "content-type",
                                )
                            }
                            relayed_sse = response.content

                            async def _upstream_sse_relay():
                                yield relayed_sse

                            return StreamingResponse(
                                _upstream_sse_relay(),
                                media_type="text/event-stream",
                                headers=relay_headers,
                            )

                        if buffered_stream_ccr and response.status_code == 200 and not resp_json:
                            logger.warning(
                                f"[{request_id}] CCR: rejecting malformed buffered 200 reply "
                                f"(content-type={response.headers.get('content-type')!r}, "
                                f"body_bytes={len(response.content)})"
                            )
                            return Response(
                                content=json.dumps(
                                    {
                                        "error": {
                                            "type": "upstream_protocol_error",
                                            "message": "Upstream returned an invalid buffered response.",
                                        }
                                    }
                                ),
                                status_code=502,
                                media_type="application/json",
                            )

                        if buffered_stream_ccr and response.status_code == 200 and resp_json:
                            sse_headers = {
                                k: v
                                for k, v in response_headers.items()
                                if k.lower()
                                not in (
                                    "content-encoding",
                                    "content-length",
                                    "transfer-encoding",
                                    "content-type",
                                )
                            }

                            def _sse_error_event(message: str) -> bytes:
                                error_event = {
                                    "type": "error",
                                    "error": {"type": "api_error", "message": message},
                                }
                                return f"event: error\ndata: {json.dumps(error_event)}\n\n".encode()

                            # Residual headroom_retrieve is only a hard failure when it
                            # is NOT an intentional mixed-tool skip. When the model
                            # emitted headroom_retrieve alongside a client tool (#839),
                            # the handler deliberately leaves both tool_use blocks in
                            # place for the client to resolve — a legal turn that the
                            # non-streaming path returns as 200. Fall through to the
                            # SSE resynthesis below so the stream:true path matches it
                            # and preserves both blocks. Use the shared, provider-generic
                            # residual-CCR signal (not an Anthropic-only branch).
                            from headroom.ccr.response_handler import RESIDUAL_CCR_ERROR

                            if (
                                self.ccr_response_handler
                                and self.ccr_response_handler.residual_ccr_status(
                                    resp_json, "anthropic"
                                )
                                == RESIDUAL_CCR_ERROR
                            ):
                                logger.warning(
                                    f"[{request_id}] CCR: Buffered streaming response still "
                                    "contains an unresolved headroom_retrieve after handling; "
                                    "failing closed"
                                )

                                async def _residual_ccr_error_sse():
                                    yield _sse_error_event(
                                        "Unable to safely complete streamed CCR retrieval."
                                    )

                                return StreamingResponse(
                                    _residual_ccr_error_sse(),
                                    media_type="text/event-stream",
                                    headers=sse_headers,
                                    status_code=502,
                                )

                            try:
                                sse_events = self._response_to_sse(resp_json, "anthropic")
                            except ValueError as sse_err:
                                logger.warning(
                                    f"[{request_id}] CCR: Failed to convert buffered response "
                                    f"to SSE: {sse_err}"
                                )

                                async def _conversion_error_sse():
                                    yield _sse_error_event(
                                        "Unable to safely convert buffered response to SSE."
                                    )

                                return StreamingResponse(
                                    _conversion_error_sse(),
                                    media_type="text/event-stream",
                                    headers=sse_headers,
                                    status_code=502,
                                )

                            async def _buffered_ccr_sse():
                                for event in sse_events:
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

                async def _buffered_ccr_operation_salvaging():
                    """Never trade a successful upstream turn for a synthesized error.

                    Everything the buffered path does after the provider answers
                    — server-side retrieval, memory tool calls, turn hooks, usage
                    accounting, caching, SSE resynthesis — is post-processing on
                    a turn that already succeeded and was already billed. When a
                    step raised unexpectedly the whole turn surfaced as a generic
                    ``api_error``, so the client lost a complete 69KB answer the
                    provider had produced (#3088), and the cause was unlogged.

                    Relay the upstream's own answer instead. Two things are
                    deliberately preserved: the exception is logged with a
                    traceback so the real defect stays diagnosable rather than
                    being papered over, and a response the client cannot safely
                    consume is never salvaged — see ``_can_salvage``.
                    """
                    try:
                        return await _buffered_ccr_operation()
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        salvaged = _salvageable_upstream.get("resp_json")
                        if salvaged is None or not self._can_salvage_buffered_upstream(salvaged):
                            raise
                        logger.error(
                            f"[{request_id}] CCR: buffered post-processing failed after a "
                            "successful upstream turn; relaying the upstream response "
                            "instead of failing the request (#3088)",
                            exc_info=True,
                        )
                        try:
                            events = self._response_to_sse(salvaged, "anthropic")
                        except Exception:
                            logger.error(
                                f"[{request_id}] CCR: could not resynthesize the salvaged "
                                "upstream response; failing the request",
                                exc_info=True,
                            )
                            raise

                        async def _salvaged_sse():
                            for event in events:
                                yield event

                        return StreamingResponse(
                            _salvaged_sse(),
                            media_type="text/event-stream",
                        )

                if buffered_stream_ccr:
                    operation = asyncio.create_task(_buffered_ccr_operation_salvaging())

                    # Holds out for the real status, then keeps the stream alive
                    # once waiting silently would risk the client's idle
                    # watchdog. Both halves live in one shared place so the
                    # OpenAI twin cannot drift from it (#3079).
                    _buffered_call = buffered_ccr_asgi_call(
                        operation=operation,
                        fmt=ANTHROPIC_ERROR_FORMAT,
                        grace_seconds=getattr(
                            self.config,
                            "buffered_ccr_grace_seconds",
                            DEFAULT_BUFFERED_CCR_GRACE_SECONDS,
                        ),
                        record_failed=self.metrics.record_failed,
                        request_id=request_id,
                    )

                    class _BufferedCCRResponse(Response):
                        async def __call__(self, scope, receive, send):  # noqa: ANN001
                            await _buffered_call(scope, receive, send)

                    return _BufferedCCRResponse(media_type="text/event-stream")
                return await _buffered_ccr_operation_salvaging()
            except HTTPException:
                # FastAPI HTTPException carries its own status code, headers,
                # and client-facing message (e.g. 429 with Retry-After, 413 for
                # oversized bodies). Let FastAPI's own exception handler produce
                # the response — swallowing it into the 502 catch-all below
                # would regress rate-limit and budget responses to 502 with no
                # Retry-After header. The outer finally still runs.
                raise
            except Exception as e:
                await self.metrics.record_failed(provider=provider_name)
                # Log full error details internally for debugging
                logger.error(f"[{request_id}] Request failed: {type(e).__name__}: {e}")

                # Try fallback if enabled
                if self.config.fallback_enabled and self.config.fallback_provider == "openai":
                    logger.info(f"[{request_id}] Attempting fallback to OpenAI")
                    # Convert to OpenAI format and retry
                    # (simplified - would need message format conversion)

                # Return sanitized error message to client (don't expose internal details)
                return JSONResponse(
                    status_code=502,
                    content={
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": "An error occurred while processing your request. Please try again.",
                        },
                    },
                )
            finally:
                # Unit 2: always emit pre-upstream stage timings exactly
                # once per request, even on early/error paths.
                await _finalize_pre_upstream()
        finally:
            # Unit 4: defense-in-depth. The inner try/finally above
            # already calls this on every normal exit path, but an
            # unexpected exception between the semaphore acquire and the
            # inner try (e.g. AttributeError in a transform, OOM in a
            # deep-copy) would otherwise leak the pre-upstream semaphore
            # permanently. The emit function is idempotent.
            await _finalize_pre_upstream()

    async def handle_anthropic_batch_create(
        self,
        request: Request,
    ) -> Response:
        """Handle Anthropic POST /v1/messages/batches endpoint with compression.

        Anthropic batch format:
        {
            "requests": [
                {
                    "custom_id": "req-1",
                    "params": {
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": "Hello"}]
                    }
                },
                ...
            ]
        }

        This method applies compression to each request's messages before forwarding.
        """
        from fastapi.responses import JSONResponse, Response

        from headroom.ccr import CCRToolInjector
        from headroom.proxy.helpers import MAX_REQUEST_BODY_SIZE, _read_request_json
        from headroom.proxy.modes import is_cache_mode
        from headroom.utils import extract_user_query

        start_time = time.time()
        request_id = await self._next_request_id()

        # Check request body size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "type": "error",
                    "error": {
                        "type": "request_too_large",
                        "message": f"Request body too large. Maximum size is {MAX_REQUEST_BODY_SIZE // (1024 * 1024)}MB",
                    },
                },
            )

        # Parse request
        try:
            body = await _read_request_json(request)
        except (json.JSONDecodeError, ValueError) as e:
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"Invalid request body: {e!s}",
                    },
                },
            )

        requests_list = body.get("requests", [])
        if not requests_list:
            return JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Missing or empty 'requests' field in batch request",
                    },
                },
            )

        # Extract headers
        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        client = classify_client(headers, default="claude")
        tags = extract_tags(headers)
        # PR-A5 (P5-49): strip internal x-headroom-* before forwarding upstream.
        from headroom.proxy.helpers import (
            _strip_internal_headers,
            log_outbound_headers,
            merge_extra_headers,
        )

        _pre_strip_count = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        # Always the configured Anthropic target; no per-request override.
        headers = merge_extra_headers(
            headers,
            self.config.anthropic_extra_headers,
            upstream_url=None,
            config=self.config,
        )
        log_outbound_headers(
            forwarder="anthropic_batch",
            stripped_count=_pre_strip_count,
            request_id=request_id,
        )

        # Track compression stats across all batch requests
        total_original_tokens = 0
        total_optimized_tokens = 0
        total_tokens_saved = 0
        compressed_requests = []
        pipeline_timing: dict[str, float] = {}

        from headroom.transforms.cold_prefix import anthropic_cache_ttl_seconds

        # Apply compression to each request in the batch
        for batch_req in requests_list:
            custom_id = batch_req.get("custom_id", "")
            params = batch_req.get("params", {})
            canonical_params = dict(params)
            original_tools = canonical_params.get("tools")
            messages = params.get("messages", [])
            original_messages = copy.deepcopy(messages)
            model = params.get("model", "unknown")
            cache_ttl_seconds = anthropic_cache_ttl_seconds(
                model, original_messages, params.get("system")
            )

            if not messages or not self.config.optimize:
                # No messages or optimization disabled - pass through unchanged
                compressed_requests.append(
                    {
                        "custom_id": custom_id,
                        "params": canonical_params,
                    }
                )
                continue

            if original_tools is not None:
                sorted_tools = self._sort_tools_deterministically(original_tools)
                if sorted_tools != original_tools:
                    canonical_params["tools"] = sorted_tools

            # Apply optimization
            original_tokens = 0  # Initialize before try to prevent UnboundLocalError
            try:
                context_limit = self.anthropic_provider.get_context_limit(model)
                frozen_message_count = (
                    self._strict_previous_turn_frozen_count(original_messages, 0)
                    if is_cache_mode(self.config.mode)
                    else 0
                )
                if is_cache_mode(self.config.mode):
                    optimized_messages = messages
                    _, original_tokens = await self._count_tokens_offloaded(model, messages)
                    optimized_tokens = original_tokens
                else:
                    from headroom.proxy.helpers import COMPRESSION_TIMEOUT_SECONDS

                    # Offload off the event loop (#1701): an inline apply()
                    # blocks every other request for the duration; a timeout
                    # here is caught below and passes the item through.
                    result = await self._run_compression_in_executor(
                        lambda messages=messages, model=model, context_limit=context_limit, frozen_message_count=frozen_message_count, cache_ttl_seconds=cache_ttl_seconds: (
                            self.anthropic_pipeline.apply(
                                messages=messages,
                                model=model,
                                model_limit=context_limit,
                                context=extract_user_query(messages),
                                frozen_message_count=frozen_message_count,
                                request_id=request_id,
                                cache_ttl_seconds=cache_ttl_seconds,
                                **proxy_pipeline_kwargs(self.config),
                            )
                        ),
                        timeout=COMPRESSION_TIMEOUT_SECONDS,
                    )

                    optimized_messages = result.messages
                    for k, v in result.timing.items():
                        pipeline_timing[k] = pipeline_timing.get(k, 0.0) + v
                    # Use pipeline's token counts for consistency with pipeline logs
                    original_tokens = result.tokens_before
                    optimized_tokens = result.tokens_after
                # Guard: if "optimization" inflated tokens, revert to originals
                if optimized_tokens > original_tokens:
                    logger.warning(
                        f"[{request_id}] Batch item optimization inflated tokens "
                        f"({original_tokens} -> {optimized_tokens}), reverting"
                    )
                    optimized_messages = messages
                    optimized_tokens = original_tokens

                total_original_tokens += original_tokens
                total_optimized_tokens += optimized_tokens
                tokens_saved = original_tokens - optimized_tokens
                total_tokens_saved += tokens_saved

                # CCR Tool Injection: Inject retrieval tool if compression occurred
                tools = canonical_params.get("tools")
                if self.config.ccr_inject_tool and tokens_saved > 0:
                    injector = CCRToolInjector(
                        provider="anthropic",
                        inject_tool=True,
                        inject_system_instructions=self.config.ccr_inject_system_instructions,
                    )
                    optimized_messages, tools, was_injected = injector.process_request(
                        optimized_messages, tools
                    )
                    if was_injected:
                        logger.debug(
                            f"[{request_id}] CCR: Injected retrieval tool for batch request '{custom_id}'"
                        )

                # Create compressed batch request
                compressed_params = {**params, "messages": optimized_messages}
                if tools is not None:
                    sorted_tools = self._sort_tools_deterministically(tools)
                    if sorted_tools != tools:
                        tools = sorted_tools
                    if tools or original_tools is not None:
                        if tools != original_tools:
                            compressed_params["tools"] = tools
                compressed_requests.append(
                    {
                        "custom_id": custom_id,
                        "params": compressed_params,
                    }
                )

                if tokens_saved > 0:
                    logger.debug(
                        f"[{request_id}] Batch request '{custom_id}': "
                        f"{original_tokens:,} -> {optimized_tokens:,} tokens "
                        f"(saved {tokens_saved:,})"
                    )

            except Exception as e:
                logger.warning(
                    f"[{request_id}] Optimization failed for batch request '{custom_id}': {e}"
                )
                # Same fail-open accounting as the single-message path: a
                # timeout means the budget was too tight, not a code bug.
                reason = (
                    "timeout" if isinstance(e, asyncio.TimeoutError | TimeoutError) else "error"
                )
                self.metrics.record_compression_failed(reason)
                # Pass through unchanged on failure
                compressed_requests.append(batch_req)
                total_optimized_tokens += original_tokens

        # Update body with compressed requests
        body["requests"] = compressed_requests

        optimization_latency = (time.time() - start_time) * 1000

        # Forward request to Anthropic
        url = f"{self.ANTHROPIC_API_URL}/v1/messages/batches"

        try:
            # Body is always mutated for batch (compressed requests).
            response = await self._retry_request(
                "POST",
                url,
                headers,
                body,
                body_mutated=True,
                mutation_reasons=["batch_compression"],
                request_id=request_id,
                forwarder_name="anthropic_batch",
                path_for_log="/v1/messages/batches",
                timeout=self._anthropic_buffered_request_timeout(),
            )

            # Batch create: tokens accumulated across all requests in
            # the batch. The funnel records it as a single observation
            # under the synthetic model name "batch" — same as
            # pre-refactor, just routed through the canonical path so
            # batch traffic appears in RequestLog + PERF (it didn't
            # before — sites 4/5/6 were "metrics-only").
            await self._record_request_outcome(
                RequestOutcome(
                    request_id=request_id,
                    provider="anthropic",
                    model="batch",
                    status_code=response.status_code,
                    original_tokens=total_original_tokens,
                    optimized_tokens=total_optimized_tokens,
                    output_tokens=0,
                    tokens_saved=total_tokens_saved,
                    attempted_input_tokens=total_optimized_tokens + total_tokens_saved,
                    total_latency_ms=optimization_latency,
                    overhead_ms=optimization_latency,
                    pipeline_timing=pipeline_timing,
                    num_messages=len(compressed_requests),
                    tags=tags,
                    client=client,
                )
            )

            # Log compression stats
            if total_tokens_saved > 0:
                savings_percent = (
                    (total_tokens_saved / total_original_tokens * 100)
                    if total_original_tokens > 0
                    else 0
                )
                logger.info(
                    f"[{request_id}] Batch ({len(compressed_requests)} requests): "
                    f"{total_original_tokens:,} -> {total_optimized_tokens:,} tokens "
                    f"(saved {total_tokens_saved:,}, {savings_percent:.1f}%)"
                )

            # Store batch context for CCR result processing
            if response.status_code == 200 and self.config.ccr_inject_tool:
                try:
                    response_data = response.json()
                    batch_id = response_data.get("id")
                    if batch_id:
                        await self._store_anthropic_batch_context(
                            batch_id,
                            requests_list,
                            headers.get("x-api-key"),
                        )
                except Exception as e:
                    logger.warning(f"[{request_id}] Failed to store batch context: {e}")

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
            await self.metrics.record_failed(provider="anthropic")
            logger.error(f"[{request_id}] Batch request failed: {type(e).__name__}: {e}")
            return JSONResponse(
                status_code=502,
                content={
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "An error occurred while processing your batch request. Please try again.",
                    },
                },
            )

    async def handle_anthropic_batch_passthrough(
        self,
        request: Request,
        batch_id: str | None = None,
    ) -> Response:
        """Handle Anthropic batch passthrough endpoints.

        Used for:
        - GET /v1/messages/batches - List batches
        - GET /v1/messages/batches/{batch_id} - Get batch
        - GET /v1/messages/batches/{batch_id}/results - Get batch results
        - POST /v1/messages/batches/{batch_id}/cancel - Cancel batch
        """
        from fastapi.responses import Response

        request_id = await self._next_request_id()
        start_time = time.time()
        path = request.url.path
        url = f"{self.ANTHROPIC_API_URL}{path}"

        # Preserve query string parameters (e.g., limit, after_id for list endpoint)
        if request.url.query:
            url = f"{url}?{request.url.query}"

        headers = dict(request.headers.items())
        headers.pop("host", None)
        client = classify_client(headers, default="claude")
        tags = extract_tags(headers)
        # PR-A5 (P5-49): strip internal x-headroom-* before forwarding upstream.
        from headroom.proxy.helpers import (
            _strip_internal_headers,
            log_outbound_headers,
            merge_extra_headers,
        )

        _pre_strip_count = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        # Always the configured Anthropic target; no per-request override.
        headers = merge_extra_headers(
            headers,
            self.config.anthropic_extra_headers,
            upstream_url=None,
            config=self.config,
        )
        log_outbound_headers(
            forwarder="anthropic_batch_passthrough",
            stripped_count=_pre_strip_count,
            request_id=None,
        )

        from starlette.requests import ClientDisconnect

        try:
            body = await request.body()
        except ClientDisconnect:
            logger.debug("Client disconnected during body read for anthropic batch passthrough")
            return Response(status_code=204)

        response = await self.http_client.request(  # type: ignore[union-attr]
            method=request.method,
            url=url,
            headers=headers,
            content=body,
            timeout=self._anthropic_buffered_request_timeout(),
        )

        # Batch passthrough: no compression, no transforms — but we
        # still record the request so dashboards see the upstream call
        # happened. Same funnel as the other 5 anthropic sites.
        latency_ms = (time.time() - start_time) * 1000
        await self._record_request_outcome(
            RequestOutcome(
                request_id=request_id,
                provider="anthropic",
                model="passthrough:batches",
                status_code=response.status_code,
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

        # Remove compression headers
        response_headers = dict(response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("content-length", None)

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
        )

    async def _store_anthropic_batch_context(
        self,
        batch_id: str,
        requests_list: list[dict[str, Any]],
        api_key: str | None,
    ) -> None:
        """Store batch context for CCR result processing.

        Args:
            batch_id: The batch ID from the API response.
            requests_list: The original batch requests.
            api_key: The API key for continuation calls.
        """
        from headroom.ccr import BatchContext, BatchRequestContext, get_batch_context_store

        store = get_batch_context_store()
        context = BatchContext(
            batch_id=batch_id,
            provider="anthropic",
            api_key=api_key,
            api_base_url=self.ANTHROPIC_API_URL,
        )

        for batch_req in requests_list:
            custom_id = batch_req.get("custom_id", "")
            params = batch_req.get("params", {})
            context.add_request(
                BatchRequestContext(
                    custom_id=custom_id,
                    messages=params.get("messages", []),
                    tools=params.get("tools"),
                    model=params.get("model", ""),
                    extras={
                        "max_tokens": params.get("max_tokens", 4096),
                        "system": params.get("system"),
                    },
                )
            )

        await store.store(context)
        logger.debug(f"Stored batch context for {batch_id} with {len(requests_list)} requests")

    async def handle_anthropic_batch_results(
        self,
        request: Request,
        batch_id: str,
    ) -> Response:
        """Handle Anthropic batch results with CCR post-processing.

        This endpoint:
        1. Fetches raw results from Anthropic
        2. Detects CCR tool calls in each result
        3. Executes retrieval and makes continuation calls
        4. Returns processed results with complete responses
        """
        from fastapi.responses import Response

        from headroom.ccr import BatchResultProcessor, get_batch_context_store

        request_id = await self._next_request_id()
        start_time = time.time()

        # Forward request to get raw results
        url = f"{self.ANTHROPIC_API_URL}/v1/messages/batches/{batch_id}/results"

        if request.url.query:
            url = f"{url}?{request.url.query}"

        headers = dict(request.headers.items())
        headers.pop("host", None)
        client = classify_client(headers, default="claude")
        tags = extract_tags(headers)
        # PR-A5 (P5-49): strip internal x-headroom-* before forwarding upstream.
        from headroom.proxy.helpers import (
            _strip_internal_headers,
            log_outbound_headers,
            merge_extra_headers,
        )

        _pre_strip_count = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        # Always the configured Anthropic target; no per-request override.
        headers = merge_extra_headers(
            headers,
            self.config.anthropic_extra_headers,
            upstream_url=None,
            config=self.config,
        )
        log_outbound_headers(
            forwarder="anthropic_batch_results",
            stripped_count=_pre_strip_count,
            request_id=None,
        )

        response = await self.http_client.get(  # type: ignore[union-attr]
            url,
            headers=headers,
            timeout=self._anthropic_buffered_request_timeout(),
        )

        if response.status_code != 200:
            # Error - pass through
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        # Parse results - Anthropic batch results are JSONL format
        raw_content = response.content.decode("utf-8")
        results = []
        for line in raw_content.strip().split("\n"):
            if line.strip():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not results:
            # No results to process
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        # Check if we have context and CCR processing is enabled
        store = get_batch_context_store()
        batch_context = await store.get(batch_id)

        if batch_context is None or not self.config.ccr_inject_tool:
            # No context or CCR disabled - pass through
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        # Process results with CCR handler
        processor = BatchResultProcessor(self.http_client)  # type: ignore[arg-type]
        processed = await processor.process_results(batch_id, results, "anthropic")

        # Convert back to JSONL format
        processed_lines = []
        for p in processed:
            processed_lines.append(json.dumps(p.result))
            if p.was_processed:
                logger.info(
                    f"CCR: Processed batch result {p.custom_id} "
                    f"({p.continuation_rounds} continuation rounds)"
                )

        processed_content = "\n".join(processed_lines)

        # Batch results, post-CCR processing. Like the other batch
        # sites, no token accounting but we record the request so it's
        # visible in dashboards + headroom perf.
        latency_ms = (time.time() - start_time) * 1000
        await self._record_request_outcome(
            RequestOutcome(
                request_id=request_id,
                provider="anthropic",
                model="batch:ccr-processed",
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
            content=processed_content.encode("utf-8"),
            status_code=200,
            media_type="application/jsonl",
        )
