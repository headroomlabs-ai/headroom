"""HeadroomEngine — request/response hook facade (Chunks 2 + 4.2a/4.2b/4.2c + 4.3-i + 5).

Composes the existing compression subsystems behind a clean hook interface.
Does NOT reimplement compression; delegates to injected ``CompressionPipeline``
instances via the ``ports.CompressionPipeline`` Protocol.

Design notes
------------
- **Dependency injection**: pipelines, config, usage_reporter are injected;
  no global state is read or written inside this module.
- **No silent fallbacks**: unregistered (provider, flavor) pairs raise loudly.
- **Passthrough fidelity**: when ``CompressionDecision.should_compress`` is
  False, ``on_request`` returns ``ctx.raw_body`` byte-identical (same object,
  no re-serialization).
- **Chunk 4.2a — real Anthropic path**: when ``anthropic_components`` is
  provided the engine orchestrates the full handler compression-core (mode
  branching, frozen-count, tool-sort, prepare_outbound_body_bytes) using the
  SAME callables the handler uses.
- **Chunk 4.2b — CCR request-side**: when ``ccr_components`` is additionally
  provided, the engine runs the full CCR request-side pipeline (workspace
  resolution, marker scan + session-sticky tool injection, compression tracking,
  proactive expansion) AFTER compression-core, using the SAME callables the
  handler uses. Memory injection (4.2c) runs between compression tracking and
  proactive expansion — the ordering seam is clearly marked.
- **Chunk 4.2c — memory injection**: when ``memory_components`` is provided,
  the engine reads ``RequestContext.prefetched_memory_context`` (pre-fetched by
  the async handler before calling ``on_request``) and appends it to the latest
  non-frozen user turn. Cache-mode and bypass requests skip injection. The
  engine never ``await``s.
- **Chunk 4.3-i — production component model**: component shapes refined for
  multi-session production use. ``CCRComponents.session_turn_counters`` is now
  a ``dict[str, int]`` keyed by session_id (per-session, not global).
  ``MemoryComponents`` holds only per-proxy bits (``memory_handler`` +
  ``default_user_id``); per-request memory context arrives via
  ``RequestContext.prefetched_memory_context`` (option (a) async-bridge:
  the async handler ``await``s the search, stores the result in
  ``RequestContext``, then calls ``engine.on_request``).
- **Chunk 5 — OpenAI chat path**: ``openai_components`` wires the
  ``_on_request_openai_chat`` path.  Key differences from Anthropic:
    * tools are NOT sorted (the live handler never sorts OpenAI tools);
    * the outbound body is ALWAYS canonically serialized (``body_mutated=True``
      is the live handler's default — original bytes are never preserved);
    * streaming prepends ``stream_options: {include_usage: True}`` (mirrors the
      live handler's pre-send injection at line ~2026-2029).
  ``(Provider.OPENAI, Flavor.RESPONSES)`` raises ``NotImplementedError`` with a
  clear message pointing to the responses follow-up task.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable, Mapping
from typing import Any

from headroom.engine.contract import (
    Flavor,
    Provider,
    RequestContext,
    RequestDecision,
    ResponseTelemetry,
    StreamContext,
)
from headroom.engine.ports import CompressionPipeline
from headroom.proxy.auth_mode import classify_auth_mode
from headroom.proxy.compression_decision import CompressionDecision
from headroom.transforms.compression_policy import resolve_policy

logger = logging.getLogger("headroom.engine")


class AnthropicComponents:
    """Real Anthropic compression components for the engine.

    Replaces the fake-pipeline-only path when the engine should reproduce
    byte-identical output with the handler's compression-core path.

    Parameters
    ----------
    pipeline:
        The real ``TransformPipeline`` for Anthropic (same object the
        server builds in HeadroomProxy.__init__).
    provider:
        The AnthropicProvider (used for ``get_context_limit``).
    session_tracker_store:
        The ``SessionTrackerStore`` the engine owns (separate from the
        server's store so prefix-tracker state is engine-private).
    get_compression_cache:
        Callable ``(session_id: str) -> CompressionCache`` — same
        semantics as ``HeadroomProxy._get_compression_cache``.
    config:
        The ``ProxyConfig`` (mode, optimize, hooks, …).
    usage_reporter:
        Commercial gate for ``CompressionDecision.decide``.
    """

    def __init__(
        self,
        *,
        pipeline: Any,
        provider: Any,
        session_tracker_store: Any,
        get_compression_cache: Callable[[str], Any],
        config: Any,
        usage_reporter: Any | None,
    ) -> None:
        self.pipeline = pipeline
        self.provider = provider
        self.session_tracker_store = session_tracker_store
        self.get_compression_cache = get_compression_cache
        self.config = config
        self.usage_reporter = usage_reporter


class OpenAIComponents:
    """Real OpenAI compression components for the engine (Chunk 5).

    Mirrors ``AnthropicComponents`` for the OpenAI chat path.  Injected into
    ``HeadroomEngine`` to enable ``_on_request_openai_chat``.

    Parameters
    ----------
    pipeline:
        The real ``TransformPipeline`` for OpenAI (``HeadroomProxy.openai_pipeline``).
    provider:
        The OpenAIProvider (used for ``get_context_limit``).
    session_tracker_store:
        The ``SessionTrackerStore`` the engine owns (engine-private; separate
        from the server's store so prefix-tracker state does not leak).
    get_compression_cache:
        Callable ``(session_id: str) -> CompressionCache`` — same semantics
        as ``HeadroomProxy._get_compression_cache``.
    config:
        The ``ProxyConfig`` (mode, optimize, …).
    usage_reporter:
        Commercial gate for ``CompressionDecision.decide``.
    """

    def __init__(
        self,
        *,
        pipeline: Any,
        provider: Any,
        session_tracker_store: Any,
        get_compression_cache: Callable[[str], Any],
        config: Any,
        usage_reporter: Any | None,
    ) -> None:
        self.pipeline = pipeline
        self.provider = provider
        self.session_tracker_store = session_tracker_store
        self.get_compression_cache = get_compression_cache
        self.config = config
        self.usage_reporter = usage_reporter


class MemoryComponents:
    """Injectable memory-injection components for the engine (Chunk 4.2c / 4.3-i).

    **Production shape (4.3-i)**: holds only per-proxy bits.  Per-request
    inputs arrive through ``RequestContext.prefetched_memory_context`` (the
    pre-fetched context string) and ``RequestContext.headers_view``
    (``x-headroom-user-id`` header → user-id derivation).

    The async handler (4.3-ii) is responsible for:
      1. Deriving ``memory_user_id`` from ``request.headers``.
      2. ``await``-ing ``memory_handler.search_and_format_context(...)`` within
         the async request scope.
      3. Passing the result as ``RequestContext.prefetched_memory_context``.

    The engine never ``await``s — it reads ``ctx.prefetched_memory_context``
    directly and performs the synchronous placement step.

    When this is ``None`` on ``HeadroomEngine``, the memory step is a no-op
    (byte-identical to the pre-4.2c behaviour for all existing tests).

    Parameters
    ----------
    memory_handler:
        The live memory handler instance (or ``None``). Used for:
          - ``inject_context`` sub-gate (``memory_handler.config.inject_context``).
          - Deriving the user-id fallback when the header is absent.
        Pass ``None`` to skip injection unconditionally (same as no components).
    default_user_id:
        Fallback user-id when the ``x-headroom-user-id`` request header is
        absent.  Mirrors the handler's ``os.environ.get("USER", ...)`` fallback.
        Defaults to the ``USER`` / ``USERNAME`` env var, or ``"default"``.
    """

    def __init__(
        self,
        *,
        memory_handler: Any | None,
        default_user_id: str | None = None,
    ) -> None:
        self.memory_handler = memory_handler
        if default_user_id is None:
            import os as _os

            self.default_user_id: str = _os.environ.get(
                "USER", _os.environ.get("USERNAME", "default")
            )
        else:
            # Explicit value (including "") is used as-is; empty string → gate fails.
            self.default_user_id = default_user_id


class CCRComponents:
    """Injectable CCR request-side components for the engine (Chunk 4.2b / 4.3-i).

    When this is provided to ``HeadroomEngine``, the engine runs the full
    CCR request-side pipeline after compression-core. When ``None``, the
    CCR steps are skipped entirely (no-op), which preserves byte-identical
    output for all existing CCR-OFF fixtures.

    **Production shape (4.3-i)**: the turn counter is now session-keyed via
    ``session_turn_counters: dict[str, int]``.  The engine derives the
    session_id from ``AnthropicComponents.session_tracker_store`` (same call
    as for frozen-count), then increments
    ``session_turn_counters[session_id]`` when compression tracking fires.
    This is the correct production semantics: ``self._turn_counter`` on the
    proxy handler was a single global int, which is incorrect for a
    multi-session proxy (a new session starts counting from wherever the
    previous session left off).

    Parameters
    ----------
    ccr_context_tracker:
        Live ``CCRContextTracker`` instance, or ``None`` when CCR context
        tracking is disabled. When ``None``, compression-tracking and
        proactive-expansion steps are skipped.
    get_compression_store:
        Callable ``() -> CompressionStore`` — returns the store used for
        ``get_metadata(hash_key)`` lookups during compression tracking.
        Injected so tests can supply a stub store without touching the global.
    session_turn_counters:
        Mutable ``dict[str, int]`` mapping session_id → per-session turn
        counter.  The engine increments ``session_turn_counters[session_id]``
        when compression tracking fires, creating the key on first use.
        Callers pass an empty ``{}`` on construction; the engine mutates in
        place so each session accumulates its own count independently.
    """

    def __init__(
        self,
        *,
        ccr_context_tracker: Any | None,
        get_compression_store: Callable[[], Any],
        session_turn_counters: dict[str, int] | None = None,
    ) -> None:
        self.ccr_context_tracker = ccr_context_tracker
        self.get_compression_store = get_compression_store
        # Per-session turn counters: session_id → count.
        # The engine mutates this dict in place on each compression-tracking event.
        self.session_turn_counters: dict[str, int] = (
            session_turn_counters if session_turn_counters is not None else {}
        )


class HeadroomEngine:
    """Facade that composes Headroom compression behind hook-shaped entry points.

    ``on_request`` is the load-bearing method. Two operating modes:

    **Fake-pipeline mode** (Chunks 1-2 tests, legacy): ``anthropic_components``
    is None; the engine uses ``pipelines`` to dispatch and applies a simplified
    (non-mode-branching) pipeline call. Existing Chunk 2 tests continue to pass
    because this path is unchanged.

    **Real-Anthropic mode** (Chunk 4.2a/4.2b): ``anthropic_components`` is set.
    The engine owns the full compression-core orchestration for Anthropic
    requests: mode-branching (token/non-cache/cache-delta), frozen-count
    derivation, tool-sort, and ``prepare_outbound_body_bytes``. It faithfully
    reproduces what ``AnthropicHandlerMixin.handle_messages`` does for
    compression-core. When ``ccr_components`` is also provided, the CCR
    request-side pipeline (steps 1-4) runs after compression-core.

    **Real-OpenAI-Chat mode** (Chunk 5): ``openai_components`` is set.
    The engine orchestrates the full OpenAI chat compression-core: mode-branch,
    frozen-count, pipeline.apply, and canonical serialization.  Key differences
    from Anthropic (preserved to match the live handler exactly):
      - Tools are NOT sorted (``handle_openai_chat`` never calls
        ``_sort_tools_deterministically``).
      - The outbound body is ALWAYS re-serialized canonically (``body_mutated=True``
        is the live handler's effective default; it never passes
        ``original_body_bytes`` to ``_retry_request`` / ``_stream_response``).
      - Streaming injects ``stream_options: {include_usage: True}`` before
        serialization (mirrors live handler lines ~2026-2029).

    Parameters
    ----------
    pipelines:
        Mapping from ``(Provider, Flavor)`` to a ``CompressionPipeline``
        implementor.  Fakes satisfy this in tests; used by the legacy path.
    config:
        Config object forwarded verbatim to ``CompressionDecision.decide``.
        Only ``config.optimize: bool`` is read there.
    usage_reporter:
        Commercial gate forwarded to ``CompressionDecision.decide``.
        ``None`` means no licensing → always allow compression.
    salt:
        Salt bytes for session key derivation (kept for CCR proactive-expansion
        wiring; not consumed in current chunks).
    anthropic_components:
        When set, the engine uses the real Anthropic orchestration path for
        Anthropic/Messages requests (Chunk 4.2a). When None, falls back to
        the fake-pipeline path (Chunks 1-2 behaviour).
    ccr_components:
        When set (and anthropic_components is also set), the engine runs the
        full CCR request-side pipeline after compression-core (Chunk 4.2b).
        When None, CCR steps are a no-op — existing CCR-OFF tests are unchanged.
    memory_components:
        When set (and anthropic_components is also set), the engine runs the
        memory injection step between CCR compression tracking and CCR
        proactive expansion (Chunk 4.2c). When None, memory step is a no-op
        — existing tests are byte-identical to the pre-4.2c behaviour.
    openai_components:
        When set, the engine uses the real OpenAI orchestration path for
        OpenAI/Chat requests (Chunk 5). When None, falls back to the
        fake-pipeline path (Chunks 1-2 behaviour).
    """

    def __init__(
        self,
        *,
        pipelines: Mapping[tuple[Provider, Flavor], CompressionPipeline],
        config: Any,
        usage_reporter: Any | None,
        salt: bytes,
        anthropic_components: AnthropicComponents | None = None,
        ccr_components: CCRComponents | None = None,
        memory_components: MemoryComponents | None = None,
        openai_components: OpenAIComponents | None = None,
    ) -> None:
        self._pipelines = dict(pipelines)
        self._config = config
        self._usage_reporter = usage_reporter
        self._salt = salt
        self._anthropic_components = anthropic_components
        self._ccr_components = ccr_components
        self._memory_components = memory_components
        self._openai_components = openai_components

    # ── Request hook ──────────────────────────────────────────────────────────

    def on_request(
        self,
        ctx: RequestContext,
        *,
        _session_tracker_store_override: Any | None = None,
    ) -> RequestDecision:
        """Process an inbound request.

        For registered ``(provider, flavor)`` combos: classify auth mode,
        decide whether to compress, and either return the raw body unchanged
        (passthrough) or run the pipeline and return the mutated body.

        Parameters
        ----------
        ctx:
            The request context (provider, flavor, raw_body, headers, …).
        _session_tracker_store_override:
            Internal. When set, the Anthropic real path uses this store
            instead of ``self._anthropic_components.session_tracker_store``.
            Intended for the Chunk 4.3-ii shadow hook: the handler seeds a
            controlled store from its own frozen-count snapshot so the engine
            and the live path see identical prefix-tracker state for this
            one call. The engine's private store is NOT modified.

        Raises
        ------
        KeyError
            If ``(ctx.provider, ctx.flavor)`` has no registered pipeline
            AND no real-component path handles it.
        ValueError
            If the raw body cannot be parsed as JSON (malformed request).
        """
        # Real Anthropic path (Chunk 4.2a + 4.2b)
        if (
            ctx.provider == Provider.ANTHROPIC
            and ctx.flavor == Flavor.MESSAGES
            and self._anthropic_components is not None
        ):
            return self._on_request_anthropic_real(
                ctx, _session_tracker_store_override=_session_tracker_store_override
            )

        # Real OpenAI chat path (Chunk 5)
        if ctx.provider == Provider.OPENAI and ctx.flavor == Flavor.CHAT:
            if self._openai_components is not None:
                return self._on_request_openai_chat(ctx)
            # Fall through to fake-pipeline path if no openai_components.

        # OpenAI Responses path — not yet wired (follow-up task).
        if ctx.provider == Provider.OPENAI and ctx.flavor == Flavor.RESPONSES:
            raise NotImplementedError(
                "HeadroomEngine: (Provider.OPENAI, Flavor.RESPONSES) is not yet wired. "
                "The /v1/responses engine path is handled in a follow-up task. "
                "Use the live OpenAI responses handler directly for now."
            )

        # Legacy fake-pipeline path (Chunks 1-2)
        key = (ctx.provider, ctx.flavor)
        if key not in self._pipelines:
            raise KeyError(
                f"No pipeline registered for provider={ctx.provider!r}, "
                f"flavor={ctx.flavor!r}. Register it in the pipelines mapping."
            )

        return self._on_request_fake_pipeline(ctx, self._pipelines[key])

    # ── Real Anthropic orchestration (Chunk 4.2a + 4.2b) ─────────────────────

    def _on_request_anthropic_real(
        self,
        ctx: RequestContext,
        *,
        _session_tracker_store_override: Any | None = None,
    ) -> RequestDecision:
        """Reproduce the handler's compression-core + CCR request-side path.

        Mirrors ``AnthropicHandlerMixin.handle_messages`` through:
          image compress → CompressionDecision → mode-branch pipeline.apply →
          tool-sort → [CCR: workspace-resolve, marker-scan, tool-inject,
          system-instruction-inject, compression-tracking, proactive-expansion]
          → prepare_outbound_body_bytes.

        CCR steps are a no-op when ``self._ccr_components`` is None (all
        existing CCR-OFF fixtures remain byte-identical).

        Ordering seam for 4.2c (memory injection):
            Memory injection runs AFTER compression tracking (step 3) and
            BEFORE proactive expansion (step 4). The comment marked
            ``# ── 4.2c SEAM: memory injection goes here ──`` is the exact
            insertion point. Do NOT move step 4 above that comment.

        Excluded from this chunk: hooks, pipeline_extension events, security
        scan, traffic_learner.

        Parameters
        ----------
        _session_tracker_store_override:
            When set, use this store for session-tracker lookup INSTEAD of
            ``ac.session_tracker_store``. The engine's own store is untouched.
            Used by the Chunk 4.3-ii shadow hook to seed controlled state.
        """
        from headroom.cache.compression_cache import CompressionCache  # noqa: F401
        from headroom.proxy.helpers import (
            BodyMutationTracker,
            prepare_outbound_body_bytes,
        )
        from headroom.proxy.image_compression_decision import ImageCompressionDecision
        from headroom.proxy.modes import is_cache_mode, is_token_mode
        from headroom.utils import extract_user_query

        ac = self._anthropic_components
        assert ac is not None

        original_body_bytes = ctx.raw_body

        # Parse body — raises loudly on malformed JSON
        try:
            body: dict[str, Any] = json.loads(original_body_bytes)
        except json.JSONDecodeError as exc:
            raise ValueError(f"on_request(anthropic): unparseable JSON body: {exc}") from exc

        messages: list[dict[str, Any]] = list(body.get("messages") or [])
        model: str = body.get("model", "unknown")
        # Preserve a deep copy of the original client messages (mirrors deep_copy
        # at handler line ~595) for use in the cache-delta path.
        original_client_messages: list[dict[str, Any]] = copy.deepcopy(messages)

        # Bypass: skip ALL compression when the caller explicitly opts out.
        headers = dict(ctx.headers_view)
        _bypass = (
            headers.get("x-headroom-bypass", "").lower() == "true"
            or headers.get("x-headroom-mode", "").lower() == "passthrough"
        )

        body_mutation_tracker = BodyMutationTracker()

        # Auth mode + policy (computed once; used by all three pipeline sites)
        auth_mode = classify_auth_mode(ctx.headers_view)
        compression_policy = resolve_policy(auth_mode)

        # Compression decision
        _decision = CompressionDecision.decide(
            headers=ctx.headers_view,
            config=ac.config,
            usage_reporter=ac.usage_reporter,
            messages=messages,
        )

        if not _decision.should_compress or _bypass:
            # No-compression path. The legacy handler ALWAYS sorts tools
            # deterministically at the pre-send site (handler ~line 1634),
            # unconditionally — there is no bypass guard around that call.
            # Empirically confirmed: bypass + unsorted tools → sorted outbound.
            # Engine must match: sort tools (if present), then byte-faithful
            # serialize (passthrough when unchanged, canonical when mutated).
            tools = body.get("tools")
            if tools is not None:
                from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin

                sorted_tools = AnthropicHandlerMixin._sort_tools_deterministically(tools)
                if sorted_tools != tools:
                    body_mutation_tracker.mark_mutated("tool_sort")
                body["tools"] = sorted_tools

            if not body_mutation_tracker.mutated:
                try:
                    parsed_original = json.loads(original_body_bytes)
                    if parsed_original != body:
                        body_mutation_tracker.mark_mutated("structural_diff_vs_original")
                except (json.JSONDecodeError, ValueError):
                    body_mutation_tracker.mark_mutated("original_unparseable")

            outbound_bytes, _source = prepare_outbound_body_bytes(
                body=body,
                original_body_bytes=original_body_bytes,
                body_mutated=body_mutation_tracker.mutated,
            )
            return RequestDecision(
                body=outbound_bytes,
                telemetry=ResponseTelemetry(compressed=False),
            )

        # --- Image compression (before text compression, same order as handler) ---
        _image_decision = ImageCompressionDecision.decide(
            headers=ctx.headers_view, config=ac.config, messages=messages
        )
        if _image_decision.should_compress and not is_cache_mode(ac.config.mode):
            from headroom.proxy.helpers import _get_image_compressor

            compressor = None
            try:
                compressor = _get_image_compressor()
                if compressor and compressor.has_images(messages):
                    messages = compressor.compress(messages, provider="anthropic")
                    body_mutation_tracker.mark_mutated("image_compression")
            finally:
                if compressor and hasattr(compressor, "close"):
                    compressor.close()

        # --- Session / frozen-count derivation ---
        # Use the store override when provided (Chunk 4.3-ii shadow hook);
        # otherwise fall back to the engine's own private session store.
        _effective_store = (
            _session_tracker_store_override
            if _session_tracker_store_override is not None
            else ac.session_tracker_store
        )
        session_id = _effective_store.compute_session_id(ctx, model, messages)
        prefix_tracker = _effective_store.get_or_create(session_id, "anthropic")
        frozen_message_count = prefix_tracker.get_frozen_message_count()
        if is_cache_mode(ac.config.mode):
            # Mirrors _strict_previous_turn_frozen_count at handler line ~890.
            frozen_message_count = _strict_previous_turn_frozen_count(
                original_client_messages, frozen_message_count
            )

        # --- Context limit ---
        context_limit = ac.provider.get_context_limit(model)

        # --- hooks/biases (skipped in 4.2a — not present in golden corpus) ---
        biases = None
        request_id = ctx.request_id

        optimized_messages = messages

        # --- Mode branch: token / non-cache / cache-delta ---
        if is_token_mode(ac.config.mode):
            comp_cache = ac.get_compression_cache(session_id)

            # Zone 1: swap cached compressed versions into working copy
            working_messages = comp_cache.apply_cached(messages)

            # Clamp frozen_message_count (mirrors handler lines ~1039-1042)
            cache_frozen_count = comp_cache.compute_frozen_count(messages)
            frozen_message_count = min(frozen_message_count, cache_frozen_count)
            comp_cache.mark_stable_from_messages(messages, frozen_message_count)

            result = ac.pipeline.apply(
                messages=working_messages,
                model=model,
                model_limit=context_limit,
                context=extract_user_query(working_messages),
                frozen_message_count=frozen_message_count,
                biases=biases,
                request_id=request_id,
                compression_policy=compression_policy,
            )

            if result.messages != working_messages:
                comp_cache.update_from_result(messages, result.messages)

            optimized_messages = result.messages
            # Mirror handler line ~1064: always use pipeline result.
            # Structural diff check below detects any real mutation.

        elif not is_cache_mode(ac.config.mode):
            result = ac.pipeline.apply(
                messages=messages,
                model=model,
                model_limit=context_limit,
                context=extract_user_query(messages),
                frozen_message_count=frozen_message_count,
                biases=biases,
                request_id=request_id,
                compression_policy=compression_policy,
            )

            if result.messages != messages:
                optimized_messages = result.messages
                # Do NOT mark mutation explicitly here; structural diff below
                # detects the actual byte change. Handler mirrors this: no
                # explicit mark at lines ~1099-1104 for the non-cache path.

        else:
            # Cache-delta path
            previous_original_messages = prefix_tracker.get_last_original_messages()
            previous_forwarded_messages = prefix_tracker.get_last_forwarded_messages()
            delta = _extract_cache_stable_delta(
                original_client_messages,
                previous_original_messages,
                previous_forwarded_messages,
            )
            if delta is not None:
                stable_forwarded_prefix, delta_messages = delta
                if delta_messages:
                    result = ac.pipeline.apply(
                        messages=delta_messages,
                        model=model,
                        model_limit=context_limit,
                        context=extract_user_query(delta_messages),
                        frozen_message_count=0,
                        biases=biases,
                        request_id=request_id,
                        compression_policy=compression_policy,
                    )
                    optimized_messages = stable_forwarded_prefix + result.messages
                    # Mirror the handler: no explicit mark_mutated here.
                    # The structural diff check below will detect any real change.
                else:
                    optimized_messages = stable_forwarded_prefix
                    # No explicit mutation mark — structural diff detects if needed.
            else:
                # Conservative fallback for cache mode
                optimized_messages = messages

        # --- Tool sort (ALWAYS when tools present) ---
        tools = body.get("tools")
        if tools is not None:
            from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin

            sorted_tools = AnthropicHandlerMixin._sort_tools_deterministically(tools)
            if sorted_tools != tools:
                body_mutation_tracker.mark_mutated("tool_sort")
            body["tools"] = sorted_tools
            tools = body["tools"]  # keep local alias in sync

        # ── CCR request-side (Chunk 4.2b) ────────────────────────────────────
        # Steps 1-4 are a no-op when ccr_components is None or CCR config flags
        # are off, preserving byte-identical output for all existing CCR-OFF
        # fixtures.
        ccr_tool_injected = False
        ccr_workspace_key = ""
        ccr_workspace_label: str | None = None

        ccr = self._ccr_components
        if ccr is not None and not _bypass:
            # Step 1: workspace resolution ────────────────────────────────────
            # Adapted from AnthropicHandlerMixin._resolve_ccr_workspace to work
            # from (headers_view, body) instead of a FastAPI Request object.
            # Fail-closed: ("", None) → skip CCR tracking + expansion.
            ccr_workspace_key, ccr_workspace_label = _resolve_ccr_workspace(ctx.headers_view, body)

            # Step 2: marker scan + session-sticky tool injection ─────────────
            # Gated on the same config flags as the handler.
            if ac.config.ccr_inject_tool or ac.config.ccr_inject_system_instructions:
                from headroom.ccr import CCRToolInjector
                from headroom.proxy.helpers import apply_session_sticky_ccr_tool

                inject_system_instructions = ac.config.ccr_inject_system_instructions
                if inject_system_instructions and frozen_message_count > 0:
                    # Cache hot zone — skip to preserve prefix cache bytes.
                    logger.info(
                        "[%s] CCR(engine): skipping system instruction injection "
                        "(frozen prefix=%d) to preserve cache",
                        request_id,
                        frozen_message_count,
                    )
                    inject_system_instructions = False

                inject_tool = ac.config.ccr_inject_tool
                if inject_tool and frozen_message_count > 0:
                    logger.info(
                        "[%s] CCR(engine): deferring tool injection "
                        "(frozen prefix=%d) to preserve cache",
                        request_id,
                        frozen_message_count,
                    )
                    inject_tool = False

                # Scan for compression markers; always with inject_tool=False
                # because tool-list injection goes through the sticky helper.
                injector = CCRToolInjector(
                    provider="anthropic",
                    inject_tool=False,
                    inject_system_instructions=inject_system_instructions,
                )
                injector.scan_for_markers(optimized_messages)

                # System-instruction injection: only when frozen==0 and compressed.
                if inject_system_instructions and injector.has_compressed_content:
                    optimized_messages = injector.inject_into_system_message(optimized_messages)
                    body_mutation_tracker.mark_mutated("ccr_system_instruction_inject")

                # Sticky-on tool registration (PR-B7): once a session has done CCR
                # the retrieve tool stays in body["tools"] every turn.
                if inject_tool:
                    tools, ccr_tool_injected = apply_session_sticky_ccr_tool(
                        provider="anthropic",
                        session_id=session_id,
                        request_id=request_id,
                        existing_tools=tools,
                        has_compressed_content_this_turn=injector.has_compressed_content,
                    )
                    if ccr_tool_injected:
                        body["tools"] = tools
                        body_mutation_tracker.mark_mutated("ccr_tool_inject")
                        logger.debug(
                            "[%s] CCR(engine): tool registered (session=%s, "
                            "compressed_this_turn=%s, hashes_seen=%d)",
                            request_id,
                            session_id,
                            injector.has_compressed_content,
                            len(injector.detected_hashes),
                        )

                # Step 3: compression tracking ────────────────────────────────
                # Gated on: has_compressed_content AND ccr_context_tracker AND
                # workspace_key resolved. Fail-closed when workspace is empty —
                # tracked under empty key would be un-matchable in analyze_query.
                if injector.has_compressed_content:
                    if ccr.ccr_context_tracker and ccr_workspace_key:
                        # Per-session turn counter: increment only for this session.
                        ccr.session_turn_counters[session_id] = (
                            ccr.session_turn_counters.get(session_id, 0) + 1
                        )
                        _session_turn = ccr.session_turn_counters[session_id]
                        for hash_key in injector.detected_hashes:
                            store = ccr.get_compression_store()
                            entry = store.get_metadata(hash_key)
                            if entry:
                                ccr.ccr_context_tracker.track_compression(
                                    hash_key=hash_key,
                                    turn_number=_session_turn,
                                    tool_name=entry.get("tool_name"),
                                    original_count=entry.get("original_item_count", 0),
                                    compressed_count=entry.get("compressed_item_count", 0),
                                    workspace_key=ccr_workspace_key,
                                    query_context=entry.get("query_context", ""),
                                    sample_content=entry.get("compressed_content", "")[:500],
                                )
                    elif ccr.ccr_context_tracker and not ccr_workspace_key:
                        # Explicit fail-closed log — not a silent skip.
                        logger.info(
                            "[%s] CCR(engine): workspace unresolved; skipping "
                            "track_compression (fail-closed — no x-headroom-cwd / "
                            "x-headroom-project-id header and no cwd: in system prompt)",
                            request_id,
                        )

        # ── Memory injection (Chunk 4.2c / 4.3-i) ───────────────────────────
        # Runs AFTER compression tracking (step 3) and BEFORE proactive
        # expansion (step 4). Mirrors handler lines ~1424-1504.
        # When memory_components is None this entire block is skipped —
        # preserving byte-identical output for all pre-4.2c tests.
        #
        # Production seam (4.3-i): the async handler pre-fetches
        # memory_handler.search_and_format_context and stores the result in
        # ctx.prefetched_memory_context before calling engine.on_request.
        # The engine reads that value here — no awaiting needed.
        mc = self._memory_components
        if mc is not None and not _bypass:
            from headroom.proxy.helpers import get_memory_injection_mode
            from headroom.proxy.memory_decision import MemoryDecision

            # Derive user_id: x-headroom-user-id header → mc.default_user_id fallback.
            # mc.default_user_id was set at MemoryComponents construction time
            # (env var resolved then, not here) so no further env lookup is needed.
            _header_user_id = dict(ctx.headers_view).get("x-headroom-user-id", "")
            memory_user_id: str | None = _header_user_id if _header_user_id else mc.default_user_id

            mem_decision = MemoryDecision.decide(
                headers=ctx.headers_view,
                memory_handler=mc.memory_handler,
                memory_user_id=memory_user_id,
                mode_name=get_memory_injection_mode(),
            )
            if mem_decision.inject:
                # Sub-gate: inject_context flag on the memory handler config.
                _inject_ctx = mc.memory_handler is not None and getattr(
                    getattr(mc.memory_handler, "config", None), "inject_context", True
                )
                if _inject_ctx:
                    # Use the pre-fetched context from the request context.
                    memory_context: str | None = ctx.prefetched_memory_context
                    if memory_context and not is_cache_mode(ac.config.mode):
                        optimized_messages = _append_context_to_latest_non_frozen_user_turn(
                            optimized_messages,
                            memory_context,
                            frozen_message_count=frozen_message_count,
                        )
                        body_mutation_tracker.mark_mutated("memory_injection")
                        logger.debug(
                            "[%s] Memory(engine): injected %d bytes into latest "
                            "non-frozen user turn",
                            request_id,
                            len(memory_context),
                        )
        # ─────────────────────────────────────────────────────────────────────

        # Step 4: proactive expansion ─────────────────────────────────────────
        # ORDERING NOTE: In the full handler this runs AFTER memory injection.
        # The 4.2c memory seam above is the canonical insertion point.
        # Gated on the same workspace and config flags as the handler.
        if (
            ccr is not None
            and not _bypass
            and ccr.ccr_context_tracker is not None
            and ac.config.ccr_proactive_expansion
            and ccr_workspace_key
        ):
            from headroom.proxy.modes import is_cache_mode as _is_cache_mode

            # Extract user query from messages (same loop as handler lines ~1340-1351).
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
                recommendations = ccr.ccr_context_tracker.analyze_query(
                    user_query,
                    ccr.session_turn_counters.get(session_id, 0),
                    workspace_key=ccr_workspace_key,
                )
                if recommendations:
                    expansions = ccr.ccr_context_tracker.execute_expansions(recommendations)
                    if expansions:
                        expansion_text = ccr.ccr_context_tracker.format_expansions_for_context(
                            expansions,
                            workspace_label=ccr_workspace_label,
                        )
                        logger.info(
                            "[%s] CCR(engine): proactively expanded %d context(s) "
                            "based on query relevance",
                            request_id,
                            len(expansions),
                        )
                        if _is_cache_mode(ac.config.mode):
                            logger.info(
                                "[%s] CCR(engine): skipping proactive expansion append "
                                "in cache mode to preserve next-turn prefix stability",
                                request_id,
                            )
                        else:
                            optimized_messages = _append_context_to_latest_non_frozen_user_turn(
                                optimized_messages,
                                expansion_text,
                                frozen_message_count=frozen_message_count,
                            )
                            body_mutation_tracker.mark_mutated("ccr_proactive_expansion")

        # --- Reassemble body ---
        body["messages"] = optimized_messages

        # --- Structural mutation safety-net (mirrors handler lines ~1654-1660) ---
        if not body_mutation_tracker.mutated:
            try:
                parsed_original = json.loads(original_body_bytes)
                if parsed_original != body:
                    body_mutation_tracker.mark_mutated("structural_diff_vs_original")
            except (json.JSONDecodeError, ValueError):
                body_mutation_tracker.mark_mutated("original_unparseable")

        # --- Byte-faithful forward (mirrors prepare_outbound_body_bytes) ---
        outbound_bytes, _source = prepare_outbound_body_bytes(
            body=body,
            original_body_bytes=original_body_bytes,
            body_mutated=body_mutation_tracker.mutated,
        )

        compressed = body_mutation_tracker.mutated
        bytes_saved = max(0, len(original_body_bytes) - len(outbound_bytes))

        return RequestDecision(
            body=outbound_bytes,
            telemetry=ResponseTelemetry(
                bytes_saved=bytes_saved,
                compressed=compressed,
                ccr_fired=ccr_tool_injected,
            ),
        )

    # ── Real OpenAI chat orchestration (Chunk 5) ─────────────────────────────

    def _on_request_openai_chat(self, ctx: RequestContext) -> RequestDecision:
        """Reproduce the OpenAI /v1/chat/completions compression-core.

        Mirrors ``OpenAIHandlerMixin.handle_openai_chat`` through:
          CompressionDecision → mode-branch pipeline.apply → (streaming:
          inject stream_options) → canonical serialization.

        Intentional differences from the Anthropic path that are faithfully
        preserved here to match the live handler byte-for-byte:

        1. **No tool sort** — ``handle_openai_chat`` never calls
           ``_sort_tools_deterministically``.  Do NOT add it.

        2. **Always canonical** — the live handler always calls
           ``_retry_request`` / ``_stream_response`` without
           ``original_body_bytes`` or ``body_mutated=False``, so the
           effective behaviour is always canonical serialization.
           ``prepare_outbound_body_bytes`` is NOT called here; we use
           ``serialize_body_canonical`` directly.

        3. **stream_options injection** — when ``body["stream"]`` is truthy the
           live handler injects ``body["stream_options"] = {"include_usage": True}``
           (or sets the key inside an existing dict) at lines ~2026-2029,
           BEFORE forwarding the bytes.  This produces a byte-level difference
           vs the inbound body and is captured in the streaming golden fixtures.

        Excluded from this chunk: CCR injection, memory injection, hooks,
        pipeline_extension events, prefix-tracker.update_from_response,
        cache, rate-limiting, image compression.  These are all controlled OFF
        in the golden recorder's ``_DEFAULT_CONFIG_KWARGS``, so the 16 chat
        golden fixtures do not exercise them.
        """
        from headroom.proxy.helpers import serialize_body_canonical
        from headroom.proxy.modes import is_cache_mode, is_token_mode
        from headroom.utils import extract_user_query

        oc = self._openai_components
        assert oc is not None

        original_body_bytes = ctx.raw_body

        # Parse body — raises loudly on malformed JSON
        try:
            body: dict[str, Any] = json.loads(original_body_bytes)
        except json.JSONDecodeError as exc:
            raise ValueError(f"on_request(openai_chat): unparseable JSON body: {exc}") from exc

        messages: list[dict[str, Any]] = list(body.get("messages") or [])
        model: str = body.get("model", "unknown")
        original_client_messages: list[dict[str, Any]] = copy.deepcopy(messages)

        # Bypass detection (same gates as the Anthropic path)
        headers = dict(ctx.headers_view)
        _bypass = (
            headers.get("x-headroom-bypass", "").lower() == "true"
            or headers.get("x-headroom-mode", "").lower() == "passthrough"
        )

        # Auth mode + per-auth-mode compression policy
        auth_mode = classify_auth_mode(ctx.headers_view)
        compression_policy = resolve_policy(auth_mode)

        # Compression decision
        _decision = CompressionDecision.decide(
            headers=ctx.headers_view,
            config=oc.config,
            usage_reporter=oc.usage_reporter,
            messages=messages,
        )

        optimized_messages = messages

        if _decision.should_compress and not _bypass:
            # --- Session / frozen-count derivation ---
            session_id = oc.session_tracker_store.compute_session_id(ctx, model, messages)
            prefix_tracker = oc.session_tracker_store.get_or_create(session_id, "openai")
            frozen_message_count = prefix_tracker.get_frozen_message_count()
            if is_cache_mode(oc.config.mode):
                frozen_message_count = _strict_previous_turn_frozen_count(
                    original_client_messages, frozen_message_count
                )

            # --- Context limit ---
            context_limit = oc.provider.get_context_limit(model)

            biases = None
            request_id = ctx.request_id

            # --- Mode branch: token / non-cache / cache-delta ---
            if is_token_mode(oc.config.mode):
                comp_cache = oc.get_compression_cache(session_id)

                working_messages = comp_cache.apply_cached(messages)

                # Re-freeze boundary (mirrors handler lines ~1469-1470)
                frozen_message_count = comp_cache.compute_frozen_count(messages)
                comp_cache.mark_stable_from_messages(messages, frozen_message_count)

                result = oc.pipeline.apply(
                    messages=working_messages,
                    model=model,
                    model_limit=context_limit,
                    context=extract_user_query(working_messages),
                    frozen_message_count=frozen_message_count,
                    biases=biases,
                    request_id=request_id,
                    compression_policy=compression_policy,
                )

                if result.messages != working_messages:
                    comp_cache.update_from_result(messages, result.messages)

                optimized_messages = result.messages

            else:
                result = oc.pipeline.apply(
                    messages=messages,
                    model=model,
                    model_limit=context_limit,
                    context=extract_user_query(messages),
                    frozen_message_count=frozen_message_count,
                    biases=biases,
                    request_id=request_id,
                    compression_policy=compression_policy,
                )

                if result.messages != messages:
                    optimized_messages = result.messages

        # --- Reassemble body ---
        body["messages"] = optimized_messages

        # --- Streaming: inject stream_options (mirrors handler lines ~2026-2029) ---
        # The live handler injects this BEFORE forwarding bytes, so it appears in the
        # outbound body captured by the golden fixtures' CapturingTransport.
        if body.get("stream"):
            if "stream_options" not in body:
                body["stream_options"] = {"include_usage": True}
            elif isinstance(body.get("stream_options"), dict):
                body["stream_options"]["include_usage"] = True

        # --- Always canonical serialization ---
        # The live handler never passes ``original_body_bytes`` or
        # ``body_mutated=False`` to ``_retry_request`` / ``_stream_response``,
        # so the effective behaviour is always ``serialize_body_canonical``.
        # We replicate that directly — no passthrough path for OpenAI chat.
        outbound_bytes = serialize_body_canonical(body)

        compressed = outbound_bytes != original_body_bytes
        bytes_saved = max(0, len(original_body_bytes) - len(outbound_bytes))

        return RequestDecision(
            body=outbound_bytes,
            telemetry=ResponseTelemetry(
                bytes_saved=bytes_saved,
                compressed=compressed,
            ),
        )

    # ── Legacy fake-pipeline path (Chunks 1-2) ────────────────────────────────

    def _on_request_fake_pipeline(
        self, ctx: RequestContext, pipeline: CompressionPipeline
    ) -> RequestDecision:
        """Simplified path used by Chunk 2 tests with fake pipelines.

        Preserves the original Chunk 2 semantics exactly so those tests
        continue passing.
        """
        # Parse body — raises loudly on malformed JSON
        try:
            body: dict[str, Any] = json.loads(ctx.raw_body)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"on_request: unparseable JSON body for "
                f"provider={ctx.provider!r}, flavor={ctx.flavor!r}: {exc}"
            ) from exc

        messages: list[dict[str, Any]] = body.get("messages") or []
        model: str = body.get("model", "")

        # Classify auth mode (pure, <10us, never raises)
        auth_mode = classify_auth_mode(ctx.headers_view)

        # Decision: should we compress?
        decision = CompressionDecision.decide(
            headers=ctx.headers_view,
            config=self._config,
            usage_reporter=self._usage_reporter,
            messages=messages,
        )

        if not decision.should_compress:
            # Return raw body BYTE-IDENTICAL — same object, no re-serialization.
            # This is load-bearing for prefix-cache safety.
            return RequestDecision(
                body=ctx.raw_body,
                telemetry=ResponseTelemetry(compressed=False),
            )

        # Resolve per-auth-mode compression policy
        policy = resolve_policy(auth_mode)

        # Delegate to the injected pipeline
        result = pipeline.apply(
            messages,
            model,
            compression_policy=policy,
        )

        # Reconstruct body with compressed messages
        body["messages"] = result.messages
        compressed_bytes = json.dumps(body).encode()

        bytes_saved = max(0, len(ctx.raw_body) - len(compressed_bytes))
        tokens_in = getattr(result, "tokens_before", 0)
        tokens_out = getattr(result, "tokens_after", 0)

        return RequestDecision(
            body=compressed_bytes,
            telemetry=ResponseTelemetry(
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                bytes_saved=bytes_saved,
                compressed=True,
                ccr_fired=False,
            ),
        )

    # ── Response hooks (Chunk 2 stubs — Chunk 3+ extends these) ─────────────

    def on_response(self, ctx: RequestContext, raw_response: bytes) -> bytes:
        """Forward the upstream response unchanged.

        Chunk 3 will extend this with CCR proactive-expansion injection and
        token telemetry parsing.
        """
        return raw_response

    def on_response_chunk(self, sc: StreamContext, chunk: bytes) -> bytes:
        """Forward a streaming chunk unchanged.

        Chunk 3 will add SSE parsing for streaming token telemetry.
        """
        return chunk

    def on_response_end(self, sc: StreamContext, outcome: Any) -> ResponseTelemetry:
        """Finalize a streaming session and return its telemetry.

        Safe to call on normal completion OR abort (``outcome`` may be an
        Exception or ``None``).  Chunk 3 will accumulate streaming token
        counts here.
        """
        return ResponseTelemetry()


# ── Private helpers (mirrors static methods on AnthropicHandlerMixin) ─────────


def _append_context_to_latest_non_frozen_user_turn(
    messages: list[dict[str, Any]],
    context_text: str,
    *,
    frozen_message_count: int,
) -> list[dict[str, Any]]:
    """Append context to the first text block of the latest non-frozen user turn.

    Direct port of ``AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn``.
    Used by both the memory injection step (4.2c) and the CCR proactive-expansion
    step (4.2b) in ``_on_request_anthropic_real``.

    Returns the input list unchanged if no eligible user text block exists
    (last message is an assistant turn, a tool result, or has no text block).
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
        new_content: list[dict[str, Any]] = []
        appended = False
        for block in content:
            if not appended and isinstance(block, dict) and block.get("type") == "text":
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


def _resolve_ccr_workspace(
    headers_view: Mapping[str, str],
    body: dict[str, Any],
) -> tuple[str, str | None]:
    """Resolve (workspace_key, workspace_label) for CCR scoping.

    Adapted from ``AnthropicHandlerMixin._resolve_ccr_workspace`` to work
    from ``(headers_view, body)`` instead of a FastAPI ``Request`` object.
    The engine has no FastAPI request; all header/body signals are available
    through ``ctx.headers_view`` and the parsed ``body`` dict respectively.

    Tier order is identical to the handler:
      x-headroom-project-id → x-headroom-cwd → CLI override (N/A here,
      project_root_override=None) → cwd: line in system prompt.

    Returns ``("", None)`` on any failure — fail-closed, not silent.
    A warning is logged so the absence is observable.
    """
    from headroom.memory.storage_router import (
        ProjectResolver,
    )
    from headroom.memory.storage_router import (
        RequestContext as _StorageCtx,
    )
    from headroom.memory.storage_router import (
        extract_system_prompt as _extract_sys_prompt,
    )

    try:
        storage_ctx = _StorageCtx(
            headers=dict(headers_view),
            system_prompt=_extract_sys_prompt(body),
            base_user_id=dict(headers_view).get("x-headroom-user-id", ""),
            project_root_override=None,
        )
        ident = ProjectResolver().resolve(storage_ctx)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "event=ccr_workspace_resolve_failed error=%s; "
            "CCR proactive expansion disabled for this request",
            exc,
        )
        return "", None

    if ident is None:
        return "", None
    return ident[0], ident[1]


def _strict_previous_turn_frozen_count(
    messages: list[dict[str, Any]],
    base_frozen_count: int,
) -> int:
    """Freeze all prior turns; only the final turn is mutable.

    Direct port of ``AnthropicHandlerMixin._strict_previous_turn_frozen_count``.
    """
    if not messages:
        return base_frozen_count
    final_idx = len(messages) - 1
    if messages[final_idx].get("role") == "user":
        return max(base_frozen_count, final_idx)
    return len(messages)


def _extract_cache_stable_delta(
    current_messages: list[dict[str, Any]],
    previous_original_messages: list[dict[str, Any]] | None,
    previous_forwarded_messages: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Return (stable_forwarded_prefix, appended_delta_messages) when safe.

    Direct port of ``AnthropicHandlerMixin._extract_cache_stable_delta``.
    """
    if not previous_original_messages or previous_forwarded_messages is None:
        return None
    prefix_len = len(previous_original_messages)
    if len(current_messages) < prefix_len:
        return None
    if current_messages[:prefix_len] != previous_original_messages:
        return None
    return (
        copy.deepcopy(previous_forwarded_messages),
        copy.deepcopy(current_messages[prefix_len:]),
    )
