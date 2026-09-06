"""``RequestOutcome``: the canonical value type for "what happened during
one completed proxy request."

Per the P0 audit (``docs/superpowers/specs/P0-proxy-pipeline-audit.md``),
18 ``metrics.record_request`` call sites across four handler files
disagreed on argument shape — 9 of 18 omitted ``cached=``, 7 of 18
omitted ``attempted_input_tokens=``, only 4 sites emitted a structured
PERF log at all. This module is the structural fix: every handler
converges on building a :class:`RequestOutcome` at end-of-request and
hands it to :func:`emit_request_outcome` (also exposed as
:meth:`HeadroomProxy._record_request_outcome`), which owns the four
downstream effects (Prometheus, cost tracker, request logger, PERF
log).

Note: this is **output unification, not input unification**. Provider
APIs (Anthropic ``/v1/messages``, OpenAI Responses WS, Gemini
``generateContent``, Bedrock, Vertex) stay wildly different — the proxy
talks each upstream in its native dialect. This dataclass standardises
only the *observation* about a completed request. Provider-specific
concepts (Anthropic's 5m/1h cache TTL splits, OpenAI's
inferred-write flag, Gemini's read-only cache count) live as optional
fields with neutral defaults; handlers populate what their provider
actually reports.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from headroom.proxy.tool_schema_savings_policy import (
    headline_tokens_saved,
    tool_schema_saved_from_tags,
)

logger = logging.getLogger("headroom.proxy")


@dataclass(frozen=True)
class RequestOutcome:
    """Immutable, value-equal snapshot of a completed request.

    Construction policy: every field that downstream consumers read MUST
    be either required (no default) or have a neutral default that makes
    the consumer's behaviour identical to "field not present". This keeps
    the contract honest — a handler that forgets a field doesn't silently
    produce wrong metrics; it produces zeros, which the dashboard can
    surface as a missing-data condition (P3 follow-up).
    """

    # ── Identity ──────────────────────────────────────────────────────
    request_id: str
    provider: str
    model: str

    # ── Tokens (required — every site has these) ──────────────────────
    # original_tokens: pre-compression request size, for `tok_before`
    # optimized_tokens: post-compression size actually forwarded, for
    #     ``tok_after``. MUST be counted with the SAME tokenizer as
    #     ``original_tokens`` — every derived quantity is a delta between the
    #     two (``tokens_saved``, ``tokens_inflated``, ``attempted_input_tokens``,
    #     and the beacon's ``eligible_pct`` / ``yield_pct``), so mixing scales
    #     silently corrupts all of them. Handlers used to pass the provider's
    #     ``usage.prompt_tokens`` here because it also fed billing; that made
    #     ``tok_after`` a provider count against a locally-estimated
    #     ``tok_before``. On a gpt-4o-mini turn where our estimator undercounted
    #     by 2 tokens that shipped as ``eligible_pct: 120`` — a structurally
    #     impossible ratio — plus a phantom ``tok_inflated``. Provider-reported
    #     input now lives in ``provider_input_tokens``.
    # provider_input_tokens: the provider's own prompt-token count for this
    #     request, when it reported one (0 otherwise). This is the billed
    #     quantity, so cost and volume totals use it in preference to
    #     ``optimized_tokens``. Kept separate precisely because it is on the
    #     provider's tokenizer scale and must never be differenced against
    #     ``original_tokens``.
    # output_tokens: response tokens from upstream
    # tokens_saved: original - optimized (or 0 if compression bypassed)
    # attempted_input_tokens: denominator for active-savings-percent.
    #     The compressible portion only — excludes user messages, system
    #     prompts, prior assistant turns, frozen prefix bytes. This is the
    #     field 7 of 18 audit sites forgot to pass, collapsing
    #     ``active_savings_percent`` to 0 (#454 / #455).
    original_tokens: int
    optimized_tokens: int
    output_tokens: int
    tokens_saved: int
    attempted_input_tokens: int
    # Optional correlation id for the human-readable PERF line. Most requests
    # use ``request_id`` for both storage identity and log correlation. A
    # long-lived WebSocket session is different: each emitted feed row needs a
    # unique request id, while operators still need every line from the socket
    # under one greppable session prefix.
    perf_request_id: str | None = None
    # Optional so the 18 existing emit sites need no change: a handler that has
    # no provider count (or whose optimized_tokens is already provider-scaled)
    # leaves it 0 and billing falls back to optimized_tokens, exactly as before.
    provider_input_tokens: int = 0

    # ── Cache (provider-agnostic; unused fields stay 0) ───────────────
    # Anthropic populates all five (read + write + 5m + 1h + uncached).
    # OpenAI populates read + inferred-write + uncached, and sets
    # ``cache_inferred=True`` so the dashboard can warn that the write
    # column is an estimate rather than an upstream-reported counter.
    # Gemini populates read only.
    # Bedrock mirrors Anthropic (it forwards Anthropic-shape usage).
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0
    uncached_input_tokens: int = 0
    cache_inferred: bool = False
    # Response-cache hit (Headroom's own semantic cache served the
    # response from a prior call — completely distinct from
    # upstream-prompt-cache `cache_read_tokens`). True means the proxy
    # never reached the provider at all. Used to drive the
    # Prometheus ``cached`` counter and dashboard "response cache" row.
    from_response_cache: bool = False

    # ── Output composition ────────────────────────────────────────────
    # ``output_tokens`` pools two quantities that different levers move in
    # opposite directions: reasoning-effort routing cuts thinking, verbosity
    # steering cuts visible text. Summed, neither can be attributed — a turn
    # whose thinking dropped 4,000 tokens while its prose grew 200 is
    # indistinguishable from one where nothing happened.
    #
    # ``None`` means "we could not tell", which is NOT zero: Anthropic reports
    # no thinking count at all, so claiming 0 there would assert that no
    # thinking occurred and corrupt any average over mixed-provider traffic.
    # ``thinking_inferred`` marks a count Headroom derived by tokenizing the
    # response's thinking blocks rather than one the provider reported — the
    # same distinction ``cache_inferred`` draws on the input side.
    thinking_tokens: int | None = None
    thinking_inferred: bool = False
    # Why the model stopped. ``"max_tokens"`` (Anthropic) / ``"length"``
    # (OpenAI) mean a token ceiling cut the response off — the one unambiguous,
    # provider-supplied feedback signal an adaptive ceiling can run a control
    # loop on. The verbosity signals (user interrupted, user replied too fast to
    # have read it) need transcript inference, which is why the verbosity
    # controller was written and never wired to anything.
    stop_reason: str | None = None
    # 0-based position of this turn in its conversation. An output token is
    # billed once at the output rate and again as input on every later turn,
    # so what a wasted token actually costs depends on how much conversation
    # is left — which cannot be computed without knowing where we are in it.
    turn_index: int = 0

    # Upstream HTTP status for this request (200 on success or response-cache
    # hit). When >= 500 (e.g. a 529 Overloaded returned after retry
    # exhaustion) the funnel records a failed request instead of feeding the
    # savings/cost stats, so an upstream failure can't inflate save-rate.
    status_code: int = 200

    # ── Timing ────────────────────────────────────────────────────────
    # total_latency_ms: wall-clock end-to-end for this request
    # overhead_ms: time spent in compression dispatch only (subset of total)
    # ttfb_ms: time to first upstream byte for streaming paths; 0 for
    #     non-streaming or when unmeasured (no None — convention is 0)
    # pipeline_timing: optional per-stage breakdown surfaced on dashboards
    total_latency_ms: float = 0.0
    overhead_ms: float = 0.0
    ttfb_ms: float = 0.0
    pipeline_timing: dict[str, float] | None = None

    # ── Transforms + diagnostics ──────────────────────────────────────
    # transforms_applied: tuple (immutable) of every transform that ran.
    #     RequestLog still wants list[str]; the funnel converts at the
    #     boundary.
    # waste_signals: per-router signals captured during routing (counts
    #     of skipped vs applied units etc.); dashboards summarise.
    # num_messages: messages in the original request (for ``msgs=N`` in
    #     PERF), counted from body.input/body.messages.
    # turn_id: stable hash of the conversation prefix; used by
    #     dashboards to group multi-turn sessions.
    # request_messages: only populated when ``config.log_full_messages``
    #     is enabled (off by default — message bodies are sensitive).
    # tags: client-provided routing/identification tags.
    # client: identified harness driving the request (codex /
    #     claude-code / aider / cursor / opencode / zed / ...).
    #     ``None`` when neither the ``X-Client`` header nor the
    #     User-Agent matched a known harness. Populated by handlers
    #     via :func:`headroom.proxy.auth_mode.classify_client`. The
    #     funnel surfaces this in the PERF log (``client=X``) and
    #     copies it into ``RequestLog.tags["client"]`` so dashboards
    #     can slice by harness without a separate column. This is the
    #     one-field-add that proves the refactor pays out: per-
    #     harness visibility appears across EVERY handler with zero
    #     new bookkeeping at the call sites.
    transforms_applied: tuple[str, ...] = ()
    waste_signals: dict[str, int] | None = None
    num_messages: int = 0
    turn_id: str | None = None
    request_messages: list[dict[str, Any]] | None = None
    # Post-compression messages actually sent upstream, paired with
    # ``request_messages`` (pre-compression) so consumers can diff the two.
    # Only populated when a caller threads in the pre-compression snapshot
    # (``original_messages``); otherwise ``request_messages`` carries the sent
    # body for backward compatibility and this stays ``None``.
    compressed_messages: list[dict[str, Any]] | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    client: str | None = None
    project: str | None = None

    # ── Derived (computed once, no caching needed — properties are cheap) ─

    @property
    def cache_hit(self) -> bool:
        """True iff EITHER upstream reported a cache read OR the response
        was served from Headroom's own response cache.

        Two distinct concepts collapsed into one observable boolean for
        downstream consumers (Prometheus ``cached`` counter, RequestLog
        ``cache_hit`` flag). The dataclass tracks them separately so
        dashboards can split them; the derived property unifies them.

        Pre-refactor 9 of 18 sites hardcoded this to False — this property
        makes "I forgot to compute it" structurally impossible.
        """
        return self.cache_read_tokens > 0 or self.from_response_cache

    @property
    def visible_output_tokens(self) -> int | None:
        """Output tokens excluding thinking, or ``None`` when the split is
        unknown.

        The quantity verbosity steering actually targets. Callers must handle
        ``None`` rather than defaulting it to ``output_tokens``: on a provider
        that reports no thinking count, treating the whole output as visible
        would credit steering with reductions that reasoning-effort routing
        produced.

        Clamped at zero because an inferred count comes from Headroom's
        tokenizer while ``output_tokens`` is on the provider's scale; the two
        can disagree by a token or two on a short response.
        """
        if self.thinking_tokens is None:
            return None
        return max(0, self.output_tokens - self.thinking_tokens)

    @property
    def cache_hit_pct(self) -> int:
        """Cache read share of (read + write), rounded to int percent.

        Returns 0 when neither read nor write fired (a request that did no
        cache work; distinguishing this from "0% hit rate on real cache
        work" requires looking at the absolute values, not the ratio).
        """
        denom = self.cache_read_tokens + self.cache_write_tokens
        if denom <= 0:
            return 0
        return round(self.cache_read_tokens / denom * 100)

    @property
    def savings_pct(self) -> float:
        """Compression savings as a fraction of the original request size.

        This is the proxy-side ratio: ``tokens_saved / original_tokens``.
        The dashboard headline "active savings percent" uses a different
        ratio (``tokens_saved / attempted_input_tokens``) — see the
        Prometheus metric for the active calculation.
        """
        if self.original_tokens <= 0:
            return 0.0
        return self.tokens_saved / self.original_tokens * 100.0

    @property
    def tokens_inflated(self) -> int:
        """Tokens the forwarded request grew by, if it ended up larger.

        ``tokens_saved`` is clamped at zero, so a request that leaves the
        proxy *bigger* than it arrived is indistinguishable from one the
        proxy simply could not compress: both report ``tok_saved=0``. That
        ambiguity hides real regressions — anything that adds to the body
        after compression (proactive context expansion, memory injection)
        can outweigh the compression it sits on top of and still look like
        a neutral turn.

        Report the swallowed amount alongside it so the two cases are
        distinguishable. This is diagnostic only: it deliberately does not
        feed ``tokens_saved`` or ``attempted_input_tokens``, because
        ``attempted_input_tokens = optimized_tokens + tokens_saved`` is a
        size, not a signed delta, and because injection paths already book
        their own cost through the retrieval-drawback channel — letting a
        negative land here too would count it twice.
        """
        return max(0, self.optimized_tokens - self.original_tokens)

    @classmethod
    def from_stream(
        cls,
        *,
        body: dict[str, Any],
        provider: str,
        model: str,
        request_id: str,
        original_tokens: int,
        optimized_tokens: int,
        output_tokens: int,
        tokens_saved: int,
        transforms_applied: list[str] | tuple[str, ...],
        total_latency_ms: float,
        overhead_ms: float,
        tags: dict[str, str] | None,
        client: str | None,
        log_full_messages: bool = False,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        cache_write_5m_tokens: int = 0,
        cache_write_1h_tokens: int = 0,
        uncached_input_tokens: int = 0,
        cache_inferred: bool = False,
        thinking_tokens: int | None = None,
        thinking_inferred: bool = False,
        stop_reason: str | None = None,
        turn_index: int = 0,
        ttfb_ms: float = 0.0,
        pipeline_timing: dict[str, float] | None = None,
        waste_signals: dict[str, int] | None = None,
        original_messages: list[dict] | None = None,
    ) -> RequestOutcome:
        """Construct an outcome from the locals available at streaming
        finalize. Three streaming finalizers
        (``_finalize_stream_response``, ``_stream_response_bedrock``,
        ``_stream_openai_via_backend``) each duplicated the same body- and
        config-derived fields inline. This classmethod is the single
        construction point so derivation logic can't drift apart again.

        Centralises six derivations:

          * ``attempted_input_tokens = optimized_tokens + tokens_saved``
          * ``num_messages = len(body["messages"])``
          * ``request_messages`` conditional on ``log_full_messages``
          * ``turn_id`` via ``compute_turn_id`` — pre-refactor only the
            Bedrock site computed this; sites 1 and 3 silently dropped it,
            breaking multi-turn-session grouping on Anthropic-SSE and
            OpenAI-via-backend traffic
          * ``transforms_applied`` list → tuple (frozen-dataclass contract)
          * ``tags or {}`` normalization
        """
        from headroom.proxy.helpers import compute_turn_id

        request_items = body.get("messages")
        turn_messages = request_items
        if request_items is None:
            request_items = body.get("contents", [])
            if isinstance(request_items, list):
                turn_messages = []
                for item in request_items:
                    if not isinstance(item, dict):
                        continue
                    parts = item.get("parts")
                    text = ""
                    if isinstance(parts, list):
                        text = "\n".join(
                            str(part.get("text"))
                            for part in parts
                            if isinstance(part, dict) and part.get("text")
                        )
                    role = "assistant" if item.get("role") == "model" else "user"
                    turn_messages.append({"role": role, "content": text})
        system = body.get("system")
        if system is None:
            system = body.get("systemInstruction")

        # ``request_items`` is ``body["messages"]`` (or ``body["contents"]``
        # for Gemini, falling back to ``[]``) — the post-compression list the
        # caller already mutated in place before finalize. When a
        # caller threads in ``original_messages`` (the pre-compression
        # snapshot), log it as ``request_messages`` and the sent body as
        # ``compressed_messages`` so the two sides stay diffable. Callers that
        # don't thread it in (gemini ``contents``, OpenAI-via-backend) keep the
        # prior behaviour: sent body under ``request_messages``, no compressed
        # side. Both sides share the ``log_full_messages`` gate.
        if not log_full_messages:
            log_request_messages = None
            log_compressed_messages = None
        elif original_messages is not None:
            log_request_messages = original_messages
            log_compressed_messages = request_items
        else:
            log_request_messages = request_items
            log_compressed_messages = None

        return cls(
            request_id=request_id,
            provider=provider,
            model=model,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            output_tokens=output_tokens,
            tokens_saved=tokens_saved,
            attempted_input_tokens=optimized_tokens + tokens_saved,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_write_5m_tokens=cache_write_5m_tokens,
            cache_write_1h_tokens=cache_write_1h_tokens,
            uncached_input_tokens=uncached_input_tokens,
            cache_inferred=cache_inferred,
            thinking_tokens=thinking_tokens,
            thinking_inferred=thinking_inferred,
            stop_reason=stop_reason,
            turn_index=turn_index,
            total_latency_ms=total_latency_ms,
            overhead_ms=overhead_ms,
            ttfb_ms=ttfb_ms,
            pipeline_timing=pipeline_timing,
            transforms_applied=tuple(transforms_applied),
            waste_signals=waste_signals,
            num_messages=len(request_items) if isinstance(request_items, list) else 0,
            turn_id=compute_turn_id(model, system, turn_messages),
            tags=tags or {},
            client=client,
            request_messages=log_request_messages,
            compressed_messages=log_compressed_messages,
        )


# ── The funnel ───────────────────────────────────────────────────────


async def emit_request_outcome(handler: Any, outcome: RequestOutcome) -> None:
    """Single funnel for per-request bookkeeping. The contract.

    Owns the downstream effects in canonical order:

      0. ``telemetry.session.record_outcome(...)`` — anonymous session beacon
         (opt-in, no-op unless ``HEADROOM_TELEMETRY`` is on). Runs ahead of the
         5xx guard below so session error rates see upstream failures.
      1. ``handler.metrics.record_request(...)`` — Prometheus / SavingsTracker
      2. ``handler.cost_tracker.record_tokens(...)`` — cost dashboard
         (skipped when cost_tracker is None, i.e. ``--no-cost``)
      3. ``handler.logger.log(RequestLog(...))`` — per-request log feed
         (skipped when logger is None, i.e. ``--no-request-logging``)
      4. structured PERF log line — consumed by ``headroom perf``

    A rejected outcome (``status_code >= 400``, e.g. a 429 rate limit or a 529
    surfaced after retry exhaustion) short-circuits before effects 1-4: it
    records the request under the counter that names what happened and returns,
    so a turn the provider never billed cannot feed the success stats.

    Takes the handler as a free argument rather than ``self`` so this
    function is callable from:
    * ``HeadroomProxy._record_request_outcome`` (production)
    * any test dummy that has the three required attributes
      (``metrics``, ``cost_tracker``, optionally ``logger``)
    * any provider handler mixin

    The handler argument is structurally typed (duck-typed); no formal
    Protocol — the requirement is simply that ``handler.metrics`` exists
    and is awaitable-compatible. We could lift this to a typing.Protocol
    if/when another contract surface emerges, but YAGNI.
    """
    from headroom.copilot_auth import consume_request_routed_to_copilot
    from headroom.proxy.cost import _summarize_transforms
    from headroom.proxy.models import RequestLog
    from headroom.proxy.project_context import get_current_project
    from headroom.proxy.savings_attribution import (
        encode,
        from_tags,
        public_tags,
        timings_from_tags,
    )
    from headroom.telemetry.session import record_outcome

    # GitHub Copilot: requests routed to the Copilot API travel on the OpenAI or
    # Anthropic wire, so the handlers stamp the wire provider. Relabel to
    # "copilot" here — the single outcome funnel — so the dashboard shows the
    # real upstream instead of "openai"/"anthropic". Keyed on the per-request
    # flag set in build_copilot_upstream_url; never touches non-Copilot traffic.
    # Done before the 5xx guard so a failed Copilot request is attributed too.
    # consume_* reads AND clears the flag (called unconditionally via short-circuit
    # order) so it cannot leak onto a later outcome in the same execution context.
    if consume_request_routed_to_copilot() and outcome.provider in ("openai", "anthropic"):
        import dataclasses

        outcome = dataclasses.replace(outcome, provider="copilot")

    # 0. Anonymous session beacon. Opt-in and a no-op unless HEADROOM_TELEMETRY
    #    is explicitly on, in which case it folds this outcome into an in-memory
    #    per-session aggregate and POSTs one content-free event when the session
    #    goes idle (off-thread — see telemetry.session.post_session_event).
    #
    #    Placed before the 5xx short-circuit, for the same reason the Copilot
    #    relabel above is: a session's failure count has to see upstream
    #    failures, and per-session error rate is exactly the signal that shows a
    #    provider going flaky for real users. Everything below this point is
    #    success-only bookkeeping.
    #
    #    Synchronous but allocation-light, and swallows its own exceptions — the
    #    beacon must never add latency to, or take down, the request path.
    record_outcome(outcome)

    # A rejected turn must not feed the savings/cost/log success stats; that
    # would let a request the provider never billed inflate the save-rate.
    # Record it under the counter that names what happened, and stop.
    #
    # This covers 4xx as well as 5xx. A 4xx is an error the PROXY served but the
    # PROVIDER did not: nothing was generated, so nothing was billed, so
    # compression on that turn saved exactly nothing. Counting it anyway is not a
    # rounding error — measured on a real session where 143 of 300 turns came
    # back 429 ("Usage credits are required for fast mode"), 46.5% of the
    # headline `total_saved` was compression on turns Anthropic rejected, and
    # `requests.rate_limited` still read 0 because only Headroom's own limiter
    # ever incremented it. The 60M `tokens.input` and the $8.90 compression
    # savings shown against a $2.26 measured spend came from the same place.
    #
    # 429 goes to record_rate_limited rather than record_failed: an upstream rate
    # limit is the one 4xx a user is expected to act on (back off, raise a cap),
    # and folding it into a generic failure count hides exactly that. Both
    # counters are already exported and neither feeds savings.
    if outcome.status_code >= 400:
        if outcome.status_code == 429:
            await handler.metrics.record_rate_limited(provider=outcome.provider)
        else:
            await handler.metrics.record_failed(provider=outcome.provider)
        return

    # Outcome stage: hand the meter's reading to any pipeline extension that
    # adapts to it — a ceiling that was hit, an effort setting that did or did
    # not pay off. The snapshot is frozen, so an extension can learn from the
    # measurement without being able to rewrite it: extensions declare what
    # they did, the core records what happened, and attribution is the core's
    # arithmetic over both.
    #
    # Best-effort and last in line: a handler without a pipeline manager, or an
    # extension that raises, must not affect the response or the stats below.
    _pipeline = getattr(handler, "pipeline_extensions", None)
    if _pipeline is not None and getattr(_pipeline, "enabled", False):
        try:
            from headroom.pipeline import OutcomeSnapshot, PipelineStage

            _pipeline.emit(
                PipelineStage.OUTCOME_OBSERVED,
                operation="proxy.outcome",
                request_id=outcome.request_id,
                provider=outcome.provider,
                model=outcome.model,
                outcome=OutcomeSnapshot(
                    request_id=outcome.request_id,
                    provider=outcome.provider,
                    model=outcome.model,
                    output_tokens=outcome.output_tokens,
                    thinking_tokens=outcome.thinking_tokens,
                    thinking_inferred=outcome.thinking_inferred,
                    stop_reason=outcome.stop_reason,
                    input_tokens=outcome.provider_input_tokens or outcome.optimized_tokens,
                    cache_read_tokens=outcome.cache_read_tokens,
                    cache_write_tokens=outcome.cache_write_tokens,
                    turn_index=outcome.turn_index,
                    transforms_applied=tuple(str(t) for t in (outcome.transforms_applied or ())),
                ),
            )
        except Exception:  # noqa: BLE001 - telemetry must never break a response
            logger.debug("OUTCOME_OBSERVED emit failed", exc_info=True)

    # Output-shaping savings ledger (counterfactual estimator). The shaper
    # tags each request's (arm, stratum) onto ``transforms_applied``; feed the
    # observed output tokens to the recorder so it can produce an honest
    # reduction estimate. Best-effort: never let bookkeeping break a response.
    output_tokens_saved_est = 0
    if any(str(t).startswith("output_shaper:") for t in outcome.transforms_applied):
        try:
            from headroom.proxy.output_savings import get_recorder

            _rec = get_recorder()

            def _record_and_estimate() -> int:
                _rec.record_from_labels(outcome.transforms_applied, outcome.output_tokens)
                return _rec.estimate_request_savings(
                    outcome.transforms_applied, outcome.output_tokens
                )

            # Both calls take the recorder lock, and the every-Nth record also
            # does a full read-modify-write of the ledger file — run them
            # together off the event loop (#18) so a slow flush can't stall it.
            output_tokens_saved_est = await asyncio.to_thread(_record_and_estimate)
        except Exception:  # pragma: no cover - defensive
            pass

    # Project attribution: explicit outcome field wins, else the value the
    # HTTP middleware / WS accept captured from ``X-Headroom-Project``.
    project = outcome.project or get_current_project()

    # Tool-schema savings (deferral + turn-hook tool shrink) live in per-request
    # tags and never move tok_before/after; aggregate them into Metrics so the
    # session summary / cost summary / all-layers total can surface the layer.
    tool_search_saved = tool_schema_saved_from_tags(outcome.tags or {})
    savings_breakdown = from_tags(outcome.tags)

    # Stage timings contributed from OUTSIDE the handler, folded in here rather
    # than in each handler so every provider picks them up from one place.
    #
    # The handler's own timings win a name collision, which cannot happen while
    # extension stages carry the ``ext:`` prefix but is the safe way round if
    # that ever changes: a plugin must not be able to overwrite a measurement
    # the pipeline made of itself.
    extension_timing = timings_from_tags(outcome.tags)
    pipeline_timing = (
        {**extension_timing, **(outcome.pipeline_timing or {})}
        if extension_timing
        else outcome.pipeline_timing
    )

    # Billed input volume. Prefer the provider's own count where it reported one
    # — that is what the invoice charges for, and it is the number cache math is
    # already expressed in. Falls back to our local ``optimized_tokens`` when the
    # provider stayed silent (streaming without usage, or a non-reporting
    # backend), which is the pre-split behaviour for every handler.
    #
    # Deliberately NOT used for any delta: differencing this against
    # ``original_tokens`` mixes tokenizer scales. See the field docs.
    billed_input_tokens = outcome.provider_input_tokens or outcome.optimized_tokens

    # 1. Prometheus / SavingsTracker.
    await handler.metrics.record_request(
        provider=outcome.provider,
        model=outcome.model,
        input_tokens=billed_input_tokens,
        output_tokens=outcome.output_tokens,
        tokens_saved=outcome.tokens_saved,
        latency_ms=outcome.total_latency_ms,
        cached=outcome.cache_hit,
        overhead_ms=outcome.overhead_ms,
        ttfb_ms=outcome.ttfb_ms,
        pipeline_timing=pipeline_timing,
        waste_signals=outcome.waste_signals,
        cache_read_tokens=outcome.cache_read_tokens,
        cache_write_tokens=outcome.cache_write_tokens,
        cache_write_5m_tokens=outcome.cache_write_5m_tokens,
        cache_write_1h_tokens=outcome.cache_write_1h_tokens,
        uncached_input_tokens=outcome.uncached_input_tokens,
        attempted_input_tokens=outcome.attempted_input_tokens,
        output_tokens_saved=output_tokens_saved_est,
        project=project,
        client=outcome.client,
        tool_search_saved=tool_search_saved,
        local_input_tokens=outcome.optimized_tokens,
        savings_attribution=savings_breakdown,
    )

    # 2. Cost tracker (optional).
    cost_tracker = getattr(handler, "cost_tracker", None)
    if cost_tracker is not None:
        cost_tracker.record_tokens(
            outcome.model,
            outcome.tokens_saved,
            billed_input_tokens,
            cache_read_tokens=outcome.cache_read_tokens,
            cache_write_tokens=outcome.cache_write_tokens,
            cache_write_5m_tokens=outcome.cache_write_5m_tokens,
            cache_write_1h_tokens=outcome.cache_write_1h_tokens,
            uncached_tokens=outcome.uncached_input_tokens,
            cache_inferred=outcome.cache_inferred,
            output_tokens=outcome.output_tokens,
            # Same figure already handed to metrics.record_request above. The
            # cost tracker feeds the dashboard's per-model table, which read
            # compression only while its own headline counted both layers.
            tool_schema_saved=tool_search_saved,
        )

    # 3. Per-request log (optional). The ``client`` outcome field is
    #    copied into ``tags["client"]`` so the dashboard's existing
    #    tag-based filtering surfaces per-harness slicing for free —
    #    no new RequestLog column needed. The original ``outcome.tags``
    #    dict is not mutated (frozen dataclass + defensive copy).
    request_logger = getattr(handler, "logger", None)
    if request_logger is not None:
        log_tags = public_tags(outcome.tags)
        if outcome.client:
            log_tags["client"] = outcome.client
        if project:
            log_tags["project"] = project
        request_logger.log(
            RequestLog(
                request_id=outcome.request_id,
                # Request logs are consumed by browsers in arbitrary time zones.
                # Include the UTC offset so relative-age calculations represent
                # the same instant regardless of where the proxy runs.
                timestamp=datetime.now(timezone.utc).isoformat(),
                provider=outcome.provider,
                model=outcome.model,
                input_tokens_original=outcome.original_tokens,
                input_tokens_optimized=outcome.optimized_tokens,
                output_tokens=outcome.output_tokens,
                tokens_saved=outcome.tokens_saved,
                savings_percent=outcome.savings_pct,
                optimization_latency_ms=outcome.overhead_ms,
                total_latency_ms=outcome.total_latency_ms,
                tags=log_tags,
                cache_hit=outcome.cache_hit,
                cache_read_tokens=outcome.cache_read_tokens,
                cache_write_tokens=outcome.cache_write_tokens,
                uncached_input_tokens=outcome.uncached_input_tokens,
                transforms_applied=list(outcome.transforms_applied),
                savings_breakdown=savings_breakdown,
                waste_signals=outcome.waste_signals,
                request_messages=outcome.request_messages,
                compressed_messages=outcome.compressed_messages,
                turn_id=outcome.turn_id,
            )
        )

    # 4. Structured PERF log line. ``client=X`` is appended only when
    #    a harness was identified — keeps the unidentified-traffic
    #    line unchanged, and gives ``headroom perf --client X``
    #    parsers a clean key to filter on.
    client_part = f" client={outcome.client}" if outcome.client else ""
    # ``cached=1`` marks a turn answered from Headroom's own response cache.
    # Such a turn never contacts the upstream, so it has no outbound_request
    # line, no upstream stage timings, and all-zero token counters — which
    # made it indistinguishable in the logs from a turn that died silently
    # (#3019). Appended only on a hit, so every other PERF line is unchanged
    # and existing parsers keep working (``_parse_kv`` reads trailing
    # key=value pairs after ``transforms=`` the same way it reads
    # ``client=``).
    cached_part = " cached=1" if outcome.from_response_cache else ""
    # Tool-schema DEFERRAL savings can't move tok_before/after (those count messages
    # only), so a tool-heavy turn shows tok_saved=0 while genuinely saving thousands of
    # tool-definition tokens. `tool_saved` carries that component and `total_saved` is
    # the sum every user-facing surface reports — see tool_schema_savings_policy for why
    # compaction is already inside tok_saved and must not be added twice.
    tool_saved = tool_schema_saved_from_tags(outcome.tags or {})
    total_saved = headline_tokens_saved(outcome.tokens_saved, outcome.tags or {})
    encoded_savings = encode(savings_breakdown)
    logger.info(
        f"[{outcome.perf_request_id or outcome.request_id}] PERF "
        f"model={outcome.model} msgs={outcome.num_messages} "
        f"tok_before={outcome.original_tokens} tok_after={outcome.optimized_tokens} "
        f"tok_saved={outcome.tokens_saved} "
        f"tok_inflated={outcome.tokens_inflated} "
        f"tool_saved={tool_saved} "
        f"total_saved={total_saved} "
        f"cache_read={outcome.cache_read_tokens} cache_write={outcome.cache_write_tokens} "
        f"cache_hit_pct={outcome.cache_hit_pct} "
        f"opt_ms={outcome.overhead_ms:.0f} "
        f"total_ms={outcome.total_latency_ms:.0f} "
        f"tok_out={outcome.output_tokens} "
        f"ttfb_ms={outcome.ttfb_ms:.0f} "
        f"savings={encoded_savings} "
        f"transforms={_summarize_transforms(list(outcome.transforms_applied))}"
        f"{client_part}"
        f"{cached_part}"
    )
