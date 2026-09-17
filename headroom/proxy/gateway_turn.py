"""Gateway turn contract — one model turn split across two HTTP calls.

An external gateway (Kong, LiteLLM, a custom sidecar) owns routing, so Headroom
never sees the provider response on the ``/v1/compress`` path. The turn
contract closes that gap in two halves:

* the **request half** (``POST /v1/compress`` with a top-level ``gateway``
  object) returns the complete provider body, a ``turn_id``, routing advice
  and the *obligations* the gateway must redeem for the transforms Headroom
  applied;
* the **response half** (``POST /v1/compress/response``) takes the provider's
  answer under that ``turn_id`` and either finishes the turn (``done``) or
  asks the gateway to call the provider again (``redrive``).

The rule behind it is the proxy's own "no shrink without reload", moved to the
API boundary: a transform whose reload step needs the response runs only when
the gateway said it can re-drive (``gateway.can_redrive``). A body without a
``gateway`` key is *legacy mode* and this module is never consulted — the
handler keeps today's stateless/session behaviour byte-for-byte.

The subtle part is :class:`SuspendedHookRunner`. Turn hooks (``turn_hooks``)
re-drive the model by ``await call_model(messages)`` inside one coroutine. On
the proxy path that await is an upstream HTTP call; here the "call" is the
gateway's next request, so the hook coroutine is parked as an ``asyncio.Task``
between the two HTTP requests and handed the provider's JSON when it arrives.
The hook never learns the difference, which is what keeps the extension
packages unchanged.

Everything here is in-process state with the same replica-affinity requirement
as the replay cache and prefix tracker: the gateway must pin a session to one
Headroom replica (``gateway.session_affinity``); when it cannot, re-driving
transforms are switched off for the turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from headroom.proxy.outcome import RequestOutcome
from headroom.proxy.turn_hooks import (
    TurnContext,
    merge_provider_headers,
    registered_turn_hooks,
    run_request_hooks,
    run_response_hooks,
)

logger = logging.getLogger(__name__)

# Keys that steer Headroom itself and must never reach the provider.
BODY_CONTROL_KEYS: frozenset[str] = frozenset({"config", "gateway", "token_budget"})

OBLIGATION_REDRIVE = "redrive"
OBLIGATION_RELAY_USAGE = "relay_usage"

ANTHROPIC_BETA_HEADER = "anthropic-beta"
# Request headers the gateway may hand over; every other name is ignored.
GATEWAY_REQUEST_HEADERS: frozenset[str] = frozenset({ANTHROPIC_BETA_HEADER})

DEFAULT_TURN_TTL_SECONDS = 120.0
DEFAULT_MAX_PENDING_TURNS = 10_000
DEFAULT_MAX_REDRIVES = 8

# Strong references to fire-and-forget bookkeeping tasks (outcome recording
# for an expired turn, cancelling its parked coroutine). asyncio keeps only a
# weak reference to a task, so without this set a GC pass could drop the task
# before it ran.
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
# Capabilities                                                                 #
# --------------------------------------------------------------------------- #


class GatewayRequestError(ValueError):
    """A malformed ``gateway`` block; the handler answers 400 with the message."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = str(message)


@dataclass(frozen=True)
class GatewayCapabilities:
    """What the gateway declared it can do for this turn.

    ``redrive_allowed`` is the one the transforms gate on: a gateway that
    cannot pin the session to this replica cannot be asked to come back with
    the provider response, so ``session_affinity=False`` disables re-driving
    even when ``can_redrive`` is set.
    """

    can_redrive: bool = False
    can_relay_response: bool = False
    session_affinity: bool = True
    plugin_version: str | None = None
    # Client request headers the gateway forwarded (lowercase name -> value),
    # only the ones Headroom merges into the ``headers`` it hands back
    # (``anthropic-beta`` today). Never forwarded anywhere by Headroom.
    request_headers: dict[str, str] = field(default_factory=dict)

    @property
    def redrive_allowed(self) -> bool:
        return self.can_redrive and self.session_affinity

    @property
    def client_anthropic_beta(self) -> str | None:
        return self.request_headers.get(ANTHROPIC_BETA_HEADER)

    def echo(self) -> dict[str, bool]:
        """The capabilities Headroom honoured, echoed back to the gateway."""
        return {
            "can_redrive": self.can_redrive,
            "can_relay_response": self.can_relay_response,
            "session_affinity": self.session_affinity,
        }


_BOOL_FIELDS: tuple[tuple[str, bool], ...] = (
    ("can_redrive", False),
    ("can_relay_response", False),
    ("session_affinity", True),
)


def parse_gateway_block(body: dict[str, Any]) -> GatewayCapabilities | None:
    """``None`` in legacy mode (no ``gateway`` key); raises on a malformed block.

    Presence of the key is the mode switch, so ``{"gateway": {}}`` is gateway
    mode with every default. Unknown keys inside the block are ignored so a
    newer plugin can talk to an older Headroom.
    """
    if "gateway" not in body:
        return None
    raw = body["gateway"]
    if not isinstance(raw, dict):
        raise GatewayRequestError(f"gateway must be an object, got {type(raw).__name__}.")
    values: dict[str, Any] = {}
    for name, default in _BOOL_FIELDS:
        value = raw.get(name, default)
        if not isinstance(value, bool):
            raise GatewayRequestError(f"gateway.{name} must be a boolean, got {value!r}.")
        values[name] = value
    plugin_version = raw.get("plugin_version")
    values["plugin_version"] = str(plugin_version) if plugin_version is not None else None
    values["request_headers"] = _parse_request_headers(raw.get("request_headers"))
    return GatewayCapabilities(**values)


def _parse_request_headers(raw: Any) -> dict[str, str]:
    """``gateway.request_headers``: an object of header name -> string value.

    Names are lowercased; only the ones in :data:`GATEWAY_REQUEST_HEADERS` are
    kept (a gateway may hand over the whole client header map). A non-object
    block or a non-string value is a 400, like the boolean flags.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise GatewayRequestError(
            f"gateway.request_headers must be an object, got {type(raw).__name__}."
        )
    out: dict[str, str] = {}
    for name, value in raw.items():
        key = str(name).strip().lower()
        if key not in GATEWAY_REQUEST_HEADERS:
            continue
        if value is None:
            continue
        if not isinstance(value, str):
            raise GatewayRequestError(
                f"gateway.request_headers[{key!r}] must be a string, got {value!r}."
            )
        out[key] = value
    return out


def provider_for_model(model_name: str) -> str:
    """The tracker-provider inference ``/v1/compress`` already uses."""
    lowered = (model_name or "").lower()
    return "anthropic" if ("claude" in lowered or "anthropic" in lowered) else "openai"


# --------------------------------------------------------------------------- #
# Usage normalization                                                          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NormalizedUsage:
    """Provider usage reduced to the four counters Headroom acts on.

    ``None`` means the provider (or the gateway's relay) did not report the
    field — distinct from 0, which is an assertion. That distinction is what
    keeps an OpenAI relay with no cache fields from being read as "provider
    confirmed fully cold" and wiping the tracker's cached-prefix state.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read: int | None = None
    cache_write: int | None = None

    @property
    def has_cache_signal(self) -> bool:
        return self.cache_read is not None or self.cache_write is not None

    @property
    def should_apply(self) -> bool:
        """Same rule as ``/v1/usage``: a positive count, or BOTH fields at 0.

        A relay whose only present cache field is 0 (an OpenAI-mapped gateway
        naturally sends ``cached_tokens: 0`` with no write field) carries no
        positive signal and must not reset the tracker.
        """
        read, write = self.cache_read or 0, self.cache_write or 0
        if read + write > 0:
            return True
        return self.cache_read is not None and self.cache_write is not None

    def merged_with(self, other: NormalizedUsage) -> NormalizedUsage:
        """Sum the billed counters across re-drive rounds; cache fields come
        from the latest round that carried them (the last provider call is the
        one whose cache state describes the returned prefix)."""

        def _sum(a: int | None, b: int | None) -> int | None:
            if a is None and b is None:
                return None
            return (a or 0) + (b or 0)

        return NormalizedUsage(
            input_tokens=_sum(self.input_tokens, other.input_tokens),
            output_tokens=_sum(self.output_tokens, other.output_tokens),
            cache_read=other.cache_read if other.has_cache_signal else self.cache_read,
            cache_write=other.cache_write if other.has_cache_signal else self.cache_write,
        )


def _usage_int(usage: dict[str, Any], name: str) -> int | None:
    if name not in usage:
        return None
    value = usage[name]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise GatewayRequestError(f"usage.{name} must be an integer, got {value!r}.")
    if value < 0:
        raise GatewayRequestError(f"usage.{name} must be non-negative, got {value}.")
    return int(value)


def normalize_usage(usage: dict[str, Any]) -> NormalizedUsage:
    """Accept Anthropic, OpenAI chat, OpenAI Responses and Kong ``ai.proxy``
    usage shapes; raise ``GatewayRequestError`` (a ``ValueError``) on booleans or negative counts.

    Descends once into a nested ``usage`` key so a gateway can relay the whole
    provider response body, or Kong's log ``ai.proxy`` block, unchanged.
    """
    if not isinstance(usage, dict):
        raise GatewayRequestError("usage must be an object.")
    nested = usage.get("usage")
    if isinstance(nested, dict):
        usage = nested

    input_tokens = _usage_int(usage, "input_tokens")
    if input_tokens is None:
        input_tokens = _usage_int(usage, "prompt_tokens")
    output_tokens = _usage_int(usage, "output_tokens")
    if output_tokens is None:
        output_tokens = _usage_int(usage, "completion_tokens")

    cache_read = _usage_int(usage, "cache_read_input_tokens")
    cache_write = _usage_int(usage, "cache_creation_input_tokens")
    if cache_read is None:
        # OpenAI reports reads only (no write counter exists), nested under
        # prompt_tokens_details (chat), input_tokens_details (Responses) or —
        # in Kong's flattened log shape — at the top level.
        for details_key in ("prompt_tokens_details", "input_tokens_details"):
            details = usage.get(details_key)
            if isinstance(details, dict):
                cache_read = _usage_int(details, "cached_tokens")
                if cache_read is not None:
                    break
        if cache_read is None:
            cache_read = _usage_int(usage, "cached_tokens")

    return NormalizedUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read=cache_read,
        cache_write=cache_write,
    )


# --------------------------------------------------------------------------- #
# Suspended hook runner                                                        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Step:
    """One driver step: the hook wants the provider called again, or is done.

    ``response`` on a ``done`` step is the replacement the hook produced, or
    ``None`` when the gateway should forward the provider response it already
    holds (the latest one it posted).
    """

    kind: str  # "redrive" | "done"
    messages: list[dict[str, Any]] | None = None
    tools: Any = None
    response: dict[str, Any] | None = None

    @property
    def is_redrive(self) -> bool:
        return self.kind == "redrive"


class SuspendedHookRunner:
    """Run ``run_response_hooks`` as a task that parks at every ``call_model``.

    The hook coroutine and the HTTP handlers share one event loop (uvicorn's),
    so the handshake is two futures: ``_suspend`` resolves when the hook asks
    for a re-drive, ``_resume`` resolves when the gateway brings the answer.
    The driver waits on ``{task, _suspend}`` and reports whichever finished.

    Nothing here leaks: ``cancel()`` cancels the task AND awaits it, and the
    signal futures are plain futures rather than helper tasks, so a cancelled
    turn leaves ``asyncio.all_tasks()`` exactly as it found it.
    """

    def __init__(
        self,
        ctx: TurnContext,
        *,
        run: Callable[..., Any] | None = None,
    ) -> None:
        self._ctx = ctx
        self._run = run or run_response_hooks
        self._task: asyncio.Task[Any] | None = None
        self._suspend: asyncio.Future[None] | None = None
        self._resume: asyncio.Future[dict[str, Any]] | None = None
        # The provider response the gateway holds right now; a hook that hands
        # this exact object back has replaced nothing.
        self._latest: dict[str, Any] | None = None
        self.pending_request: list[dict[str, Any]] | None = None
        self.state = "idle"  # idle | running | awaiting_redrive | done
        self.rounds = 0
        self.last_step: Step | None = None

    async def _call_model(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        self.pending_request = messages
        self._resume = loop.create_future()
        self.state = "awaiting_redrive"
        if self._suspend is not None and not self._suspend.done():
            self._suspend.set_result(None)
        return await self._resume

    async def start(self, response: dict[str, Any]) -> Step:
        if self._task is not None:
            raise RuntimeError("SuspendedHookRunner.start() called twice")
        self._latest = response
        self.state = "running"
        self._task = asyncio.create_task(self._run(self._ctx, response, self._call_model))
        return await self._drive()

    async def resume(self, provider_response: dict[str, Any]) -> Step:
        if self.state != "awaiting_redrive" or self._resume is None:
            raise RuntimeError(f"SuspendedHookRunner.resume() in state {self.state!r}")
        self._latest = provider_response
        self.rounds += 1
        self.state = "running"
        fut, self._resume = self._resume, None
        fut.set_result(provider_response)
        return await self._drive()

    async def _drive(self) -> Step:
        assert self._task is not None
        loop = asyncio.get_running_loop()
        self._suspend = loop.create_future()
        try:
            await asyncio.wait({self._task, self._suspend}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            # The driver — the response-half handler — was cancelled (client
            # disconnect, shutdown). Left alone the hook task keeps running:
            # a gateway retry would then find it "awaiting_redrive" and feed
            # it the round-0 response as the answer to a re-drive it never
            # sent, or answer "done" and leave it parked forever. Cancel it
            # with the driver; the retry gets a plain "done" (hooks are
            # best-effort by contract).
            await self.cancel()
            raise
        if not self._task.done():
            # Parked inside call_model: the gateway must call the provider.
            self.last_step = Step("redrive", messages=self.pending_request, tools=self._ctx.tools)
            return self.last_step
        self._suspend = None
        self.state = "done"
        final: Any = None
        if not self._task.cancelled():
            try:
                final = self._task.result()
            except Exception:
                # run_response_hooks already swallows hook exceptions; this
                # catches a broken runner, which must not fail the turn.
                logger.exception("gateway turn: response hook runner failed")
        replaced = final if isinstance(final, dict) and final is not self._latest else None
        self.last_step = Step("done", response=replaced)
        return self.last_step

    async def cancel(self) -> None:
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if self._resume is not None and not self._resume.done():
            self._resume.cancel()
        self._resume = None
        self._suspend = None
        self.state = "done"


# --------------------------------------------------------------------------- #
# Pending turns                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class PendingTurn:
    """Everything the response half needs from the request half."""

    turn_id: str
    session_key: str | None
    session_id: str | None
    provider: str
    model: str
    body: dict[str, Any]
    ctx: TurnContext | None
    obligations: list[str]
    outcome_draft: RequestOutcome | None
    created_at: float
    deadline: float
    rounds: int = 0
    max_rounds: int = DEFAULT_MAX_REDRIVES
    runner: SuspendedHookRunner | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    client: str | None = None
    # Billed usage accumulated across every response-half call for the turn.
    usage: NormalizedUsage | None = None
    # A second response-half call while one is being driven would race the
    # parked coroutine; the handler answers 409 instead.
    in_flight: bool = False
    # config.mode="ccr" with markers inserted and re-drive allowed: the
    # response half answers headroom_retrieve calls itself (see arm_ccr_redrive).
    ccr_armed: bool = False

    @property
    def hooks_armed(self) -> bool:
        """The response half has something to run: turn hooks and/or CCR."""
        return OBLIGATION_REDRIVE in self.obligations and self.ctx is not None


class PendingTurnRegistry:
    """In-process, thread-safe registry of turns awaiting their response half.

    Lazy sweep on every ``register``/``get`` (the same pattern as the
    compression-cache registry) so no background task is needed. Expired or
    capacity-evicted turns are handed to ``on_expire`` — the proxy wires it to
    record the deferred outcome draft and cancel a parked coroutine. Never
    raises out of the callback.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float | None = None,
        max_entries: int | None = None,
        on_expire: Callable[[PendingTurn], None] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.ttl_seconds = (
            ttl_seconds
            if ttl_seconds is not None
            else _env_float("HEADROOM_GATEWAY_TURN_TTL_SECONDS", DEFAULT_TURN_TTL_SECONDS)
        )
        self.max_entries = (
            max_entries
            if max_entries is not None
            else _env_int("HEADROOM_GATEWAY_MAX_PENDING_TURNS", DEFAULT_MAX_PENDING_TURNS)
        )
        self._on_expire = on_expire
        self._clock = clock
        self._turns: OrderedDict[str, PendingTurn] = OrderedDict()
        self._lock = threading.Lock()
        self._expired = 0
        self._evicted = 0
        # Outcome drafts whose expiry ran with no event loop to record them
        # on; flushed by the next expiry that does (see on_turn_expired).
        self.orphaned_outcomes: list[RequestOutcome] = []

    def register(self, turn: PendingTurn) -> None:
        now = self._clock()
        with self._lock:
            dropped = self._sweep_locked(now)
            while len(self._turns) >= self.max_entries and self._turns:
                # Oldest first, but not one being driven right now: evicting
                # it cancels the hook mid-step and its replacement response is
                # lost (the sweep skips in-flight turns for the same reason).
                # Only when every turn is in flight does the bound win.
                oldest = next((t for t in self._turns.values() if not t.in_flight), None)
                if oldest is None:
                    _, oldest = self._turns.popitem(last=False)
                else:
                    del self._turns[oldest.turn_id]
                self._evicted += 1
                dropped.append(oldest)
            self._turns[turn.turn_id] = turn
        self._notify(dropped)

    def get(self, turn_id: str) -> PendingTurn | None:
        now = self._clock()
        with self._lock:
            dropped = self._sweep_locked(now)
            turn = self._turns.get(turn_id)
        self._notify(dropped)
        return turn

    def pop(self, turn_id: str) -> PendingTurn | None:
        with self._lock:
            return self._turns.pop(turn_id, None)

    def sweep(self, now: float | None = None) -> list[PendingTurn]:
        with self._lock:
            dropped = self._sweep_locked(self._clock() if now is None else now)
        self._notify(dropped)
        return dropped

    def _sweep_locked(self, now: float) -> list[PendingTurn]:
        expired = [t for t in self._turns.values() if t.deadline <= now and not t.in_flight]
        for turn in expired:
            self._turns.pop(turn.turn_id, None)
        self._expired += len(expired)
        return expired

    def _notify(self, dropped: list[PendingTurn]) -> None:
        if not dropped or self._on_expire is None:
            return
        for turn in dropped:
            try:
                self._on_expire(turn)
            except Exception:  # bookkeeping must never break a request
                logger.exception("gateway turn %s: on_expire callback failed", turn.turn_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._turns)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "pending": len(self._turns),
                "expired": self._expired,
                "evicted": self._evicted,
                "ttl_seconds": self.ttl_seconds,
                "max_entries": self.max_entries,
            }


def _spawn(coro: Any) -> asyncio.Task[Any] | None:
    """Schedule bookkeeping on the running loop; ``None`` when there is none."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        coro.close()
        return None
    task = loop.create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return task


def on_turn_expired(proxy: Any, turn: PendingTurn) -> None:
    """Registry ``on_expire`` hook: record the deferred outcome, drop the coroutine.

    The sweep runs inside ``register``/``get``, which only the HTTP handlers
    call, so in production there is always a running loop and both the
    outcome record (``proxy._record_request_outcome`` is async — it shields the
    funnel and appends to the ledger in a worker thread) and the coroutine
    cancel are scheduled as tasks on it. Without a loop (direct registry use
    from a thread or a test) the draft cannot be recorded safely — the funnel's
    async primitives belong to the server loop — so it is parked on the
    registry's ``orphaned_outcomes`` list and flushed by the next expiry that
    does run on the loop. Recording is the safe direction to fail: a dropped
    draft is a missing row, a misrecorded one is a wrong dashboard.
    """
    registry = getattr(proxy, "gateway_turns", None)
    orphans: list[RequestOutcome] | None = getattr(registry, "orphaned_outcomes", None)

    if turn.runner is not None:
        _spawn(turn.runner.cancel())

    draft = turn.outcome_draft
    turn.outcome_draft = None
    to_record: list[RequestOutcome] = []
    if orphans:
        to_record.extend(orphans)
        orphans.clear()
    if draft is not None:
        to_record.append(draft)
    for outcome in to_record:
        if _spawn(proxy._record_request_outcome(outcome)) is None:
            if orphans is not None:
                orphans.append(outcome)
            logger.warning(
                "gateway turn %s expired outside an event loop; outcome deferred",
                turn.turn_id,
            )


# --------------------------------------------------------------------------- #
# Request half                                                                 #
# --------------------------------------------------------------------------- #


def _count_tools_with(tokenizer: Any) -> Callable[[Any], int]:
    return lambda value: tokenizer.count_text(json.dumps(value, default=str)) if value else 0


def compact_gateway_tools(tools: Any) -> tuple[Any, list[str]]:
    """Deterministic, request-only tool schema compaction (chat-handler parity).

    Layer 1 (annotation stripping) always runs; layer 2 (description
    truncation) only behind ``HEADROOM_TOOL_DESC_MAX_CHARS``. The compaction
    module caches by tools digest and hands back the SAME cached list on every
    hit, so the result is deep-copied before a hook may mutate it in place —
    otherwise one turn's hook edits would leak into every later turn's
    "identical" compaction and break byte stability.

    Native tool-search deferral runs AFTER this, in :func:`defer_gateway_tools`,
    in the chat handler's order (compaction, deferral, hooks, history repair).
    """
    if not isinstance(tools, list) or not tools:
        return tools, []
    from headroom.proxy.tool_schema_compaction import (
        compact_tool_descriptions,
        compact_tools,
        tool_desc_max_chars,
    )

    transforms: list[str] = []
    payload, modified, _, _ = compact_tools({"tools": tools})
    if modified and payload.get("tools") is not None:
        tools = copy.deepcopy(payload["tools"])
        transforms.append("tool_schema_compaction")
    max_chars = tool_desc_max_chars()
    if max_chars > 0:
        payload, modified, _, _ = compact_tool_descriptions({"tools": tools}, max_chars)
        if modified and payload.get("tools") is not None:
            tools = copy.deepcopy(payload["tools"])
            if "tool_schema_compaction" not in transforms:
                transforms.append("tool_schema_compaction")
            transforms.append("tool_desc_compaction")
    return tools, transforms


# Emitted when a turn hook (the tool-search extension's native tier) deferred
# tool schemas: it writes core's ``tool_search_deferred_*`` tags and the label
# is derived from them here, so /stats, the outcome and the response half see
# exactly what the chat path's built-in deferral produces.
NATIVE_DEFERRAL_LABEL = "tool_search:native_deferral"
TOOL_SEARCH_REPAIR_LABEL = "router:tool_search_repair"


def repair_tool_search_history(
    messages: list[dict[str, Any]], tools: Any, *, provider: str
) -> tuple[list[dict[str, Any]], int]:
    """Tool-search history repair (chat-handler parity, request-only).

    Once a turn deferred, the client's transcript carries Anthropic's
    ``server_tool_use`` / ``tool_search_tool_result`` blocks for good, and
    upstream validates every ``tool_reference`` in them against THIS request's
    tools. A side request with a different, smaller tools array (Claude Code's
    Stop-hook evaluator, ``/compact``) would 400 — so, as on the chat path,
    the blocks the outbound tools cannot support are dropped. Deterministic
    (same request -> same output), and it must be the LAST stage that can
    invalidate a reference: after deferral and after the hooks.
    """
    if provider != "anthropic":
        return messages, 0
    from headroom.proxy.helpers import strip_unsupported_tool_search_blocks

    repaired, stripped = strip_unsupported_tool_search_blocks(messages, tools)
    if not stripped:
        return messages, 0
    return repaired, int(stripped)


@dataclass
class RequestTransformResult:
    messages: list[dict[str, Any]]
    tools: Any
    ctx: TurnContext | None
    transforms: list[str]
    redrive_armed: bool
    # Headers the gateway must SET on the provider request (name -> value);
    # empty when no transform needs one.
    headers: dict[str, str] = field(default_factory=dict)
    # Messages were rewritten by a stage the pipeline's own count never saw
    # (a hook fold, history repair): the caller must recount ``tokens_after``.
    messages_rewritten: bool = False


class RequestTransformer:
    """Tool compaction + request-side turn hooks for one gateway turn.

    Synchronous and CPU-bound (token counts, schema walks), so the handler
    runs it on the compression executor — in session mode after
    ``finalize_turn`` and before the tracker snapshots, because the hook
    output is what the gateway forwards and therefore what ``record_returned``
    must remember. A hook that is not a pure function of its input will bust
    the replayed prefix in session mode, exactly as it would on the proxy path.
    """

    def __init__(
        self,
        *,
        provider: str,
        model_name: str,
        tools: Any,
        config: Any,
        tags: dict[str, Any],
        caps: GatewayCapabilities,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.tools = tools
        self.config = config
        self.tags = tags
        self.caps = caps
        self.result: RequestTransformResult | None = None
        self._tokenizer: Any = None

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from headroom.tokenizers import get_tokenizer

            self._tokenizer = get_tokenizer(self.model_name)
        return self._tokenizer

    def count_messages(self, messages: list[dict[str, Any]], fallback: int) -> int:
        """Recount after a hook folded; the pipeline's own count is stale then.
        Same per-model tokenizer the derived pipelines use, so the scale
        matches ``tokens_before``. Falls back to the pipeline count."""
        try:
            return int(self._get_tokenizer().count_messages(messages))
        except Exception:
            return fallback

    @property
    def folded_messages(self) -> bool:
        return self.result is not None and self.result.messages_rewritten

    def run(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        transforms: list[str] = []
        headers: dict[str, str] = {}
        messages_rewritten = False
        tools, tool_transforms = compact_gateway_tools(self.tools)
        transforms.extend(tool_transforms)

        ctx: TurnContext | None = None
        redrive_armed = False
        hooks = registered_turn_hooks()
        count_messages: Callable[[list[dict[str, Any]]], int] | None = None
        count_tools: Callable[[Any], int] | None = None
        try:
            tokenizer = self._get_tokenizer()
            count_messages = tokenizer.count_messages
            count_tools = _count_tools_with(tokenizer)
        except Exception:
            logger.debug("gateway turn: tokenizer unavailable; hook savings uncounted")

        # Built even with no hook registered: the response half drives CCR
        # retrieval through the same context (messages + live tools).
        ctx = TurnContext(
            provider=self.provider,
            model=self.model_name,
            messages=messages,
            tools=tools,
            config=self.config,
            tags=self.tags,
            count_messages=count_messages,
            count_tools=count_tools,
        )
        if hooks:
            msg_before = _safe_count(count_messages, messages)
            tools_before = _safe_count(count_tools, tools)
            deferred_before = self.tags.get("tool_search_deferred_tools")
            run_request_hooks(ctx, stream_safe_only=not self.caps.redrive_allowed)
            messages = ctx.messages
            tools = ctx.tools
            # Provider headers the hooks asked for become the contract's
            # ``headers`` object; the gateway sets them on the provider call.
            headers = merge_provider_headers(self.caps.request_headers, ctx.provider_headers)
            deferred_after = self.tags.get("tool_search_deferred_tools")
            if deferred_after and deferred_after != deferred_before:
                transforms.append(
                    f"{NATIVE_DEFERRAL_LABEL}:{int(deferred_after)}tools:"
                    f"{int(self.tags.get('tool_search_deferred_tokens', 0) or 0)}tok"
                )
            msg_after = _safe_count(count_messages, messages)
            if msg_before is not None and msg_after is not None and msg_after < msg_before:
                transforms.append("turn_hook")
                messages_rewritten = True
            tools_after = _safe_count(count_tools, tools)
            if tools_before is not None and tools_after is not None:
                saved = max(0, tools_before - tools_after)
                if saved > 0:
                    self.tags["turn_hook_tools_saved_tokens"] = (
                        int(self.tags.get("turn_hook_tools_saved_tokens", 0) or 0) + saved
                    )
                    transforms.append(f"turn_hook:tools:{saved}tok")
            # Only hooks that ran can re-drive, and with redrive allowed every
            # hook ran; a fold-only (stream_safe) hook never comes back.
            redrive_armed = self.caps.redrive_allowed and any(
                getattr(h, "on_response", None) is not None and not getattr(h, "stream_safe", False)
                for h in hooks
            )

        # History repair last: nothing past this point may change the tools
        # the references are validated against (CCR injection only adds).
        try:
            messages, stripped = repair_tool_search_history(messages, tools, provider=self.provider)
            if stripped:
                transforms.append(f"{TOOL_SEARCH_REPAIR_LABEL}:{stripped}blocks")
                messages_rewritten = True
                ctx.messages = messages
        except Exception:
            logger.exception("gateway turn: tool-search history repair failed; messages unchanged")

        self.result = RequestTransformResult(
            messages=messages,
            tools=tools,
            ctx=ctx,
            transforms=transforms,
            redrive_armed=redrive_armed,
            headers=headers,
            messages_rewritten=messages_rewritten,
        )
        return messages


def _safe_count(counter: Callable[[Any], int] | None, value: Any) -> int | None:
    if counter is None:
        return None
    try:
        return int(counter(value))
    except Exception:
        return None


def arm_ccr_redrive(
    result: RequestTransformResult,
    *,
    provider: str,
    caps: GatewayCapabilities,
    mode: str | None,
    ccr_hashes: list[str],
) -> bool:
    """``config.mode="ccr"`` on the gateway contract: inject ``headroom_retrieve``
    and make the response half answer it.

    Only when markers were actually inserted (nothing to retrieve otherwise)
    and the gateway can re-drive — a marker the model cannot resolve is a
    shrink without reload. The tool goes into the live ``ctx.tools`` too, so a
    re-drive request carries it. Chat-completions and Anthropic tool shapes
    only (``create_ccr_tool_definition``); the Responses API flat shape is
    not produced here.
    """
    if mode != "ccr" or not caps.redrive_allowed or not ccr_hashes:
        return False
    from headroom.ccr import CCR_TOOL_NAME
    from headroom.ccr.tool_injection import create_ccr_tool_definition

    tools = list(result.tools) if isinstance(result.tools, list) else []
    present = any(
        isinstance(t, dict)
        and (
            t.get("name") == CCR_TOOL_NAME
            or (isinstance(t.get("function"), dict) and t["function"].get("name") == CCR_TOOL_NAME)
        )
        for t in tools
    )
    if not present:
        tools.append(create_ccr_tool_definition(provider))
        result.transforms.append("ccr_tool_injected")
    result.tools = tools
    if result.ctx is not None:
        result.ctx.tools = tools
    return True


def make_response_runner(proxy: Any, turn: PendingTurn) -> Callable[..., Any]:
    """The coroutine the ``SuspendedHookRunner`` parks for this turn.

    CCR retrieval first (the model asked for original bytes; it must have them
    before any hook sees a "final" answer), then the registered turn hooks —
    the order the chat path uses. Both re-drive through the same suspending
    ``call_model``. A CCR failure never propagates: the gateway forwards the
    response it holds, tool call and all, which is what the proxy path does
    when a continuation fails.
    """

    async def run(ctx: TurnContext, response: dict[str, Any], call_model: Any) -> Any:
        current = response
        handler = getattr(proxy, "ccr_response_handler", None)
        if turn.ccr_armed and handler is not None:
            try:
                if handler.has_ccr_tool_calls(current, turn.provider):

                    async def api_call_fn(messages: list[dict[str, Any]], tools: Any) -> Any:
                        # Mirrors the chat path's continuation closure: the
                        # re-drive request carries the handler's tools.
                        if tools is not None:
                            ctx.tools = tools
                        return await call_model(messages)

                    current = await handler.handle_response(
                        current, ctx.messages, ctx.tools, api_call_fn, provider=turn.provider
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("gateway turn %s: CCR retrieval failed", turn.turn_id)
        return await run_response_hooks(ctx, current, call_model)

    return run


def build_provider_body(
    body: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: Any,
) -> dict[str, Any]:
    """The complete provider request: every pass-through field, final
    ``messages`` and ``tools``; never the Headroom control keys."""
    out = {k: v for k, v in body.items() if k not in BODY_CONTROL_KEYS}
    out["messages"] = messages
    if tools is not None:
        out["tools"] = tools
    return out


def build_route(request: Any, provider_body: dict[str, Any], provider: str) -> dict[str, Any]:
    """Routing advice for the gateway. Headroom only advises — the gateway
    picks the upstream — but a routing extension's model choice is written
    into the body, as ``BackendResolver.for_request`` does on the proxy path,
    so ``route.model`` and ``body.model`` never disagree."""
    from headroom.proxy.route_advice import advice_from

    advice = advice_from(request)
    if advice is not None:
        provider_body["model"] = advice.model
    return {
        "model": provider_body.get("model"),
        "provider": (advice.provider if advice is not None and advice.provider else provider),
        "service_tier": provider_body.get("service_tier"),
        "reason": advice.reason if advice is not None else "",
    }


def compute_obligations(caps: GatewayCapabilities, *, redrive_armed: bool) -> list[str]:
    obligations: list[str] = []
    if caps.redrive_allowed and redrive_armed:
        obligations.append(OBLIGATION_REDRIVE)
    if caps.can_relay_response:
        obligations.append(OBLIGATION_RELAY_USAGE)
    return obligations


def new_turn_id() -> str:
    return uuid.uuid4().hex


def gateway_response_fields(
    *,
    request: Any,
    body: dict[str, Any],
    caps: GatewayCapabilities,
    provider: str,
    messages: list[dict[str, Any]],
    tools: Any,
    obligations: list[str],
    turn_id: str,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The keys gateway mode adds to every ``/v1/compress`` answer, including
    the fail-open ones (originals, no obligations, no headers)."""
    provider_body = build_provider_body(body, messages, tools)
    return {
        "body": provider_body,
        "turn_id": turn_id,
        "route": build_route(request, provider_body, provider),
        "obligations": list(obligations),
        "gateway": caps.echo(),
        "headers": dict(headers or {}),
    }


# --------------------------------------------------------------------------- #
# Output shaping                                                               #
# --------------------------------------------------------------------------- #


def shape_gateway_body(
    provider_body: dict[str, Any],
    *,
    provider: str,
    model_name: str,
    config: Any,
    input_tokens: int,
    transforms: list[str],
) -> bool:
    """Verbosity steering on the provider body (chat-handler parity, request-only).

    Same gates as both chat handlers: ``shaper_enabled_for(config)`` (the
    ``proxy_output_shaper`` rollout / ``HEADROOM_OUTPUT_SHAPER``),
    ``steering_allowed_for(config)`` (off in ``mode="cache"``), the
    conversation-stable ``HEADROOM_OUTPUT_HOLDOUT`` arm, and the level from
    ``resolve_verbosity_level``. Anthropic bodies get the block appended to
    ``system``; OpenAI chat bodies to the last system/developer message.

    The (arm, stratum) label always goes on ``transforms`` when the shaper is
    enabled — treatment or control — because that label is what the outcome
    funnel (``record_from_labels`` / ``estimate_request_savings``) reads to
    turn the relayed ``output_tokens`` into output savings. The steering block
    is byte-stable per level, so turn N+1 reproduces turn N's prefix.

    Mutates ``provider_body`` in place on copies of the fields it writes, so a
    session snapshot taken before this call (``record_returned``) never sees
    the steering block: the client resends its own system prompt every turn
    and the block is re-appended deterministically. Never raises.
    """
    try:
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
            shape_request,
            shaper_enabled_for,
            steering_allowed_for,
        )

        settings = OutputShaperSettings.from_env(
            enabled=shaper_enabled_for(config),
            steering_enabled=steering_allowed_for(config),
        )
        if not settings.enabled:
            return False
        try:
            holdout = float(runtime_env.getenv("HEADROOM_OUTPUT_HOLDOUT", "0") or "0")
        except ValueError:
            holdout = 0.0
        arm = assign_arm(conversation_key_from_body(provider_body), holdout)
        messages = provider_body.get("messages")
        turn_kind = classify_turn(messages if isinstance(messages, list) else []).value
        stratum = stratum_key(
            turn_kind=turn_kind,
            input_tokens=int(input_tokens or 0),
            model=model_name,
            has_tools=bool(provider_body.get("tools")),
        )
        transforms.append(stratum_label(arm, stratum))
        if arm != "treatment":
            return False
        level, _src = resolve_verbosity_level(settings)

        if provider == "anthropic":
            if provider_body.get("system") is not None:
                provider_body["system"] = copy.deepcopy(provider_body["system"])
            result = shape_request(provider_body, settings, level_override=level)
        else:
            if isinstance(messages, list):
                # The injector mutates the target message dict in place; give
                # it a private copy so the session snapshot stays pristine.
                shallow = list(messages)
                target_index = None
                for index, message in enumerate(shallow):
                    if isinstance(message, dict) and message.get("role") in ("system", "developer"):
                        target_index = index
                if target_index is not None:
                    shallow[target_index] = copy.deepcopy(shallow[target_index])
                provider_body["messages"] = shallow
            result = shape_openai_chat_request(provider_body, settings, level_override=level)
        if result.changed:
            transforms.extend(result.labels or [])
        return bool(result.changed)
    except Exception:  # shaping is best-effort; never fail the turn
        logger.exception("gateway turn: output shaping failed; body left as built")
        return False


def register_pending_turn(
    proxy: Any,
    *,
    turn_id: str,
    session_key: str | None,
    session_id: str | None,
    provider: str,
    model: str,
    body: dict[str, Any],
    ctx: TurnContext | None,
    obligations: list[str],
    outcome_draft: RequestOutcome | None,
    tags: dict[str, Any],
    client: str | None,
    ccr_armed: bool = False,
) -> PendingTurn | None:
    """Register the turn when there is something to wait for; ``None`` otherwise."""
    registry = getattr(proxy, "gateway_turns", None)
    if registry is None or not obligations:
        return None
    now = time.time()
    turn = PendingTurn(
        turn_id=turn_id,
        session_key=session_key,
        session_id=session_id,
        provider=provider,
        model=model,
        body=body,
        ctx=ctx,
        obligations=list(obligations),
        outcome_draft=outcome_draft,
        created_at=now,
        deadline=now + registry.ttl_seconds,
        max_rounds=_env_int("HEADROOM_GATEWAY_MAX_REDRIVES", DEFAULT_MAX_REDRIVES),
        tags=tags,
        client=client,
        ccr_armed=ccr_armed,
    )
    registry.register(turn)
    return turn


# --------------------------------------------------------------------------- #
# Response half                                                                #
# --------------------------------------------------------------------------- #


def _error(status: int, error_type: str, message: str) -> Any:
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status, content={"error": {"type": error_type, "message": message}}
    )


async def apply_session_usage(
    proxy: Any, turn: PendingTurn, usage: NormalizedUsage | None
) -> tuple[int | None, bool]:
    """Feed provider-confirmed cache counts to the session tracker.

    Same rules and same locking as ``/v1/usage``: only under the session turn
    lock, on the executor, and only when the usage carries a positive cache
    signal or a genuine both-zero cold assertion. Returns
    ``(frozen_message_count, applied)``; ``(None, False)`` for a stateless
    turn or an expired session. A lock timeout is reported as not applied
    rather than raised: the turn is finishing and the gateway has nothing to
    retry with.
    """
    if turn.session_key is None:
        return None, False
    tracker = proxy.session_tracker_store.peek(turn.session_key)
    if tracker is None:
        return None, False
    if usage is None or not usage.should_apply:
        return tracker.get_frozen_message_count(), False
    comp_cache = proxy._peek_compression_cache(turn.session_key)
    cache_read = usage.cache_read or 0
    cache_write = usage.cache_write or 0

    from headroom.proxy.handlers.openai import _SESSION_TURN_LOCK_TIMEOUT_SECONDS
    from headroom.proxy.helpers import COMPRESSION_TIMEOUT_SECONDS

    def _apply() -> int | None:
        lock = comp_cache.session_turn_lock if comp_cache is not None else None
        if lock is not None and not lock.acquire(timeout=_SESSION_TURN_LOCK_TIMEOUT_SECONDS):
            raise TimeoutError(f"session turn lock busy for {turn.session_id!r}")
        try:
            last_returned = tracker.get_last_forwarded_messages()
            if not last_returned:
                return None
            tracker.update_from_response(
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                messages=last_returned,
                original_messages=tracker.get_last_original_messages(),
            )
            return int(tracker.get_frozen_message_count())
        finally:
            if lock is not None:
                lock.release()

    try:
        frozen = await proxy._run_compression_in_executor(
            _apply, timeout=COMPRESSION_TIMEOUT_SECONDS
        )
    except TimeoutError:
        logger.warning(
            "gateway turn %s: session %r busy; usage relay not applied",
            turn.turn_id,
            turn.session_id,
        )
        return tracker.get_frozen_message_count(), False
    if frozen is None:
        return tracker.get_frozen_message_count(), False
    return frozen, True


def complete_outcome(
    draft: RequestOutcome,
    usage: NormalizedUsage | None,
    *,
    status: int,
    latency_ms: float | None,
) -> RequestOutcome:
    """Fill the response-side fields of a deferred outcome draft.

    Only fields ``RequestOutcome`` already carries: billed input lands in
    ``provider_input_tokens`` (never ``optimized_tokens`` — different
    tokenizer scale), output and the two cache counters as reported, provider
    status so a 5xx is funnelled as a failure, and the gateway's provider
    latency added to the compress-side wall clock.
    """
    usage = usage or NormalizedUsage()
    return dataclasses.replace(
        draft,
        output_tokens=usage.output_tokens or 0,
        provider_input_tokens=usage.input_tokens or 0,
        cache_read_tokens=usage.cache_read or 0,
        cache_write_tokens=usage.cache_write or 0,
        status_code=status,
        total_latency_ms=draft.total_latency_ms + float(latency_ms or 0.0),
    )


def _emit_response_received(proxy: Any, turn: PendingTurn, response: Any, status: int) -> None:
    pipeline = getattr(proxy, "pipeline_extensions", None)
    if pipeline is None or not getattr(pipeline, "enabled", False):
        return
    try:
        from headroom.pipeline import PipelineStage

        pipeline.emit(
            PipelineStage.RESPONSE_RECEIVED,
            operation="gateway.turn",
            request_id=(turn.outcome_draft.request_id if turn.outcome_draft else turn.turn_id),
            provider=turn.provider,
            model=turn.model,
            response=response,
            metadata={
                "path": "/v1/compress/response",
                "stream": False,
                "status_code": status,
                "turn_id": turn.turn_id,
                "rounds": turn.rounds,
            },
        )
    except Exception:  # telemetry must never break a response
        logger.debug("RESPONSE_RECEIVED emit failed", exc_info=True)


def _redrive_payload(turn: PendingTurn, step: Step) -> dict[str, Any]:
    """The ``redrive`` answer for ``step``; also what a stale retry gets again.

    Mirrors ``_hook_call_model`` on the proxy path: the hook's message list plus
    its (possibly reloaded) tools, everything else exactly as the gateway sent.
    """
    redrive_body = dict(turn.body)
    redrive_body["messages"] = step.messages
    if step.tools is not None:
        redrive_body["tools"] = step.tools
    return {
        "action": "redrive",
        "turn_id": turn.turn_id,
        "request": redrive_body,
        "round": turn.rounds,
    }


async def handle_compress_response(proxy: Any, request: Any) -> Any:
    """``POST /v1/compress/response`` — finish or re-drive a pending turn."""
    from fastapi.responses import JSONResponse

    from headroom.proxy.helpers import _read_request_json

    try:
        body = await _read_request_json(request)
    except Exception:
        return _error(400, "invalid_request", "Invalid JSON in request body.")
    if not isinstance(body, dict):
        return _error(400, "invalid_request", "Request body must be a JSON object.")

    turn_id = body.get("turn_id")
    if not isinstance(turn_id, str) or not turn_id.strip() or len(turn_id) > 128:
        return _error(400, "invalid_request", "Missing or invalid turn_id.")

    status = body.get("status", 200)
    if status is None:
        status = 200
    if isinstance(status, bool) or not isinstance(status, int) or not (100 <= status <= 599):
        return _error(400, "invalid_request", "status must be an HTTP status code.")
    latency_ms = body.get("latency_ms")
    if latency_ms is not None and (
        isinstance(latency_ms, bool) or not isinstance(latency_ms, (int, float)) or latency_ms < 0
    ):
        return _error(400, "invalid_request", "latency_ms must be a non-negative number.")
    raw_usage = body.get("usage")
    if raw_usage is not None and not isinstance(raw_usage, dict):
        return _error(400, "invalid_request", "usage must be an object when present.")
    response = body.get("response")
    if response is not None and not isinstance(response, dict):
        return _error(400, "invalid_request", "response must be an object when present.")
    # Validate the usage before touching the turn: a bad relay must not leave
    # a half-driven coroutine behind.
    usage: NormalizedUsage | None = None
    if raw_usage is not None:
        try:
            usage = normalize_usage(raw_usage)
        except GatewayRequestError as e:
            return _error(400, "invalid_request", e.message)

    registry = getattr(proxy, "gateway_turns", None)
    turn = registry.get(turn_id) if registry is not None else None
    if turn is None:
        return _error(
            404,
            "unknown_turn",
            f"No pending turn {turn_id!r} (never registered, already finished, or expired).",
        )
    if OBLIGATION_REDRIVE in turn.obligations and response is None:
        return _error(
            400,
            "missing_response",
            "This turn carries the 'redrive' obligation; the provider response is required.",
        )
    if turn.in_flight:
        return _error(409, "turn_busy", f"Turn {turn_id!r} is already being driven.")

    # `round` makes the half idempotent. It names which provider call this
    # response answers: 0 is the original forward, n is the n-th re-drive
    # (the `round` this endpoint returned). Without it a gateway whose socket
    # dropped after we answered "redrive" would re-post round 0 and the parked
    # hook would take it as the answer to a re-drive the gateway never sent —
    # and the usage below would bill that response twice.
    round_no = body.get("round")
    if round_no is not None and (
        isinstance(round_no, bool) or not isinstance(round_no, int) or round_no < 0
    ):
        return _error(400, "invalid_request", "round must be a non-negative integer.")
    if round_no is not None and round_no != turn.rounds:
        if round_no > turn.rounds:
            return _error(
                400,
                "round_out_of_sequence",
                f"Turn {turn_id!r} is waiting for round {turn.rounds}, got {round_no}.",
            )
        # A stale retry: repeat the last answer without consuming anything.
        last = turn.runner.last_step if turn.runner is not None else None
        if last is not None and last.is_redrive:
            return JSONResponse(_redrive_payload(turn, last))
        return _error(
            409,
            "round_already_answered",
            f"Round {round_no} of turn {turn_id!r} was already answered.",
        )

    turn.in_flight = True
    try:
        if usage is not None:
            turn.usage = usage if turn.usage is None else turn.usage.merged_with(usage)

        step: Step | None = None
        if turn.hooks_armed and turn.ctx is not None:
            runner = turn.runner
            try:
                if runner is None:
                    if status == 200 and isinstance(response, dict):
                        runner = turn.runner = SuspendedHookRunner(
                            turn.ctx, run=make_response_runner(proxy, turn)
                        )
                        step = await runner.start(response)
                elif runner.state == "awaiting_redrive":
                    step = await runner.resume(response)  # type: ignore[arg-type]
                else:
                    # A repeated call after the hooks finished (gateway retry):
                    # answer the same thing again rather than re-running.
                    step = runner.last_step
            except Exception:
                # A failure to start/resume ends the turn with the gateway's
                # own response; hooks are best-effort by contract.
                logger.exception("gateway turn %s: hook driver failed", turn_id)
                if runner is not None:
                    await runner.cancel()
                step = Step("done", response=None)

        if step is not None and step.is_redrive:
            turn.rounds += 1
            if turn.rounds > turn.max_rounds:
                logger.warning(
                    "gateway turn %s: hit max re-drive rounds (%d); returning latest response",
                    turn_id,
                    turn.max_rounds,
                )
                if turn.runner is not None:
                    await turn.runner.cancel()
                step = Step("done", response=None)
            else:
                # Refresh the deadline: the gateway is actively working the turn.
                turn.deadline = time.time() + (registry.ttl_seconds if registry else 0)
                return JSONResponse(_redrive_payload(turn, step))

        final_response = step.response if step is not None else None
        frozen, applied = await apply_session_usage(proxy, turn, turn.usage)
        if turn.outcome_draft is not None:
            outcome = complete_outcome(
                turn.outcome_draft, turn.usage, status=status, latency_ms=latency_ms
            )
            turn.outcome_draft = None
            await proxy._record_request_outcome(outcome)
        _emit_response_received(
            proxy, turn, final_response if final_response is not None else response, status
        )
        if registry is not None:
            registry.pop(turn_id)
        return JSONResponse(
            {
                "action": "done",
                "turn_id": turn_id,
                "response": final_response,
                "frozen_message_count": frozen,
                "usage_applied": applied,
                "rounds": turn.rounds,
            }
        )
    finally:
        turn.in_flight = False
