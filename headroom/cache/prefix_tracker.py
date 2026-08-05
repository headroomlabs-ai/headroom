"""Prefix Cache Tracker — session-scoped state for cache-aware compression.

Tracks provider prefix cache state between turns so the transform pipeline
can freeze already-cached messages and only compress new content.

Problem: Clients like Claude Code already manage prefix caching (up to 4
cache_control breakpoints, growing-prefix strategy). If Headroom compresses
or modifies messages in the cached prefix, it invalidates the cache —
replacing a 90% read discount (Anthropic) or 50% (OpenAI) with a 25%
write penalty.

Solution: After each API response, record how many tokens the provider
cached. On the next turn, freeze that many messages so the transform
pipeline skips them entirely.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Provider cache economics for cost comparisons
_PROVIDER_READ_DISCOUNT = {
    "anthropic": 0.9,  # 90% discount on reads
    "openai": 0.5,  # 50% discount on reads
    "gemini": 0.9,
    "bedrock": 0.9,
}

_PROVIDER_WRITE_PENALTY = {
    "anthropic": 0.25,  # 25% surcharge on writes
    "openai": 0.0,  # No write penalty
    "gemini": 0.0,
    "bedrock": 0.25,
}

# Default prompt-cache lifetime per provider, in seconds. Used by
# `classify_cache_miss` to decide whether a miss is most likely a TTL
# lapse (idle longer than this) versus a prefix-content change. Anthropic's
# default ephemeral cache is 5 minutes (matches
# headroom.cache.anthropic.ANTHROPIC_CACHE_TTL_SECONDS); the others are best-
# effort defaults and only matter once those providers are wired in. A
# session that opts into Anthropic's 1h cache breakpoint can override this
# via the tracker config (see PrefixFreezeConfig.cache_ttl_seconds).
_PROVIDER_CACHE_TTL_SECONDS = {
    "anthropic": 300,  # 5 minutes (default ephemeral cache)
    "openai": 300,  # automatic prefix cache, ~5-10 min; conservative floor
    "gemini": 300,
    "bedrock": 300,
}


@dataclass
class PrefixFreezeConfig:
    """Configuration for cache-aware prefix freezing."""

    enabled: bool = True
    min_cached_tokens: int = 1024  # Min cached tokens to activate freeze
    session_ttl_seconds: int = 600  # Session tracker cleanup TTL
    force_compress_threshold: float = 0.5  # Bust cache if compression saves > this fraction
    # Provider prompt-cache lifetime used by `classify_cache_miss` to tell a
    # TTL lapse from a prefix change. `None` falls back to the per-provider
    # default in `_PROVIDER_CACHE_TTL_SECONDS`. Set to 3600 for a session that
    # uses Anthropic's 1h cache breakpoint so idle-gap attribution stays honest.
    cache_ttl_seconds: int | None = None
    # Cap on concurrent conversation lineages tracked per session id (#2085).
    # A fan-out storm (many parallel subagents sharing one fallback id) evicts
    # the shortest-chain lineage instead of growing without bound. Raise it
    # for workspaces that genuinely run more concurrent conversations on one
    # model + system prompt.
    max_lineages_per_session: int = 32


@dataclass
class FreezeStats:
    """Statistics from prefix freezing for metrics/dashboard."""

    busts_avoided: int = 0
    tokens_preserved: int = 0
    compression_foregone_tokens: int = 0
    net_benefit_tokens: int = 0  # tokens_preserved - compression_foregone
    frozen_message_count: int = 0
    turn_number: int = 0


# Cache-miss attribution verdicts. `reason` is one of these literals so
# metrics/dashboard can bucket without re-deriving the logic. See
# PrefixCacheTracker.classify_cache_miss.
MISS_TTL_EXPIRY = "ttl_expiry"
MISS_PREFIX_CHANGE = "prefix_change"
MISS_COLD_START = "cold_start"  # no prior cached prefix to miss against
MISS_UNKNOWN = "unknown"  # expected a hit, content stable, idle within TTL


@dataclass
class CacheMissAttribution:
    """Why a turn that expected a prompt-cache hit missed instead.

    Produced by :meth:`PrefixCacheTracker.classify_cache_miss`. ``is_miss``
    is False when the turn actually hit cache (or there was nothing to hit),
    in which case ``reason`` is informational only.
    """

    is_miss: bool
    reason: str  # one of the MISS_* literals
    idle_seconds: float = 0.0
    cache_ttl_seconds: int = 0
    expected_cached_tokens: int = 0
    cache_read_tokens: int = 0
    prefix_changed: bool = False
    ttl_exceeded: bool = False


def _strip_cache_control(obj: Any) -> Any:
    """Recursively drop ``cache_control`` for content-only equality checks.

    Clients (notably Claude Code) move the cache_control breakpoint to the newest
    message on every call, so the exact same message carries cache_control on one
    turn and not the next. That per-call annotation must be ignored when deciding
    whether this turn append-only-extends the previous one — otherwise a moved
    marker spuriously fails the check and we skip the byte-identical replay,
    busting the cache."""
    if isinstance(obj, dict):
        return {k: _strip_cache_control(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip_cache_control(v) for v in obj]
    return obj


# Keys that carry NO semantic payload for the model — transport / caching-directive
# / telemetry / client-routing annotations that clients attach and vary turn-to-turn.
# Grounded in provider API docs (Anthropic Messages, OpenAI Chat+Responses, Bedrock
# Converse) + client-library field inventories (litellm, Vercel AI SDK, opencode,
# Claude Code, Cline). Dropped from the cross-turn prefix-equality key ONLY.
#
# NOTE ON SAFETY: this projection is a COMPARISON KEY, never a source to rebuild
# forwarded bytes — the cache-stable-delta path always forwards the previously
# forwarded bytes + the raw appended delta. So dropping these can't deprive the
# model. What we must NOT do is drop a *semantic* field (that would mask a real
# divergence and replay a stale prefix), which is why: (1) reasoning SIGNATURES are
# NOT in this set (Anthropic 400s if a thinking block is altered/missing, and a
# present/absent flip is a real divergence we want to detect); (2) tool inputs /
# arguments / json payloads are treated as OPAQUE and compared verbatim (see
# _OPAQUE_PAYLOAD_KEYS) so a user key that happens to be named "index"/"state" is
# never stripped from inside a tool call.
_NON_SEMANTIC_KEYS = frozenset(
    {
        # cache-breakpoint markers (moved to the newest block every turn)
        "cache_control",  # Anthropic (per-block)
        "cachePoint",  # Bedrock (per-block content block)
        # litellm unified-message / tool annotations
        "caller",  # litellm programmatic-tool tag on tool_use
        "provider_specific_fields",
        "reasoning_content",  # litellm display echo (the paired signature is separate)
        "reasoning_items",
        "annotations",  # citation/display metadata
        # OpenAI response echoes that can ride on assistant messages
        "system_fingerprint",
        "service_tier",
        # Vercel AI SDK / opencode part transport
        "providerMetadata",
        "providerOptions",
        "callProviderMetadata",
        "state",
        "providerExecuted",
        "synthetic",
        "ignored",
        # streaming-assembly artifact
        "index",
    }
)

# Values under these keys are opaque semantic payloads (tool-call input, OpenAI
# stringified arguments, Bedrock tool_result json). They are compared VERBATIM — we
# never recurse into them to strip "noise" keys, because arbitrary user data there
# may legitimately contain keys that collide with _NON_SEMANTIC_KEYS (e.g. an
# `input` of {"state": "CA", "index": 3}). Recursing would corrupt the comparison.
_OPAQUE_PAYLOAD_KEYS = frozenset({"input", "arguments", "json"})


def _canonicalize_for_prefix_compare(obj: Any) -> Any:
    """Representation-agnostic canonical form for cross-turn prefix equality.

    Providers accept several *equivalent* encodings for the same message, and real
    clients vary them turn-to-turn; a raw-dict prefix compare then fails spuriously
    and drops cache mode to raw (uncompressed) forwarding. This normalizes ONLY
    representation:
      * drops non-semantic annotation / cache-directive / telemetry keys
        (_NON_SEMANTIC_KEYS) at any message/block level;
      * wraps a bare string ``content`` into ``[{"type": "text", "text": ...}]``
        (Anthropic's string sugar, which litellm flips per turn);
      * leaves tool ``input`` / ``arguments`` / ``json`` payloads verbatim
        (_OPAQUE_PAYLOAD_KEYS) so user data is never corrupted;
      * KEEPS all real content (text, tool name/input, tool_result content, reasoning
        signatures, ids) so two messages canonicalize-equal iff they are semantically
        identical.

    Used ONLY as a comparison key for the cache-stable delta path; the original,
    unmodified messages are always what gets forwarded.
    """
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key in _NON_SEMANTIC_KEYS:
                continue
            if key in _OPAQUE_PAYLOAD_KEYS:
                out[key] = value  # verbatim — do not recurse into user payloads
            elif key == "content" and isinstance(value, str):
                out[key] = [{"type": "text", "text": value}]
            else:
                out[key] = _canonicalize_for_prefix_compare(value)
        return out
    if isinstance(obj, list):
        canon = [_canonicalize_for_prefix_compare(value) for value in obj]
        # Drop blocks that projected to {} — a pure cache-directive content block
        # (e.g. Bedrock {"cachePoint": {...}}) whose only key was non-semantic. Left
        # in place it would be an empty-dict entry, so a directive block moving
        # position across turns would spuriously fail the length/order compare.
        return [value for value in canon if value != {}]
    return obj


def extract_cache_stable_delta(
    current_messages: list[dict[str, Any]],
    previous_original_messages: list[dict[str, Any]] | None,
    previous_forwarded_messages: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Return ``(stable_forwarded_prefix, appended_delta_messages)`` when the current
    request append-only-extends the previous one, else ``None``.

    Provider-agnostic delta engine for cache mode. "Append-only" is decided by comparing
    the *canonicalized* prefix (:func:`_canonicalize_for_prefix_compare`, which ignores
    per-turn transport / cache-directive / client-annotation noise across
    Anthropic / OpenAI / Bedrock and the common clients), so a moved cache marker or
    shape churn does not spuriously collapse cache mode to raw forwarding. On a match the
    caller replays the byte-identical previously-forwarded prefix and compresses ONLY the
    appended delta.

    This is a COMPARISON + slice only: the returned prefix is the previously-forwarded
    bytes verbatim and the delta is the raw appended messages — never a rebuild from the
    canonical projection — so the projection dropping non-semantic fields is safe.
    """
    if not previous_original_messages or previous_forwarded_messages is None:
        return None
    prefix_len = len(previous_original_messages)
    if len(current_messages) < prefix_len:
        return None
    if _canonicalize_for_prefix_compare(
        current_messages[:prefix_len]
    ) != _canonicalize_for_prefix_compare(previous_original_messages):
        return None
    return (
        copy.deepcopy(previous_forwarded_messages),
        copy.deepcopy(current_messages[prefix_len:]),
    )


def overlay_cached_prefix(
    optimized_messages: list[dict[str, Any]],
    current_original_messages: list[dict[str, Any]],
    previous_original_messages: list[dict[str, Any]] | None,
    previous_forwarded_messages: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Replay the previously-forwarded (cached, compressed) prefix byte-identical.

    Provider-agnostic cache-safety guard for the freeze path. When a message is
    "frozen", the compression pipeline may emit the agent's ORIGINAL bytes for
    it — but the provider cached whatever we FORWARDED last turn (the compressed
    form). Forwarding the original then mismatches the cached prefix and busts
    the prompt cache from that point (100% of observed misses were this
    ``prefix_change``). This overlays the exact previously-forwarded prefix onto
    the corresponding leading messages so the forwarded prefix stays byte-for-byte
    what the provider hashed for its cache key.

    Safe only when this turn append-only-extends the previous turn (the standard
    growing-conversation shape): the previous ORIGINAL messages must be an exact
    prefix of the current ORIGINAL messages, and there is exactly one forwarded
    message per original. Otherwise the previous forwarded bytes may not
    correspond to the same positions, so we return ``optimized_messages``
    unchanged (accept a possible bust rather than forward wrong content).

    This makes freezing byte-identical in BOTH proxy modes, so the only remaining
    difference between them is how large a mutable (still-compressible) tail each
    leaves — not whether the frozen prefix busts the cache.
    """
    prev_orig = previous_original_messages
    prev_fwd = previous_forwarded_messages
    if not prev_orig or not prev_fwd:
        return optimized_messages
    n = len(prev_orig)
    # Positional 1:1 correspondence between prev_orig[i] and prev_fwd[i] holds
    # only when last turn forwarded exactly one message per original (the
    # append-only, no-injection shape update_from_response records). If the
    # counts differ, an injected / dropped / merged message shifted the
    # mapping, so replaying prev_fwd[i] at position i could forward the wrong
    # content — bail (leave this turn's output untouched) rather than risk it.
    if len(prev_fwd) != n:
        logger.debug(
            "overlay: forwarded/original count mismatch (prev_fwd=%d, prev_orig=%d) "
            "— skipping cached-prefix replay (possible bust)",
            len(prev_fwd),
            n,
        )
        return optimized_messages
    # Append-only guard on CONTENT ONLY, message-by-message. Replay the
    # previously-forwarded (cached, compressed) bytes for the longest LEADING
    # run of messages that is byte-for-byte (content-canonical) identical to
    # what we forwarded last turn, and stop at the first divergence.
    #
    # This is the cache-safety centerpiece for token mode (which relies solely
    # on this replay; cache mode is already byte-stable by construction). The
    # prior all-or-nothing guard busted the ENTIRE cached prefix the moment any
    # single leading message failed to canonicalize-equal last turn — most
    # commonly the just-added assistant turn, whose client-resent form can
    # differ trivially from the copy we reconstructed and recorded. Stopping at
    # the first divergence instead keeps the (much larger) cache-hit region
    # up to that point and only re-forwards from the changed message onward.
    #
    # Comparison uses the shared canonicalizer (not just cache_control
    # stripping) so it is robust to ALL per-turn transport / annotation churn —
    # cache_control movement (Anthropic), litellm `caller`,
    # provider_specific_fields, streaming `index`, string<->block content shape,
    # etc. Content stability is what the provider's prefix cache actually keys
    # on. Safe by construction: we only replay prev_fwd[k] where
    # current_original[k] canonicalize-equals prev_orig[k], and prev_fwd[k]
    # positionally corresponds to prev_orig[k] (guaranteed by the count check
    # above), so no wrong bytes are ever forwarded.
    limit = min(n, len(current_original_messages), len(optimized_messages))
    k = 0
    while k < limit and _canonicalize_for_prefix_compare(
        current_original_messages[k]
    ) == _canonicalize_for_prefix_compare(prev_orig[k]):
        k += 1
    if k == 0:
        logger.debug(
            "overlay: prefix diverged at message 0 — no cached-prefix replay "
            "(cold prefix or client rewrote history head)"
        )
        return optimized_messages
    if k < n:
        logger.debug(
            "overlay: cached-prefix replay for %d/%d leading messages "
            "(diverged at %d — re-forwarding tail fresh)",
            k,
            n,
            k,
        )
    # Replay the cached (compressed) prefix byte-identical up to the first
    # divergence; keep this turn's freshly-produced output for the rest.
    return list(prev_fwd[:k]) + list(optimized_messages[k:])


def normalize_message_cache_control(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Own message-level cache_control placement so breakpoints stay bounded.

    Two forces pile up cache_control markers turn over turn: clients move the
    breakpoint to the newest message each call, and ``overlay_cached_prefix``
    replays the markers that rode on each turn's then-newest message. Anthropic
    hard-errors at **>4 cache_control blocks total** (system + tools + messages),
    so on a long conversation the accumulation eventually 400s.

    Fix: strip EVERY message-level cache_control and re-place a **single**
    ephemeral breakpoint on the last block of the last block-style message. One
    breakpoint caches the whole message prefix up to it, and — because the
    provider's cache key is message CONTENT, not marker presence (moving the
    breakpoint forward is the documented client pattern and it hits) — stripping
    and re-placing markers never busts. system/tools breakpoints live outside
    ``messages`` and are left untouched (they still count toward the 4 limit, so
    holding messages to one breakpoint leaves room for them).

    Headroom owns WHERE the breakpoint goes; the client still owns WHAT it says:
    the re-placed marker reuses the newest client marker verbatim, so an explicit
    ``ttl`` (e.g. ``"1h"``) survives consolidation instead of silently
    downgrading to the 5-minute default (#2375).

    Only block-style (list) content can carry cache_control; string content is
    left as-is. Returns the input unchanged when there is nothing to normalize.
    """
    changed = False
    out: list[dict[str, Any]] = []
    last_block_idx = -1
    last_marker: dict[str, Any] | None = None
    for i, msg in enumerate(messages):
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            had = False
            for b in content:
                if isinstance(b, dict) and "cache_control" in b:
                    had = True
                    # The newest marker in message order is the client's current
                    # intent (older ones are replay leftovers) — keep it.
                    if isinstance(b["cache_control"], dict):
                        last_marker = b["cache_control"]
            stripped = [
                {k: v for k, v in b.items() if k != "cache_control"} if isinstance(b, dict) else b
                for b in content
            ]
            out.append({**msg, "content": stripped} if had else msg)
            changed = changed or had
            if stripped and isinstance(stripped[-1], dict):
                last_block_idx = i
        else:
            out.append(msg)
    # Re-place exactly one breakpoint on the last block-style message.
    if last_block_idx >= 0:
        msg = out[last_block_idx]
        content = list(msg["content"])
        marker = dict(last_marker) if last_marker else {"type": "ephemeral"}
        content[-1] = {**content[-1], "cache_control": marker}
        out[last_block_idx] = {**msg, "content": content}
        changed = True
    return out if changed else messages


class PrefixCacheTracker:
    """Tracks provider prefix cache state across turns in a session.

    Usage:
        tracker = PrefixCacheTracker("anthropic")

        # Before compression (turn 2+):
        frozen = tracker.get_frozen_message_count()
        result = pipeline.apply(messages, model, frozen_message_count=frozen)

        # After API response:
        tracker.update_from_response(
            cache_read_tokens=usage["cache_read_input_tokens"],
            cache_write_tokens=usage["cache_creation_input_tokens"],
            messages=optimized_messages,
            tokenizer=tokenizer,
        )
    """

    def __init__(self, provider: str, config: PrefixFreezeConfig | None = None):
        self.provider = provider
        self.config = config or PrefixFreezeConfig()
        self._cached_token_count: int = 0
        self._cached_message_count: int = 0
        self._turn_number: int = 0
        self._last_activity: float = time.time()
        self._last_original_messages: list[dict[str, Any]] = []
        self._last_forwarded_messages: list[dict[str, Any]] = []
        # Idle gap (seconds) since the PREVIOUS turn's response, captured by
        # SessionTrackerStore.get_or_create at fetch time — BEFORE it refreshes
        # _last_activity. Without this snapshot, seconds_since_activity() reads
        # ~0 on every request (the fetch itself bumps the clock), so the
        # net-cost/TTL P_alive gate could never see idle time. The handler reads
        # this and forwards it to the pipeline as `idle_seconds`.
        self._idle_seconds_at_fetch: float = 0.0

        # Session-scoped ReadMaturationManager (Mechanism B), created
        # lazily by the handler when read maturation is enabled. Rides
        # here so it shares the session's affinity and TTL cleanup.
        self.read_maturation_manager: Any = None

        # Stats
        self._busts_avoided: int = 0
        self._tokens_preserved: int = 0
        self._compression_foregone_tokens: int = 0

    def get_frozen_message_count(self) -> int:
        """How many leading messages to skip compression on the next turn.

        Returns 0 on turn 0 (cold start) or if caching is disabled/below threshold.
        """
        if not self.config.enabled:
            return 0
        if self._turn_number == 0:
            return 0
        if self._cached_token_count < self.config.min_cached_tokens:
            return 0
        return self._cached_message_count

    def update_from_response(
        self,
        cache_read_tokens: int,
        cache_write_tokens: int,
        messages: list[dict[str, Any]],
        message_token_counts: list[int] | None = None,
        original_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Update tracker with cache metrics from the API response.

        Called after every API call. Computes how many messages to freeze
        on the next turn based on the cache_read_tokens reported.

        Args:
            cache_read_tokens: Tokens read from cache (cache hit portion).
            cache_write_tokens: Tokens written to cache (new cache entries).
            messages: The messages that were sent to the API.
            message_token_counts: Pre-computed token counts per message.
                If None, estimates from content length.
        """
        self._last_activity = time.time()
        self._turn_number += 1
        self._last_original_messages = copy.deepcopy(original_messages or messages)
        self._last_forwarded_messages = copy.deepcopy(messages)

        # Compute total cached tokens (read + write = what's in cache now)
        total_cached = cache_read_tokens + cache_write_tokens

        if total_cached == 0:
            self._cached_token_count = 0
            self._cached_message_count = 0
            return

        # Estimate per-message token counts if not provided
        if message_token_counts is None:
            message_token_counts = self._estimate_message_tokens(messages)

        # Walk messages from the start, accumulating tokens until we exceed
        # the cached amount. All messages within the cached prefix are frozen.
        accumulated = 0
        frozen_count = 0
        for i, tok_count in enumerate(message_token_counts):
            accumulated += tok_count
            if accumulated <= total_cached:
                frozen_count = i + 1
            else:
                break

        self._cached_token_count = total_cached
        self._cached_message_count = frozen_count

        logger.debug(
            "PrefixCacheTracker[%s]: turn=%d, cached=%d tokens, "
            "frozen=%d/%d messages (read=%d, write=%d)",
            self.provider,
            self._turn_number,
            total_cached,
            frozen_count,
            len(messages),
            cache_read_tokens,
            cache_write_tokens,
        )

    def get_last_original_messages(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._last_original_messages)

    def get_last_forwarded_messages(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._last_forwarded_messages)

    def resolved_cache_ttl_seconds(self) -> int:
        """Effective prompt-cache lifetime for this session's provider."""
        if self.config.cache_ttl_seconds is not None:
            return self.config.cache_ttl_seconds
        return _PROVIDER_CACHE_TTL_SECONDS.get(self.provider, 300)

    def classify_cache_miss(
        self,
        cache_read_tokens: int,
        current_forwarded_messages: list[dict[str, Any]],
        idle_seconds: float | None = None,
    ) -> CacheMissAttribution:
        """Attribute *this turn's* cache outcome: hit, TTL lapse, or prefix change.

        Call this BEFORE :meth:`update_from_response` — it reads the state
        captured from the *previous* turn (``_cached_token_count``,
        ``_last_forwarded_messages``, ``_last_activity``), all of which
        ``update_from_response`` overwrites.

        Attribution only fires when the previous turn left a cacheable prefix
        (``_cached_token_count > 0``); the very first warm turn has nothing to
        miss against, so it is reported as ``cold_start`` with ``is_miss=False``.

        When a hit was expected but ``cache_read_tokens == 0``:

        * If the idle gap since the last turn exceeded the provider cache TTL,
          the cache entry had already lapsed — ``ttl_expiry``. **TTL wins ties:**
          once the entry expired, a coincident prefix change is moot (the issue
          asks "should I move 5m → 1h?", which only the TTL signal answers).
        * Otherwise, if the forwarded prefix changed versus last turn, the new
          bytes couldn't match the cached prefix — ``prefix_change``.
        * If neither signal fires (stable prefix, within TTL) we can't explain
          it from local state — ``unknown`` (e.g. provider-side eviction).

        A partial read (``0 < cache_read_tokens``) counts as a hit here; the
        existing model-aware bust detection in PrometheusMetrics already covers
        partial-invalidation accounting, and double-counting it as a "miss"
        would muddy the 5m-vs-1h signal this method exists to provide.

        Returns a :class:`CacheMissAttribution`; ``is_miss`` is False for hits
        and cold starts.
        """
        if idle_seconds is None:
            idle_seconds = self.seconds_since_activity()
        ttl = self.resolved_cache_ttl_seconds()
        expected = self._cached_token_count

        # Nothing was cached last turn → cold start, not a miss.
        if expected <= 0:
            return CacheMissAttribution(
                is_miss=False,
                reason=MISS_COLD_START,
                idle_seconds=idle_seconds,
                cache_ttl_seconds=ttl,
                expected_cached_tokens=expected,
                cache_read_tokens=cache_read_tokens,
            )

        # We expected a hit. A non-zero read means the prefix cache worked.
        if cache_read_tokens > 0:
            return CacheMissAttribution(
                is_miss=False,
                reason="hit",
                idle_seconds=idle_seconds,
                cache_ttl_seconds=ttl,
                expected_cached_tokens=expected,
                cache_read_tokens=cache_read_tokens,
            )

        # Full miss on a prefix we expected cached. Attribute it.
        ttl_exceeded = idle_seconds > ttl
        prefix_changed = not self._forwarded_prefix_stable(current_forwarded_messages)

        if ttl_exceeded:
            reason = MISS_TTL_EXPIRY  # TTL wins ties (see docstring)
        elif prefix_changed:
            reason = MISS_PREFIX_CHANGE
        else:
            reason = MISS_UNKNOWN

        return CacheMissAttribution(
            is_miss=True,
            reason=reason,
            idle_seconds=idle_seconds,
            cache_ttl_seconds=ttl,
            expected_cached_tokens=expected,
            cache_read_tokens=cache_read_tokens,
            prefix_changed=prefix_changed,
            ttl_exceeded=ttl_exceeded,
        )

    def _forwarded_prefix_stable(self, current_forwarded_messages: list[dict[str, Any]]) -> bool:
        """True if last turn's forwarded prefix is still an exact prefix of this turn's.

        The cached prefix is whatever we forwarded last turn. If those exact
        messages still lead the current forwarded list, the bytes the provider
        hashed for its cache key are unchanged, so a miss can't be blamed on
        content. Anything else (a frozen message rewritten, the prefix
        reordered, the list now shorter) counts as a prefix change.
        """
        prev = self._last_forwarded_messages
        if not prev:
            # No recorded prefix to compare — can't claim it changed.
            return True
        if len(current_forwarded_messages) < len(prev):
            return False
        return current_forwarded_messages[: len(prev)] == prev

    def record_bust_avoided(self, tokens_preserved: int, compression_foregone: int) -> None:
        """Record when we chose to preserve cache over compressing."""
        self._busts_avoided += 1
        self._tokens_preserved += tokens_preserved
        self._compression_foregone_tokens += compression_foregone

    def should_force_compress(
        self,
        message_index: int,
        message_tokens: int,
        estimated_compressed_tokens: int,
    ) -> bool:
        """Check if compression savings outweigh cache preservation.

        Returns True if we should bust the cache and compress anyway.
        This happens when compression would save a large fraction of tokens
        AND the savings exceed the cache read discount.
        """
        if message_index >= self._cached_message_count:
            return True  # Not in frozen prefix, always compress

        if message_tokens == 0:
            return False

        savings_fraction = (message_tokens - estimated_compressed_tokens) / message_tokens

        # Would compression savings exceed the cache read discount?
        read_discount = _PROVIDER_READ_DISCOUNT.get(self.provider, 0.5)
        return savings_fraction > read_discount

    @property
    def is_expired(self) -> bool:
        """Check if this tracker has been idle beyond TTL."""
        return (time.time() - self._last_activity) > self.config.session_ttl_seconds

    def seconds_since_activity(self) -> float:
        """Wall-clock seconds since this tracker last saw activity.

        #856 P3b feeds this to the net-cost gate as an idle signal: as it
        approaches the provider's prompt-cache TTL (~300s for Anthropic),
        P_alive decays toward 0 and deep edits near cache lapse become free.
        Distinct from :attr:`is_expired`, which uses the much longer
        session-tracker *cleanup* TTL (``session_ttl_seconds``), not the cache
        TTL.

        Wiring caveat: ``SessionTrackerStore.get_or_create`` refreshes
        ``_last_activity`` on access, so a caller that wants the idle gap
        since the *previous turn's response* must read this before fetching
        the tracker for the current request (or the store must capture it at
        fetch time). ``update_from_response`` is the per-turn activity stamp.
        """
        return max(0.0, time.time() - self._last_activity)

    @property
    def stats(self) -> FreezeStats:
        """Return stats for dashboard/metrics."""
        return FreezeStats(
            busts_avoided=self._busts_avoided,
            tokens_preserved=self._tokens_preserved,
            compression_foregone_tokens=self._compression_foregone_tokens,
            net_benefit_tokens=self._tokens_preserved - self._compression_foregone_tokens,
            frozen_message_count=self._cached_message_count,
            turn_number=self._turn_number,
        )

    @staticmethod
    def _estimate_message_tokens(messages: list[dict[str, Any]]) -> list[int]:
        """Rough token count per message (chars / 3.5).

        Counts text, tool_result content, and tool_use input fields
        for accurate Anthropic-format estimation.
        """
        counts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                chars = len(content)
            elif isinstance(content, list):
                chars = 0
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type", "")
                    if block_type == "text":
                        chars += len(block.get("text", ""))
                    elif block_type == "tool_result":
                        inner = block.get("content", "")
                        if isinstance(inner, str):
                            chars += len(inner)
                        elif isinstance(inner, list):
                            chars += sum(
                                len(b.get("text", "")) for b in inner if isinstance(b, dict)
                            )
                    elif block_type == "tool_use":
                        inp = block.get("input")
                        if isinstance(inp, str):
                            chars += len(inp)
                        elif isinstance(inp, dict):
                            chars += len(json.dumps(inp, separators=(",", ":")))
                    else:
                        text = block.get("text", "")
                        if text:
                            chars += len(text)
            else:
                chars = 0
            # OpenAI function-calling: the assistant's command lives in the
            # top-level `tool_calls` (or legacy `function_call`) field, NOT in
            # `content` (which is empty/None on a tool-call turn). Anthropic puts
            # the equivalent in a `tool_use` content BLOCK (counted above), but
            # the OpenAI shape was never counted here. That under-counted every
            # tool-based assistant turn to ~0, so the frozen-prefix estimate
            # overshot the real cache boundary and froze the NEWEST delta — which
            # is why OpenAI/Kimi (fireworks) tool harnesses got ~zero compression
            # while text/back-tick harnesses (command in `content`) compressed.
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    chars += len(str(fn.get("name", ""))) + len(str(fn.get("arguments", "")))
            fc = msg.get("function_call")
            if isinstance(fc, dict):
                chars += len(str(fc.get("name", ""))) + len(str(fc.get("arguments", "")))
            # Add overhead for role, block structure, etc.
            chars += 20
            counts.append(max(1, int(chars / 3.5)))
        return counts


def _lineage_snapshot(obj: Any) -> Any:
    """Structural copy of a canonical projection for lineage-chain storage.

    Copies dict/list structure (immutable leaves are shared, so this costs
    structure, not bytes) to isolate the stored chain from downstream in-place
    mutation of the request, and normalizes NaN to a sentinel — ``json.loads``
    accepts bare ``NaN`` and ``NaN != NaN``, so a byte-identical resend would
    otherwise read as a rewritten history on every turn. Values inside opaque
    tool payloads are walked for copying only; no keys are dropped (see
    ``_OPAQUE_PAYLOAD_KEYS``).
    """
    if isinstance(obj, dict):
        return {k: _lineage_snapshot(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_lineage_snapshot(v) for v in obj]
    if isinstance(obj, float) and obj != obj:
        return "\x00nan"
    return obj


class SessionTrackerStore:
    """Manages PrefixCacheTracker instances across sessions.

    Keyed by session ID (from x-headroom-session-id header or computed hash).
    Within one session id, ``resolve_tracker`` keys trackers by conversation
    lineage so concurrent conversations sharing a fallback id do not thrash
    one tracker's frozen-prefix state (#2085).
    Automatically cleans up expired sessions.
    """

    def __init__(self, default_config: PrefixFreezeConfig | None = None):
        self._trackers: dict[str, PrefixCacheTracker] = {}
        self._default_config = default_config or PrefixFreezeConfig()
        self._last_cleanup: float = time.time()
        self._cleanup_interval: float = 60.0  # Cleanup every 60s
        # Conversation lineages per session id: tracker key -> snapshot of the
        # canonicalized messages of the last request that lineage served. The
        # first lineage lives under the bare session id (single-conversation
        # sessions behave exactly as before); later lineages get unique
        # "<session_id>\x00<n>" keys — NUL cannot appear in an HTTP header
        # value, so a synthetic key can never collide with a client-supplied
        # x-headroom-session-id.
        self._lineages: dict[str, OrderedDict[str, list[Any]]] = {}
        self._lineage_counter = itertools.count(1)

    def get_or_create(self, session_id: str, provider: str) -> PrefixCacheTracker:
        """Get existing tracker or create a new one for this session."""
        self._maybe_cleanup()

        if session_id in self._trackers:
            tracker = self._trackers[session_id]
            # Snapshot idle-since-last-response BEFORE bumping the access clock,
            # so the net-cost/TTL gate sees the true gap (see the attribute's
            # docstring in PrefixCacheTracker.__init__).
            tracker._idle_seconds_at_fetch = max(0.0, time.time() - tracker._last_activity)
            tracker._last_activity = time.time()
            return tracker

        tracker = PrefixCacheTracker(provider, self._default_config)
        tracker._idle_seconds_at_fetch = 0.0  # cold start: nothing cached to lapse
        self._trackers[session_id] = tracker
        return tracker

    def resolve_tracker(
        self,
        session_id: str,
        provider: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> PrefixCacheTracker:
        """Resolve the tracker for THIS conversation within a session id (#2085).

        Concurrent conversations often share a fallback session id — same
        model + system prompt covers a Claude Code session together with its
        parallel subagents, or several sessions in one workspace. On a shared
        tracker their interleaved histories cross-contaminate the
        frozen-prefix state: freeze never stabilizes and the provider prompt
        cache is re-written on nearly every call.

        Lineage resolution keys trackers by conversation content instead:
        reuse the tracker whose previous request messages are a prefix of the
        incoming history (client histories are append-only, so a
        conversation's next request always extends its previous one); start a
        fresh lineage when the history diverges or was rewritten (client-side
        compaction — the provider cache line is gone then anyway).
        Byte-identical histories (templated fan-outs before they diverge)
        intentionally share a tracker: their provider cache line is identical
        too, so sharing is harmless.

        The session id itself is never altered, so session-sticky state keyed
        on it elsewhere (beta headers, CCR/memory registries, the compression
        cache) is unaffected.

        Args:
            session_id: Session identity from :meth:`compute_session_id`.
            provider: Provider name for a newly created tracker.
            messages: The request's original client messages, as captured
                right after the INPUT_RECEIVED pipeline extension and before
                security-scan/hook/image-compression mutation — the same
                snapshot ``update_from_response`` records, so the chain
                compares like against like across turns. ``None``/empty
                (legacy callers, stub stores in tests) falls back to plain
                :meth:`get_or_create`.

        Returns:
            The ``PrefixCacheTracker`` for this conversation's lineage.
        """
        if not messages or not self._default_config.enabled:
            # No lineage signal, or prefix freeze is disabled (there is no
            # frozen state to protect): legacy one-tracker-per-session-id.
            return self.get_or_create(session_id, provider)

        # Prune expired trackers BEFORE matching, so a dead lineage cannot win
        # the match. This also arms the cleanup interval: the get_or_create
        # calls below cannot re-trigger a prune mid-function, so the family
        # read here stays attached through the stamp at the end.
        self._maybe_cleanup()

        # The repo's canonical cross-turn equivalence, shared with the
        # cache-stable delta path: a moved cache breakpoint, string<->block
        # content sugar, or per-turn transport annotations must not read as
        # a rewritten history. Object comparison plus one structural snapshot
        # per request (~1ms on a 2MB history — same order as the handler's
        # existing request deepcopy; no serialization or hashing).
        canon = _canonicalize_for_prefix_compare(messages)
        if not canon:
            # Degenerate: every message projected away (pure directive
            # content) — no lineage signal to match on.
            return self.get_or_create(session_id, provider)
        snap = _lineage_snapshot(canon)

        family = self._lineages.setdefault(session_id, OrderedDict())

        # Longest recorded chain that prefixes the incoming history wins.
        best_key: str | None = None
        best_len = -1
        for key, chain in family.items():
            if len(chain) > len(snap) or len(chain) <= best_len:
                continue
            if snap[: len(chain)] == chain:
                best_key, best_len = key, len(chain)

        if best_key is None:
            cap = self._default_config.max_lineages_per_session
            if len(family) >= cap:
                # Family is full: over-cap conversations share one overflow
                # tracker instead of evicting an established lineage. Any
                # eviction policy degrades EVERY conversation once the
                # working set exceeds the cap (under round-robin the victim
                # is always the conversation about to arrive), while overflow
                # sharing degrades only the over-cap tail — to exactly the
                # pre-lineage shared-tracker behavior — and established
                # lineages keep their state. A cap <= 0 therefore disables
                # lineage splitting entirely.
                overflow_key = f"{session_id}\x00overflow"
                if overflow_key not in self._trackers:
                    logger.info(
                        "SessionTrackerStore: lineage cap %d reached for session %s; "
                        "over-cap conversations share an overflow tracker (raise "
                        "PrefixFreezeConfig.max_lineages_per_session if this "
                        "workspace genuinely runs more concurrent conversations)",
                        cap,
                        session_id,
                    )
                return self.get_or_create(overflow_key, provider)
            if not family:
                # First lineage rides the bare session id; this also adopts a
                # tracker created earlier via plain get_or_create.
                best_key = session_id
            else:
                best_key = f"{session_id}\x00{next(self._lineage_counter)}"

        # get_or_create (not a private fetch) so test stubs that patch the
        # instance method keep intercepting tracker creation; its internal
        # cleanup is interval-gated and was armed above, so it cannot prune
        # the family before the stamp below.
        tracker = self.get_or_create(best_key, provider)
        family[best_key] = snap
        return tracker

    def compute_session_id(
        self,
        request: Any,
        model: str,
        messages: list[dict[str, Any]],
    ) -> str:
        """Compute a session ID from the request.

        Priority:
        1. x-headroom-session-id header (explicit)
        2. Hash of (model + system prompt) — stable per conversation

        The system prompt is harvested from the LEADING run of ``role:"system"``
        entries in ``messages`` (everything before the first non-system turn).
        Anthropic carries the system prompt as a top-level ``body["system"]``
        field instead, so its handler prepends that as a synthetic
        ``role:"system"`` message before calling this — otherwise every
        Anthropic conversation on the same model would collapse to one session id
        and their session-sticky state (CCR/memory tools, beta headers, frozen
        prefix) would cross-contaminate.

        Only the leading run counts: agentic clients (Claude Code) interleave
        ``role:"system"`` reminder turns INTO the history as it grows (hook
        output, skills lists, truncation notices). Hashing those would rotate
        the session id mid-conversation, orphaning the prefix tracker and every
        other session-sticky subsystem each time a reminder lands (#2085).
        """
        # Check for explicit session header
        if hasattr(request, "headers"):
            session_header = request.headers.get("x-headroom-session-id")
            if session_header:
                return str(session_header)

        # Fall back to hashing model + the leading system-text run.
        system_parts: list[str] = []
        for msg in messages:
            if msg.get("role") != "system":
                break
            content = msg.get("content", "")
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        system_parts.append(block.get("text", ""))

        system_content = json.dumps(system_parts, ensure_ascii=False, separators=(",", ":"))
        key = f"{model}:{system_content}"
        return hashlib.md5(key.encode()).hexdigest()[:16]  # nosec B324

    def _maybe_cleanup(self) -> None:
        """Remove expired trackers periodically."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        expired = [sid for sid, tracker in self._trackers.items() if tracker.is_expired]
        for sid in expired:
            del self._trackers[sid]

        if expired:
            # Keep the lineage index in step with the tracker map.
            for base in list(self._lineages):
                family = self._lineages[base]
                for key in [k for k in family if k not in self._trackers]:
                    del family[key]
                if not family:
                    del self._lineages[base]
            logger.debug("SessionTrackerStore: cleaned up %d expired sessions", len(expired))

        self._last_cleanup = now

    @property
    def active_sessions(self) -> int:
        """Number of active session trackers (one per conversation lineage)."""
        return len(self._trackers)
