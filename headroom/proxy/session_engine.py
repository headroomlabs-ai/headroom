"""Session-turn engine — the single cache-management brain for both modes.

One conversation turn, from the cache's point of view, is always the same
three-step dance regardless of who owns the upstream call:

1. **Prepare** (:func:`prepare_turn`): decide how many leading messages are
   frozen, mark the stable prefix, and swap previously-computed compressed
   bytes into the working copy (``apply_cached`` — "Zone 1").
2. Run the compression pipeline over the prepared input (owned by the
   caller: the proxy handlers wrap it in background/cold-start/backpressure
   orchestration, the sidecar path runs it inline on the executor).
3. **Finalize** (:func:`finalize_turn`): replay last turn's exact
   previously-forwarded/returned prefix over any residual drift the pipeline
   introduced (``overlay_cached_prefix``), so the bytes that leave the
   process are byte-identical to what the provider already cached.

Historically the proxy request handlers (anthropic + openai token mode) and
the sidecar ``/v1/compress`` session path each carried their own inline copy
of steps 1 and 3. This module is the shared implementation: a
cache-management fix landed here reaches BOTH modes at once.

Freeze policies
---------------

The one deliberate behavioural difference between the modes lives in step 1,
and it is a *policy parameter*, not a fork of the code:

``FREEZE_POLICY_CONFIRMED_CLAMP`` — ``min(tracker_frozen, cache_count)``.
    The proxy sees the provider's responses, so ``tracker_frozen`` is the
    provider-confirmed cached prefix (from ``cache_read_input_tokens``).
    Freezing is clamped by BOTH bounds: never past what the provider
    actually has cached (freezing more would forgo compression of content
    that is not yet cache-protected — the #327 posture), and never past what
    the local cache can byte-replay (freezing a message whose entry was
    evicted would pass through raw original bytes).

``FREEZE_POLICY_REPLAYABLE`` — ``max(cache_count, explicit_frozen or 0)``.
    Freeze everything the local cache can byte-replay. Used by callers with
    no provider-confirmed count to clamp against: the sidecar ``/v1/compress``
    endpoint (it never sees the provider's response — whatever it previously
    RETURNED is the provider's cache contract, so every already-returned
    message must come back byte-identical), and the OpenAI proxy token path
    (its tracker feeds cache mode, not token mode). Recompressing an
    already-returned message — even into a *smaller* form — is a bust: the
    drift was observed in practice, and ``overlay_cached_prefix``'s
    non-inflation guard cannot repair a shrunken form (replaying the larger
    original bytes would "inflate" the candidate). Freezing the entire
    locally-replayable prefix eliminates that recompression outright.
    Over-freezing relative to the provider's real cache only forgoes tail
    compression; it can never bust. An explicit ``frozen_message_count``
    from the caller still wins when larger — the caller may know more about
    the provider cache than local state does.

Why the Anthropic proxy path cannot simply adopt the replayable posture: its
provider-confirmed clamp deliberately KEEPS not-yet-cached content
compressible, and its overlay inputs (tracker snapshots) are refreshed on
every response, so drift repair is reliable there. Each posture is correct
for the information its mode actually has.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from headroom.cache.prefix_tracker import overlay_cached_prefix

logger = logging.getLogger(__name__)

FREEZE_POLICY_CONFIRMED_CLAMP = "confirmed_clamp"
FREEZE_POLICY_REPLAYABLE = "replayable"

_FREEZE_POLICIES = (FREEZE_POLICY_CONFIRMED_CLAMP, FREEZE_POLICY_REPLAYABLE)


@dataclass(frozen=True)
class TurnPrep:
    """Result of :func:`prepare_turn`.

    ``frozen_message_count`` is what the pipeline must be told to skip;
    ``pipeline_input`` is the working copy with previously-compressed bytes
    swapped in (never the caller's list — ``apply_cached`` copies).
    """

    frozen_message_count: int
    pipeline_input: list[dict[str, Any]]


@dataclass(frozen=True)
class TurnFinal:
    """Result of :func:`finalize_turn`.

    ``messages`` are the bytes to forward/return; ``replayed`` says whether
    the overlay restored last turn's prefix over pipeline drift; ``tokens``
    is the recount of ``messages`` when a ``count_tokens`` hook was supplied
    and the overlay actually fired (None otherwise — the pipeline's own
    count is still valid when nothing was replaced).
    """

    messages: list[dict[str, Any]]
    replayed: bool
    tokens: int | None = None


def prepare_turn(
    comp_cache: Any,
    messages: list[dict[str, Any]],
    *,
    policy: str,
    tracker_frozen: int | None = None,
    explicit_frozen: int | None = None,
) -> TurnPrep:
    """Freeze decision + stable marking + cached-byte swap for one turn.

    Args:
        comp_cache: the session's ``CompressionCache``.
        messages: the caller's RAW message list (never mutated).
        policy: ``FREEZE_POLICY_CONFIRMED_CLAMP`` or ``FREEZE_POLICY_REPLAYABLE`` —
            see the module docstring for why they differ.
        tracker_frozen: provider-confirmed frozen count (proxy policy only;
            ``None`` means "nothing confirmed" and freezes 0 there).
        explicit_frozen: caller-pinned frozen count (sidecar policy only;
            wins when larger than the locally-derived bound).
    """
    if policy not in _FREEZE_POLICIES:
        raise ValueError(f"unknown freeze policy: {policy!r}")

    cache_count = comp_cache.compute_frozen_count(messages)
    if policy == FREEZE_POLICY_CONFIRMED_CLAMP:
        # Never freeze past the provider-confirmed prefix, and never past
        # what local state can byte-replay.
        frozen = min(tracker_frozen or 0, cache_count)
    else:
        # Freeze the entire locally-replayable prefix; an explicit caller
        # pin may extend it (the caller vouches the provider cached those
        # exact raw bytes, so passing them through untouched is correct).
        frozen = max(cache_count, explicit_frozen or 0)

    comp_cache.mark_stable_from_messages(messages, frozen)
    return TurnPrep(
        frozen_message_count=frozen,
        pipeline_input=comp_cache.apply_cached(messages),
    )


def finalize_turn(
    result_messages: list[dict[str, Any]],
    original_messages: list[dict[str, Any]],
    prev_original: list[dict[str, Any]] | None,
    prev_returned: list[dict[str, Any]] | None,
    *,
    count_tokens: Callable[[list[dict[str, Any]]], int] | None = None,
    confirmed_frozen_count: int | None = None,
) -> TurnFinal:
    """Replay last turn's exact forwarded/returned prefix over pipeline drift.

    ``overlay_cached_prefix`` self-guards (positional alignment, append-only
    shape, non-inflation), so calling this is always safe: when replay is not
    provably correct it returns the pipeline's own output unchanged.

    ``confirmed_frozen_count`` is forwarded to ``overlay_cached_prefix`` as
    the unconditional-replay floor: positions the provider has confirmed
    cached are always replayed byte-identical, while beyond the floor the
    size bound decides between drift repair (a shrinking replay) and letting
    a fresh improvement through (an inflating one). Callers with no
    provider-confirmed count pass None and keep the fully size-bounded
    behavior.

    ``count_tokens`` is invoked only when the overlay actually replaced
    bytes — the pipeline's own token count is still accurate otherwise. A
    failing hook falls back to "no recount" rather than failing the turn.
    """
    final = overlay_cached_prefix(
        result_messages,
        original_messages,
        prev_original,
        prev_returned,
        confirmed_frozen_count=confirmed_frozen_count,
    )
    replayed = final != result_messages
    tokens: int | None = None
    if replayed and count_tokens is not None:
        try:
            tokens = count_tokens(final)
        except Exception as e:
            # Fail-open: the turn still forwards, but the caller keeps the
            # pipeline's count of messages that are NOT being forwarded —
            # tokens_saved accounting is stale for this turn. Loud, not
            # silent: a tokenizer that cannot count the replayed form is a
            # bug worth surfacing even though it must not fail the request.
            logger.warning(
                "finalize_turn: token recount of replayed prefix failed "
                "(%s: %s); keeping the pipeline's pre-overlay count",
                type(e).__name__,
                e,
            )
            tokens = None
    return TurnFinal(messages=final, replayed=replayed, tokens=tokens)
