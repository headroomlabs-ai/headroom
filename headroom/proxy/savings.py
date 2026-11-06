"""One per-request savings record: counterfactual cost minus actual cost.

Every savings surface (/stats, the durable ledger, persistent savings, the
dashboard, the PERF line) is meant to derive from the :class:`SavingsRecord`
this module computes, so no surface prices savings its own way.

Two rulers meet here and are only ever combined through RATIOS:

* Headroom's tokenizer gives per-item counts of the request as the client sent
  it (ORIGINAL) and as Headroom forwarded it (SENT). Both are counted with the
  same tokenizer, so a tokenizer bias cancels in their difference.
* The provider's usage block gives exact billed totals (uncached, cache read,
  cache write) but no per-item detail.

``k = billed / sent`` converts Headroom units into provider units, calibrated
on this request's own content. The counterfactual cache behaviour mirrors the
actual request's: removed tokens in items the previous request already carried
would have been cache reads to the extent the actual request hit its cache
(``warm``); removed tokens in new items would have been misses, written to
cache in the proportion the actual request wrote (``w``). Old items whose sent
copy Headroom changed since the previous request (``bust``) are misses that
Headroom caused, charged back against the saving.

Output tokens are identical in the counterfactual by construction; output
shaping is measured separately against its holdout and stays out of this
number.

``headroom.pricing`` is imported inside functions, as elsewhere on the proxy
path: its package ``__init__`` pulls in litellm (~4s), which must not land on
proxy startup.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from headroom.log_safety import describe_exception, safe_id

if TYPE_CHECKING:
    from headroom.pricing.counterfactual import CacheMix, CacheRates

logger = logging.getLogger(__name__)

#: OpenAI caches prompts of at least 1024 tokens, in 128-token blocks.
OPENAI_MIN_CACHEABLE_TOKENS = 1024
OPENAI_CACHE_BLOCK_TOKENS = 128
#: Used when the catalog has no ``prompt_cache_min_tokens`` for a model.
DEFAULT_MIN_CACHEABLE_TOKENS = 1024


@dataclass(frozen=True, slots=True)
class ItemCounts:
    """One request item (tools block, system, or one input/message item).

    Hashes identify the item's content; counts are in Headroom tokens. Lists of
    items are aligned by position between ORIGINAL and SENT, so an item
    Headroom dropped entirely keeps its slot with ``sent_hash=""`` and
    ``sent_tokens=0``. Tool-schema deferral is removal inside the tools item.
    """

    original_hash: str
    original_tokens: int
    sent_hash: str
    sent_tokens: int


@dataclass(frozen=True)
class ItemMeasure:
    """The Headroom-unit quantities the savings formula needs."""

    #: ``S``: tokens of the whole sent body.
    sent_tokens: int
    #: ``R_old``: removed tokens (signed: an item Headroom grew counts against
    #: it) in items within the longest prefix of the ORIGINAL body identical to
    #: the previous ORIGINAL body.
    removed_old: int
    #: ``R_new``: removed tokens in every other item, signed likewise.
    removed_new: int
    #: ``P_S``: tokens of the longest prefix of the SENT body identical to the
    #: previous SENT body.
    sent_shared_prefix: int
    #: Sent tokens of old items that fell outside the sent shared prefix: cache
    #: misses caused only by Headroom changing what it had sent before.
    bust: int
    #: False when the old/new split is approximate (the totals stay exact):
    #: Headroom changed the item count and the original/sent pairing had to be
    #: approximated, or the baseline was lost (``baseline_lost``).
    aligned: bool = True
    #: True when the ledger lost what this conversation last sent (its
    #: baseline or an in-flight request was evicted under the budget, or its
    #: baseline may be truncated), so ``removed_new`` may hold old items. Such
    #: a measure is also ``aligned=False``. Its record prices the new removal
    #: at the cache-read rate, the lowest any input token costs, so the lost
    #: knowledge can only understate the saving. A conversation's first
    #: request is not lost: nothing was sent before, so everything is new.
    baseline_lost: bool = False
    #: Tokens Headroom added to new items (the negative part of
    #: ``removed_new``), so the read-rate floor covers only what it removed.
    grown_new: int = 0
    #: Which measured request this is, for :meth:`ItemLedger.settle`.
    ticket: int = field(default=0, compare=False)

    @property
    def removed(self) -> int:
        return self.removed_old + self.removed_new


@dataclass(frozen=True)
class SavingsRecord:
    """What one request saved, in dollars and in both token rulers."""

    saved_usd: float
    actual_input_cost_usd: float
    counterfactual_input_cost_usd: float
    #: Removed tokens in Headroom units (``R``).
    removed_tokens: int
    #: The same removal in provider units (``k * R``).
    saved_tokens: float
    k: float
    warm: float
    write_share: float
    #: True when usage was missing (aborted stream) or summed over server-side
    #: iterations, so the cache split could not be read from this request.
    estimated: bool = False
    #: False when the catalog has no price for the model.
    priced: bool = True


def _common_prefix_len(a: Sequence[str], b: Sequence[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def measure_items(
    items: Sequence[ItemCounts],
    previous: Sequence[ItemCounts] | None,
) -> ItemMeasure:
    """Split one request's removal into old/new items against the previous request.

    ``previous`` is the same conversation's previous request, or ``None`` on its
    first request, where everything is new and nothing is shared.
    """
    n_old = _common_prefix_len(
        [i.original_hash for i in items], [i.original_hash for i in previous or ()]
    )
    n_sent_shared = _common_prefix_len(
        [i.sent_hash for i in items], [i.sent_hash for i in previous or ()]
    )
    removed = [i.original_tokens - i.sent_tokens for i in items]
    sent = [max(0, i.sent_tokens) for i in items]
    return ItemMeasure(
        sent_tokens=sum(sent),
        removed_old=sum(removed[:n_old]),
        removed_new=sum(removed[n_old:]),
        sent_shared_prefix=sum(sent[:n_sent_shared]),
        bust=sum(sent[n_sent_shared:n_old]),
        grown_new=-sum(r for r in removed[n_old:] if r < 0),
    )


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


#: One request item as (identity text, billable text): the identity is hashed
#: to tell items apart, the billable text is what gets tokenized.
Item = tuple[str, str]


def _item(value: Any) -> Item:
    identity = _dump(value)
    if (
        isinstance(value, dict)
        and value.get("type") == "reasoning"
        and "encrypted_content" in value
    ):
        # The provider bills the reasoning it decrypts, not the ciphertext,
        # which would otherwise be most of a long Codex transcript's count.
        billable = _dump({k: v for k, v in value.items() if k != "encrypted_content"})
        return identity, billable
    return identity, identity


def responses_items(body: dict[str, Any]) -> list[Item]:
    """A ``/v1/responses`` body as items in prompt order.

    Tools, then instructions, then each ``input`` item: the order the provider
    caches the prompt in, which is what the shared-prefix comparison needs.
    Codex's ``additional_tools`` input items stay where the client put them.
    """
    items = [_item(body.get("tools") or []), _item(body.get("instructions") or "")]
    # OpenAI-compatible upstreams accept ``messages`` in place of ``input``.
    raw = body.get("input") if body.get("input") is not None else body.get("messages")
    if isinstance(raw, list):
        items.extend(_item(item) for item in raw)
    elif raw:
        items.append(_item(raw))
    return items


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=8).hexdigest()


#: Items kept across all conversations' baselines and requests awaiting their
#: outcome, combined. A long Codex session carries 1000+ items; bounding items,
#: not conversations, bounds memory: up to twice this while one conversation's
#: baseline and its in-flight request exceed it (see :class:`ItemLedger`).
MAX_STORED_ITEMS = 100_000

#: Conversations remembered as having lost their baseline, a memory bound
#: (one short key each). Past it the ledger can no longer tell a first request
#: from a lost one and treats every request without a baseline as lost: priced
#: at the cache-read floor, which can understate a saving but never overstate it.
MAX_LOST_KEYS = 100_000


def _align(
    original: list[tuple[str, int]],
    sent: list[tuple[str, int]],
    previous: Sequence[ItemCounts] | None,
) -> tuple[list[ItemCounts], bool]:
    """Pair original and sent items, and say whether the pairing is exact.

    Same item count: by position (compression rewrites items in place). A
    different count (a hook folded or dropped items, or Headroom inserted
    some): identical items at either end pair up; the unmatched originals in
    between are removed, and the unmatched sent items in between are folded
    into the first of their slots. The totals stay exact either way; only
    which items count as old is approximate, hence ``aligned=False``. An item
    compressed in place still pairs at the head when the previous request
    paired the same original with the same sent copy at that position.

    Equal counts pair by position even if a fold and an insert cancel out.
    """
    if len(original) == len(sent):
        return [ItemCounts(oh, ot, sh, st) for (oh, ot), (sh, st) in zip(original, sent)], True
    limit = min(len(original), len(sent))
    prev = previous or ()

    def same_as_before(i: int) -> bool:
        return (
            i < len(prev)
            and original[i][0] == prev[i].original_hash
            and sent[i][0] == prev[i].sent_hash
        )

    head = 0
    while head < limit and (original[head][0] == sent[head][0] or same_as_before(head)):
        head += 1
    tail = 0
    while (
        tail < limit - head
        and original[len(original) - 1 - tail][0] == sent[len(sent) - 1 - tail][0]
    ):
        tail += 1
    mid_original = original[head : len(original) - tail]
    mid_sent = sent[head : len(sent) - tail]
    if len(mid_sent) == 1:
        folded_hash = mid_sent[0][0]
    else:
        folded_hash = _digest("\x00".join(h for h, _ in mid_sent)) if mid_sent else ""
    folded_tokens = sum(t for _, t in mid_sent)
    middle = [ItemCounts(oh, ot, "", 0) for oh, ot in mid_original] or [ItemCounts("", 0, "", 0)]
    middle[0] = ItemCounts(
        middle[0].original_hash, middle[0].original_tokens, folded_hash, folded_tokens
    )
    pairs = [ItemCounts(oh, ot, sh, st) for (oh, ot), (sh, st) in zip(original[:head], sent[:head])]
    pairs += middle
    pairs += [ItemCounts(h, t, h, t) for h, t in original[len(original) - tail :]]
    return pairs, False


class ItemLedger:
    """The previous request of each conversation, for the old/new item split.

    One instance per process. A request is measured against the conversation's
    last SUCCESSFUL request and becomes the baseline only once its own outcome
    succeeds (:meth:`settle`): a failed call never reached the provider's cache.
    Each measure carries a ticket, issued in measure order, so a slow earlier
    turn settling late cannot replace a newer turn's baseline. Items re-sent
    from the baseline reuse its token counts, so a long transcript is
    tokenized once rather than every turn.

    Baselines and requests awaiting their outcome share one budget,
    :data:`MAX_STORED_ITEMS`. Past it, whichever is least recent goes first,
    by ticket: a pending request by its own ticket (its slot is reserved when
    the ticket is issued, so a slow measure keeps its place), a baseline by
    the last ticket that used it. Settling counts as a use. A request being
    admitted never evicts its own conversation's baseline: that baseline is
    replaced only when a newer request for the conversation succeeds. When
    only that baseline and one pending request are left, both are kept, even
    past the budget, so a conversation larger than half the budget still
    advances; memory then peaks at twice :data:`MAX_STORED_ITEMS` items.

    A request is kept only up to its first :data:`MAX_STORED_ITEMS` items: the
    comparison is a prefix, so the next request still gets the shared-prefix
    credit for them.

    Whatever is evicted, its conversation is remembered as lost, so its next
    request is marked ``baseline_lost`` (and ``aligned=False``): its record
    prices the new removal at the cache-read floor instead of as new, at the
    miss rate. So is a request measured against a baseline of exactly
    :data:`MAX_STORED_ITEMS` items, which may be truncated. A conversation's
    first request is new, not lost. The memory of losses is bounded by
    :data:`MAX_LOST_KEYS`; past it, every request without a baseline is
    treated as lost, and a forgotten loss's stale baseline is dropped with it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # key -> (settled ticket, items, last ticket that used it), oldest use first.
        self._previous: OrderedDict[str, tuple[int, list[ItemCounts], int]] = OrderedDict()
        # ticket -> (key, items), in ticket order.
        self._pending: OrderedDict[int, tuple[str, list[ItemCounts]]] = OrderedDict()
        # key -> newest ticket whose items were evicted, oldest loss first.
        self._lost: OrderedDict[str, int] = OrderedDict()
        # True once a loss was forgotten: a missing baseline may then be a lost one.
        self._lost_overflowed = False
        self._stored = 0
        self._pending_items = 0
        self._seq = 0

    def measure(
        self,
        key: str,
        original: Sequence[Item],
        sent: Sequence[Item],
        count: Callable[[str], int],
    ) -> ItemMeasure:
        """Measure one request against ``key``'s baseline; settle it later."""
        with self._lock:
            self._seq += 1
            seq = self._seq
            entry = self._previous.get(key)
            if entry is not None:
                self._previous[key] = (entry[0], entry[1], seq)
                self._previous.move_to_end(key)
            lost = key in self._lost or (entry is None and self._lost_overflowed)
            self._pending[seq] = (key, [])
        previous = entry[1] if entry else None
        known: dict[str, int] = {}
        for item in previous or ():
            known[item.original_hash] = item.original_tokens
            known[item.sent_hash] = item.sent_tokens

        def counted(items: Sequence[Item]) -> list[tuple[str, int]]:
            out = []
            for identity, billable in items:
                digest = _digest(identity)
                tokens = known.get(digest)
                if tokens is None:
                    tokens = known[digest] = int(count(billable))
                out.append((digest, tokens))
            return out

        try:
            items, aligned = _align(counted(original), counted(sent), previous)
        except BaseException:
            # The caller logs and never settles: release the reserved slot.
            with self._lock:
                self._pending.pop(seq, None)
            raise
        if previous is not None and len(previous) == MAX_STORED_ITEMS:
            # Possibly a truncated baseline: items past it may be old.
            lost = True
        with self._lock:
            if seq in self._pending:  # not already evicted while counting
                kept = items[:MAX_STORED_ITEMS]
                self._pending[seq] = (key, kept)
                self._pending_items += len(kept)
                self._evict(key)
        return replace(
            measure_items(items, previous),
            aligned=aligned and not lost,
            baseline_lost=lost,
            ticket=seq,
        )

    def settle(self, measure: ItemMeasure, *, succeeded: bool) -> None:
        """Make a measured request its conversation's baseline if it succeeded."""
        seq = measure.ticket
        with self._lock:
            pending = self._pending.pop(seq, None)
            if pending is None:
                return
            key, items = pending
            self._pending_items -= len(items)
            if not succeeded:
                return
            current = self._previous.get(key)
            if current is not None and current[0] > seq:
                return
            # Moving items from pending to stored never grows the total.
            self._stored -= len(current[1]) if current else 0
            self._previous[key] = (seq, items, self._seq)
            self._previous.move_to_end(key)
            self._stored += len(items)
            if self._lost.get(key, seq) < seq:
                # Newer than everything evicted: the baseline is whole again.
                del self._lost[key]

    def _evict(self, admitting: str) -> None:
        """Forget the least recent entries until the combined budget holds.

        Called with the lock held, right after admitting a request for
        ``admitting``, whose baseline is never a candidate. Each pass removes
        one entry until only that baseline and one pending request are left:
        each holds at most :data:`MAX_STORED_ITEMS` items, so they stay.
        """
        while self._stored + self._pending_items > MAX_STORED_ITEMS:
            oldest_ticket = next(iter(self._pending), None)
            oldest_key = next((k for k in self._previous if k != admitting), None)
            if oldest_key is None and len(self._pending) <= 1:
                return
            # On a tie the baseline goes: the request that used it is newer.
            if oldest_key is not None and (
                oldest_ticket is None or self._previous[oldest_key][2] <= oldest_ticket
            ):
                settled, evicted, _ = self._previous.pop(oldest_key)
                self._stored -= len(evicted)
                self._mark_lost(oldest_key, settled)
            else:
                ticket, (key, dropped) = self._pending.popitem(last=False)
                self._pending_items -= len(dropped)
                self._mark_lost(key, ticket)

    def _mark_lost(self, key: str, ticket: int) -> None:
        """Remember that ``key``'s items up to ``ticket`` were evicted (lock held)."""
        self._lost[key] = max(ticket, self._lost.get(key, 0))
        self._lost.move_to_end(key)
        while len(self._lost) > MAX_LOST_KEYS:
            forgotten, _ = self._lost.popitem(last=False)
            if not self._lost_overflowed:
                self._lost_overflowed = True
                logger.info(
                    "savings: lost-baseline memory full (MAX_LOST_KEYS=%d); requests "
                    "without a baseline now price at the cache-read floor",
                    MAX_LOST_KEYS,
                )
            # Unmarked, a stale baseline would price old items as new: drop it,
            # so the conversation counts as lost under the overflow rule.
            stale = self._previous.pop(forgotten, None)
            if stale is not None:
                self._stored -= len(stale[1])


_ledger = ItemLedger()


def get_item_ledger() -> ItemLedger:
    """Process-wide ledger: one conversation spans many requests and handlers."""
    return _ledger


def prefix_key(conversation: str | None, model: str, original: Sequence[Item]) -> str:
    """Which previous request a request is compared against.

    The conversation's explicit identity (``savings_conversation_key``) plus
    the tools and instructions: a side request in the same session (compaction,
    review) carries a different prefix and must not replace the conversation's
    previous request. Without an identity, the model plus the first input item
    stand in. Requests that collide on the key overwrite each other's previous
    request, shortening the shared prefix: more of the removal is then priced
    as new, at the miss rate, which overstates the saving.
    """
    head = "\x00".join(identity for identity, _ in original[:2])
    if conversation:
        # The model too: the provider's cache is per model, and so are counts.
        return _digest("\x00".join([conversation, model, head]))
    return _digest("\x00".join([model, *(identity for identity, _ in original[:3])]))


def _input_cost(
    rates: CacheRates, *, uncached: float, read: float, write_5m: float, write_1h: float
) -> float:
    return (
        uncached * rates.uncached
        + read * rates.read
        + write_5m * rates.write_5m
        + write_1h * rates.write_1h
    )


def _min_cacheable_tokens(model: str, provider: str | None) -> int:
    if provider == "openai":
        return OPENAI_MIN_CACHEABLE_TOKENS
    from headroom.pricing.counterfactual import catalog_min_cacheable_tokens

    return catalog_min_cacheable_tokens(model) or DEFAULT_MIN_CACHEABLE_TOKENS


def _unpriced(measure: ItemMeasure, *, k: float, estimated: bool) -> SavingsRecord:
    return SavingsRecord(
        saved_usd=0.0,
        actual_input_cost_usd=0.0,
        counterfactual_input_cost_usd=0.0,
        removed_tokens=measure.removed,
        saved_tokens=k * measure.removed,
        k=k,
        warm=0.0,
        write_share=0.0,
        estimated=estimated,
        priced=False,
    )


def compute_savings_record(
    measure: ItemMeasure,
    usage: CacheMix | None,
    *,
    model: str,
    provider: str | None = None,
    estimated: bool = False,
) -> SavingsRecord:
    """Price one request's removal as ``counterfactual_cost - actual_cost``.

    ``usage`` is the provider's billed input as a :class:`CacheMix` (normalized
    here, so an inferred OpenAI write counts as the uncached slice it is).
    ``None`` or an empty mix means the provider reported nothing (an aborted
    stream); the record then prices at list with ``k = 1`` and is marked
    ``estimated``.

    For a request the provider ran as several server-side iterations, pass the
    FIRST iteration's usage when it is available (``estimated=False``); the
    later iterations are a cost the caller books separately. Pass the summed
    usage with ``estimated=True`` only when per-iteration usage is not
    available: the sum describes more prompts than the one sent body, so
    ``k`` is not calibrated from it (``k = 1``) and only its cache ratios are
    used. ``provider`` must be given for OpenAI's 128-token cache blocks to
    apply.

    Never raises: a savings figure must not be able to fail a request.
    """
    try:
        record = _compute(measure, usage, model=model, provider=provider, estimated=estimated)
        # An approximate item pairing leaves the totals exact but not the
        # old/new split the price depends on.
        return record if measure.aligned else replace(record, estimated=True)
    except Exception as exc:  # noqa: BLE001 - pricing must never fail a request
        logger.debug(
            "savings: record failed for model=%s: %s", safe_id(model), describe_exception(exc)
        )
        return _unpriced(measure, k=1.0, estimated=True)


def _compute(
    measure: ItemMeasure,
    usage: CacheMix | None,
    *,
    model: str,
    provider: str | None,
    estimated: bool,
) -> SavingsRecord:
    from headroom.pricing.counterfactual import LONG_CONTEXT_THRESHOLD_TOKENS, resolve_rates

    def rates_for(billed: float, *, for_billing: bool = False) -> CacheRates | None:
        return resolve_rates(
            model,
            long_context=billed > LONG_CONTEXT_THRESHOLD_TOKENS,
            provider=provider,
            for_billing=for_billing,
        )

    removed = measure.removed
    # With the baseline lost, the new items may be old ones the provider had
    # cached. Their removal prices at the read rate, the floor of any input
    # token: the cache ratio and minimum below describe a prefix the ledger no
    # longer knows. Tokens Headroom added to new items stay at the miss rate.
    floor = measure.removed_new + measure.grown_new if measure.baseline_lost else 0
    mix = usage.normalized() if usage is not None else None
    if mix is None or mix.billed <= 0:
        actual_rates = rates_for(measure.sent_tokens)
        cf_rates = rates_for(measure.sent_tokens + removed)
        if actual_rates is None or cf_rates is None:
            return _unpriced(measure, k=1.0, estimated=True)
        actual = measure.sent_tokens * actual_rates.uncached
        counterfactual = (
            measure.sent_tokens + removed - floor
        ) * cf_rates.uncached + floor * cf_rates.read
        return SavingsRecord(
            saved_usd=counterfactual - actual,
            actual_input_cost_usd=actual,
            counterfactual_input_cost_usd=counterfactual,
            removed_tokens=removed,
            saved_tokens=float(removed),
            k=1.0,
            warm=0.0,
            write_share=0.0,
            estimated=True,
        )

    billed = mix.billed
    writes = mix.write_5m + mix.write_1h
    k = billed / measure.sent_tokens if measure.sent_tokens > 0 and not estimated else 1.0
    hit = mix.read / billed
    sigma = measure.sent_shared_prefix / measure.sent_tokens if measure.sent_tokens > 0 else 0.0
    warm = min(1.0, hit / sigma) if sigma > 0 else 0.0
    misses = mix.uncached + writes
    w = writes / misses if misses > 0 else 0.0

    # The counterfactual must obey its own cache minimum. The catalog's
    # minimum is used unless this request was demonstrably cached or written
    # below it; then the catalog is wrong and no minimum is applied, the
    # direction that cannot inflate the saving.
    min_cacheable = _min_cacheable_tokens(model, provider)
    proven = mix.read if mix.read > 0 else (billed if writes > 0 else min_cacheable)
    threshold = min_cacheable if proven >= min_cacheable else 0
    # Removed or busted old tokens could only have been reads if the
    # counterfactual's old prefix was itself cacheable.
    raw_warm = warm
    if k * (measure.sent_shared_prefix + measure.removed_old) < threshold:
        warm = 0.0

    # The counterfactual's buckets are built whole, not as deltas clamped onto
    # the actual ones: a counterfactual smaller than the request (an old item
    # Headroom grew) must not inherit reads or writes it was too small to make.
    # Its reads are the actual reads plus the removed/busted old tokens that
    # would have hit, minus the reads a grown old item contributed (those go
    # whatever the prefix size), never more than its whole billed input, and
    # none below its cache minimum. ``cf_billed`` is ``k * original`` when k is
    # calibrated; the clamp matters only for summed (estimated) usage.
    cf_billed = max(0.0, billed + k * removed)
    gained = k * (max(0, measure.removed_old) + measure.bust) * warm
    lost = k * min(0, measure.removed_old) * raw_warm
    cf_read = min(max(0.0, mix.read + gained + lost), cf_billed)
    if cf_read < threshold:
        # Never fires when nothing changed: the actual reads are 0 or proven
        # at least the threshold.
        cf_read = 0.0
    if provider == "openai" and cf_read != mix.read:
        # Hits come in whole 128-token blocks: round down, but a counterfactual
        # that adds reads keeps at least the actual ones. The ``!=`` keeps an
        # unchanged request exactly at the actual reads.
        blocks = (cf_read // OPENAI_CACHE_BLOCK_TOKENS) * OPENAI_CACHE_BLOCK_TOKENS
        cf_read = max(blocks, mix.read) if cf_read > mix.read else blocks
    if floor > 0:
        cf_read = min(cf_read + k * floor, cf_billed)
    # Everything else missed, written in the proportion the request wrote,
    # unless the counterfactual is smaller than anything shown cacheable: a
    # write it could not make would inflate the saving.
    cf_miss = cf_billed - cf_read
    cf_write = cf_miss * w if cf_billed >= min(min_cacheable, proven) else 0.0
    w5_share = mix.write_5m / writes if writes > 0 else 1.0
    cf_buckets = {
        "uncached": cf_miss - cf_write,
        "read": cf_read,
        "write_5m": cf_write * w5_share,
        "write_1h": cf_write * (1.0 - w5_share),
    }
    actual_buckets = {
        "uncached": float(mix.uncached),
        "read": float(mix.read),
        "write_5m": float(mix.write_5m),
        "write_1h": float(mix.write_1h),
    }

    # Tokens the request was billed for price at billing rates on both sides
    # (the counterfactual's at its own context tier); only the tokens a
    # counterfactual bucket holds beyond the actual one price at the
    # savings-estimate rates. Every priced quantity is nonnegative.
    actual_rates = rates_for(billed, for_billing=True)
    cf_billing_rates = rates_for(cf_billed, for_billing=True)
    cf_rates = rates_for(cf_billed)
    if actual_rates is None or cf_billing_rates is None or cf_rates is None:
        return _unpriced(measure, k=k, estimated=estimated)

    actual = _input_cost(actual_rates, **actual_buckets)
    counterfactual = _input_cost(
        cf_billing_rates, **{b: min(n, actual_buckets[b]) for b, n in cf_buckets.items()}
    ) + _input_cost(cf_rates, **{b: max(0.0, n - actual_buckets[b]) for b, n in cf_buckets.items()})
    return SavingsRecord(
        saved_usd=counterfactual - actual,
        actual_input_cost_usd=actual,
        counterfactual_input_cost_usd=counterfactual,
        removed_tokens=removed,
        saved_tokens=k * removed,
        k=k,
        warm=warm,
        write_share=w,
        estimated=estimated,
    )


__all__ = [
    "Item",
    "ItemCounts",
    "ItemLedger",
    "ItemMeasure",
    "SavingsRecord",
    "compute_savings_record",
    "get_item_ledger",
    "measure_items",
    "prefix_key",
    "responses_items",
]
