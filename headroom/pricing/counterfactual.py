"""Pricing for tokens Headroom kept OFF the wire.

Every savings figure Headroom reports is a counterfactual: *what would this
request have cost had we not removed those tokens?* That question has one
correct answer and one tempting wrong one.

The wrong one — ``tokens_saved * input_cost_per_token`` — is true exactly once,
for the first request of a cache window. After that the tokens we removed would
have been served from the provider's prompt cache at a fraction of list price
(Anthropic 0.10x, OpenAI 0.50x, Gemini 0.10x), so list pricing overstates the
saving by up to an order of magnitude. Measured on a real 238M-input-token
corpus that was 87.8% cache reads, list pricing valued the removed tokens at
2.73x the rate actually paid for the tokens that *were* sent.

The right answer depends on WHERE in the request the removed tokens sat, because
a prompt is not uniformly priced:

``Region.PREFIX``
    The stable head of the prompt — tool definitions above all, since providers
    serialize the tool array before the system prompt and the turns. These are
    the most cacheable tokens in the entire request: written once, read on every
    subsequent turn until the TTL lapses. On a warm turn they are cache *reads*,
    not a pro-rata slice of the request's mix. Splitting them proportionally
    hands them a share of the 1.25x write bucket they would never have occupied
    — on a 90/10 read/write request that prices them at 0.215x instead of 0.10x,
    still ~2x high. :func:`split_tokens` therefore fills the read bucket FIRST
    (``min(tokens, mix.read)``) and only spills the remainder into write/list.

``Region.LIVE_ZONE``
    The freshly appended delta at the tail. Handlers freeze the cached prefix
    byte-for-byte for prefix-cache safety and compress only this region, so the
    tokens removed here could never have been billed as cache reads — the read
    bucket belongs to the frozen prefix Headroom did not touch. Pricing these by
    the whole-request mix values a warm turn's compression at ~0.1x, an order of
    magnitude UNDER what the provider would have charged. The read bucket is
    excluded outright.

Both regions inherit the observed request's own cache mix for the buckets they
do span, which is what makes TTL expiry fall out of real traffic instead of
being modelled: an expired window shows up as a write-heavy mix on the actual
request, so the counterfactual re-write is measured per request. That matters
most for providers that publish no TTL at all (OpenAI) — there is nothing to
model, only something to observe.

Provider- and harness-agnostic by construction. Rates come from LiteLLM's
per-model catalog; the structural multipliers in :mod:`headroom.pricing.cache_ttl`
fill only the gaps the catalog leaves (notably the 1h write rate above the
200k-context threshold, which no catalog row publishes). A request that reports
no usable cache breakdown at all — the MCP tool path, a non-reporting gateway, a
streaming turn with no usage frame — prices at list and says so via
:attr:`PricedSavings.basis` rather than inventing a mix. Honest and labelled
beats precise and wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any

from headroom.pricing.cache_ttl import CACHE_WRITE_MULTIPLIERS

logger = logging.getLogger(__name__)

#: Context size at which the major catalogs publish a second, higher price tier
#: (LiteLLM spells it ``*_above_200k_tokens``). A request's billed prompt is
#: compared against this to pick which rate applies.
LONG_CONTEXT_THRESHOLD_TOKENS = 200_000

#: Provider-level cache discount ratios, as a fraction of the base input price.
#: FALLBACK ONLY — the per-model LiteLLM catalog is always preferred, because
#: these go stale per model and per context tier. Kept here (rather than in
#: ``proxy/cost.py``, where they used to live) so every consumer shares one copy.
CACHE_ECONOMICS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "read_multiplier": 0.1,
        "write_multiplier": 1.25,
        "label": "Explicit breakpoints, 5-min TTL",
    },
    "openai": {
        "read_multiplier": 0.5,
        "write_multiplier": 1.0,
        "label": "Automatic, no TTL control",
    },
    "gemini": {
        "read_multiplier": 0.1,
        "write_multiplier": 1.0,
        "label": "Explicit cachedContent, configurable TTL",
    },
    "bedrock": {
        "read_multiplier": 0.1,
        "write_multiplier": 1.25,
        "label": "Same as Anthropic (Bedrock)",
    },
}


class Region(str, Enum):
    """Where in the request the counterfactual tokens would have sat.

    ``str`` mixin so a Region survives a JSON round-trip through the ledger and
    the telemetry payloads without a custom encoder.
    """

    #: Stable head of the prompt: tool definitions, system prompt, frozen turns.
    #: Cache reads first — see the module docstring.
    PREFIX = "prefix"
    #: Freshly appended tail. Never a cache read.
    LIVE_ZONE = "live_zone"


#: Why a set of rates is what it is, most to least trustworthy. Surfaced on
#: every priced figure so a reader can tell a measured number from a guess.
BASIS_CATALOG = "catalog"
BASIS_CATALOG_TTL_RATIO = "catalog+ttl-ratio"
BASIS_PROVIDER_RATIO = "provider-ratio"
BASIS_LIST = "list"
BASIS_NO_MIX = "no-mix"
BASIS_UNPRICED = "unpriced"

#: Ordered worst-to-best. Blending figures of differing provenance reports the
#: WEAKEST basis present, so one unpriced request cannot launder a total into
#: looking catalog-grade.
_BASIS_RANK: dict[str, int] = {
    BASIS_UNPRICED: 0,
    BASIS_NO_MIX: 1,
    BASIS_LIST: 2,
    BASIS_PROVIDER_RATIO: 3,
    BASIS_CATALOG_TTL_RATIO: 4,
    BASIS_CATALOG: 5,
}


def weakest_basis(*bases: str | None) -> str:
    """Return the least-trustworthy basis among ``bases``.

    An aggregate is only as sound as its worst input, so totals report that
    rather than the basis of whichever row happened to be priced best.
    """
    present = [b for b in bases if b]
    if not present:
        return BASIS_UNPRICED
    return min(present, key=lambda b: _BASIS_RANK.get(b, 0))


def _coerce_int(value: Any) -> int:
    """Non-negative int, defaulting to 0. Never raises."""
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class CacheMix:
    """One request's provider-reported input breakdown.

    Every field is optional because provider coverage genuinely differs, and the
    difference is load-bearing rather than incidental (see
    ``RequestOutcome``'s cache block, which this mirrors):

    * Anthropic / Bedrock / Vertex — all five: read, write, the 5m/1h write
      split, and uncached.
    * OpenAI — read and uncached only. Its ``cached_tokens`` has no companion
      write counter, so handlers DERIVE a write figure from the uncached slice
      and set ``inferred_write``. That derived value is the same tokens as
      ``uncached``, so counting it as a write would double-count it AND apply a
      write premium OpenAI does not charge. :meth:`normalized` drops it.
    * Gemini — read only.
    * MCP tool path, non-reporting gateways, streaming turns with no usage
      frame — nothing at all, which :meth:`has_signal` reports as False.
    """

    read: int = 0
    write_5m: int = 0
    write_1h: int = 0
    #: Total writes when the provider reports no TTL split. Only the excess over
    #: ``write_5m + write_1h`` is used, so passing all three is safe.
    write_total: int = 0
    uncached: int = 0
    #: True when ``write_*`` was derived by Headroom rather than billed by the
    #: provider. Such writes are dropped entirely — see the class docstring.
    inferred_write: bool = False

    @classmethod
    def from_usage(
        cls,
        *,
        cache_read_tokens: Any = 0,
        cache_write_tokens: Any = 0,
        cache_write_5m_tokens: Any = 0,
        cache_write_1h_tokens: Any = 0,
        uncached_input_tokens: Any = 0,
        cache_inferred: Any = False,
    ) -> CacheMix:
        """Build from the keyword names every proxy call site already uses.

        Keeps the handlers, ``RequestOutcome`` and the metrics path spelling
        these the one way they always have, so adopting cache-aware pricing is a
        constructor call rather than a rename across 18 emit sites.
        """
        return cls(
            read=_coerce_int(cache_read_tokens),
            write_5m=_coerce_int(cache_write_5m_tokens),
            write_1h=_coerce_int(cache_write_1h_tokens),
            write_total=_coerce_int(cache_write_tokens),
            uncached=_coerce_int(uncached_input_tokens),
            inferred_write=bool(cache_inferred),
        )

    def normalized(self) -> CacheMix:
        """Resolve the write buckets into a consistent, billable shape.

        Two corrections, both of which change the money:

        1. ``inferred_write`` zeroes every write bucket. A derived write is the
           same tokens as ``uncached`` and carries no provider write premium.
        2. A provider that reports a write total but no TTL split (or splits
           that undershoot the total) has the remainder attributed to 5m — the
           default TTL a request gets when it does not ask for the extended one.
           Attributing it to 1h instead would inflate the counterfactual by 60%
           of the write premium on traffic that never asked for 1h.
        """
        if self.inferred_write:
            return CacheMix(
                read=self.read,
                uncached=self.uncached,
                inferred_write=True,
            )
        split = self.write_5m + self.write_1h
        remainder = max(0, self.write_total - split)
        return CacheMix(
            read=self.read,
            write_5m=self.write_5m + remainder,
            write_1h=self.write_1h,
            write_total=split + remainder,
            uncached=self.uncached,
        )

    @property
    def billed(self) -> int:
        """Total billed input tokens implied by this mix, after normalization."""
        n = self.normalized()
        return n.read + n.write_5m + n.write_1h + n.uncached

    def has_signal(self) -> bool:
        """True when the provider reported enough to price a counterfactual.

        False means list pricing is the only honest answer — reported as
        ``BASIS_NO_MIX`` rather than silently assuming a cache hit rate.
        """
        return self.billed > 0

    def is_long_context(self, *, local_tokens: int = 0) -> bool:
        """True when this request billed above the catalogs' 200k price tier.

        ``local_tokens`` lets a caller contribute its own forwarded-token count
        for requests where the provider reported no breakdown; the larger of the
        two decides, so a long request is never priced at the cheap tier merely
        because usage was missing.
        """
        return max(self.billed, _coerce_int(local_tokens)) > LONG_CONTEXT_THRESHOLD_TOKENS


@dataclass(frozen=True)
class TokenSplit:
    """Counterfactual tokens apportioned across the buckets that would bill them.

    Floats, not ints: these are *shares* of a request's removed tokens, and
    rounding each request to whole tokens would bias a long session's total.
    """

    read: float = 0.0
    write_5m: float = 0.0
    write_1h: float = 0.0
    uncached: float = 0.0

    @property
    def total(self) -> float:
        return self.read + self.write_5m + self.write_1h + self.uncached


def split_tokens(tokens: int, mix: CacheMix, region: Region | str) -> TokenSplit:
    """Apportion ``tokens`` across billing buckets for ``region``.

    ``PREFIX`` fills the read bucket first, capped at the read tokens the
    request actually reported: prefix content sits ahead of every cache
    breakpoint, so on a warm turn it IS the read. ``LIVE_ZONE`` excludes reads
    outright — the live zone is appended after the frozen prefix and has never
    been cached.

    Whatever a region cannot attribute falls to ``uncached`` (list price), which
    is the conservative direction: it is the most expensive non-write bucket, so
    an unpriceable request overstates rather than silently zeroes.
    """
    tokens = _coerce_int(tokens)
    if tokens <= 0:
        return TokenSplit()

    mix = mix.normalized()
    # Normalize before comparing. `Region` is a str enum, so a caller that
    # passes the bare string "prefix" compares EQUAL to Region.PREFIX but is not
    # IDENTICAL to it — and an `is` test would silently route it down the
    # live-zone branch, pricing a cached prefix as if it had never been cached.
    # Callers that cannot import Region at module scope (see `proxy/cost.py`,
    # which must keep litellm off the startup path) pass the string.
    region = Region(region)

    read_part = float(min(tokens, mix.read)) if region is Region.PREFIX else 0.0
    rest = tokens - read_part
    if rest <= 0:
        return TokenSplit(read=read_part)

    denom = mix.write_5m + mix.write_1h + mix.uncached
    if denom <= 0:
        # Nothing left to apportion against. A warm PREFIX turn legitimately
        # lands here once its reads absorb every token; otherwise this is the
        # no-signal case and list price is the honest fallback.
        return TokenSplit(read=read_part, uncached=rest)

    w5 = rest * mix.write_5m / denom
    w1h = rest * mix.write_1h / denom
    # Subtract rather than compute independently so the parts sum to `tokens`
    # exactly, with no float drift accumulating over a long session.
    return TokenSplit(read=read_part, write_5m=w5, write_1h=w1h, uncached=rest - w5 - w1h)


@dataclass(frozen=True)
class CacheRates:
    """Per-token prices for each input bucket, plus where they came from."""

    read: float
    write_5m: float
    write_1h: float
    uncached: float
    basis: str = BASIS_CATALOG

    def price(self, split: TokenSplit) -> float:
        return (
            split.read * self.read
            + split.write_5m * self.write_5m
            + split.write_1h * self.write_1h
            + split.uncached * self.uncached
        )


def _litellm() -> Any | None:
    """Import LiteLLM lazily; absent is a supported configuration, not an error.

    Headroom's own dependency spec excludes litellm on Python 3.14, and gateway
    deployments often run without it. Callers degrade to list pricing.
    """
    try:
        import litellm
    except Exception:  # pragma: no cover - environment-dependent
        return None
    return litellm


#: ``resolve_rates`` runs on every priced request and ``model`` comes straight
#: off a client-supplied request body, so the memo is a BOUNDED LRU, never a
#: plain dict: a request-facing proxy must not let a caller grow a cache for
#: free by sending a fresh model string each time. LRU eviction means a model
#: that stops being sent falls out and simply re-resolves if it returns — a cost
#: question, never a correctness one. Mirrors the cap
#: ``savings_tracker._resolve_litellm_model`` already applies for the same
#: reason (and for the same noisy-probe symptom: LiteLLM prints its "Provider
#: List" banner on every failed lookup).
_RATE_CACHE_MAXSIZE = 256


@lru_cache(maxsize=_RATE_CACHE_MAXSIZE)
def resolve_rates(
    model: str,
    *,
    long_context: bool = False,
    provider: str | None = None,
) -> CacheRates | None:
    """Resolve per-bucket input rates for ``model``, or ``None`` if unpriceable.

    Preference order, strongest first:

    1. **Catalog** — LiteLLM's per-model published rates, including the
       long-context tier and, where the row has it,
       ``cache_creation_input_token_cost_above_1hr``. Anthropic publishes a real
       1h write rate (Sonnet: 2.00x base), and reading it beats deriving it.
    2. **Catalog + TTL ratio** — a catalog row that prices 5m writes but not 1h
       ones, which is every row at the above-200k tier. The 1h rate is derived
       from the tier's base input price and the structural multiplier in
       :mod:`headroom.pricing.cache_ttl`. Only applied when the row shows a real
       write premium: a provider that does not bill for cache writes at all
       (OpenAI, Gemini) has no 5m/1h trade to derive, and inventing one would
       charge them a premium their invoice never shows.
    3. **Provider ratio** — :data:`CACHE_ECONOMICS`, for a model the catalog
       prices for input but not for cache.
    4. ``None`` — model unknown to the catalog. The caller falls back to list.

    A model whose input price is legitimately ``0.0`` (a free or local model)
    returns all-zero rates rather than ``None``: free must cost nothing, not
    fall through to a blended estimate.

    Memoized — see :data:`_RATE_CACHE_MAXSIZE`. Tests that swap the LiteLLM
    catalog between cases must call ``resolve_rates.cache_clear()`` in between,
    or the previous case's rates leak into the next.
    """
    litellm = _litellm()
    if litellm is None:
        return None

    try:
        from headroom.pricing.litellm_pricing import resolve_litellm_model

        info = litellm.model_cost.get(resolve_litellm_model(model), {}) or {}
    except Exception:
        return None

    base = info.get("input_cost_per_token")
    # `is None` distinguishes "unknown model" from "genuinely free". `not base`
    # treated a real 0.0 as unavailable and billed a fallback rate — phantom
    # savings on a model that costs nothing.
    if base is None:
        return None
    if long_context:
        base = info.get("input_cost_per_token_above_200k_tokens") or base
    base = float(base)

    def _tier(field: str, default: float) -> float:
        """Read ``field``, preferring its above-200k variant on long requests."""
        if long_context:
            hi = info.get(f"{field}_above_200k_tokens")
            if hi:
                return float(hi)
        value = info.get(field)
        return float(value) if value is not None else default

    read = _tier("cache_read_input_token_cost", base)
    # A provider that does not bill cache writes leaves this absent; writes then
    # cost the same as ordinary input, which is exactly OpenAI's and Gemini's
    # actual behaviour.
    write_5m = _tier("cache_creation_input_token_cost", base)

    basis = BASIS_CATALOG
    write_1h_raw = info.get("cache_creation_input_token_cost_above_1hr")
    if long_context:
        # No catalog publishes a combined 1h + above-200k rate, so the long
        # tier always derives. Ratio basis, and labelled as such.
        write_1h_raw = None
    if write_1h_raw:
        write_1h = float(write_1h_raw)
    elif write_5m > base:
        # Real write premium present but no 1h rate: derive structurally.
        write_1h = base * CACHE_WRITE_MULTIPLIERS["1h"]
        basis = BASIS_CATALOG_TTL_RATIO
    else:
        # No write premium at all (OpenAI, Gemini). There is no 1h trade to
        # price; a 1h write costs what any write costs.
        write_1h = write_5m

    if read == base and write_5m == base and base > 0:
        # Catalog priced the model but published no cache rates. Fall back to
        # the provider ratio table if we can identify the provider.
        econ = CACHE_ECONOMICS.get((provider or "").split(":")[-1].strip().lower())
        if econ:
            return CacheRates(
                read=base * float(econ["read_multiplier"]),
                write_5m=base * float(econ["write_multiplier"]),
                write_1h=base * float(econ["write_multiplier"]),
                uncached=base,
                basis=BASIS_PROVIDER_RATIO,
            )

    return CacheRates(read=read, write_5m=write_5m, write_1h=write_1h, uncached=base, basis=basis)


@dataclass(frozen=True)
class PricedSavings:
    """What a set of counterfactual tokens was worth, and how sure we are.

    ``usd`` is the headline: what the provider would actually have charged for
    these tokens given the cache mix of the request they were removed from.
    ``usd_list`` is the same tokens at flat list price — the upper bound, what
    Headroom reported before this module existed, and the figure budget
    enforcement keeps using because it is monotonic in tokens and independent of
    provider reporting.

    ``usd <= usd_list`` always holds for PREFIX savings. It does NOT hold for
    LIVE_ZONE savings on a cold turn: a token written into an Anthropic cache
    costs 1.25x list, so removing it saves more than list. That is real money,
    not an accounting artifact, and clamping it would under-report Headroom.
    """

    usd: float
    usd_list: float
    basis: str
    tokens: int = 0
    split: TokenSplit | None = None

    @property
    def ratio(self) -> float:
        """``usd / usd_list`` — how far the honest figure sits below list.

        1.0 means list pricing happened to be right (a fully cold request).
        Returns 1.0 when there is nothing to compare.
        """
        return self.usd / self.usd_list if self.usd_list else 1.0


def price_savings(
    tokens: int,
    *,
    model: str,
    mix: CacheMix | None = None,
    region: Region = Region.LIVE_ZONE,
    long_context: bool | None = None,
    local_tokens: int = 0,
    provider: str | None = None,
    fallback_rate_per_token: float | None = None,
) -> PricedSavings:
    """Price ``tokens`` that Headroom kept out of one request.

    The single entry point every savings surface should call. ``region`` decides
    the apportionment (see :func:`split_tokens`), ``mix`` supplies the observed
    cache breakdown, and ``fallback_rate_per_token`` is the blended rate used
    when the model cannot be priced at all — the MCP tool path, which never
    learns the agent's upstream model, depends on it.

    Never raises: a savings figure must not be able to fail a request.
    """
    tokens = _coerce_int(tokens)
    if tokens <= 0:
        return PricedSavings(usd=0.0, usd_list=0.0, basis=BASIS_CATALOG, tokens=0)

    mix = (mix or CacheMix()).normalized()
    if long_context is None:
        long_context = mix.is_long_context(local_tokens=local_tokens)

    try:
        rates = resolve_rates(model, long_context=long_context, provider=provider)
    except Exception:  # pragma: no cover - defensive; pricing must never raise
        logger.debug("counterfactual: rate resolution failed for %s", model, exc_info=True)
        rates = None

    if rates is None:
        if fallback_rate_per_token is None:
            return PricedSavings(usd=0.0, usd_list=0.0, basis=BASIS_UNPRICED, tokens=tokens)
        flat = float(tokens) * float(fallback_rate_per_token)
        return PricedSavings(usd=flat, usd_list=flat, basis=BASIS_LIST, tokens=tokens)

    usd_list = float(tokens) * rates.uncached

    if not mix.has_signal():
        # No breakdown to apportion against. List price IS the answer here, but
        # it is labelled so a reader knows it is a ceiling, not a measurement.
        return PricedSavings(
            usd=usd_list,
            usd_list=usd_list,
            basis=BASIS_NO_MIX,
            tokens=tokens,
            split=TokenSplit(uncached=float(tokens)),
        )

    split = split_tokens(tokens, mix, region)
    return PricedSavings(
        usd=rates.price(split),
        usd_list=usd_list,
        basis=rates.basis,
        tokens=tokens,
        split=split,
    )


__all__ = [
    "BASIS_CATALOG",
    "BASIS_CATALOG_TTL_RATIO",
    "BASIS_LIST",
    "BASIS_NO_MIX",
    "BASIS_PROVIDER_RATIO",
    "BASIS_UNPRICED",
    "CACHE_ECONOMICS",
    "LONG_CONTEXT_THRESHOLD_TOKENS",
    "CacheMix",
    "CacheRates",
    "PricedSavings",
    "Region",
    "TokenSplit",
    "price_savings",
    "resolve_rates",
    "split_tokens",
    "weakest_basis",
]
