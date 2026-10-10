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
import math
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any

from headroom.cache_economics import CACHE_ECONOMICS
from headroom.pricing.cache_ttl import CACHE_WRITE_MULTIPLIERS

logger = logging.getLogger(__name__)

#: Prompt size at which most catalog rows publish a second, higher price tier
#: (LiteLLM spells it ``*_above_200k_tokens``). The default for
#: :func:`long_context_threshold`, which reads each model's own threshold from
#: its row: Claude Haiku 5.5's tier starts at 100K, GPT-5.x's at 272K.
LONG_CONTEXT_THRESHOLD_TOKENS = 200_000

#: A catalog row's long-context input rate; the number is the threshold in
#: thousands of tokens. Each row publishes at most one.
_LONG_CONTEXT_INPUT_FIELD = re.compile(r"^input_cost_per_token_above_(\d+)k_tokens$")

#: Provider-level cache discount ratios, as a fraction of the base input price.
#: FALLBACK ONLY — the per-model LiteLLM catalog is always preferred, because
#: these go stale per model and per context tier. The lightweight shared table
#: is also used by the dashboard, without importing the pricing package there.


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

    def is_long_context(
        self, *, local_tokens: int = 0, threshold: int = LONG_CONTEXT_THRESHOLD_TOKENS
    ) -> bool:
        """True when this request billed above ``threshold``, the model's long-context tier.

        ``local_tokens`` lets a caller contribute its own forwarded-token count
        for requests where the provider reported no breakdown; the larger of the
        two decides, so a long request is never priced at the cheap tier merely
        because usage was missing. Pass :func:`long_context_threshold` for the
        model so a tier that starts below 200K (Haiku 5.5: 100K) is honoured.
        """
        return max(self.billed, _coerce_int(local_tokens)) > threshold


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
    read_is_catalog: bool = True
    write_is_catalog: bool = True

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


def _catalog_row(model: str) -> dict[str, Any] | None:
    """``model``'s LiteLLM catalog row, ``{}`` when unlisted, ``None`` without LiteLLM."""
    litellm = _litellm()
    if litellm is None:
        return None
    try:
        from headroom.pricing.litellm_pricing import resolve_litellm_model

        return litellm.model_cost.get(resolve_litellm_model(model), {}) or {}
    except Exception:
        return None


def long_context_tier(info: dict[str, Any]) -> tuple[int, str] | None:
    """The row's long-context tier as ``(threshold_tokens, field_suffix)``, if any."""
    for key in info:
        match = _LONG_CONTEXT_INPUT_FIELD.match(key)
        if match:
            thousands = int(match.group(1))
            return thousands * 1000, f"_above_{thousands}k_tokens"
    return None


@lru_cache(maxsize=_RATE_CACHE_MAXSIZE)
def long_context_threshold(model: str) -> int:
    """Billed-prompt size above which ``model`` pays its long-context rates.

    Read from the catalog row's ``*_above_<N>k_tokens`` fields: 100K for Claude
    Haiku 5.5, 200K for Sonnet 4 / 4.5, 272K for GPT-5.x. A model with no tier,
    or one the catalog does not list, gets :data:`LONG_CONTEXT_THRESHOLD_TOKENS`;
    that is harmless, because :func:`resolve_rates` then has no tier to switch
    to and returns the base rates either way.

    Memoized on the same bounded LRU as :func:`resolve_rates`, for the same
    reason; tests that swap the catalog clear both.
    """
    tier = long_context_tier(_catalog_row(model) or {})
    return tier[0] if tier else LONG_CONTEXT_THRESHOLD_TOKENS


def _canonical_cache_rate(litellm: Any, model: str, field: str, *, long_context: bool) -> float:
    """Resolve an absent catalog slice through the model's billing calculator.

    LiteLLM versions and model adapters differ: missing rates can contribute
    zero, list price, or a model-specific discount. Never invent our own ratio.
    The result is cached with the other resolved rates, not probed per request.
    """
    tokens = long_context_threshold(model) + 1 if long_context else 1
    try:
        input_cost, _ = litellm.cost_per_token(
            model=model, prompt_tokens=tokens, completion_tokens=0, **{field: tokens}
        )
        rate = float(input_cost) / tokens
    except Exception:
        return 0.0
    return rate if math.isfinite(rate) and rate >= 0 else 0.0


@lru_cache(maxsize=_RATE_CACHE_MAXSIZE)
def resolve_rates(
    model: str,
    *,
    long_context: bool = False,
    provider: str | None = None,
    for_billing: bool = False,
) -> CacheRates | None:
    """Resolve per-bucket input rates for ``model``, or ``None`` if unpriceable.

    ``for_billing`` resolves absent cache rates through LiteLLM's model-specific
    billing calculation. Savings estimates use the default fallbacks below
    instead; those estimates must not manufacture billed cache charges.

    Preference order, strongest first:

    1. **Catalog** — LiteLLM's per-model published rates, including the
       long-context tier (the row's own ``*_above_<N>k_tokens`` fields, see
       :func:`long_context_threshold`) and, where the row has it,
       ``cache_creation_input_token_cost_above_1hr``. Anthropic publishes a real
       1h write rate (Sonnet: 2.00x base), and reading it beats deriving it.
    2. **Catalog + TTL ratio** — a catalog row that prices 5m writes but not 1h
       ones, which is every row at its long-context tier. The 1h rate is derived
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

        resolved_model = resolve_litellm_model(model)
        info = litellm.model_cost.get(resolved_model, {}) or {}
    except Exception:
        return None

    base = info.get("input_cost_per_token")
    # `is None` distinguishes "unknown model" from "genuinely free". `not base`
    # treated a real 0.0 as unavailable and billed a fallback rate — phantom
    # savings on a model that costs nothing.
    if base is None:
        return None
    tier = long_context_tier(info) if long_context else None
    if tier:
        long_base = info.get(f"input_cost_per_token{tier[1]}")
        if long_base is not None:
            base = long_base
    base = float(base)

    def _tier(field: str, default: float) -> float:
        """Read ``field``, preferring its long-context variant on long requests."""
        if tier:
            hi = info.get(f"{field}{tier[1]}")
            if hi is not None:
                return float(hi)
        value = info.get(field)
        return float(value) if value is not None else default

    # Counterfactual savings retain their existing, labelled estimate fallback.
    # Billed slices consult the canonical model calculator below when the
    # catalog omits a rate; missing metadata does not imply free cache.
    missing_cache_rate = 0.0 if for_billing else base
    read = _tier("cache_read_input_token_cost", missing_cache_rate)
    # Savings estimates value otherwise unpriced writes at ordinary input;
    # billed provider-reported writes require their own catalog rate.
    write_5m = _tier("cache_creation_input_token_cost", missing_cache_rate)
    read_is_catalog = info.get("cache_read_input_token_cost") is not None or (
        tier is not None and info.get(f"cache_read_input_token_cost{tier[1]}") is not None
    )
    write_is_catalog = info.get("cache_creation_input_token_cost") is not None or (
        tier is not None and info.get(f"cache_creation_input_token_cost{tier[1]}") is not None
    )
    if for_billing and not read_is_catalog:
        read = _canonical_cache_rate(
            litellm, resolved_model, "cache_read_input_tokens", long_context=long_context
        )
    if for_billing and not write_is_catalog:
        write_5m = _canonical_cache_rate(
            litellm, resolved_model, "cache_creation_input_tokens", long_context=long_context
        )

    basis = BASIS_CATALOG
    write_1h_raw = info.get("cache_creation_input_token_cost_above_1hr")
    if tier:
        # The long tier's 1h rate is its own field
        # (``cache_creation_input_token_cost_above_1hr_above_100k_tokens``);
        # most tiered rows omit it, and the rate then derives below. Ratio
        # basis, and labelled as such.
        write_1h_raw = info.get(f"cache_creation_input_token_cost_above_1hr{tier[1]}")
    if write_1h_raw is not None:
        write_1h = float(write_1h_raw)
    elif write_5m > base:
        # Real write premium present but no 1h rate: derive structurally.
        write_1h = base * CACHE_WRITE_MULTIPLIERS["1h"]
        basis = BASIS_CATALOG_TTL_RATIO
    else:
        # No write premium at all (OpenAI, Gemini). There is no 1h trade to
        # price; a 1h write costs what any write costs.
        write_1h = write_5m

    if (
        not for_billing
        and not read_is_catalog
        and not write_is_catalog
        and read == base
        and write_5m == base
        and base > 0
    ):
        # Catalog priced the model but published no cache rates. Fall back to
        # the provider ratio table if we can identify the provider.
        # Model-only cost consumers do not have a request provider to pass.
        # Prefer explicit routing context, otherwise use the catalog identity.
        resolved_provider = provider or info.get("litellm_provider") or ""
        provider_key = (
            resolved_provider.split(":")[-1].strip().lower()
            if isinstance(resolved_provider, str)
            else ""
        )
        econ = CACHE_ECONOMICS.get(provider_key)
        if econ:
            return CacheRates(
                read=base * float(econ["read_multiplier"]),
                write_5m=base * float(econ["write_multiplier"]),
                write_1h=base * float(econ["write_multiplier"]),
                uncached=base,
                basis=BASIS_PROVIDER_RATIO,
                read_is_catalog=read_is_catalog,
                write_is_catalog=write_is_catalog,
            )

    return CacheRates(
        read=read,
        write_5m=write_5m,
        write_1h=write_1h,
        uncached=base,
        basis=basis,
        read_is_catalog=read_is_catalog,
        write_is_catalog=write_is_catalog,
    )


def long_context_premium_avoided_usd(
    model: str,
    mix: CacheMix,
    *,
    local_tokens: int = 0,
    output_tokens: int = 0,
    provider: str | None = None,
) -> float:
    """Long-context premium a request's forwarded input and output avoided, in USD.

    For a request whose uncompressed prompt would have passed ``model``'s
    long-context threshold but whose forwarded prompt did not: the provider
    re-prices the WHOLE request above the threshold, so compression saved the
    long card on every forwarded input token, on top of the removed tokens
    themselves. This is that difference, forwarded input at the long card minus
    the same input at the base card. When ``output_tokens`` is supplied, include
    the premium avoided on that generated output too. These tokens were still
    generated; this is compression savings, not output-shaping savings.

    ``local_tokens`` stands in as uncached input when the provider reported no
    breakdown. Returns 0.0 for a model with no tier or no catalog row.
    """
    try:
        long_rates = resolve_rates(model, long_context=True, provider=provider)
        base_rates = resolve_rates(model, long_context=False, provider=provider)
    except Exception:  # pragma: no cover - defensive; pricing must never raise
        return 0.0
    if long_rates is None or base_rates is None:
        return 0.0
    if mix.has_signal():
        n = mix.normalized()
        split = TokenSplit(
            read=float(n.read),
            write_5m=float(n.write_5m),
            write_1h=float(n.write_1h),
            uncached=float(n.uncached),
        )
    else:
        split = TokenSplit(uncached=float(max(_coerce_int(local_tokens), 0)))
    input_premium = max(0.0, long_rates.price(split) - base_rates.price(split))
    output_premium = 0.0
    if max(_coerce_int(output_tokens), 0):
        try:
            info = _catalog_row(model) or {}
            tier = long_context_tier(info)
            base_output = info.get("output_cost_per_token")
            long_output = info.get(f"output_cost_per_token{tier[1]}") if tier else None
            if base_output is not None and long_output is not None:
                output_premium = max(0.0, float(long_output) - float(base_output)) * max(
                    _coerce_int(output_tokens), 0
                )
        except Exception:  # Pricing metadata must not break telemetry.
            pass
    return input_premium + output_premium


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
        long_context = mix.is_long_context(
            local_tokens=local_tokens, threshold=long_context_threshold(model)
        )

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
    "long_context_premium_avoided_usd",
    "long_context_threshold",
    "long_context_tier",
    "price_savings",
    "resolve_rates",
    "split_tokens",
    "weakest_basis",
]
