"""DeepSeek peak/off-peak pricing structure.

DeepSeek prices DeepSeek-V4.1-Flash (``deepseek-flash``) and DeepSeek-V4-Pro
(``deepseek-v4-pro``) on Beijing-time peak windows, and peak is exactly twice
off-peak. Because the ratio is structural, it is applied here instead of being
transcribed into every row of :mod:`headroom.pricing.deepseek_prices` - a
per-tier table would double the columns, would drift, and would let one tier be
mis-typed without anything noticing.

Windows are Beijing wall time (a fixed UTC+8; the PRC has no daylight saving):
09:00-12:00 and 14:00-18:00, which are the published 01:00-04:00 and
06:00-10:00 UTC windows. Weekends are all-day off-peak only from
:data:`WEEKEND_OFF_PEAK_FROM`; before that instant they still followed the
weekday windows, so a back-dated or replayed request is not silently repriced.

Prices are the vendor's own USD list. The zh-cn page lists the same tiers in
CNY, but the two lists are not one FX apart, so CNY billing needs a switch over
two published lists rather than a baked rate - not modelled here.

Consumers that need one flat number for a model take :func:`off_peak_rates`
(the cheaper published tier). Consumers pricing a real request take
:func:`rates_for` with that request's instant.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal

#: Peak is exactly this multiple of every off-peak rate.
PEAK_MULTIPLIER: float = 2.0

#: Last date the vendor's USD rate card below was verified.
LAST_UPDATED = date(2026, 9, 14)

#: Official pricing page these rates come from.
SOURCE_URL = "https://api-docs.deepseek.com/quick_start/pricing"

#: The vendor's published card, as fetched, so the numbers below can be checked
#: without trusting prose. Values are quoted from the page, not re-derived.
#:
#: Source: :data:`SOURCE_URL`, fetched 2026-09-14 and re-verified 2026-09-15 in
#: both locales with a cache-busting query. Footnote (1) verbatim: "Use
#: ``deepseek-flash`` as the model name. The legacy names ``deepseek-v4-flash``
#: and ``deepseek-v4-flash-vision-exp`` are still accepted, but the corresponding
#: models have been retired, their requests are served by the DeepSeek-V4.1-Flash
#: model and billed at the Flash price."
VENDOR_CARD: dict[str, dict[str, object]] = {
    "deepseek-flash": {
        "model_version": "DeepSeek-V4.1-Flash",
        "off_peak": (0.003, 0.15, 0.60),
        "peak": (0.006, 0.30, 1.20),
        "max_output": "384K; upstream LiteLLM encodes it as 393216 = 384 x 1024",
        "vision": True,
        "legacy_ids": ("deepseek-v4-flash", "deepseek-v4-flash-vision-exp"),
    },
    "deepseek-v4-pro": {
        "model_version": "DeepSeek-V4-Pro-0813",
        "off_peak": (0.022, 0.66, 1.98),
        "peak": (0.044, 1.32, 3.96),
        "max_output": "384K",
        "vision": False,
        "legacy_ids": (),
    },
}

#: Off-peak ``(cache-hit input, cache-miss input, output)`` USD per 1M tokens.
#: These are the ``off_peak`` tuples of :data:`VENDOR_CARD`; the assertions in
#: ``tests/test_pricing_deepseek_tiers.py`` keep the two in step.
#:
#: ``deepseek-flash`` is the current id for DeepSeek-V4.1-Flash. The vendor still
#: accepts the retired ``deepseek-v4-flash`` and ``deepseek-v4-flash-vision-exp``
#: ids and serves them from the same model at the Flash price, so they resolve
#: through :data:`LEGACY_MODEL_IDS`.
OFF_PEAK_RATES_PER_1M: dict[str, tuple[float, float, float]] = {
    "deepseek-flash": (0.003, 0.15, 0.60),
    "deepseek-v4-pro": (0.022, 0.66, 1.98),
}

#: Retired ids the vendor still accepts, mapped to the id that now serves them.
LEGACY_MODEL_IDS: dict[str, str] = {
    "deepseek-v4-flash": "deepseek-flash",
    "deepseek-v4-flash-vision-exp": "deepseek-flash",
}

#: Peak windows in Beijing wall time, start-inclusive and end-exclusive.
PEAK_WINDOWS_BEIJING: tuple[tuple[time, time], ...] = (
    (time(9), time(12)),
    (time(14), time(18)),
)

#: Beijing is a fixed UTC+8 offset.
BEIJING_TZ = timezone(timedelta(hours=8))

#: Instant weekends became all-day off-peak: 2026-08-23 00:00 Beijing.
WEEKEND_OFF_PEAK_FROM = datetime(2026, 8, 22, 16, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class DeepSeekRates:
    """One tier of a DeepSeek model's rate card, in USD per 1M tokens."""

    cache_hit_per_1m: float
    input_per_1m: float
    output_per_1m: float
    #: DeepSeek bills no cache-write surcharge; write tokens bill as cache-miss input.
    cache_write_per_1m: float
    tier: Literal["peak", "off_peak"]


def bare_model(model: str) -> str:
    """Return ``model`` without a ``provider/`` prefix, lowercased.

    A tag suffix after ``:`` (``deepseek/deepseek-v4-pro:free``) is not stripped: such an id
    falls out of tier scope, and it is priced only if a flat table happens to carry that exact
    key - otherwise the caller's cost is unknown rather than approximated.

    Args:
        model: A model id, optionally prefixed the way a gateway writes it
            (``deepseek/deepseek-v4-pro``).

    Returns:
        The bare, lowercased id used to key the tier table.
    """
    return model.rsplit("/", 1)[-1].strip().lower()


def _canonical(model: str) -> str:
    """Return the id whose tier prices ``model``, resolving retired aliases."""
    bare = bare_model(model)
    return LEGACY_MODEL_IDS.get(bare, bare)


def _tier_rates(canonical: str, tier: Literal["peak", "off_peak"]) -> DeepSeekRates | None:
    """Build the ``tier`` rate row for a canonical id, or ``None`` if unknown."""
    rates = OFF_PEAK_RATES_PER_1M.get(canonical)
    if rates is None:
        return None
    hit, miss, out = rates
    if tier == "peak":
        hit = hit * PEAK_MULTIPLIER
        miss = miss * PEAK_MULTIPLIER
        out = out * PEAK_MULTIPLIER
    return DeepSeekRates(
        cache_hit_per_1m=hit,
        input_per_1m=miss,
        output_per_1m=out,
        cache_write_per_1m=0.0,
        tier=tier,
    )


def off_peak_rates(model: str) -> DeepSeekRates | None:
    """Return the off-peak tier for ``model``, or ``None`` when out of scope.

    Args:
        model: Model id, with or without a ``provider/`` prefix, and either a
            current id or a retired alias.

    Returns:
        The off-peak :class:`DeepSeekRates`, or ``None`` for any model outside
        the flash/pro rate card.
    """
    return _tier_rates(_canonical(model), "off_peak")


def is_peak(now: datetime) -> bool:
    """Return whether ``now`` falls in a Beijing peak window.

    Args:
        now: The instant to test. A naive value is read as UTC.

    Returns:
        ``True`` during Beijing 09:00-12:00 or 14:00-18:00 on a day the published
        windows apply; weekends are all-day off-peak once
        :data:`WEEKEND_OFF_PEAK_FROM` is in force.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    beijing = now.astimezone(BEIJING_TZ)
    if now >= WEEKEND_OFF_PEAK_FROM and beijing.weekday() >= 5:  # 5 = Saturday
        return False
    wall = beijing.time()
    return any(start <= wall < end for start, end in PEAK_WINDOWS_BEIJING)


def rates_for(model: str, now: datetime | None = None) -> DeepSeekRates | None:
    """Return the tier that prices ``model`` at ``now``.

    This is the seam every cost path uses. A caller with a request instant passes
    it, so tests and replays stay deterministic; a caller without one gets the
    wall clock.

    Args:
        model: Model id, with or without a ``provider/`` prefix, current or retired.
        now: The request instant, or ``None`` to read the clock.

    Returns:
        The applicable :class:`DeepSeekRates`, or ``None`` for any model outside
        the flash/pro rate card - the caller's cue to use its generic path.
    """
    canonical = _canonical(model)
    if canonical not in OFF_PEAK_RATES_PER_1M:
        return None
    instant = now if now is not None else datetime.now(timezone.utc)
    return _tier_rates(canonical, "peak" if is_peak(instant) else "off_peak")
