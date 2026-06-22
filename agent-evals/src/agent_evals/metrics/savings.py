"""Layer-1 savings capture: per-task header parsing + run-level ``/stats`` reader.

Two attribution surfaces, matching the Headroom proxy's two reporting channels:

* **Per-request** — every optimized response carries ``x-headroom-*`` headers
  (``-tokens-before``/``-after``/``-saved``/``-model`` plus conditional ``-transforms``,
  ``-cached``, ``-compression-failed``). :func:`parse_savings_headers` turns those into a
  :class:`~agent_evals.models.TaskSavings`, and :func:`make_response_hook` attaches that
  parse to an ``httpx`` client so the savings of every request a task issues land in a
  :class:`SavingsStore` keyed by ``task_id``.
* **Run-level** — the ``/stats`` endpoint exposes lifetime cache / prefix-freeze aggregates
  that have no per-response header and so cannot be attributed to one task.
  :func:`fetch_run_savings` reads those into a :class:`~agent_evals.models.RunSavings`.

The header names live as module-level constants so they are configured in exactly one place;
their VALUES are pinned to what the proxy emits (verified against
``headroom/proxy/handlers/{openai,anthropic}.py``). ``x-headroom-savings-percent`` is
deliberately NOT read here — it only exists on the batch path, so percent/ratio are DERIVED by
:meth:`TaskSavings.from_token_counts`.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from agent_evals.logging import get_logger
from agent_evals.models import Pricing, RunSavings, TaskSavings

logger = get_logger("metrics.savings")

# --- Per-response header names (verified against headroom proxy handlers) -------------------
# Required token headers — absent => no Headroom optimization happened on this response.
HEADER_TOKENS_BEFORE = "x-headroom-tokens-before"
HEADER_TOKENS_AFTER = "x-headroom-tokens-after"
# Emitted alongside the required headers, but DERIVED here from before/after so we never trust
# a value we can also compute; kept as a constant for documentation/lookup parity.
HEADER_TOKENS_SAVED = "x-headroom-tokens-saved"
HEADER_MODEL = "x-headroom-model"
# Conditional headers — only present when their condition holds.
HEADER_TRANSFORMS = "x-headroom-transforms"
HEADER_CACHED = "x-headroom-cached"
HEADER_COMPRESSION_FAILED = "x-headroom-compression-failed"

# String the proxy writes for a true boolean header (e.g. ``x-headroom-cached: "true"``).
_HEADER_TRUE = "true"
# Delimiter the proxy uses to join the transforms list into a single header value.
_TRANSFORMS_SEP = ","

# --- /stats payload paths (verified against proxy/server.py + proxy/cost.py) ----------------
# Run-level cache reads live under prefix_cache.totals.cache_read_tokens.
_STATS_PREFIX_CACHE_KEY = "prefix_cache"
_STATS_TOTALS_KEY = "totals"
_STATS_CACHE_READ_TOKENS_KEY = "cache_read_tokens"
# Prefix-freeze block lives under prefix_cache.prefix_freeze.
_STATS_PREFIX_FREEZE_KEY = "prefix_freeze"
_STATS_BUSTS_AVOIDED_KEY = "busts_avoided"
_STATS_TOKENS_PRESERVED_KEY = "tokens_preserved"


def _lower_keyed(headers: Mapping[str, str]) -> dict[str, str]:
    """Return a lower-cased copy of ``headers`` for case-insensitive lookup.

    ``httpx.Headers`` is already case-insensitive, but a plain ``dict`` (or any ``Mapping``)
    is not — so we normalize once here and look everything up in lower case.
    """

    return {str(k).lower(): v for k, v in headers.items()}


def _parse_bool_header(value: str | None) -> bool:
    """A conditional bool header is true iff present and equal to the proxy's true sentinel."""

    return value is not None and value.strip().lower() == _HEADER_TRUE


def _parse_transforms(value: str | None) -> list[str]:
    """Split the transforms header into an ordered, de-duplicated, non-empty list."""

    if not value:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in value.split(_TRANSFORMS_SEP):
        item = part.strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def parse_savings_headers(
    headers: Mapping[str, str],
    pricing: Pricing,
    added_latency_ms: float = 0.0,
) -> TaskSavings | None:
    """Parse the per-response ``x-headroom-*`` headers into a :class:`TaskSavings`.

    Returns ``None`` (never a fabricated zero) when the required token headers are absent —
    that signals "no Headroom optimization on this response" (e.g. a pass-through / A1 arm, or
    a non-optimized request). ``savings_percent``/``ratio``/``cost_*`` are DERIVED by
    :meth:`TaskSavings.from_token_counts`; ``x-headroom-savings-percent`` is never read (it is
    batch-path-only).
    """

    lut = _lower_keyed(headers)
    raw_before = lut.get(HEADER_TOKENS_BEFORE)
    raw_after = lut.get(HEADER_TOKENS_AFTER)
    if raw_before is None or raw_after is None:
        return None

    try:
        tokens_before = int(raw_before)
        tokens_after = int(raw_after)
    except (TypeError, ValueError):
        # The required headers are present but malformed. Fail loud, do not fabricate.
        logger.warning(
            "malformed headroom token headers; skipping savings capture",
            extra={
                "fields": {
                    HEADER_TOKENS_BEFORE: raw_before,
                    HEADER_TOKENS_AFTER: raw_after,
                }
            },
        )
        return None

    return TaskSavings.from_token_counts(
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        pricing=pricing,
        transforms=_parse_transforms(lut.get(HEADER_TRANSFORMS)),
        cached=_parse_bool_header(lut.get(HEADER_CACHED)),
        compression_failed=_parse_bool_header(lut.get(HEADER_COMPRESSION_FAILED)),
        added_latency_ms=added_latency_ms,
        source="headers",
    )


class SavingsStore:
    """Thread-safe ``task_id -> list[TaskSavings]`` accumulator.

    A single benchmark task issues many model requests; each optimized response contributes one
    :class:`TaskSavings`. :meth:`aggregate` collapses a task's requests into one
    :class:`TaskSavings` by SUMMING the raw token counts and re-deriving percent/ratio/cost via
    :meth:`TaskSavings.from_token_counts` (deriving from the summed counts, never averaging the
    per-request percentages). The lock makes concurrent ``add`` (from httpx event hooks running
    across tasks) and ``aggregate`` safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_task: dict[str, list[TaskSavings]] = {}

    def add(self, task_id: str, savings: TaskSavings) -> None:
        """Record one request's savings under ``task_id``."""

        with self._lock:
            self._by_task.setdefault(task_id, []).append(savings)

    def get(self, task_id: str) -> list[TaskSavings]:
        """Return a snapshot copy of the per-request savings recorded for ``task_id``."""

        with self._lock:
            return list(self._by_task.get(task_id, ()))

    def task_ids(self) -> list[str]:
        """Return the task ids that have at least one recorded request."""

        with self._lock:
            return list(self._by_task.keys())

    def aggregate(self, task_id: str, pricing: Pricing) -> TaskSavings | None:
        """Collapse ``task_id``'s requests into one :class:`TaskSavings`, or ``None`` if none.

        Sums ``tokens_before``/``tokens_after`` and ``added_latency_ms``; unions ``transforms``
        preserving first-seen order; ``cached``/``compression_failed`` are ORed. Percent, ratio
        and cost are re-derived from the summed token counts.
        """

        items = self.get(task_id)
        if not items:
            return None

        total_before = sum(s.tokens_before for s in items)
        total_after = sum(s.tokens_after for s in items)
        total_latency = sum(s.added_latency_ms for s in items)
        cached_any = any(s.cached for s in items)
        failed_any = any(s.compression_failed for s in items)

        merged_transforms: OrderedDict[str, None] = OrderedDict()
        for s in items:
            for t in s.transforms:
                merged_transforms.setdefault(t, None)

        return TaskSavings.from_token_counts(
            tokens_before=total_before,
            tokens_after=total_after,
            pricing=pricing,
            transforms=list(merged_transforms.keys()),
            cached=cached_any,
            compression_failed=failed_any,
            added_latency_ms=total_latency,
            source="headers",
        )


def make_response_hook(
    store: SavingsStore,
    task_id_getter: Callable[[], str | None],
    pricing: Pricing,
) -> Callable[[httpx.Response], None]:
    """Build an ``httpx`` response event-hook that attributes savings to the current task.

    Attach via ``httpx.Client(event_hooks={"response": [hook]})`` on the harness client shim.
    On each response it reads the ``x-headroom-*`` headers; if ``task_id_getter()`` resolves to
    a task id and the headers parse to a :class:`TaskSavings`, the entry is added to ``store``.
    When no task is active (``task_id_getter()`` returns ``None``) or the response carries no
    Headroom headers, nothing is recorded — savings are never fabricated.
    """

    def hook(response: httpx.Response) -> None:
        task_id = task_id_getter()
        if task_id is None:
            return
        savings = parse_savings_headers(response.headers, pricing)
        if savings is None:
            return
        store.add(task_id, savings)
        logger.debug(
            "captured task savings",
            extra={
                "fields": {
                    "task_id": task_id,
                    "tokens_before": savings.tokens_before,
                    "tokens_after": savings.tokens_after,
                    "tokens_saved": savings.tokens_saved,
                }
            },
        )

    return hook


def _coerce_int(value: Any) -> int | None:
    """Coerce a JSON leaf to ``int``, or ``None`` if it is missing / not numeric."""

    if isinstance(value, bool):  # bool is an int subclass — reject explicitly.
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def fetch_run_savings(stats_url: str, client: httpx.Client) -> RunSavings:
    """GET ``stats_url`` and map the cache / prefix-freeze aggregates into :class:`RunSavings`.

    Tolerates missing leaves: ``cache_read_tokens`` and ``busts_avoided`` default to ``0`` and
    ``prefix_freeze.tokens_preserved`` stays ``None`` when absent (it is ``Optional`` in the
    model). Reads the verified payload paths
    ``prefix_cache.totals.cache_read_tokens`` and ``prefix_cache.prefix_freeze.{busts_avoided,
    tokens_preserved}``. Raises on transport / HTTP / JSON errors — failures are loud.
    """

    response = client.get(stats_url)
    response.raise_for_status()
    payload: Any = response.json()
    if not isinstance(payload, Mapping):
        raise ValueError(f"/stats payload is not a JSON object: {type(payload).__name__}")

    prefix_cache = payload.get(_STATS_PREFIX_CACHE_KEY)
    prefix_cache = prefix_cache if isinstance(prefix_cache, Mapping) else {}

    totals = prefix_cache.get(_STATS_TOTALS_KEY)
    totals = totals if isinstance(totals, Mapping) else {}
    cache_read_tokens = _coerce_int(totals.get(_STATS_CACHE_READ_TOKENS_KEY)) or 0

    prefix_freeze = prefix_cache.get(_STATS_PREFIX_FREEZE_KEY)
    prefix_freeze = prefix_freeze if isinstance(prefix_freeze, Mapping) else {}
    busts_avoided = _coerce_int(prefix_freeze.get(_STATS_BUSTS_AVOIDED_KEY)) or 0
    tokens_preserved = _coerce_int(prefix_freeze.get(_STATS_TOKENS_PRESERVED_KEY))

    run = RunSavings(
        cache_read_tokens=cache_read_tokens,
        prefix_freeze_busts_avoided=busts_avoided,
        prefix_freeze_tokens_preserved=tokens_preserved,
    )
    logger.info(
        "fetched run savings",
        extra={
            "fields": {
                "stats_url": stats_url,
                "cache_read_tokens": run.cache_read_tokens,
                "prefix_freeze_busts_avoided": run.prefix_freeze_busts_avoided,
                "prefix_freeze_tokens_preserved": run.prefix_freeze_tokens_preserved,
            }
        },
    )
    return run
