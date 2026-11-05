"""Bounded attribution for savings that do not have a built-in metric."""

from __future__ import annotations

import base64
import json
import math
import re
import warnings
from collections.abc import Iterable, MutableMapping
from typing import Any

SAVINGS_ATTRIBUTION_TAG = "_headroom_savings_attribution"
_NAME_RE = re.compile(r"[^a-z0-9_.-]+")
_SCOPE_KEY = "headroom_savings_attribution"

# Per-request stage timings contributed from outside the handler, merged into
# ``RequestOutcome.pipeline_timing`` at the outcome funnel.
#
# An ASGI middleware wraps the handler, so every millisecond it spends lands in
# the client's latency while ``overhead_ms`` -- measured inside the handler --
# stays flat. An extension that halves the bill and adds 200ms per request is a
# trade the operator has to be able to see both halves of, and until now only
# one half reached the dashboard.
STAGE_TIMING_TAG = "_headroom_stage_timing"
_TIMING_SCOPE_KEY = "headroom_stage_timing"

# Stage names are extension-supplied, so they are capped like every other
# client-influenced label in this proxy (see MAX_DISTINCT_MODELS).
MAX_STAGES = 16

# Namespace, so an extension can never shadow a built-in transform's timing --
# ``deep_copy`` reported by a plugin and ``deep_copy`` reported by the pipeline
# must not accumulate into the same series.
STAGE_PREFIX = "ext:"

# NON-FINITE VALUES POISON EVERY CONSUMER DOWNSTREAM, and they do it long after
# the call that introduced them. Starlette's JSONResponse encodes with
# ``allow_nan=False``, so a single ``inf`` reaching ``/stats`` raises
# ``ValueError: Out of range float values are not JSON compliant`` -- and the
# value sits in the process-wide metrics totals, so the endpoint stays broken
# until restart. Prometheus is no better: Python renders ``inf``, the exposition
# format wants ``+Inf``, and the scrape fails to parse.
#
# The request itself still returns 200 throughout, which is the worst shape a
# bug can have: the extension looks healthy while the operator's dashboard and
# scrape are dead.
#
# One hour bounds a single stage inside one request -- unreachable in practice,
# and it makes overflow-to-infinity on accumulation structurally impossible
# (16 stages x 1h is nowhere near the float ceiling).
MAX_STAGE_MS = 3_600_000.0

# Source names are extension-supplied, so a ledger keeps at most this many
# rows by name. A source past it folds into ``other`` -- one row per
# (realized, estimated) -- so storage and the per-insert scan stay bounded
# while every token, dollar and event is still booked.
MAX_NAMED_ROWS = 256
OVERFLOW_SOURCE = "other"
# Core-booked sources never fold: ``tool_schema_savings_policy`` finds the
# deferral's row by name to lower it when hooks un-defer tools.
_UNFOLDED_SOURCES = frozenset({"tool_search"})

# A row's sums saturate here instead of overflowing to ``inf`` (see
# MAX_STAGE_MS for what a non-finite value does downstream). Both are far
# beyond any real saving; a saturated row is flagged ``saturated``.
MAX_ROW_USD = 1e15
MAX_ROW_TOKENS = 10**15


def _source_name(value: object) -> str:
    name = _NAME_RE.sub("_", str(value or "other").strip().lower()).strip("_.-")
    return (name or "other")[:64]


def _ledger(tags: MutableMapping[str, Any]) -> list[dict[str, Any]]:
    current = tags.get(SAVINGS_ATTRIBUTION_TAG)
    if isinstance(current, list):
        return current
    current = []
    tags[SAVINGS_ATTRIBUTION_TAG] = current
    return current


def bind_scope(tags: MutableMapping[str, Any], scope: MutableMapping[str, Any]) -> None:
    """Share the savings and timing ledgers between ASGI middleware and the handler.

    Both are bound together because an extension that reports one usually
    reports the other, and a handler that binds only savings would drop the
    timings silently -- which is the failure this call is here to prevent.
    """
    state = scope.setdefault("state", {})
    ledger = state.get(_SCOPE_KEY)
    if not isinstance(ledger, list):
        ledger = []
        state[_SCOPE_KEY] = ledger
    tags[SAVINGS_ATTRIBUTION_TAG] = ledger

    timings = state.get(_TIMING_SCOPE_KEY)
    if not isinstance(timings, dict):
        timings = {}
        state[_TIMING_SCOPE_KEY] = timings
    tags[STAGE_TIMING_TAG] = timings


def record_scope_savings(scope: MutableMapping[str, Any], source: object, **values: Any) -> None:
    state = scope.setdefault("state", {})
    ledger = state.get(_SCOPE_KEY)
    if not isinstance(ledger, list):
        ledger = []
        state[_SCOPE_KEY] = ledger
    record_savings({SAVINGS_ATTRIBUTION_TAG: ledger}, source, **values)


def record_scope_timing(scope: MutableMapping[str, Any], stage: object, ms: float) -> None:
    """Attribute milliseconds spent outside the handler to a named stage.

    Additive within one request, so a middleware that works in two passes
    (before and after ``call_next``) reports each and gets their sum. Never
    raises and never changes a response: a plugin's telemetry must not be able
    to break the request it is describing.
    """
    try:
        elapsed = float(ms)
    except (TypeError, ValueError):
        return
    # Non-positive is either a clock artifact or nothing happening; either way
    # it is not a measurement, and averaging it in would drag the mean toward
    # zero exactly where the stage is cheapest to ignore. Non-finite and
    # absurdly large are not measurements either, and they break consumers
    # rather than merely skewing them -- see MAX_STAGE_MS.
    if not math.isfinite(elapsed) or not 0.0 < elapsed <= MAX_STAGE_MS:
        return

    state = scope.setdefault("state", {})
    timings = state.get(_TIMING_SCOPE_KEY)
    if not isinstance(timings, dict):
        timings = {}
        state[_TIMING_SCOPE_KEY] = timings

    name = STAGE_PREFIX + _source_name(stage)
    if name not in timings and len(timings) >= MAX_STAGES:
        return
    timings[name] = round(float(timings.get(name, 0.0)) + elapsed, 4)


def timings_from_tags(tags: MutableMapping[str, Any] | None) -> dict[str, float]:
    """Extension stage timings carried on the request's tags, if any."""
    raw = (tags or {}).get(STAGE_TIMING_TAG)
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for name, value in list(raw.items())[:MAX_STAGES]:
        try:
            elapsed = float(value)
        except (TypeError, ValueError):
            continue
        # Re-checked rather than trusted: the ledger is a plain dict reachable
        # through ``tags``, so a handler can be handed one this module never
        # wrote. The guarantee has to hold at the read, not only at the write.
        #
        # Finiteness ONLY. ``MAX_STAGE_MS`` bounds a single sample at the write,
        # where it prevents overflow; applying it here would test it against an
        # ACCUMULATED total and silently discard a stage that legitimately ran
        # for longer across many samples -- throwing away real data to guard
        # against a value this path cannot produce.
        if math.isfinite(elapsed) and elapsed > 0.0:
            out[str(name)] = elapsed
    return out


def record_savings(
    tags: MutableMapping[str, Any],
    source: object,
    *,
    tokens: int = 0,
    usd: float = 0.0,
    realized: bool = True,
    estimated: bool = False,
    details: dict[str, Any] | None = None,
) -> None:
    """Attribute savings to a source; this never changes headline totals."""
    # Same hazard as MAX_STAGE_MS, on the amounts rather than the durations:
    # ``usd=inf`` reaches ``/stats`` and raises out of the JSON encoder, and
    # ``int(inf)`` raises OverflowError right here, inside the handler, on a
    # request that would otherwise have succeeded. Neither is a saving, so
    # neither is recorded -- the alternative is a plugin's arithmetic bug
    # taking down an endpoint it has nothing to do with.
    try:
        amount = float(usd or 0.0)
        count = int(tokens or 0)
    except (TypeError, ValueError, OverflowError):
        return
    if not math.isfinite(amount):
        return
    item: dict[str, Any] = {
        "source": _source_name(source),
        "realized": bool(realized),
        "estimated": bool(estimated),
        "tokens": max(0, count),
        "usd": round(amount, 12),
    }
    if details:
        item["details"] = {
            _source_name(key): value
            for key, value in list(details.items())[:12]
            if isinstance(value, (str, int, float, bool)) or value is None
        }
    accumulate(_ledger(tags), [item])


def from_tags(tags: MutableMapping[str, Any] | None) -> list[dict[str, Any]]:
    raw = (tags or {}).get(SAVINGS_ATTRIBUTION_TAG)
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def event_count(row: dict[str, Any]) -> int:
    """How many recordings a row stands for; a row without ``events`` is one."""
    try:
        return max(1, int(row.get("events", 1)))
    except (TypeError, ValueError, OverflowError):
        return 1


def accumulate(rows: list[dict[str, Any]], items: Iterable[dict[str, Any]]) -> None:
    """Add ``items`` to ``rows``, one row per (source, realized, estimated).

    Every writer goes through here -- the per-request ledger and a WS session's
    pending rows alike -- and no amount is ever dropped. Same-key rows sum
    ``tokens``, ``usd`` and ``events``; ``details`` stay only while every
    summed row agrees. Once ``rows`` holds ``MAX_NAMED_ROWS`` rows, a new
    source folds into ``OVERFLOW_SOURCE`` (which also takes unnamed savings),
    so the list grows by at most the overflow and ``_UNFOLDED_SOURCES`` keys.
    """
    for item in items:
        key = _key(item)
        target = _row_for(rows, key)
        if (
            target is None
            and key[0] not in _UNFOLDED_SOURCES
            and _row_count(rows) >= MAX_NAMED_ROWS
        ):
            if key[0] != OVERFLOW_SOURCE:
                # Details describe one source; the overflow row mixes several.
                item = {k: v for k, v in item.items() if k != "details"}
            key = (OVERFLOW_SOURCE, key[1], key[2])
            target = _row_for(rows, key)
        if target is None:
            target = {**item, "source": key[0], "tokens": 0, "usd": 0.0, "events": 0}
            rows.append(target)
        else:
            if target.get("details") != item.get("details"):
                target.pop("details", None)
            target["events"] = event_count(target)
        _add_into(target, item)


def _add_into(target: dict[str, Any], item: dict[str, Any]) -> None:
    """Sum ``item`` into ``target``; a total that hit a bound stays flagged."""
    tokens, tokens_clipped = _bounded(target.get("tokens"), item.get("tokens"), 0, MAX_ROW_TOKENS)
    usd, usd_clipped = _bounded(target.get("usd"), item.get("usd"), -MAX_ROW_USD, MAX_ROW_USD)
    target["tokens"] = int(tokens)
    target["usd"] = round(usd, 12)
    target["events"] += event_count(item)
    if tokens_clipped or usd_clipped or item.get("saturated"):
        target["saturated"] = True


def _bounded(current: Any, added: Any, low: float, high: float) -> tuple[float, bool]:
    """``current + added`` held inside ``[low, high]``, and whether it was clipped."""
    a, a_clipped = _operand(current, low, high)
    b, b_clipped = _operand(added, low, high)
    total = max(low, min(a + b, high))
    return total, a_clipped or b_clipped or total != a + b


def _operand(value: Any, low: float, high: float) -> tuple[float, bool]:
    # Rows are plain dicts a reader may edit (deferral reconcile, plugins), so
    # an operand can be junk; summing must not raise inside the handler. A
    # non-finite or oversized operand reads as the bound it passed, never as
    # zero, so a saturated total is not reset by the next update.
    try:
        number = float(value or 0)
    except OverflowError:  # an int beyond float range
        number = math.inf if value > 0 else -math.inf
    except (TypeError, ValueError):
        return 0.0, False
    if math.isnan(number):
        return 0.0, False
    bounded = max(low, min(number, high))
    return bounded, bounded != number


def _row_count(rows: list[Any]) -> int:
    return sum(1 for row in rows if isinstance(row, dict))


def _key(row: dict[str, Any]) -> tuple[str, bool, bool]:
    # ``realized`` defaults to True, as in every consumer of these rows.
    return (str(row.get("source")), bool(row.get("realized", True)), bool(row.get("estimated")))


def _row_for(rows: list[dict[str, Any]], key: tuple[str, bool, bool]) -> dict[str, Any] | None:
    # The ledger is a plain list reachable through ``tags``; skip what this
    # module did not write rather than raise inside the handler.
    return next((row for row in rows if isinstance(row, dict) and _key(row) == key), None)


_INTERNAL_TAGS = frozenset({SAVINGS_ATTRIBUTION_TAG, STAGE_TIMING_TAG})


def public_tags(tags: MutableMapping[str, Any] | None) -> dict[str, Any]:
    """Tags minus the internal ledgers, which are structures rather than labels.

    They are carried on ``tags`` because that is the one dict that reaches the
    outcome funnel from every handler; letting them through to ``RequestLog``
    would put a list and a dict into a string-keyed label store.

    Credential-shaped keys are dropped here as well. :func:`helpers.extract_tags`
    already refuses them at the source; this is the last gate before the
    request log for tags a hook or extension added by hand.
    """
    from headroom.proxy.internal_header_policy import is_credential_tag_key

    return {
        key: value
        for key, value in (tags or {}).items()
        if key not in _INTERNAL_TAGS and not is_credential_tag_key(str(key))
    }


def encode(items: list[dict[str, Any]]) -> str:
    if not items:
        return "none"
    payload = json.dumps(items, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def decode(value: str) -> list[dict[str, Any]]:
    if not value or value == "none":
        return []
    try:
        padded = value + "=" * (-len(value) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(padded).decode())
    except Exception:
        return []
    return (
        [dict(item) for item in decoded if isinstance(item, dict)]
        if isinstance(decoded, list)
        else []
    )


def __getattr__(name: str) -> Any:
    # MAX_SOURCES used to cap a request's ledger at 32 rows and silently drop
    # the rest; rows now coalesce per source, and sources past MAX_NAMED_ROWS
    # fold into OVERFLOW_SOURCE. Kept for importers.
    if name == "MAX_SOURCES":
        warnings.warn(
            "headroom.proxy.savings_attribution.MAX_SOURCES is unused and will be "
            "removed; rows coalesce per source, past MAX_NAMED_ROWS into 'other'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return 32
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
