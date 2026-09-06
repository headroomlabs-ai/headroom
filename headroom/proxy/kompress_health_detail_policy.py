"""Policy for the public ``detail`` field on the kompress health check.

``/health`` and ``/readyz`` are auth-exempt, so everything they serialize is
unauthenticated public data. The kompress check reports ``detail`` because
``ready=false`` on its own is ambiguous — the model may be warming in the
background, missing from the local cache, or the ML extras may not be installed
at all (#2730).

That diagnostic value must not come at the cost of leaking internals. Native
loader, model and cache failures routinely embed absolute filesystem paths,
model identifiers, endpoint URLs and configuration values in their exception
text, and a warmup slot's ``error`` string is populated from exactly those
failures. So the public field draws from a closed vocabulary: anything not in
it degrades to ``unavailable`` rather than being echoed.

The precise cause is not lost — it stays in the proxy log and in the warmup
slot, which ``/debug/warmup`` serializes behind authentication.
"""

from __future__ import annotations

# The model is loaded and serving.
KOMPRESS_DETAIL_LOADED = "loaded"
# The background warm thread is still running.
KOMPRESS_DETAIL_WARMING = "warming"
# Cache-only warm found no local artifacts; no network call was made.
KOMPRESS_DETAIL_NOT_CACHED = "model not cached"
# The warm attempt raised. The cause is in the log, not here.
KOMPRESS_DETAIL_WARM_FAILED = "warm failed"
# The ML extras are absent, or no compressor was ever constructed.
KOMPRESS_DETAIL_NOT_INSTALLED = "not installed"

# ``ContentRouter.eager_load_compressors()`` source statuses. These are fixed
# literals in the router, never interpolated, so they are safe to surface.
KOMPRESS_DETAIL_ENABLED = "enabled"
KOMPRESS_DETAIL_DEFERRED = "deferred"
KOMPRESS_DETAIL_UNAVAILABLE = "unavailable"

#: Every value ``/health`` and ``/readyz`` may report for ``kompress.detail``.
KOMPRESS_HEALTH_DETAILS = frozenset(
    {
        KOMPRESS_DETAIL_LOADED,
        KOMPRESS_DETAIL_WARMING,
        KOMPRESS_DETAIL_NOT_CACHED,
        KOMPRESS_DETAIL_WARM_FAILED,
        KOMPRESS_DETAIL_NOT_INSTALLED,
        KOMPRESS_DETAIL_ENABLED,
        KOMPRESS_DETAIL_DEFERRED,
        KOMPRESS_DETAIL_UNAVAILABLE,
    }
)

#: What an out-of-vocabulary value collapses to.
DEFAULT_KOMPRESS_HEALTH_DETAIL = KOMPRESS_DETAIL_UNAVAILABLE


def public_detail(value: str | None) -> str:
    """Return ``value`` if it is a known detail, else the bounded fallback.

    This is the single choke point for the field. Route *every* candidate
    through it — including strings that look safe at the call site, such as a
    warmup slot's ``source_status`` — so a future code path cannot introduce a
    leak by writing free-form text into the slot.
    """
    if not isinstance(value, str):
        return DEFAULT_KOMPRESS_HEALTH_DETAIL
    normalized = value.strip().lower()
    if normalized in KOMPRESS_HEALTH_DETAILS:
        return normalized
    return DEFAULT_KOMPRESS_HEALTH_DETAIL
