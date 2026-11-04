"""One mismatched provider count must reach billing and delta math differently.

Splitting ``optimized_tokens`` (local) from ``provider_input_tokens`` (provider)
is only half the fix. Two consumers derive their own numbers downstream, and each
needs the OTHER side of the split:

* ``telemetry.session._fold`` accumulates ``tokens.input``, a billed/volume figure
  that sits beside ``output``/``cache_read``/``cache_write``/``uncached`` — all
  provider-reported. It must prefer the provider count.
* ``PrometheusMetrics.record_request`` reconstructs the durable savings ledger as
  ``tokens_before = input_tokens + tokens_saved``, ``tokens_after = input_tokens``.
  That is a DELTA, so pairing a provider ``input_tokens`` with a locally-counted
  ``tokens_saved`` straddles two rulers — local 10->6 with the provider reporting
  8 would record 12->8.

These drive the real funnel with one deliberately mismatched pair and assert both
semantics, rather than asserting on the dataclass alone.
"""

from __future__ import annotations

import pytest

from headroom.proxy.outcome import RequestOutcome

# Local 10 -> 6 (saved 4); the provider says the prompt was 8. Every number below
# is derived from exactly this one mismatch.
_LOCAL_ORIGINAL = 10
_LOCAL_OPTIMIZED = 6
_LOCAL_SAVED = 4
_PROVIDER_INPUT = 8


def _outcome() -> RequestOutcome:
    return RequestOutcome(
        request_id="r1",
        provider="openai",
        model="gpt-4o-mini",
        original_tokens=_LOCAL_ORIGINAL,
        optimized_tokens=_LOCAL_OPTIMIZED,
        provider_input_tokens=_PROVIDER_INPUT,
        output_tokens=5,
        tokens_saved=_LOCAL_SAVED,
        attempted_input_tokens=_LOCAL_OPTIMIZED + _LOCAL_SAVED,
    )


def test_beacon_input_is_the_billed_provider_count() -> None:
    """``tokens.input`` is a volume figure and must not silently become local."""
    from headroom.telemetry import session as sess_mod

    sess = sess_mod._Session(sid="s1", started=0.0, last_seen=0.0)  # type: ignore[attr-defined]
    sess_mod._fold(sess, _outcome(), now=0.0, source="proxy")  # type: ignore[attr-defined]

    assert sess.input_tokens == _PROVIDER_INPUT, (
        "the beacon's input volume must use the provider count, not the local one"
    )
    # The local pair still drives the reduction ratios.
    assert sess.original_tokens == _LOCAL_ORIGINAL
    assert sess.tokens_saved == _LOCAL_SAVED


def test_beacon_falls_back_to_local_when_no_provider_count() -> None:
    """Providers that report no usage must behave exactly as before the split."""
    from headroom.telemetry import session as sess_mod

    o = RequestOutcome(
        request_id="r2",
        provider="anthropic",
        model="claude-sonnet-4-6",
        original_tokens=_LOCAL_ORIGINAL,
        optimized_tokens=_LOCAL_OPTIMIZED,
        output_tokens=5,
        tokens_saved=_LOCAL_SAVED,
        attempted_input_tokens=_LOCAL_OPTIMIZED + _LOCAL_SAVED,
    )
    sess = sess_mod._Session(sid="s2", started=0.0, last_seen=0.0)  # type: ignore[attr-defined]
    sess_mod._fold(sess, o, now=0.0, source="proxy")  # type: ignore[attr-defined]

    assert sess.input_tokens == _LOCAL_OPTIMIZED


@pytest.mark.asyncio
async def test_ledger_delta_stays_on_the_local_ruler(monkeypatch) -> None:
    """Drive the real record_request and capture what reaches the ledger.

    The ledger stores a delta, so both ends must be local. With the billed input
    (8, provider) paired against a local tokens_saved (4), the old shape recorded
    12 -> 8 for a request that actually went 10 -> 6.
    """
    from headroom.proxy import prometheus_metrics as pm

    seen: dict = {}

    def fake_record_savings_event(**kw):
        seen.update(kw)

    monkeypatch.setattr(pm.savings_ledger, "record_savings_event", fake_record_savings_event)

    metrics = pm.PrometheusMetrics()
    metrics._stateless = False  # the ledger write is skipped when stateless

    await metrics.record_request(
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=_PROVIDER_INPUT,  # billed/volume: provider's count
        local_input_tokens=_LOCAL_OPTIMIZED,  # same ruler as tokens_saved
        output_tokens=5,
        tokens_saved=_LOCAL_SAVED,
        latency_ms=1.0,
    )

    assert seen, "no ledger event recorded"
    assert seen["tokens_before"] == _LOCAL_ORIGINAL, "before must be the local original"
    assert seen["tokens_after"] == _LOCAL_OPTIMIZED, "after must be the local optimized"
    # The mixed-ruler shape this replaces.
    assert (seen["tokens_before"], seen["tokens_after"]) != (
        _PROVIDER_INPUT + _LOCAL_SAVED,
        _PROVIDER_INPUT,
    )
    # Volume still counted on the billed figure.
    assert metrics.tokens_input_total == _PROVIDER_INPUT


@pytest.mark.asyncio
async def test_ledger_falls_back_to_billed_when_local_omitted(monkeypatch) -> None:
    """Pre-split callers keep their existing (single-value) behaviour."""
    from headroom.proxy import prometheus_metrics as pm

    seen: dict = {}
    monkeypatch.setattr(pm.savings_ledger, "record_savings_event", lambda **kw: seen.update(kw))
    metrics = pm.PrometheusMetrics()
    metrics._stateless = False

    await metrics.record_request(
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=_PROVIDER_INPUT,
        output_tokens=5,
        tokens_saved=_LOCAL_SAVED,
        latency_ms=1.0,
    )

    assert seen["tokens_after"] == _PROVIDER_INPUT


def test_record_request_defaults_local_to_billed_when_omitted() -> None:
    """Callers that never pass local_input_tokens keep pre-split behaviour."""
    import inspect

    from headroom.proxy.prometheus_metrics import PrometheusMetrics

    sig = inspect.signature(PrometheusMetrics.record_request)
    assert sig.parameters["local_input_tokens"].default is None
