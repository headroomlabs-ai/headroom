"""Attribution and timing contributed by proxy extensions.

An extension that changes the bill has to be able to say so, or the operator
sees a different total with nothing to explain it. The savings half of this
already existed but only reached two of the three handler families; the timing
half did not exist at all, so an extension's own latency was invisible —
``overhead_ms`` is measured inside the handler that the extension wraps.
"""

from __future__ import annotations

import math

import pytest

from headroom.proxy.savings_attribution import (
    MAX_STAGE_MS,
    MAX_STAGES,
    SAVINGS_ATTRIBUTION_TAG,
    STAGE_PREFIX,
    STAGE_TIMING_TAG,
    bind_scope,
    from_tags,
    public_tags,
    record_scope_savings,
    record_scope_timing,
    timings_from_tags,
)


def _scope() -> dict:
    return {"type": "http", "method": "POST"}


# --- savings, from middleware ------------------------------------------------


def test_middleware_savings_reach_the_handlers_tags() -> None:
    """The contract: middleware records into the scope before the handler runs,
    the handler binds, and the outcome funnel reads one ledger."""
    scope = _scope()
    record_scope_savings(scope, "routemegood", usd=0.42)

    tags: dict = {}
    bind_scope(tags, scope)

    (row,) = from_tags(tags)
    assert row["source"] == "routemegood"
    assert row["usd"] == 0.42


def test_savings_can_be_money_without_being_tokens() -> None:
    """The gap this closes. Every other savings channel computes
    ``saved = before - after`` and three of them refuse a non-positive value,
    so an extension that routes a request to a cheaper model — same tokens,
    smaller bill — could only report by inventing a token count nobody saved."""
    scope = _scope()
    record_scope_savings(scope, "model_router", tokens=0, usd=1.75)

    tags: dict = {}
    bind_scope(tags, scope)
    (row,) = from_tags(tags)
    assert row["tokens"] == 0
    assert row["usd"] == 1.75


def test_a_projection_is_not_a_measurement() -> None:
    scope = _scope()
    record_scope_savings(scope, "guess", usd=1.0, realized=False)
    record_scope_savings(scope, "guess", usd=1.0, realized=True)

    tags: dict = {}
    bind_scope(tags, scope)
    assert sorted(row["realized"] for row in from_tags(tags)) == [False, True]


# --- timing ------------------------------------------------------------------


def test_middleware_timing_reaches_the_handlers_tags() -> None:
    scope = _scope()
    record_scope_timing(scope, "routemegood", 12.5)

    tags: dict = {}
    bind_scope(tags, scope)
    assert timings_from_tags(tags) == {f"{STAGE_PREFIX}routemegood": 12.5}


def test_timing_is_additive_within_one_request() -> None:
    """A middleware works in two passes — before ``call_next`` and after — and
    should be able to report each without tracking the total itself."""
    scope = _scope()
    record_scope_timing(scope, "ext", 4.0)
    record_scope_timing(scope, "ext", 2.5)

    tags: dict = {}
    bind_scope(tags, scope)
    assert timings_from_tags(tags) == {f"{STAGE_PREFIX}ext": 6.5}


def test_extension_stages_are_namespaced() -> None:
    """``deep_copy`` reported by a plugin and ``deep_copy`` measured by the
    pipeline must not accumulate into the same series."""
    scope = _scope()
    record_scope_timing(scope, "deep_copy", 1.0)

    tags: dict = {}
    bind_scope(tags, scope)
    assert list(timings_from_tags(tags)) == [f"{STAGE_PREFIX}deep_copy"]


@pytest.mark.parametrize(
    "bad",
    [
        0,
        -1.0,
        None,
        "slow",
        float("nan"),
        float("inf"),
        float("-inf"),
        1e400,
        MAX_STAGE_MS + 1,
    ],
)
def test_a_non_measurement_is_not_recorded(bad) -> None:
    """Zero and negative are clock artifacts, not observations; averaging them
    in would drag the mean down exactly where the stage is cheapest to skip.

    Non-finite is worse than skew. Starlette encodes ``/stats`` with
    ``allow_nan=False``, so one ``inf`` raises out of the JSON encoder — and it
    lands in process-wide metrics totals, so the endpoint stays broken until
    restart while the request that caused it returns 200.
    """
    scope = _scope()
    record_scope_timing(scope, "ext", bad)

    tags: dict = {}
    bind_scope(tags, scope)
    assert timings_from_tags(tags) == {}


def test_a_poisoned_ledger_is_rejected_on_read_too() -> None:
    """The ledger is a plain dict reachable through ``tags``, so a handler can
    be handed one this module never wrote. The guarantee holds at the read."""
    assert timings_from_tags({STAGE_TIMING_TAG: {"ext:a": float("inf"), "ext:b": 2.0}}) == {
        "ext:b": 2.0
    }


def test_accumulation_cannot_overflow_to_infinity() -> None:
    """Two finite values can sum to ``inf``. Bounding each SAMPLE makes that
    unreachable rather than merely unlikely."""
    scope = _scope()
    for _ in range(4):
        record_scope_timing(scope, "ext", MAX_STAGE_MS)

    tags: dict = {}
    bind_scope(tags, scope)
    (total,) = timings_from_tags(tags).values()
    assert math.isfinite(total)


def test_an_accumulated_total_may_exceed_the_per_sample_bound() -> None:
    """The bound is on one sample, not on the sum. Testing it against the
    accumulated total would silently discard a stage that legitimately ran
    longer across many samples — throwing away real data to guard a value the
    write path cannot produce."""
    scope = _scope()
    for _ in range(3):
        record_scope_timing(scope, "ext", MAX_STAGE_MS)

    tags: dict = {}
    bind_scope(tags, scope)
    assert timings_from_tags(tags) == {f"{STAGE_PREFIX}ext": MAX_STAGE_MS * 3}


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_a_non_finite_amount_is_not_a_saving(bad) -> None:
    """Pre-existing, and the same crash: ``usd=inf`` reaches ``/stats`` and
    raises out of the JSON encoder."""
    scope = _scope()
    record_scope_savings(scope, "buggy", usd=bad)

    tags: dict = {}
    bind_scope(tags, scope)
    assert from_tags(tags) == []


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_a_non_finite_token_count_does_not_raise_inside_the_handler(bad) -> None:
    """``int(inf)`` is an OverflowError, raised on a request that would
    otherwise have succeeded. A plugin's arithmetic bug must not become the
    proxy's 500."""
    scope = _scope()
    record_scope_savings(scope, "buggy", tokens=bad)

    tags: dict = {}
    bind_scope(tags, scope)
    assert from_tags(tags) == []


def test_a_real_saving_still_records_after_the_guards() -> None:
    """The direction that must not be lost while hardening the other one."""
    scope = _scope()
    record_scope_savings(scope, "routemegood", tokens=10, usd=0.5)
    record_scope_timing(scope, "routemegood", 3.0)

    tags: dict = {}
    bind_scope(tags, scope)
    assert from_tags(tags)[0]["usd"] == 0.5
    assert timings_from_tags(tags) == {f"{STAGE_PREFIX}routemegood": 3.0}


def test_stage_cardinality_is_capped() -> None:
    """Stage names are extension-supplied, so they are bounded like every other
    client-influenced label in this proxy."""
    scope = _scope()
    for i in range(MAX_STAGES * 4):
        record_scope_timing(scope, f"stage-{i}", 1.0)

    tags: dict = {}
    bind_scope(tags, scope)
    assert len(timings_from_tags(tags)) == MAX_STAGES


def test_an_existing_stage_still_accumulates_at_the_cap() -> None:
    """The cap bounds distinct names, not measurements. A stage already being
    tracked must keep accumulating or its total silently stops growing."""
    scope = _scope()
    for i in range(MAX_STAGES):
        record_scope_timing(scope, f"stage-{i}", 1.0)
    record_scope_timing(scope, "stage-0", 5.0)

    tags: dict = {}
    bind_scope(tags, scope)
    assert timings_from_tags(tags)[f"{STAGE_PREFIX}stage-0"] == 6.0


def test_recording_before_any_bind_still_works() -> None:
    """Ordering is not guaranteed: middleware runs first, and on a path where
    the handler never binds, nothing should raise."""
    scope = _scope()
    record_scope_timing(scope, "ext", 1.0)
    record_scope_savings(scope, "ext", usd=1.0)
    assert scope["state"]


def test_recording_after_bind_is_seen_by_the_already_bound_tags() -> None:
    """A middleware measures its own post-response work AFTER the handler has
    bound. Sharing one object rather than copying is what makes that land."""
    tags: dict = {}
    scope = _scope()
    bind_scope(tags, scope)
    record_scope_timing(scope, "ext", 3.0)
    record_scope_savings(scope, "ext", usd=0.5)

    assert timings_from_tags(tags) == {f"{STAGE_PREFIX}ext": 3.0}
    assert from_tags(tags)[0]["usd"] == 0.5


def test_bind_is_idempotent() -> None:
    tags: dict = {}
    scope = _scope()
    bind_scope(tags, scope)
    record_scope_timing(scope, "ext", 1.0)
    bind_scope(tags, scope)
    record_scope_timing(scope, "ext", 1.0)
    assert timings_from_tags(tags) == {f"{STAGE_PREFIX}ext": 2.0}


def test_timings_from_tags_tolerates_junk() -> None:
    for junk in (
        None,
        {},
        {STAGE_TIMING_TAG: "nope"},
        {STAGE_TIMING_TAG: []},
        {STAGE_TIMING_TAG: {"a": "b"}},
    ):
        assert timings_from_tags(junk) == {}


# --- the ledgers are structures, not labels ---------------------------------


def test_neither_ledger_leaks_into_request_log_tags() -> None:
    """They ride on ``tags`` because that is the one dict reaching the outcome
    funnel from every handler. A list and a dict must not land in a
    string-keyed label store."""
    tags: dict = {"client": "claude-code"}
    scope = _scope()
    bind_scope(tags, scope)
    record_scope_savings(scope, "ext", usd=1.0)
    record_scope_timing(scope, "ext", 1.0)

    assert public_tags(tags) == {"client": "claude-code"}
    assert SAVINGS_ATTRIBUTION_TAG not in public_tags(tags)
    assert STAGE_TIMING_TAG not in public_tags(tags)


# --- through the outcome funnel ---------------------------------------------

pytest.importorskip("fastapi")


class _Harness:
    """Just enough of HeadroomProxy to drive the real funnel method.

    Mirrors ``tests/test_request_outcome.py::_FunnelHarness`` — the real
    implementation is bound to the harness, so nothing under test is mocked.
    """

    def __init__(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from headroom.proxy.server import HeadroomProxy

        self.metrics = MagicMock()
        self.metrics.record_request = AsyncMock()
        self.cost_tracker = MagicMock()
        self.logger = None
        self._record_request_outcome = HeadroomProxy._record_request_outcome.__get__(
            self, type(self)
        )


def _outcome(**overrides):
    from headroom.proxy.outcome import RequestOutcome

    defaults = {
        "request_id": "req-1",
        "provider": "anthropic",
        "model": "claude-sonnet-4",
        "original_tokens": 1000,
        "optimized_tokens": 1000,
        "output_tokens": 50,
        "tokens_saved": 0,
        "attempted_input_tokens": 1000,
    }
    defaults.update(overrides)
    return RequestOutcome(**defaults)


@pytest.mark.asyncio
async def test_extension_timing_reaches_pipeline_timing() -> None:
    """The whole point of the timing half: ``pipeline_timing`` is what
    ``/stats``, the dashboard's Performance panel and
    ``headroom_transform_timing_ms_*`` are all built on."""
    scope = _scope()
    record_scope_timing(scope, "routemegood", 8.0)
    tags: dict = {}
    bind_scope(tags, scope)

    h = _Harness()
    await h._record_request_outcome(_outcome(tags=tags, pipeline_timing={"deep_copy": 1.0}))

    timing = h.metrics.record_request.await_args.kwargs["pipeline_timing"]
    assert timing == {"deep_copy": 1.0, f"{STAGE_PREFIX}routemegood": 8.0}


@pytest.mark.asyncio
async def test_a_handler_timing_wins_a_name_collision() -> None:
    """Namespacing makes this unreachable today; it is asserted so that if the
    prefix ever goes, a plugin still cannot overwrite a measurement the
    pipeline made of itself."""
    tags = {STAGE_TIMING_TAG: {"deep_copy": 99.0}}

    h = _Harness()
    await h._record_request_outcome(_outcome(tags=tags, pipeline_timing={"deep_copy": 1.0}))

    timing = h.metrics.record_request.await_args.kwargs["pipeline_timing"]
    assert timing["deep_copy"] == 1.0


@pytest.mark.asyncio
async def test_no_extension_timing_leaves_pipeline_timing_untouched() -> None:
    """Including identity: a request with no extension must pass the handler's
    own dict through, not a rebuilt copy of it."""
    original = {"deep_copy": 1.0}

    h = _Harness()
    await h._record_request_outcome(_outcome(pipeline_timing=original))

    assert h.metrics.record_request.await_args.kwargs["pipeline_timing"] is original


@pytest.mark.asyncio
async def test_extension_savings_reach_the_metrics_call() -> None:
    scope = _scope()
    record_scope_savings(scope, "routemegood", usd=0.42, tokens=0)
    tags: dict = {}
    bind_scope(tags, scope)

    h = _Harness()
    await h._record_request_outcome(_outcome(tags=tags))

    attribution = h.metrics.record_request.await_args.kwargs["savings_attribution"]
    assert [(row["source"], row["usd"]) for row in attribution] == [("routemegood", 0.42)]
