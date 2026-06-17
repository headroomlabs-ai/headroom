"""Tests for :class:`headroom.proxy.outcome.RequestOutcome` and the
:meth:`HeadroomProxy._record_request_outcome` funnel.

The point of this file is the *contract* — every behavioural assertion
here is a thing that, prior to the funnel, lived inline at one or more
of the 18 metrics-emit sites identified in
``docs/superpowers/specs/P0-proxy-pipeline-audit.md``. Locking the
contract in tests means future migrations onto the funnel cannot
silently regress the wire shape.
"""

from __future__ import annotations

import logging
from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from headroom.proxy.outcome import RequestOutcome

# ── Value-type contract ────────────────────────────────────────────────


def _outcome(**overrides: Any) -> RequestOutcome:
    """Construct a RequestOutcome with sensible defaults; override fields per test."""
    defaults: dict[str, Any] = {
        "request_id": "req-1",
        "provider": "anthropic",
        "model": "claude-sonnet-4",
        "original_tokens": 1000,
        "optimized_tokens": 300,
        "output_tokens": 50,
        "tokens_saved": 700,
        "attempted_input_tokens": 800,
    }
    defaults.update(overrides)
    return RequestOutcome(**defaults)


def test_outcome_is_frozen() -> None:
    """Mutability would let a handler patch the outcome after handing it
    to the funnel — bypassing the contract. Must error."""
    o = _outcome()
    with pytest.raises(FrozenInstanceError):
        o.cache_read_tokens = 999  # type: ignore[misc]


def test_cache_hit_is_derived_not_stored() -> None:
    """Pre-refactor, 9 of 18 ``RequestLog`` sites hardcoded ``cache_hit=False``
    even when ``cache_read_tokens > 0``. Deriving from the actual value
    makes "forgot to compute it" structurally impossible."""
    assert _outcome(cache_read_tokens=0).cache_hit is False
    assert _outcome(cache_read_tokens=1).cache_hit is True
    assert _outcome(cache_read_tokens=500).cache_hit is True


def test_cache_hit_pct_handles_zero_denominator() -> None:
    """No reads + no writes is a no-cache request, not a 0%-hit cache request.
    Returning 0 here is correct as long as dashboards distinguish via the
    absolute ``cache_read_tokens`` / ``cache_write_tokens`` values."""
    assert _outcome(cache_read_tokens=0, cache_write_tokens=0).cache_hit_pct == 0


def test_cache_hit_pct_rounds_to_int() -> None:
    """PERF log line consumed by ``headroom perf`` parses an integer here;
    keep the type contract tight."""
    o = _outcome(cache_read_tokens=2, cache_write_tokens=1)  # 66.66%
    assert o.cache_hit_pct == 67
    assert isinstance(o.cache_hit_pct, int)


def test_savings_pct_handles_zero_original() -> None:
    """A request with 0 original tokens — e.g. an empty body — should not
    raise ZeroDivisionError. Sites pre-refactor handled this inconsistently."""
    assert _outcome(original_tokens=0).savings_pct == 0.0


def test_savings_pct_basic() -> None:
    assert _outcome(original_tokens=1000, tokens_saved=300).savings_pct == 30.0


def test_provider_specific_fields_default_to_zero() -> None:
    """Anthropic's 5m/1h cache TTL splits don't exist on OpenAI / Gemini.
    The dataclass defaults them to 0 so non-Anthropic handlers don't
    have to know about them."""
    o = _outcome(provider="openai", cache_read_tokens=100, cache_write_tokens=200)
    assert o.cache_write_5m_tokens == 0
    assert o.cache_write_1h_tokens == 0
    # And OpenAI's "inferred" flag defaults False — only the OpenAI
    # handler sets it True after running _infer_openai_cache_write_tokens.
    assert o.cache_inferred is False


def test_optional_fields_default_to_neutral_values() -> None:
    """Handlers that don't have a field (e.g. Bedrock with no waste_signals)
    must not have to pass anything — defaults handle it."""
    o = _outcome()
    assert o.ttfb_ms == 0.0
    assert o.pipeline_timing is None
    assert o.waste_signals is None
    assert o.transforms_applied == ()
    assert o.turn_id is None
    assert o.request_messages is None
    assert o.tags == {}
    assert o.client is None  # unidentified harness


def test_client_field_round_trips() -> None:
    """The ``client`` field is the proof point that the refactor pays
    out across harnesses — one field-add gives every dashboard a
    per-harness dimension for free.
    """
    o = _outcome(client="codex")
    assert o.client == "codex"


def test_stream_outcome_derives_gemini_contents_metadata() -> None:
    outcome = RequestOutcome.from_stream(
        body={
            "systemInstruction": {"parts": [{"text": "sys"}]},
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
        },
        provider="vertex:google",
        model="gemini-2.0-flash",
        request_id="req-gemini-stream",
        original_tokens=12,
        optimized_tokens=10,
        output_tokens=3,
        tokens_saved=2,
        transforms_applied=["compress"],
        total_latency_ms=25.0,
        overhead_ms=4.0,
        tags={"route": "vertex"},
        client="codex",
        log_full_messages=True,
    )

    assert outcome.provider == "vertex:google"
    assert outcome.num_messages == 1
    assert outcome.request_messages == [{"role": "user", "parts": [{"text": "hello"}]}]
    assert outcome.turn_id is not None


# ── classify_client — the harness ID source ─────────────────────────


def test_classify_client_recognises_known_harness_user_agents() -> None:
    from headroom.proxy.auth_mode import classify_client

    cases = [
        ({"User-Agent": "codex-cli/0.30.0 (osx)"}, "codex"),
        ({"User-Agent": "claude-code/1.4.2"}, "claude-code"),
        ({"User-Agent": "claude-cli/2.0"}, "claude-code"),  # aliased
        ({"User-Agent": "cursor/0.42.1 (electron)"}, "cursor"),
        ({"User-Agent": "aider/0.50.0"}, "aider"),
        ({"User-Agent": "zed/0.143.0"}, "zed"),
        ({"User-Agent": "opencode/1.0"}, "opencode"),
        ({"User-Agent": "github-copilot/x.y.z"}, "copilot"),
    ]
    for headers, expected in cases:
        assert classify_client(headers) == expected, headers


def test_classify_client_x_client_header_wins_over_user_agent() -> None:
    from headroom.proxy.auth_mode import classify_client

    # X-Client wins even when UA matches a different harness
    h = {"User-Agent": "codex-cli/0.30.0", "X-Client": "my-custom-harness"}
    assert classify_client(h) == "my-custom-harness"


def test_classify_client_recognises_codex_desktop_originator() -> None:
    from headroom.proxy.auth_mode import classify_client

    assert classify_client({"originator": "Codex Desktop"}) == "codex"


def test_classify_client_returns_none_for_unknown_traffic() -> None:
    """``None`` is the loud "unidentified" signal — downstream consumers
    can group these as "unknown" rather than silently bucketing into
    a default that would mislead dashboards."""
    from headroom.proxy.auth_mode import classify_client

    assert classify_client({"User-Agent": "Mozilla/5.0"}) is None
    assert classify_client({}) is None
    assert classify_client({"User-Agent": ""}) is None


# ── Funnel contract (_record_request_outcome) ──────────────────────────


class _CollectingLogger:
    """Minimal stand-in for ``RequestLogger``."""

    def __init__(self) -> None:
        self.logs: list[Any] = []

    def log(self, entry: Any) -> None:
        self.logs.append(entry)


class _FunnelHarness:
    """Pulls just enough of HeadroomProxy onto an object to exercise
    ``_record_request_outcome`` without instantiating the full proxy.

    The harness assigns the real method to ``self`` via descriptor
    binding so the implementation is exactly the production one — no
    forking, no mock-the-thing-you're-testing.
    """

    def __init__(self, *, with_cost_tracker: bool = True, with_logger: bool = True) -> None:
        from headroom.proxy.server import HeadroomProxy

        self.metrics = MagicMock()
        self.metrics.record_request = AsyncMock()
        self.cost_tracker = MagicMock() if with_cost_tracker else None
        self.logger = _CollectingLogger() if with_logger else None
        # Bind the real method to this harness.
        self._record_request_outcome = HeadroomProxy._record_request_outcome.__get__(
            self, type(self)
        )


@pytest.mark.asyncio
async def test_funnel_calls_metrics_with_full_kwargs() -> None:
    """The funnel must pass EVERY field that
    ``PrometheusMetrics.record_request`` knows about, not the
    pre-refactor "pass-what-was-convenient" subset. Otherwise a handler
    that forgets to populate a field silently degrades dashboard data."""
    h = _FunnelHarness()
    o = _outcome(
        provider="openai",
        model="gpt-4",
        optimized_tokens=300,
        output_tokens=50,
        tokens_saved=700,
        attempted_input_tokens=800,
        cache_read_tokens=200,
        cache_write_tokens=100,
        cache_write_5m_tokens=50,
        cache_write_1h_tokens=50,
        uncached_input_tokens=0,
        total_latency_ms=1234.5,
        overhead_ms=12.3,
        ttfb_ms=200.0,
        pipeline_timing={"phase": 1.0},
        waste_signals={"skipped": 3},
    )
    await h._record_request_outcome(o)

    h.metrics.record_request.assert_awaited_once()
    kwargs = h.metrics.record_request.await_args.kwargs
    assert kwargs["provider"] == "openai"
    assert kwargs["model"] == "gpt-4"
    assert kwargs["input_tokens"] == 300  # optimized → input
    assert kwargs["output_tokens"] == 50
    assert kwargs["tokens_saved"] == 700
    assert kwargs["latency_ms"] == 1234.5
    assert kwargs["cached"] is True  # derived from cache_read > 0
    assert kwargs["overhead_ms"] == 12.3
    assert kwargs["ttfb_ms"] == 200.0
    assert kwargs["pipeline_timing"] == {"phase": 1.0}
    assert kwargs["waste_signals"] == {"skipped": 3}
    assert kwargs["cache_read_tokens"] == 200
    assert kwargs["cache_write_tokens"] == 100
    assert kwargs["cache_write_5m_tokens"] == 50
    assert kwargs["cache_write_1h_tokens"] == 50
    assert kwargs["uncached_input_tokens"] == 0
    assert kwargs["attempted_input_tokens"] == 800


@pytest.mark.asyncio
async def test_funnel_passes_canonical_record_tokens_shape() -> None:
    """``cost_tracker.record_tokens`` takes ``(model, tokens_saved,
    optimized_tokens)`` positionally and the cache args as kwargs. The
    funnel preserves this — moving anything to positional would break
    sites that pass kwargs explicitly."""
    h = _FunnelHarness()
    o = _outcome(
        model="claude-sonnet-4",
        optimized_tokens=300,
        tokens_saved=700,
        cache_read_tokens=200,
        cache_write_tokens=100,
        cache_write_5m_tokens=80,
        cache_write_1h_tokens=20,
        uncached_input_tokens=0,
    )
    await h._record_request_outcome(o)

    h.cost_tracker.record_tokens.assert_called_once()
    args, kwargs = h.cost_tracker.record_tokens.call_args
    assert args == ("claude-sonnet-4", 700, 300)
    assert kwargs == {
        "cache_read_tokens": 200,
        "cache_write_tokens": 100,
        "cache_write_5m_tokens": 80,
        "cache_write_1h_tokens": 20,
        "uncached_tokens": 0,
        "output_tokens": 50,
    }


@pytest.mark.asyncio
async def test_funnel_skips_cost_tracker_when_absent() -> None:
    """When the proxy was started with ``--no-cost``, ``cost_tracker``
    is None and the funnel must skip step 2 silently."""
    h = _FunnelHarness(with_cost_tracker=False)
    await h._record_request_outcome(_outcome())
    # No crash, metrics still recorded.
    h.metrics.record_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_funnel_logs_request_with_derived_cache_hit() -> None:
    """The RequestLog row needs cache_hit derived from cache_read>0, not
    the hardcoded False that 9 of 18 pre-refactor sites used."""
    h = _FunnelHarness()
    await h._record_request_outcome(_outcome(cache_read_tokens=200, cache_write_tokens=100))
    assert len(h.logger.logs) == 1
    log_entry = h.logger.logs[0]
    assert log_entry.cache_hit is True


@pytest.mark.asyncio
async def test_funnel_skips_request_log_when_logger_absent() -> None:
    """Same pattern as cost_tracker — optional surface."""
    h = _FunnelHarness(with_logger=False)
    await h._record_request_outcome(_outcome())
    h.metrics.record_request.assert_awaited_once()  # still happens


@pytest.mark.asyncio
async def test_funnel_emits_perf_log_with_canonical_shape(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``headroom perf`` parses this exact ``key=value`` format. Changing
    it breaks the analyzer. The contract: model, msgs, tok_before,
    tok_after, tok_saved, cache_read, cache_write, cache_hit_pct,
    opt_ms, transforms — in that order, space-separated."""
    h = _FunnelHarness()
    # Direct handler attach: caplog otherwise drops propagation-disabled
    # records (the proxy disables ``headroom.*`` propagation once started).
    target = logging.getLogger("headroom.proxy")
    captured: list[logging.LogRecord] = []

    class _H(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _H(level=logging.INFO)
    target.addHandler(handler)
    prior_level = target.level
    target.setLevel(logging.INFO)
    try:
        await h._record_request_outcome(
            _outcome(
                request_id="req-perf",
                model="gpt-4",
                original_tokens=1000,
                optimized_tokens=300,
                tokens_saved=700,
                cache_read_tokens=200,
                cache_write_tokens=100,
                num_messages=5,
                overhead_ms=12.0,
                transforms_applied=("smart_crusher", "content_router"),
            )
        )
    finally:
        target.removeHandler(handler)
        target.setLevel(prior_level)

    perf_lines = [r.getMessage() for r in captured if " PERF " in r.getMessage()]
    assert len(perf_lines) == 1
    line = perf_lines[0]
    assert "[req-perf] PERF " in line
    assert "model=gpt-4" in line
    assert "msgs=5" in line
    assert "tok_before=1000" in line
    assert "tok_after=300" in line
    assert "tok_saved=700" in line
    assert "cache_read=200" in line
    assert "cache_write=100" in line
    assert "cache_hit_pct=67" in line  # 200/(200+100) * 100 = 67
    assert "opt_ms=12" in line


# ── Funnel: per-client analytics surface ─────────────────────────────


@pytest.mark.asyncio
async def test_funnel_appends_client_to_perf_log_when_set() -> None:
    """``headroom perf --client X`` filtering relies on the ``client=X``
    token at the end of the PERF line. Absent client means no token —
    the PERF line stays clean for unidentified traffic."""
    h = _FunnelHarness()
    target = logging.getLogger("headroom.proxy")
    captured: list[logging.LogRecord] = []

    class _H(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _H(level=logging.INFO)
    target.addHandler(handler)
    prior_level = target.level
    target.setLevel(logging.INFO)
    try:
        await h._record_request_outcome(_outcome(client="codex"))
    finally:
        target.removeHandler(handler)
        target.setLevel(prior_level)

    perf_lines = [r.getMessage() for r in captured if " PERF " in r.getMessage()]
    assert len(perf_lines) == 1
    assert "client=codex" in perf_lines[0]


@pytest.mark.asyncio
async def test_funnel_omits_client_from_perf_log_when_unidentified() -> None:
    """When ``client`` is None the PERF line must NOT include a
    bogus ``client=`` token — that would mislead the parser into
    bucketing unidentified traffic as the empty string."""
    h = _FunnelHarness()
    target = logging.getLogger("headroom.proxy")
    captured: list[logging.LogRecord] = []

    class _H(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _H(level=logging.INFO)
    target.addHandler(handler)
    prior_level = target.level
    target.setLevel(logging.INFO)
    try:
        await h._record_request_outcome(_outcome(client=None))
    finally:
        target.removeHandler(handler)
        target.setLevel(prior_level)

    perf_lines = [r.getMessage() for r in captured if " PERF " in r.getMessage()]
    assert len(perf_lines) == 1
    assert "client=" not in perf_lines[0]


@pytest.mark.asyncio
async def test_funnel_stamps_client_into_request_log_tags() -> None:
    """Dashboards already filter on RequestLog.tags. Copying ``client``
    into tags gives per-harness slicing for free with no new column."""
    h = _FunnelHarness()
    await h._record_request_outcome(_outcome(client="aider"))
    assert len(h.logger.logs) == 1
    assert h.logger.logs[0].tags.get("client") == "aider"


# ── from_stream classmethod (streaming-finalizer construction shape) ──
#
# Three streaming finalizers (``_finalize_stream_response``,
# ``_stream_response_bedrock``, ``_stream_openai_via_backend``) each used
# to construct ``RequestOutcome(...)`` inline with the same body- and
# config-derived fields — ``attempted_input_tokens``, ``num_messages``,
# ``request_messages``, ``turn_id``, tuple-conversion of
# ``transforms_applied``, ``tags`` normalization. One site (Bedrock)
# computed ``turn_id``; the other two silently dropped it — a real bug
# the helper fixes by computing it uniformly. ``from_stream`` is the
# canonical construction point so the three finalizers cannot drift
# apart on derivation logic again.


def _stream_kwargs(**overrides: Any) -> dict[str, Any]:
    """Minimal kwargs for ``RequestOutcome.from_stream``; override per test."""
    base: dict[str, Any] = {
        "body": {"messages": [{"role": "user", "content": "hi"}]},
        "provider": "anthropic",
        "model": "claude-sonnet-4",
        "request_id": "req-1",
        "original_tokens": 1000,
        "optimized_tokens": 300,
        "output_tokens": 50,
        "tokens_saved": 700,
        "transforms_applied": ["smart_crusher"],
        "total_latency_ms": 1234.5,
        "overhead_ms": 12.3,
        "tags": {"a": "b"},
        "client": "codex",
        "log_full_messages": False,
    }
    base.update(overrides)
    return base


def test_from_stream_returns_request_outcome() -> None:
    o = RequestOutcome.from_stream(**_stream_kwargs())
    assert isinstance(o, RequestOutcome)


def test_from_stream_derives_attempted_input_tokens_from_optimized_plus_saved() -> None:
    """One of the six derivations the three finalizers each computed
    inline. Centralising it makes the dashboard's active-savings
    denominator structurally consistent across providers (#454/#455)."""
    o = RequestOutcome.from_stream(**_stream_kwargs(optimized_tokens=300, tokens_saved=700))
    assert o.attempted_input_tokens == 1000


def test_from_stream_counts_messages_from_body() -> None:
    """``num_messages`` powers PERF ``msgs=N``. Computing it from the
    body in one place prevents the historical drift where some sites
    used ``original_messages`` and others used ``body["messages"]``."""
    body = {"messages": [{"role": "user", "content": "1"}, {"role": "user", "content": "2"}]}
    assert RequestOutcome.from_stream(**_stream_kwargs(body=body)).num_messages == 2


def test_from_stream_handles_missing_messages_key() -> None:
    """Empty body — e.g. a probe request — must yield num_messages=0,
    not raise KeyError. All three pre-refactor sites used
    ``len(body.get("messages", []))`` so the contract is already
    "default to 0"."""
    assert RequestOutcome.from_stream(**_stream_kwargs(body={})).num_messages == 0


def test_from_stream_always_computes_turn_id() -> None:
    """The bug the helper is fixing: pre-refactor, only the Bedrock
    finalizer called ``compute_turn_id``. Sites 1 and 3 silently dropped
    it, breaking the dashboard's multi-turn-session grouping for every
    Anthropic-SSE and OpenAI-via-backend request. The helper computes
    it uniformly."""
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "system": "you are helpful",
    }
    o = RequestOutcome.from_stream(**_stream_kwargs(body=body, model="claude-sonnet-4"))
    assert o.turn_id is not None
    assert isinstance(o.turn_id, str)
    # Stable: same body+model produces the same turn_id.
    o2 = RequestOutcome.from_stream(**_stream_kwargs(body=body, model="claude-sonnet-4"))
    assert o.turn_id == o2.turn_id


def test_from_stream_converts_transforms_to_tuple() -> None:
    """``transforms_applied`` is typed as ``tuple[str, ...]`` on the
    dataclass (frozen → must be hashable/immutable). Callers pass lists.
    The helper does the conversion so no caller has to remember."""
    o = RequestOutcome.from_stream(**_stream_kwargs(transforms_applied=["a", "b"]))
    assert o.transforms_applied == ("a", "b")
    assert isinstance(o.transforms_applied, tuple)


def test_from_stream_normalises_none_tags_to_empty_dict() -> None:
    """``tags=None`` is the common case (no routing tags); the dataclass
    contract is ``dict[str, str]``. Pre-refactor each site wrote
    ``tags or {}``; the helper does it once."""
    assert RequestOutcome.from_stream(**_stream_kwargs(tags=None)).tags == {}


def test_from_stream_omits_request_messages_when_log_full_messages_disabled() -> None:
    """``log_full_messages=False`` is the default; message bodies are
    sensitive (tool outputs, secrets) and must not land in the request
    log unless explicitly enabled."""
    body = {"messages": [{"role": "user", "content": "secret"}]}
    o = RequestOutcome.from_stream(**_stream_kwargs(body=body, log_full_messages=False))
    assert o.request_messages is None


def test_from_stream_includes_request_messages_when_log_full_messages_enabled() -> None:
    """Same path, opt-in for full-message logging — used by
    /transformations/feed when the operator enables it."""
    body = {"messages": [{"role": "user", "content": "hi"}]}
    o = RequestOutcome.from_stream(**_stream_kwargs(body=body, log_full_messages=True))
    assert o.request_messages == body["messages"]


def test_from_stream_threads_provider_specific_cache_fields() -> None:
    """Anthropic populates all five cache fields; OpenAI-via-backend
    sets ``cache_inferred=True``; Gemini populates read only. The
    helper must pass each through without forcing every caller to
    pass every field."""
    o = RequestOutcome.from_stream(
        **_stream_kwargs(),
        cache_read_tokens=100,
        cache_write_tokens=200,
        cache_write_5m_tokens=150,
        cache_write_1h_tokens=50,
        uncached_input_tokens=10,
    )
    assert o.cache_read_tokens == 100
    assert o.cache_write_tokens == 200
    assert o.cache_write_5m_tokens == 150
    assert o.cache_write_1h_tokens == 50
    assert o.uncached_input_tokens == 10
    assert o.cache_inferred is False  # default — only set True by OpenAI sites


def test_from_stream_threads_waste_signals_for_openai_via_backend_site() -> None:
    """Only the OpenAI-via-backend finalizer populates ``waste_signals``;
    the helper threads it through as an optional kwarg."""
    o = RequestOutcome.from_stream(
        **_stream_kwargs(),
        waste_signals={"skipped_units": 3, "applied_units": 7},
    )
    assert o.waste_signals == {"skipped_units": 3, "applied_units": 7}
