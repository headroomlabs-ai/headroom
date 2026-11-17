"""Self-check for the beacon session aggregator.

This was ``headroom.telemetry.session.demo()``, run by hand with
``python -m headroom.telemetry.session``. It lives here so CI runs it. It covers
OTLP value encoding, the rate arithmetic, strategy and stack staging, the MCP
record path, flush overrides, the model allowlist and the config snapshot.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from headroom.telemetry import session as S
from headroom.telemetry.session import (
    _HIST_EDGES,
    _TURN_KINDS,
    MAX_STRATEGIES,
    SessionAggregator,
    _any_value,
    _config_snapshot,
    _McpCompression,
    _pct,
    _public_model,
    _safe_slug,
    _staged_shapes,
    _staged_stacks,
    _staged_strategies,
    _staged_tool_shapes,
    _tool_shape_key,
    record_compression,
    record_mcp_compression,
    record_stack,
    resource_attributes,
)

_STAGED = (_staged_strategies, _staged_stacks, _staged_shapes, _staged_tool_shapes)


@pytest.fixture
def isolated_session(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Keep the self-check off the real home directory and module state.

    resource_attributes() mints and persists an install id under the config
    dir, so point HOME and the Headroom dirs at tmp_path and reset the memoised
    id. The staging dicts are module globals: clear them before and after, so a
    failed assertion cannot leak staged rows into later beacon tests.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HEADROOM_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setattr(S, "_install_id", None)
    monkeypatch.setattr(S, "_aggregator", S._aggregator)
    # tests/conftest.py turns the beacon off for every test. The staging helpers
    # below are no-ops while it is off, so switch it back on for this test.
    monkeypatch.setenv("HEADROOM_BEACON", "on")
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
    for staged in _STAGED:
        staged.clear()
    yield tmp_path
    for staged in _STAGED:
        staged.clear()


def test_session_selfcheck(monkeypatch: pytest.MonkeyPatch, isolated_session) -> None:
    class FakeOutcome:
        provider = "anthropic"
        model = "claude-3-5-sonnet-20241022"
        optimized_tokens = 100
        output_tokens = 20
        tokens_saved = 50
        cache_read_tokens = 10
        overhead_ms = 1.5
        status_code = 200
        # Widened so subclasses below can override with a different arity —
        # RequestOutcome declares tuple[str, ...] too.
        transforms_applied: tuple[str, ...] = ("crush", "dedupe")

    # bool must not be encoded as int (bool subclasses int).
    assert _any_value(True) == {"boolValue": True}
    assert _any_value(1) == {"intValue": "1"}
    assert _any_value(2.5) == {"doubleValue": 2.5}
    assert _any_value({"a": [1, 2]})["kvlistValue"]["values"][0]["key"] == "a"
    assert _any_value(None) == {}, "None must not serialise as the text 'None'"

    # Ratios: the three the product is judged on, plus a zero-denominator guard.
    assert _pct(25, 100) == 25.0
    assert _pct(1, 3) == 33.33
    assert _pct(5, 0) == 0.0

    class Rich(FakeOutcome):
        original_tokens = 1000
        attempted_input_tokens = 400  # 40% eligible
        tokens_saved = 300  # 30% overall, 75% yield on eligible
        cache_read_tokens = 500  # 50% cache read
        cache_write_tokens = 100
        uncached_input_tokens = 400
        total_latency_ms = 1000.0
        overhead_ms = 50.0  # 5% overhead
        from_response_cache = True
        tags = {"tool_search_deferred_tokens": "800", "turn_hook_tools_saved_tokens": 200}

    rates_out: list[dict[str, Any]] = []
    ra = SessionAggregator(rates_out.append)
    ra.record(Rich(), now=100.0)
    ra.flush_all()
    r = rates_out[0]
    assert r["rates"]["saved_pct"] == 30.0, r["rates"]
    assert r["rates"]["eligible_pct"] == 40.0, r["rates"]
    assert r["rates"]["yield_pct"] == 75.0, r["rates"]
    assert r["rates"]["cache_read_pct"] == 50.0, r["rates"]
    assert r["rates"]["overhead_pct"] == 5.0, r["rates"]
    # Tool savings are invisible in `saved` by design; they must not be lost.
    assert r["tokens"]["tool_saved"] == 1000, r["tokens"]
    # ...and the all-layers rates are the ones that do count them: 1300 saved
    # of 2000 sent, 1300 of 1400 attempted. Both denominators grow too.
    assert r["rates"]["all_layers_saved_pct"] == 65.0, r["rates"]
    assert r["rates"]["all_layers_yield_pct"] == 92.86, r["rates"]
    assert r["compression"]["response_cache_hits"] == 1
    assert r["tokens"]["cache_write"] == 100 and r["tokens"]["uncached"] == 400

    emitted: list[dict[str, Any]] = []
    agg = SessionAggregator(emitted.append, idle_s=10.0)

    agg.record(FakeOutcome(), now=1000.0)
    agg.record(FakeOutcome(), now=1002.0)
    assert emitted == [], "a live session must not emit"

    # Quiet past the timeout, then activity -> the old burst closes.
    agg.record(FakeOutcome(), now=1100.0)
    assert len(emitted) == 1, emitted
    event = emitted[0]
    assert event["session"]["turns"] == 2
    assert event["session"]["ended"] == "idle"
    assert event["session"]["duration_s"] == 2
    assert event["tokens"]["saved"] == 100
    assert event["tokens"]["input"] == 200
    assert event["compression"]["transforms"] == {"crush": 2, "dedupe": 2}
    assert event["providers"] == ["anthropic"]
    assert event["failures"] == 0
    assert event["failure_statuses"] == {}

    # The new burst is a distinct session, not a continuation.
    agg.flush_all()
    assert len(emitted) == 2, emitted
    assert emitted[1]["session"]["turns"] == 1

    # --- per-strategy compression -----------------------------------------
    # Compression runs on the executor thread before its request's outcome
    # arrives, so events are staged and drained by the next outcome. That is
    # what keeps the first turn's numbers while letting only an outcome open a
    # session. `_staged_*` is module state, so clear it between cases.
    _staged_strategies.clear()
    _staged_stacks.clear()

    strat: list[dict[str, Any]] = []
    sa = SessionAggregator(strat.append, idle_s=10.0)
    record_compression("smart_crusher", 1000, 400)
    record_compression("smart_crusher", 500, 300)
    record_compression("code_aware", 800, 800)
    assert sa._current is None, "a compression event must not open a session"
    sa.record(FakeOutcome(), now=2000.0)
    sa.flush_all()
    by = {row["strategy"]: row for row in strat[-1]["compression"]["by_strategy"]}
    assert by["smart_crusher"] == {
        "strategy": "smart_crusher",
        "n": 2,
        "tokens_in": 1500,
        "tokens_out": 700,
    }, by
    # A strategy that ran and saved nothing must still appear: "ran 800 tokens
    # through and removed none" is the finding, and dropping it would make
    # every strategy look effective.
    assert by["code_aware"]["tokens_in"] == by["code_aware"]["tokens_out"] == 800, by
    assert strat[-1]["session"]["turns"] == 1, "compression events are not turns"
    # A list of records, not an object keyed by strategy: the type must not
    # change as strategies are added. See the note in payload().
    assert isinstance(strat[-1]["compression"]["by_strategy"], list)
    assert [r["strategy"] for r in strat[-1]["compression"]["by_strategy"]] == sorted(
        r["strategy"] for r in strat[-1]["compression"]["by_strategy"]
    ), "sorted so heartbeats are byte-comparable"

    # Draining is exhaustive: a second session must not re-count the first
    # session's events.
    assert not _staged_strategies, "record() drains everything it staged"
    again: list[dict[str, Any]] = []
    sb = SessionAggregator(again.append, idle_s=10.0)
    sb.record(FakeOutcome(), now=3000.0)
    sb.flush_all()
    assert again[-1]["compression"]["by_strategy"] == [], again[-1]

    # An abandoned request — compression ran, the outcome never arrived — must
    # not invent a session. Before staging, this emitted a phantom turns=0 row
    # with all-zero tokens that inflated fleet session and install counts.
    ghost: list[dict[str, Any]] = []
    sc = SessionAggregator(ghost.append, idle_s=10.0)
    record_compression("smart_crusher", 900, 100)
    sc.flush_all()
    assert ghost == [], "no outcome, no session"
    _staged_strategies.clear()

    # Over the cardinality cap, extra strategies are dropped rather than
    # allowed to grow the payload without bound.
    cap: list[dict[str, Any]] = []
    cc = SessionAggregator(cap.append, idle_s=10.0)
    for i in range(MAX_STRATEGIES + 5):
        record_compression(f"s{i}", 100, 50)
    cc.record(FakeOutcome(), now=4000.0)
    cc.flush_all()
    assert len(cap[-1]["compression"]["by_strategy"]) == MAX_STRATEGIES, cap[-1]
    _staged_strategies.clear()

    # Strategy names are slugged, never passed through: the observer protocol
    # takes a free string and this is the only chokepoint before the wire.
    assert _safe_slug("smart_crusher") == "smart_crusher"
    assert _safe_slug("../../etc/passwd") == "other"

    # --- stack detection ---------------------------------------------------
    # Environment-only detection answers "proxy" for every install that points
    # an agent at a persistent proxy instead of using `headroom wrap` — i.e.
    # most of the fleet. The per-request slugs are the only real signal.
    _staged_stacks.clear()
    for _ in range(9):
        record_stack("wrap_claude")
    record_stack("wrap_cursor")
    assert resource_attributes()["headroom.stack"] == "wrap_claude", "dominant stack wins"
    _staged_stacks.clear()
    for _ in range(5):
        record_stack("wrap_claude")
    for _ in range(5):
        record_stack("wrap_cursor")
    assert resource_attributes()["headroom.stack"] == "mixed", "no dominant stack"
    _staged_stacks.clear()
    record_stack("../../etc/passwd")
    assert not _staged_stacks, "junk slugs never reach the wire"
    assert resource_attributes()["headroom.stack"] == "proxy", "falls back with no signal"

    assert emitted[1]["session"]["id"] != emitted[0]["session"]["id"]
    assert emitted[1]["session"]["ended"] == "shutdown"

    # Nothing derived from prompt content or the model id reaches the wire.
    wire = json.dumps(emitted)
    assert "sonnet" not in wire and "claude" not in wire, wire

    # Model ids reach the wire only when a public registry knows them. A
    # fine-tune id carries an org name and must never survive.
    assert _public_model("ft:gpt-4o:acme-corp:internal-bot:abc123") is None
    assert _public_model("acme-internal-llama") is None
    assert _public_model("") is None
    if _public_model("gpt-4o") is None:
        print("  (litellm unavailable — model field degrades to absent)")
    else:
        assert _public_model("gpt-4o") == "gpt-4o"

    # Reason slugs are validated, not trusted.
    assert _safe_slug("bypass_header") == "bypass_header"
    assert _safe_slug("/Users/me/secret/path.py") == "other"
    assert _safe_slug(None) == "other"

    # Low compression must be explainable: a passthrough session and a
    # ran-but-found-nothing session must not look the same.
    class Bypassed(FakeOutcome):
        original_tokens = 5000
        attempted_input_tokens = 0
        tokens_saved = 0
        transforms_applied = ()
        tags = {"passthrough_reason": "bypass_header"}

    class Barren(FakeOutcome):
        original_tokens = 5000
        attempted_input_tokens = 4000
        tokens_saved = 12
        tags: dict[str, str] = {}

    diag: list[dict[str, Any]] = []
    agg_bypassed = SessionAggregator(diag.append)
    agg_bypassed.record(Bypassed(), now=5000.0)
    agg_bypassed.flush_all()
    agg_barren = SessionAggregator(diag.append)
    agg_barren.record(Barren(), now=6000.0)
    agg_barren.flush_all()

    bypassed, barren = diag[0], diag[1]
    assert bypassed["tokens"]["attempted"] == 0
    assert bypassed["skips"] == {"passthrough:bypass_header": 1}
    assert bypassed["compression"]["passthrough_turns"] == 1
    assert barren["tokens"]["attempted"] == 4000
    assert barren["skips"] == {}
    # Both saved ~nothing, but the reason is now distinguishable.
    assert bypassed["tokens"]["saved"] == 0 and barren["tokens"]["saved"] == 12

    # A live session heartbeats under ONE id with cumulative totals, so the
    # highest-seq row is the whole session and dedupe is last-write-wins.
    beats: list[dict[str, Any]] = []
    hb = SessionAggregator(beats.append, idle_s=900.0, flush_s=100.0)
    hb.record(FakeOutcome(), now=7000.0)
    hb.record(FakeOutcome(), now=7050.0)
    assert beats == [], "must not emit before the flush interval"

    hb.record(FakeOutcome(), now=7100.0)  # first heartbeat
    hb.record(FakeOutcome(), now=7150.0)
    hb.record(FakeOutcome(), now=7250.0)  # second heartbeat
    hb.flush_all()  # final

    assert len(beats) == 3, beats
    ids = {b["session"]["id"] for b in beats}
    assert len(ids) == 1, f"one session must not fragment into {len(ids)} ids"
    assert [b["session"]["seq"] for b in beats] == [0, 1, 2]
    assert [b["session"]["final"] for b in beats] == [False, False, True]
    assert [b["session"]["ended"] for b in beats] == ["active", "active", "shutdown"]

    # Cumulative, not deltas: turns and tokens only ever climb, and the last
    # row alone reconstructs the session.
    assert [b["session"]["turns"] for b in beats] == [3, 5, 5]
    assert [b["tokens"]["saved"] for b in beats] == [150, 250, 250]

    # Dedupe by (install, session id), keep max seq -> exactly one row.
    latest: dict[str, dict[str, Any]] = {}
    for b in beats:
        key = b["session"]["id"]
        if key not in latest or b["session"]["seq"] > latest[key]["session"]["seq"]:
            latest[key] = b
    assert len(latest) == 1
    only = next(iter(latest.values()))
    assert only["session"]["turns"] == 5 and only["tokens"]["saved"] == 250

    # Losing a heartbeat costs nothing — the survivor still restates everything.
    survivors = [beats[0], beats[2]]
    assert max(s["session"]["seq"] for s in survivors) == 2
    assert survivors[-1]["session"]["turns"] == 5

    # Going quiet starts a genuinely new session, not a continuation.
    hb.record(FakeOutcome(), now=90000.0)
    hb.flush_all()
    assert beats[-1]["session"]["id"] != beats[0]["session"]["id"]
    assert beats[-1]["session"]["turns"] == 1

    # A transform label carrying a stratum suffix must be reduced to its prefix
    # — otherwise output_shaper smuggles the model tier and a size bucket out.
    class Shaped(FakeOutcome):
        transforms_applied = ("output_shaper:stratum:sonnet|tool_result|8k", "crush")

    shaped: list[dict[str, Any]] = []
    agg_s = SessionAggregator(shaped.append)
    agg_s.record(Shaped(), now=1500.0)
    agg_s.flush_all()
    assert shaped[0]["compression"]["transforms"] == {"output_shaper": 1, "crush": 1}
    assert "sonnet" not in json.dumps(shaped[0]), shaped[0]

    # A 5xx counts as a failure without poisoning the token stats.
    class Failed(FakeOutcome):
        status_code = 529

    class Broke(FakeOutcome):
        status_code = 500

    agg2 = SessionAggregator(emitted.append)
    agg2.record(Failed(), now=2000.0)
    agg2.record(Failed(), now=2001.0)
    agg2.record(Broke(), now=2002.0)
    agg2.flush_all()
    assert emitted[-1]["failures"] == 3
    # Provider load-shedding and our own 500s have to be separable, or the
    # count says "0.7% of turns failed" and nothing about whose fault it is.
    assert emitted[-1]["failure_statuses"] == {"529": 2, "500": 1}

    # Flushing an empty aggregator is a no-op, not a null event.
    before = len(emitted)
    SessionAggregator(emitted.append).flush_all()
    assert len(emitted) == before

    # MCP turns must be distinguishable from proxy turns in the same session.
    # Blending them would corrupt eligible_pct: everything handed to the tool is
    # eligible by construction, so MCP turns always read 100% and would drag the
    # proxy's real eligibility ceiling upward.
    mixed: list[dict[str, Any]] = []
    mx = SessionAggregator(mixed.append)
    mx.record(FakeOutcome(), now=9000.0)
    mx.record(
        _McpCompression(
            original_tokens=1000,
            attempted_input_tokens=1000,
            optimized_tokens=300,
            tokens_saved=700,
        ),
        source="mcp",
        now=9001.0,
    )
    mx.flush_all()
    mixed_ev = mixed[0]
    assert mixed_ev["sources"] == {"proxy": 1, "mcp": 1}, mixed_ev["sources"]
    assert mixed_ev["session"]["turns"] == 2
    # An MCP-only shim contributes tokens but no provider and no latency.
    assert mixed_ev["tokens"]["saved"] == 700 + 50
    assert mixed_ev["providers"] == ["anthropic"]  # only the proxy turn had one

    mcp_only: list[dict[str, Any]] = []
    mo = SessionAggregator(mcp_only.append)
    mo.record(
        _McpCompression(
            original_tokens=800,
            attempted_input_tokens=800,
            optimized_tokens=200,
            tokens_saved=600,
        ),
        source="mcp",
        now=9100.0,
    )
    mo.flush_all()
    only_ev = mcp_only[0]
    assert only_ev["sources"] == {"mcp": 1}
    assert only_ev["rates"]["eligible_pct"] == 100.0
    assert only_ev["rates"]["yield_pct"] == 75.0
    assert only_ev["rates"]["saved_pct"] == 75.0
    assert only_ev["providers"] == [] and only_ev["models"] == []
    # No upstream call means no latency, and the ratio must not divide by zero.
    assert only_ev["rates"]["overhead_pct"] == 0.0

    # record_mcp_compression: a real call records, a degenerate one does not.
    # Beacon forced on for this block so the assertions cannot pass merely
    # because the ambient environment has it disabled.
    with monkeypatch.context() as mp:
        mp.setenv("HEADROOM_BEACON", "on")
        # DO_NOT_TRACK and offline mode outrank an explicit opt-in, so they have to
        # be cleared here or these assertions test the wrong thing.
        mp.delenv("DO_NOT_TRACK", raising=False)
        mp.delenv("HEADROOM_OFFLINE", raising=False)
        mp.setattr(S, "_aggregator", SessionAggregator(lambda _p: None))
        record_mcp_compression(original_tokens=0, compressed_tokens=0)
        assert S._aggregator._current is None, "zero-token call recorded"

        record_mcp_compression(original_tokens=500, compressed_tokens=100)
        live = S._aggregator._current
        assert live is not None, "valid MCP call did not record"
        assert live.sources == {"mcp": 1}
        assert live.tokens_saved == 400

        # A compression that grew the content must not report negative savings.
        record_mcp_compression(original_tokens=100, compressed_tokens=250)
        assert S._aggregator._current.tokens_saved == 400

        # DO_NOT_TRACK outranks an explicit HEADROOM_BEACON=on.
        mp.setenv("DO_NOT_TRACK", "1")
        mp.setattr(S, "_aggregator", SessionAggregator(lambda _p: None))
        record_mcp_compression(original_tokens=500, compressed_tokens=100)
        assert S._aggregator._current is None, "DO_NOT_TRACK was ignored"

    # flush_all must honour an emit override. The atexit path depends on this:
    # the normal sink defers to a daemon thread, and daemon threads are killed
    # before they finish once the interpreter is shutting down, so a thread
    # started there never posts. Without the override, every session shorter
    # than FLUSH_INTERVAL_S would report nothing at all.
    default_sink: list[dict[str, Any]] = []
    override_sink: list[dict[str, Any]] = []
    ex = SessionAggregator(default_sink.append)
    ex.record(FakeOutcome(), now=8000.0)
    ex.flush_all(emit=override_sink.append)
    assert override_sink and not default_sink, "flush_all ignored the emit override"
    assert override_sink[0]["session"]["final"] is True

    # An emit that blows up must not propagate to the caller.
    def boom(_payload: dict[str, Any]) -> None:
        raise RuntimeError("collector down")

    agg3 = SessionAggregator(boom, idle_s=1.0)
    agg3.record(FakeOutcome(), now=3000.0)
    agg3.record(FakeOutcome(), now=3100.0)
    agg3.flush_all()

    # A malformed outcome must not raise either.
    SessionAggregator(emitted.append).record(object(), now=4000.0)

    # --- schema v2 ---------------------------------------------------------
    # The full v2 surface is covered by tests/test_beacon_schema_v2.py. What
    # belongs here is the part that has to hold for the module on its own: the
    # additions are additive, and the two things they read that nobody had read
    # before — the environment, and the output_shaper stratum label — cannot
    # carry anything out with them.
    _staged_shapes.clear()
    _staged_tool_shapes.clear()

    class V2(FakeOutcome):
        original_tokens = 1000
        attempted_input_tokens = 400
        tokens_saved = 300
        ttfb_ms = 300.0
        num_messages = 30
        # Hyphenated, as `classify_client` really returns it.
        client = "claude-code"
        cache_write_5m_tokens = 80
        cache_write_1h_tokens = 20
        waste_signals = {"reread": 900, "reread_compressed": 400, "json_bloat": 50}
        # Carries a model family, a turn kind and a size bucket. Only the last
        # two may leave, and neither the label nor the family may.
        transforms_applied = ("output_shaper:stratum:sonnet|tool_result|l|tools", "crush")

    class V2Limited(V2):
        status_code = 429

    v2_out: list[dict[str, Any]] = []
    v2 = SessionAggregator(v2_out.append)
    v2.record(V2(), now=10_000.0)
    v2.record(V2Limited(), now=10_045.0)
    v2.record(V2(), now=10_090.0)
    v2.flush_all()
    row = v2_out[-1]

    # v1 is untouched by all of it. A 4xx in particular must not reach the
    # failure counters, which have been 5xx-only in every row of the corpus.
    assert row["failures"] == 0 and row["failure_statuses"] == {}, row
    assert row["errors"] == {"count": 1, "by_status": {"429": 1}}, row["errors"]
    assert row["compression"]["transforms"] == {"output_shaper": 3, "crush": 3}

    # The regret label: saved_pct alone cannot say whether compression cost the
    # agent anything, and this is the counter-pressure on it.
    assert row["quality"]["reread_compressed_tokens"] == 1200, row["quality"]
    assert row["quality"]["waste"] == [{"kind": "json_bloat", "tokens": 150}], (
        "waste kinds are allowlisted"
    )

    # Distributions ship their bucket edges, so a row is readable without
    # knowing which client version wrote it.
    assert row["hist"]["turn_tokens"]["edges"] == list(_HIST_EDGES["turn_tokens"])
    assert sum(row["hist"]["turn_tokens"]["counts"]) == 3
    assert len(row["hist"]["saved_pct"]["counts"]) == len(_HIST_EDGES["saved_pct"]) + 1

    # Cumulative like every v1 counter, so max(seq) still reconstructs the
    # session and a dropped heartbeat still costs nothing.
    assert sum(row["trajectory"]["turns"]) == row["session"]["turns"]
    assert row["trajectory"]["kinds"] == "s1e1s1", row["trajectory"]
    assert set(row["trajectory"]["kinds"]) - set("0123456789") <= _TURN_KINDS

    # The first turn has no predecessor, so the cache curve counts turns - 1.
    assert sum(row["cache"]["hits"]) + sum(row["cache"]["misses"]) == 2

    # Strata reach the wire as validated enums; the model family does not.
    # Hyphenated client ids must survive; _safe_slug alone would drop them.
    assert row["clients"] == [{"client": "claude_code", "n": 3}], row["clients"]
    # Lists of records, never objects keyed by a value that came from data —
    # an object's COLUMN TYPE in DuckDB is a function of which keys showed up.
    assert {"dim": "arm", "value": "treatment", "n": 3} in row["strata"], row["strata"]
    assert {"dim": "input_bucket", "value": "l", "n": 3} in row["strata"], row["strata"]
    assert isinstance(row["config"], list) and isinstance(row["quality"]["waste"], list)
    assert "sonnet" not in json.dumps(row), "the stratum label leaked a model family"

    # Structural descriptor only, never TOIN's structure_hash.
    assert _tool_shape_key(object()) is None
    assert (
        _tool_shape_key(
            type(
                "Sig",
                (),
                {"field_count": 17, "max_depth": 3, "has_arrays": True, "has_id_like_field": True},
            )()
        )
        == "f16d3_ai"
    )

    # The environment is read through an allowlist, and every value that gets
    # past it is slugged — so a variable holding a path or a URL reports
    # "other" rather than its contents.
    with monkeypatch.context() as mp:
        mp.setenv("HEADROOM_MODE", "/Users/someone/private")
        mp.setenv("HEADROOM_KOMPRESS_ENDPOINT", "https://secret.internal/v1")
        snapshot = _config_snapshot()
        assert snapshot.get("mode") == "other", snapshot
        assert "kompress_endpoint" not in snapshot, snapshot

    _staged_shapes.clear()
    _staged_tool_shapes.clear()
