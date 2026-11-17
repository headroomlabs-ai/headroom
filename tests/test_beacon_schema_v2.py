"""Beacon schema v2: the new signals, and proof the old ones did not move.

The corpus has years of rows whose meaning depends on the v1 payload staying
exactly what it has always been. Schema v2 is additive, and this file is where
that claim is mechanical rather than asserted: `test_v1_*` locks the shape and
the arithmetic of every pre-existing field, `test_v2_*` covers what was added,
and `test_allowlist_covers_every_emitted_key` enforces the deploy-ordering rule
that a key the worker does not allow is a key that is silently discarded.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any

import pytest

from headroom.telemetry import session as S

# ---------------------------------------------------------------------------
# The v1 payload, spelled out. Changing anything in this literal is a schema
# break for every consumer of the existing corpus, and should be very hard to
# do by accident.
# ---------------------------------------------------------------------------

V1_TOP = {
    "schema_version",
    "session",
    "tokens",
    "rates",
    "compression",
    "skips",
    "sources",
    "providers",
    "models",
    "failures",
    "failure_statuses",
}

V1_NESTED = {
    "session": {"id", "seq", "duration_s", "turns", "ended", "final"},
    "tokens": {
        "original",
        "attempted",
        "input",
        "output",
        "saved",
        "tool_saved",
        "cache_read",
        "cache_write",
        "uncached",
    },
    "rates": {
        "saved_pct",
        "eligible_pct",
        "yield_pct",
        "all_layers_saved_pct",
        "all_layers_yield_pct",
        "cache_read_pct",
        "overhead_pct",
    },
    "compression": {
        "transforms",
        "by_strategy",
        "overhead_ms_total",
        "latency_ms_total",
        "overhead_ms_per_turn",
        "latency_ms_per_turn",
        "passthrough_turns",
        "response_cache_hits",
    },
}

V2_TOP = {
    "quality",
    "hist",
    "shapes",
    "cache",
    "trajectory",
    "clients",
    "errors",
    "strata",
    "config",
}


class Outcome:
    """A fully-populated RequestOutcome stand-in. `_fold` is duck-typed."""

    provider = "anthropic"
    model = "gpt-4o"
    original_tokens = 1000
    attempted_input_tokens = 400
    optimized_tokens = 700
    provider_input_tokens = 700
    output_tokens = 20
    tokens_saved = 300
    cache_read_tokens = 500
    cache_write_tokens = 100
    cache_write_5m_tokens = 80
    cache_write_1h_tokens = 20
    uncached_input_tokens = 400
    cache_inferred = False
    from_response_cache = False
    status_code = 200
    total_latency_ms = 1000.0
    overhead_ms = 50.0
    ttfb_ms = 300.0
    num_messages = 30
    client = "claude-code"  # real ids are hyphenated; see test_client_ids_survive
    transforms_applied: tuple[str, ...] = ("crush",)
    tags: dict[str, Any] = {}
    waste_signals: dict[str, int] | None = None


def emit(
    *outcomes: Any, start: float = 1000.0, step: float = 45.0, source: str = "proxy"
) -> dict[str, Any]:
    """Fold a sequence of outcomes into one session and return its final row."""
    rows: list[dict[str, Any]] = []
    agg = S.SessionAggregator(rows.append)
    for index, outcome in enumerate(outcomes):
        agg.record(outcome, now=start + index * step, source=source)
    agg.flush_all()
    return rows[-1]


@pytest.fixture(autouse=True)
def _clean_staging():
    """Staging is module state; a leak across tests would double-count."""
    S._staged_strategies.clear()
    S._staged_stacks.clear()
    S._staged_shapes.clear()
    S._staged_tool_shapes.clear()
    yield
    S._staged_strategies.clear()
    S._staged_stacks.clear()
    S._staged_shapes.clear()
    S._staged_tool_shapes.clear()


@pytest.fixture
def beacon_on(monkeypatch):
    """conftest turns the beacon off for hermeticity; the record_* entry points
    short-circuit on that, so anything testing them has to turn it back on."""
    monkeypatch.setenv("HEADROOM_BEACON", "on")
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)


# ---------------------------------------------------------------------------
# v1: nothing moved
# ---------------------------------------------------------------------------


def test_v1_keys_all_present_and_unchanged():
    payload = emit(Outcome())
    assert V1_TOP <= set(payload), f"v1 key(s) dropped: {V1_TOP - set(payload)}"
    for section, expected in V1_NESTED.items():
        assert expected <= set(payload[section]), (
            f"{section} lost key(s): {expected - set(payload[section])}"
        )


def test_v1_arithmetic_is_byte_for_byte_what_it_was():
    """The five ratios the product is judged on, plus the counters they derive
    from. These values predate schema v2 and must survive it exactly."""
    payload = emit(Outcome())
    assert payload["tokens"] == {
        "original": 1000,
        "attempted": 400,
        "input": 700,
        "output": 20,
        "saved": 300,
        "tool_saved": 0,
        "cache_read": 500,
        "cache_write": 100,
        "uncached": 400,
    }
    assert payload["rates"]["saved_pct"] == 30.0
    assert payload["rates"]["eligible_pct"] == 40.0
    assert payload["rates"]["yield_pct"] == 75.0
    assert payload["rates"]["cache_read_pct"] == 50.0
    assert payload["rates"]["overhead_pct"] == 5.0
    assert payload["compression"]["transforms"] == {"crush": 1}
    assert payload["sources"] == {"proxy": 1}
    assert payload["providers"] == ["anthropic"]


def test_v1_failures_stay_5xx_only_when_4xx_is_present():
    """The single most dangerous change in v2: 4xx is now counted, and it must
    be counted somewhere `failures` cannot see. Widening `failures` would have
    silently redefined it — a session that rate-limited would start reading as
    a session that failed.
    """

    class RateLimited(Outcome):
        status_code = 429

    class Overloaded(Outcome):
        status_code = 529

    payload = emit(Outcome(), RateLimited(), RateLimited(), Overloaded())
    assert payload["failures"] == 1, "a 4xx leaked into the 5xx failure count"
    assert payload["failure_statuses"] == {"529": 1}
    assert payload["errors"] == {"count": 2, "by_status": {"429": 2}}


def test_v1_transforms_still_collapse_the_stratum_suffix():
    """v2 reads the stratum label too, but the v1 `transforms` map must still
    see only the prefix — the suffix carries a model family."""

    class Shaped(Outcome):
        transforms_applied = ("output_shaper:stratum:sonnet|tool_result|l|tools", "crush")

    payload = emit(Shaped())
    assert payload["compression"]["transforms"] == {"output_shaper": 1, "crush": 1}


def test_v1_cumulative_dedupe_still_holds_with_v2_fields():
    """The corpus dedupes on max(seq) per (install, session) and never sums
    across heartbeats. Every v2 field has to be cumulative for that to keep
    working."""
    rows: list[dict[str, Any]] = []
    agg = S.SessionAggregator(rows.append, idle_s=900.0, flush_s=100.0)
    for index in range(6):
        agg.record(Outcome(), now=7000.0 + index * 50)
    agg.flush_all()

    assert len({row["session"]["id"] for row in rows}) == 1
    assert [row["session"]["seq"] for row in rows] == list(range(len(rows)))
    for field in ("turns",):
        values = [row["session"][field] for row in rows]
        assert values == sorted(values), f"session.{field} is not monotonic"
    # Every v2 counter must climb or hold, never reset.
    for row_a, row_b in zip(rows, rows[1:]):
        assert row_b["quality"]["reread_tokens"] >= row_a["quality"]["reread_tokens"]
        assert sum(row_b["trajectory"]["turns"]) >= sum(row_a["trajectory"]["turns"])
        for name, hist in row_b["hist"].items():
            assert sum(hist["counts"]) >= sum(row_a["hist"][name]["counts"]), name
    # The last row alone reconstructs the session.
    assert rows[-1]["session"]["turns"] == 6
    assert sum(rows[-1]["trajectory"]["turns"]) == 6


def test_mcp_shim_still_folds_without_the_new_fields():
    """`_McpCompression` knows five fields. v2 reads a dozen more through the
    same duck-typed `get`, and must not require any of them."""
    payload = emit(
        S._McpCompression(
            original_tokens=800,
            attempted_input_tokens=800,
            optimized_tokens=200,
            tokens_saved=600,
        ),
        source="mcp",
    )
    assert payload["sources"] == {"mcp": 1}
    assert payload["rates"]["yield_pct"] == 75.0
    assert payload["trajectory"]["kinds"] == "s1"
    assert payload["clients"] == []


def test_a_malformed_outcome_still_cannot_raise():
    rows: list[dict[str, Any]] = []
    agg = S.SessionAggregator(rows.append)
    agg.record(object(), now=1.0)
    agg.flush_all()


# ---------------------------------------------------------------------------
# v2: the new signals
# ---------------------------------------------------------------------------


def test_quality_reports_the_regret_label():
    class Regretful(Outcome):
        waste_signals = {
            "reread": 900,
            "reread_compressed": 400,
            "json_bloat": 50,
            "base64": 7,
        }

    payload = emit(Regretful(), Regretful(), Outcome())
    assert payload["quality"]["reread_tokens"] == 1800
    assert payload["quality"]["reread_compressed_tokens"] == 800
    assert payload["quality"]["waste"] == [
        {"kind": "base64", "tokens": 14},
        {"kind": "json_bloat", "tokens": 100},
    ]
    assert payload["quality"]["waste_turns"] == 2, "the clean turn was counted"


def test_waste_keys_are_allowlisted_not_copied():
    class Sneaky(Outcome):
        waste_signals = {"json_bloat": 10, "some_future_key": 99, "/etc/passwd": 5}

    payload = emit(Sneaky())
    assert payload["quality"]["waste"] == [{"kind": "json_bloat", "tokens": 10}]


def test_histograms_ship_their_own_edges_and_count_defined_samples():
    class Small(Outcome):
        original_tokens = 500
        tokens_saved = 0
        ttfb_ms = 0.0  # 0 means unmeasured, not instant

    payload = emit(Outcome(), Outcome(), Small())
    turn_tokens = payload["hist"]["turn_tokens"]
    assert turn_tokens["edges"] == [1000, 4000, 16000, 64000, 128000, 256000]
    assert len(turn_tokens["counts"]) == len(turn_tokens["edges"]) + 1
    assert sum(turn_tokens["counts"]) == 3, "observed on every turn"
    assert turn_tokens["counts"][0] == 1, "the 500-token turn is under the first edge"
    assert sum(payload["hist"]["ttfb_ms"]["counts"]) == 2, "unmeasured ttfb excluded"
    # 30% saved -> the [20, 40) bucket, which is index 4.
    assert payload["hist"]["saved_pct"]["counts"][4] == 2


def test_cache_gap_curve_excludes_the_first_turn():
    class Cold(Outcome):
        cache_read_tokens = 0

    payload = emit(Outcome(), Outcome(), Cold(), step=45.0)
    hits = payload["cache"]["hits"]
    misses = payload["cache"]["misses"]
    assert sum(hits) + sum(misses) == 2, "first turn has no predecessor to measure"
    assert payload["cache"]["gap_edges_s"] == [30.0, 60.0, 120.0, 300.0, 600.0, 1800.0]
    # A 45s gap lands in the [30, 60) bucket.
    assert hits[1] == 1 and misses[1] == 1
    assert payload["cache"]["write_5m"] == 240
    assert payload["cache"]["write_1h"] == 60


def test_trajectory_buckets_are_log_spaced_and_kinds_are_run_length_encoded():
    class Bypassed(Outcome):
        attempted_input_tokens = 0
        tokens_saved = 0
        tags = {"passthrough_reason": "bypass_header"}

    payload = emit(*([Outcome()] * 4), Bypassed(), Bypassed(), *([Outcome()] * 3))
    assert payload["trajectory"]["turns"] == [1, 2, 4, 2, 0, 0, 0, 0, 0, 0]
    assert payload["trajectory"]["kinds"] == "s4p2s3"
    assert payload["trajectory"]["kinds_truncated"] is False
    assert payload["trajectory"]["msgs_max"] == 30
    # And the v1 passthrough counter is untouched by any of it.
    assert payload["compression"]["passthrough_turns"] == 2


def test_trajectory_kind_alphabet_is_closed():
    class Cached(Outcome):
        from_response_cache = True

    class Barren(Outcome):
        tokens_saved = 0

    class Ineligible(Outcome):
        attempted_input_tokens = 0
        tokens_saved = 0

    class Failed(Outcome):
        status_code = 500

    class Limited(Outcome):
        status_code = 429

    payload = emit(Cached(), Failed(), Limited(), Barren(), Outcome(), Ineligible())
    kinds = payload["trajectory"]["kinds"]
    assert kinds == "c1f1e1z1s1o1"
    assert set(re.findall(r"[a-z]", kinds)) <= S._TURN_KINDS


def test_trajectory_string_is_capped():
    outcomes = []
    for index in range(S.MAX_TRAJECTORY_RUNS * 3):
        cls = Outcome if index % 2 else type("Alt", (Outcome,), {"status_code": 500})
        outcomes.append(cls())
    payload = emit(*outcomes)
    assert payload["trajectory"]["kinds_truncated"] is True
    assert len(re.findall(r"[a-z]", payload["trajectory"]["kinds"])) == S.MAX_TRAJECTORY_RUNS


def test_clients_and_strata(beacon_on):
    class Shaped(Outcome):
        transforms_applied = ("output_shaper:stratum:sonnet|tool_result|l|tools",)

    class Control(Outcome):
        client = "claude-vscode"
        transforms_applied = ("output_shaper:control:opus|user|s|notools",)

    class Anonymous(Outcome):
        client = None

    payload = emit(Shaped(), Control(), Anonymous())
    assert payload["clients"] == [
        {"client": "claude_code", "n": 1},
        {"client": "claude_vscode", "n": 1},
    ], "unknown client not bucketed"
    strata = {(r["dim"], r["value"]): r["n"] for r in payload["strata"]}
    assert strata[("arm", "treatment")] == 1 and strata[("arm", "control")] == 1
    assert strata[("turn_kind", "tool_result")] == 1 and strata[("turn_kind", "user")] == 1
    assert strata[("input_bucket", "l")] == 1 and strata[("input_bucket", "s")] == 1
    assert strata[("tools", "tools")] == 1 and strata[("tools", "notools")] == 1
    # Model family is the one stratum component that must never ship.
    assert "sonnet" not in json.dumps(payload)
    assert "opus" not in json.dumps(payload)


def test_client_ids_survive_slugging():
    """Every harness `classify_client` can return has to round-trip, or the
    field reports "other" for the clients that matter. Asserted against the
    real vocabulary rather than a value invented here — the first version of
    this test used "claude_code", which is not an id the proxy ever emits, and
    so passed while `claude-code` was being discarded.
    """
    from headroom.proxy.auth_mode import CLIENT_UA_MAP

    known = sorted({value for _, value in CLIENT_UA_MAP})
    assert "claude-code" in known, "the vocabulary moved; this test needs updating"

    outcomes = [type(f"C{i}", (Outcome,), {"client": name})() for i, name in enumerate(known)]
    clients = {r["client"]: r["n"] for r in emit(*outcomes)["clients"]}
    assert "other" not in clients, (
        f"a real client id was discarded: {sorted(set(known) - set(clients))}"
    )
    assert len(clients) == len(known)
    assert clients["claude_code"] == 1


def test_client_slug_still_rejects_junk():
    payload = emit(type("Junk", (Outcome,), {"client": "../../etc/passwd"})())
    assert payload["clients"] == [{"client": "other", "n": 1}]


def test_content_shape_table(beacon_on):
    S.record_content_shape("search", "smart_crusher", 1000, 400)
    S.record_content_shape("search", "smart_crusher", 500, 300)
    S.record_content_shape("diff", "passthrough", 800, 800)
    payload = emit(Outcome())
    rows = {(r["content"], r["strategy"]): r for r in payload["shapes"]["by_content"]}
    assert rows[("search", "smart_crusher")] == {
        "content": "search",
        "strategy": "smart_crusher",
        "n": 2,
        "tokens_in": 1500,
        "tokens_out": 700,
    }
    assert rows[("diff", "passthrough")]["tokens_in"] == 800
    assert isinstance(payload["shapes"]["by_content"], list), "a list, never a map"


def test_content_shape_cannot_open_a_session(beacon_on):
    rows: list[dict[str, Any]] = []
    agg = S.SessionAggregator(rows.append)
    S.record_content_shape("search", "smart_crusher", 1000, 400)
    agg.flush_all()
    assert rows == [], "a compression event invented a session"


def test_content_shape_slugs_free_strings(beacon_on):
    S.record_content_shape("../../etc/passwd", "SELECT * FROM x", 100, 50)
    payload = emit(Outcome())
    assert payload["shapes"]["by_content"][0]["content"] == "other"
    assert payload["shapes"]["by_content"][0]["strategy"] == "other"


def test_tool_shape_is_a_descriptor_never_a_hash(beacon_on):
    class Signature:
        structure_hash = "deadbeefcafe0123456789ab"
        field_count = 17
        max_depth = 3
        has_arrays = True
        has_nested_objects = True
        has_id_like_field = True
        has_score_like_field = False
        has_timestamp_like_field = True
        has_status_like_field = False
        has_error_like_field = False
        has_message_like_field = False

    S.record_tool_shape(Signature(), 5000, 1200)
    payload = emit(Outcome())
    assert payload["shapes"]["by_tool"] == [
        {"shape": "f16d3_anit", "n": 1, "tokens_in": 5000, "tokens_out": 1200}
    ]
    assert "deadbeef" not in json.dumps(payload), "structure_hash reached the wire"


def test_tool_shape_rejects_a_non_signature(beacon_on):
    S.record_tool_shape(object(), 5000, 1200)
    S.record_tool_shape(None, 5000, 1200)
    payload = emit(Outcome())
    assert payload["shapes"]["by_tool"] == []


def test_shape_tables_are_capped(beacon_on):
    """The cap is a guard against a caller inventing keys, not a budget — see
    test_content_shape_cap_exceeds_its_vocabulary for the other side."""
    for index in range(S.MAX_SHAPE_ROWS + 10):
        S.record_content_shape(f"c{index}", "smart_crusher", 100, 50)
    payload = emit(Outcome())
    assert len(payload["shapes"]["by_content"]) == S.MAX_SHAPE_ROWS


def test_config_is_an_allowlist_and_every_value_is_slugged(monkeypatch):
    monkeypatch.setenv("HEADROOM_MODE", "optimize")
    monkeypatch.setenv("HEADROOM_KOMPRESS_ENDPOINT", "https://secret.acme.internal/v1")
    monkeypatch.setenv("HEADROOM_PROXY_TOKEN", "sk-live-abc123")
    monkeypatch.setenv("HEADROOM_SAVINGS_PROFILE", "/Users/someone/private/path")
    payload = emit(Outcome())
    config = {r["key"]: r["value"] for r in payload["config"]}
    assert config["mode"] == "optimize"
    assert config["savings_profile"] == "other", "a path shipped verbatim"
    assert "kompress_endpoint" not in config
    assert "proxy_token" not in config
    wire = json.dumps(payload)
    assert "acme" not in wire and "sk-live" not in wire and "/Users/" not in wire


def test_config_omits_unset_variables(monkeypatch):
    for name in S._CONFIG_ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HEADROOM_DEDUPE", "  ")
    assert emit(Outcome())["config"] == []


def test_a_v2_payload_failure_cannot_lose_the_v1_event():
    """The v2 sections claim they cannot cost a v1 number. Built inline in one
    dict literal they could: any raise aborts the literal and drops the whole
    event, `tokens.saved` included."""
    from unittest.mock import patch

    rows: list[dict[str, Any]] = []
    agg = S.SessionAggregator(rows.append)
    with patch.object(S, "_config_snapshot", side_effect=RuntimeError("boom")):
        agg.record(Outcome(), now=1000.0)
        agg.flush_all()
    assert rows, "a v2 failure dropped the entire session event"
    assert rows[0]["tokens"]["saved"] == 300, "v1 counters survived intact"
    assert "config" not in rows[0], "the failed v2 section is absent, not partial"


def test_a_v2_fold_failure_cannot_cost_a_v1_counter_or_a_heartbeat():
    from unittest.mock import patch

    rows: list[dict[str, Any]] = []
    agg = S.SessionAggregator(rows.append, idle_s=900.0, flush_s=100.0)
    with patch.object(S, "_fold_extra", side_effect=RuntimeError("boom")):
        for index in range(4):
            agg.record(Outcome(), now=7000.0 + index * 60)
        agg.flush_all()
    assert len(rows) >= 2, "a v2 fold failure suppressed the heartbeat"
    assert rows[-1]["session"]["turns"] == 4
    assert rows[-1]["tokens"]["saved"] == 1200


# ---------------------------------------------------------------------------
# client and receiver must agree
# ---------------------------------------------------------------------------


def test_allowlist_covers_every_emitted_key():
    """The worker drops any top-level key not in ALLOWED_KEYS, permanently and
    silently. A client that emits a key the worker has not been taught is a
    signal that looks healthy and stores nothing, so the two files have to be
    checked against each other rather than by eye.
    """
    worker = (Path(__file__).resolve().parents[1] / "deploy/beacon/worker.js").read_text()
    match = re.search(r"const ALLOWED_KEYS = \[(.*?)\n\];", worker, re.S)
    assert match, "ALLOWED_KEYS not found in worker.js"
    without_comments = re.sub(r"//[^\n]*", "", match.group(1))
    allowed = set(re.findall(r"'([a-z_]+)'", without_comments))

    emitted = set(emit(Outcome()))
    missing = emitted - allowed
    assert not missing, f"worker.js would silently discard: {sorted(missing)}"
    assert V1_TOP <= allowed, "a v1 key was removed from the allowlist"
    assert V2_TOP <= allowed


def test_schema_version_was_bumped():
    assert S.SCHEMA_VERSION == 2
    assert emit(Outcome())["schema_version"] == 2


def test_worst_case_payload_fits_the_receivers_body_limit():
    """The client must not be able to build a body the Worker will refuse.

    Not a formality. The cap is checked on what ARRIVES, so an uncompressed
    upload -- the kill switch is set, or the gzip fallback fired -- is what has
    to fit. And a 413 correctly does NOT trigger the uncompressed retry (see
    the gzip tests), so a body over the cap is simply lost.

    Read from worker.js rather than hardcoded, so raising one and not the other
    fails here instead of in production.
    """
    worker = (Path(__file__).resolve().parents[1] / "deploy/beacon/worker.js").read_text()
    match = re.search(r"const MAX_BODY_BYTES = (\d+) \* 1024;", worker)
    assert match, "MAX_BODY_BYTES not found in worker.js"
    cap = int(match.group(1)) * 1024

    # Every bounded table driven to its cap at once.
    for index in range(S.MAX_SHAPE_ROWS + 20):
        S.record_content_shape(f"content{index}", f"strategy{index}", 9000, 3000)
        S.record_tool_shape(
            type("Sig", (), {"field_count": 8, "max_depth": index % 10, "has_arrays": True})(),
            5000,
            1200,
        )
    for index in range(S.MAX_STRATEGIES + 5):
        S.record_compression(f"strategy{index}", 9000, 3000)

    outcomes = []
    for index in range(1200):
        outcomes.append(
            type(
                f"O{index}",
                (Outcome,),
                {
                    "status_code": 400 + (index % 99),
                    "client": f"harness_{index % 70}",
                    "transforms_applied": (
                        f"output_shaper:stratum:m|kind{index % 40}|b{index % 9}|tools",
                    ),
                    "tags": {"passthrough_reason": f"reason{index % 20}"},
                    "waste_signals": dict.fromkeys(
                        (*S._WASTE_KEYS, "reread", "reread_compressed"), 9
                    ),
                },
            )()
        )
    payload = emit(*outcomes)
    body = json.dumps(
        S.build_otlp_logs(payload, S.resource_attributes()), separators=(",", ":")
    ).encode()
    assert len(body) < cap, (
        f"worst-case OTLP body is {len(body):,} B against a {cap:,} B cap — the "
        "Worker would 413 it and it would not be retried"
    )
    # Gzipped it is a rounding error, which is the normal path.
    assert len(gzip.compress(body, S._GZIP_LEVEL)) < cap // 10


# ---------------------------------------------------------------------------
# headroom telemetry --show
# ---------------------------------------------------------------------------


def test_telemetry_show_prints_the_real_schema_not_a_mockup():
    """The command exists so a user can verify what leaves their machine
    without reading two source files and trusting they match the installed
    copy. That only holds if it renders the same payload the beacon builds —
    so it is asserted against the live key set, not against a fixture."""
    from click.testing import CliRunner

    from headroom.cli.main import main as cli

    result = CliRunner().invoke(cli, ["telemetry", "--show", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["schema_version"] == S.SCHEMA_VERSION
    assert V1_TOP <= set(report["session_event"])
    assert V2_TOP <= set(report["session_event"])
    assert set(report["session_event"]) == set(emit(Outcome())), (
        "--show has drifted from the payload the beacon actually sends"
    )


def test_telemetry_show_reports_the_beacon_state(monkeypatch):
    from click.testing import CliRunner

    from headroom.cli.main import main as cli

    monkeypatch.setenv("HEADROOM_BEACON", "off")
    off = CliRunner().invoke(cli, ["telemetry", "--show", "--json"])
    assert json.loads(off.output)["beacon_enabled"] is False

    monkeypatch.setenv("HEADROOM_BEACON", "on")
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    on = CliRunner().invoke(cli, ["telemetry", "--show", "--json"])
    assert json.loads(on.output)["beacon_enabled"] is True


def test_telemetry_command_does_not_mint_an_install_id(tmp_path, monkeypatch):
    """Inspecting telemetry — very plausibly on the way to switching it off —
    must not be the thing that gives a machine its identifier. `install_id()`
    creates and persists one as a side effect, so this command reads the file
    directly instead."""
    from click.testing import CliRunner

    from headroom.cli.main import main as cli

    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    # The id is memoised in a module global once anything has read it, so the
    # cache has to be cleared or this test reads the developer's real id and
    # never exercises the create path at all.
    monkeypatch.setattr(S, "_install_id", None)

    # BOTH paths. `--show` is the one that matters and the one that regressed:
    # filtering the id out of the rendered report is not enough, because
    # resource_attributes() mints it while building that report.
    for argv in (["telemetry"], ["telemetry", "--show"], ["telemetry", "--json"]):
        monkeypatch.setattr(S, "_install_id", None)
        result = CliRunner().invoke(cli, argv)
        assert result.exit_code == 0, result.output
        assert not list(tmp_path.rglob("install_id")), f"{argv} created an install id"
    assert "none yet" in CliRunner().invoke(cli, ["telemetry"]).output


def test_clients_map_is_capped_against_a_hostile_header():
    """`classify_client` returns the `X-Client` header verbatim when a caller
    sets one, so this field is attacker-influenced. Junk already collapses to
    "other"; the cap is what stops a caller minting a new valid-looking slug
    per request and growing the payload without bound."""
    outcomes = [
        type(f"C{i}", (Outcome,), {"client": f"harness_{i}"})() for i in range(S.MAX_SHAPES + 25)
    ]
    clients = emit(*outcomes)["clients"]
    assert len(clients) == S.MAX_SHAPES


# ---------------------------------------------------------------------------
# shape stability
# ---------------------------------------------------------------------------


def key_shape(node: Any, path: str = "") -> set[str]:
    """Every key path in a payload, with list contents collapsed.

    Numeric key segments are normalised, because a map keyed by a number is
    safe: DuckDB cannot read "429" as a struct field name, so it infers MAP —
    which is what the v1 `failure_statuses` column has always been across the
    whole corpus. Identifier-shaped keys are the dangerous ones.
    """
    shape: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            name = "<num>" if str(key).isdigit() else key
            shape.add(f"{path}.{name}")
            shape |= key_shape(value, f"{path}.{name}")
    elif isinstance(node, list):
        for item in node:
            shape |= key_shape(item, f"{path}[]")
    return shape


def test_payload_shape_does_not_depend_on_the_data():
    """The invariant behind every "list of records, not an object" note in
    payload(): two sessions with nothing in common must produce the SAME set of
    key paths.

    An object keyed by something that came from the data breaks this, and the
    consequence is not cosmetic — DuckDB infers such an object as a STRUCT
    while its keys are few and identifier-shaped and as a MAP once they are
    not, so the COLUMN TYPE becomes a function of the contents. Measured on the
    real corpus: two harnesses gave STRUCT(claude_code, codex) and a query for
    `clients.cursor` failed to bind; twenty-two gave MAP(VARCHAR, BIGINT) and
    the same query ran. That is the break that took out the transforms report.

    This test is the general guard, so a future field cannot reintroduce it.
    """

    class Left(Outcome):
        client = "claude-code"
        status_code = 200
        transforms_applied = ("output_shaper:stratum:sonnet|tool_result|l|tools",)
        waste_signals = {"json_bloat": 50, "reread": 10}

    class Right(Outcome):
        client = "aider"
        status_code = 200
        transforms_applied = ("output_shaper:control:opus|new_user_ask|xs|notools",)
        waste_signals = {"base64": 7, "whitespace": 3, "repetition": 11}

    import os

    os.environ["HEADROOM_MODE"] = "optimize"
    try:
        left = key_shape(emit(Left()))
    finally:
        os.environ.pop("HEADROOM_MODE", None)
    os.environ["HEADROOM_DEDUPE"] = "on"
    os.environ["HEADROOM_LOSSLESS"] = "on"
    try:
        right = key_shape(emit(Right()))
    finally:
        os.environ.pop("HEADROOM_DEDUPE", None)
        os.environ.pop("HEADROOM_LOSSLESS", None)

    assert left == right, (
        "payload shape varies with its contents — these paths differ, which "
        f"means a column type will too: {sorted(left ^ right)}"
    )


def test_the_fields_that_carry_open_vocabularies_are_lists():
    """Belt and braces on the test above, naming the four that were objects."""
    payload = emit(Outcome())
    for field in ("clients", "config", "strata"):
        assert isinstance(payload[field], list), f"{field} must be a list of records"
    assert isinstance(payload["quality"]["waste"], list)
    # ...and the records have fixed field names, which is what makes the type
    # stable no matter how the vocabulary grows.
    assert isinstance(payload["shapes"]["by_content"], list)


def test_trajectory_string_is_a_faithful_prefix_once_truncated():
    """Past the cap the string must STOP, not keep growing its last run.

    Letting a matching kind keep incrementing looks harmless and is not: once a
    run has been dropped the remaining turns are no longer contiguous, so a run
    of `s6` would claim six consecutive `s` turns where the session actually
    did `s q s q s q`. Every run in the string has to be true.
    """
    runs_at_cap = []
    for index in range(S.MAX_TRAJECTORY_RUNS):
        runs_at_cap.append(
            type(f"A{index}", (Outcome,), {"status_code": 500 if index % 2 else 200})()
        )
    # Then a tail that alternates between the last run's kind and another.
    tail = []
    for index in range(20):
        tail.append(type(f"B{index}", (Outcome,), {"status_code": 200 if index % 2 else 429})())

    payload = emit(*runs_at_cap, *tail)
    kinds = payload["trajectory"]["kinds"]
    assert payload["trajectory"]["kinds_truncated"] is True
    runs = re.findall(r"([a-z])(\d+)", kinds)
    assert len(runs) == S.MAX_TRAJECTORY_RUNS
    # The prefix is exactly the pre-cap history: alternating runs of length 1.
    assert all(int(n) == 1 for _, n in runs), f"a run grew after truncation: {kinds}"
    # And the uncapped turn counter still accounts for every turn.
    assert sum(payload["trajectory"]["turns"]) == payload["session"]["turns"]
    assert payload["session"]["turns"] == S.MAX_TRAJECTORY_RUNS + 20


def test_content_shape_cap_exceeds_its_vocabulary(beacon_on):
    """The cap must not be able to bite in normal operation. It drops rows
    first-come-wins, which would bias the table toward whatever a session
    routed early — and that table is what a routing policy gets fitted on."""
    from headroom.transforms.content_detector import ContentType
    from headroom.transforms.content_router import CompressionStrategy

    # +1 each for the "other" bucket a free string collapses to.
    cross_product = (len(list(ContentType)) + 1) * (len(list(CompressionStrategy)) + 1)
    assert S.MAX_SHAPE_ROWS >= cross_product, (
        f"{cross_product} legitimate (content, strategy) pairs exist but the cap "
        f"is {S.MAX_SHAPE_ROWS}"
    )

    for content in ContentType:
        for strategy in CompressionStrategy:
            S.record_content_shape(content.value, strategy.value, 100, 50)
    rows = emit(Outcome())["shapes"]["by_content"]
    assert len(rows) == len(list(ContentType)) * len(list(CompressionStrategy)), (
        "a legitimate content/strategy combination was dropped"
    )


def test_the_client_budget_stays_under_every_deployed_receiver():
    """`_MAX_WIRE_BYTES` is only useful while it is below what receivers accept.

    The current Worker allows 256KB, but an install upgrades on its own schedule
    and may be pointed at an older deployment whose cap is 64KB. Raising the
    client budget past that would silently reintroduce the 413 this bounds.
    """
    worker = (Path(__file__).resolve().parents[1] / "deploy/beacon/worker.js").read_text()
    match = re.search(r"const MAX_BODY_BYTES = (\d+) \* 1024;", worker)
    assert match, "MAX_BODY_BYTES not found in worker.js"
    assert S._MAX_WIRE_BYTES <= int(match.group(1)) * 1024
    assert S._MAX_WIRE_BYTES <= 64 * 1024, (
        f"the client budget is {S._MAX_WIRE_BYTES:,} B, over the 64KB cap of the "
        "oldest Worker still in service"
    )


def test_the_worst_case_body_actually_sent_fits_the_oldest_receiver():
    """The test above bounds what can be BUILT; this bounds what is SENT.

    `_fit_to_wire` is the thing standing between the two, and uncompressed v2
    crosses 64KB well before the shape caps are reached.
    """
    for index in range(S.MAX_SHAPE_ROWS + 20):
        S.record_content_shape(f"content{index}", f"strategy{index}", 9000, 3000)
        S.record_tool_shape(
            type("Sig", (), {"field_count": 8, "max_depth": index % 10, "has_arrays": True})(),
            5000,
            1200,
        )
    outcomes = [
        type(
            f"O{index}",
            (Outcome,),
            {"status_code": 400 + (index % 99), "client": f"harness_{index % 70}"},
        )()
        for index in range(600)
    ]
    payload = emit(*outcomes)
    saved = payload["tokens"]["saved"]
    sent = S._fit_to_wire(payload, S.resource_attributes(), compress=False)
    assert len(sent) <= 64 * 1024, f"sent {len(sent):,} B, over the 64KB receiver cap"
    assert payload["tokens"]["saved"] == saved, "trimming cost a v1 counter"
