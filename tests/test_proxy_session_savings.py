"""Tests for per-session savings attribution (x-claude-code-session-id)."""

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from headroom import savings_ledger as L  # noqa: E402
from headroom.proxy.outcome import RequestOutcome, emit_request_outcome  # noqa: E402
from headroom.proxy.savings_attribution import public_tags  # noqa: E402
from headroom.proxy.savings_tracker import (  # noqa: E402
    DEFAULT_MAX_SESSIONS,
    DEFAULT_SESSION_INACTIVITY_HOURS,
    SavingsTracker,
)
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402
from headroom.proxy.session_policy import (  # noqa: E402
    CLAUDE_CODE_SESSION_HEADER,
    HEADROOM_SESSION_HEADER,
    SESSION_TAG,
    bind_session_tag,
    classify_session,
    session_from_tags,
)

UTC = timezone.utc

# ---------------------------------------------------------------------------
# classify_session / tag binding
# ---------------------------------------------------------------------------


def test_classify_session_prefers_claude_code_header():
    both = {
        CLAUDE_CODE_SESSION_HEADER: "cc-session",
        HEADROOM_SESSION_HEADER: "hr-session",
    }
    assert classify_session(both) == "cc-session"
    assert classify_session({HEADROOM_SESSION_HEADER: "hr-session"}) == "hr-session"
    assert classify_session({"user-agent": "claude-code/1.0"}) is None
    assert classify_session({CLAUDE_CODE_SESSION_HEADER: "   "}) is None
    assert classify_session(object()) is None


def test_classify_session_sanitizes_client_value():
    uuid_like = "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"
    assert classify_session({CLAUDE_CODE_SESSION_HEADER: f"  {uuid_like}  "}) == uuid_like
    assert len(classify_session({CLAUDE_CODE_SESSION_HEADER: "x" * 300})) == 128


def test_bind_session_tag_sets_internal_key_only_when_present():
    tags: dict[str, object] = {}
    bind_session_tag(tags, {CLAUDE_CODE_SESSION_HEADER: "s1"})
    assert tags == {SESSION_TAG: "s1"}
    assert session_from_tags(tags) == "s1"

    empty: dict[str, object] = {}
    bind_session_tag(empty, {"user-agent": "claude-code"})
    assert empty == {}
    assert session_from_tags(empty) is None
    assert session_from_tags(None) is None


def test_session_tag_never_reaches_public_request_log_tags():
    tags = {SESSION_TAG: "s1", "mode": "cache"}
    assert public_tags(tags) == {"mode": "cache"}


# ---------------------------------------------------------------------------
# SavingsTracker per-session aggregation
# ---------------------------------------------------------------------------


def test_tracker_accumulates_per_session_and_persists(tmp_path):
    path = tmp_path / "savings.json"
    tracker = SavingsTracker(path=str(path))

    tracker.record_request(model="gpt-4o", input_tokens=1000, tokens_saved=400, session_id="s-a")
    tracker.record_request(model="gpt-4o", input_tokens=500, tokens_saved=100, session_id="s-a")
    tracker.record_request(model="gpt-4o", input_tokens=200, tokens_saved=50, session_id="s-b")
    tracker.record_request(model="gpt-4o", input_tokens=99, tokens_saved=9)  # unattributed

    sessions = tracker.lifetime_response()["sessions"]
    assert list(sessions) == ["s-a", "s-b"]  # sorted by tokens saved desc
    assert sessions["s-a"]["requests"] == 2
    assert sessions["s-a"]["tokens_saved"] == 500
    assert sessions["s-a"]["total_input_tokens"] == 1500
    assert sessions["s-a"]["savings_percent"] == pytest.approx(25.0)
    assert sessions["s-b"]["requests"] == 1
    assert sessions["s-a"]["last_activity_at"] is not None

    # Unattributed traffic still lands in the lifetime totals.
    assert tracker.stats_preview()["lifetime"]["requests"] == 4

    # Survives a restart via the persisted JSON state.
    reloaded = SavingsTracker(path=str(path))
    assert reloaded.lifetime_response()["sessions"]["s-a"]["tokens_saved"] == 500
    assert reloaded.session_response("s-a")["tokens_saved"] == 500


def test_session_entry_shape_matches_project_entry(tmp_path):
    tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
    tracker.record_request(
        model="gpt-4o", input_tokens=100, tokens_saved=10, session_id="s", project="p"
    )
    session_row = tracker.session_response("s")
    assert session_row is not None
    project_row = tracker.lifetime_response()["projects"]["p"]
    assert set(session_row) == set(project_row)


def test_tracker_evicts_sessions_idle_beyond_inactivity_window(tmp_path):
    tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
    now = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
    tracker.record_request(
        model="gpt-4o", input_tokens=10, tokens_saved=1, session_id="old", timestamp=now
    )
    idle_at = now - timedelta(hours=DEFAULT_SESSION_INACTIVITY_HOURS + 1)
    tracker.record_request(
        model="gpt-4o", input_tokens=10, tokens_saved=2, session_id="idle", timestamp=idle_at
    )
    assert set(tracker.lifetime_response()["sessions"]) == {"old", "idle"}

    # A later session's record sweeps the one that has now been idle too long.
    tracker.record_request(
        model="gpt-4o", input_tokens=10, tokens_saved=3, session_id="fresh", timestamp=now
    )
    sessions = tracker.lifetime_response()["sessions"]
    assert "idle" not in sessions
    assert set(sessions) == {"old", "fresh"}
    # A session already evicted starts a fresh entry when it resumes: the prior figure was discarded by design, and resurrecting it would make the idle bound unenforceable.
    tracker.record_request(
        model="gpt-4o", input_tokens=10, tokens_saved=5, session_id="idle", timestamp=now
    )
    resumed = tracker.session_response("idle")
    assert resumed is not None
    assert resumed["tokens_saved"] == 5


def test_tracker_evicts_idle_sessions_on_load(tmp_path):
    path = tmp_path / "savings.json"
    now = datetime.now(UTC)
    path.write_text(
        json.dumps(
            {
                "schema_version": 6,
                "lifetime": {},
                "display_session": None,
                "history": [],
                "sessions": {
                    "recent": {
                        "requests": 1,
                        "tokens_saved": 5,
                        "last_activity_at": now.isoformat(),
                    },
                    "stale": {
                        "requests": 1,
                        "tokens_saved": 9,
                        "last_activity_at": (
                            now - timedelta(hours=DEFAULT_SESSION_INACTIVITY_HOURS + 2)
                        ).isoformat(),
                    },
                },
            }
        )
    )
    sessions = SavingsTracker(path=str(path)).lifetime_response()["sessions"]
    assert set(sessions) == {"recent"}


def test_tracker_caps_session_cardinality_by_recency(tmp_path):
    tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
    base = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
    for i in range(DEFAULT_MAX_SESSIONS + 5):
        tracker.record_request(
            model="gpt-4o",
            input_tokens=10,
            tokens_saved=1,
            session_id=f"s-{i:04d}",
            # Each session records slightly later, so s-0 is least recent.
            timestamp=base + timedelta(seconds=i),
        )
    sessions = tracker.lifetime_response()["sessions"]
    assert len(sessions) == DEFAULT_MAX_SESSIONS
    assert "s-0000" not in sessions
    assert f"s-{DEFAULT_MAX_SESSIONS + 4:04d}" in sessions


def test_tracker_normalizes_persisted_session_state(tmp_path):
    path = tmp_path / "savings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 6,
                "lifetime": {},
                "display_session": None,
                "history": [],
                "sessions": {
                    "ok": {"requests": "2", "tokens_saved": 10},
                    "": {"requests": 1},
                    "bad-entry": "not-a-dict",
                },
            }
        )
    )
    tracker = SavingsTracker(path=str(path))
    sessions = tracker.lifetime_response()["sessions"]
    assert set(sessions) == {"ok"}
    assert sessions["ok"]["requests"] == 2
    assert sessions["ok"]["tokens_saved"] == 10
    assert sessions["ok"]["compression_savings_usd"] == 0.0


def test_record_request_without_session_matches_legacy_totals(tmp_path):
    path = tmp_path / "savings.json"
    tracker = SavingsTracker(path=str(path))
    tracker.record_request(model="gpt-4o", input_tokens=100, tokens_saved=40)
    tracker.record_request(model="gpt-4o", input_tokens=200, tokens_saved=60)

    assert tracker.lifetime_response()["sessions"] == {}
    assert tracker.stats_preview()["lifetime"]["tokens_saved"] == 100
    persisted = json.loads(path.read_text())
    # Every legacy top-level key survives alongside the new sessions map.
    assert set(persisted) >= {"schema_version", "lifetime", "display_session", "history"}
    assert persisted["sessions"] == {}


# ---------------------------------------------------------------------------
# Ledger schema v3
# ---------------------------------------------------------------------------


def _events_env(monkeypatch, tmp_path):
    path = tmp_path / "savings_events.jsonl"
    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(path))
    return path


def test_ledger_records_session_and_bumps_schema(monkeypatch, tmp_path):
    path = _events_env(monkeypatch, tmp_path)
    assert L.record_savings_event(
        tokens_before=1000, tokens_after=400, model=None, client="c", session="s-1"
    )
    event = json.loads(path.read_text().splitlines()[0])
    assert event["v"] == L.SCHEMA_VERSION == 3
    assert event["session"] == "s-1"


def test_ledger_omits_session_when_unavailable(monkeypatch, tmp_path):
    path = _events_env(monkeypatch, tmp_path)
    assert L.record_savings_event(tokens_before=1000, tokens_after=400, model=None, client="c")
    # Usable-but-empty ids are omitted too: absence is the "unattributed" marker, so a blank or whitespace id must not become a bucket.
    assert L.record_savings_event(
        tokens_before=1000, tokens_after=400, model=None, client="c", session="   "
    )
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert all("session" not in event for event in lines)


def test_ledger_aggregates_v2_records_without_session(monkeypatch, tmp_path):
    path = _events_env(monkeypatch, tmp_path)
    legacy = {
        "v": 2,
        "ts": datetime.now(UTC).isoformat(),
        "before": 1000,
        "after": 400,
        "saved": 600,
        "cost_usd": 0.001,
        "model": "unknown",
        "client": "c",
        "source": "proxy",
        "pid": 1,
    }
    path.write_text(json.dumps(legacy) + "\n")
    L.record_savings_event(
        tokens_before=500, tokens_after=100, model=None, client="c", session="s-1"
    )
    report = L.aggregate_savings()
    # Both events aggregate; the v2 line simply has no session to read.
    assert report.lifetime["calls"] == 2
    assert report.lifetime["tokens_saved"] == 600 + 400


# ---------------------------------------------------------------------------
# Outcome funnel -> metrics -> tracker
# ---------------------------------------------------------------------------


def _emit_outcome(proxy, *, session_tag=None):
    outcome = RequestOutcome(
        request_id="req-1",
        provider="openai",
        model="gpt-4o",
        original_tokens=1000,
        optimized_tokens=600,
        output_tokens=20,
        tokens_saved=400,
        attempted_input_tokens=1000,
        tags={SESSION_TAG: session_tag} if session_tag else {},
    )
    asyncio.run(emit_request_outcome(proxy, outcome))


def test_funnel_attributes_savings_from_session_tag(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "savings.json"))
    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(tmp_path / "events.jsonl"))
    config = ProxyConfig(cache_enabled=False, rate_limit_enabled=False, log_requests=False)

    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy
        _emit_outcome(proxy, session_tag="sess-1")
        _emit_outcome(proxy)  # unattributed

        sessions = proxy.metrics.savings_tracker.lifetime_response()["sessions"]
        assert sessions["sess-1"]["tokens_saved"] == 400
        assert sessions["sess-1"]["requests"] == 1

        # The ledger line carries the session id when the metrics path supplied one; the unattributed request's line does not.
        events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        assert [event.get("session") for event in events] == ["sess-1", None]


def test_metrics_record_request_threads_session_to_tracker(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "savings.json"))
    config = ProxyConfig(cache_enabled=False, rate_limit_enabled=False, log_requests=False)

    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy
        asyncio.run(
            proxy.metrics.record_request(
                provider="openai",
                model="gpt-4o",
                input_tokens=120,
                output_tokens=24,
                tokens_saved=30,
                latency_ms=15.0,
                session_id="sess-2",
            )
        )
        assert proxy.metrics.savings_tracker.session_response("sess-2")["tokens_saved"] == 30


# ---------------------------------------------------------------------------
# GET /stats/sessions/<session_id>
# ---------------------------------------------------------------------------


def _make_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "savings.json"))
    return ProxyConfig(cache_enabled=False, rate_limit_enabled=False, log_requests=False)


def _loopback_client(app):
    # A real loopback peer + a loopback Host header, passing both gates of require_loopback (client IP and the DNS-rebinding Host check).
    return TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345))


def test_stats_sessions_endpoint_returns_entry_and_404(tmp_path, monkeypatch):
    with TestClient(create_app(_make_config(tmp_path, monkeypatch))) as client:
        proxy = client.app.state.proxy
        proxy.metrics.savings_tracker.record_request(
            model="gpt-4o", input_tokens=600, tokens_saved=200, session_id="sess-3"
        )

        loopback = _loopback_client(client.app)
        resp = loopback.get("/stats/sessions/sess-3")
        assert resp.status_code == 200, resp.text
        assert resp.json() == proxy.metrics.savings_tracker.session_response("sess-3")
        assert resp.json()["tokens_saved"] == 200
        assert resp.json()["requests"] == 1

        missing = loopback.get("/stats/sessions/never-seen")
        assert missing.status_code == 404
        assert missing.json() == {"error": "session_not_found", "session_id": "never-seen"}


def test_stats_sessions_endpoint_is_loopback_only(tmp_path, monkeypatch):
    # A vanilla TestClient presents client.host="testclient", not a loopback IP, so require_loopback returns 404: invisible, not merely forbidden.
    with TestClient(create_app(_make_config(tmp_path, monkeypatch))) as client:
        proxy = client.app.state.proxy
        proxy.metrics.savings_tracker.record_request(
            model="gpt-4o", input_tokens=600, tokens_saved=200, session_id="sess-4"
        )
        assert client.get("/stats/sessions/sess-4").status_code == 404


def test_stats_lifetime_strips_sessions_for_network_callers(tmp_path, monkeypatch):
    with TestClient(create_app(_make_config(tmp_path, monkeypatch))) as client:
        proxy = client.app.state.proxy
        proxy.metrics.savings_tracker.record_request(
            model="gpt-4o", input_tokens=600, tokens_saved=200, session_id="sess-5"
        )
        # Vanilla TestClient is a non-loopback caller for the metadata gate.
        network = client.get("/stats-lifetime").json()
        assert "sessions" not in network
        assert "projects" not in network
        loopback = _loopback_client(client.app).get("/stats-lifetime").json()
        assert loopback["sessions"]["sess-5"]["tokens_saved"] == 200
