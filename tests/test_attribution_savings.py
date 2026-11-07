"""Attribution labels through the savings surfaces: tracker, ledger, CLI, proxy."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest
from click.testing import CliRunner

from headroom import savings_ledger
from headroom.proxy.attribution import set_current_attribution
from headroom.proxy.savings_tracker import (
    DEFAULT_MAX_LABELS_PER_DIMENSION,
    SavingsTracker,
)


@pytest.fixture(autouse=True)
def _clean_context():
    set_current_attribution(None)
    yield
    set_current_attribution(None)


# ── SavingsTracker ────────────────────────────────────────────────────


def test_tracker_books_each_dimension_and_persists(tmp_path):
    path = tmp_path / "savings.json"
    tracker = SavingsTracker(path=path)
    tracker.record_request(
        model="gpt-4o",
        input_tokens=600,
        tokens_saved=400,
        project="web",
        attribution={"host": "studio", "client": "claude-code", "project": "ignored"},
    )
    tracker.record_request(
        model="gpt-4o",
        input_tokens=300,
        tokens_saved=100,
        attribution={"host": "studio", "client": "codex"},
    )
    tracker.record_request(
        model="gpt-4o", input_tokens=200, tokens_saved=50, attribution={"host": "build01"}
    )
    tracker.flush()

    by_dim = SavingsTracker(path=path).snapshot()["by_dimension"]
    assert by_dim["host"]["studio"]["requests"] == 2
    assert by_dim["host"]["studio"]["tokens_saved"] == 500
    assert list(by_dim["host"]) == ["studio", "build01"]
    assert by_dim["host"]["studio"]["savings_percent"] == pytest.approx(500 / 1400 * 100, 0.01)
    assert set(by_dim["client"]) == {"claude-code", "codex"}
    # project keeps its own legacy map and is surfaced from it, once.
    assert by_dim["project"] == {"web": by_dim["project"]["web"]}
    assert by_dim["project"]["web"]["tokens_saved"] == 400


def test_unattributed_requests_change_only_the_aggregate(tmp_path):
    tracker = SavingsTracker(path=tmp_path / "savings.json")
    tracker.record_request(model="gpt-4o", input_tokens=600, tokens_saved=400)
    snapshot = tracker.snapshot()
    assert snapshot["by_dimension"] == {}
    assert snapshot["lifetime"]["tokens_saved"] == 400


def test_labels_per_dimension_are_capped_keeping_the_largest(tmp_path):
    tracker = SavingsTracker(path=tmp_path / "savings.json")
    for i in range(DEFAULT_MAX_LABELS_PER_DIMENSION + 10):
        tracker.record_request(
            model="gpt-4o",
            input_tokens=10,
            tokens_saved=i + 1,
            attribution={"host": f"h{i:03d}"},
        )
    hosts = tracker.snapshot()["by_dimension"]["host"]
    assert len(hosts) == DEFAULT_MAX_LABELS_PER_DIMENSION
    assert f"h{DEFAULT_MAX_LABELS_PER_DIMENSION + 9:03d}" in hosts
    assert "h000" not in hosts


def test_state_without_dimensions_loads_and_garbage_is_dropped(tmp_path):
    path = tmp_path / "savings.json"
    tracker = SavingsTracker(path=path)
    tracker.record_request(model="gpt-4o", input_tokens=100, tokens_saved=10)
    tracker.flush()
    raw = json.loads(path.read_text())
    raw.pop("dimensions", None)
    path.write_text(json.dumps(raw))
    assert SavingsTracker(path=path).snapshot()["by_dimension"] == {}

    raw["dimensions"] = {
        "host": {"ok": {"requests": 2, "tokens_saved": "7"}, "bad": "x"},
        "Bad Name": {"a": {}},
        "project": {"shadow": {"requests": 1}},
    }
    path.write_text(json.dumps(raw))
    by_dim = SavingsTracker(path=path).snapshot()["by_dimension"]
    assert by_dim == {"host": {"ok": by_dim["host"]["ok"]}}
    assert by_dim["host"]["ok"]["tokens_saved"] == 7


# ── Ledger ────────────────────────────────────────────────────────────


def _event(path, saved, *, client="claude-code", dims=None, when=None):
    assert savings_ledger.record_savings_event(
        tokens_before=1000,
        tokens_after=1000 - saved,
        model="gpt-4o",
        client=client,
        source="proxy",
        path=path,
        timestamp=when,
        attribution=dims,
    )


def test_ledger_writes_dims_without_repeating_client(tmp_path):
    path = tmp_path / "events.jsonl"
    _event(path, 100, dims={"host": "studio", "client": "claude-code"})
    _event(path, 100)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["dims"] == {"host": "studio"}
    assert "dims" not in lines[1]


def test_ledger_groups_and_filters_by_dimension(tmp_path):
    path = tmp_path / "events.jsonl"
    _event(path, 400, dims={"host": "studio", "project": "web"})
    _event(path, 100, dims={"host": "studio", "project": "api"}, client="codex")
    _event(path, 50, dims={"host": "build01", "project": "web"})
    _event(path, 25)  # pre-attribution event

    report = savings_ledger.aggregate_savings(path, group_by=["host", "client", "Host"])
    assert list(report.by_dimension) == ["host", "client"]
    hosts = {row["host"]: row for row in report.by_dimension["host"]}
    assert hosts["studio"]["tokens_saved"] == 500
    assert hosts["build01"]["tokens_saved"] == 50
    assert hosts["unknown"]["tokens_saved"] == 25
    assert {row["client"] for row in report.by_dimension["client"]} == {"claude-code", "codex"}
    assert report.dimensions == ["client", "host", "project"]

    scoped = savings_ledger.aggregate_savings(path, group_by=["project"], where={"host": "studio"})
    assert scoped.lifetime["tokens_saved"] == 500
    assert {row["project"]: row["tokens_saved"] for row in scoped.by_dimension["project"]} == {
        "web": 400,
        "api": 100,
    }
    assert scoped.to_dict()["where"] == {"host": "studio"}


def test_ledger_windows_respect_the_filter(tmp_path):
    path = tmp_path / "events.jsonl"
    old = datetime.now(timezone.utc) - timedelta(days=10)
    _event(path, 100, dims={"host": "a"}, when=old.isoformat())
    _event(path, 200, dims={"host": "a"})
    _event(path, 400, dims={"host": "b"})
    report = savings_ledger.aggregate_savings(path, where={"host": "a"})
    assert report.windows["last_7_days"]["tokens_saved"] == 200
    assert report.windows["last_30_days"]["tokens_saved"] == 300


# ── CLI ───────────────────────────────────────────────────────────────


def test_savings_cli_by_and_where(tmp_path, monkeypatch):
    from headroom.cli import savings as savings_cli

    path = tmp_path / "events.jsonl"
    _event(path, 400, dims={"host": "studio"})
    _event(path, 100, dims={"host": "build01"})
    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(path))
    monkeypatch.setattr(savings_cli, "_proxy_routing", lambda: None)

    runner = CliRunner()
    result = runner.invoke(savings_cli.savings, ["--by", "host"])
    assert result.exit_code == 0, result.output
    assert "Savings by host:" in result.output
    assert result.output.index("studio") < result.output.index("build01")

    result = runner.invoke(savings_cli.savings, ["--where", "host=build01", "--json"])
    data = json.loads(result.output)
    assert data["lifetime"]["tokens_saved"] == 100
    assert data["where"] == {"host": "build01"}

    result = runner.invoke(savings_cli.savings, ["--where", "nonsense"])
    assert result.exit_code != 0

    result = runner.invoke(savings_cli.savings, ["--by", "team"])
    assert "No attributed events" in result.output
    assert "host" in result.output


# ── Through the proxy ─────────────────────────────────────────────────


def _proxy_client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "savings.json"))
    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(tmp_path / "events.jsonl"))
    config = ProxyConfig(cache_enabled=False, rate_limit_enabled=False, log_requests=False)
    return TestClient(create_app(config), base_url="http://127.0.0.1", client=("127.0.0.1", 1))


def _emit(proxy, **fields):
    from headroom.proxy.outcome import RequestOutcome, emit_request_outcome

    outcome = RequestOutcome(
        request_id=fields.pop("request_id", "req-1"),
        provider="openai",
        model="gpt-4o",
        original_tokens=1000,
        optimized_tokens=600,
        output_tokens=20,
        tokens_saved=400,
        attempted_input_tokens=1000,
        **fields,
    )
    asyncio.run(emit_request_outcome(proxy, outcome))


def test_funnel_books_context_labels_and_outcome_fields_win(tmp_path, monkeypatch):
    with _proxy_client(tmp_path, monkeypatch) as client:
        proxy = client.app.state.proxy
        set_current_attribution({"host": "studio", "project": "from-header", "team": "infra"})
        _emit(proxy, project="late-project", client="codex", attribution={"team": "core"})
        set_current_attribution(None)

        by_dim = client.get("/stats").json()["savings"]["by_dimension"]
        assert by_dim["host"]["studio"]["tokens_saved"] == 400
        assert set(by_dim["project"]) == {"late-project"}
        assert set(by_dim["client"]) == {"codex"}
        assert set(by_dim["team"]) == {"core"}

        lifetime = client.get("/stats-lifetime").json()
        assert lifetime["by_dimension"]["host"]["studio"]["requests"] == 1

    events = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["dims"] == {"host": "studio", "project": "late-project", "team": "core"}


def test_snapshot_carries_labels_to_pipeline_extensions(tmp_path, monkeypatch):
    from headroom.pipeline import PipelineStage

    seen = []

    class Listener:
        def on_pipeline_event(self, event):
            if event.stage is PipelineStage.OUTCOME_OBSERVED:
                seen.append(event.outcome)

    with _proxy_client(tmp_path, monkeypatch) as client:
        proxy = client.app.state.proxy
        listener = Listener()
        proxy.pipeline_extensions.add(listener)
        proxy.pipeline_extensions.add(listener)  # idempotent
        set_current_attribution({"host": "studio"})
        _emit(proxy, conversation_key="conv-1")
        set_current_attribution(None)

    snapshot = seen[-1]
    assert dict(snapshot.attribution) == {"host": "studio"}
    assert snapshot.conversation_key == "conv-1"
    assert snapshot.original_tokens == 1000
    with pytest.raises(TypeError):
        snapshot.attribution["host"] = "x"  # type: ignore[index]


def test_middleware_resolves_labels_from_request(tmp_path, monkeypatch):
    import headroom.proxy.server as server_module

    captured = []
    real = server_module.set_current_attribution

    def capture(labels):
        captured.append(dict(labels or {}))
        real(labels)

    monkeypatch.setattr(server_module, "set_current_attribution", capture)
    with _proxy_client(tmp_path, monkeypatch) as client:
        client.get(
            "/health",
            headers={
                "X-Headroom-Project": "web",
                "User-Agent": "claude-cli/2.1.0 (external, sdk-cli)",
            },
        )
        assert captured[-1] == {"project": "web", "client": "claude-code"}
        client.get("/p/aider-proj/health")
        assert captured[-1]["project"] == "aider-proj"


def test_network_callers_do_not_see_labels(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "savings.json"))
    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(tmp_path / "events.jsonl"))
    config = ProxyConfig(cache_enabled=False, rate_limit_enabled=False, log_requests=False)
    app = create_app(config)
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 1)) as local:
        set_current_attribution({"host": "studio"})
        _emit(local.app.state.proxy)
        set_current_attribution(None)
    with TestClient(app, base_url="http://10.0.0.5", client=("10.0.0.9", 1)) as remote:
        stats = remote.get("/stats")
        assert stats.status_code == 200
        assert "by_dimension" not in stats.json()["savings"]
        lifetime = remote.get("/stats-lifetime")
        assert lifetime.status_code == 200
        assert "by_dimension" not in lifetime.json()


def test_proxy_token_holders_see_labels_like_recent_requests(tmp_path, monkeypatch):
    """Labels follow the per-request metadata gate, which admits a verified
    proxy-token holder (a dashboard behind an authenticating reverse proxy)."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "savings.json"))
    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("HEADROOM_PROXY_TOKEN", "s3cret-token")
    monkeypatch.setenv("HEADROOM_HOST", "0.0.0.0")
    config = ProxyConfig(cache_enabled=False, rate_limit_enabled=False, log_requests=False)
    app = create_app(config)
    token = {"x-headroom-proxy-token": "s3cret-token"}
    with TestClient(app, base_url="http://headroom", client=("10.0.4.7", 5000)) as client:
        set_current_attribution({"team": "infra"})
        _emit(client.app.state.proxy)
        set_current_attribution(None)
        stats = client.get("/stats", headers=token).json()
        assert stats["savings"]["by_dimension"]["team"]["infra"]["requests"] == 1
        lifetime = client.get("/stats-lifetime", headers=token).json()
        assert lifetime["by_dimension"]["team"]["infra"]["requests"] == 1
        assert client.get("/stats").status_code == 401


def test_snapshot_falls_back_to_the_client_named_conversation(tmp_path, monkeypatch):
    from headroom.pipeline import PipelineStage
    from headroom.proxy.attribution import (
        conversation_hint,
        set_current_conversation,
    )

    assert conversation_hint({"x-claude-code-session-id": "abc-1"}) == "abc-1"
    assert (
        conversation_hint({"x-headroom-session-id": "mine", "x-claude-code-session-id": "x"})
        == "mine"
    )
    assert conversation_hint({}) is None

    seen = []

    class Listener:
        def on_pipeline_event(self, event):
            if event.stage is PipelineStage.OUTCOME_OBSERVED:
                seen.append(event.outcome)

    with _proxy_client(tmp_path, monkeypatch) as client:
        proxy = client.app.state.proxy
        proxy.pipeline_extensions.add(Listener())
        set_current_conversation("cc-session-7")
        try:
            _emit(proxy)  # handler derived no key: the client's wins
            _emit(proxy, request_id="req-2", conversation_key="handler-key")
        finally:
            set_current_conversation(None)
    assert [s.conversation_key for s in seen[-2:]] == ["cc-session-7", "handler-key"]
