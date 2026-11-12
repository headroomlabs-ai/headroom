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
def _clean_context(monkeypatch):
    monkeypatch.delenv("HEADROOM_ATTRIBUTION_DIMENSIONS", raising=False)
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
        attribution={"host": "studio", "client": "claude-code", "project": "web-label"},
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
    # by_dimension.project is the attribution label, like any other dimension;
    # the project argument feeds the older projects totals.
    assert set(by_dim["project"]) == {"web-label"}
    assert by_dim["project"]["web-label"]["tokens_saved"] == 400
    assert set(SavingsTracker(path=path).snapshot()["projects"]) == {"web"}


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


def test_a_new_dimension_evicts_the_smallest_idle_one(tmp_path):
    from headroom.proxy.savings_tracker import DEFAULT_MAX_DIMENSIONS

    tracker = SavingsTracker(path=tmp_path / "savings.json")
    crowd = {f"junk{i}": "x" for i in range(DEFAULT_MAX_DIMENSIONS - 1)}
    tracker.record_request(model="gpt-4o", input_tokens=10, tokens_saved=1, attribution=crowd)
    tracker.record_request(
        model="gpt-4o", input_tokens=600, tokens_saved=400, attribution={"team": "infra"}
    )
    assert len(tracker.snapshot()["by_dimension"]) == DEFAULT_MAX_DIMENSIONS
    # Full. A dimension seen now takes the place of a small idle one, not of
    # the one with real savings, and the cap holds.
    tracker.record_request(
        model="gpt-4o", input_tokens=300, tokens_saved=100, attribution={"client": "codex"}
    )
    by_dim = tracker.snapshot()["by_dimension"]
    assert len(by_dim) == DEFAULT_MAX_DIMENSIONS
    assert by_dim["client"]["codex"]["tokens_saved"] == 100
    assert by_dim["team"]["infra"]["tokens_saved"] == 400
    assert sum(name.startswith("junk") for name in by_dim) == DEFAULT_MAX_DIMENSIONS - 2


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
    assert set(by_dim) == {"host", "project"}
    assert by_dim["host"] == {"ok": by_dim["host"]["ok"]}
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


def test_ledger_refuses_to_group_by_a_name_that_is_a_row_field(tmp_path):
    path = tmp_path / "events.jsonl"
    _event(path, 400, dims={"calls": "infra", "host": "studio"})
    _event(path, 100, dims={"calls": "web"})
    for name in sorted(savings_ledger.ROW_FIELDS):
        with pytest.raises(ValueError, match=name):
            savings_ledger.aggregate_savings(path, group_by=[name])
    # The label is still recorded and still filters; nothing was merged away.
    report = savings_ledger.aggregate_savings(path, group_by=["host"], where={"calls": "infra"})
    assert report.lifetime["tokens_saved"] == 400
    assert [(row["host"], row["calls"]) for row in report.by_dimension["host"]] == [("studio", 1)]
    assert "calls" in report.dimensions


def test_ledger_rejects_names_that_are_not_dimensions(tmp_path):
    path = tmp_path / "events.jsonl"
    _event(path, 400, dims={"team": "infra"})
    with pytest.raises(ValueError, match="cost center"):
        savings_ledger.aggregate_savings(path, where={"cost center": "42"})
    with pytest.raises(ValueError, match="cost center"):
        savings_ledger.aggregate_savings(path, group_by=["cost center"])
    # A bare string is one name, not its characters.
    report = savings_ledger.aggregate_savings(path, group_by="team")
    assert [row["team"] for row in report.by_dimension["team"]] == ["infra"]


def test_ledger_lists_every_dimension_whatever_the_filter(tmp_path):
    path = tmp_path / "events.jsonl"
    _event(path, 400, dims={"team": "infra"})
    _event(path, 100, dims={"region": "eu"})
    report = savings_ledger.aggregate_savings(path, where={"team": "infra"})
    assert report.dimensions == ["client", "region", "team"]
    assert report.lifetime["tokens_saved"] == 400
    # Unlabelled events are the ``unknown`` group, and can be selected as such.
    unknown = savings_ledger.aggregate_savings(path, where={"team": "unknown"})
    assert unknown.lifetime["tokens_saved"] == 100
    # ``client`` filters on the event's own key.
    by_harness = savings_ledger.aggregate_savings(path, where={"client": "claude-code"})
    assert by_harness.lifetime["tokens_saved"] == 500


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

    # The proxy-wide routing rollup is not shown beside a filtered report.
    routing = {"decisions": 3, "savings_usd": 20.0}
    monkeypatch.setattr(savings_cli, "_proxy_routing", lambda: routing)
    assert "routing" in json.loads(runner.invoke(savings_cli.savings, ["--json"]).output)
    filtered = runner.invoke(savings_cli.savings, ["--where", "host=studio", "--json"])
    assert "routing" not in json.loads(filtered.output)

    result = runner.invoke(savings_cli.savings, ["--by", "team"])
    assert "No attributed events" in result.output
    assert "host" in result.output


def test_savings_cli_filter_edges(tmp_path, monkeypatch):
    from headroom.cli import savings as savings_cli

    path = tmp_path / "events.jsonl"
    _event(path, 400, dims={"team": "infra"})
    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(path))
    monkeypatch.setattr(savings_cli, "_proxy_routing", lambda: None)
    runner = CliRunner()

    # A name that is not a dimension is an error, not an unfiltered report.
    for args in (["--where", "cost center=42"], ["--by", "cost center"]):
        result = runner.invoke(savings_cli.savings, args)
        assert result.exit_code == 2, result.output
        assert "not a dimension name" in result.output
    result = runner.invoke(savings_cli.savings, ["--where", "team=a", "--where", "team=b"])
    assert result.exit_code == 2
    assert "given twice" in result.output

    # A filter that matches nothing says so, and names the filter.
    result = runner.invoke(savings_cli.savings, ["--where", "team=Infra"])
    assert result.exit_code == 0, result.output
    assert "No savings recorded for team=Infra" in result.output
    assert "No savings recorded yet" not in result.output

    # ``--by client`` prints one client section, not two.
    result = runner.invoke(savings_cli.savings, ["--by", "client"])
    assert result.output.count("Savings by client:") == 1


def test_savings_cli_rejects_by_a_row_field(tmp_path, monkeypatch):
    from headroom.cli import savings as savings_cli

    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(tmp_path / "events.jsonl"))
    result = CliRunner().invoke(savings_cli.savings, ["--by", "calls"])
    assert result.exit_code == 2
    assert "cannot group by calls" in result.output


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
    monkeypatch.setenv("HEADROOM_ATTRIBUTION_DIMENSIONS", "default,host,team")
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


def test_a_dimension_turned_off_is_not_booked_from_the_legacy_field(tmp_path, monkeypatch):
    from headroom.proxy import attribution

    monkeypatch.setenv(attribution.ATTRIBUTION_DIMENSIONS_ENV_VAR, "default,-client")
    with _proxy_client(tmp_path, monkeypatch) as client:
        _emit(client.app.state.proxy, project="web", client="codex")
        by_dim = client.get("/stats").json()["savings"]["by_dimension"]
        assert "client" not in by_dim
        assert set(by_dim["project"]) == {"web"}
        attribution.bind_label("client", "codex")
        assert "client" not in attribution.get_current_attribution()
    event = json.loads((tmp_path / "events.jsonl").read_text().splitlines()[-1])
    # The ledger's own client key predates labels and is unchanged.
    assert event["client"] == "codex"


def test_a_handler_cannot_book_a_dimension_that_is_off(tmp_path, monkeypatch):
    from headroom.proxy import attribution

    monkeypatch.setenv(attribution.ATTRIBUTION_DIMENSIONS_ENV_VAR, "default,squad,-team")
    with _proxy_client(tmp_path, monkeypatch) as client:
        _emit(client.app.state.proxy, attribution={"team": "x", "squad": "blue", "other": "y"})
        by_dim = client.get("/stats").json()["savings"]["by_dimension"]
        assert set(by_dim) == {"squad"}
    monkeypatch.setenv(attribution.ATTRIBUTION_DIMENSIONS_ENV_VAR, "none")
    with _proxy_client(tmp_path, monkeypatch) as client:
        _emit(client.app.state.proxy, project="web", client="codex", attribution={"team": "x"})
        assert client.get("/stats").json()["savings"]["by_dimension"] == {"squad": by_dim["squad"]}


def test_client_named_labels_do_not_change_the_request_log_tags(tmp_path, monkeypatch):
    from headroom.proxy import attribution

    monkeypatch.setenv(attribution.ATTRIBUTION_DIMENSIONS_ENV_VAR, "labels")
    with _proxy_client(tmp_path, monkeypatch) as client:
        proxy = client.app.state.proxy
        sink = _LogSink()
        with monkeypatch.context() as patch:
            patch.setattr(proxy, "logger", sink, raising=False)
            set_current_attribution({"project": "web", "client": "cursor"})
            _emit(proxy, project="api", client="codex")
            set_current_attribution(None)
        tags = sink.entries[-1].tags
        # by_dimension books what the client named; the tags stay what was detected.
        assert (tags["client"], tags["project"]) == ("codex", "api")
        by_dim = client.get("/stats").json()["savings"]["by_dimension"]
        assert set(by_dim["client"]) == {"cursor"}


def test_project_turned_off_leaves_only_the_older_projects_totals(tmp_path, monkeypatch):
    from headroom.proxy import attribution

    monkeypatch.setenv(attribution.ATTRIBUTION_DIMENSIONS_ENV_VAR, "default,-project")
    with _proxy_client(tmp_path, monkeypatch) as client:
        _emit(client.app.state.proxy, project="web", client="codex")
        lifetime = client.get("/stats-lifetime").json()
        assert "project" not in lifetime["by_dimension"]
        assert set(lifetime["by_dimension"]["client"]) == {"codex"}
        assert set(lifetime["projects"]) == {"web"}


def _capture_middleware_labels(monkeypatch):
    """Record what the request middleware binds, while still binding it."""
    import headroom.proxy.server as server_module

    captured = []
    real = server_module.set_current_attribution

    def capture(labels):
        captured.append(dict(labels or {}))
        real(labels)

    monkeypatch.setattr(server_module, "set_current_attribution", capture)
    return captured


class _LogSink:
    def __init__(self):
        self.entries = []

    def log(self, entry):
        self.entries.append(entry)


def test_a_replaced_resolver_is_not_overridden_by_the_legacy_field(tmp_path, monkeypatch):
    from headroom.proxy import attribution

    captured = _capture_middleware_labels(monkeypatch)
    try:
        attribution.register_dimension("client", lambda request: "custom-client", replace=True)
        attribution.register_dimension("project", lambda request: "custom-project", replace=True)
        with _proxy_client(tmp_path, monkeypatch) as client:
            proxy = client.app.state.proxy
            # The middleware resolves the labels on a real request ...
            client.get("/health", headers={"X-Client": "codex", "X-Headroom-Project": "web"})
            resolved = captured[-1]
            assert resolved["client"] == "custom-client"
            assert resolved["project"] == "custom-project"
            # ... and a handler that identified the harness and the project
            # itself does not undo them at the outcome funnel.
            sink = _LogSink()
            with monkeypatch.context() as patch:
                patch.setattr(proxy, "logger", sink, raising=False)
                set_current_attribution(resolved)
                _emit(proxy, project="web", client="codex")
                set_current_attribution(None)

            lifetime = client.get("/stats-lifetime").json()
            for by_dim in (
                client.get("/stats").json()["savings"]["by_dimension"],
                lifetime["by_dimension"],
            ):
                assert set(by_dim["client"]) == {"custom-client"}
                assert set(by_dim["project"]) == {"custom-project"}
            # The older projects totals still record what the proxy detected.
            assert set(lifetime["projects"]) == {"web"}
            tags = sink.entries[-1].tags
            assert (tags["client"], tags["project"]) == ("custom-client", "custom-project")
    finally:
        attribution.unregister_dimension("client")
        attribution.unregister_dimension("project")

    event = json.loads((tmp_path / "events.jsonl").read_text().splitlines()[-1])
    # So does the ledger's own client key; its dims carry the resolved project.
    assert event["client"] == "codex"
    assert event["dims"]["project"] == "custom-project"
    assert "client" not in event["dims"]


def test_open_labels_book_a_builtin_dimension_that_is_not_selected(tmp_path, monkeypatch):
    from headroom.proxy import attribution

    monkeypatch.setenv(attribution.ATTRIBUTION_DIMENSIONS_ENV_VAR, "labels")
    captured = _capture_middleware_labels(monkeypatch)
    with _proxy_client(tmp_path, monkeypatch) as client:
        client.get("/health", headers={"X-Headroom-Labels": "project=web; team=infra"})
        assert captured[-1] == {"project": "web", "team": "infra"}

        set_current_attribution(captured[-1])
        # ``client`` is not listed, so the harness the handler saw is not booked.
        _emit(client.app.state.proxy, client="codex")
        set_current_attribution(None)
        by_dim = client.get("/stats").json()["savings"]["by_dimension"]
        assert set(by_dim) == {"project", "team"}
        assert set(by_dim["project"]) == {"web"}
    event = json.loads((tmp_path / "events.jsonl").read_text().splitlines()[-1])
    assert event["dims"] == {"project": "web", "team": "infra"}


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
        assert "by_dimension" not in stats.json()["persistent_savings"]
        lifetime = remote.get("/stats-lifetime")
        assert lifetime.status_code == 200
        assert "by_dimension" not in lifetime.json()
        # Whatever the key, the label itself is nowhere in either response.
        assert "studio" not in stats.text
        assert "studio" not in lifetime.text


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
