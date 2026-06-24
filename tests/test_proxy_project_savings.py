"""Tests for per-project savings attribution (X-Headroom-Project)."""

import asyncio
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.outcome import RequestOutcome, emit_request_outcome  # noqa: E402
from headroom.proxy.project_context import (  # noqa: E402
    classify_project,
    get_current_project,
    set_current_project,
    split_project_path,
    with_project_prefix,
)
from headroom.proxy.savings_tracker import (  # noqa: E402
    DEFAULT_MAX_PROJECTS,
    SavingsTracker,
    sanitize_project_name,
)
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

# ---------------------------------------------------------------------------
# sanitize_project_name / classify_project
# ---------------------------------------------------------------------------


def test_sanitize_project_name_normalizes_and_caps():
    assert sanitize_project_name("  api-server  ") == "api-server"
    assert sanitize_project_name("a" * 300) == "a" * 128
    assert sanitize_project_name("x\x00\x1by") == "xy"
    assert sanitize_project_name("") is None
    assert sanitize_project_name("   ") is None
    assert sanitize_project_name(None) is None
    assert sanitize_project_name(42) is None


def test_sanitize_project_name_decodes_percent_encoded_non_ascii():
    """Percent-encoded non-ASCII cwd names (issue #1069) must decode to Unicode."""
    import urllib.parse

    chinese = "第二大脑共享"
    encoded = urllib.parse.quote(chinese, safe="-_.() ")
    assert sanitize_project_name(encoded) == chinese

    mixed = "test-中文-项目"
    encoded_mixed = urllib.parse.quote(mixed, safe="-_.() ")
    assert sanitize_project_name(encoded_mixed) == mixed

    # Plain ASCII names must still pass through unchanged.
    assert sanitize_project_name("my-project") == "my-project"


def test_classify_project_reads_header_case_insensitively():
    assert classify_project({"x-headroom-project": "frontend"}) == "frontend"
    assert classify_project({"X-Headroom-Project": " frontend "}) == "frontend"
    assert classify_project({"user-agent": "claude-code/1.0"}) is None
    assert classify_project(object()) is None


def test_split_project_path_extracts_and_strips():
    assert split_project_path("/p/frontend/v1/messages") == ("frontend", "/v1/messages")
    assert split_project_path("/p/my%20repo/v1/chat/completions") == (
        "my repo",
        "/v1/chat/completions",
    )
    assert split_project_path("/p/frontend") == ("frontend", "/")
    # No prefix / unusable name: path passes through untouched.
    assert split_project_path("/v1/messages") == (None, "/v1/messages")
    assert split_project_path("/p//v1/messages") == (None, "/p//v1/messages")
    assert split_project_path("/p/%20%20/v1") == (None, "/p/%20%20/v1")


def test_with_project_prefix_round_trips_through_split():
    url = with_project_prefix("http://127.0.0.1:8787/v1", "my repo")
    assert url == "http://127.0.0.1:8787/p/my%20repo/v1"
    path = url.removeprefix("http://127.0.0.1:8787")
    assert split_project_path(path) == ("my repo", "/v1")

    # Bare host (anthropic-style base) and unusable names.
    assert with_project_prefix("http://127.0.0.1:8787", "api") == "http://127.0.0.1:8787/p/api"
    assert with_project_prefix("http://127.0.0.1:8787/v1", "  ") == "http://127.0.0.1:8787/v1"
    assert with_project_prefix("http://127.0.0.1:8787/v1", None) == "http://127.0.0.1:8787/v1"


def test_project_contextvar_roundtrip():
    set_current_project("  demo  ")
    assert get_current_project() == "demo"
    set_current_project(None)
    assert get_current_project() is None


# ---------------------------------------------------------------------------
# SavingsTracker per-project aggregation
# ---------------------------------------------------------------------------


def test_tracker_accumulates_per_project_and_persists(tmp_path):
    path = tmp_path / "savings.json"
    tracker = SavingsTracker(path=str(path))

    tracker.record_request(model="gpt-4o", input_tokens=1000, tokens_saved=400, project="api")
    tracker.record_request(model="gpt-4o", input_tokens=500, tokens_saved=100, project="api")
    tracker.record_request(model="gpt-4o", input_tokens=200, tokens_saved=50, project="web")
    tracker.record_request(model="gpt-4o", input_tokens=99, tokens_saved=9)  # unattributed

    projects = tracker.stats_preview()["projects"]
    assert list(projects) == ["api", "web"]  # sorted by tokens saved desc
    assert projects["api"]["requests"] == 2
    assert projects["api"]["tokens_saved"] == 500
    assert projects["api"]["total_input_tokens"] == 1500
    assert projects["api"]["savings_percent"] == pytest.approx(25.0)
    assert projects["web"]["requests"] == 1
    assert projects["api"]["last_activity_at"] is not None

    # Unattributed traffic still lands in the lifetime totals.
    assert tracker.stats_preview()["lifetime"]["requests"] == 4

    # Survives a restart via the persisted JSON state.
    reloaded = SavingsTracker(path=str(path))
    assert reloaded.stats_preview()["projects"]["api"]["tokens_saved"] == 500


def test_tracker_migrates_v2_state_without_projects(tmp_path):
    path = tmp_path / "savings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "lifetime": {
                    "requests": 3,
                    "tokens_saved": 77,
                    "compression_savings_usd": 0.1,
                    "total_input_tokens": 500,
                    "total_input_cost_usd": 0.2,
                },
                "display_session": None,
                "history": [],
            }
        )
    )
    tracker = SavingsTracker(path=str(path))
    preview = tracker.stats_preview()
    assert preview["projects"] == {}
    assert preview["lifetime"]["tokens_saved"] == 77


def test_tracker_caps_project_cardinality(tmp_path):
    tracker = SavingsTracker(path=str(tmp_path / "savings.json"))
    for i in range(DEFAULT_MAX_PROJECTS + 5):
        tracker.record_request(
            model="gpt-4o",
            input_tokens=10,
            tokens_saved=i + 1,
            project=f"proj-{i:03d}",
        )
    projects = tracker.stats_preview()["projects"]
    assert len(projects) == DEFAULT_MAX_PROJECTS
    # The smallest buckets were evicted; the biggest savers survive.
    assert "proj-000" not in projects
    assert f"proj-{DEFAULT_MAX_PROJECTS + 4:03d}" in projects


def test_tracker_sanitizes_persisted_project_state(tmp_path):
    path = tmp_path / "savings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "lifetime": {},
                "display_session": None,
                "history": [],
                "projects": {
                    "ok": {"requests": "2", "tokens_saved": 10},
                    "": {"requests": 1},
                    "bad-entry": "not-a-dict",
                },
            }
        )
    )
    projects = SavingsTracker(path=str(path)).stats_preview()["projects"]
    assert set(projects) == {"ok"}
    assert projects["ok"]["requests"] == 2
    assert projects["ok"]["tokens_saved"] == 10
    assert projects["ok"]["compression_savings_usd"] == 0.0


def test_tracker_caps_persisted_projects_on_load(tmp_path):
    path = tmp_path / "savings.json"
    oversized = {
        f"proj-{i:03d}": {"requests": 1, "tokens_saved": i}
        for i in range(DEFAULT_MAX_PROJECTS + 10)
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "lifetime": {},
                "display_session": None,
                "history": [],
                "projects": oversized,
            }
        )
    )
    projects = SavingsTracker(path=str(path)).stats_preview()["projects"]
    assert len(projects) == DEFAULT_MAX_PROJECTS
    # Lowest tokens_saved entries are dropped, highest kept.
    assert "proj-000" not in projects
    assert f"proj-{DEFAULT_MAX_PROJECTS + 9:03d}" in projects


# ---------------------------------------------------------------------------
# End-to-end: outcome funnel -> tracker -> /stats payload
# ---------------------------------------------------------------------------


def _emit_outcome(proxy, *, project_field=None):
    outcome = RequestOutcome(
        request_id="req-1",
        provider="openai",
        model="gpt-4o",
        original_tokens=1000,
        optimized_tokens=600,
        output_tokens=20,
        tokens_saved=400,
        attempted_input_tokens=1000,
        project=project_field,
    )
    asyncio.run(emit_request_outcome(proxy, outcome))


def test_funnel_attributes_savings_from_context_and_stats_exposes_them(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "savings.json"))
    config = ProxyConfig(cache_enabled=False, rate_limit_enabled=False, log_requests=False)

    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy

        set_current_project("ctx-project")
        try:
            _emit_outcome(proxy)
        finally:
            set_current_project(None)

        # Explicit outcome.project wins over the bound context.
        _emit_outcome(proxy, project_field="field-project")

        stats = client.get("/stats").json()
        per_project = stats["savings"]["per_project"]
        assert per_project["ctx-project"]["tokens_saved"] == 400
        assert per_project["field-project"]["tokens_saved"] == 400
        assert stats["persistent_savings"]["projects"] == per_project
        assert stats["persistent_savings"]["projects_limit"] == DEFAULT_MAX_PROJECTS

        history = client.get("/stats-history").json()
        assert history["schema_version"] == 3
        assert history["projects"]["ctx-project"]["requests"] == 1


# ---------------------------------------------------------------------------
# Regression: pre-feature behavior must be unchanged
# ---------------------------------------------------------------------------


def test_record_request_without_project_matches_legacy_totals(tmp_path):
    """No-header traffic produces exactly the pre-v3 aggregates."""
    path = tmp_path / "savings.json"
    tracker = SavingsTracker(path=str(path))
    tracker.record_request(model="gpt-4o", input_tokens=100, tokens_saved=40)
    tracker.record_request(model="gpt-4o", input_tokens=200, tokens_saved=60)

    preview = tracker.stats_preview()
    assert preview["projects"] == {}
    assert preview["lifetime"]["requests"] == 2
    assert preview["lifetime"]["tokens_saved"] == 100
    assert preview["display_session"]["tokens_saved"] == 100

    persisted = json.loads(path.read_text())
    # Every legacy top-level key survives alongside the new projects map.
    assert set(persisted) >= {"schema_version", "lifetime", "display_session", "history"}
    assert persisted["projects"] == {}


def test_stats_payload_keeps_legacy_shape(tmp_path, monkeypatch):
    """Dashboard consumers of the old /stats keys must not break."""
    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "savings.json"))
    config = ProxyConfig(cache_enabled=False, rate_limit_enabled=False, log_requests=False)

    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy
        _emit_outcome(proxy)  # unattributed: no header, no context, no field

        stats = client.get("/stats").json()
        assert stats["savings"]["per_project"] == {}
        for legacy_key in ("requests", "savings", "persistent_savings", "cost"):
            assert legacy_key in stats, f"legacy /stats key {legacy_key!r} disappeared"
        assert stats["persistent_savings"]["lifetime"]["requests"] == 1

        history = client.get("/stats-history").json()
        for legacy_key in ("schema_version", "lifetime", "display_session", "retention"):
            assert legacy_key in history, f"legacy /stats-history key {legacy_key!r} disappeared"


def test_metrics_record_request_works_without_project_kwarg(tmp_path, monkeypatch):
    """Existing callers that never pass ``project=`` keep working."""
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
            )
        )
        preview = proxy.metrics.savings_tracker.stats_preview()
        assert preview["lifetime"]["tokens_saved"] == 30
        assert preview["projects"] == {}


def test_middleware_binds_project_header_to_context(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "savings.json"))
    config = ProxyConfig(cache_enabled=False, rate_limit_enabled=False, log_requests=False)

    captured: list[str | None] = []

    import headroom.proxy.server as server_module

    def _capture(project: str | None) -> None:
        captured.append(project)
        set_current_project(project)

    monkeypatch.setattr(server_module, "set_current_project", _capture)

    with TestClient(create_app(config)) as client:
        assert client.get("/health", headers={"X-Headroom-Project": " my repo "}).status_code == 200
        assert client.get("/health").status_code == 200
        # /p/<name> base-URL prefix (aider/copilot/cursor wraps): stripped
        # before routing, so the request still reaches /health.
        assert client.get("/p/my%20repo/health").status_code == 200
        # An explicit header wins over the path prefix.
        assert (
            client.get(
                "/p/prefix-project/health", headers={"X-Headroom-Project": "header-project"}
            ).status_code
            == 200
        )

    assert captured == ["my repo", None, "my repo", "header-project"]
