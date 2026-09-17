from __future__ import annotations

from types import SimpleNamespace

import pytest

from headroom.dashboard import get_dashboard_html


class _StatsStub:
    def __init__(self, calls: dict[str, int], key: str, payload: dict):
        self._calls = calls
        self._key = key
        self._payload = payload

    def get_stats(self) -> dict:
        self._calls[self._key] += 1
        return dict(self._payload)


class _ToinStub:
    def get_stats(self) -> dict:
        return {"patterns": 0}


@pytest.fixture(autouse=True)
def _stub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_REQUIRE_RUST_CORE", "false")


def test_stats_cached_query_reuses_short_ttl_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import headroom.proxy.server as server
    from headroom.proxy.server import ProxyConfig, create_app

    calls = {"store": 0, "telemetry": 0, "feedback": 0}
    now = {"value": 100.0}

    monkeypatch.setattr(server.time, "monotonic", lambda: now["value"])
    monkeypatch.setattr(
        server,
        "get_compression_store",
        lambda: _StatsStub(calls, "store", {"entry_count": 1, "max_entries": 100}),
    )
    monkeypatch.setattr(
        server,
        "get_telemetry_collector",
        lambda: _StatsStub(calls, "telemetry", {"enabled": True}),
    )
    monkeypatch.setattr(
        server,
        "get_compression_feedback",
        lambda: _StatsStub(calls, "feedback", {}),
    )

    monkeypatch.setattr(server, "get_toin", lambda: _ToinStub())

    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
        )
    )

    with TestClient(app) as client:
        first = client.get("/stats?cached=1")
        second = client.get("/stats?cached=1")
        now["value"] += 5.1
        third = client.get("/stats?cached=1")
        uncached = client.get("/stats")

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 200
    assert uncached.status_code == 200

    assert calls == {"store": 3, "telemetry": 3, "feedback": 3}
    assert first.json()["tokens"]["proxy_compression_saved"] == 0
    # The retired CLI context tools must leave no trace in the payload.
    payload = first.json()
    assert "context_tool" not in payload
    assert "cli_filtering" not in payload
    assert not any("rtk" in key or "lean_ctx" in key for key in payload["tokens"])


def test_session_summary_surfaces_codex_ws_counters() -> None:
    from headroom.proxy.cost import build_session_summary

    proxy = SimpleNamespace(
        config=SimpleNamespace(mode="token"),
        logger=SimpleNamespace(_logs=[]),
        cost_tracker=SimpleNamespace(stats=lambda: {}),
    )
    metrics = SimpleNamespace(
        requests_by_model={},
        tokens_saved_total=0,
        codex_ws_units_total=12,
        codex_ws_units_modified_total=9,
        codex_ws_unit_tokens_saved_sum=4321,
    )

    payload = build_session_summary(proxy, metrics, {}, total_tokens_before=0)

    assert payload["codex_ws"] == {
        "units_total": 12,
        "units_modified": 9,
        "tokens_saved": 4321,
    }


def test_stats_reset_clears_runtime_proxy_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import headroom.proxy.server as server
    from headroom.proxy.loopback_guard import require_loopback
    from headroom.proxy.server import ProxyConfig, create_app

    monkeypatch.setattr(
        server,
        "get_compression_store",
        lambda: _StatsStub({"store": 0}, "store", {}),
    )
    monkeypatch.setattr(
        server,
        "get_telemetry_collector",
        lambda: _StatsStub({"telemetry": 0}, "telemetry", {}),
    )
    monkeypatch.setattr(
        server,
        "get_compression_feedback",
        lambda: _StatsStub({"feedback": 0}, "feedback", {}),
    )
    monkeypatch.setattr(server, "get_toin", lambda: _ToinStub())

    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
        )
    )
    app.dependency_overrides[require_loopback] = lambda: None

    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy.metrics.tokens_saved_total = 123
        proxy.metrics.tokens_input_total = 456
        proxy.metrics.requests_total = 2

        before = client.get("/stats").json()
        reset = client.post("/stats/reset")
        after = client.get("/stats").json()

    assert before["tokens"]["proxy_compression_saved"] == 123
    assert reset.status_code == 200
    assert after["tokens"]["proxy_compression_saved"] == 0
    assert after["tokens"]["input"] == 0
    assert after["requests"]["total"] == 0


def test_dashboard_uses_cached_stats_and_lazy_history_feed_polling() -> None:
    html = get_dashboard_html()

    assert "fetch('/stats?cached=1')" in html
    assert "version: 'loading'" in html
    assert 'x-text="formatVersion(version)"' in html
    assert "return /^\\d+\\.\\d+\\.\\d+$/.test(label)" in html
    assert "return /^\\d/.test(value)" not in html
    assert "this.version = health.version || 'unknown'" in html
    assert "0.3.0" not in html
    assert "@click=\"setViewMode('history')\"" in html
    assert '@click="toggleFeed()"' in html
    assert "this.viewMode === 'history'" in html
    assert "this.feedOpen" in html
    # The retired CLI context tools left no panel, label or getter behind.
    for gone in (
        "CLI Filtering (rtk)",
        "RTK Filtered",
        "|| 'RTK'",
        "rtkShareOfTotal",
        "Lean-ctx",
        "Context Tool",
        "cliFiltering",
        "cli_filtering",
    ):
        assert gone not in html, f"dashboard still references {gone!r}"


def test_dashboard_session_metrics_do_not_repeat_proxy_tokens_without_new_context() -> None:
    html = get_dashboard_html()

    assert "proxy tokens removed" not in html
    assert "'Proxy ' + formatNumber(stats.tokens?.proxy_compression_saved" not in html
    assert "formatNumber(stats.tokens?.saved || 0) + ' tokens total'" not in html
    assert '<span class="text-sm text-gray-400">Headroom Overhead</span>' not in html
    assert '<span class="text-sm text-gray-400">TTFB (upstream)</span>' not in html
    assert "Overhead Range" in html
    assert "TTFB Range" in html
    assert "Proxy Removed" in html


def test_proxy_throughput_in_stats_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that the /stats endpoint includes a 'throughput' key in the response.

    The server's _compute_throughput closure does a fresh
    `from headroom.perf.analyzer import ...` on every call, so we patch the
    names directly on the `headroom.perf.analyzer` module so the local import
    inside the closure picks up our fakes.

    Skipped locally when headroom._core (Rust extension) is not compiled.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import headroom.perf.analyzer as _analyzer_mod

    try:
        from headroom.proxy.server import (
            _throughput_cache,
            create_app,
            require_loopback,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"headroom._core not available (Rust extension not compiled): {exc}")

    from headroom.config import ProxyConfig

    # Reset the module-level cache so CI doesn't reuse a stale value
    _throughput_cache.update({"expires_at": 0.0, "value": None})

    # Patch at the module level so the local import inside _compute_throughput
    # picks up our stubs instead of the real implementations.
    monkeypatch.setattr(
        _analyzer_mod,
        "parse_log_files",
        lambda last_n_hours=1.0: _analyzer_mod.PerfReport(),
    )
    monkeypatch.setattr(
        _analyzer_mod,
        "build_perf_summary",
        lambda report: {"throughput": {"input_wall_clock": 99.0}},
    )

    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
        )
    )
    app.dependency_overrides[require_loopback] = lambda: None

    with TestClient(app) as client:
        response = client.get("/stats")

    assert response.status_code == 200
    payload = response.json()
    assert "throughput" in payload
    assert payload["throughput"] == {"input_wall_clock": 99.0}
