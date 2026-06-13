from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from headroom.proxy import server
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app


class FakeRequestLogger:
    def __init__(self) -> None:
        self._logs: list[dict[str, object]] = []

    @property
    def logs(self) -> list[dict[str, object]]:
        return self._logs

    @logs.setter
    def logs(self, value: list[dict[str, object]]) -> None:
        self._logs = value

    def get_recent(self, limit: int) -> list[dict[str, object]]:
        return self._logs[-limit:]


class FakeLogEntry(dict[str, object]):
    def __getattr__(self, name: str) -> object:
        return self.get(name)


def test_stats_refreshes_recent_requests_when_cached() -> None:
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
            http2=False,
        )
    )
    logger = FakeRequestLogger()
    app.state.proxy.logger = logger

    first_log = FakeLogEntry(
        {
            "timestamp": "2026-06-11T10:00:00Z",
            "provider": "openai",
            "model": "gpt-4.1",
            "input_tokens_original": 100,
            "input_tokens_optimized": 60,
            "tokens_saved": 40,
            "savings_percent": 40.0,
        }
    )
    second_log = FakeLogEntry(
        {
            "timestamp": "2026-06-11T10:01:00Z",
            "provider": "anthropic",
            "model": "claude-sonnet",
            "input_tokens_original": 200,
            "input_tokens_optimized": 120,
            "tokens_saved": 80,
            "savings_percent": 40.0,
        }
    )

    with TestClient(app) as client:
        logger.logs = [first_log]
        first_response = client.get("/stats?cached=1")
        assert first_response.status_code == 200
        assert first_response.json()["recent_requests"][-1]["model"] == "gpt-4.1"

        logger.logs = [first_log, second_log]
        second_response = client.get("/stats?cached=1")
        assert second_response.status_code == 200
        second_payload = second_response.json()

    assert second_payload["recent_requests"][-1]["model"] == "claude-sonnet"
    assert second_payload["request_logs"][-1]["model"] == "claude-sonnet"


def test_agent_usage_totals_use_proxy_only_savings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_REQUIRE_RUST_CORE", "false")
    monkeypatch.setattr(
        server,
        "_get_context_tool_stats",
        lambda: {
            "tool": "rtk",
            "label": "RTK",
            "tokens_saved": 500,
            "session": {},
            "lifetime": {},
        },
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
            http2=False,
        )
    )
    logger = FakeRequestLogger()
    app.state.proxy.logger = logger

    logger.logs = [
        FakeLogEntry(
            {
                "timestamp": "2026-06-11T10:00:00Z",
                "provider": "openai",
                "model": "gpt-5.2-codex",
                "tags": {"client": "codex"},
                "input_tokens_original": 1000,
                "input_tokens_optimized": 900,
                "output_tokens": 50,
                "tokens_saved": 100,
                "savings_percent": 10.0,
            }
        )
    ]

    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy.metrics.tokens_input_total = 900
        proxy.metrics.tokens_saved_total = 100
        proxy.metrics.tokens_output_total = 50

        response = client.get("/stats")

    assert response.status_code == 200
    payload = response.json()

    assert payload["tokens"]["saved"] == 600
    assert payload["agent_usage"]["totals"]["before_tokens"] == 1000
    assert payload["agent_usage"]["totals"]["tokens_saved"] == 100
    assert payload["agent_usage"]["totals"]["savings_percent"] == 10.0
    assert payload["agent_usage"]["agents"][0]["share_of_saved_percent"] == 100.0


def test_stats_preserves_default_smart_crusher_compaction_state() -> None:
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    client = TestClient(create_app(config))

    response = client.get("/stats")

    assert response.status_code == 200
    assert response.json()["config"]["smart_crusher_with_compaction"] is None
