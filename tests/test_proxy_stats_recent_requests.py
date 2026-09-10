from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app
from headroom.rollout import resolve_rollout


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


def test_stats_exposes_actual_running_rollout_snapshot() -> None:
    rollout = resolve_rollout(
        {
            "HEADROOM_ROLLOUT_CHANNEL": "canary",
            "HEADROOM_FEATURES": "tool_result_interceptors",
        }
    )
    app = create_app(
        ProxyConfig(
            rollout=rollout,
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

    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        payload = client.get("/stats").json()["rollout"]

    assert payload == rollout.to_dict()
    assert payload["qualification_eligible"] is True


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
            "savings_breakdown": [
                {
                    "source": "tool_search",
                    "tokens": 40,
                    "usd": 0.0,
                    "realized": True,
                }
            ],
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

    # Loopback client/Host: recent_requests is served only to loopback callers.
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        logger.logs = [first_log]
        first_response = client.get("/stats?cached=1")
        assert first_response.status_code == 200
        assert first_response.json()["recent_requests"][-1]["model"] == "gpt-4.1"
        assert first_response.json()["recent_requests"][-1]["savings_breakdown"] == [
            {
                "source": "tool_search",
                "tokens": 40,
                "usd": 0.0,
                "realized": True,
            }
        ]

        logger.logs = [first_log, second_log]
        second_response = client.get("/stats?cached=1")
        assert second_response.status_code == 200
        second_payload = second_response.json()

    assert second_payload["recent_requests"][0]["model"] == "claude-sonnet"
    assert second_payload["request_logs"][-1]["model"] == "claude-sonnet"


def test_stats_recent_requests_includes_token_incomplete_requests() -> None:
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
                "request_id": "req-haiku-1",
                "timestamp": "2026-07-09T10:00:00Z",
                "provider": "anthropic",
                "model": "claude-haiku",
                "transforms_applied": [],
            }
        ),
        FakeLogEntry(
            {
                "request_id": "req-haiku-2",
                "timestamp": "2026-07-09T10:01:00Z",
                "provider": "anthropic",
                "model": "claude-haiku",
                "input_tokens_original": None,
                "input_tokens_optimized": None,
                "output_tokens": None,
                "tokens_saved": 0,
                "savings_percent": 0.0,
                "transforms_applied": [],
            }
        ),
        FakeLogEntry(
            {
                "request_id": "req-sonnet-1",
                "timestamp": "2026-07-09T10:02:00Z",
                "provider": "anthropic",
                "model": "claude-sonnet",
                "input_tokens_original": 200,
                "input_tokens_optimized": 120,
                "output_tokens": 40,
                "tokens_saved": 80,
                "savings_percent": 40.0,
                "transforms_applied": ["smart_crusher"],
            }
        ),
    ]

    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        response = client.get("/stats")

    assert response.status_code == 200
    payload = response.json()
    assert [req["model"] for req in payload["recent_requests"]] == [
        "claude-sonnet",
        "claude-haiku",
        "claude-haiku",
    ]
    assert payload["recent_requests"][0]["token_accounting_status"] == "complete"
    assert payload["recent_requests"][0]["has_exact_tokens"] is True
    assert payload["recent_requests"][1]["output_tokens"] is None
    assert payload["recent_requests"][1]["token_accounting_status"] == "partial"
    assert payload["recent_requests"][1]["tokens_saved"] == 0
    assert payload["recent_requests"][2]["input_tokens_optimized"] is None
    assert payload["recent_requests"][2]["token_accounting_status"] == "missing"
    assert payload["recent_requests"][2]["has_exact_tokens"] is False
    assert payload["summary"]["uncompressed_requests"]["unknown_token_accounting"] == 2
    assert payload["request_logs"][-1]["model"] == "claude-sonnet"


def test_agent_usage_totals_use_proxy_only_savings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_REQUIRE_RUST_CORE", "false")
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

    assert payload["tokens"]["saved"] == 100
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
    # Loopback client/Host: the `config` block is served only to loopback callers.
    client = TestClient(
        create_app(config), base_url="http://127.0.0.1", client=("127.0.0.1", 12345)
    )

    response = client.get("/stats")

    assert response.status_code == 200
    assert response.json()["config"]["smart_crusher_with_compaction"] is None
