"""``/stats`` withholds from network callers what ``/stats-lifetime`` and ``/quota`` withhold."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app
from headroom.subscription.base import QuotaTracker, get_quota_registry

PROJECT = "private-project-name"
LOGIN = "private-account-login"


class _FakeTracker(QuotaTracker):
    @property
    def key(self) -> str:
        return "fake_quota"

    @property
    def label(self) -> str:
        return "Fake"

    def is_available(self) -> bool:
        return True

    def get_stats(self) -> dict[str, Any] | None:
        return {"latest": {"login": LOGIN, "plan": "business"}}


def _app():
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
            stateless=True,
        )
    )
    app.state.proxy.metrics.savings_tracker.record_request(
        model="gpt-4.1", input_tokens=100, tokens_saved=40, project=PROJECT
    )
    return app


@contextmanager
def _client(app, base_url: str, peer: str):
    # Startup resets the quota registry, so the stand-in tracker goes in afterwards.
    with TestClient(app, base_url=base_url, client=(peer, 12345)) as client:
        get_quota_registry().register(_FakeTracker())
        yield client


@pytest.mark.parametrize("path", ["/stats", "/stats?cached=1"])
def test_stats_withholds_project_names_and_quota_stats_from_network_callers(path) -> None:
    with _client(_app(), "http://203.0.113.5", "203.0.113.9") as client:
        response = client.get(path)
    assert response.status_code == 200
    payload = response.json()
    assert "summary" in payload  # aggregate counters stay public
    assert PROJECT not in response.text
    assert LOGIN not in response.text


def test_loopback_still_sees_them_after_a_network_caller_primed_the_cache() -> None:
    app = _app()
    with _client(app, "http://203.0.113.5", "203.0.113.9") as net:
        net.get("/stats?cached=1")  # primes the shared snapshot
        body = (
            TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345))
            .get("/stats?cached=1")
            .text
        )
    assert PROJECT in body
    assert LOGIN in body


@pytest.mark.parametrize("path", ["/stats", "/stats?cached=1"])
def test_token_holder_still_sees_them(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    monkeypatch.setenv("HEADROOM_PROXY_TOKEN", "s3cret-token")
    monkeypatch.setenv("HEADROOM_HOST", "0.0.0.0")
    with _client(_app(), "http://headroom", "10.0.4.7") as client:
        body = client.get(path, headers={"x-headroom-proxy-token": "s3cret-token"}).text
    assert PROJECT in body
    assert LOGIN in body
