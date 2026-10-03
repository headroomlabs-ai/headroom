"""01-F16: a caller's bearer must never become the account a usage poller polls.

On a shared proxy, the subscription tracker used to adopt *any* caller's OAuth
bearer as the account to poll, and the Codex ``/wham/usage`` refresher polled
with any caller's bearer + ChatGPT account id. The result was published on the
operator's dashboard and the caller's credential was spent on a request they
never made. Now only a bearer from the local operator (direct loopback, not
forwarded) may be adopted; otherwise the operator-configured credential is used.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from starlette.datastructures import Headers

from headroom.subscription import codex_rate_limits
from headroom.subscription.credential_policy import is_local_operator_connection
from headroom.subscription.tracker import SubscriptionTracker


def _conn(host: str | None, headers: dict[str, str] | Headers | None = None) -> SimpleNamespace:
    client = SimpleNamespace(host=host) if host is not None else None
    return SimpleNamespace(client=client, headers=headers or {})


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
def test_direct_loopback_caller_is_the_local_operator(host: str) -> None:
    assert is_local_operator_connection(_conn(host)) is True


@pytest.mark.parametrize(
    "conn",
    [
        _conn("10.0.0.7"),
        _conn("192.168.1.20"),
        _conn(None),
        _conn("127.0.0.1", {"x-forwarded-for": "203.0.113.9"}),
        _conn("127.0.0.1", {"forwarded": "for=203.0.113.9"}),
        _conn("127.0.0.1", {"x-real-ip": "203.0.113.9"}),
        _conn("127.0.0.1", {"x-forwarded-proto": "https"}),
        _conn("127.0.0.1", {"x-forwarded-host": "public.example"}),
        _conn("127.0.0.1", {"X-Forwarded-Port": "443"}),
        _conn("127.0.0.1", Headers({"X-Forwarded-Proto": "https"})),
    ],
    ids=[
        "lan",
        "lan2",
        "no-peer",
        "xff",
        "forwarded",
        "x-real-ip",
        "xfp-only",
        "xfh-only",
        "xf-port",
        "starlette-xfp",
    ],
)
def test_network_forwarded_or_unknown_callers_are_not(conn) -> None:  # noqa: ANN001
    assert is_local_operator_connection(conn) is False


# ---------------------------------------------------------------------------
# Subscription tracker
# ---------------------------------------------------------------------------


def _tracker(tmp_path) -> SubscriptionTracker:  # noqa: ANN001
    return SubscriptionTracker(persist_path=tmp_path / "sub.json")


def test_network_caller_marks_activity_but_is_never_adopted(tmp_path) -> None:  # noqa: ANN001
    tracker = _tracker(tmp_path)
    tracker.notify_active("Bearer tenant-b-oauth-token")
    assert tracker._current_token is None
    assert tracker.is_active() is True


def test_local_operator_bearer_is_adopted(tmp_path) -> None:  # noqa: ANN001
    tracker = _tracker(tmp_path)
    tracker.notify_active("Bearer operator-oauth-token", from_local_operator=True)
    assert tracker._current_token == "operator-oauth-token"


def test_foreign_bearer_never_reaches_the_usage_poll(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "no-claude"))
    tracker = _tracker(tmp_path)
    tracker.notify_active("Bearer tenant-b-oauth-token")  # network caller
    polled: list[str | None] = []

    async def fetch(token: str | None):  # noqa: ANN202
        polled.append(token)
        return None

    tracker._client.fetch = fetch  # type: ignore[method-assign]
    asyncio.run(tracker._maybe_poll())
    # The poll ran (the caller marked activity) but with no adopted token, so
    # the client falls back to the operator's own credential.
    assert polled == [None]


# ---------------------------------------------------------------------------
# Codex /wham/usage refresher
# ---------------------------------------------------------------------------


CODEX_HEADERS = {"authorization": "Bearer tenant-b.jwt", "chatgpt-account-id": "acct-b"}


def test_codex_poll_is_not_driven_by_a_network_caller(monkeypatch) -> None:  # noqa: ANN001
    fetched: list[dict] = []

    async def _fake_fetch(url, headers):  # noqa: ANN001
        fetched.append(headers)

    monkeypatch.setattr(codex_rate_limits, "_fetch_and_store_usage", _fake_fetch)
    codex_rate_limits.get_codex_rate_limit_state()._last_poll_monotonic = 0.0

    async def run() -> bool:
        scheduled = codex_rate_limits.maybe_schedule_usage_poll(CODEX_HEADERS, min_interval_s=0.0)
        await asyncio.sleep(0)
        return scheduled

    assert asyncio.run(run()) is False
    assert fetched == []


# ---------------------------------------------------------------------------
# End to end through the real Anthropic handler
# ---------------------------------------------------------------------------


def _drive_handler(client_host: str, monkeypatch, tmp_path) -> SubscriptionTracker:  # noqa: ANN001
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app
    from headroom.subscription import tracker as tracker_mod

    tracker = _tracker(tmp_path)
    monkeypatch.setattr(tracker_mod, "get_subscription_tracker", lambda: tracker)
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
            image_optimize=False,
        )
    )
    with TestClient(app, client=(client_host, 50000)) as client:

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            return httpx.Response(
                200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        client.app.state.proxy._retry_request = _fake_retry
        resp = client.post(
            "/v1/messages",
            headers={
                "authorization": "Bearer tenant-oauth-token",
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-haiku-4-5",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200, resp.text
    return tracker


def test_handler_never_adopts_a_network_callers_bearer(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    tracker = _drive_handler("127.0.0.1", monkeypatch, tmp_path)
    # Sanity: a direct loopback caller is still adopted (single-developer case).
    assert tracker._current_token == "tenant-oauth-token"

    monkeypatch.setenv("HEADROOM_ALLOW_UNAUTHENTICATED_BIND", "1")
    tracker = _drive_handler("10.1.2.3", monkeypatch, tmp_path / "net")
    assert tracker._current_token is None
