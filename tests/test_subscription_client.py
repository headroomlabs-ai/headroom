from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from headroom.subscription.client import (
    _BETA_HEADER,
    _USAGE_URL,
    SubscriptionClient,
    _credentials_path,
    _load_credentials_file,
    read_cached_oauth_token,
)


class DummyResponse:
    def __init__(self, status_code: int, data: dict | None = None) -> None:
        self.status_code = status_code
        self._data = data or {}

    def json(self) -> dict:
        return self._data


class AsyncClientStub:
    def __init__(
        self,
        *,
        response=None,
        error: Exception | None = None,
        record: dict | None = None,
        timeout=None,
    ):
        self._response = response
        self._error = error
        self._record = record if record is not None else {}
        self._record["timeout"] = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str, headers: dict[str, str]):
        self._record["url"] = url
        self._record["headers"] = headers
        if self._error:
            raise self._error
        return self._response


def test_credentials_path_uses_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    assert _credentials_path() == tmp_path / ".credentials.json"


def test_load_credentials_file_handles_missing_invalid_and_valid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    assert _load_credentials_file() is None

    creds_path = tmp_path / ".credentials.json"
    creds_path.write_text("{invalid", encoding="utf-8")
    assert _load_credentials_file() is None

    payload = {"claudeAiOauth": {"accessToken": "token-from-file"}}
    creds_path.write_text(json.dumps(payload), encoding="utf-8")
    assert _load_credentials_file() == payload


def test_read_cached_oauth_token_prefers_env_and_checks_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", " env-token ")
    monkeypatch.setattr("headroom.subscription.client._load_credentials_file", lambda: None)
    assert read_cached_oauth_token() == "env-token"

    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(
        "headroom.subscription.client._load_credentials_file",
        lambda: {"claudeAiOauth": {"accessToken": "cached-token"}},
    )
    assert read_cached_oauth_token() == "cached-token"

    monkeypatch.setattr(
        "headroom.subscription.client._load_credentials_file",
        lambda: {
            "claudeAiOauth": {
                "accessToken": "expired-token",
                "expiresAt": 59_000,
            }
        },
    )
    monkeypatch.setattr("time.time", lambda: 60)
    assert read_cached_oauth_token() is None

    monkeypatch.setattr(
        "headroom.subscription.client._load_credentials_file",
        lambda: {"claudeAiOauth": {"accessToken": ""}},
    )
    assert read_cached_oauth_token() is None

    monkeypatch.setattr(
        "headroom.subscription.client._load_credentials_file",
        lambda: None,
    )
    assert read_cached_oauth_token() is None


@pytest.mark.asyncio
async def test_subscription_client_fetch_handles_success_and_status_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record: dict = {}
    monkeypatch.setattr(
        "headroom.subscription.client.httpx.AsyncClient",
        lambda timeout: AsyncClientStub(
            response=DummyResponse(200, {"five_hour": {"total": 1}}),
            record=record,
            timeout=timeout,
        ),
    )
    monkeypatch.setattr(
        "headroom.subscription.client.SubscriptionSnapshot.from_api_response",
        lambda data, token="": {"data": data, "token": token},
    )

    client = SubscriptionClient(timeout=3.5)
    result = await client.fetch("  explicit-token  ")
    assert result == {"data": {"five_hour": {"total": 1}}, "token": "explicit-token"}
    assert record["timeout"] == 3.5
    assert record["url"] == _USAGE_URL
    assert record["headers"] == {
        "Authorization": "Bearer explicit-token",
        "anthropic-beta": _BETA_HEADER,
        "Content-Type": "application/json",
    }

    for status_code in (401, 404, 500):
        monkeypatch.setattr(
            "headroom.subscription.client.httpx.AsyncClient",
            lambda timeout, status_code=status_code: AsyncClientStub(
                response=DummyResponse(status_code), timeout=timeout
            ),
        )
        assert await client.fetch("explicit-token") is None


@pytest.mark.asyncio
async def test_subscription_client_fetch_uses_cached_token_and_handles_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SubscriptionClient()

    monkeypatch.setattr("headroom.subscription.client.read_cached_oauth_token", lambda: None)
    assert await client.fetch() is None

    monkeypatch.setattr(
        "headroom.subscription.client.read_cached_oauth_token",
        lambda: "cached-token",
    )
    monkeypatch.setattr(
        "headroom.subscription.client.httpx.AsyncClient",
        lambda timeout: AsyncClientStub(error=httpx.TimeoutException("slow"), timeout=timeout),
    )
    assert await client.fetch() is None

    monkeypatch.setattr(
        "headroom.subscription.client.httpx.AsyncClient",
        lambda timeout: AsyncClientStub(error=RuntimeError("boom"), timeout=timeout),
    )
    assert await client.fetch() is None
