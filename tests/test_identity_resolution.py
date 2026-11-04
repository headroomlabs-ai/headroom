"""Tests for memory-partition identity resolution (WEB-02)."""

from __future__ import annotations

import pytest

from headroom.proxy import identity
from headroom.proxy.identity import resolve_memory_identity, set_identity_resolver


class _FakeRequest:
    def __init__(self, headers: dict[str, str], host: str | None) -> None:
        self.headers = headers
        self.client = type("_Client", (), {"host": host})() if host is not None else None


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HEADROOM_PROXY_TOKEN", raising=False)
    set_identity_resolver(None)
    yield
    set_identity_resolver(None)


def test_loopback_trusts_header() -> None:
    req = _FakeRequest({"x-headroom-user-id": "alice"}, "127.0.0.1")
    assert resolve_memory_identity(req) == "alice"


def test_non_loopback_ignores_header() -> None:
    req = _FakeRequest({"x-headroom-user-id": "victim@corp"}, "10.0.0.5")
    assert resolve_memory_identity(req, default="me") == "me"


def test_missing_peer_metadata_does_not_trust_header() -> None:
    req = _FakeRequest({"x-headroom-user-id": "victim@corp"}, None)
    assert resolve_memory_identity(req, default="me") == "me"


def test_non_loopback_binds_to_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_PROXY_TOKEN", "s3cret")
    req = _FakeRequest({"x-headroom-user-id": "victim@corp"}, "10.0.0.5")
    got = resolve_memory_identity(req, default="me")
    assert got.startswith("tok_")
    assert got not in {"victim@corp", "me"}


def test_legacy_user_id_allowlist_cannot_authorize_remote_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_USER_ID_ALLOWLIST", "alice,bob")
    assert (
        resolve_memory_identity(
            _FakeRequest({"x-headroom-user-id": "alice"}, "10.0.0.5"), default="me"
        )
        == "me"
    )


def test_custom_resolver_wins() -> None:
    set_identity_resolver(lambda request, *, default: "tenant-42")
    req = _FakeRequest({"x-headroom-user-id": "whatever"}, "10.0.0.5")
    assert resolve_memory_identity(req) == "tenant-42"


def test_no_header_uses_default() -> None:
    assert resolve_memory_identity(_FakeRequest({}, "127.0.0.1"), default="") == ""
    assert identity._default_os_user()  # never empty
