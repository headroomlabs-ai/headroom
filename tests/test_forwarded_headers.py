"""Tests for ``headroom.proxy.forwarded_headers`` — Phase F PR-F4.

Threat model: a malicious upstream client can forge any
``X-Forwarded-*`` header. The proxy must trust them ONLY when the
connecting peer's IP is in the configured CIDR allow-list. Default
(``HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS`` unset / empty) is
strict-secure: every forwarded header is ignored.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.datastructures import Headers, State

from headroom.proxy.forwarded_headers import (
    TRUSTED_GATEWAY_CIDRS_ENV,
    load_trusted_gateway_cidrs,
    peer_is_trusted_gateway,
    resolve_client_ip,
    trusted_forwarded_headers,
)

# ──────────────────────────────────────────────────────────────────
# Fake-request helper
# ──────────────────────────────────────────────────────────────────


def _fake_request(
    *,
    peer_host: str | None,
    forwarded_for: str | None = None,
    forwarded_proto: str | None = None,
    forwarded_host: str | None = None,
) -> Any:
    """Build a minimal duck-typed ``Request`` stand-in.

    Avoids spinning up a TestClient — we only need ``client.host``,
    ``headers``, and ``state`` for these helpers.
    """
    raw_headers: list[tuple[bytes, bytes]] = []
    if forwarded_for is not None:
        raw_headers.append((b"x-forwarded-for", forwarded_for.encode("latin-1")))
    if forwarded_proto is not None:
        raw_headers.append((b"x-forwarded-proto", forwarded_proto.encode("latin-1")))
    if forwarded_host is not None:
        raw_headers.append((b"x-forwarded-host", forwarded_host.encode("latin-1")))
    headers = Headers(raw=raw_headers)
    client = None if peer_host is None else SimpleNamespace(host=peer_host)
    return SimpleNamespace(client=client, headers=headers, state=State())


# ──────────────────────────────────────────────────────────────────
# CIDR parsing
# ──────────────────────────────────────────────────────────────────


def test_load_cidrs_unset_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TRUSTED_GATEWAY_CIDRS_ENV, raising=False)
    assert load_trusted_gateway_cidrs() == ()


def test_load_cidrs_empty_string_is_empty() -> None:
    assert load_trusted_gateway_cidrs("") == ()
    assert load_trusted_gateway_cidrs("   ") == ()


def test_load_cidrs_single() -> None:
    cidrs = load_trusted_gateway_cidrs("10.0.0.0/8")
    assert len(cidrs) == 1
    assert str(cidrs[0]) == "10.0.0.0/8"


def test_load_cidrs_multiple() -> None:
    cidrs = load_trusted_gateway_cidrs("10.0.0.0/8,172.16.0.0/12,fd00::/8")
    assert [str(c) for c in cidrs] == ["10.0.0.0/8", "172.16.0.0/12", "fd00::/8"]


def test_load_cidrs_whitespace_tolerant() -> None:
    cidrs = load_trusted_gateway_cidrs(" 10.0.0.0/8 , 172.16.0.0/12 ")
    assert [str(c) for c in cidrs] == ["10.0.0.0/8", "172.16.0.0/12"]


def test_load_cidrs_trailing_comma_tolerant() -> None:
    cidrs = load_trusted_gateway_cidrs("10.0.0.0/8,")
    assert [str(c) for c in cidrs] == ["10.0.0.0/8"]


def test_load_cidrs_host_bits_normalized() -> None:
    """``10.0.0.1/8`` is accepted as ``10.0.0.0/8`` (operator-friendly)."""
    cidrs = load_trusted_gateway_cidrs("10.0.0.1/8")
    assert str(cidrs[0]) == "10.0.0.0/8"


def test_load_cidrs_malformed_raises_loud() -> None:
    """Malformed CIDR must raise — silent skip would mask config typos."""
    with pytest.raises(ValueError):
        load_trusted_gateway_cidrs("not-a-cidr")


def test_load_cidrs_partial_malformed_raises_loud() -> None:
    """One bad entry in a multi-CIDR list still raises — we don't degrade."""
    with pytest.raises(ValueError):
        load_trusted_gateway_cidrs("10.0.0.0/8,not-a-cidr,fd00::/8")


def test_load_cidrs_reads_env_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TRUSTED_GATEWAY_CIDRS_ENV, "10.0.0.0/8")
    cidrs = load_trusted_gateway_cidrs()
    assert [str(c) for c in cidrs] == ["10.0.0.0/8"]


# ──────────────────────────────────────────────────────────────────
# Membership check (peer_is_trusted_gateway)
# ──────────────────────────────────────────────────────────────────


def test_peer_membership_empty_allowlist_is_false() -> None:
    assert peer_is_trusted_gateway("10.0.0.5", ()) is False


def test_peer_membership_none_peer_is_false() -> None:
    cidrs = load_trusted_gateway_cidrs("10.0.0.0/8")
    assert peer_is_trusted_gateway(None, cidrs) is False


def test_peer_membership_in_v4_cidr() -> None:
    cidrs = load_trusted_gateway_cidrs("10.0.0.0/8")
    assert peer_is_trusted_gateway("10.0.0.5", cidrs) is True


def test_peer_membership_outside_v4_cidr() -> None:
    cidrs = load_trusted_gateway_cidrs("10.0.0.0/8")
    assert peer_is_trusted_gateway("8.8.8.8", cidrs) is False


def test_peer_membership_in_v6_cidr() -> None:
    """IPv6: ``fd00::1`` ∈ ``fd00::/8`` (allow-list parity test)."""
    cidrs = load_trusted_gateway_cidrs("fd00::/8")
    assert peer_is_trusted_gateway("fd00::1", cidrs) is True


def test_peer_membership_v4_mapped_v6() -> None:
    """``::ffff:10.0.0.1`` resolves to IPv4 for matching."""
    cidrs = load_trusted_gateway_cidrs("10.0.0.0/8")
    assert peer_is_trusted_gateway("::ffff:10.0.0.1", cidrs) is True


def test_peer_membership_v4_not_in_v6_only_cidr() -> None:
    cidrs = load_trusted_gateway_cidrs("fd00::/8")
    assert peer_is_trusted_gateway("10.0.0.5", cidrs) is False


def test_peer_membership_v6_not_in_v4_only_cidr() -> None:
    cidrs = load_trusted_gateway_cidrs("10.0.0.0/8")
    assert peer_is_trusted_gateway("fd00::1", cidrs) is False


def test_peer_membership_evaluates_all_cidrs() -> None:
    cidrs = load_trusted_gateway_cidrs("10.0.0.0/8,172.16.0.0/12,fd00::/8")
    # last-CIDR hit ensures we don't short-circuit early
    assert peer_is_trusted_gateway("172.16.5.5", cidrs) is True
    assert peer_is_trusted_gateway("fd00::beef", cidrs) is True


def test_peer_membership_malformed_peer_is_false() -> None:
    cidrs = load_trusted_gateway_cidrs("10.0.0.0/8")
    assert peer_is_trusted_gateway("not-an-ip", cidrs) is False


# ──────────────────────────────────────────────────────────────────
# resolve_client_ip / trusted_forwarded_headers
# ──────────────────────────────────────────────────────────────────


def test_default_strict_ignores_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env unset → X-Forwarded-* IGNORED even from a 10.x peer."""
    monkeypatch.delenv(TRUSTED_GATEWAY_CIDRS_ENV, raising=False)
    req = _fake_request(
        peer_host="10.0.0.5",
        forwarded_for="203.0.113.7",
        forwarded_proto="https",
        forwarded_host="api.example.com",
    )
    assert resolve_client_ip(req) == "10.0.0.5"
    assert trusted_forwarded_headers(req) == {"for": "", "proto": "", "host": ""}


def test_allowlisted_peer_honors_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TRUSTED_GATEWAY_CIDRS_ENV, "10.0.0.0/8")
    req = _fake_request(
        peer_host="10.0.0.5",
        forwarded_for="203.0.113.7",
        forwarded_proto="https",
        forwarded_host="api.example.com",
    )
    assert resolve_client_ip(req) == "203.0.113.7"
    assert trusted_forwarded_headers(req) == {
        "for": "203.0.113.7",
        "proto": "https",
        "host": "api.example.com",
    }


def test_non_allowlisted_peer_ignores_forwarded_and_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv(TRUSTED_GATEWAY_CIDRS_ENV, "10.0.0.0/8")
    req = _fake_request(
        peer_host="8.8.8.8",  # NOT in 10.0.0.0/8
        forwarded_for="203.0.113.7",
        forwarded_proto="https",
        forwarded_host="api.example.com",
    )
    with caplog.at_level(logging.WARNING, logger="headroom.proxy.forwarded_headers"):
        ip = resolve_client_ip(req)
        fwd = trusted_forwarded_headers(req)
    assert ip == "8.8.8.8"
    assert fwd == {"for": "", "proto": "", "host": ""}
    # Structured rejection event MUST be emitted with full context.
    rejections = [r for r in caplog.records if r.message == "forwarded_headers_rejected"]
    assert len(rejections) == 1, f"expected one rejection event, got {len(rejections)}"
    rec = rejections[0]
    # ``logging.makeLogRecord``-style: we set extras via ``extra=`` kwargs;
    # they end up as attributes on the record.
    assert getattr(rec, "event", None) == "forwarded_headers_rejected"
    assert getattr(rec, "peer_ip", None) == "8.8.8.8"
    assert getattr(rec, "forwarded_for", None) == "203.0.113.7"
    assert getattr(rec, "forwarded_proto", None) == "https"
    assert getattr(rec, "forwarded_host", None) == "api.example.com"


def test_no_forwarded_headers_no_rejection_log(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Direct client (no X-Forwarded-* at all) must NOT spam rejection logs."""
    monkeypatch.setenv(TRUSTED_GATEWAY_CIDRS_ENV, "10.0.0.0/8")
    req = _fake_request(peer_host="8.8.8.8")
    with caplog.at_level(logging.WARNING, logger="headroom.proxy.forwarded_headers"):
        assert resolve_client_ip(req) == "8.8.8.8"
        assert trusted_forwarded_headers(req) == {"for": "", "proto": "", "host": ""}
    rejections = [r for r in caplog.records if r.message == "forwarded_headers_rejected"]
    assert rejections == []


def test_empty_headers_with_allowlisted_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow-listed peer + no X-Forwarded-* headers → empty dict, no error."""
    monkeypatch.setenv(TRUSTED_GATEWAY_CIDRS_ENV, "10.0.0.0/8")
    req = _fake_request(peer_host="10.0.0.5")
    assert resolve_client_ip(req) == "10.0.0.5"  # falls back to peer IP
    assert trusted_forwarded_headers(req) == {"for": "", "proto": "", "host": ""}


def test_ipv6_allowlisted_peer_honors_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TRUSTED_GATEWAY_CIDRS_ENV, "fd00::/8")
    req = _fake_request(
        peer_host="fd00::1",
        forwarded_for="2001:db8::42",
        forwarded_proto="https",
        forwarded_host="api.example.com",
    )
    assert resolve_client_ip(req) == "2001:db8::42"
    assert trusted_forwarded_headers(req) == {
        "for": "2001:db8::42",
        "proto": "https",
        "host": "api.example.com",
    }


def test_ipv4_mapped_v6_peer_honors_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``::ffff:10.0.0.1`` peer matches a v4 allow-list."""
    monkeypatch.setenv(TRUSTED_GATEWAY_CIDRS_ENV, "10.0.0.0/8")
    req = _fake_request(
        peer_host="::ffff:10.0.0.1",
        forwarded_for="203.0.113.7",
    )
    assert resolve_client_ip(req) == "203.0.113.7"


def test_multiple_cidrs_in_env_all_evaluated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TRUSTED_GATEWAY_CIDRS_ENV, "10.0.0.0/8,172.16.0.0/12,fd00::/8")
    for peer in ("10.5.5.5", "172.16.5.5", "fd00::beef"):
        req = _fake_request(peer_host=peer, forwarded_for="203.0.113.7")
        assert resolve_client_ip(req) == "203.0.113.7", f"peer={peer}"


def test_comma_whitespace_tolerance_in_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TRUSTED_GATEWAY_CIDRS_ENV, "10.0.0.0/8 , 172.16.0.0/12")
    for peer in ("10.5.5.5", "172.16.5.5"):
        req = _fake_request(peer_host=peer, forwarded_for="203.0.113.7")
        assert resolve_client_ip(req) == "203.0.113.7", f"peer={peer}"


def test_x_forwarded_for_takes_leftmost(monkeypatch: pytest.MonkeyPatch) -> None:
    """``X-Forwarded-For: client, p1, p2`` → leftmost is the origin."""
    monkeypatch.setenv(TRUSTED_GATEWAY_CIDRS_ENV, "10.0.0.0/8")
    req = _fake_request(
        peer_host="10.0.0.5",
        forwarded_for="203.0.113.7, 10.0.0.99, 10.0.0.5",
    )
    assert resolve_client_ip(req) == "203.0.113.7"


def test_no_client_no_forwarded_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``request.client`` is None (TestClient/UDS): IP is empty string."""
    monkeypatch.delenv(TRUSTED_GATEWAY_CIDRS_ENV, raising=False)
    req = _fake_request(peer_host=None)
    assert resolve_client_ip(req) == ""
    assert trusted_forwarded_headers(req) == {"for": "", "proto": "", "host": ""}


# ──────────────────────────────────────────────────────────────────
# Caching on request.state
# ──────────────────────────────────────────────────────────────────


def test_resolution_cached_on_request_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeat calls within a request must not re-parse the env var.

    We assert behaviourally: poison ``request.state.client_ip`` after the
    first call, then verify the second call returns the cached value
    (proving the second call hit the cache, not the resolver).
    """
    monkeypatch.setenv(TRUSTED_GATEWAY_CIDRS_ENV, "10.0.0.0/8")
    req = _fake_request(peer_host="10.0.0.5", forwarded_for="203.0.113.7")
    first = resolve_client_ip(req)
    assert first == "203.0.113.7"
    # Mutate the cached value; second call should observe it.
    req.state.client_ip = "SENTINEL"
    assert resolve_client_ip(req) == "SENTINEL"


def test_trusted_forwarded_headers_returns_defensive_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutating the returned dict must not corrupt request-state cache."""
    monkeypatch.setenv(TRUSTED_GATEWAY_CIDRS_ENV, "10.0.0.0/8")
    req = _fake_request(
        peer_host="10.0.0.5",
        forwarded_for="203.0.113.7",
        forwarded_proto="https",
        forwarded_host="api.example.com",
    )
    fwd = trusted_forwarded_headers(req)
    fwd["proto"] = "POISONED"
    again = trusted_forwarded_headers(req)
    assert again["proto"] == "https"


# ──────────────────────────────────────────────────────────────────
# Integration with FastAPI Request via TestClient
# ──────────────────────────────────────────────────────────────────


def test_integration_with_real_fastapi_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: a real ``Request`` flows through helpers correctly.

    Uses Starlette's TestClient with a custom ``client=("10.0.0.5", ...)``
    so the peer IP looks like a trusted gateway. Default TestClient
    sets ``request.client.host == "testclient"``, which is not a valid
    IP literal and so always fails the gate (covered separately below).
    """
    monkeypatch.setenv(TRUSTED_GATEWAY_CIDRS_ENV, "10.0.0.0/8")
    app = FastAPI()

    @app.get("/whoami")
    def whoami(request: Request) -> dict[str, Any]:
        return {
            "client_ip": resolve_client_ip(request),
            "forwarded": trusted_forwarded_headers(request),
        }

    client = TestClient(app, client=("10.0.0.5", 50000))
    resp = client.get(
        "/whoami",
        headers={
            "X-Forwarded-For": "203.0.113.42",
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "api.example.com",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["client_ip"] == "203.0.113.42"
    assert body["forwarded"] == {
        "for": "203.0.113.42",
        "proto": "https",
        "host": "api.example.com",
    }


def test_integration_default_strict_ignores_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TRUSTED_GATEWAY_CIDRS_ENV, raising=False)
    app = FastAPI()

    @app.get("/whoami")
    def whoami(request: Request) -> dict[str, Any]:
        return {
            "client_ip": resolve_client_ip(request),
            "forwarded": trusted_forwarded_headers(request),
        }

    client = TestClient(app, client=("10.0.0.5", 50000))
    resp = client.get(
        "/whoami",
        headers={"X-Forwarded-For": "203.0.113.42", "X-Forwarded-Proto": "https"},
    )
    body = resp.json()
    # Env unset → strict-secure: peer IP is the answer, X-Forwarded-* ignored.
    assert body["client_ip"] == "10.0.0.5"
    assert body["forwarded"] == {"for": "", "proto": "", "host": ""}


def test_integration_non_ip_peer_fails_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default TestClient host is the literal ``"testclient"`` — not an IP.

    Even if the operator wrote a 0.0.0.0/0 allow-list (the worst-case
    "trust everyone" config), a non-IP peer literal must still fail
    parsing and the gate must reject it. Belt-and-suspenders.
    """
    monkeypatch.setenv(TRUSTED_GATEWAY_CIDRS_ENV, "0.0.0.0/0")
    app = FastAPI()

    @app.get("/whoami")
    def whoami(request: Request) -> dict[str, Any]:
        return {
            "client_ip": resolve_client_ip(request),
            "forwarded": trusted_forwarded_headers(request),
        }

    client = TestClient(app)
    resp = client.get(
        "/whoami",
        headers={"X-Forwarded-For": "203.0.113.42"},
    )
    body = resp.json()
    assert body["client_ip"] == "testclient"
    assert body["forwarded"] == {"for": "", "proto": "", "host": ""}
