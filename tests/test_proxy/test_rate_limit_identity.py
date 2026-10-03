"""Rate-limit identity (VAPT 01-F3)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from headroom.proxy.rate_limit_identity import rate_limit_identity


def _request(peer: str, *, authenticated: bool = False, headers=None):
    return SimpleNamespace(
        client=SimpleNamespace(host=peer),
        headers=headers or {},
        state=SimpleNamespace(proxy_authenticated=authenticated),
    )


# ── identity ────────────────────────────────────────────────────────────────


def test_untrusted_identity_ignores_the_credential() -> None:
    a = rate_limit_identity(_request("203.0.113.9", headers={"authorization": "Bearer k1"}))
    b = rate_limit_identity(_request("203.0.113.9", headers={"x-api-key": "k2"}))
    c = rate_limit_identity(_request("203.0.113.9"))
    assert a == b == c == "peer:203.0.113.9"


def test_trusted_identity_is_per_credential_and_owned_by_the_peer() -> None:
    a = rate_limit_identity(
        _request("203.0.113.9", authenticated=True, headers={"authorization": "Bearer k1"})
    )
    b = rate_limit_identity(
        _request("203.0.113.9", authenticated=True, headers={"authorization": "Bearer k2"})
    )
    assert a != b
    assert a.startswith("peer:203.0.113.9|cred:") and b.startswith("peer:203.0.113.9|cred:")


@pytest.mark.parametrize(
    "headers",
    [
        {"authorization": "Bearer sk-ant-api03-SAME-PREFIX-A"},
        {"x-api-key": "sk-ant-api03-SAME-PREFIX-A"},
        {"x-goog-api-key": "AIzaSyD-SAME-PREFIX-A"},
    ],
)
def test_trusted_identity_uses_the_whole_credential_not_a_prefix(headers) -> None:
    other = {k: v[:-1] + "B" for k, v in headers.items()}
    a = rate_limit_identity(_request("127.0.0.1", headers=headers))
    b = rate_limit_identity(_request("127.0.0.1", headers=other))
    assert a != b


def test_bucket_names_carry_no_credential_material() -> None:
    ident = rate_limit_identity(
        _request("127.0.0.1", headers={"authorization": "Bearer sk-live-secret-value"})
    )
    assert "sk-live" not in ident
    assert "secret" not in ident


def test_loopback_is_trusted_without_a_token() -> None:
    keyed = {"authorization": "Bearer k1"}
    assert rate_limit_identity(_request("127.0.0.1", headers=keyed)) != rate_limit_identity(
        _request("127.0.0.1")
    )


def test_trusted_gateway_peer_alone_does_not_authenticate_the_caller(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS", "10.20.0.0/16")
    keyed = {"authorization": "Bearer k1"}
    assert rate_limit_identity(_request("10.20.3.4", headers=keyed)) == "peer:10.20.3.4"
    assert "|cred:" in rate_limit_identity(_request("10.20.3.4", authenticated=True, headers=keyed))


def test_loopback_gateway_does_not_authenticate_forwarded_callers(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS", "127.0.0.0/8")
    keyed = {"authorization": "Bearer k1", "x-forwarded-for": "203.0.113.9"}
    assert rate_limit_identity(_request("127.0.0.1", headers=keyed)) == "peer:203.0.113.9"


def test_loopback_gateway_config_keeps_direct_loopback_callers_per_credential(
    monkeypatch,
) -> None:
    """A loopback gateway CIDR only matters for requests that forward a client address."""
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS", "127.0.0.0/8")
    a = rate_limit_identity(_request("127.0.0.1", headers={"authorization": "Bearer k1"}))
    b = rate_limit_identity(_request("127.0.0.1", headers={"authorization": "Bearer k2"}))
    assert a != b
    assert a.startswith("peer:127.0.0.1|cred:") and b.startswith("peer:127.0.0.1|cred:")


def test_unusable_forwarded_address_still_counts_as_relayed(monkeypatch) -> None:
    """A gateway-relayed request with an empty leftmost hop is not a direct caller."""
    monkeypatch.setenv("HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS", "127.0.0.0/8")
    keyed = {"authorization": "Bearer k1", "x-forwarded-for": ", 203.0.113.9"}
    assert "|cred:" not in rate_limit_identity(_request("127.0.0.1", headers=keyed))


def test_ipv6_peers_are_grouped_by_slash_64() -> None:
    a = rate_limit_identity(_request("2001:db8:1:2::1"))
    b = rate_limit_identity(_request("2001:db8:1:2:ffff:ffff:ffff:ffff"))
    c = rate_limit_identity(_request("2001:db8:1:3::1"))
    assert a == b
    assert a != c


def test_ipv4_mapped_ipv6_is_the_ipv4_peer() -> None:
    assert rate_limit_identity(_request("::ffff:203.0.113.9")) == rate_limit_identity(
        _request("203.0.113.9")
    )
