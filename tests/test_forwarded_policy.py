"""Tests for pure trusted-forwarded-header policy."""

from __future__ import annotations

from headroom.proxy.forwarded_policy import (
    ForwardedHeaderInputs,
    header_first,
    parse_cidr_list,
    peer_is_trusted_gateway,
    resolve_forwarded_headers,
)


def test_resolve_trusted_peer_honors_sanitized_forwarded_values() -> None:
    cidrs = parse_cidr_list("10.0.0.0/8")
    result = resolve_forwarded_headers(
        ForwardedHeaderInputs(
            peer_host="10.0.0.5",
            forwarded_for="203.0.113.7, 10.0.0.99",
            forwarded_proto=" https ",
            forwarded_host=" api.example.com ",
        ),
        cidrs,
    )

    assert result.trusted is True
    assert result.rejected is False
    assert result.client_ip == "203.0.113.7"
    assert result.forwarded == {
        "for": "203.0.113.7",
        "proto": "https",
        "host": "api.example.com",
    }


def test_resolve_untrusted_peer_rejects_forwarded_values() -> None:
    cidrs = parse_cidr_list("10.0.0.0/8")
    result = resolve_forwarded_headers(
        ForwardedHeaderInputs(
            peer_host="8.8.8.8",
            forwarded_for="203.0.113.7",
            forwarded_proto="https",
            forwarded_host="api.example.com",
        ),
        cidrs,
    )

    assert result.trusted is False
    assert result.rejected is True
    assert result.client_ip == "8.8.8.8"
    assert result.forwarded == {"for": "", "proto": "", "host": ""}


def test_resolve_direct_untrusted_peer_without_forwarded_values_is_not_rejection() -> None:
    result = resolve_forwarded_headers(ForwardedHeaderInputs(peer_host="8.8.8.8"), ())

    assert result.trusted is False
    assert result.rejected is False
    assert result.client_ip == "8.8.8.8"
    assert result.forwarded == {"for": "", "proto": "", "host": ""}


def test_peer_trust_handles_ipv4_mapped_ipv6() -> None:
    cidrs = parse_cidr_list("10.0.0.0/8")

    assert peer_is_trusted_gateway("::ffff:10.0.0.1", cidrs) is True


def test_header_first_uses_leftmost_forwarded_for_hop() -> None:
    assert header_first("203.0.113.7, 10.0.0.99, 10.0.0.5") == "203.0.113.7"
