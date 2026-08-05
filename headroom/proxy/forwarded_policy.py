"""Pure policy for trusted forwarded headers."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass

Network = ipaddress.IPv4Network | ipaddress.IPv6Network
Address = ipaddress.IPv4Address | ipaddress.IPv6Address


@dataclass(frozen=True)
class ForwardedHeaderInputs:
    """Raw connection and forwarded-header values before trust evaluation."""

    peer_host: str
    forwarded_for: str = ""
    forwarded_proto: str = ""
    forwarded_host: str = ""

    @property
    def has_forwarded_headers(self) -> bool:
        return bool(self.forwarded_for or self.forwarded_proto or self.forwarded_host)


@dataclass(frozen=True)
class ForwardedHeaderResolution:
    """Deterministic forwarded-header trust decision."""

    client_ip: str
    forwarded: dict[str, str]
    trusted: bool
    rejected: bool


def parse_cidr_list(raw: str) -> tuple[Network, ...]:
    """Parse a comma-separated CIDR list. Empty / whitespace returns empty."""
    if not raw or not raw.strip():
        return ()
    nets: list[Network] = []
    for chunk in raw.split(","):
        entry = chunk.strip()
        if not entry:
            continue
        nets.append(ipaddress.ip_network(entry, strict=False))
    return tuple(nets)


def normalize_ip(host: str) -> Address | None:
    """Parse ``host`` as IP, normalizing IPv4-mapped IPv6 to IPv4."""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return None
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    return addr


def peer_is_trusted_gateway(
    peer_host: str | None,
    cidrs: tuple[Network, ...],
) -> bool:
    """Return True iff ``peer_host`` is inside any allow-listed CIDR."""
    if not cidrs or peer_host is None:
        return False
    addr = normalize_ip(peer_host)
    if addr is None:
        return False
    for net in cidrs:
        if isinstance(addr, ipaddress.IPv4Address) and isinstance(net, ipaddress.IPv6Network):
            continue
        if isinstance(addr, ipaddress.IPv6Address) and isinstance(net, ipaddress.IPv4Network):
            continue
        if addr in net:
            return True
    return False


def header_first(value: str) -> str:
    """Return the leftmost element of a comma-separated header value."""
    if not value:
        return ""
    head, _, _ = value.partition(",")
    return head.strip()


def resolve_forwarded_headers(
    inputs: ForwardedHeaderInputs,
    cidrs: tuple[Network, ...],
) -> ForwardedHeaderResolution:
    """Resolve client IP and sanitized forwarded headers from pure inputs."""
    trusted = peer_is_trusted_gateway(inputs.peer_host or None, cidrs)
    if trusted:
        forwarded_for = header_first(inputs.forwarded_for)
        return ForwardedHeaderResolution(
            client_ip=forwarded_for or inputs.peer_host,
            forwarded={
                "for": forwarded_for,
                "proto": inputs.forwarded_proto.strip(),
                "host": inputs.forwarded_host.strip(),
            },
            trusted=True,
            rejected=False,
        )

    return ForwardedHeaderResolution(
        client_ip=inputs.peer_host,
        forwarded={"for": "", "proto": "", "host": ""},
        trusted=False,
        rejected=inputs.has_forwarded_headers,
    )
