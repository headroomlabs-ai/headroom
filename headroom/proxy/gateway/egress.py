"""Credential-specific destination authorization."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import socket
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import NoReturn
from urllib.parse import parse_qsl, quote, urlsplit

from headroom.proxy.gateway.config import RouteConfig
from headroom.proxy.gateway.credentials import CredentialLease
from headroom.proxy.gateway.destinations import (
    SOURCE_KINDS,
    https_destination,
    normalized_origin,
    path_within,
    validate_audience,
)
from headroom.proxy.gateway.errors import GatewayEgressDenied

_CLIENT_AUTH_HEADERS = frozenset(
    {
        "authorization",
        "x-api-key",
        "x-goog-api-key",
        "x-headroom-proxy-token",
        "proxy-authorization",
    }
)


@dataclass(frozen=True, slots=True)
class AuthorizedDestination:
    hostname: str
    port: int
    addresses: tuple[str, ...]
    url: str


def resolve_addresses(hostname: str, port: int) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(record[4][0])
            for record in socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        )
    )


class EgressPolicy:
    """Authorize the final URL and resolved addresses before revealing a secret."""

    def __init__(self, resolver: Callable[[str, int], Iterable[str]] | None = None) -> None:
        self._resolver = resolver or resolve_addresses

    def authorize(
        self,
        lease: CredentialLease,
        url: str,
        *,
        resolved_addresses: Iterable[str] | None = None,
        route: RouteConfig | None = None,
    ) -> AuthorizedDestination:
        try:
            parsed = https_destination(url)
            validate_audience(lease.provider, url, project=lease.project, region=lease.region)
            if SOURCE_KINDS.get(lease.provider) != lease.source_kind:
                self._deny()
            origin = normalized_origin(url)
            if origin not in {normalized_origin(value) for value in lease.allowed_origins}:
                self._deny()
            if not any(path_within(parsed.path, prefix) for prefix in lease.allowed_path_prefixes):
                self._deny()
            private = False
            if route is not None:
                if (
                    route.provider != lease.provider
                    or origin != normalized_origin(route.upstream_origin)
                    or not path_within(parsed.path, route.upstream_path_prefix)
                ):
                    self._deny()
                private = (
                    route.private_network
                    and lease.provider == "compatible"
                    and lease.source_kind == "env"
                )
            addresses = (
                tuple(resolved_addresses)
                if resolved_addresses is not None
                else tuple(self._resolver(parsed.hostname or "", parsed.port or 443))
            )
            if not addresses:
                self._deny()
            for raw_address in addresses:
                address = ipaddress.ip_address(raw_address)
                if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
                    address = address.ipv4_mapped
                if (
                    address.is_loopback
                    or address.is_link_local
                    or address.is_multicast
                    or address.is_unspecified
                    or address.is_reserved
                    or (not address.is_global and not (private and address.is_private))
                ):
                    self._deny()
            return AuthorizedDestination(parsed.hostname or "", parsed.port or 443, addresses, url)
        except (ValueError, OSError):
            self._deny()

    @staticmethod
    def _deny() -> NoReturn:
        raise GatewayEgressDenied(
            status_code=502,
            code="gateway_egress_denied",
            message="Upstream destination is not authorized for this credential",
        )


def build_managed_upstream_headers(
    client_headers: Mapping[str, str],
    lease: CredentialLease,
    url: str,
    *,
    resolved_addresses: Iterable[str] | None = None,
    route: RouteConfig | None = None,
    method: str = "GET",
    body: bytes = b"",
) -> dict[str, str]:
    """Strip every caller credential before attaching one authorized lease."""

    EgressPolicy().authorize(lease, url, resolved_addresses=resolved_addresses, route=route)
    result = {
        name.lower(): value
        for name, value in client_headers.items()
        if name.lower() not in _CLIENT_AUTH_HEADERS
        and name.lower() not in {"host", "cookie", "forwarded", "proxy-connection"}
        and not name.lower().startswith(("x-headroom-", "x-forwarded-", "x-amz-", "x-goog-"))
    }
    if lease.provider == "bedrock":
        result.update(_aws_sigv4_headers(lease, method, url, result, body))
    else:
        result.update(lease.authorization_headers())
    return result


def _aws_sigv4_headers(
    lease: CredentialLease,
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes,
) -> dict[str, str]:
    """Sign one Bedrock request from an opaque AWS SDK credential object."""

    credentials = lease.secret._reveal_for_egress()
    access_key = getattr(credentials, "access_key", None)
    secret_key = getattr(credentials, "secret_key", None)
    session_token = getattr(credentials, "token", None)
    if not isinstance(access_key, str) or not isinstance(secret_key, str):
        raise TypeError("AWS credential object is missing signing fields")

    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if parsed.port not in (None, 443):
        host = f"{host}:{parsed.port}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    date = timestamp[:8]
    region = host.removeprefix("bedrock-runtime.").removesuffix(".amazonaws.com")
    payload_hash = hashlib.sha256(body).hexdigest()
    signing_headers = {"host": host, "x-amz-date": timestamp}
    if session_token:
        signing_headers["x-amz-security-token"] = str(session_token)
    content_type = headers.get("content-type")
    if content_type:
        signing_headers["content-type"] = content_type.strip()
    canonical_headers = "".join(
        f"{name}:{signing_headers[name]}\n" for name in sorted(signing_headers)
    )
    signed_headers = ";".join(sorted(signing_headers))
    canonical_query = "&".join(
        f"{quote(key, safe='-_.~')}={quote(value, safe='-_.~')}"
        for key, value in sorted(parse_qsl(parsed.query, keep_blank_values=True))
    )
    canonical_request = "\n".join(
        (
            method.upper(),
            quote(parsed.path or "/", safe="/-_.~"),
            canonical_query,
            canonical_headers,
            signed_headers,
            payload_hash,
        )
    )
    scope = f"{date}/{region}/bedrock/aws4_request"
    string_to_sign = "\n".join(
        (
            "AWS4-HMAC-SHA256",
            timestamp,
            scope,
            hashlib.sha256(canonical_request.encode()).hexdigest(),
        )
    )

    def sign(key: bytes, value: str) -> bytes:
        return hmac.new(key, value.encode(), hashlib.sha256).digest()

    signing_key = sign(
        sign(sign(sign(("AWS4" + secret_key).encode(), date), region), "bedrock"),
        "aws4_request",
    )
    signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()
    result = {
        "authorization": (
            f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        ),
        "host": host,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": timestamp,
    }
    if session_token:
        result["x-amz-security-token"] = str(session_token)
    return result
