"""Runtime helpers for Factory Droid integration.

``headroom wrap droid`` routes Droid through Headroom by pointing Droid's
Factory gateway at the local proxy via the ``FACTORY_API_BASE_URL`` environment
variable. The proxy compresses the Anthropic-shaped ``/api/llm/a/v1/messages``
inference route and forwards every other Factory ``/api/*`` path verbatim to
the real upstream resolved here. OpenAI-shaped ``/api/llm/o/*`` traffic is
forwarded uncompressed for now.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
from urllib.parse import urlsplit, urlunsplit

DEFAULT_FACTORY_API_URL = "https://api.factory.ai"
_DNS_HOST_RE = re.compile(
    r"(?=.{1,253}\.?$)"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.?"
)


def _is_valid_host(host: str) -> bool:
    """Return whether `host` is an IP literal or a syntactically valid DNS name."""
    try:
        ipaddress.ip_address(host.rstrip("."))
    except ValueError:
        return _DNS_HOST_RE.fullmatch(host) is not None
    return True


def _is_self_host(host: str) -> bool:
    """Return whether `host` names this machine (loopback or the unspecified address).

    Forwarding there would loop the proxy back to itself when a leftover proxy
    address sits in ``FACTORY_API_BASE_URL``.
    """
    normalized = host.rstrip(".").lower()
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        try:
            address = ipaddress.ip_address(socket.inet_aton(normalized))
        except OSError:
            return False
    return address.is_loopback or address.is_unspecified


def canonical_factory_api_url(value: object) -> str | None:
    """Return a strict Factory upstream URL, or None when unsafe or malformed.

    Only for resolving and validating the upstream; running-vs-requested proxy
    comparisons use the wrap module's shared URL normaliser.
    """
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except ValueError:
        return None
    scheme = parsed.scheme.lower()
    host = parsed.hostname
    if (
        scheme not in {"http", "https"}
        or not host
        or not _is_valid_host(host)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or _is_self_host(host)
    ):
        return None

    host = host.lower()
    rendered_host = f"[{host}]" if ":" in host else host
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        rendered_host = f"{rendered_host}:{port}"
    return urlunsplit((scheme, rendered_host, parsed.path.rstrip("/"), "", ""))


def proxy_base_url(port: int) -> str:
    """Return the local Headroom base URL Droid targets via FACTORY_API_BASE_URL."""
    return f"http://127.0.0.1:{port}"


def resolve_factory_upstream(explicit: str | None = None) -> str:
    """Resolve the real Factory upstream the proxy forwards to.

    Precedence: an explicit ``--factory-api-url``, then the caller's existing
    ``FACTORY_API_BASE_URL`` (an enterprise or EU gateway they already use),
    then the public default.
    """
    candidate = explicit or os.environ.get("FACTORY_API_BASE_URL") or DEFAULT_FACTORY_API_URL
    return candidate.strip().rstrip("/")
