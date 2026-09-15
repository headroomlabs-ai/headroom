"""Runtime helpers for Factory Droid integration.

``headroom wrap droid`` routes Droid through Headroom by pointing Droid's
Factory gateway at the local proxy via the ``FACTORY_API_BASE_URL`` environment
variable. The proxy compresses the Anthropic-shaped ``/api/llm/a/v1/messages``
inference route and forwards every other Factory REST path verbatim to the real
upstream resolved here, so all Droid models (including Droid Core) are
compressed on the user's Factory subscription with no ``customModels`` edits.
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


def _is_loopback_host(host: str) -> bool:
    normalized = host.rstrip(".").lower()
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        pass
    try:
        packed = socket.inet_aton(normalized)
    except OSError:
        return False
    return ipaddress.ip_address(packed).is_loopback


def canonical_factory_api_url(value: object) -> str | None:
    """Return a strict Factory upstream URL, or None when unsafe or malformed."""
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
        or host is None
        or not _is_valid_host(host)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or _is_loopback_host(host)
    ):
        return None

    host = host.lower()
    rendered_host = f"[{host}]" if ":" in host else host
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        rendered_host = f"{rendered_host}:{port}"
    path = parsed.path.rstrip("/")
    return urlunsplit((scheme, rendered_host, path, "", ""))


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
    return candidate.rstrip("/")
