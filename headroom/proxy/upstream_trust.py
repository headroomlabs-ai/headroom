"""Which upstreams are allowed to receive the operator's *own* credentials.

``x-headroom-base-url`` lets a client pick the upstream for a single request, so
OpenAI-compatible gateways (LiteLLM, Azure, self-hosted vLLM) route through the
dedicated handlers instead of the generic passthrough. That is a deliberate
feature and this module does not take it away.

What it takes away is the credential that used to ride along. ``*_extra_headers``
is operator-configured, marked ``secret=True`` in the settings store, and its own
help text suggests an API key as the example value. It was merged into the
upstream-bound header set *before* the destination was resolved, so a request
carrying ``X-Headroom-Base-Url: https://attacker.example`` reached the attacker's
host with the operator's gateway key attached — one request, no user interaction,
from anything able to talk to the proxy port.

The rule here is the one ``copilot_auth.is_copilot_upstream_url`` already applies
to Headroom's own Copilot token, generalized: **a secret only travels to a host
the operator designated.** Designated means one of

* the origin of a resolved provider API target (``ANTHROPIC_TARGET_API_URL``,
  ``OPENAI_TARGET_API_URL``, and the Gemini/Vertex/Cloud Code equivalents), or
* an entry in ``HEADROOM_UPSTREAM_ALLOWED_HOSTS`` (comma-separated bare hosts,
  ``host:port`` or full URLs).

Anything else still gets proxied — the request is not blocked — it just does not
get the operator's headers.

Matching is on the parsed origin (scheme, hostname, port), never the URL string:
comparing whole strings lets ``https://api.anthropic.com@evil.example`` and
``https://api.anthropic.com.evil.example`` through, and a base URL matches while
base+path does not. Scheme and port are part of the match, so ``http://`` or another
port of a designated host is not designated. A designation without a scheme means
https, and one without a port means the scheme's default port. Exact hostname
equality only; no wildcards, because a suffix rule that forgets the label boundary
is the usual way this class of check fails open.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import urlparse

from headroom.config import resolve_omlx_target_api_url

logger = logging.getLogger("headroom.proxy")

ALLOWED_HOSTS_ENV = "HEADROOM_UPSTREAM_ALLOWED_HOSTS"

#: Config attributes holding an operator-designated upstream.
_API_URL_ATTRS = (
    "anthropic_api_url",
    "openai_api_url",
    "gemini_api_url",
    "cloudcode_api_url",
    "vertex_api_url",
    "bedrock_api_url",
    "omlx_target_api_url",
)

# Hosts that are always operator-designated: they are what the provider targets
# resolve to when nothing is overridden, so omitting them would refuse the
# headers on a completely default install.
_DEFAULT_HOSTS = frozenset(
    {
        "api.anthropic.com",
        "api.openai.com",
    }
)

# Warn once per origin rather than once per request; a client looping on a
# rejected host would otherwise flood the log.
_warned_origins: set[str] = set()


def url_host(value: str | None) -> str | None:
    """Return the lowercase hostname for ``value``, tolerating a missing scheme.

    ``urlparse("api.example.com/v1").hostname`` is ``None`` — the whole value is
    read as a path — so a scheme-less configured URL would otherwise contribute
    nothing to the trusted set and silently widen or narrow the check.
    """

    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    parsed = urlparse(candidate)
    if not parsed.hostname and "//" not in candidate:
        parsed = urlparse(f"//{candidate}")
    host = parsed.hostname
    return host.lower() if host else None


_DEFAULT_PORTS = {"http": 80, "https": 443}

Origin = tuple[str, str, int]


def _origin(value: str | None) -> Origin | None:
    """Return ``(scheme, host, port)`` for ``value``, or ``None`` without a usable host.

    No scheme means https and no port means the scheme's default, so a bare
    ``llm.corp`` designates ``https://llm.corp:443`` and ``llm.corp:8443``
    designates ``https://llm.corp:8443``.
    """

    candidate = (value or "").strip()
    if not candidate:
        return None
    try:
        parsed = urlparse(candidate)
        if not parsed.hostname and "//" not in candidate:
            parsed = urlparse(f"//{candidate}")
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        # A malformed port or bracketed host.
        return None
    if not host:
        return None
    scheme = (parsed.scheme or "https").lower()
    return scheme, host.lower(), port if port is not None else _DEFAULT_PORTS.get(scheme, 0)


def trusted_upstream_origins(config: Any = None) -> frozenset[Origin]:
    """Origins (scheme, host, port) permitted to receive operator-configured secret headers."""

    origins: set[Origin] = {("https", host, 443) for host in _DEFAULT_HOSTS}
    candidates = [getattr(config, attr, None) for attr in _API_URL_ATTRS]
    candidates.append(resolve_omlx_target_api_url(environ=os.environ))
    candidates.extend(os.environ.get(ALLOWED_HOSTS_ENV, "").split(","))
    for candidate in candidates:
        origin = _origin(candidate) if isinstance(candidate, str) else None
        if origin:
            origins.add(origin)
    return frozenset(origins)


def is_trusted_upstream(url: str | None, config: Any = None) -> bool:
    """True when ``url`` is on an origin the operator designated.

    ``None``/empty means "no per-request override" — the handler is going to the
    configured target — so it is trusted. ``http://api.openai.com`` and
    ``https://api.openai.com:8443`` are not ``https://api.openai.com``.
    """

    if not url:
        return True
    origin = _origin(url)
    if not origin:
        # Unparseable destination: refuse rather than guess.
        return False
    return origin in trusted_upstream_origins(config)


def warn_untrusted_once(url: str | None, *, request_id: str | None = None) -> None:
    """Log the refusal once per origin, with the remedy in the message."""

    origin = _origin(url)
    host = f"{origin[0]}://{origin[1]}:{origin[2]}" if origin else "<unparseable>"
    if host in _warned_origins:
        return
    _warned_origins.add(host)
    prefix = f"[{request_id}] " if request_id else ""
    logger.warning(
        "%supstream_extra_headers_withheld host=%s reason=not_operator_designated. "
        "The configured extra headers are secret and were NOT sent to this host. "
        "If this upstream is legitimate, add it to %s (comma-separated; a bare host "
        "means https on port 443, otherwise give host:port or the full URL).",
        prefix,
        host,
        ALLOWED_HOSTS_ENV,
    )


def reset_warning_state() -> None:
    """Test hook: clear the once-per-host warning memo."""

    _warned_origins.clear()
