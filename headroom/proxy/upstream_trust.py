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

* a host in the resolved provider API targets (``ANTHROPIC_TARGET_API_URL``,
  ``OPENAI_TARGET_API_URL``, and the Gemini/Vertex/Cloud Code equivalents), or
* a host listed in ``HEADROOM_UPSTREAM_ALLOWED_HOSTS`` (comma-separated).

Anything else still gets proxied — the request is not blocked — it just does not
get the operator's headers.

Matching is on the parsed hostname, never the URL string: comparing whole strings
lets ``https://api.anthropic.com@evil.example`` and ``https://api.anthropic.com.evil.example``
through, and a base URL matches while base+path does not. Exact hostname equality
only; no wildcards, because a suffix rule that forgets the label boundary is the
usual way this class of check fails open.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import urlparse

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

# Warn once per destination rather than once per request; a client looping on a
# rejected host would otherwise flood the log.
_warned_hosts: set[str] = set()


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


def _env_allowed_hosts() -> set[str]:
    raw = os.environ.get(ALLOWED_HOSTS_ENV, "")
    hosts: set[str] = set()
    for entry in raw.split(","):
        # Accept a bare host or a full URL, so operators can paste either.
        host = url_host(entry) if entry.strip() else None
        if host:
            hosts.add(host)
    return hosts


def trusted_upstream_hosts(config: Any = None) -> frozenset[str]:
    """Hosts permitted to receive operator-configured secret headers."""

    hosts = set(_DEFAULT_HOSTS)
    for attr in _API_URL_ATTRS:
        host = url_host(getattr(config, attr, None))
        if host:
            hosts.add(host)
    hosts |= _env_allowed_hosts()
    return frozenset(hosts)


def is_trusted_upstream(url: str | None, config: Any = None) -> bool:
    """True when ``url`` is a destination the operator designated.

    ``None``/empty means "no per-request override" — the handler is going to the
    configured target — so it is trusted.
    """

    if not url:
        return True
    host = url_host(url)
    if not host:
        # Unparseable destination: refuse rather than guess.
        return False
    return host in trusted_upstream_hosts(config)


def warn_untrusted_once(url: str | None, *, request_id: str | None = None) -> None:
    """Log the refusal once per host, with the remedy in the message."""

    host = url_host(url) or "<unparseable>"
    if host in _warned_hosts:
        return
    _warned_hosts.add(host)
    prefix = f"[{request_id}] " if request_id else ""
    logger.warning(
        "%supstream_extra_headers_withheld host=%s reason=not_operator_designated. "
        "The configured extra headers are secret and were NOT sent to this host. "
        "If this upstream is legitimate, add it to %s (comma-separated hosts).",
        prefix,
        host,
        ALLOWED_HOSTS_ENV,
    )


def reset_warning_state() -> None:
    """Test hook: clear the once-per-host warning memo."""

    _warned_hosts.clear()
