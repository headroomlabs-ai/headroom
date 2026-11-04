"""Which credential a background usage poller may use (VAPT 01-F16).

Usage pollers (the Claude subscription tracker, the Codex ``/wham/usage``
refresher) call a provider on the proxy's behalf and show the result on the
operator's dashboard. The account they poll must be the **operator's**, so a
credential *learned from traffic* is adopted only from the **local operator**: a
loopback peer whose request was not forwarded by another hop. On a shared proxy (several principals behind one Headroom), a network caller's
``Authorization`` bearer must never become the polled account — that would spend
the caller's credential on a request they never made and publish their usage on
someone else's dashboard. Without a learned token the pollers fall back to the
operator-configured credential (``CLAUDE_CODE_OAUTH_TOKEN`` or the proxy user's
own Claude Code credentials file), as before.

This keeps the single-developer experience: Claude Code on the same machine,
where the OAuth token often lives in the OS keychain rather than a file, still
gets its subscription window without extra setup.
"""

from __future__ import annotations

from typing import Any

from headroom.proxy.loopback_guard import is_loopback_host

# Any of these means the request crossed another hop (a gateway or reverse
# proxy on the same host), so the loopback peer is not the end caller. The
# whole ``X-Forwarded-*`` family counts (``-For``, ``-Proto``, ``-Host``, and
# any added later), so a gateway that sends only one of them still fails closed.
_FORWARDING_HEADERS = frozenset({"forwarded", "x-real-ip"})
_FORWARDING_HEADER_PREFIX = "x-forwarded-"


def is_local_operator_connection(conn: Any) -> bool:
    """True only for a direct loopback caller (HTTP request or WebSocket).

    Fails closed: a missing peer address, a non-loopback peer, or any
    forwarding header returns ``False``. Trusted-gateway CIDRs deliberately do
    not count — a gateway fronts other principals by definition.
    """
    client = getattr(conn, "client", None)
    host = getattr(client, "host", None) if client is not None else None
    if not isinstance(host, str) or not is_loopback_host(host):
        return False
    headers = getattr(conn, "headers", None)
    if headers is not None:
        try:
            for name in headers.keys():
                name = name.lower()
                if name in _FORWARDING_HEADERS or name.startswith(_FORWARDING_HEADER_PREFIX):
                    return False
        except Exception:
            return False
    return True
