"""mitmproxy addon: keep Claude Code's Remote Control while routing through Headroom.

Claude Code v2.1.196+ disables Remote Control whenever ``ANTHROPIC_BASE_URL``
names a host other than ``api.anthropic.com`` (issue #1779). This addon lets
``ANTHROPIC_BASE_URL`` stay unset: Claude Code reaches ``api.anthropic.com``
through ``HTTPS_PROXY`` (mitmproxy), and the addon hands only the Messages API
calls to the local Headroom proxy. Every other request is left alone.

Run it with mitmproxy decrypting ``api.anthropic.com`` only, so claude.ai (the
Remote Control bridge), OAuth and every other host stay a raw tunnel::

    mitmdump --listen-host 127.0.0.1 --listen-port 8790 \
      --allow-hosts '^api\\.anthropic\\.com(:443)?$' \
      -s examples/claude_code_remote_control/route_to_headroom.py

See ``README.md`` next to this file for the Claude Code side.

Environment:
    HEADROOM_RC_UPSTREAM: ``host:port`` of the Headroom proxy
        (default ``127.0.0.1:8787``).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mitmproxy import http

ANTHROPIC_HOST = "api.anthropic.com"
MESSAGES_PATH = "/v1/messages"

# Loop guard. If the Headroom process inherited HTTPS_PROXY (it is often started
# by a Claude Code hook, with Claude Code's environment), its upstream call comes
# back through this proxy. The marker set on the way in tells the two apart. It
# must NOT start with ``x-headroom-``: Headroom strips that prefix before
# forwarding upstream, which would erase the marker and loop forever.
HOP_HEADER = "x-rc-proxy-hop"


def _upstream() -> tuple[str, int]:
    host, _, port = os.environ.get("HEADROOM_RC_UPSTREAM", "127.0.0.1:8787").rpartition(":")
    return host or "127.0.0.1", int(port)


def _is_messages_call(request: http.Request) -> bool:
    return request.pretty_host == ANTHROPIC_HOST and request.path.startswith(MESSAGES_PATH)


def request(flow: http.HTTPFlow) -> None:
    """Send ``/v1/messages*`` to Headroom, once; leave everything else untouched."""
    req = flow.request
    if not _is_messages_call(req):
        return
    if HOP_HEADER in req.headers:
        # Second pass: Headroom is calling upstream through us. Go to Anthropic.
        del req.headers[HOP_HEADER]
        return
    req.headers[HOP_HEADER] = "1"
    req.scheme = "http"
    req.host, req.port = _upstream()


def responseheaders(flow: http.HTTPFlow) -> None:
    """Stream bodies so server-sent events reach Claude Code as they arrive."""
    if flow.response is not None:
        flow.response.stream = True
