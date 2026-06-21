"""SSL context builder for the Headroom upstream httpx client.

Respects the standard CA-bundle environment variables used by Python
(``SSL_CERT_FILE``), requests (``REQUESTS_CA_BUNDLE``), and Node.js /
Claude Code (``NODE_EXTRA_CA_CERTS``) so that enterprise / corporate
deployments with custom certificate authorities work without extra
configuration.

Priority order (first match wins):
1. ``SSL_CERT_FILE``  — replacement semantics (only these CAs are trusted)
2. ``REQUESTS_CA_BUNDLE`` — replacement semantics
3. ``NODE_EXTRA_CA_CERTS`` — **additive** semantics (extra roots loaded
   on top of the default/system trust store, matching Node.js behavior)
"""

from __future__ import annotations

import logging
import os
import ssl

logger = logging.getLogger("headroom.proxy")

_REPLACEMENT_CA_VARS = (
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
)


def find_ca_bundle() -> str | ssl.SSLContext | None:
    """Return a CA verification target for httpx's ``verify=`` parameter.

    ``SSL_CERT_FILE`` and ``REQUESTS_CA_BUNDLE`` use **replacement**
    semantics: the returned path becomes the *only* trust store.

    ``NODE_EXTRA_CA_CERTS`` uses **additive** semantics (matching Node.js):
    an ``ssl.SSLContext`` is returned that contains the default/system
    roots *plus* the extra certificate, so public upstreams stay reachable
    when the extra bundle contains only a private/internal root.

    Returns ``None`` when no env var is set (or all paths are missing),
    which signals to the caller to use httpx's default TLS verification.
    """
    for var in _REPLACEMENT_CA_VARS:
        path = os.environ.get(var)
        if path and os.path.isfile(path):
            logger.info(
                "event=ssl_ca_bundle_loaded env_var=%s path=%s",
                var,
                path,
            )
            return path
        if path and not os.path.isfile(path):
            logger.warning(
                "event=ssl_ca_bundle_missing env_var=%s path=%r (skipped)",
                var,
                path,
            )

    node_path = os.environ.get("NODE_EXTRA_CA_CERTS")
    if node_path and os.path.isfile(node_path):
        ctx = ssl.create_default_context()
        ctx.load_verify_locations(cafile=node_path)
        ctx.set_alpn_protocols(["h2", "http/1.1"])
        logger.info(
            "event=ssl_ca_bundle_loaded env_var=NODE_EXTRA_CA_CERTS path=%s additive=true",
            node_path,
        )
        return ctx
    if node_path and not os.path.isfile(node_path):
        logger.warning(
            "event=ssl_ca_bundle_missing env_var=NODE_EXTRA_CA_CERTS path=%r (skipped)",
            node_path,
        )

    return None
