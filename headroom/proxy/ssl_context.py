"""SSL context builder for the Headroom upstream httpx client.

Respects the standard CA-bundle environment variables used by Python
(``SSL_CERT_FILE``), requests (``REQUESTS_CA_BUNDLE``), and Node.js /
Claude Code (``NODE_EXTRA_CA_CERTS``) so that enterprise / corporate
deployments with custom certificate authorities work without extra
configuration.

Priority order (first match wins):
1. ``SSL_CERT_FILE``
2. ``REQUESTS_CA_BUNDLE``
3. ``NODE_EXTRA_CA_CERTS``
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("headroom.proxy")

_CA_BUNDLE_ENV_VARS = (
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
)


def find_ca_bundle() -> str | None:
    """Return the CA bundle path if any CA-bundle env var points to a file.

    Iterates ``SSL_CERT_FILE``, ``REQUESTS_CA_BUNDLE``, and
    ``NODE_EXTRA_CA_CERTS`` in that order.  The first variable that is set
    *and* points to an existing file is returned as a string path so that
    httpx can build its own SSL context (with correct ALPN setup for HTTP/2).

    Returns ``None`` when no env var is set (or all paths are missing),
    which signals to the caller to use httpx's default TLS verification.
    """
    for var in _CA_BUNDLE_ENV_VARS:
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
    return None
