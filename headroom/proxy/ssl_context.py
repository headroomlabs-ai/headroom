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


def _relax_x509_strict_for_custom_ca(ctx: ssl.SSLContext, *, path: str) -> ssl.SSLContext:
    """Relax OpenSSL strict-mode checks for an operator-provided CA bundle.

    Python 3.13 / newer OpenSSL can reject some enterprise or private PKI
    roots that platform TLS stacks accept, for example roots without a
    keyUsage extension. Clearing only ``VERIFY_X509_STRICT`` keeps certificate
    verification, hostname verification, expiry checks, and chain validation
    enabled while making custom CA bundles usable in those environments.
    """
    strict_flag = getattr(ssl, "VERIFY_X509_STRICT", 0)
    if strict_flag and ctx.verify_flags & strict_flag:
        ctx.verify_flags &= ~strict_flag
        logger.info("event=ssl_x509_strict_disabled_for_custom_ca path=%s", path)
    return ctx


def _replacement_ca_context(path: str) -> ssl.SSLContext:
    """Build a replacement trust-store context from a CA bundle path."""
    ctx = ssl.create_default_context(cafile=path)
    ctx.set_alpn_protocols(["h2", "http/1.1"])
    return _relax_x509_strict_for_custom_ca(ctx, path=path)


def _additive_ca_context(path: str) -> ssl.SSLContext:
    """Build an additive trust-store context from a CA bundle path."""
    ctx = ssl.create_default_context()
    ctx.load_verify_locations(cafile=path)
    ctx.set_alpn_protocols(["h2", "http/1.1"])
    return _relax_x509_strict_for_custom_ca(ctx, path=path)


def find_ca_bundle() -> ssl.SSLContext | None:
    """Return a CA verification target for httpx's ``verify=`` parameter.

    ``SSL_CERT_FILE`` and ``REQUESTS_CA_BUNDLE`` use **replacement**
    semantics: the returned context trusts that bundle as its trust store.

    ``NODE_EXTRA_CA_CERTS`` uses **additive** semantics (matching Node.js):
    the returned context contains the default/system roots *plus* the extra
    certificate, so public upstreams stay reachable when the extra bundle
    contains only a private/internal root.

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
            return _replacement_ca_context(path)
        if path and not os.path.isfile(path):
            logger.warning(
                "event=ssl_ca_bundle_missing env_var=%s path=%r (skipped)",
                var,
                path,
            )

    node_path = os.environ.get("NODE_EXTRA_CA_CERTS")
    if node_path and os.path.isfile(node_path):
        logger.info(
            "event=ssl_ca_bundle_loaded env_var=NODE_EXTRA_CA_CERTS path=%s additive=true",
            node_path,
        )
        return _additive_ca_context(node_path)
    if node_path and not os.path.isfile(node_path):
        logger.warning(
            "event=ssl_ca_bundle_missing env_var=NODE_EXTRA_CA_CERTS path=%r (skipped)",
            node_path,
        )

    return None
