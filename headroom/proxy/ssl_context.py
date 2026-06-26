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

Strict-mode toggle (``HEADROOM_TLS_STRICT``):
    Python 3.13 + OpenSSL 3.x enable ``VERIFY_X509_STRICT`` by default, which
    enforces RFC 5280 §4.2.1.9 — a CA cert's ``basicConstraints`` MUST be
    marked critical. Corporate TLS-inspection roots (Zscaler, Netskope, …)
    commonly set ``CA:TRUE`` *without* the critical bit, so the chain is
    rejected with ``Basic Constraints of CA cert not marked critical`` even
    though the root is correctly installed and trusted. A CA bundle env var
    can't fix this — the cert is found, it's the strict check that fails.

    Setting ``HEADROOM_TLS_STRICT=0`` clears *only* ``VERIFY_X509_STRICT`` from
    every TLS context Headroom controls (the httpx upstream client AND the
    urllib3/requests stack used by ``huggingface_hub`` for model downloads).
    Chain validation, signature checks, expiry, and hostname verification all
    stay on — this is strictly narrower than ``verify=False``. Default is
    strict (the flag stays set) to match Python's own default.
"""

from __future__ import annotations

import logging
import os
import ssl
from typing import Any, cast

logger = logging.getLogger("headroom.proxy")

_REPLACEMENT_CA_VARS = (
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
)

# Env var that opts out of OpenSSL's RFC 5280 strict CA-constraint checks.
TLS_STRICT_ENV = "HEADROOM_TLS_STRICT"

# Values (case-insensitive) that mean "turn strict mode OFF".
_TLS_STRICT_OFF_VALUES = frozenset({"0", "false", "no", "off"})


def tls_strict_disabled() -> bool:
    """True when ``HEADROOM_TLS_STRICT`` opts out of OpenSSL strict mode.

    Default (unset / any other value) is strict, matching Python 3.13's own
    default. Only the explicit off-values flip it.
    """
    return os.environ.get(TLS_STRICT_ENV, "").strip().lower() in _TLS_STRICT_OFF_VALUES


def _clear_x509_strict(ctx: ssl.SSLContext, *, reason: str) -> ssl.SSLContext:
    """Clear only ``VERIFY_X509_STRICT`` from a context, leaving all else on.

    Keeps certificate verification, hostname verification, expiry checks, and
    chain validation enabled — this is far narrower than disabling verify.
    """
    strict_flag = getattr(ssl, "VERIFY_X509_STRICT", 0)
    if strict_flag and ctx.verify_flags & strict_flag:
        ctx.verify_flags &= ~strict_flag
        logger.info("event=ssl_x509_strict_disabled reason=%s", reason)
    return ctx


def _relax_x509_strict_for_custom_ca(ctx: ssl.SSLContext, *, path: str) -> ssl.SSLContext:
    """Relax OpenSSL strict-mode checks for an operator-provided CA bundle.

    Python 3.13 / newer OpenSSL can reject some enterprise or private PKI
    roots that platform TLS stacks accept, for example roots without a
    keyUsage extension. Clearing only ``VERIFY_X509_STRICT`` keeps certificate
    verification, hostname verification, expiry checks, and chain validation
    enabled while making custom CA bundles usable in those environments.

    A custom CA bundle is itself a strong signal of a corporate PKI, so the
    strict flag is relaxed here regardless of ``HEADROOM_TLS_STRICT`` (the
    historical behavior). The env toggle additionally covers the case where
    the corporate root lives in the *default* trust store and no bundle var
    is set — see :func:`build_httpx_verify`.
    """
    return _clear_x509_strict(ctx, reason=f"custom_ca:{path}")


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


def _default_strict_relaxed_context() -> ssl.SSLContext:
    """Default trust store, but with ``VERIFY_X509_STRICT`` cleared.

    Used when no custom CA bundle is configured (the corporate root lives in
    the OS/default trust store) but ``HEADROOM_TLS_STRICT=0`` asks us to
    tolerate a non-critical ``basicConstraints`` CA. Mirrors what httpx builds
    for ``verify=True`` (default context + ALPN), minus the strict flag.
    """
    ctx = ssl.create_default_context()
    ctx.set_alpn_protocols(["h2", "http/1.1"])
    return _clear_x509_strict(ctx, reason="env_toggle")


def build_httpx_verify() -> ssl.SSLContext | bool:
    """Return the value for httpx's ``verify=`` parameter.

    Resolution order:

    1. A custom CA bundle env var (``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE`` /
       ``NODE_EXTRA_CA_CERTS``) → a context trusting that bundle, with strict
       mode already relaxed (corporate PKI signal).
    2. No bundle, but ``HEADROOM_TLS_STRICT=0`` → the default trust store with
       ``VERIFY_X509_STRICT`` cleared, so a corporate root that's installed in
       the OS store but trips RFC 5280 strict mode still validates.
    3. Otherwise → ``True`` (httpx's default strict verification).

    Returning ``True`` rather than a hand-built context in the common case
    keeps httpx's own default behavior (including its certifi fallback) intact.
    """
    ca_ctx = find_ca_bundle()
    if ca_ctx is not None:
        return ca_ctx
    if tls_strict_disabled():
        return _default_strict_relaxed_context()
    return True


def apply_global_tls_relaxation() -> bool:
    """Strip ``VERIFY_X509_STRICT`` from urllib3's context builder when opted in.

    The proxy's upstream httpx client is handled explicitly via
    :func:`build_httpx_verify`, but model downloads go through
    ``huggingface_hub`` → ``requests`` → ``urllib3``, which builds its own
    context via ``urllib3.util.ssl_.create_urllib3_context`` and sets
    ``VERIFY_X509_STRICT`` independently (urllib3 ≥ 2.5). That path never sees
    our httpx context, so a corporate-MITM user hits the same
    ``Basic Constraints ... not marked critical`` rejection on a model cache
    miss.

    When ``HEADROOM_TLS_STRICT=0`` this monkeypatches
    ``create_urllib3_context`` to clear the strict flag from every context it
    returns. The patch is idempotent (guarded by a sentinel attribute) and a
    no-op when urllib3 isn't importable. Returns True if a patch was applied
    (or was already in place), False otherwise.

    Call this as early as possible — before ``huggingface_hub`` / ``requests``
    import and cache their context — i.e. at CLI startup.
    """
    if not tls_strict_disabled():
        return False

    strict_flag = getattr(ssl, "VERIFY_X509_STRICT", 0)
    if not strict_flag:
        return False

    try:
        import urllib3.util.ssl_ as _u3ssl
    except Exception:  # pragma: no cover - urllib3 always present in practice
        logger.debug("event=ssl_urllib3_patch_skipped reason=import_failed")
        return False

    if getattr(_u3ssl.create_urllib3_context, "_headroom_strict_relaxed", False):
        return True

    _orig = _u3ssl.create_urllib3_context

    def _relaxed_create_urllib3_context(*args: Any, **kwargs: Any) -> ssl.SSLContext:
        # urllib3's create_urllib3_context signature varies across versions;
        # forward verbatim and cast the (Any-typed) result back to SSLContext.
        ctx = cast(ssl.SSLContext, _orig(*args, **kwargs))
        if ctx.verify_flags & strict_flag:
            ctx.verify_flags &= ~strict_flag
        return ctx

    _relaxed_create_urllib3_context._headroom_strict_relaxed = True  # type: ignore[attr-defined]
    _u3ssl.create_urllib3_context = _relaxed_create_urllib3_context  # type: ignore[assignment]
    logger.info("event=ssl_x509_strict_disabled reason=urllib3_global_patch")
    return True
