"""TLS trust policy for every outbound connection Headroom makes.

Corporate networks (Zscaler, Netskope, Palo Alto, Cisco Secure Access, ...)
inspect TLS by re-signing every certificate with a company root. IT installs
that root in the **operating system** trust store (macOS Keychain, Windows
certificate store), which is why browsers, curl and Claude Code keep working.
Python does not read that store: httpx trusts certifi's bundle and the stdlib
reads a static OpenSSL file, so Headroom used to reject the re-signed chain
and fail every upstream request with ``CERTIFICATE_VERIFY_FAILED``.

Default policy (``HEADROOM_CERT_STORE=system,bundled``, mirroring Claude
Code's ``CLAUDE_CODE_CERT_STORE``): verify through the OS trust store via
``truststore`` *and* trust certifi's public bundle. On macOS and Windows the
OS verifier does the chain check, so a corporate root that IT pushed to the
machine just works and OpenSSL's strict-mode quirks never apply.

Sources, in the order they are resolved:
1. ``SSL_CERT_FILE``  — replacement semantics (only these CAs are trusted)
2. ``REQUESTS_CA_BUNDLE`` — replacement semantics
3. Otherwise the ``HEADROOM_CERT_STORE`` sources (``system``, ``bundled``),
   plus the **additive** bundles ``HEADROOM_CA_BUNDLE`` and
   ``NODE_EXTRA_CA_CERTS`` (extra roots on top, matching Node.js behavior).

``HEADROOM_CERT_STORE=bundled`` restores the pre-0.40 behavior (no OS store).

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
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any, cast

logger = logging.getLogger("headroom.proxy")

CERT_STORE_ENV = "HEADROOM_CERT_STORE"
EXTRA_CA_ENV = "HEADROOM_CA_BUNDLE"

_CERT_STORE_SOURCES = frozenset({"system", "bundled"})
# Spellings people reach for; a typo'd opt-out must not silently re-enable a source.
_CERT_STORE_ALIASES = {
    "bundle": "bundled",
    "certifi": "bundled",
    "os": "system",
    "native": "system",
}
_DEFAULT_CERT_STORE = frozenset({"system", "bundled"})

# Additive bundles, loaded on top of whatever trust store is in force.
_ADDITIVE_CA_VARS = (EXTRA_CA_ENV, "NODE_EXTRA_CA_CERTS")

# Set by Headroom's CLI (and inherited by the proxy processes it spawns) so
# proxy startup only injects process-wide when Headroom owns the process.
PROCESS_TRUST_ENV = "HEADROOM_PROCESS_TRUST"

_process_trust_injected = False

# Built contexts, keyed by ALPN + every env var that shapes them. Building one
# parses certifi's ~140 roots, so it must not run per WebSocket connection.
_system_ctx_cache: dict[tuple[Any, ...], ssl.SSLContext] = {}

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


def _require_tls12(ctx: ssl.SSLContext) -> ssl.SSLContext:
    """Refuse TLS < 1.2 explicitly (Python's default, stated so it cannot regress)."""
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx


def _replacement_ca_context(path: str) -> ssl.SSLContext:
    """Build a replacement trust-store context from a CA bundle path."""
    ctx = _require_tls12(ssl.create_default_context(cafile=path))
    ctx.set_alpn_protocols(["h2", "http/1.1"])
    return _relax_x509_strict_for_custom_ca(ctx, path=path)


def _additive_ca_context(path: str) -> ssl.SSLContext:
    """Build an additive trust-store context from a CA bundle path."""
    ctx = _require_tls12(ssl.create_default_context())
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

    additive = _additive_ca_paths()
    if additive:
        ctx = _additive_ca_context(additive[0])
        for path in additive[1:]:
            ctx.load_verify_locations(cafile=path)
        return ctx

    return None


def _additive_ca_paths(*, log: bool = True) -> list[str]:
    """Existing files named by the additive CA vars (logged unless ``log=False``)."""
    paths: list[str] = []
    for var in _ADDITIVE_CA_VARS:
        path = os.environ.get(var)
        if not path:
            continue
        if not log:
            if os.path.isfile(path):
                paths.append(path)
            continue
        if os.path.isfile(path):
            logger.info(
                "event=ssl_ca_bundle_loaded env_var=%s path=%s additive=true",
                var,
                path,
            )
            paths.append(path)
        else:
            logger.warning(
                "event=ssl_ca_bundle_missing env_var=%s path=%r (skipped)",
                var,
                path,
            )
    return paths


def _replacement_ca_var_set() -> bool:
    """True when ``SSL_CERT_FILE``/``REQUESTS_CA_BUNDLE`` name an existing file."""
    return any(
        (path := os.environ.get(var)) and os.path.isfile(path) for var in _REPLACEMENT_CA_VARS
    )


def cert_store_sources() -> frozenset[str]:
    """Trust sources selected by ``HEADROOM_CERT_STORE`` (default: system,bundled).

    Unknown tokens are ignored with a warning; a value with no known token falls
    back to the default rather than trusting nothing.
    """
    raw = os.environ.get(CERT_STORE_ENV)
    if raw is None or not raw.strip():
        return _DEFAULT_CERT_STORE
    tokens = {
        _CERT_STORE_ALIASES.get(t.strip().lower(), t.strip().lower())
        for t in raw.split(",")
        if t.strip()
    }
    unknown = tokens - _CERT_STORE_SOURCES
    if unknown:
        logger.warning(
            "event=ssl_cert_store_unknown env_var=%s ignored=%s",
            CERT_STORE_ENV,
            ",".join(sorted(unknown)),
        )
    chosen = frozenset(tokens & _CERT_STORE_SOURCES)
    return chosen or _DEFAULT_CERT_STORE


def system_trust_available() -> bool:
    """True when the ``truststore`` package can verify through the OS store."""
    try:
        import truststore  # noqa: F401
    except Exception:
        return False
    return True


def _system_store_enabled() -> bool:
    return (
        "system" in cert_store_sources()
        and not _replacement_ca_var_set()
        and system_trust_available()
    )


def _build_system_context(alpn: list[str]) -> ssl.SSLContext:
    """A context that verifies through the OS trust store. Raises if it cannot.

    ``truststore`` hands chain validation to macOS Security.framework / Windows
    CryptoAPI (OpenSSL + the system bundle on Linux). Certificates loaded with
    ``load_verify_locations`` are trusted *in addition* to the OS roots, so
    certifi (``bundled``) and the additive bundle vars stack on top. Built once
    per configuration and cached.
    """
    key = (
        tuple(alpn),
        os.environ.get(CERT_STORE_ENV),
        *(os.environ.get(var) for var in _ADDITIVE_CA_VARS),
    )
    cached = _system_ctx_cache.get(key)
    if cached is not None:
        return cached
    import truststore

    ctx: ssl.SSLContext = _require_tls12(truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT))
    # Match create_default_context() on 3.13+: an explicitly trusted
    # intermediate (a common "Zscaler Intermediate Root CA" export) is a
    # valid anchor. Only affects the OpenSSL (Linux) backend.
    partial_chain = getattr(ssl, "VERIFY_X509_PARTIAL_CHAIN", 0)
    if partial_chain:
        ctx.verify_flags |= partial_chain
    if "bundled" in cert_store_sources():
        try:
            import certifi

            ctx.load_verify_locations(cafile=certifi.where())
        except Exception:  # certifi is optional; the OS store still applies
            logger.debug("event=ssl_certifi_unavailable")
    for path in _additive_ca_paths():
        ctx.load_verify_locations(cafile=path)
    ctx.set_alpn_protocols(alpn)
    _system_ctx_cache[key] = ctx
    return ctx


def describe_trust_policy() -> dict[str, Any]:
    """What trust configuration is in force, for ``/health`` and ``doctor``."""
    replacement = next(
        (
            {"env_var": var, "path": os.environ[var]}
            for var in _REPLACEMENT_CA_VARS
            if os.environ.get(var) and os.path.isfile(os.environ[var])
        ),
        None,
    )
    return {
        "cert_store": sorted(cert_store_sources()),
        "system_store_available": system_trust_available(),
        "system_store_active": _system_store_enabled(),
        "replacement_bundle": replacement,
        "additive_bundles": [
            {"env_var": var, "path": os.environ[var]}
            for var in _ADDITIVE_CA_VARS
            if os.environ.get(var) and os.path.isfile(os.environ[var])
        ],
        "tls_strict_disabled": tls_strict_disabled(),
        "process_injected": _process_trust_injected,
    }


def _default_strict_relaxed_context() -> ssl.SSLContext:
    """Default trust store, but with ``VERIFY_X509_STRICT`` cleared.

    Used when no custom CA bundle is configured (the corporate root lives in
    the OS/default trust store) but ``HEADROOM_TLS_STRICT=0`` asks us to
    tolerate a non-critical ``basicConstraints`` CA. Mirrors what httpx builds
    for ``verify=True`` (default context + ALPN), minus the strict flag.
    """
    ctx = _require_tls12(ssl.create_default_context())
    ctx.set_alpn_protocols(["h2", "http/1.1"])
    return _clear_x509_strict(ctx, reason="env_toggle")


def _replacement_ca_path() -> str | None:
    """The existing file named by ``SSL_CERT_FILE``/``REQUESTS_CA_BUNDLE``, if any."""
    for var in _REPLACEMENT_CA_VARS:
        path = os.environ.get(var)
        if path and os.path.isfile(path):
            return path
    return None


def _has_configured_trust() -> bool:
    """True when Headroom's trust settings call for something other than a library default."""
    return (
        _system_store_enabled()
        or _replacement_ca_path() is not None
        or bool(_additive_ca_paths(log=False))
        or tls_strict_disabled()
    )


def _build_context(alpn: list[str], fallback: Callable[[], ssl.SSLContext]) -> ssl.SSLContext:
    """Build the verifying context Headroom's trust configuration calls for.

    Every branch returns a concrete, certificate-verifying ``SSLContext``;
    ``fallback`` supplies the library-default equivalent. Resolution order:

    0. The OS trust store (plus certifi and the additive bundle vars) when
       ``HEADROOM_CERT_STORE`` includes ``system`` — the default — and no
       replacement bundle var is set.
    1. ``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE`` → only that bundle, strict
       mode relaxed (corporate PKI signal).
    2. ``HEADROOM_CA_BUNDLE`` / ``NODE_EXTRA_CA_CERTS`` → default roots plus
       those, strict mode relaxed.
    3. ``HEADROOM_TLS_STRICT=0`` → the default trust store with
       ``VERIFY_X509_STRICT`` cleared.
    4. Otherwise ``fallback()``.
    """
    if _system_store_enabled():
        try:
            return _build_system_context(alpn)
        except Exception as exc:
            # Never let the OS-store path take the proxy down: fall back to the
            # bundled behavior and say so.
            logger.warning("event=ssl_system_store_failed error=%r (falling back to bundled)", exc)
    replacement = _replacement_ca_path()
    additive = _additive_ca_paths()
    if replacement is not None:
        logger.info("event=ssl_ca_bundle_loaded path=%s", replacement)
        ctx = _replacement_ca_context(replacement)
    elif additive:
        ctx = _additive_ca_context(additive[0])
        for path in additive[1:]:
            ctx.load_verify_locations(cafile=path)
    elif tls_strict_disabled():
        ctx = _default_strict_relaxed_context()
    else:
        ctx = fallback()
    ctx.set_alpn_protocols(alpn)
    return ctx


def _bundled_default_context() -> ssl.SSLContext:
    """Exactly what httpx builds for ``verify=True``: certifi, strict defaults.

    Built here rather than passing ``True`` so every upstream client receives a
    concrete, always-verifying ``SSLContext``: no code path can hand httpx a
    value that disables verification.
    """
    cert_dir = os.environ.get("SSL_CERT_DIR")
    if cert_dir:
        ctx = ssl.create_default_context(capath=cert_dir)
    else:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    return _require_tls12(ctx)


def _stdlib_default_context() -> ssl.SSLContext:
    """What ``ssl=True`` / urlopen build: the stdlib default trust store."""
    return _require_tls12(ssl.create_default_context())


def build_httpx_verify() -> ssl.SSLContext:
    """Return the value for httpx's ``verify=`` parameter: always a verifying context.

    Headroom's configured trust (see :func:`_build_context`) when there is one,
    else httpx's own default (certifi) built explicitly.
    """
    return _build_context(["h2", "http/1.1"], _bundled_default_context)


def build_urlopen_context() -> ssl.SSLContext | None:
    """Return Headroom's configured TLS context for ``urllib.request.urlopen``.

    ``None`` means "use urlopen's own default" (Python's default trust store),
    which is what the legacy ``HEADROOM_CERT_STORE=bundled`` path wants when no
    bundle or strict-mode toggle is configured. urllib.request/http.client only
    implements HTTP/1.1 framing, so the context only offers ``http/1.1``:
    offering h2 can make a TLS-inspecting MITM negotiate a protocol it cannot
    parse.
    """
    if not _has_configured_trust():
        return None
    return _build_context(["http/1.1"], _stdlib_default_context)


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


_ENV_PROXY_VARS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")


def find_system_proxy() -> str | None:
    """Return a proxy URL from system settings, including macOS System Preferences.

    httpx trust_env=True reads HTTP_PROXY / HTTPS_PROXY env vars but not the
    macOS System Configuration proxy (System Settings → Network → Proxies).
    urllib.request.getproxies() reads both env vars and the system config;
    we only use it when none of the env vars are set, so we never layer a
    system proxy on top of explicit env-based proxy handling.
    """
    if any(os.environ.get(var) for var in _ENV_PROXY_VARS):
        return None  # ponytail: trust_env already handles this path
    proxies = urllib.request.getproxies()
    url = proxies.get("https") or proxies.get("http")
    if url:
        _log_detected_system_proxy(url)
    return url


def _log_detected_system_proxy(url: str) -> None:
    """Log a detected system proxy URL with any userinfo (credentials) redacted.

    Tolerates malformed/hostless URLs -- a garbled PAC or system-proxy entry
    shouldn't crash proxy startup, so falls back to logging scheme only.
    """
    parsed = urllib.parse.urlparse(url)
    if not parsed.hostname:
        logger.info("event=system_proxy_detected scheme=%s", parsed.scheme or "unknown")
        return
    try:
        port = parsed.port
    except ValueError:
        # e.g. "http://proxy.example:notaport" -- ParseResult.port raises
        # rather than returning None for a non-numeric port.
        logger.info("event=system_proxy_detected host=%s", parsed.hostname)
        return
    netloc = parsed.hostname + (f":{port}" if port else "")
    safe = parsed._replace(netloc=netloc)
    logger.info("event=system_proxy_detected url=%s", safe.geturl())


def build_websocket_ssl() -> ssl.SSLContext:
    """Return the value for ``websockets.connect(ssl=...)`` on a ``wss://`` URL.

    WebSocket upgrades are HTTP/1.1, so the context only offers ``http/1.1``.
    Without configured trust it is what ``ssl=True`` would build (the stdlib
    default, which on Windows loads the machine store).
    """
    return _build_context(["http/1.1"], _stdlib_default_context)


def ensure_process_trust() -> bool:
    """Make every TLS client in this *application* process use the OS store.

    Marks the process (``HEADROOM_PROCESS_TRUST=1``) so proxy processes the CLI
    spawns inherit the decision; see :func:`ensure_process_trust_if_owned`.

    Headroom's own upstream clients get an explicit context from the builders
    above, but third-party code (``huggingface_hub`` model downloads, tiktoken
    vocab fetches, OTEL exporters, ...) builds its own. ``truststore``'s
    process-wide injection covers those in one place instead of chasing every
    call site. It is only ever called from Headroom's entry points (CLI and
    proxy startup) — never on library import — because injecting into an
    embedding application's process is not ours to decide.

    Skipped when the operator pinned a replacement bundle (``SSL_CERT_FILE`` /
    ``REQUESTS_CA_BUNDLE``) or opted out via ``HEADROOM_CERT_STORE``.
    Idempotent. Returns True when injection is in place.
    """
    global _process_trust_injected
    os.environ[PROCESS_TRUST_ENV] = "1"
    if _process_trust_injected:
        return True
    if not _system_store_enabled():
        return False
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception as exc:
        logger.warning("event=ssl_process_trust_failed error=%r", exc)
        return False
    _process_trust_injected = True
    logger.info("event=ssl_process_trust_injected source=os_trust_store")
    return True


def ensure_process_trust_if_owned() -> bool:
    """:func:`ensure_process_trust`, but only in a process Headroom's CLI owns.

    Proxy startup calls this: uvicorn workers spawned by ``headroom proxy``
    inherit ``HEADROOM_PROCESS_TRUST=1`` and get injection, while an application
    that embeds the proxy app keeps its own process-wide TLS behavior (its
    Headroom upstream clients still get the OS store via explicit contexts).
    """
    if os.environ.get(PROCESS_TRUST_ENV) != "1":
        return False
    return ensure_process_trust()
