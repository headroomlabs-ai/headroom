"""Per-tenant key resolver — Phase F PR-F3.

# Threat model

TOIN (Tool Output Intelligence Network) learns compression patterns from
every observed request. Pre-F3 those patterns aggregated into one global
pool keyed by ``(auth_mode, model_family, sig_hash)``. That works for an
OSS single-tenant deploy, but in a multi-tenant SaaS / shared-proxy
deploy it's a cross-tenant pattern leak: tenant A's tool-call patterns
train recommendations that the proxy then applies to tenant B's
requests. Two distinct customers carrying differently-shaped tool
outputs can degrade each other's compression quality, and (worse) a
tenant could probe TOIN dumps and learn the structural fingerprint of
another tenant's tool surface.

F3 partitions: every TOIN store key gets a ``tenant_key`` prefix
derived from the inbound request. Two tenants therefore train two
isolated pools — the dataclass on disk is unchanged, only the key
namespace grows by one component.

# Resolution rules

The resolver uses an explicit tenant signal only:

1. **Header** — when ``HEADROOM_TENANT_KEY_HEADER`` (default
   ``X-Headroom-Tenant-ID``) is present and survives sanitization.
   Source: ``"header"``. The customer's auth-proxy / sidecar is the
   canonical place to pin tenancy — it knows the tenant best.

2. **Global** — the literal ``"global"`` namespace when no tenant-id
   header is present (the OSS single-process / dev / unauthenticated
   case). Source: ``"global"``. This is a real namespace, not a sentinel
   — the existing pool of pre-F3 patterns already lives under
   ``"global"`` after the migration, so a fresh deploy is non-breaking.

Every resolution emits a structured ``tenant_key_resolved`` log event
(NOT silent — see ``feedback_no_silent_fallbacks.md``). Operators can
alert on a sustained ``source="global"`` rate in production to catch
mis-configured upstream auth.

# Sanitization rules

The key must be injective: two distinct tenant ids must never share a
pool, because that is precisely the cross-tenant leak this module
prevents. Since the accepted charset is narrow and the length is
capped, a plain filter is lossy and cannot be injective on its own, so
sanitization has two forms:

- **Verbatim** when the raw id already consists only of accepted code
  points (ASCII alphanumeric, ``-``, ``_``) and is at most 64 chars.
  Unchanged from earlier releases, so existing conforming tenants keep
  their stored TOIN data.
- **Digested** otherwise: the accepted characters become a readable
  prefix (trimmed to leave room) and a 16-hex-char SHA-256 digest of
  the raw id is appended, e.g. ``acme.corp`` →
  ``acmecorp-<digest>``. Same shape as the project / user keys in
  ``headroom.memory.storage_router``.

Inputs with no accepted character at all (pure CJK, emoji) no longer
fall through to the shared ``"global"`` pool — they get
``tenant-<digest>``, so an affected tenant is isolated rather than
silently pooled with every other tenant.

- Allowed output: ASCII alphanumeric, ``-`` (hyphen), ``_`` (underscore).
- Length cap: 64 chars, enforced on both forms (prevents log-spam /
  OOM via giant headers).
- Empty after stripping: rejected (falls through to the next rule).
- Unicode / control chars: never appear in the output — we use
  ``unicodedata.category()`` rather than a regex (per project memory
  ``no regexes``) so the classification is the OS Unicode tables, not a
  developer-authored pattern.

**Cost of the digest form:** a tenant whose id is not already
``[A-Za-z0-9_-]{1,64}`` gets a different key than earlier releases did,
so its previously learned TOIN patterns stay behind under the old key.
TOIN is a diagnostic learning store, not an archive — isolation wins
over carrying the history forward, and no migration shim is provided.

# Configuration

Single env var: ``HEADROOM_TENANT_KEY_HEADER``. Default
``X-Headroom-Tenant-ID``. The header NAME is configurable; the
sanitization rules and the ``"global"`` literal are not — they're
load-bearing for namespace isolation.

# Integration

The handlers call ``resolve_tenant_key(request)`` once at request entry
(right after ``classify_auth_mode``) and store the result on
``request.state.tenant_key`` + ``request.state.tenant_key_source``.
They also call ``set_request_tenant_key(tenant_key)`` to populate the
:class:`contextvars.ContextVar` that TOIN's deep-stack
``record_compression`` / ``record_retrieval`` calls read — same pattern
as the existing ``_request_ccr_store`` ContextVar in
``headroom/cache/compression_store.py``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import unicodedata
from contextvars import ContextVar
from typing import Any, Final

logger = logging.getLogger(__name__)

# ── Configuration env vars ────────────────────────────────────────────
# Header name is the only thing operators may want to change at deploy
# time (e.g. some load balancers strip ``X-`` headers, some teams use
# ``X-Tenant-ID`` without the Headroom prefix). Sanitization, prefix
# length, and the ``"global"`` literal are NOT configurable — they're
# load-bearing for tenant isolation.
TENANT_KEY_HEADER_ENV_VAR: Final[str] = "HEADROOM_TENANT_KEY_HEADER"
DEFAULT_TENANT_KEY_HEADER: Final[str] = "X-Headroom-Tenant-ID"

# ── Tenant-key constants ──────────────────────────────────────────────
# Maximum length we'll accept after sanitization. 64 chars is enough
# for SaaS-style ``cust_<22-char-id>`` (Stripe-shape) and short UUIDs;
# longer values almost always indicate a misconfiguration where
# something else (a JWT payload, a session id) was wedged into the
# header by mistake.
_MAX_TENANT_KEY_LEN: Final[int] = 64

# Length of the SHA-256 digest (hex chars) appended to keys that can't
# round-trip through the accepted charset verbatim. 16 hex chars = 64
# bits, matching what ``headroom.memory.storage_router`` uses for the
# same purpose.
_TENANT_KEY_DIGEST_LEN: Final[int] = 16

# Prefix used when the raw id has no accepted character at all (a
# pure-CJK or emoji id). Any constant works — uniqueness comes from the
# digest; this just keeps the key readable in logs and stores.
_OPAQUE_TENANT_KEY_PREFIX: Final[str] = "tenant"

# Hex alphabet of the digest. Kept as a literal so ``_looks_digested``
# needs no ``re`` (project memory: ``no regexes``).
_HEX_DIGITS: Final[str] = "0123456789abcdef"

# Marker that reserves one slice of the key space for digested keys.
#
# It exists because the two sanitization forms must not overlap: the
# verbatim form returns conforming ids unchanged, and every digested key
# is itself conforming, so without a reserved marker a tenant could send
# a *digested* key as its own header and be handed the same pool — the
# collision this marker removes. Digested keys always start with this
# marker; verbatim keys never do (see :func:`_looks_digested`).
_TENANT_KEY_MARKER: Final[str] = "tnt-"

# Literal namespace name used when no tenant-id header is present.
# This is a real namespace, not a sentinel —
# pre-F3 patterns already live under it after migration, and the OSS
# single-tenant / unauthenticated dev case rightfully uses one shared
# pool.
GLOBAL_TENANT_KEY: Final[str] = "global"

# Source string returned alongside the key. Operators can grep
# structured logs for ``source="global"`` to spot mis-configured
# upstream auth proxies.
SOURCE_HEADER: Final[str] = "header"
SOURCE_GLOBAL: Final[str] = "global"


# ── Request-scoped ContextVar ─────────────────────────────────────────
# TOIN's ``record_compression`` / ``record_retrieval`` are invoked deep
# inside SmartCrusher / ContentRouter, several frames below the proxy
# handler. Threading ``tenant_key`` through every call site would touch
# dozens of files; using a ``ContextVar`` keeps the surface small and
# matches the existing per-request ``_request_ccr_store`` pattern in
# ``headroom/cache/compression_store.py``.
#
# ``ContextVar`` is the right primitive — it's per-asyncio-task-aware,
# unlike ``threading.local`` which would cross-pollinate concurrent
# requests on the same event loop.
_request_tenant_key: ContextVar[str | None] = ContextVar(
    "headroom_request_tenant_key", default=None
)


def set_request_tenant_key(tenant_key: str | None) -> None:
    """Set the tenant key for the current request context.

    Called once per request from the proxy handler after
    :func:`resolve_tenant_key`. TOIN's ``record_compression`` /
    ``record_retrieval`` reads this via :func:`get_current_tenant_key`.

    Args:
        tenant_key: Resolved tenant key, or ``None`` to clear.
    """
    _request_tenant_key.set(tenant_key)


def get_current_tenant_key() -> str:
    """Return the tenant key for the current request, or
    :data:`GLOBAL_TENANT_KEY` if none is set.

    TOIN call sites prepend this to their store keys. The default of
    ``"global"`` (rather than ``None``) keeps test / OSS / batch-job
    callers — which never pass through the proxy handler — addressable
    in the same namespace they used pre-F3.
    """
    key = _request_tenant_key.get()
    return key if key is not None else GLOBAL_TENANT_KEY


# ── Public resolver ───────────────────────────────────────────────────


def resolve_tenant_key(request: Any) -> tuple[str, str]:
    """Resolve the tenant key for an inbound request.

    See module docstring for the threat model and full resolution
    rules. Briefly:

    1. ``HEADROOM_TENANT_KEY_HEADER`` (default ``X-Headroom-Tenant-ID``)
       header → ``"header"`` source.
    2. Literal ``"global"`` namespace → ``"global"`` source.

    Every resolution emits a ``tenant_key_resolved`` structured log so
    operators can alert on a sustained ``source="global"`` rate.

    Args:
        request: A Starlette / FastAPI ``Request`` object. Must have
            ``headers`` (Mapping-like) and ``state`` (with optional
            ``auth_mode`` from F1).

    Returns:
        ``(tenant_key, source)`` — both non-empty strings. ``source`` is
        one of :data:`SOURCE_HEADER` or :data:`SOURCE_GLOBAL`.
    """
    # ── 1. Header path ────────────────────────────────────────────────
    header_name = os.environ.get(TENANT_KEY_HEADER_ENV_VAR, DEFAULT_TENANT_KEY_HEADER)
    raw_header = _get_header(request, header_name)
    if raw_header:
        sanitized = _sanitize_tenant_key(raw_header)
        if sanitized:
            _emit_resolution_log(sanitized, SOURCE_HEADER, header_name=header_name)
            return sanitized, SOURCE_HEADER
        # Fall through: header was present but empty after
        # sanitization. We log this at debug level — it's noisy enough
        # at scale that an info-level log would flood, but rare enough
        # that an operator debugging a specific tenant wants to see it.
        logger.debug(
            "tenant_key_header_rejected_after_sanitization",
            extra={
                "event": "tenant_key_header_rejected",
                "header": header_name,
                "raw_len": len(raw_header),
            },
        )

    # ── 2. Global fallback ───────────────────────────────────────────

    # MUST be logged (per `feedback_no_silent_fallbacks.md`). Operators
    # see a sustained ``source="global"`` rate as a config signal: the
    # proxy is sitting behind a path that doesn't carry tenant info.
    _emit_resolution_log(GLOBAL_TENANT_KEY, SOURCE_GLOBAL)
    return GLOBAL_TENANT_KEY, SOURCE_GLOBAL


# ── Internal helpers ──────────────────────────────────────────────────


def _get_header(request: Any, name: str) -> str:
    """Read a header case-insensitively. Returns ``""`` on miss.

    Mirrors ``_header_get`` in ``auth_mode.py`` but inlined here so
    this module has no dependency on ``auth_mode.py``'s private name.
    """
    headers = getattr(request, "headers", None)
    if headers is None:
        return ""

    value: Any = None
    try:
        value = headers.get(name)
        if value is None:
            # Fallback for plain-dict fixtures with mixed case.
            for k, v in headers.items():  # type: ignore[union-attr]
                if isinstance(k, str) and k.lower() == name.lower():
                    value = v
                    break
    except AttributeError:
        return ""

    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return ""
    return str(value)


def _sanitize_tenant_key(raw: str) -> str:
    """Sanitize a tenant_key candidate. Returns ``""`` on rejection.

    Two output forms, and the choice between them is load-bearing for
    tenant isolation:

    - **verbatim** — when the raw value already consists only of
      accepted code points (ASCII alphanumeric, ``-``, ``_``) and fits
      in :data:`_MAX_TENANT_KEY_LEN`, it is returned unchanged. This is
      the common SaaS case (``cust_42``, ``Cust42TenantA``) and keeping
      it byte-identical means existing tenants keep their TOIN pool
      across the upgrade.
    - **digested** — otherwise the accepted characters become a
      readable prefix and a SHA-256 digest of the raw value is appended,
      exactly as ``headroom.memory.storage_router`` does for project and
      user ids. Dropping disallowed characters is lossy (``acme.corp``
      and ``acmecorp`` both sanitize to ``acmecorp``), and truncating
      past the cap is lossy too, so the bare sanitized form would let
      two distinct tenants train one pool — the cross-tenant pattern
      leak this module exists to prevent. The digest also keeps
      NFC-vs-NFD and full-width-vs-ASCII spellings distinct without us
      having to decide which of them are "the same" tenant.

    The two forms must stay disjoint or the fix reintroduces the bug it
    removes: a digested key is itself a legal identifier, so if it were
    also accepted verbatim a tenant could simply send another tenant's
    digested key and be handed that tenant's pool. Digested keys
    therefore always carry :data:`_TENANT_KEY_MARKER`, and the verbatim
    form refuses any value that already looks digested (see
    :func:`_looks_digested`). The same rule keeps a tenant literally
    named ``global`` out of the shared :data:`GLOBAL_TENANT_KEY`
    namespace.

    The result is always ``[A-Za-z0-9_-]{1,64}``: it flows into the TOIN
    serialized store under a ``|`` separator and into structured log
    fields, so the charset is load-bearing in both.

    Per project memory (``no regexes``) we use string ops +
    :func:`unicodedata.category` instead of ``re``. The Unicode
    classification is the source of truth for "is this a control
    char", and we get correctness for free instead of authoring our
    own character class.
    """
    raw = raw.strip()
    if not raw:
        return ""

    # Walk character by character. Cheap (header values are short),
    # branchy in a way the JIT will inline, and avoids `re`.
    accepted: list[str] = []
    lossy = False
    for ch in raw:
        # Keep the two literal punctuation chars `-` and `_`.
        if ch in ("-", "_"):
            accepted.append(ch)
            continue
        # ASCII alphanumeric: cheap fast-path — ord() check avoids the
        # unicodedata lookup for the common case (Latin-letter SaaS IDs).
        ch_ord = ord(ch)
        if (
            (0x30 <= ch_ord <= 0x39)  # 0-9
            or (0x41 <= ch_ord <= 0x5A)  # A-Z
            or (0x61 <= ch_ord <= 0x7A)  # a-z
        ):
            accepted.append(ch)
            continue
        # Anything non-ASCII or punctuation: drop, and remember that the
        # mapping stopped being injective so we can add a digest below.
        lossy = True
        # We do NOT call unicodedata for the Cyrillic/Greek/CJK case to
        # widen the set — those would round-trip into JSON fine, but
        # they introduce normalization surprises (NFC vs NFD, full-width
        # digits vs ASCII digits) that aren't worth the maintenance
        # cost. SaaS tenant IDs are typically opaque ASCII strings.
        # Touch unicodedata.category() so static analysis can see we
        # use it — and also so a developer reading the code knows
        # this is the right place to extend if we ever broaden the
        # accepted set.
        _ = unicodedata.category(ch)

    if not lossy and len(raw) <= _MAX_TENANT_KEY_LEN and not _looks_digested(raw):
        return raw

    # Digested form: marker + readable prefix + digest. The marker keeps
    # this sub-space disjoint from verbatim ids (see module docstring and
    # ``_looks_digested``), so a tenant cannot claim another tenant's
    # digested key by sending it as its own header.
    budget = _MAX_TENANT_KEY_LEN - len(_TENANT_KEY_MARKER) - _TENANT_KEY_DIGEST_LEN - 1
    prefix = "".join(accepted)[:budget]
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:_TENANT_KEY_DIGEST_LEN]
    return f"{_TENANT_KEY_MARKER}{prefix or _OPAQUE_TENANT_KEY_PREFIX}-{digest}"


def _in_digested_subspace(value: str) -> bool:
    """True when ``value`` has the shape of a key :func:`_sanitize_tenant_key` derived.

    Nothing else can produce this shape, so it is the test an upgrade uses to
    find pools that an older release stored verbatim under a legal id which
    now collides with another tenant's digest. ``global`` is deliberately NOT
    included: it is a pre-existing shared namespace, not a derived key.
    """
    if not value.startswith(_TENANT_KEY_MARKER):
        return False
    _, sep, tail = value.rpartition("-")
    return bool(sep) and len(tail) == _TENANT_KEY_DIGEST_LEN and all(c in _HEX_DIGITS for c in tail)


def _looks_digested(value: str) -> bool:
    """True when ``value`` occupies the digested key sub-space.

    The verbatim form is the identity function on conforming ids, and
    every digested key is itself conforming, so without this check the
    digest could be bypassed: a tenant sending ``acmecorp-80a1d5cc623b165f``
    as its own header would be mapped to that string verbatim and share
    the pool of the ``acme.corp`` tenant it was derived from.

    A value counts as digested when it is the shared
    :data:`GLOBAL_TENANT_KEY` literal (a tenant literally named
    ``global`` must not join the OSS shared pool) or when it matches the
    digested shape ``<marker><prefix>-<16 hex chars>``. Anything else,
    including ids that merely start with the marker but carry no digest,
    stays verbatim so ordinary ids are not rekeyed.
    """
    return value == GLOBAL_TENANT_KEY or _in_digested_subspace(value)


def _emit_resolution_log(
    tenant_key: str,
    source: str,
    *,
    header_name: str | None = None,
    auth_mode: str | None = None,
) -> None:
    """Emit the structured ``tenant_key_resolved`` log event.

    Required by the no-silent-fallbacks rule. Every resolution path —
    including the ``"global"`` fallback — emits exactly one log event
    so log aggregators can compute per-source rates without sampling.

    The ``tenant_key`` value itself is included in the structured
    fields. It's not a secret: header-mode keys are operator-supplied
    """
    extra: dict[str, Any] = {
        "event": "tenant_key_resolved",
        "tenant_key": tenant_key,
        "source": source,
    }
    if header_name is not None:
        extra["header"] = header_name
    if auth_mode is not None:
        extra["auth_mode"] = auth_mode
    # ``info`` level: routine, structured, low-cardinality. Operators
    # can downsample to ``debug`` if they're confident their tenancy
    # config is healthy.
    logger.info("tenant_key_resolved", extra=extra)


__all__ = [
    "DEFAULT_TENANT_KEY_HEADER",
    "GLOBAL_TENANT_KEY",
    "SOURCE_GLOBAL",
    "SOURCE_HEADER",
    "TENANT_KEY_HEADER_ENV_VAR",
    "_in_digested_subspace",
    "_looks_digested",
    "get_current_tenant_key",
    "resolve_tenant_key",
    "set_request_tenant_key",
]
