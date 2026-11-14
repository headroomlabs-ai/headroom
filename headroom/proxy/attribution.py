"""Request attribution: named labels that say who and where a request came from.

Every aggregate the proxy reports (savings, spend, request counts) is a sum over
requests. Splitting those sums needs a label on each request: which project,
which client harness, which machine, which team. Before this module each label
was its own axis with its own plumbing; project attribution alone touches the
middleware, a context variable, the outcome funnel, the metrics call and the
savings tracker. This module gives every label the same plumbing once.

A *dimension* is a name (``project``, ``host``, ``team``) and a *resolver* that
reads it from an inbound request. Middleware resolves every enabled dimension
once per request into a context variable; the outcome funnel copies the result
onto :class:`~headroom.proxy.outcome.RequestOutcome`, the pipeline's
:class:`~headroom.pipeline.OutcomeSnapshot`, the savings tracker and the savings
ledger. Anything downstream (a dashboard table, ``headroom savings --by``, an
exporter extension) reads one ``{dimension: label}`` mapping instead of growing
another parameter.

**Labels, not identity.** Every built-in resolver reads something the client
reports about itself. That is right for attributing usage, where a client has
no reason to lie and the operator controls the clients, and wrong for
authorizing anything. Authenticated identity stays where it is, behind
:func:`headroom.proxy.identity.set_identity_resolver`; a deployment that has one
can publish it as a dimension with :func:`register_dimension`, and every
surface here picks it up.

Which dimensions are resolved is controlled by
``HEADROOM_ATTRIBUTION_DIMENSIONS``, a comma-separated list. ``default`` expands
to the built-ins, ``-name`` removes one, ``none`` turns attribution off. A name
with no registered resolver is read from the ``X-Headroom-<name>`` request
header, so an operator can add a ``team`` or ``cost-center`` dimension with no
code: list it, and have clients send the header. ``X-Headroom-Labels:
team=infra; cost-center=42`` carries several at once, and the ``labels`` token accepts any name a client
sends that way.
"""

from __future__ import annotations

import functools
import itertools
import logging
import os
import threading
from collections.abc import Callable, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

logger = logging.getLogger(__name__)

ATTRIBUTION_DIMENSIONS_ENV_VAR: Final[str] = "HEADROOM_ATTRIBUTION_DIMENSIONS"

#: Long enough for any FQDN or directory name a client can report, short
#: enough that a misbehaving client cannot bloat persisted state.
LABEL_MAX_LENGTH: Final[int] = 128

#: Dimension names become JSON keys, CLI arguments and header suffixes.
DIMENSION_NAME_MAX_LENGTH: Final[int] = 32

#: Upper bound on dimensions resolved per request. Each one costs a bounded
#: map in the savings tracker, so this caps persisted state, not just CPU.
MAX_DIMENSIONS: Final[int] = 16

#: Header prefix the fallback resolver reads for a dimension with no resolver.
HEADER_PREFIX: Final[str] = "x-headroom-"

#: One header carrying several labels, ``team=infra; cost-center=42``. Lets a
#: client whose header set is fixed at install time (Codex maps each header to
#: an environment variable in config.toml) send any label set through one
#: variable. A dedicated ``X-Headroom-<name>`` header wins over this one.
LABELS_HEADER: Final[str] = "x-headroom-labels"

#: Only this much of ``X-Headroom-Labels`` is read. It is parsed on every
#: request, so its cost must not be the client's to choose. Sized to the
#: longest header :func:`format_labels` can emit (every label at full length
#: and every character percent-encoded), so whatever it builds is read whole.
LABELS_HEADER_MAX_LENGTH: Final[int] = MAX_DIMENSIONS * (
    DIMENSION_NAME_MAX_LENGTH + len("=") + 3 * LABEL_MAX_LENGTH
) + (MAX_DIMENSIONS - 1) * len("; ")

#: In ``HEADROOM_ATTRIBUTION_DIMENSIONS``, accept every label a client sends in
#: ``X-Headroom-Labels`` as its own dimension, not only the listed ones. Off by
#: default: each new dimension is persisted state, so an operator opts in.
OPEN_LABELS_TOKEN: Final[str] = "labels"

#: Words with a meaning in ``HEADROOM_ATTRIBUTION_DIMENSIONS``; never dimension names.
RESERVED_TOKENS: Final[frozenset[str]] = frozenset({"default", "none", OPEN_LABELS_TOKEN})

#: What the savings ledger and ``headroom savings`` show for an event with no
#: label in a dimension. Never a label itself, so the two cannot be confused.
UNLABELED: Final[str] = "unknown"

_DIMENSION_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_-.")

EMPTY_ATTRIBUTION: Mapping[str, str] = MappingProxyType({})


def sanitize_dimension_name(value: Any) -> str | None:
    """Normalize a dimension name; ``None`` when unusable.

    Lowercase ASCII letters, digits, ``_``, ``-`` and ``.`` only: the name is a
    JSON key, a CLI argument and a header suffix, and anything else would need
    escaping in at least one of them. It may not start with ``-`` (that reads
    as "remove" in ``HEADROOM_ATTRIBUTION_DIMENSIONS`` and as a flag on the
    command line) or ``.``.
    """
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if not cleaned or len(cleaned) > DIMENSION_NAME_MAX_LENGTH or cleaned[0] in "-.":
        return None
    if any(ch not in _DIMENSION_NAME_CHARS for ch in cleaned):
        return None
    return cleaned


def sanitize_label(value: Any, *, decode: bool = True) -> str | None:
    """Normalize a label; ``None`` when unusable.

    Same rule as project names (:mod:`headroom.proxy.project_name_policy`):
    percent-decoded, printable characters only, trimmed, length-capped.

    ``decode`` is for a value read straight from a header. A label that was
    already resolved is cleaned with ``decode=False``: decoding is not
    idempotent (``%2541`` becomes ``%41`` and then ``A``), so it happens once,
    where the header is read, and never again on the way to storage.

    The label is cut to length before trailing space is trimmed, so cleaning a
    clean label changes nothing. ``unknown`` is not a label: it is what the
    ledger shows when there is none (:data:`UNLABELED`).
    """
    if decode:
        from headroom.proxy.project_name_policy import sanitize_project_name

        cleaned = sanitize_project_name(value)
    elif isinstance(value, str):
        cleaned = "".join(ch for ch in value if ch.isprintable()).strip() or None
    else:
        cleaned = None
    if cleaned is None:
        return None
    cleaned = cleaned[:LABEL_MAX_LENGTH].rstrip()
    if not cleaned or cleaned == UNLABELED:
        return None
    return cleaned


@dataclass(frozen=True)
class AttributionRequest:
    """What a resolver may read about an inbound request.

    ``headers`` keys are lowercased. ``peer`` is the TCP peer address (no port).
    ``path_project`` is the project named by a ``/p/<name>`` base-URL prefix,
    which the middleware strips before routing and so must hand over itself.
    """

    headers: Mapping[str, str] = field(default_factory=dict)
    path: str = ""
    peer: str | None = None
    path_project: str | None = None

    def header(self, name: str) -> str | None:
        value = self.headers.get(name.lower())
        return value if isinstance(value, str) else None


#: Returns the label as it should be stored, or nothing. A resolver that reads
#: a header itself decides whether to percent-decode it (:func:`sanitize_label`).
Resolver = Callable[[AttributionRequest], Any]


@dataclass(frozen=True)
class _Registration:
    resolver: Resolver
    default: bool
    client_settable: bool = False


_registry: dict[str, _Registration] = {}
_builtins: dict[str, _Registration] = {}
_registry_lock = threading.Lock()
_builtins_installed = False
_builtins_lock = threading.Lock()


def _ensure_builtins() -> None:
    # Lazy so importing this module stays cheap and cycle-free; the built-in
    # resolvers import the auth, project and tenant modules.
    global _builtins_installed
    if _builtins_installed:
        return
    with _builtins_lock:
        if _builtins_installed:
            return
        from headroom.proxy.attribution_builtins import install_builtin_dimensions

        install_builtin_dimensions()
        # Only once they are all there: a concurrent first caller waits on the
        # lock instead of seeing a partial registry.
        _builtins_installed = True


def _register_if_absent(name: str, resolver: Resolver, *, default: bool = True) -> None:
    """Register a built-in without displacing a resolver an extension installed."""
    # Built-ins read what the client reports about itself, so a client may
    # also name them through the open ``labels`` token.
    registration = _Registration(resolver=resolver, default=default, client_settable=True)
    with _registry_lock:
        _builtins[name] = registration
        _registry.setdefault(name, registration)


def register_dimension(
    name: str,
    resolver: Resolver,
    *,
    default: bool = True,
    replace: bool = False,
    client_settable: bool = False,
) -> str:
    """Register ``resolver`` as the source of dimension ``name``.

    ``default`` puts the dimension in the ``default`` set, so it is resolved
    without operator configuration. An extension that needs a dimension for its
    own report (a cost-per-branch table needs ``branch``) registers it this way.
    ``replace`` must be passed to override an existing resolver, so two
    extensions cannot silently fight over one name. Replacing a built-in
    (``project``, ``client``) makes the new resolver the only source of that
    label: what a handler knows about the harness or project no longer
    overrides it.

    ``client_settable`` decides what the open ``labels`` token does when the
    operator has not listed this dimension. Off, the default, keeps the name
    reserved: a client cannot supply it in ``X-Headroom-Labels``, which is what
    a resolver backed by authenticated identity needs. On, a client may name it
    there, as it may for the built-ins.
    """
    cleaned = sanitize_dimension_name(name)
    if cleaned is None or cleaned in RESERVED_TOKENS:
        raise ValueError(f"invalid attribution dimension name: {name!r}")
    _ensure_builtins()
    with _registry_lock:
        if cleaned in _registry and not replace:
            raise ValueError(f"attribution dimension {cleaned!r} is already registered")
        _registry[cleaned] = _Registration(
            resolver=resolver, default=default, client_settable=client_settable
        )
    return cleaned


def unregister_dimension(name: str) -> None:
    """Remove a registered resolver (tests, extension teardown).

    A built-in that was replaced gets its own resolver back.
    """
    cleaned = sanitize_dimension_name(name) or ""
    with _registry_lock:
        builtin = _builtins.get(cleaned)
        if builtin is not None:
            _registry[cleaned] = builtin
        else:
            _registry.pop(cleaned, None)


def registered_dimensions() -> tuple[str, ...]:
    _ensure_builtins()
    with _registry_lock:
        return tuple(_registry)


def default_dimensions() -> tuple[str, ...]:
    _ensure_builtins()
    with _registry_lock:
        return tuple(name for name, reg in _registry.items() if reg.default)


def _parse_spec(
    spec: str | None, defaults: tuple[str, ...]
) -> tuple[tuple[str, ...], frozenset[str]]:
    """The selected dimension names, and the names removed with ``-name``."""
    if spec is None or not spec.strip():
        return defaults[:MAX_DIMENSIONS], frozenset()
    selected: list[str] = []
    removed: set[str] = set()
    for raw in spec.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token == "none":
            selected.clear()
            continue
        if token == "default":
            selected.extend(defaults)
            continue
        if token.startswith("-"):
            name = sanitize_dimension_name(token[1:])
            if name:
                removed.add(name)
            continue
        name = sanitize_dimension_name(token)
        if name is None:
            logger.warning("ignoring invalid attribution dimension %r in %s", raw, spec)
            continue
        selected.append(name)
    ordered = [name for name in dict.fromkeys(selected) if name not in removed]
    # ``labels`` is a switch, not a dimension, so it never takes one of the slots.
    names = [name for name in ordered if name != OPEN_LABELS_TOKEN]
    if len(names) > MAX_DIMENSIONS:
        logger.warning(
            "attribution: %d dimensions configured, keeping the first %d",
            len(names),
            MAX_DIMENSIONS,
        )
        kept = set(names[:MAX_DIMENSIONS])
        ordered = [name for name in ordered if name == OPEN_LABELS_TOKEN or name in kept]
    return tuple(ordered), frozenset(removed)


def parse_dimension_spec(spec: str | None, defaults: Iterable[str]) -> tuple[str, ...]:
    """Expand a ``HEADROOM_ATTRIBUTION_DIMENSIONS`` value into dimension names.

    Unset or empty means ``default``. Order is preserved and duplicates dropped;
    invalid names are skipped with a warning rather than failing startup.
    """
    return _parse_spec(spec, tuple(defaults))[0]


@functools.lru_cache(maxsize=16)
def _parsed_spec(
    spec: str | None, defaults: tuple[str, ...]
) -> tuple[tuple[str, ...], frozenset[str]]:
    # Called on every request: parse (and warn about) a given value once.
    return _parse_spec(spec, defaults)


def _current_spec(
    environ: Mapping[str, str] | None,
) -> tuple[tuple[str, ...], frozenset[str]]:
    env = os.environ if environ is None else environ
    return _parsed_spec(env.get(ATTRIBUTION_DIMENSIONS_ENV_VAR), default_dimensions())


def enabled_dimensions(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Dimensions resolved on each request, from the environment."""
    return _current_spec(environ)[0]


def excluded_dimensions(environ: Mapping[str, str] | None = None) -> frozenset[str]:
    """Names removed with ``-name``.

    Nothing re-admits them: not the open ``labels`` token, not a handler
    binding the label.
    """
    return _current_spec(environ)[1]


def dimension_enabled(name: str, environ: Mapping[str, str] | None = None) -> bool:
    """Whether a label the proxy binds for ``name`` is recorded.

    Listed in ``HEADROOM_ATTRIBUTION_DIMENSIONS``, or an unregistered name
    under the open ``labels`` token that was not removed with ``-name``. A
    registered dimension that is not listed is not enabled: under ``labels``
    only a client can name it, in ``X-Headroom-Labels``, and only when it is
    client-settable (see :func:`resolve_attribution`).
    """
    if name in RESERVED_TOKENS:
        return False
    enabled, excluded = _current_spec(environ)
    if name in enabled:
        return True
    if OPEN_LABELS_TOKEN not in enabled or name in excluded:
        return False
    return name not in registered_dimensions()


def _builtin_label_applies(name: str, environ: Mapping[str, str] | None = None) -> bool:
    """Whether a value the proxy itself derived for ``name`` is recorded.

    Handlers know ``project`` and ``client`` independently of the resolvers (a
    Codex project context, the harness of a WebSocket session). Those values
    count while the dimension is enabled and still resolved by its built-in.
    Once ``register_dimension(..., replace=True)`` hands the name to another
    resolver, that resolver is the only source of the label.
    """
    if not dimension_enabled(name, environ):
        return False
    with _registry_lock:
        registration = _registry.get(name)
        builtin = _builtins.get(name)
    if registration is None:
        return True
    return builtin is not None and registration.resolver is builtin.resolver


def _label_pairs(value: Any) -> dict[str, str]:
    """Every well-formed pair in a ``name=value; name=value`` string, sanitized."""
    if not isinstance(value, str) or not value.strip():
        return {}
    if len(value) > LABELS_HEADER_MAX_LENGTH:
        # Keep whole pairs only: a pair cut at the limit would be booked under
        # a shortened label. A separator right at the limit ends a whole pair.
        head = value[: LABELS_HEADER_MAX_LENGTH + 1]
        value = head[: max(head.rfind(","), head.rfind(";"), 0)]
    pairs: dict[str, str] = {}
    for chunk in value.replace(",", ";").split(";"):
        raw_name, sep, raw_label = chunk.partition("=")
        if not sep:
            continue
        name = sanitize_dimension_name(raw_name)
        label = sanitize_label(raw_label.strip())
        if name is not None and name not in RESERVED_TOKENS and label is not None:
            pairs[name] = label
    return pairs


def parse_labels(value: Any) -> dict[str, str]:
    """Parse ``name=value; name=value`` (``,`` also separates) into labels.

    Malformed pairs are skipped; the result is sanitized and bounded.
    """
    return dict(itertools.islice(_label_pairs(value).items(), MAX_DIMENSIONS))


_LABEL_ESCAPES = str.maketrans({"%": "%25", ",": "%2C", ";": "%3B"})


def format_labels(labels: Mapping[str, str]) -> str:
    """Inverse of :func:`parse_labels`, for building ``X-Headroom-Labels``.

    The parser splits on ``,`` and ``;`` and then percent-decodes each label
    once, so ``%``, ``,`` and ``;`` in a label are percent-encoded here. Every
    label that :func:`normalize_attribution` keeps comes back unchanged,
    ``release%41`` and ``design, ops`` included.
    """
    return "; ".join(
        f"{name}={label.translate(_LABEL_ESCAPES)}"
        for name, label in normalize_attribution(labels).items()
    )


def header_name_for(dimension: str) -> str:
    """Request header the fallback resolver reads for ``dimension``."""
    return HEADER_PREFIX + dimension.replace("_", "-").replace(".", "-")


def _header_resolver(dimension: str) -> Resolver:
    header = header_name_for(dimension)

    def resolve(request: AttributionRequest) -> str | None:
        direct = request.header(header)
        if direct:
            return sanitize_label(direct)
        return _label_pairs(request.header(LABELS_HEADER)).get(dimension)

    return resolve


def _is_credential(dimension: str) -> bool:
    from headroom.proxy.internal_header_policy import is_credential_header

    return bool(is_credential_header(header_name_for(dimension)))


def resolve_attribution(
    request: AttributionRequest,
    dimensions: Iterable[str] | None = None,
) -> dict[str, str]:
    """Resolve every enabled dimension for ``request``.

    A dimension whose resolver yields nothing usable is absent from the result,
    never present with an empty or placeholder label: "unattributed" is a
    reporting decision, not a value. A resolver that raises is logged and
    skipped so one broken extension cannot fail the request.

    With the open ``labels`` token, ``X-Headroom-Labels`` may name any
    dimension that was not resolved above. That includes a registered one that
    is not selected, provided it is client-settable (the built-ins are; see
    :func:`register_dimension`). A selected resolver keeps precedence, even
    when it yields nothing, and a name removed with ``-name`` stays out.
    """
    _ensure_builtins()
    names = enabled_dimensions() if dimensions is None else tuple(dimensions)
    excluded = excluded_dimensions() if dimensions is None else frozenset()
    with _registry_lock:
        registry = dict(_registry)
    labels: dict[str, str] = {}
    for name in names:
        if name in RESERVED_TOKENS:
            continue
        registration = registry.get(name)
        if registration is not None:
            resolver = registration.resolver
        elif _is_credential(name):
            # Never turn a credential header into a label that lands in
            # stats, logs and exports.
            continue
        else:
            resolver = _header_resolver(name)
        try:
            value = resolver(request)
        except Exception:  # noqa: BLE001 - attribution must never fail a request
            logger.debug("attribution resolver %r failed", name, exc_info=True)
            continue
        label = sanitize_label(value, decode=False)
        if label is not None:
            labels[name] = label
    if OPEN_LABELS_TOKEN in names:
        for name, label in _label_pairs(request.header(LABELS_HEADER)).items():
            if len(labels) >= MAX_DIMENSIONS:
                break
            if name in labels or name in excluded or _is_credential(name):
                continue
            registration = registry.get(name)
            if registration is not None and (name in names or not registration.client_settable):
                # A selected resolver has the last word; an unselected one
                # that is not client-settable keeps its name reserved.
                continue
            labels[name] = label
    return labels


def normalize_attribution(raw: Any) -> dict[str, str]:
    """Coerce a persisted or caller-supplied mapping into clean labels."""
    if not isinstance(raw, Mapping):
        return {}
    cleaned: dict[str, str] = {}
    for key, value in raw.items():
        name = sanitize_dimension_name(key)
        label = sanitize_label(value, decode=False)
        if name is not None and name not in RESERVED_TOKENS and label is not None:
            cleaned[name] = label
        if len(cleaned) >= MAX_DIMENSIONS:
            break
    return cleaned


# ── Request-scoped binding ────────────────────────────────────────────

_current_attribution: ContextVar[Mapping[str, str]] = ContextVar(
    "headroom_current_attribution", default=EMPTY_ATTRIBUTION
)


def set_current_attribution(labels: Mapping[str, str] | None) -> None:
    """Bind the active request's labels for downstream outcome recording."""
    _current_attribution.set(MappingProxyType(dict(labels)) if labels else EMPTY_ATTRIBUTION)


def get_current_attribution() -> Mapping[str, str]:
    """Labels bound to the current request context (read-only, maybe empty)."""
    return _current_attribution.get()


#: Headers that name the client's own conversation, in priority order. Used only
#: to group requests into conversations for reporting (``OutcomeSnapshot.
#: conversation_key`` when the handler derived none), never for accounting.
CONVERSATION_HEADERS: Final[tuple[str, ...]] = (
    "x-headroom-session-id",
    "x-claude-code-session-id",
)

_current_conversation: ContextVar[str | None] = ContextVar(
    "headroom_current_conversation", default=None
)


def conversation_hint(headers: Mapping[str, str]) -> str | None:
    """The client-named conversation for a request, from :data:`CONVERSATION_HEADERS`."""
    for name in CONVERSATION_HEADERS:
        label = sanitize_label(headers.get(name))
        if label is not None:
            return label
    return None


def set_current_conversation(value: str | None) -> None:
    """Bind the conversation key; ``value`` is already decoded (:func:`conversation_hint`)."""
    _current_conversation.set(sanitize_label(value, decode=False))


def get_current_conversation() -> str | None:
    return _current_conversation.get()


def bind_label(dimension: str, value: Any) -> None:
    """Add or replace one label on the current request.

    For values known only after the middleware ran, such as a client a handler
    identified from the request body. ``None`` removes the label. Ignored for a
    dimension that is not enabled, and for a new one past ``MAX_DIMENSIONS``.
    """
    name = sanitize_dimension_name(dimension)
    if name is None:
        return
    labels = dict(_current_attribution.get())
    label = sanitize_label(value, decode=False)
    if label is not None and not dimension_enabled(name):
        return  # turned off by the operator: never added, still removable
    if label is None:
        labels.pop(name, None)
    elif name in labels or len(labels) < MAX_DIMENSIONS:
        labels[name] = label
    else:
        return
    set_current_attribution(labels)


def bind_builtin_label(dimension: str, value: Any) -> None:
    """:func:`bind_label` for a value a handler derived for a built-in dimension.

    A no-op when the dimension is off or its resolver was replaced.
    """
    if _builtin_label_applies(dimension):
        bind_label(dimension, value)


def merge_attribution(*layers: Mapping[str, str] | None) -> dict[str, str]:
    """Merge label mappings; later layers win, empty labels are dropped.

    At most ``MAX_DIMENSIONS`` labels come back. Past that, later layers keep
    their place as well as their value, so what a handler knows is not the
    part that is cut.
    """
    cleaned = [normalize_attribution(layer) for layer in layers if layer]
    merged: dict[str, str] = {}
    for layer in cleaned:
        merged.update(layer)
    if len(merged) <= MAX_DIMENSIONS:
        return merged
    keep = list(dict.fromkeys(name for layer in reversed(cleaned) for name in layer))
    kept = frozenset(keep[:MAX_DIMENSIONS])
    return {name: label for name, label in merged.items() if name in kept}


# ── OpenTelemetry ─────────────────────────────────────────────────────

OTEL_ATTRIBUTION_ENV_VAR: Final[str] = "HEADROOM_OTEL_ATTRIBUTION"

#: OTel attribute namespace for labels, so they cannot collide with the
#: canonical instrument attributes (``provider``, ``model``, ``source``).
OTEL_ATTRIBUTE_PREFIX: Final[str] = "headroom.attribution."

_otel_provider: Callable[[], Mapping[str, str]] | None = None


def otel_dimensions(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Dimensions exported as OTel attributes; empty when export is off.

    Off by default: every label becomes a metric series, and a backend bills
    or drops by series count, so the operator chooses which to pay for.
    ``HEADROOM_OTEL_ATTRIBUTION=host,project`` exports those two; ``all``
    exports every resolved dimension.
    """
    env = os.environ if environ is None else environ
    raw = (env.get(OTEL_ATTRIBUTION_ENV_VAR) or "").strip().lower()
    if not raw or raw in ("0", "false", "off", "no"):
        return ()
    if raw in ("1", "true", "on", "yes", "all", "*"):
        return ("*",)
    return tuple(n for n in (sanitize_dimension_name(p) for p in raw.split(",")) if n)


def install_otel_attribution(environ: Mapping[str, str] | None = None) -> bool:
    """Attach the current request's labels to every Headroom OTel datapoint.

    Rides :func:`headroom.observability.metrics.register_otel_metric_attribute_provider`,
    the seam that exists for exactly this. Returns whether export is on.
    Idempotent; calling again with export off removes the provider.
    """
    global _otel_provider
    from headroom.observability.metrics import (
        register_otel_metric_attribute_provider,
        unregister_otel_metric_attribute_provider,
    )

    if _otel_provider is not None:
        unregister_otel_metric_attribute_provider(_otel_provider)
        _otel_provider = None
    selected = otel_dimensions(environ)
    if not selected:
        return False
    export_all = "*" in selected
    allowed = frozenset(selected)

    def provider() -> Mapping[str, str]:
        return {
            OTEL_ATTRIBUTE_PREFIX + name: label
            for name, label in get_current_attribution().items()
            if export_all or name in allowed
        }

    _otel_provider = register_otel_metric_attribute_provider(provider)
    return True


__all__ = [
    "ATTRIBUTION_DIMENSIONS_ENV_VAR",
    "AttributionRequest",
    "DIMENSION_NAME_MAX_LENGTH",
    "EMPTY_ATTRIBUTION",
    "LABELS_HEADER",
    "LABELS_HEADER_MAX_LENGTH",
    "LABEL_MAX_LENGTH",
    "MAX_DIMENSIONS",
    "OTEL_ATTRIBUTE_PREFIX",
    "OTEL_ATTRIBUTION_ENV_VAR",
    "OPEN_LABELS_TOKEN",
    "RESERVED_TOKENS",
    "UNLABELED",
    "Resolver",
    "CONVERSATION_HEADERS",
    "bind_builtin_label",
    "bind_label",
    "conversation_hint",
    "get_current_conversation",
    "set_current_conversation",
    "default_dimensions",
    "enabled_dimensions",
    "excluded_dimensions",
    "format_labels",
    "get_current_attribution",
    "header_name_for",
    "install_otel_attribution",
    "merge_attribution",
    "normalize_attribution",
    "otel_dimensions",
    "parse_dimension_spec",
    "parse_labels",
    "register_dimension",
    "registered_dimensions",
    "resolve_attribution",
    "sanitize_dimension_name",
    "sanitize_label",
    "set_current_attribution",
    "unregister_dimension",
]
