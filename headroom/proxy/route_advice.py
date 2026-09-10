"""Per-request backend selection, published by an extension.

WHY THIS EXISTS
    Headroom picks its egress backend ONCE, at startup: `create_proxy_backend`
    returns a single `Backend` (or None for the direct Anthropic path) and
    every request goes through it. That is right for "run this whole proxy
    against Bedrock instead of Anthropic", and it is the wrong shape for "this
    request is cheaper on a different provider than the last one".

    Meanwhile `ModelRouter` already lets an extension change `body["model"]`
    per request -- but only within whatever protocol the request arrived in,
    because a model id alone cannot move a request to another provider.

    So a router extension can currently decide something Headroom has no way to
    carry out. This module is the missing half: a routing decision that names a
    PROVIDER as well as a model, and a per-request backend to serve it.

THE CONTRACT, AND WHY IT IS VENDOR-NEUTRAL
    An extension sets `request.state.headroom_route` to a `RouteAdvice`.
    Headroom reads it, resolves a backend for `advice.provider`, and dispatches
    there. Nothing in core names any particular extension.

ABSENT MEANS UNCHANGED
    This is the property that matters most. With no extension installed, no
    extension setting the field, an unresolvable provider, or a backend that
    fails to construct, `backend_for()` returns exactly what
    `self.anthropic_backend` would have been -- including None, which keeps the
    direct-API path. Nothing about the default deployment moves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

STATE_ATTR = "headroom_route"

# Providers that speak the Anthropic Messages API natively, i.e. the shape the
# proxy already holds. Everything else needs a translating backend.
_NATIVE = ("anthropic",)


@dataclass(frozen=True)
class RouteAdvice:
    """A routing decision an extension wants Headroom to carry out.

    Only `model` and `provider` are required; the rest is explanatory and is
    logged rather than acted on, so an extension can be as terse as it likes.
    """

    model: str
    provider: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("RouteAdvice needs a model")


def advice_from(request: Any) -> RouteAdvice | None:
    """The advice an extension published, if any. Never raises.

    Duck-typed on purpose: an extension should not have to import Headroom to
    talk to Headroom. Anything carrying `.model` (and optionally `.provider`)
    works, which also keeps extensions from breaking when this dataclass gains
    a field.
    """
    obj = getattr(getattr(request, "state", None), STATE_ATTR, None)
    if obj is None:
        return None
    model = getattr(obj, "model", None)
    if not isinstance(model, str) or not model:
        return None
    provider = getattr(obj, "provider", "") or ""
    return RouteAdvice(
        model=model,
        provider=provider if isinstance(provider, str) else "",
        reason=str(getattr(obj, "reason", "") or "")[:400],
    )


class BackendResolver:
    """Lazily builds and caches one translating backend per provider.

    Construction is not free -- the Bedrock path calls out to AWS to enumerate
    inference profiles -- so backends are built on first use for a provider and
    kept. A provider that fails to construct is remembered as a failure and not
    retried on every request; the request falls back to the default backend
    rather than erroring, because a routing preference must never be able to
    take traffic down.
    """

    def __init__(self, default: Any = None) -> None:
        self._default = default
        self._cache: dict[str, Any] = {}
        self._failed: set[str] = set()

    @property
    def default(self) -> Any:
        return self._default

    def for_request(self, request: Any, *, body: dict | None = None) -> Any:
        """The backend to serve this request. Falls back to the default."""
        advice = advice_from(request)
        if advice is None:
            return self._default

        provider = advice.provider or _provider_of(advice.model)
        if not provider or provider in _NATIVE:
            # Same protocol the proxy already speaks -- a `body["model"]`
            # rewrite is enough and the extension has already done it. Nothing
            # to switch.
            return self._default
        if provider in self._failed:
            return self._default

        backend = self._cache.get(provider)
        if backend is None:
            try:
                backend = self._build(provider)
            except Exception as exc:  # noqa: BLE001 - see the class docstring
                log.warning("route advice: building %r raised (%s)", provider, exc)
                backend = None
            if backend is None:
                self._failed.add(provider)
                return self._default
            self._cache[provider] = backend

        if body is not None:
            # The extension chose the model; make sure the body agrees, since
            # it could not safely write a foreign model id itself. Log here
            # too -- a caller asking without a body is only checking WHICH
            # backend will serve the request, and would log the same decision
            # a second time.
            body["model"] = advice.model
            log.info(
                "route advice: %s via %s (%s)",
                advice.model,
                provider,
                advice.reason or "no reason given",
            )
        return backend

    def _build(self, provider: str) -> Any:
        # Validate the name FIRST. `LiteLLMBackend` accepts an unknown provider
        # happily -- the registry falls through to a generic pass-through
        # config -- so a typo builds a backend that only fails later, at
        # request time, with an error that points nowhere near the typo.
        if not _known_provider(provider):
            log.warning("route advice: %r is not a litellm provider; ignoring", provider)
            return None
        try:
            from headroom.backends.litellm import LiteLLMBackend

            return LiteLLMBackend(provider=provider)
        except Exception as exc:  # noqa: BLE001 - never fail a request for this
            log.warning(
                "route advice: cannot build a backend for %r (%s); "
                "falling back to the configured backend",
                provider,
                exc,
            )
            return None


def resolver_for(handler: Any) -> BackendResolver:
    """The handler's resolver, built once and reused.

    Lives here rather than on a handler mixin so every protocol handler can
    reach it without depending on a sibling mixin. Rebuilt if
    `anthropic_backend` is reassigned (tests do this), so the resolver can
    never serve a stale default.
    """
    default = getattr(handler, "anthropic_backend", None)
    cached = getattr(handler, "_route_resolver_cache", None)
    if cached is None or cached.default is not default:
        cached = BackendResolver(default)
        handler._route_resolver_cache = cached
    return cached


def _known_provider(provider: str) -> bool:
    """Is this a provider litellm actually knows about?"""
    try:
        import litellm

        names = {getattr(p, "value", None) or str(p) for p in getattr(litellm, "provider_list", [])}
        return provider in names
    except Exception:  # noqa: BLE001
        return False


def _provider_of(model: str) -> str:
    """Provider from a `provider/model` spec, else from litellm's table."""
    if "/" in model:
        return model.split("/", 1)[0]
    try:
        import litellm

        entry = litellm.model_cost.get(model) or {}
        return str(entry.get("litellm_provider") or "")
    except Exception:  # noqa: BLE001
        return ""
