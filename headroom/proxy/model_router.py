"""Cost-aware model routing (issue #1706).

Complementary to content compression: route a request to a cheaper (or more
capable) model based on request characteristics, so callers can stretch quota
and control spend without changing their client.

This is an opt-in, config-driven **mechanism**, not an opinionated built-in
policy. The operator declares an ordered list of rules mapping request
characteristics to a target model; the router picks the first rule whose
conditions all match and records the decision, with a human-readable reason,
so routing is observable and never a black box. When disabled (the default) or
when no rule matches, the original model is returned unchanged, so behavior is
identical to today.

The router is a pure component: no I/O, no global state, fully unit-testable.
Wiring into the request path (reading the decision, rewriting the outgoing
model, logging, and metrics) lives in the handlers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelRoute:
    """One ordered routing rule.

    A rule matches when every condition that is set is satisfied (logical AND).
    Conditions left as ``None``/empty are ignored. Rules are evaluated in order
    and the first match wins.
    """

    to_model: str
    """Model to route to when this rule matches."""

    max_input_tokens: int | None = None
    """Match only when estimated input tokens are <= this (cheap for small requests)."""

    min_input_tokens: int | None = None
    """Match only when estimated input tokens are >= this."""

    require_no_tools: bool = False
    """Match only when the request declares no tools (a proxy for low-risk work)."""

    from_models: tuple[str, ...] = ()
    """Restrict this rule to these source models. Empty = any source model."""

    name: str = ""
    """Human-readable label surfaced in decision logs."""

    def matches(self, *, model: str, input_tokens: int, has_tools: bool) -> bool:
        """True when every set condition is satisfied for this request."""
        if self.from_models and model not in self.from_models:
            return False
        if self.require_no_tools and has_tools:
            return False
        if self.max_input_tokens is not None and input_tokens > self.max_input_tokens:
            return False
        if self.min_input_tokens is not None and input_tokens < self.min_input_tokens:
            return False
        # A rule whose ``to_model`` equals the current model still MATCHES (strict
        # first-match-wins): it is a no-op (``changed`` is False) that short-circuits
        # later rules, which lets an operator write an explicit exemption rule.
        return True


@dataclass(frozen=True)
class ModelRouterConfig:
    """Configuration for :class:`ModelRouter`. Disabled by default."""

    enabled: bool = False
    routes: tuple[ModelRoute, ...] = ()

    @classmethod
    def from_env(cls, enabled_raw: str | None, routes_raw: str | None) -> ModelRouterConfig:
        """Build config from env-style strings, failing open to disabled.

        ``routes_raw`` is a JSON array of rule objects, e.g.::

            [{"name": "small->mini", "max_input_tokens": 4000,
              "require_no_tools": true, "to_model": "gpt-5.4-mini"}]

        A malformed value logs a warning and disables routing rather than
        raising, so a bad config can never take the proxy down.
        """
        enabled = _truthy(enabled_raw)
        routes = _parse_routes(routes_raw)
        if enabled and not routes:
            logger.warning("model router enabled but no valid routes configured; disabling")
            return cls(enabled=False, routes=())
        return cls(enabled=enabled and bool(routes), routes=routes)


@dataclass(frozen=True)
class ModelDecision:
    """The outcome of a routing evaluation for one request."""

    original_model: str
    routed_model: str
    matched: bool
    reason: str
    rule_name: str = ""

    @property
    def changed(self) -> bool:
        """True when the caller should rewrite the outgoing model."""
        return self.matched and self.routed_model != self.original_model


class ModelRouter:
    """Selects an outgoing model from ordered, config-driven rules."""

    def __init__(self, config: ModelRouterConfig | None) -> None:
        self._config = config or ModelRouterConfig()

    @property
    def enabled(self) -> bool:
        return self._config.enabled and bool(self._config.routes)

    def select(self, *, model: str, input_tokens: int, has_tools: bool) -> ModelDecision:
        """Return the routing decision for a request.

        Never raises: on a disabled router or no matching rule, returns a
        non-matching decision that leaves the original model in place.
        """
        if not self.enabled:
            return ModelDecision(model, model, matched=False, reason="router disabled")
        if not isinstance(model, str) or not model:
            return ModelDecision(model, model, matched=False, reason="no source model")

        for route in self._config.routes:
            if route.matches(model=model, input_tokens=input_tokens, has_tools=has_tools):
                reason = (
                    f"matched rule {route.name or route.to_model!r}: "
                    f"{model} -> {route.to_model} "
                    f"(input_tokens={input_tokens}, has_tools={has_tools})"
                )
                return ModelDecision(
                    original_model=model,
                    routed_model=route.to_model,
                    matched=True,
                    reason=reason,
                    rule_name=route.name,
                )
        return ModelDecision(model, model, matched=False, reason="no rule matched")


def estimate_input_tokens(messages: object, tools: object = None, system: object = None) -> int:
    """Cheap, tokenizer-free estimate of request input size, for routing only.

    Uses a ~4-chars-per-token heuristic over the serialized message, tool, and
    system content. ``system`` covers Anthropic's top-level ``system`` field
    (string or content-block list), which is not part of ``messages`` but can
    dominate request size, so omitting it would let a large system prompt route
    as if the request were tiny. This is deliberately approximate: it runs on
    the hot path purely to pick a route tier, so it must not pay for a real
    tokenizer. It never raises.
    """
    try:
        chars = 0
        if isinstance(messages, list):
            for msg in messages:
                chars += (
                    len(str(msg.get("content", ""))) if isinstance(msg, dict) else len(str(msg))
                )
        if tools:
            chars += len(str(tools))
        if system:
            chars += len(str(system))
        return chars // 4
    except Exception:  # noqa: BLE001 — estimation must never break the request path
        return 0


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}


def _parse_routes(routes_raw: str | None) -> tuple[ModelRoute, ...]:
    if not routes_raw or not routes_raw.strip():
        return ()
    try:
        parsed = json.loads(routes_raw)
    except (ValueError, TypeError) as exc:
        logger.warning("invalid HEADROOM_MODEL_ROUTES JSON; ignoring: %s", exc)
        return ()
    if not isinstance(parsed, list):
        logger.warning("HEADROOM_MODEL_ROUTES must be a JSON array; ignoring")
        return ()

    routes: list[ModelRoute] = []
    for i, entry in enumerate(parsed):
        route = _route_from_entry(entry, i)
        if route is not None:
            routes.append(route)
    return tuple(routes)


_INVALID = object()
"""Sentinel: a route field was present but malformed (fail open, skip the route)."""

_ALLOWED_ROUTE_KEYS = frozenset(
    {"to_model", "max_input_tokens", "min_input_tokens", "require_no_tools", "from_models", "name"}
)


def _route_from_entry(entry: object, index: int) -> ModelRoute | None:
    """Parse one route object, failing open (skip) on any malformed condition.

    A silently-broadened rule (e.g. an unparseable ``max_input_tokens`` treated
    as "no cap") could route far more traffic than the operator intended, so an
    invalid condition disables just that rule rather than widening it.
    """
    if not isinstance(entry, dict):
        logger.warning("model route #%d is not an object; skipping", index)
        return None
    unknown_keys = set(entry) - _ALLOWED_ROUTE_KEYS
    if unknown_keys:
        # A misspelled condition (e.g. "max_input_token") would otherwise be
        # ignored, silently widening the rule. Reject unknown keys instead.
        logger.warning(
            "model route #%d has unknown key(s) %s; skipping route", index, sorted(unknown_keys)
        )
        return None
    to_model = entry.get("to_model")
    if not isinstance(to_model, str) or not to_model:
        logger.warning("model route #%d missing string 'to_model'; skipping", index)
        return None

    max_tokens = _strict_opt_int(entry, "max_input_tokens", index)
    min_tokens = _strict_opt_int(entry, "min_input_tokens", index)
    if max_tokens is _INVALID or min_tokens is _INVALID:
        return None

    require_no_tools = entry.get("require_no_tools", False)
    if not isinstance(require_no_tools, bool):
        logger.warning(
            "model route #%d 'require_no_tools' must be a boolean; skipping route", index
        )
        return None

    from_models_raw = entry.get("from_models", [])
    if not isinstance(from_models_raw, list) or not all(
        isinstance(m, str) for m in from_models_raw
    ):
        logger.warning(
            "model route #%d 'from_models' must be a list of strings; skipping route", index
        )
        return None

    return ModelRoute(
        to_model=to_model,
        max_input_tokens=max_tokens,  # type: ignore[arg-type]
        min_input_tokens=min_tokens,  # type: ignore[arg-type]
        require_no_tools=require_no_tools,
        from_models=tuple(from_models_raw),
        name=str(entry.get("name", "")),
    )


def _strict_opt_int(entry: dict, key: str, index: int) -> int | None | object:
    """Return the int at ``key``, ``None`` if absent, or ``_INVALID`` if malformed.

    Accepts JSON integers and digit strings; rejects booleans, floats, and
    non-numeric values so a typo cannot silently remove a token bound.
    """
    if key not in entry or entry[key] is None:
        return None
    value = entry[key]
    if isinstance(value, bool):
        logger.warning("model route #%d '%s' must be an integer, not a boolean", index, key)
        return _INVALID
    if isinstance(value, int):
        parsed = value
    else:
        try:
            parsed = int(str(value))
        except (ValueError, TypeError):
            logger.warning("model route #%d '%s' is not a valid integer", index, key)
            return _INVALID
    if parsed < 0:
        logger.warning("model route #%d '%s' must be non-negative", index, key)
        return _INVALID
    return parsed
