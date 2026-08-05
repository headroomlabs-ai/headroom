"""Turn hooks — observe and optionally re-drive a single model turn.

A *turn hook* wraps the proxy's buffered upstream call: it sees the outbound
request and the model's response, and may call the model again (via
``call_model``) to return a *replacement* response — transparently to the client.
That covers a range of turn-level behaviors an extension might want: resolving an
injected tool call, enforcing a guardrail, retrying on a bad response, serving a
cached answer, or running a small model→proxy→model loop before handing back the
final answer.

Hooks are registered by opt-in proxy extensions (``proxy/extensions.py``). This
module is **inert unless a hook is registered**: the runner helpers return their
input unchanged, so with no hooks the proxy behaves exactly as if this module did
not exist. Hooks must never raise — a failing hook is logged and skipped so it
cannot take the proxy down.

Stability: the ``TurnHook`` protocol and the registry/runner functions are part
of the extension surface (see ``proxy/extensions.py``); signature changes follow
the same deprecation policy.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)

# Re-drive the model with a message list; returns the provider's response JSON.
# The exact message/response shapes are provider-native (Anthropic / OpenAI /
# Google), matching whatever the surrounding handler already works with.
CallModel = Callable[[list[dict[str, Any]]], Awaitable[dict[str, Any]]]


@dataclass
class TurnContext:
    """The request side of one model turn, as the proxy forwards it upstream.

    ``tools`` and ``messages`` are the live objects the handler is about to send
    (or just sent); a hook's ``on_request`` may mutate them in place.
    """

    provider: str  # "anthropic" | "openai" | "google" | ...
    model: str
    messages: list[dict[str, Any]]
    tools: Any = None  # provider-native tools value (list, or None)
    config: Any = None


@runtime_checkable
class TurnHook(Protocol):
    """A registered turn observer. Both methods are optional (a hook may define
    either); missing methods are simply skipped."""

    name: str

    # Optional: set True if this hook's ``on_request`` needs no later re-drive
    # (a fold-only hook). Such hooks run on streaming turns too; hooks that may
    # re-drive in ``on_response`` leave it unset and run buffered-only. Absent ⇒
    # False (conservative). See :func:`run_request_hooks`.
    stream_safe: bool

    def on_request(self, ctx: TurnContext) -> None:
        """Inspect / mutate ``ctx`` (e.g. ``ctx.tools``) before it goes upstream."""

    async def on_response(
        self, ctx: TurnContext, response: dict[str, Any], call_model: CallModel
    ) -> dict[str, Any] | None:
        """Return a replacement response, or ``None`` to leave it unchanged.

        May ``await call_model(messages)`` to re-drive the model (e.g. to resolve
        an injected tool call), looping as needed before returning the final."""


_hooks: list[TurnHook] = []


def register_turn_hook(hook: TurnHook) -> None:
    """Register a hook. Called by an extension's ``install(app, config)``."""
    _hooks.append(hook)
    log.info("registered turn hook: %s", getattr(hook, "name", type(hook).__name__))


def registered_turn_hooks() -> list[TurnHook]:
    return list(_hooks)


def clear_turn_hooks() -> None:
    """Test/reset helper."""
    _hooks.clear()


def run_request_hooks(ctx: TurnContext, *, stream_safe_only: bool = False) -> None:
    """Run each hook's ``on_request``. Inert when none registered; never raises.

    ``on_request`` mutates the outbound request before the upstream send, so it is
    safe on a streamed turn *as long as the hook needs no later re-drive*. A hook
    that may re-drive the model in ``on_response`` (e.g. defer a tool, then reload
    it when the model asks) can't run on a stream — the bytes are already flowing,
    there's nothing to re-drive. So on streaming turns the handler passes
    ``stream_safe_only=True`` and only hooks that opt in via a truthy ``stream_safe``
    attribute run; fold-only hooks (which never re-drive) set it and thus keep
    working on streamed OpenAI-compatible traffic. Default off ⇒ conservative:
    a hook is treated as buffered-only unless it declares itself stream-safe.
    """
    for hook in _hooks:
        if stream_safe_only and not getattr(hook, "stream_safe", False):
            continue
        fn = getattr(hook, "on_request", None)
        if fn is None:
            continue
        try:
            fn(ctx)
        except Exception:  # a hook must never break the proxy
            log.exception("turn hook %r on_request failed", getattr(hook, "name", hook))


async def run_response_hooks(
    ctx: TurnContext, response: dict[str, Any], call_model: CallModel
) -> dict[str, Any]:
    """Run every hook's ``on_response``, chaining any replacements.

    Returns the (possibly replaced) response. Inert when no hooks are registered
    (returns ``response`` unchanged); a failing hook is logged and skipped.
    """
    current = response
    for hook in _hooks:
        fn = getattr(hook, "on_response", None)
        if fn is None:
            continue
        try:
            replacement = await fn(ctx, current, call_model)
        except Exception:
            log.exception("turn hook %r on_response failed", getattr(hook, "name", hook))
            continue
        if replacement is not None:
            current = replacement
    return current
