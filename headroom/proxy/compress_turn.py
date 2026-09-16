"""Compress-turn extensions: the seam that lets a gateway contract live outside
``POST /v1/compress``'s handler.

The compress handler does one thing well — it compresses a message list, in
legacy or session mode — and it exposes five moments to an extension that
wants to wrap that into a richer, gateway-facing turn:

1. ``begin`` — the handler has parsed the body; the extension decides whether
   this request is one of its (a gateway would look for its ``gateway`` block),
   validates its own fields (raise :class:`CompressTurnError` for a 400) and
   returns a per-request :class:`CompressTurn`, or ``None`` to stay out.
2. ``prepare`` — model, tags and config are resolved; the turn may build
   whatever it needs for the executor (a tokenizer, hooks, a scope binding).
3. ``transform`` — EXECUTOR SIDE. Called once, after the pipeline (legacy) or
   after the session engine's prefix replay (session mode) and before the
   session tracker records what was returned. What this returns IS what the
   caller forwards to the provider, so it must be a pure function of its input
   in session mode or it busts the replayed prefix.
4. ``finish`` — back on the event loop with the final messages and counts:
   the turn adds its own transform labels, extra response fields and may
   replace the outgoing messages (a gateway builds the provider ``body`` here).
5. ``commit`` — the handler has built the :class:`RequestOutcome`; the turn
   may take ownership of recording it (return ``True`` to defer) when it is
   waiting for more information, such as a gateway's relayed usage.

Plus ``fail_open_fields`` for the three fail-open answers (bypass header,
empty messages, compression timeout), where the caller must still get
whatever it needs to forward the original request.

Exactly one extension claims a request: the first whose ``begin`` returns a
turn. The built-in gateway turn contract (``headroom.proxy.gateway_extension``)
is registered through this same seam, so a third-party contract has nothing
core does not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from headroom.proxy.outcome import RequestOutcome

log = logging.getLogger(__name__)


class CompressTurnError(ValueError):
    """Raised from ``begin`` for a malformed extension-specific request; the
    handler answers 400 ``invalid_request`` with ``message`` — a validation
    sentence the extension wrote for the caller, never an exception's text."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = str(message)


@dataclass
class FinishedTurn:
    """What ``finish`` hands back to the handler."""

    #: Extra top-level keys merged into the JSON answer (a gateway's ``body``,
    #: ``turn_id``, ``route``, ``obligations``, ``headers`` …).
    fields: dict[str, Any] = field(default_factory=dict)
    #: The complete ``transforms_applied`` list for the answer AND the outcome
    #: (the pipeline's plus the turn's own).
    transforms: list[str] = field(default_factory=list)
    #: The messages to return; ``None`` keeps the handler's.
    messages: list[dict[str, Any]] | None = None


@runtime_checkable
class CompressTurn(Protocol):
    """One request's turn, created by :meth:`CompressTurnExtension.begin`."""

    def fail_open_fields(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Fields to merge into a fail-open answer (originals, no obligations)."""

    def prepare(self, *, model_name: str, tags: dict[str, Any], config: Any) -> None:
        """Called once before the executor with the resolved model and tags."""

    def transform(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Executor side; see the module docstring for the exact moment."""

    @property
    def folded_messages(self) -> bool:
        """Whether ``transform`` changed the messages' token count (the handler
        must then recount ``tokens_after``)."""

    def count_messages(self, messages: list[dict[str, Any]], fallback: int) -> int:
        """Recount with the turn's own tokenizer; ``fallback`` on failure."""

    def finish(
        self,
        *,
        messages: list[dict[str, Any]],
        tokens_before: int,
        transforms_applied: list[str],
        mode: str | None,
        ccr_hashes: list[str],
    ) -> FinishedTurn:
        """Event-loop side, after the executor; before the outcome is built."""

    def commit(
        self,
        outcome: RequestOutcome,
        *,
        session_key: str | None,
        session_id: str | None,
    ) -> bool:
        """The outcome is built. Return ``True`` to take ownership of recording
        it later; ``False`` and the handler records it now."""


@runtime_checkable
class CompressTurnExtension(Protocol):
    name: str

    def begin(
        self,
        *,
        proxy: Any,
        request: Any,
        body: dict[str, Any],
        client: str | None,
    ) -> CompressTurn | None:
        """Claim the request (return a turn) or decline (``None``). Raise
        :class:`CompressTurnError` for a 400."""


def _registry(proxy: Any) -> list[CompressTurnExtension]:
    """Per-proxy registry (one proxy per app, so two apps in one process — the
    test suite, an embedded proxy — never see each other's contracts)."""
    registry = getattr(proxy, "compress_turn_extensions", None)
    if registry is None:
        registry = []
        try:
            proxy.compress_turn_extensions = registry
        except Exception:  # a proxy that cannot hold attributes: nothing installs
            return []
    return registry


def register_compress_turn_extension(proxy: Any, extension: CompressTurnExtension) -> None:
    """Register a contract on ``proxy`` (``app.state.proxy``). Called by an
    extension's ``install(app, config)``."""
    _registry(proxy).append(extension)
    log.info("registered compress-turn extension: %s", getattr(extension, "name", extension))


def unregister_compress_turn_extension(proxy: Any, extension: CompressTurnExtension) -> None:
    registry = _registry(proxy)
    while extension in registry:
        registry.remove(extension)


def registered_compress_turn_extensions(proxy: Any) -> list[CompressTurnExtension]:
    return list(_registry(proxy))


def begin_compress_turn(
    *, proxy: Any, request: Any, body: dict[str, Any], client: str | None
) -> CompressTurn | None:
    """The first extension that claims the body wins. A ``CompressTurnError``
    propagates (the handler answers 400); any other exception disables that
    extension for this request only (fail-open, logged)."""
    for extension in registered_compress_turn_extensions(proxy):
        try:
            turn = extension.begin(proxy=proxy, request=request, body=body, client=client)
        except CompressTurnError:
            raise
        except Exception:  # an extension must never break /v1/compress
            log.exception(
                "compress-turn extension %r failed to begin; ignored for this request",
                getattr(extension, "name", extension),
            )
            continue
        if turn is not None:
            return turn
    return None
