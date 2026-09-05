"""Canonical Headroom pipeline lifecycle and extension contracts."""

from __future__ import annotations

import importlib.metadata
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "headroom.pipeline_extension"
ENV_VAR = "HEADROOM_PIPELINE_EXTENSIONS"


class PipelineStage(str, Enum):
    """Stable lifecycle stages for the canonical Headroom pipeline."""

    SETUP = "setup"
    PRE_START = "pre_start"
    POST_START = "post_start"
    INPUT_RECEIVED = "input_received"
    INPUT_CACHED = "input_cached"
    INPUT_ROUTED = "input_routed"
    INPUT_COMPRESSED = "input_compressed"
    INPUT_REMEMBERED = "input_remembered"
    PRE_SEND = "pre_send"
    POST_SEND = "post_send"
    RESPONSE_RECEIVED = "response_received"
    OUTCOME_OBSERVED = "outcome_observed"


CANONICAL_PIPELINE_STAGES: tuple[PipelineStage, ...] = (
    PipelineStage.SETUP,
    PipelineStage.PRE_START,
    PipelineStage.POST_START,
    PipelineStage.INPUT_RECEIVED,
    PipelineStage.INPUT_CACHED,
    PipelineStage.INPUT_ROUTED,
    PipelineStage.INPUT_COMPRESSED,
    PipelineStage.INPUT_REMEMBERED,
    PipelineStage.PRE_SEND,
    PipelineStage.POST_SEND,
    PipelineStage.RESPONSE_RECEIVED,
    PipelineStage.OUTCOME_OBSERVED,
)


@dataclass(frozen=True)
class OutcomeSnapshot:
    """What actually happened on one request. Read-only by construction.

    Emitted at :attr:`PipelineStage.OUTCOME_OBSERVED` so an extension can learn
    from a response — a token ceiling that was hit, an effort setting that did
    or did not pay off — without being able to rewrite the measurement it is
    learning from. Extensions declare what they did; the core records what
    happened; attribution is the core's arithmetic over both.

    ``thinking_tokens`` is ``None`` when the provider reported no split and none
    could be inferred. That is NOT zero: Anthropic reports no thinking count at
    all, so treating unknown as zero would credit visible-text levers with
    reductions that reasoning-effort levers produced.
    """

    request_id: str = ""
    provider: str = ""
    model: str = ""
    output_tokens: int = 0
    thinking_tokens: int | None = None
    thinking_inferred: bool = False
    stop_reason: str | None = None
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    turn_index: int = 0
    transforms_applied: tuple[str, ...] = ()

    @property
    def truncated(self) -> bool:
        """Whether a token ceiling cut the response off.

        The one unambiguous, provider-supplied feedback signal available to an
        adaptive lever — which is why a ceiling can run a closed control loop
        where verbosity steering (whose signals need transcript inference)
        cannot.
        """
        return self.stop_reason in ("max_tokens", "length")

    @property
    def visible_output_tokens(self) -> int | None:
        """Output tokens excluding thinking, or ``None`` when unknown."""
        if self.thinking_tokens is None:
            return None
        return max(0, self.output_tokens - self.thinking_tokens)


@dataclass
class PipelineEvent:
    """Event emitted at a canonical pipeline stage.

    Extensions may mutate ``messages``, ``tools``, ``headers`` or ``metadata``
    in place, or return a replacement ``PipelineEvent`` from
    ``on_pipeline_event``.
    """

    stage: PipelineStage
    operation: str
    request_id: str = ""
    provider: str = ""
    model: str = ""
    messages: list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    headers: dict[str, str] | None = None
    response: Any = None
    outcome: OutcomeSnapshot | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineExtension(Protocol):
    """Request lifecycle extension contract for the canonical pipeline."""

    def on_pipeline_event(self, event: PipelineEvent) -> PipelineEvent | None:
        """Handle a canonical pipeline event."""


def _resolve_enabled(enabled: Iterable[str] | None) -> set[str]:
    """Resolve the enabled entry-point names.

    Precedence: the explicit argument, else ``HEADROOM_PIPELINE_EXTENSIONS``,
    else nothing. Mirrors :mod:`headroom.proxy.extensions` so operators learn
    one rule for both seams.
    """
    raw: Iterable[str]
    if enabled is not None:
        raw = enabled
    else:
        raw = (os.environ.get(ENV_VAR) or "").split(",")
    return {n.strip() for n in raw if n and n.strip()}


def discover_pipeline_extensions(
    enabled: Iterable[str] | None = None,
) -> list[PipelineExtension]:
    """Load explicitly-enabled pipeline extensions from Python entry points.

    **Opt-in.** Discovery enumerates every registered entry point, but only
    those the operator named are loaded. Merely installing a package — as a
    transitive dependency, say — must not silently start rewriting requests.

    An unaudited package in the same environment could otherwise rewrite the
    messages of every request on live traffic. The proxy-extension seam has
    always worked this way; this brings the pipeline seam in line.

    Extensions passed directly to :class:`PipelineExtensionManager` are
    unaffected — constructing one and handing it over is already explicit.

    Enable with ``HEADROOM_PIPELINE_EXTENSIONS=name1,name2``; the literal ``*``
    enables everything discovered (only where every package is trusted).
    """

    discovered: list[PipelineExtension] = []
    try:
        entries = list(importlib.metadata.entry_points(group=ENTRY_POINT_GROUP))
    except Exception as exc:  # noqa: BLE001 - importlib metadata varies by runtime
        log.debug("pipeline extensions: entry-point enumeration failed: %s", exc)
        return discovered

    enabled_set = _resolve_enabled(enabled)
    if not enabled_set:
        if entries:
            log.info(
                "pipeline extensions discovered but disabled (opt-in): %s. "
                "Enable with %s=<name1,name2>.",
                ",".join(sorted(e.name for e in entries)),
                ENV_VAR,
            )
        return discovered

    wildcard = "*" in enabled_set
    if not wildcard:
        missing = enabled_set - {e.name for e in entries}
        if missing:
            log.warning(
                "pipeline extensions requested but not found: %s (available: %s)",
                ",".join(sorted(missing)),
                ",".join(sorted(e.name for e in entries)) or "<none>",
            )
        entries = [e for e in entries if e.name in enabled_set]

    for entry in entries:
        try:
            extension = entry.load()
        except Exception as exc:  # noqa: BLE001 - third-party load failures are isolated
            log.warning("pipeline extension %r failed to load: %s", entry.name, exc)
            continue

        if isinstance(extension, type):
            try:
                extension = extension()
            except Exception as exc:  # noqa: BLE001
                log.warning("pipeline extension %r failed to initialize: %s", entry.name, exc)
                continue

        discovered.append(extension)

    return discovered


def summarize_routing_markers(transforms_applied: list[str]) -> list[str]:
    """Return the routed transform markers emitted by ContentRouter."""

    return [item for item in transforms_applied if item.startswith("router:")]


class PipelineExtensionManager:
    """Dispatch canonical pipeline events to configured extensions."""

    def __init__(
        self,
        *,
        hooks: Any = None,
        extensions: list[Any] | None = None,
        discover: bool = True,
    ) -> None:
        resolved: list[Any] = []
        if hooks is not None and callable(getattr(hooks, "on_pipeline_event", None)):
            resolved.append(hooks)
        if extensions:
            resolved.extend(extensions)
        if discover:
            resolved.extend(discover_pipeline_extensions())
        self._extensions = resolved

    @property
    def enabled(self) -> bool:
        return bool(self._extensions)

    def emit(
        self,
        stage: PipelineStage,
        *,
        operation: str,
        request_id: str = "",
        provider: str = "",
        model: str = "",
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        headers: dict[str, str] | None = None,
        response: Any = None,
        outcome: OutcomeSnapshot | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PipelineEvent:
        """Emit a canonical lifecycle event and return the final event state."""

        event = PipelineEvent(
            stage=stage,
            operation=operation,
            request_id=request_id,
            provider=provider,
            model=model,
            messages=messages,
            tools=tools,
            headers=headers,
            response=response,
            outcome=outcome,
            metadata=metadata or {},
        )

        for extension in self._extensions:
            handler = getattr(extension, "on_pipeline_event", None)
            if not callable(handler):
                continue
            try:
                updated = handler(event)
            except Exception as exc:  # noqa: BLE001 - preserve hook fail-open behavior
                log.warning(
                    "pipeline extension %r failed during %s: %s",
                    type(extension).__name__,
                    stage.value,
                    exc,
                )
                continue
            if isinstance(updated, PipelineEvent):
                event = updated

        return event
