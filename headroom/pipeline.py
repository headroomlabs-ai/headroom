"""Canonical Headroom pipeline lifecycle and extension contracts."""

from __future__ import annotations

import importlib.metadata
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "headroom.pipeline_extension"


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
)


@dataclass
class PipelineEvent:
    """Event emitted at a canonical pipeline stage.

    Extensions may mutate ``messages``, ``tools``, ``headers``, or ``metadata`` in
    place, or return a replacement ``PipelineEvent`` from ``on_pipeline_event``.
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
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineExtension(Protocol):
    """Request lifecycle extension contract for the canonical pipeline."""

    def on_pipeline_event(self, event: PipelineEvent) -> PipelineEvent | None:
        """Handle a canonical pipeline event."""


def discover_pipeline_extensions() -> list[PipelineExtension]:
    """Load registered pipeline extensions from Python entry points."""

    discovered: list[PipelineExtension] = []
    try:
        entries = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception as exc:  # noqa: BLE001 - importlib metadata varies by runtime
        log.debug("pipeline extensions: entry-point enumeration failed: %s", exc)
        return discovered

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
