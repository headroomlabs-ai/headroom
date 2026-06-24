from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass

from headroom.pipeline import (
    CANONICAL_PIPELINE_STAGES,
    ENTRY_POINT_GROUP,
    PipelineEvent,
    PipelineExtensionManager,
    PipelineStage,
    discover_pipeline_extensions,
    summarize_routing_markers,
)


@dataclass
class FakeEntryPoint:
    name: str
    value: object

    def load(self):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def test_discover_pipeline_extensions_handles_load_and_init_failures(
    monkeypatch,
) -> None:
    class WorkingExtension:
        def on_pipeline_event(self, event: PipelineEvent):  # noqa: ANN001, ANN201
            return event

    class NeedsInit:
        def __init__(self) -> None:
            raise RuntimeError("bad init")

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group=None: (
            [
                FakeEntryPoint("working-instance", WorkingExtension()),
                FakeEntryPoint("working-class", WorkingExtension),
                FakeEntryPoint("bad-load", RuntimeError("bad load")),
                FakeEntryPoint("bad-init", NeedsInit),
            ]
            if group == ENTRY_POINT_GROUP
            else []
        ),
    )

    discovered = discover_pipeline_extensions()
    assert len(discovered) == 2
    assert all(callable(getattr(ext, "on_pipeline_event", None)) for ext in discovered)


def test_discover_pipeline_extensions_handles_enumeration_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group=None: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert discover_pipeline_extensions() == []


def test_pipeline_manager_emit_and_summary(monkeypatch) -> None:
    class Hook:
        def __init__(self) -> None:
            self.seen: list[str] = []

        def on_pipeline_event(self, event: PipelineEvent):  # noqa: ANN001, ANN201
            self.seen.append(event.stage.value)
            event.metadata["hook"] = True
            return event

    class ReplacingExtension:
        def on_pipeline_event(self, event: PipelineEvent):  # noqa: ANN001, ANN201
            return PipelineEvent(
                stage=event.stage,
                operation=event.operation,
                request_id=event.request_id,
                provider=event.provider,
                model=event.model,
                messages=event.messages,
                tools=event.tools,
                headers=event.headers,
                response=event.response,
                metadata={**event.metadata, "replaced": True},
            )

    class BrokenExtension:
        def on_pipeline_event(self, event: PipelineEvent):  # noqa: ANN001, ANN201
            raise RuntimeError("boom")

    hook = Hook()
    monkeypatch.setattr(
        "headroom.pipeline.discover_pipeline_extensions",
        lambda: [BrokenExtension()],
    )

    manager = PipelineExtensionManager(
        hooks=hook,
        extensions=[object(), ReplacingExtension()],
        discover=True,
    )

    assert manager.enabled is True
    event = manager.emit(
        PipelineStage.INPUT_RECEIVED,
        operation="compress",
        request_id="req-1",
        provider="openai",
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        metadata={"start": True},
    )

    assert hook.seen == ["input_received"]
    assert event.metadata == {"start": True, "hook": True, "replaced": True}
    assert event.request_id == "req-1"

    disabled = PipelineExtensionManager(discover=False)
    assert disabled.enabled is False

    assert summarize_routing_markers(["router:smart", "other", "router:cheap"]) == [
        "router:smart",
        "router:cheap",
    ]
    assert PipelineStage.SETUP in CANONICAL_PIPELINE_STAGES
    assert PipelineStage.RESPONSE_RECEIVED in CANONICAL_PIPELINE_STAGES
