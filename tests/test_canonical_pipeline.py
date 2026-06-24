from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

from headroom.client import HeadroomClient
from headroom.compress import compress
from headroom.config import HeadroomConfig, HeadroomMode, TransformResult
from headroom.hooks import CompressionHooks
from headroom.pipeline import (
    CANONICAL_PIPELINE_STAGES,
    PipelineExtensionManager,
    PipelineStage,
    summarize_routing_markers,
)
from headroom.providers.base import Provider, TokenCounter
from headroom.transforms import ContentRouter, TransformPipeline
from headroom.transforms.content_router import CompressionStrategy


class RecordingExtension:
    def __init__(self) -> None:
        self.stages: list[PipelineStage] = []

    def on_pipeline_event(self, event):
        self.stages.append(event.stage)
        return None


class MutatingExtension:
    def on_pipeline_event(self, event):
        if event.stage == PipelineStage.INPUT_RECEIVED:
            event.messages = [{"role": "user", "content": "mutated"}]
        return event


class ReplacingExtension:
    def on_pipeline_event(self, event):
        return type(event)(
            stage=event.stage,
            operation=event.operation,
            model=event.model,
            messages=[{"role": "user", "content": "replaced"}],
            metadata={"replaced": True},
        )


class RaisingExtension:
    def on_pipeline_event(self, event):
        raise RuntimeError("boom")


class RecordingHooks(CompressionHooks):
    def __init__(self) -> None:
        self.stages: list[PipelineStage] = []
        self.post_event = None

    def pre_compress(self, messages, ctx):
        return messages

    def compute_biases(self, messages, ctx):
        return {}

    def post_compress(self, event):
        self.post_event = event

    def on_pipeline_event(self, event):
        self.stages.append(event.stage)
        return None


class StubPipeline:
    def apply(self, messages, model, **kwargs):
        return TransformResult(
            messages=messages,
            tokens_before=20,
            tokens_after=8,
            transforms_applied=["router:text:kompress", "kompress:user:0.40"],
        )

    def _get_tokenizer(self, model):
        return StubTokenCounter()


class StubTokenCounter(TokenCounter):
    def count_text(self, text: str) -> int:
        return len(text.split())

    def count_message(self, message: dict[str, Any]) -> int:
        content = message.get("content", "")
        if isinstance(content, str):
            return len(content.split())
        return 1

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        return sum(self.count_message(message) for message in messages)


class StubProvider(Provider):
    @property
    def name(self) -> str:
        return "openai"

    def get_token_counter(self, model: str) -> TokenCounter:
        return StubTokenCounter()

    def get_context_limit(self, model: str) -> int:
        return 128000

    def supports_model(self, model: str) -> bool:
        return True


class DummyCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"id": "resp_123", "messages": kwargs["messages"]}


class DummyOriginalClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=DummyCompletions())


def test_pipeline_extension_manager_uses_canonical_stage_contract():
    recorder = RecordingExtension()
    manager = PipelineExtensionManager(
        extensions=[recorder, MutatingExtension()],
        discover=False,
    )

    event = manager.emit(
        PipelineStage.INPUT_RECEIVED,
        operation="test",
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert list(CANONICAL_PIPELINE_STAGES)[0] is PipelineStage.SETUP
    assert summarize_routing_markers(["router:text:kompress", "smart:kept=3"]) == [
        "router:text:kompress"
    ]
    assert recorder.stages == [PipelineStage.INPUT_RECEIVED]
    assert event.messages == [{"role": "user", "content": "mutated"}]


def test_default_transform_pipeline_always_uses_content_router() -> None:
    config = HeadroomConfig()

    pipeline = TransformPipeline(config)

    assert any(isinstance(transform, ContentRouter) for transform in pipeline.transforms)
    assert not any(type(transform).__name__ == "SmartCrusher" for transform in pipeline.transforms)


def test_content_router_protects_instruction_roles_but_compresses_tool_outputs() -> None:
    class Tokenizer:
        def count_text(self, text: str) -> int:
            return max(1, len(text.split()))

    router = ContentRouter()
    calls: list[str] = []

    def fake_compress(text: str, **kwargs: Any) -> SimpleNamespace:
        calls.append(text)
        return SimpleNamespace(
            compressed="COMPRESSED",
            compression_ratio=0.1,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    router.compress = fake_compress  # type: ignore[method-assign]
    tool_text = "tool output " * 120
    messages = [
        {"role": "system", "content": "system instructions " * 120},
        {"role": "developer", "content": "developer instructions " * 120},
        {"role": "user", "content": "user prompt " * 120},
        {"role": "tool", "tool_call_id": "call_1", "content": tool_text},
    ]

    result = router.apply(messages, Tokenizer())

    assert result.messages[0]["content"] == messages[0]["content"]
    assert result.messages[1]["content"] == messages[1]["content"]
    assert result.messages[2]["content"] == messages[2]["content"]
    assert result.messages[3]["content"] == "COMPRESSED"
    assert calls == [tool_text]


def test_pipeline_extension_manager_replaces_events_and_ignores_failures(caplog):
    recorder = RecordingExtension()
    manager = PipelineExtensionManager(
        extensions=[recorder, RaisingExtension(), ReplacingExtension(), object()],
        discover=False,
    )

    with caplog.at_level("WARNING", logger="headroom.pipeline"):
        event = manager.emit(
            PipelineStage.PRE_SEND,
            operation="test",
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert manager.enabled is True
    assert recorder.stages == [PipelineStage.PRE_SEND]
    assert event.messages == [{"role": "user", "content": "replaced"}]
    assert event.metadata == {"replaced": True}


def test_discover_pipeline_extensions_handles_load_and_init_failures(monkeypatch):
    pipeline_module = importlib.import_module("headroom.pipeline")

    class Entry:
        def __init__(self, name, loader):
            self.name = name
            self._loader = loader

        def load(self):
            return self._loader()

    class ExtensionClass:
        def on_pipeline_event(self, event):
            return event

    class FailingInit:
        def __init__(self):
            raise RuntimeError("init failed")

    entries = [
        Entry("instance", lambda: RecordingExtension()),
        Entry("class", lambda: ExtensionClass),
        Entry("load-fail", lambda: (_ for _ in ()).throw(RuntimeError("load failed"))),
        Entry("init-fail", lambda: FailingInit),
    ]

    monkeypatch.setattr(
        pipeline_module.importlib.metadata,
        "entry_points",
        lambda group: entries if group == pipeline_module.ENTRY_POINT_GROUP else [],
    )

    discovered = pipeline_module.discover_pipeline_extensions()

    assert len(discovered) == 2
    assert hasattr(discovered[0], "on_pipeline_event")
    assert hasattr(discovered[1], "on_pipeline_event")


def test_discover_pipeline_extensions_returns_empty_when_entrypoint_lookup_fails(monkeypatch):
    pipeline_module = importlib.import_module("headroom.pipeline")
    monkeypatch.setattr(
        pipeline_module.importlib.metadata,
        "entry_points",
        lambda group: (_ for _ in ()).throw(RuntimeError("lookup failed")),
    )

    assert pipeline_module.discover_pipeline_extensions() == []


def test_compress_emits_canonical_pipeline_events(monkeypatch):
    hooks = RecordingHooks()
    compress_module = importlib.import_module("headroom.compress")
    monkeypatch.setattr(compress_module, "_get_pipeline", lambda: StubPipeline())

    result = compress(
        [{"role": "user", "content": "hello world"}],
        model="gpt-4o",
        hooks=hooks,
    )

    assert result.tokens_before == 20
    assert result.tokens_after == 8
    assert hooks.post_event is not None
    assert hooks.post_event.tokens_saved == 12
    assert hooks.stages == [
        PipelineStage.INPUT_RECEIVED,
        PipelineStage.INPUT_ROUTED,
        PipelineStage.INPUT_COMPRESSED,
    ]


def test_headroom_client_emits_canonical_pipeline_events(tmp_path):
    recorder = RecordingExtension()
    original = DummyOriginalClient()
    config = HeadroomConfig(
        store_url=f"jsonl://{tmp_path / 'headroom.jsonl'}",
        default_mode=HeadroomMode.OPTIMIZE,
        pipeline_extensions=[recorder],
        discover_pipeline_extensions=False,
    )
    client = HeadroomClient(
        original_client=original,
        provider=StubProvider(),
        store_url=f"jsonl://{tmp_path / 'headroom-client.jsonl'}",
        enable_cache_optimizer=False,
        config=config,
    )
    client._pipeline = StubPipeline()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello world"}],
    )

    assert response["id"] == "resp_123"
    assert recorder.stages == [
        PipelineStage.SETUP,
        PipelineStage.INPUT_RECEIVED,
        PipelineStage.INPUT_ROUTED,
        PipelineStage.INPUT_COMPRESSED,
        PipelineStage.PRE_SEND,
        PipelineStage.POST_SEND,
        PipelineStage.RESPONSE_RECEIVED,
    ]
