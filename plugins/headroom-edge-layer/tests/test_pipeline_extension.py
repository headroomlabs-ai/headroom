from __future__ import annotations

import pytest
from headroom.cache.backends.memory import InMemoryBackend
from headroom.cache.compression_store import CompressionStore
from headroom.pipeline import PipelineExtensionManager, PipelineStage

from headroom_edge_layer.config import EdgeLayerConfig
from headroom_edge_layer.pipeline_extension import EdgeLayerExtension


@pytest.fixture
def manager() -> tuple[PipelineExtensionManager, CompressionStore]:
    store = CompressionStore(backend=InMemoryBackend())
    extension = EdgeLayerExtension(EdgeLayerConfig(), store=store)
    return PipelineExtensionManager(extensions=[extension], discover=False), store


def _grep_messages(n: int) -> list[dict]:
    text = "\n".join(f"foo.py:{i}:def line_{i}(): pass  # matches bar" for i in range(1, n + 1))
    return [
        {"role": "assistant", "content": [
            {"type": "text", "text": "grep for bar"},
            {"type": "tool_use", "id": "tu1", "name": "Grep", "input": {"pattern": "bar"}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": [{"type": "text", "text": text}]}]},
    ]


def test_only_acts_on_input_received_stage(manager) -> None:
    mgr, _store = manager
    messages = _grep_messages(60)
    event = mgr.emit(PipelineStage.PRE_SEND, operation="t", request_id="r1", messages=messages)
    # PRE_SEND is not INPUT_RECEIVED — the extension must no-op, leaving the
    # manager's own passthrough event with the original messages untouched.
    assert event.messages == messages


def test_grep_is_compressed_at_input_received(manager) -> None:
    mgr, _store = manager
    messages = _grep_messages(60)
    event = mgr.emit(PipelineStage.INPUT_RECEIVED, operation="t", request_id="r1", messages=messages)
    result_text = event.messages[1]["content"][0]["content"][0]["text"]
    assert "Retrieve more: hash=" in result_text
    assert len(result_text) < len(messages[1]["content"][0]["content"][0]["text"])


def test_none_messages_is_a_no_op(manager) -> None:
    mgr, _store = manager
    event = mgr.emit(PipelineStage.INPUT_RECEIVED, operation="t", request_id="r1", messages=None)
    assert event.messages is None


def test_per_edge_flags_disable_individually() -> None:
    store = CompressionStore(backend=InMemoryBackend())
    config = EdgeLayerConfig(enable_grep=False)
    extension = EdgeLayerExtension(config, store=store)
    mgr = PipelineExtensionManager(extensions=[extension], discover=False)
    messages = _grep_messages(60)
    event = mgr.emit(PipelineStage.INPUT_RECEIVED, operation="t", request_id="r1", messages=messages)
    result_text = event.messages[1]["content"][0]["content"][0]["text"]
    assert result_text == messages[1]["content"][0]["content"][0]["text"]  # grep edge disabled -> unchanged


def test_thinking_blocks_pass_through_untouched(manager) -> None:
    mgr, _store = manager
    thinking_block = {"type": "thinking", "thinking": "must never be read or written", "signature": "sig123"}
    messages = [
        {"role": "assistant", "content": [thinking_block, {"type": "text", "text": "answer"}]},
    ]
    event = mgr.emit(PipelineStage.INPUT_RECEIVED, operation="t", request_id="r1", messages=messages)
    assert event.messages[0]["content"][0] == thinking_block


def test_deterministic_output_for_identical_input(manager) -> None:
    mgr, _store = manager
    messages = _grep_messages(60)
    event_1 = mgr.emit(PipelineStage.INPUT_RECEIVED, operation="t", request_id="r1", messages=messages)
    event_2 = mgr.emit(PipelineStage.INPUT_RECEIVED, operation="t", request_id="r2", messages=messages)
    assert event_1.messages == event_2.messages


def test_extension_failure_fails_open_per_pipeline_manager() -> None:
    """The pipeline manager itself catches extension exceptions and skips
    that stage rather than failing the request — verified here so a bug in
    one edge can't take an unrelated request down.
    """

    class _Explode:
        def on_pipeline_event(self, event):
            raise RuntimeError("boom")

    mgr = PipelineExtensionManager(extensions=[_Explode()], discover=False)
    messages = _grep_messages(5)
    event = mgr.emit(PipelineStage.INPUT_RECEIVED, operation="t", request_id="r1", messages=messages)
    assert event.messages == messages
