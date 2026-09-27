from __future__ import annotations

import pytest
from headroom.cache.backends.memory import InMemoryBackend
from headroom.cache.compression_store import CompressionStore

from headroom_edge_layer.config import EdgeLayerConfig
from headroom_edge_layer.file_read_edge import compress_file_reads


@pytest.fixture
def store() -> CompressionStore:
    return CompressionStore(backend=InMemoryBackend())


@pytest.fixture
def config() -> EdgeLayerConfig:
    return EdgeLayerConfig()


def _read_messages(text_1: str, text_2: str) -> list[dict]:
    return [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "r1", "name": "Read", "input": {"file_path": "/a.py"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "r1", "content": [{"type": "text", "text": text_1}]}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "checking again"},
            {"type": "tool_use", "id": "r2", "name": "Read", "input": {"file_path": "/a.py"}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "r2", "content": [{"type": "text", "text": text_2}]}]},
    ]


def _numbered_lines(n: int) -> str:
    return "\n".join(f"   {i}\tdef line_{i}(): pass" for i in range(1, n + 1))


def test_identical_repeat_read_becomes_unchanged_marker(store, config) -> None:
    content = _numbered_lines(50)
    messages = _read_messages(content, content)
    result, ledger = compress_file_reads(messages, config=config, store=store, request_id="r1")

    first_read_text = result[1]["content"][0]["content"][0]["text"]
    second_read_text = result[3]["content"][0]["content"][0]["text"]
    assert first_read_text == content  # earlier copy untouched
    assert "Retrieve original: hash=" in second_read_text
    assert len(ledger) == 1
    assert ledger[0].edge == "file_read_repeat"


def test_small_edit_becomes_a_diff(store, config) -> None:
    original = _numbered_lines(50)
    edited = original.replace("line_25", "line_25_renamed")
    messages = _read_messages(original, edited)
    result, ledger = compress_file_reads(messages, config=config, store=store, request_id="r1")

    second_read_text = result[3]["content"][0]["content"][0]["text"]
    assert "Diff since message" in second_read_text
    assert "line_25_renamed" in second_read_text
    assert len(ledger) == 1
    assert ledger[0].edge == "file_read_diff"


def test_large_rewrite_is_left_as_a_full_read(store, config) -> None:
    original = _numbered_lines(50)
    rewritten = "\n".join(f"   {i}\tCOMPLETELY_DIFFERENT_LINE_{i}_XYZXYZXYZ" for i in range(1, 51))
    messages = _read_messages(original, rewritten)
    result, ledger = compress_file_reads(messages, config=config, store=store, request_id="r1")

    second_read_text = result[3]["content"][0]["content"][0]["text"]
    assert second_read_text == rewritten  # too different to trust a diff
    assert ledger == []


def test_single_read_of_a_file_is_left_alone(store, config) -> None:
    content = _numbered_lines(50)
    messages = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "r1", "name": "Read", "input": {"file_path": "/a.py"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "r1", "content": [{"type": "text", "text": content}]}]},
    ]
    result, ledger = compress_file_reads(messages, config=config, store=store, request_id="r1")
    assert result == messages
    assert ledger == []


def test_never_touches_thinking_blocks(store, config) -> None:
    content = _numbered_lines(50)
    messages = _read_messages(content, content)
    # Sneak a thinking block into the assistant message right before the repeat read.
    messages[2]["content"].insert(0, {"type": "thinking", "thinking": "must never be read or written"})
    result, _ = compress_file_reads(messages, config=config, store=store, request_id="r1")
    thinking_block = result[2]["content"][0]
    assert thinking_block == {"type": "thinking", "thinking": "must never be read or written"}


def test_ttl_is_refreshed_on_every_replay(store, config) -> None:
    content = _numbered_lines(50)
    messages = _read_messages(content, content)
    _, ledger_1 = compress_file_reads(messages, config=config, store=store, request_id="r1")
    hash_1 = ledger_1[0].hash_key
    entry_before = store.retrieve(hash_1)
    assert entry_before is not None

    _, ledger_2 = compress_file_reads(messages, config=config, store=store, request_id="r2")
    hash_2 = ledger_2[0].hash_key
    assert hash_1 == hash_2  # same content -> same config-scoped hash
    assert store.retrieve(hash_2) is not None  # still resolvable after the "replay"
