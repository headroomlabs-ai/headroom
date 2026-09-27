from __future__ import annotations

import pytest
from headroom.cache.backends.memory import InMemoryBackend
from headroom.cache.compression_store import CompressionStore

from headroom_edge_layer.config import EdgeLayerConfig
from headroom_edge_layer.grep_edge import compress_grep_result
from headroom_edge_layer.scoring import TokenOverlapScorer


@pytest.fixture
def store() -> CompressionStore:
    return CompressionStore(backend=InMemoryBackend())


@pytest.fixture
def config() -> EdgeLayerConfig:
    return EdgeLayerConfig(grep_per_file_cap=20)


def _grep_text(n: int, path: str = "foo.py") -> str:
    return "\n".join(f"{path}:{i}:def line_{i}(): pass  # matches bar" for i in range(1, n + 1))


def test_prunes_file_over_cap_and_keeps_all_file_paths(store, config) -> None:
    text = _grep_text(60)
    compressed, entry = compress_grep_result(
        text,
        tool_name="Grep",
        tool_call_id="t1",
        tool_input={"pattern": "bar"},
        intent_query="find bar usages",
        config=config,
        store=store,
        scorer=TokenOverlapScorer(),
        request_id="r1",
    )
    assert entry is not None
    assert "foo.py:" in compressed  # file path never dropped
    assert "Retrieve more: hash=" in compressed
    assert "60" in compressed  # true match count named in the marker


def test_leaves_result_under_cap_unchanged(store, config) -> None:
    text = _grep_text(5)
    compressed, entry = compress_grep_result(
        text,
        tool_name="Grep",
        tool_call_id="t1",
        tool_input={"pattern": "bar"},
        intent_query="find bar",
        config=config,
        store=store,
        scorer=TokenOverlapScorer(),
        request_id="r1",
    )
    assert entry is None
    assert compressed == text


@pytest.mark.parametrize(
    "tool_input,intent_query",
    [
        ({"pattern": "bar", "flags": ["-l"]}, "find bar"),
        ({"pattern": "bar", "flags": ["--files-with-matches"]}, "find bar"),
        ({"pattern": "bar"}, "rename every usage of bar"),
        ({"pattern": "bar"}, "replace all references to bar"),
    ],
)
def test_enumeration_guard_skips_pruning(store, config, tool_input, intent_query) -> None:
    text = _grep_text(60)
    compressed, entry = compress_grep_result(
        text,
        tool_name="Grep",
        tool_call_id="t1",
        tool_input=tool_input,
        intent_query=intent_query,
        config=config,
        store=store,
        scorer=TokenOverlapScorer(),
        request_id="r1",
    )
    assert entry is None
    assert compressed == text


def test_already_compressed_text_is_left_alone(store, config) -> None:
    text = "[60 lines compressed to 20. Retrieve more: hash=abcdef0123456789abcdef01]"
    compressed, entry = compress_grep_result(
        text,
        tool_name="Grep",
        tool_call_id="t1",
        tool_input={"pattern": "bar"},
        intent_query="find bar",
        config=config,
        store=store,
        scorer=TokenOverlapScorer(),
        request_id="r1",
    )
    assert entry is None
    assert compressed == text


def test_deterministic_across_repeated_calls(store, config) -> None:
    text = _grep_text(60)
    out1, _ = compress_grep_result(
        text, tool_name="Grep", tool_call_id="t1", tool_input={"pattern": "bar"},
        intent_query="find bar", config=config, store=store, scorer=TokenOverlapScorer(), request_id="r1",
    )
    out2, _ = compress_grep_result(
        text, tool_name="Grep", tool_call_id="t1", tool_input={"pattern": "bar"},
        intent_query="find bar", config=config, store=store, scorer=TokenOverlapScorer(), request_id="r2",
    )
    assert out1 == out2
