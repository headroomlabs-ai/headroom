from __future__ import annotations

import pytest
from headroom.cache.backends.memory import InMemoryBackend
from headroom.cache.compression_store import CompressionStore

from headroom_edge_layer.config import EdgeLayerConfig
from headroom_edge_layer.rerank_edge import compress_log_or_web
from headroom_edge_layer.scoring import TokenOverlapScorer


@pytest.fixture
def store() -> CompressionStore:
    return CompressionStore(backend=InMemoryBackend())


@pytest.fixture
def config() -> EdgeLayerConfig:
    return EdgeLayerConfig(rerank_trigger_tokens=100, rerank_min_tokens_kept=50, rerank_target_ratio=0.2)


def _log_with_error(n_lines: int, error_at: int) -> str:
    lines = [f"line {i}: routine build output, nothing interesting here at all" for i in range(n_lines)]
    lines[error_at] = "ERROR: build failed with exit code 1 in module widget"
    return "\n".join(lines)


def test_below_trigger_is_left_unchanged(store, config) -> None:
    text = "short log\nwith just a few lines\nnothing big"
    compressed, entry = compress_log_or_web(
        text, tool_name="Bash", tool_call_id="t1", intent_query="check the build",
        config=config, store=store, scorer=TokenOverlapScorer(), request_id="r1",
    )
    assert entry is None
    assert compressed == text


def test_error_line_survives_elision(store, config) -> None:
    text = _log_with_error(300, error_at=150)
    compressed, entry = compress_log_or_web(
        text, tool_name="Bash", tool_call_id="t1", intent_query="why did the build fail",
        config=config, store=store, scorer=TokenOverlapScorer(), request_id="r1",
    )
    assert entry is not None
    assert "ERROR: build failed" in compressed
    assert len(compressed) < len(text)


def test_head_and_tail_lines_survive_elision(store, config) -> None:
    lines = [f"line {i}: filler filler filler filler filler content" for i in range(500)]
    lines[0] = "FIRST LINE MARKER"
    lines[-1] = "LAST LINE MARKER"
    text = "\n".join(lines)
    compressed, entry = compress_log_or_web(
        text, tool_name="WebFetch", tool_call_id="t1", intent_query="irrelevant query xyz",
        config=config, store=store, scorer=TokenOverlapScorer(), request_id="r1",
    )
    assert entry is not None
    assert "FIRST LINE MARKER" in compressed
    assert "LAST LINE MARKER" in compressed


def test_elided_chunks_are_retrievable_via_marker(store, config) -> None:
    text = _log_with_error(300, error_at=150)
    compressed, entry = compress_log_or_web(
        text, tool_name="Bash", tool_call_id="t1", intent_query="why did the build fail",
        config=config, store=store, scorer=TokenOverlapScorer(), request_id="r1",
    )
    assert entry is not None
    assert "Retrieve more: hash=" in compressed
    import re

    hash_key = re.search(r"hash=([0-9a-f]{12,24})", compressed).group(1)
    stored = store.retrieve(hash_key)
    assert stored is not None


def test_already_compressed_text_is_left_alone(store, config) -> None:
    text = "[300 lines compressed to 100. Retrieve more: hash=abcdef0123456789abcdef01]" * 30
    compressed, entry = compress_log_or_web(
        text, tool_name="Bash", tool_call_id="t1", intent_query="anything",
        config=config, store=store, scorer=TokenOverlapScorer(), request_id="r1",
    )
    assert entry is None
    assert compressed == text


def test_deterministic_across_repeated_calls(store, config) -> None:
    text = _log_with_error(300, error_at=150)
    out1, _ = compress_log_or_web(
        text, tool_name="Bash", tool_call_id="t1", intent_query="why did the build fail",
        config=config, store=store, scorer=TokenOverlapScorer(), request_id="r1",
    )
    out2, _ = compress_log_or_web(
        text, tool_name="Bash", tool_call_id="t1", intent_query="why did the build fail",
        config=config, store=store, scorer=TokenOverlapScorer(), request_id="r2",
    )
    assert out1 == out2
