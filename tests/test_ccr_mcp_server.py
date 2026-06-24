from __future__ import annotations

import asyncio
import json

import pytest

from headroom.cache.compression_store import (
    get_compression_store,
    reset_compression_store,
)
from tests._mcp_stub import import_module_with_mcp_stub

mcp_server = import_module_with_mcp_stub("headroom.ccr.mcp_server")


def test_shared_stats_work_without_fcntl(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(mcp_server, "_HAS_FCNTL", False)
    monkeypatch.setattr(mcp_server, "fcntl", None)
    monkeypatch.setattr(mcp_server, "SHARED_STATS_DIR", tmp_path)
    monkeypatch.setattr(mcp_server, "SHARED_STATS_FILE", tmp_path / "session_stats.jsonl")
    monkeypatch.setattr(mcp_server.os, "getpid", lambda: 4242)
    monkeypatch.setattr(mcp_server.time, "time", lambda: 1001.0)

    event = {"type": "compress", "timestamp": 1000.0}
    mcp_server._append_shared_event(event)

    raw_lines = mcp_server.SHARED_STATS_FILE.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 1
    assert json.loads(raw_lines[0]) == {"type": "compress", "timestamp": 1000.0, "pid": 4242}

    events = mcp_server._read_shared_events(window_seconds=60)
    assert events == [{"type": "compress", "timestamp": 1000.0, "pid": 4242}]


# --- Shared compression store wiring ---------------------------------------
# MCP's _get_local_store() must return the get_compression_store() singleton —
# the same instance the proxy and response_handler use — so content compressed
# on either side is retrievable in-process. These pin that wiring so a private
# store can't creep back.


@pytest.fixture
def fresh_store():
    reset_compression_store()
    yield
    reset_compression_store()


def test_mcp_uses_shared_singleton_store(fresh_store) -> None:
    """MCP's store is the global singleton, not a private instance."""
    server = mcp_server.HeadroomMCPServer(check_proxy=False)
    assert server._get_local_store() is get_compression_store()


def test_mcp_retrieves_proxy_stored_content(fresh_store) -> None:
    """Content stored via the singleton (as the proxy does) is retrievable
    through MCP's local-store path. The HTTP fallback is disabled so this
    passes only via the shared store."""
    original = '{"some": "original proxy-compressed content"}'
    hash_key = get_compression_store().store(original, '{"compressed": true}')

    server = mcp_server.HeadroomMCPServer(check_proxy=False)
    result = asyncio.run(server._retrieve_content(hash_key, query=None))

    assert result.get("source") == "local"
    assert result["original_content"] == original


def test_compress_savings_percent_tracks_token_counts(fresh_store) -> None:
    """``savings_percent`` must be the *removed* percentage derived from the
    token counts — never the retained percentage. Regression for the inversion
    where ``(1 - compression_ratio)`` reported a no-op (0% saved) as 100%."""
    pytest.importorskip("mcp", reason="MCP SDK required")
    server = mcp_server.HeadroomMCPServer(check_proxy=False)

    # Repetitive JSON array — the shape the engine actually compresses.
    content = json.dumps([{"id": i, "status": "ok", "kind": "run"} for i in range(40)])
    result = server._compress_content(content)

    orig = result["original_tokens"]
    comp = result["compressed_tokens"]
    expected = round((1 - comp / orig) * 100, 1) if orig > 0 else 0

    # Reported savings agrees with the token fields (and with tokens_saved).
    assert result["savings_percent"] == expected
    assert 0.0 <= result["savings_percent"] <= 100.0
    if result["tokens_saved"] == 0:
        assert result["savings_percent"] == 0.0  # not inverted to 100
    else:
        assert result["savings_percent"] > 0.0


def test_mcp_retrieve_with_nonmatching_query_returns_full_content(fresh_store) -> None:
    """A query that matches no item above the relevance floor must still return
    the stored entry (it exists and is unexpired) rather than the "Content not
    found" error, which is reserved for genuine misses."""
    original = "the the the the the the the the the the\n" * 5
    hash_key = get_compression_store().store(original, "<<small>>")
    # Precondition: the query genuinely matches nothing above the BM25 floor.
    assert get_compression_store().search(hash_key, "zzqx_nonmatching_token") == []

    server = mcp_server.HeadroomMCPServer(check_proxy=False)
    result = asyncio.run(server._retrieve_content(hash_key, query="zzqx_nonmatching_token"))

    assert "error" not in result
    assert result.get("source") == "local"
    assert result["original_content"] == original
    assert result["count"] == 0


def test_mcp_retrieve_missing_hash_still_errors(fresh_store) -> None:
    """A genuinely missing hash must still report "Content not found"."""
    server = mcp_server.HeadroomMCPServer(check_proxy=False)
    result = asyncio.run(server._retrieve_content("nonexistent_hash", query="anything"))
    assert "Content not found" in result.get("error", "")
