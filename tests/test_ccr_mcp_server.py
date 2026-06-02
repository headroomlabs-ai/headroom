from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

from headroom.ccr import mcp_server


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


def test_token_savings_percent_uses_saved_tokens() -> None:
    assert mcp_server._token_savings_percent(100, 25) == 75.0
    assert mcp_server._token_savings_percent(67, 67) == 0.0
    assert mcp_server._token_savings_percent(10, 12) == 0.0
    assert mcp_server._token_savings_percent(0, 0) == 0.0


def test_mcp_compress_reports_savings_from_token_counts(monkeypatch) -> None:
    fake_compress_module = ModuleType("headroom.compress")

    def fake_compress(messages, model):
        assert messages == [{"role": "tool", "content": "original"}]
        assert model == "claude-sonnet-4-5-20250929"
        return SimpleNamespace(
            messages=[{"role": "tool", "content": "compressed"}],
            tokens_before=67,
            tokens_after=67,
            transforms_applied=["fake"],
            compression_ratio=0.0,
        )

    fake_compress_module.compress = fake_compress
    monkeypatch.setitem(sys.modules, "headroom.compress", fake_compress_module)

    class DummyStore:
        def store(self, **kwargs):
            self.kwargs = kwargs
            return "dummyhash"

    class DummyStats:
        def record_compression(self, *args):
            self.args = args

    server = mcp_server.HeadroomMCPServer.__new__(mcp_server.HeadroomMCPServer)
    server._local_store = DummyStore()
    server._stats = DummyStats()

    result = mcp_server.HeadroomMCPServer._compress_content(server, "original")

    assert result["tokens_saved"] == 0
    assert result["savings_percent"] == 0.0
