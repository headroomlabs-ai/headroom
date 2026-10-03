"""Stateless mode must not write to the workspace — end to end and per writer.

``--stateless`` is the flag regulated and ephemeral deployments pick so that no
request content lands on disk. Before this fix the CCR retrieval store (whose
default backend became SQLite) wrote verbatim tool output to ``ccr_store.db``
under a stateless proxy, and several smaller persisters (licence cache, MCP
session stats, savings ledger, subscription state, memory sync state, the
update-check cache) had no stateless check at all.

The end-to-end test snapshots a private workspace, drives a tool-heavy turn
through ``/v1/compress`` (which stores CCR originals exactly as the data plane
does) and asserts nothing appeared. The control test runs the same turn without
``--stateless`` and asserts the store file *does* appear, so the assertion is
known to be reachable. The remaining tests pin each writer to the shared
``paths.persistence_allowed()`` predicate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom import paths
from headroom.cache import compression_store as cs
from headroom.cache.backends import InMemoryBackend

# The runtime log is the one file stateless mode is documented to keep writing
# (it is created owner-only and carries no request content by default).
_ALLOWED_TOP_LEVEL = {"logs"}


@pytest.fixture(autouse=True)
def _isolate_process_state(monkeypatch):
    """Process-global flags and singletons must never leak between tests."""
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)
    monkeypatch.delenv("HEADROOM_CCR_SQLITE_PATH", raising=False)
    paths.set_process_stateless(False)
    paths._reset_persistence_notices()
    cs.reset_compression_store()
    yield
    paths.set_process_stateless(False)
    paths._reset_persistence_notices()
    cs.reset_compression_store()
    try:
        from headroom.telemetry.toin import reset_toin

        reset_toin()
    except Exception:
        pass


@pytest.fixture
def workspace(tmp_path, monkeypatch) -> Path:
    """A private HOME and workspace so the test can assert on the whole tree."""
    home = tmp_path / "home"
    home.mkdir()
    ws = home / ".headroom"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(ws))
    monkeypatch.chdir(tmp_path)
    return ws


def _snapshot(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.relative_to(root).parts[0] not in _ALLOWED_TOP_LEVEL
    }


def _tool_heavy_messages() -> list[dict]:
    large = json.dumps(
        [
            {
                "id": i,
                "name": f"Item {i}",
                "description": (
                    f"Detailed description for item {i}: status active, created on "
                    f"2024-01-{(i % 28) + 1:02d}, category=electronics, price={i * 10.99:.2f}."
                ),
                "tags": ["electronics", "sale", "featured"],
            }
            for i in range(200)
        ]
    )
    return [
        {"role": "user", "content": "What items are available?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "list", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": large},
        {"role": "user", "content": "Summarize the first 5 items."},
    ]


def _run_turn(stateless: bool) -> None:
    from headroom.proxy.server import ProxyConfig, create_app

    config = ProxyConfig(
        stateless=stateless,
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        resp = client.post(
            "/v1/compress", json={"messages": _tool_heavy_messages(), "model": "gpt-4"}
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["tokens_saved"] > 0, "turn must actually exercise the CCR store"


# ---- end to end -----------------------------------------------------------


def test_stateless_turn_writes_nothing_to_the_workspace(workspace):
    before = _snapshot(workspace)
    _run_turn(stateless=True)
    after = _snapshot(workspace)
    assert after - before == set(), f"stateless proxy wrote: {sorted(after - before)}"
    assert not (workspace / "ccr_store.db").exists()
    assert isinstance(cs.get_compression_store()._backend, InMemoryBackend)


def test_stateful_turn_does_write_the_ccr_store(workspace):
    """Control: the same turn without --stateless persists, so the assertion above is live."""
    _run_turn(stateless=False)
    assert (workspace / "ccr_store.db").exists()


def _ccr_db_state(workspace: Path) -> tuple[int, dict[str, bytes]]:
    """Row count plus the exact bytes of ccr_store.db and its WAL/SHM files."""
    import sqlite3

    db = workspace / "ccr_store.db"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT COUNT(*) FROM ccr_entries").fetchone()[0]
    finally:
        conn.close()
    files = {p.name: p.read_bytes() for p in sorted(workspace.glob("ccr_store.db*"))}
    return rows, files


def test_stateless_switch_leaves_an_existing_sqlite_store_untouched(workspace):
    """A SQLite singleton built before the flag is swapped out, not cleared.

    Entering stateless mode must not write to (or empty) a ccr_store.db that
    already holds entries; it only stops using it.
    """
    from headroom.proxy.server import ProxyConfig, _apply_stateless_persistence

    stateful = cs.get_compression_store()  # default backend: SQLite
    assert not isinstance(stateful._backend, InMemoryBackend)
    kept = stateful.store('{"rows": [1, 2, 3]}', '{"rows": [1]}')
    rows_before, bytes_before = _ccr_db_state(workspace)
    assert rows_before == 1

    paths.set_process_stateless(True)
    _apply_stateless_persistence(ProxyConfig(stateless=True))
    stateless = cs.get_compression_store()
    assert isinstance(stateless._backend, InMemoryBackend)
    fresh = stateless.store('{"rows": [4, 5, 6]}', '{"rows": [4]}')
    assert stateless.retrieve(fresh) is not None

    rows_after, bytes_after = _ccr_db_state(workspace)
    assert rows_after == rows_before, "stateless switch deleted persisted CCR entries"
    assert bytes_after == bytes_before, "stateless switch wrote to ccr_store.db"
    assert stateful.retrieve(kept) is not None


# ---- the shared predicate ---------------------------------------------------


def test_persistence_allowed_follows_the_stateless_flag(caplog):
    assert paths.persistence_allowed("x") is True
    paths.set_process_stateless(True)
    import logging

    with caplog.at_level(logging.INFO, logger="headroom.paths"):
        assert paths.persistence_allowed("x") is False
        assert paths.persistence_allowed("x") is False
        assert paths.persistence_allowed("y") is False
    notices = [r for r in caplog.records if "not persisting" in r.getMessage()]
    assert [r.getMessage().split("persisting ")[1].split(" to")[0] for r in notices] == ["x", "y"]


def test_persistence_allowed_honours_env(monkeypatch):
    monkeypatch.setenv("HEADROOM_STATELESS", "true")
    assert paths.persistence_allowed("x") is False


# ---- CCR backend selection --------------------------------------------------


@pytest.mark.parametrize("backend_env", [None, "sqlite"])
def test_default_ccr_backend_is_memory_under_stateless(workspace, monkeypatch, backend_env):
    if backend_env is not None:
        monkeypatch.setenv("HEADROOM_CCR_BACKEND", backend_env)
    paths.set_process_stateless(True)
    assert cs._create_default_ccr_backend() is None
    assert not (workspace / "ccr_store.db").exists()


def test_default_ccr_backend_is_sqlite_when_stateful(workspace):
    backend = cs._create_default_ccr_backend()
    assert backend is not None and not isinstance(backend, InMemoryBackend)
    assert (workspace / "ccr_store.db").exists()


# ---- individual writers -----------------------------------------------------


def test_license_cache_not_written_under_stateless(workspace, tmp_path):
    from headroom.telemetry.reporter import LicenseInfo, UsageReporter

    cache = tmp_path / "license.json"
    reporter = UsageReporter("hlk_test", cache_path=cache)
    reporter._license_info = LicenseInfo(status="active")
    paths.set_process_stateless(True)
    reporter._save_cache()
    assert not cache.exists()
    paths.set_process_stateless(False)
    reporter._save_cache()
    assert cache.exists()


def test_mcp_session_stats_not_written_under_stateless(workspace, monkeypatch):
    from headroom.ccr import mcp_server

    stats_file = workspace / "session_stats.jsonl"
    monkeypatch.setattr(mcp_server, "SHARED_STATS_DIR", workspace)
    monkeypatch.setattr(mcp_server, "SHARED_STATS_FILE", stats_file)
    paths.set_process_stateless(True)
    mcp_server._append_shared_event({"timestamp": 1.0, "type": "compression"})
    assert not stats_file.exists()
    paths.set_process_stateless(False)
    mcp_server._append_shared_event({"timestamp": 1.0, "type": "compression"})
    assert stats_file.exists()


def test_savings_ledger_not_written_under_stateless(workspace, tmp_path):
    from headroom import savings_ledger

    target = tmp_path / "events.jsonl"
    paths.set_process_stateless(True)
    assert (
        savings_ledger.record_savings_event(tokens_before=100, tokens_after=10, path=target)
        is False
    )
    assert not target.exists()
    paths.set_process_stateless(False)
    assert (
        savings_ledger.record_savings_event(tokens_before=100, tokens_after=10, path=target) is True
    )
    assert target.exists()


def test_subscription_state_not_persisted_under_stateless(workspace, tmp_path):
    from headroom.subscription.tracker import SubscriptionTracker

    persist = tmp_path / "sub.json"
    tracker = SubscriptionTracker(persist_path=persist)
    paths.set_process_stateless(True)
    tracker._persist_state()
    assert not persist.exists()
    paths.set_process_stateless(False)
    tracker._persist_state()
    assert persist.exists()


def test_memory_sync_state_not_written_under_stateless(tmp_path):
    from headroom.memory.sync import _save_sync_state

    state = tmp_path / "sync.json"
    paths.set_process_stateless(True)
    _save_sync_state(state, {"a": 1})
    assert not state.exists()
    paths.set_process_stateless(False)
    _save_sync_state(state, {"a": 1})
    assert state.exists()


def test_update_check_cache_not_written_under_stateless(workspace):
    from headroom import update_check

    paths.set_process_stateless(True)
    assert update_check.is_update_check_enabled() is False
    update_check.write_cache("9.9.9", now=1.0)
    assert not update_check._cache_path().exists()


def test_cli_stateless_flag_exports_env_for_children(monkeypatch):
    """`headroom proxy --stateless` must make HEADROOM_STATELESS visible to child processes."""
    from click.testing import CliRunner

    from headroom.cli import main

    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    captured: dict[str, str] = {}

    def fake_run_server(*_args, **_kwargs):
        captured["HEADROOM_STATELESS"] = os.environ.get("HEADROOM_STATELESS", "")

    # The CLI imports run_server lazily from headroom.proxy.server, so patch it there.
    import headroom.proxy.server as proxy_server

    monkeypatch.setattr(proxy_server, "run_server", fake_run_server)
    result = CliRunner().invoke(main, ["proxy", "--stateless", "--port", "18999"])
    assert result.exit_code == 0, result.output
    assert captured.get("HEADROOM_STATELESS") == "1"
