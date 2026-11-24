"""Isolated tests for the agy cross-process savings inbox.

These exercise :mod:`headroom.proxy.agy_savings_inbox` in a temp HOME so no
shared state is touched. The proxy funnel is replaced by a fake object whose
async ``record_request`` just records the kwargs it was called with.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from headroom.proxy import agy_savings_inbox


class FakeMetrics:
    """Stand-in for PrometheusMetrics: async record_request captures kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def record_request(self, **kwargs) -> None:
        self.calls.append(kwargs)


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point the workspace (and thus the inbox) at a throwaway dir."""

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(home / ".headroom"))
    return home


_FUNNEL_KWARGS = {
    "provider": "anthropic",
    "model": "claude-sonnet",
    "input_tokens": 1200,
    "output_tokens": 340,
    "tokens_saved": 800,
    "latency_ms": 42.5,
    "cached": False,
    "overhead_ms": 3.0,
    "ttfb_ms": 10.0,
    "cache_read_tokens": 0,
    "cache_write_tokens": 0,
    "cache_write_5m_tokens": 0,
    "cache_write_1h_tokens": 0,
    "uncached_input_tokens": 1200,
    "attempted_input_tokens": 2000,
    "project": "myproj",
    "client": "agy",
}


def _evt_files() -> list[Path]:
    return sorted(agy_savings_inbox.inbox_dir().glob("evt-*.json"))


def test_emit_event_writes_one_roundtrippable_file(isolated_home):
    agy_savings_inbox.emit_event(**_FUNNEL_KWARGS)

    files = _evt_files()
    assert len(files) == 1

    envelope = json.loads(files[0].read_text())
    assert envelope["v"] == agy_savings_inbox.SCHEMA_VERSION
    assert isinstance(envelope["event_id"], str) and envelope["event_id"]

    kwargs = envelope["kwargs"]
    assert kwargs == _FUNNEL_KWARGS
    # Sanity: the funnel-only fields are present...
    assert kwargs["output_tokens"] == 340
    assert kwargs["latency_ms"] == 42.5
    # ...and SavingsTracker-only fields never leak in.
    assert "total_input_tokens" not in kwargs
    assert "total_input_cost_usd" not in kwargs
    assert "timestamp" not in kwargs


@pytest.mark.asyncio
async def test_drain_replays_each_event_once_then_deletes(isolated_home):
    agy_savings_inbox.emit_event(**_FUNNEL_KWARGS)
    fake = FakeMetrics()

    recorded = await agy_savings_inbox.drain_inbox(fake)

    assert recorded == 1
    assert fake.calls == [_FUNNEL_KWARGS]
    assert _evt_files() == []


@pytest.mark.asyncio
async def test_redrain_dedups_and_survives_crash_window(isolated_home):
    # Normal completed drain: event recorded, file gone, id in .processed.
    agy_savings_inbox.emit_event(**_FUNNEL_KWARGS)
    fake = FakeMetrics()
    assert await agy_savings_inbox.drain_inbox(fake) == 1
    assert len(fake.calls) == 1

    # Re-draining does NOT re-record the same event id.
    assert await agy_savings_inbox.drain_inbox(fake) == 0
    assert len(fake.calls) == 1

    # Crash window: an event whose id is already in .processed but whose evt
    # file still exists (recorded, crashed before unlink) must be unlinked
    # WITHOUT being recorded again.
    inbox = agy_savings_inbox.inbox_dir()
    seen_id = "crash-1"
    (inbox / ".processed").write_text(seen_id + "\n")
    (inbox / f"evt-{seen_id}.json").write_text(
        json.dumps({"v": 1, "event_id": seen_id, "kwargs": _FUNNEL_KWARGS})
    )

    assert await agy_savings_inbox.drain_inbox(fake) == 0
    assert len(fake.calls) == 1
    assert not (inbox / f"evt-{seen_id}.json").exists()


@pytest.mark.asyncio
async def test_two_events_from_two_pids_recorded_once_each(isolated_home):
    inbox = agy_savings_inbox.inbox_dir()
    for pid in (111, 222):
        eid = f"{pid}-0-abc"
        (inbox / f"evt-{eid}.json").write_text(
            json.dumps({"v": 1, "event_id": eid, "kwargs": _FUNNEL_KWARGS})
        )

    fake = FakeMetrics()
    recorded = await agy_savings_inbox.drain_inbox(fake)

    assert recorded == 2
    assert len(fake.calls) == 2
    assert _evt_files() == []


@pytest.mark.asyncio
async def test_malformed_event_skipped_not_fatal(isolated_home):
    inbox = agy_savings_inbox.inbox_dir()
    # Bad JSON file.
    bad = inbox / "evt-bad.json"
    bad.write_text("{ this is not json")
    # A good event alongside it.
    good_id = "999-0-def"
    (inbox / f"evt-{good_id}.json").write_text(
        json.dumps({"v": 1, "event_id": good_id, "kwargs": _FUNNEL_KWARGS})
    )

    fake = FakeMetrics()
    recorded = await agy_savings_inbox.drain_inbox(fake)

    # The malformed file is dropped; the good one still recorded.
    assert recorded == 1
    assert len(fake.calls) == 1
    assert not bad.exists()
    assert _evt_files() == []


@pytest.mark.asyncio
async def test_empty_inbox_returns_zero(isolated_home):
    fake = FakeMetrics()
    assert await agy_savings_inbox.drain_inbox(fake) == 0
    assert fake.calls == []
    # Non-AGY proxy must not create savings.d on disk when unused.
    assert not (isolated_home / ".headroom" / "savings.d").exists()


@pytest.mark.asyncio
async def test_stateless_mode_skips_emit_and_drain(isolated_home, monkeypatch):
    monkeypatch.setenv("HEADROOM_STATELESS", "1")
    agy_savings_inbox.emit_event(**_FUNNEL_KWARGS)
    inbox_path = isolated_home / ".headroom" / "savings.d"
    assert not inbox_path.exists()

    # Pre-existing event file must not be unlinked or written to .processed in stateless mode.
    inbox_path.mkdir(parents=True)
    evt_file = inbox_path / "evt-stateless-1.json"
    evt_file.write_text(
        json.dumps({"v": 1, "event_id": "stateless-1", "kwargs": _FUNNEL_KWARGS}),
        encoding="utf-8",
        newline="\n",
    )
    fake = FakeMetrics()
    assert await agy_savings_inbox.drain_inbox(fake) == 0
    assert fake.calls == []
    assert evt_file.exists()
    assert not (inbox_path / ".processed").exists()


@pytest.mark.windows_newline
@pytest.mark.asyncio
async def test_processed_file_uses_lf_newlines(isolated_home):
    agy_savings_inbox.emit_event(**_FUNNEL_KWARGS)
    fake = FakeMetrics()
    assert await agy_savings_inbox.drain_inbox(fake) == 1
    raw = (isolated_home / ".headroom" / "savings.d" / ".processed").read_bytes()
    assert b"\n" in raw
    assert b"\r\n" not in raw


def test_agy_emit_enabled_reflects_env(monkeypatch):
    monkeypatch.delenv(agy_savings_inbox.AGY_INBOX_EMIT_ENV, raising=False)
    assert agy_savings_inbox.agy_emit_enabled() is False

    monkeypatch.setenv(agy_savings_inbox.AGY_INBOX_EMIT_ENV, "1")
    assert agy_savings_inbox.agy_emit_enabled() is True

    monkeypatch.setenv(agy_savings_inbox.AGY_INBOX_EMIT_ENV, "0")
    assert agy_savings_inbox.agy_emit_enabled() is False


def test_two_processes_draining_same_inbox_record_event_once(isolated_home, tmp_path):
    """Two worker processes draining the same inbox must replay one event only once."""
    import os
    import subprocess
    import sys

    agy_savings_inbox.emit_event(**_FUNNEL_KWARGS)
    assert len(_evt_files()) == 1

    sync_dir = tmp_path / "sync"
    sync_dir.mkdir()

    worker_script = """
import asyncio
import sys
import time
from pathlib import Path
from headroom.proxy import agy_savings_inbox

worker_id = sys.argv[1]
other_id = sys.argv[2]
sync_dir = Path(sys.argv[3])

class SyncMetrics:
    async def record_request(self, **kwargs):
        (sync_dir / f"{worker_id}.entered").write_text("1", encoding="utf-8")
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            if (sync_dir / f"{other_id}.entered").exists():
                break
            await asyncio.sleep(0.01)

count = asyncio.run(agy_savings_inbox.drain_inbox(SyncMetrics()))
print(count)
"""
    env = {
        **os.environ,
        "HOME": str(isolated_home),
        "HEADROOM_WORKSPACE_DIR": str(isolated_home / ".headroom"),
        "HEADROOM_OFFLINE": "1",
    }

    p1 = subprocess.Popen(
        [sys.executable, "-c", worker_script, "w1", "w2", str(sync_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    p2 = subprocess.Popen(
        [sys.executable, "-c", worker_script, "w2", "w1", str(sync_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    out1, err1 = p1.communicate(timeout=15)
    out2, err2 = p2.communicate(timeout=15)

    assert p1.returncode == 0, err1
    assert p2.returncode == 0, err2
    c1 = int(out1.strip())
    c2 = int(out2.strip())
    assert c1 + c2 == 1, f"expected 1 combined record across two workers, got {c1} + {c2}"
    assert _evt_files() == []
