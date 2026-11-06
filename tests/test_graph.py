from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace

from headroom.graph import installer, watcher


def test_get_cbm_path_prefers_path_then_install_dir(monkeypatch, tmp_path: Path) -> None:
    on_path = tmp_path / "on-path"
    installed = tmp_path / installer.CBM_BIN_NAME
    installed.write_text("bin")
    monkeypatch.setattr(installer, "CBM_BIN_DIR", tmp_path)
    monkeypatch.setattr(installer.platform, "system", lambda: "Linux")
    monkeypatch.setattr(installer.shutil, "which", lambda name: str(on_path))
    assert installer.get_cbm_path() == on_path

    monkeypatch.setattr(installer.shutil, "which", lambda name: None)
    assert installer.get_cbm_path() == installed

    installed.unlink()
    assert installer.get_cbm_path() is None


def test_code_graph_watcher_init_start_stop_and_event_filtering(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("headroom.graph.installer.get_cbm_path", lambda: tmp_path / "cbm")
    graph_watcher = watcher.CodeGraphWatcher(tmp_path)
    assert graph_watcher.cbm_binary == str(tmp_path / "cbm")

    explicit = watcher.CodeGraphWatcher(tmp_path, cbm_binary="explicit-cbm")
    assert explicit.cbm_binary == "explicit-cbm"

    missing = watcher.CodeGraphWatcher(tmp_path, cbm_binary=None)
    missing.cbm_binary = None
    assert missing.start() is False

    watchdog_mod = ModuleType("watchdog")
    events_mod = ModuleType("watchdog.events")
    observers_mod = ModuleType("watchdog.observers")

    class FileSystemEventHandler:
        pass

    class FakeObserver:
        def __init__(self) -> None:
            self.scheduled = None
            self.daemon = False
            self.started = False
            self.stopped = False
            self.join_timeout = None

        def schedule(self, handler, project_dir, recursive=True) -> None:
            self.scheduled = (handler, project_dir, recursive)

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

        def join(self, timeout=None) -> None:
            self.join_timeout = timeout

    events_mod.FileSystemEventHandler = FileSystemEventHandler
    observers_mod.Observer = FakeObserver
    monkeypatch.setitem(__import__("sys").modules, "watchdog", watchdog_mod)
    monkeypatch.setitem(__import__("sys").modules, "watchdog.events", events_mod)
    monkeypatch.setitem(__import__("sys").modules, "watchdog.observers", observers_mod)

    scheduled: list[str] = []
    monkeypatch.setattr(graph_watcher, "_schedule_reindex", lambda: scheduled.append("reindex"))

    assert graph_watcher.start() is True
    handler, project_dir, recursive = graph_watcher._observer.scheduled
    assert project_dir == str(tmp_path)
    assert recursive is True

    handler.on_any_event(SimpleNamespace(src_path=""))
    handler.on_any_event(SimpleNamespace(src_path=str(tmp_path / ".git" / "config")))
    handler.on_any_event(SimpleNamespace(src_path=str(tmp_path / "notes.txt")))
    handler.on_any_event(SimpleNamespace(src_path=str(tmp_path / ".temp.py")))
    handler.on_any_event(SimpleNamespace(src_path=str(tmp_path / "main.py~")))
    handler.on_any_event(SimpleNamespace(src_path=str(tmp_path / "main.py")))
    assert scheduled == ["reindex"]

    class FakeTimer:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    timer = FakeTimer()
    graph_watcher._debounce_timer = timer
    graph_watcher._reindex_count = 1
    graph_watcher.stop()
    assert timer.cancelled is True
    assert graph_watcher._observer is None


def test_code_graph_watcher_start_returns_false_without_watchdog(
    monkeypatch, tmp_path: Path
) -> None:
    graph_watcher = watcher.CodeGraphWatcher(tmp_path, cbm_binary="cbm")

    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("watchdog"):
            raise ImportError("missing watchdog")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert graph_watcher.start() is False


def test_code_graph_watcher_stop_handles_missing_timer_and_observer_methods(tmp_path: Path) -> None:
    graph_watcher = watcher.CodeGraphWatcher(tmp_path, cbm_binary="cbm")
    graph_watcher._observer = object()
    graph_watcher.stop()
    assert graph_watcher._observer is None

    graph_watcher.stop()


def test_schedule_reindex_replaces_existing_timer(monkeypatch, tmp_path: Path) -> None:
    graph_watcher = watcher.CodeGraphWatcher(tmp_path, debounce_seconds=3.5, cbm_binary="cbm")
    timers: list[FakeTimer] = []

    class FakeTimer:
        def __init__(self, interval, callback) -> None:
            self.interval = interval
            self.callback = callback
            self.daemon = False
            self.started = False
            self.cancelled = False
            timers.append(self)

        def start(self) -> None:
            self.started = True

        def cancel(self) -> None:
            self.cancelled = True

    monkeypatch.setattr(watcher.threading, "Timer", FakeTimer)
    graph_watcher._schedule_reindex()
    graph_watcher._schedule_reindex()

    assert len(timers) == 2
    assert timers[0].cancelled is True
    assert timers[1].started is True
    assert timers[1].daemon is True
    assert timers[1].interval == 3.5


def test_do_reindex_success_failure_timeout_and_stats(monkeypatch, tmp_path: Path) -> None:
    graph_watcher = watcher.CodeGraphWatcher(tmp_path, cbm_binary="cbm")
    graph_watcher._running = True

    monotonic_values = iter([10.0, 10.4, 20.0, 20.5, 30.0, 30.5, 40.0, 40.5])
    monkeypatch.setattr(watcher.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(watcher.time, "time", lambda: 1234.0)

    run_calls: list[list[str]] = []

    def success_run(command, **kwargs):
        run_calls.append(command)
        return SimpleNamespace(returncode=0, stderr="indexed\nchanged=7 files\n")

    monkeypatch.setattr(watcher.subprocess, "run", success_run)
    graph_watcher._do_reindex()
    assert graph_watcher.stats == {
        "running": True,
        "project_dir": str(tmp_path),
        "reindex_count": 1,
        "last_reindex": 1234.0,
        "debounce_seconds": 2.0,
    }
    assert run_calls == [
        ["cbm", "cli", "index_repository", json.dumps({"repo_path": str(tmp_path), "mode": "fast"})]
    ]

    monkeypatch.setattr(
        watcher.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(returncode=1, stderr="failed"),
    )
    graph_watcher._do_reindex()
    assert graph_watcher._reindex_count == 2

    monkeypatch.setattr(
        watcher.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            returncode=0, stderr="indexed\nchanged=oops\nstill running\n"
        ),
    )
    graph_watcher._do_reindex()
    assert graph_watcher._reindex_count == 3

    monkeypatch.setattr(
        watcher.subprocess,
        "run",
        lambda command, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired(command, 30)),
    )
    graph_watcher._do_reindex()

    monkeypatch.setattr(
        watcher.subprocess,
        "run",
        lambda command, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    graph_watcher._do_reindex()

    graph_watcher._running = False
    graph_watcher._do_reindex()

    graph_watcher._running = True
    graph_watcher.cbm_binary = None
    graph_watcher._do_reindex()
