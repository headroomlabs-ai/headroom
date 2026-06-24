from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from headroom.graph import installer, watcher


def _build_archive(member_name: str = installer.CBM_BIN_NAME) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as tar:
        data = b"#!/bin/sh\necho version\n"
        info = tarfile.TarInfo(name=member_name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return payload.getvalue()


class FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self._data


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("Darwin", "arm64", "darwin-arm64"),
        ("Darwin", "x86_64", "darwin-amd64"),
        ("Linux", "aarch64", "linux-arm64"),
        ("Linux", "arm64", "linux-arm64"),
        ("Linux", "x86_64", "linux-amd64"),
        ("Windows", "AMD64", "windows-amd64"),
    ],
)
def test_detect_platform_variants(monkeypatch, system: str, machine: str, expected: str) -> None:
    monkeypatch.setattr(installer.platform, "system", lambda: system)
    monkeypatch.setattr(installer.platform, "machine", lambda: machine)
    assert installer._detect_platform() == expected


def test_detect_platform_rejects_unknown_system(monkeypatch) -> None:
    monkeypatch.setattr(installer.platform, "system", lambda: "Solaris")
    monkeypatch.setattr(installer.platform, "machine", lambda: "sparc")
    with pytest.raises(RuntimeError, match="Unsupported platform"):
        installer._detect_platform()


def test_get_cbm_path_prefers_path_then_install_dir(monkeypatch, tmp_path: Path) -> None:
    on_path = tmp_path / "on-path"
    installed = tmp_path / installer.CBM_BIN_NAME
    installed.write_text("bin")
    monkeypatch.setattr(installer, "CBM_BIN_DIR", tmp_path)
    monkeypatch.setattr(installer.shutil, "which", lambda name: str(on_path))
    assert installer.get_cbm_path() == on_path

    monkeypatch.setattr(installer.shutil, "which", lambda name: None)
    assert installer.get_cbm_path() == installed

    installed.unlink()
    assert installer.get_cbm_path() is None


def test_download_cbm_success_and_verification_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(installer, "CBM_BIN_DIR", tmp_path)
    monkeypatch.setattr(installer, "_detect_platform", lambda: "linux-amd64")
    monkeypatch.setattr(
        installer, "urlopen", lambda url, timeout=60: FakeResponse(_build_archive())
    )

    run_calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        run_calls.append(command)
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr("subprocess.run", fake_run)
    path = installer.download_cbm(version="v1.2.3")
    assert path == tmp_path / installer.CBM_BIN_NAME
    assert path.exists()
    assert run_calls == [[str(path), "--version"]]

    monkeypatch.setattr(
        "subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert installer.download_cbm(version="v1.2.3") == path

    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="cbm v1.2.3\n"),
    )
    assert installer.download_cbm(version="v1.2.3") == path


def test_download_cbm_invalid_url_download_failure_and_extract_errors(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(installer, "CBM_BIN_DIR", tmp_path)
    monkeypatch.setattr(installer, "_detect_platform", lambda: "linux-amd64")

    monkeypatch.setattr(installer, "GITHUB_RELEASE_URL", "ftp://example.test/releases")
    with pytest.raises(RuntimeError, match="Invalid URL"):
        installer.download_cbm()

    monkeypatch.setattr(installer, "GITHUB_RELEASE_URL", "https://example.test/releases")
    monkeypatch.setattr(
        installer,
        "urlopen",
        lambda url, timeout=60: (_ for _ in ()).throw(OSError("network down")),
    )
    with pytest.raises(RuntimeError, match="Failed to download codebase-memory-mcp"):
        installer.download_cbm()

    monkeypatch.setattr(
        installer,
        "urlopen",
        lambda url, timeout=60: FakeResponse(_build_archive("some/other-binary")),
    )
    with pytest.raises(RuntimeError, match="binary not found in archive"):
        installer.download_cbm()

    monkeypatch.setattr(installer, "urlopen", lambda url, timeout=60: FakeResponse(b"not a tar"))
    with pytest.raises(RuntimeError, match="Failed to extract archive"):
        installer.download_cbm()


def test_ensure_cbm_uses_existing_or_returns_none_on_failure(monkeypatch, tmp_path: Path) -> None:
    existing = tmp_path / installer.CBM_BIN_NAME
    monkeypatch.setattr(installer, "get_cbm_path", lambda: existing)
    assert installer.ensure_cbm() == existing

    monkeypatch.setattr(installer, "get_cbm_path", lambda: None)
    monkeypatch.setattr(
        installer, "download_cbm", lambda: (_ for _ in ()).throw(RuntimeError("nope"))
    )
    assert installer.ensure_cbm() is None


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
