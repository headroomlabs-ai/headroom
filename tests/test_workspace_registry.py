"""Tests for headroom.proxy.workspace_registry.resolve_registered_cwd."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from headroom import paths
from headroom.proxy.workspace_registry import resolve_registered_cwd


def _write_marker(port: int, pid: int, payload: dict) -> None:
    d = paths.proxy_clients_dir(port)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{pid}.json").write_text(json.dumps(payload), encoding="utf-8")


def _spawn_and_reap() -> int:
    """Return a PID that is guaranteed dead (spawned, then waited on)."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_no_token_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    assert resolve_registered_cwd(8787, None) is None
    assert resolve_registered_cwd(8787, "") is None


def test_no_marker_directory_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    assert resolve_registered_cwd(8787, "some-token") is None


def test_matching_token_live_pid_returns_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_marker(
        8787,
        os.getpid(),  # this test process -- always live
        {"pid": os.getpid(), "session_token": "tok-A", "cwd": str(project_dir)},
    )
    assert resolve_registered_cwd(8787, "tok-A") == str(project_dir)


def test_wrong_token_is_none_even_with_a_live_marker_present(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_marker(
        8787,
        os.getpid(),
        {"pid": os.getpid(), "session_token": "tok-A", "cwd": str(project_dir)},
    )
    assert resolve_registered_cwd(8787, "tok-B-guessed") is None


def test_stale_marker_dead_pid_is_none_not_the_cwd(tmp_path, monkeypatch):
    """A crashed/killed session's marker must not be trusted."""
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    dead_pid = _spawn_and_reap()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_marker(
        8787,
        dead_pid,
        {"pid": dead_pid, "session_token": "tok-dead", "cwd": str(project_dir)},
    )
    assert resolve_registered_cwd(8787, "tok-dead") is None


def test_multi_session_isolation_token_resolves_only_its_own_cwd(tmp_path, monkeypatch):
    """One proxy can serve N concurrent sessions -- token A must never
    resolve project B's cwd."""
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_a.mkdir()
    project_b.mkdir()
    # Two distinct *live* PIDs: this test process and a spawned-but-blocked
    # child (killed at teardown) so both are genuinely alive at lookup time.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        _write_marker(
            8787,
            os.getpid(),
            {"pid": os.getpid(), "session_token": "tok-a", "cwd": str(project_a)},
        )
        _write_marker(
            8787,
            child.pid,
            {"pid": child.pid, "session_token": "tok-b", "cwd": str(project_b)},
        )
        assert resolve_registered_cwd(8787, "tok-a") == str(project_a)
        assert resolve_registered_cwd(8787, "tok-b") == str(project_b)
    finally:
        child.kill()
        child.wait()


def test_non_string_cwd_in_marker_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    _write_marker(8787, os.getpid(), {"pid": os.getpid(), "session_token": "tok-A", "cwd": 12345})
    assert resolve_registered_cwd(8787, "tok-A") is None


def test_malformed_marker_json_is_skipped_not_raised(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    d = paths.proxy_clients_dir(8787)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{os.getpid()}.json").write_text("{not valid json", encoding="utf-8")
    assert resolve_registered_cwd(8787, "tok-A") is None


def test_non_object_marker_json_is_skipped_not_raised(tmp_path, monkeypatch):
    """Syntactically valid JSON that isn't an object (e.g. a bare list) must
    not crash resolution via AttributeError on .get() -- this function runs
    on every request through the middleware, so one bad marker must not take
    down the whole proxy."""
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    d = paths.proxy_clients_dir(8787)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{os.getpid()}.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert resolve_registered_cwd(8787, "tok-A") is None
