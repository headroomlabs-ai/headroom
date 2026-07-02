"""Detachment of the shared proxy subprocess in ``headroom wrap``.

``_start_proxy`` launches the proxy that every wrapped agent on a port shares.
It must outlive an *ungraceful* close of the agent that happened to start it
(closing the terminal window, taskkill, a crash) -- otherwise the OS tree-kills
the proxy and breaks the other live clients, bypassing the marker-based
reference counting in ``_make_cleanup``.

On Windows that means detaching from the launcher's console and Job object via
creation flags; on POSIX ``start_new_session`` already detaches via setsid().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from headroom.cli import wrap as wrap_cli

# Windows-only creation flag; not always exported by the host's ``subprocess``.
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000


class _FakeProc:
    """Stand-in for a live proxy process (``poll() is None``)."""

    returncode = 0

    def poll(self) -> None:
        return None


def _capture_popen_kwargs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Invoke ``_start_proxy`` with all I/O stubbed; return the Popen kwargs."""
    captured: dict[str, Any] = {}

    def _fake_popen(cmd: Any, **kwargs: Any) -> _FakeProc:
        captured.clear()
        captured.update(kwargs)
        return _FakeProc()

    monkeypatch.setattr(wrap_cli.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(wrap_cli, "_get_log_path", lambda: tmp_path / "proxy.log")
    monkeypatch.setattr(wrap_cli, "_resolve_wrap_proxy_timeout_seconds", lambda: 1)

    wrap_cli._start_proxy(8787)
    return captured


def test_start_proxy_detaches_on_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Force the Windows branch regardless of the host OS, and supply the
    # Windows-only ``subprocess`` constants the host lacks on POSIX.
    monkeypatch.setattr(wrap_cli.sys, "platform", "win32")
    monkeypatch.setattr(wrap_cli.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(wrap_cli.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)

    flags = _capture_popen_kwargs(monkeypatch, tmp_path)["creationflags"]

    assert flags & 0x08000000  # CREATE_NO_WINDOW: own hidden console
    assert flags & 0x200  # CREATE_NEW_PROCESS_GROUP: ignore the parent's Ctrl-C
    assert flags & _CREATE_BREAKAWAY_FROM_JOB  # survive Job kill-on-close


def test_start_proxy_keeps_creationflags_zero_off_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(wrap_cli.sys, "platform", "linux")

    kwargs = _capture_popen_kwargs(monkeypatch, tmp_path)

    # POSIX detaches via setsid(); no Windows creation flags are applied.
    assert kwargs["creationflags"] == 0
    assert isinstance(kwargs["start_new_session"], bool)


def test_start_proxy_retries_without_breakaway_when_job_forbids_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Force the Windows branch and supply its constants, as above.
    monkeypatch.setattr(wrap_cli.sys, "platform", "win32")
    monkeypatch.setattr(wrap_cli.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(wrap_cli.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)

    # First spawn raises OSError (the launcher's Job forbids breakaway); the
    # second must succeed without CREATE_BREAKAWAY_FROM_JOB.
    seen: list[int] = []

    def _flaky_popen(cmd: Any, **kwargs: Any) -> _FakeProc:
        seen.append(kwargs["creationflags"])
        if len(seen) == 1:
            raise OSError("a process in the job cannot break away")
        return _FakeProc()

    monkeypatch.setattr(wrap_cli.subprocess, "Popen", _flaky_popen)
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(wrap_cli, "_get_log_path", lambda: tmp_path / "proxy.log")
    monkeypatch.setattr(wrap_cli, "_resolve_wrap_proxy_timeout_seconds", lambda: 1)

    wrap_cli._start_proxy(8787)

    assert len(seen) == 2  # the call was retried exactly once
    assert seen[0] & _CREATE_BREAKAWAY_FROM_JOB  # first attempt requests breakaway
    assert not seen[1] & _CREATE_BREAKAWAY_FROM_JOB  # retry drops it
    assert seen[1] & 0x08000000  # but still gets its own hidden console
    assert seen[1] & 0x200  # and still gets its own process group


class _TrackingFile:
    """Records whether the parent closed its copy of the stdio log handle."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_start_proxy_closes_log_when_both_spawns_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Both spawn attempts raise (e.g. the Job forbids breakaway AND the retry
    # still fails). The parent must not leak the stdio log file handle.
    monkeypatch.setattr(wrap_cli.sys, "platform", "win32")
    monkeypatch.setattr(wrap_cli.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(wrap_cli.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)

    log_file = _TrackingFile()
    monkeypatch.setattr(wrap_cli, "open", lambda *a, **k: log_file, raising=False)

    def _always_fails(cmd: Any, **kwargs: Any) -> _FakeProc:
        raise OSError("spawn failed on both attempts")

    monkeypatch.setattr(wrap_cli.subprocess, "Popen", _always_fails)
    monkeypatch.setattr(wrap_cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(wrap_cli, "_get_log_path", lambda: tmp_path / "proxy.log")
    monkeypatch.setattr(wrap_cli, "_resolve_wrap_proxy_timeout_seconds", lambda: 1)

    with pytest.raises(OSError):
        wrap_cli._start_proxy(8787)

    assert log_file.closed  # finally closed the handle despite the failure
