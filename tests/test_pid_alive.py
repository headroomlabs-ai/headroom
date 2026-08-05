"""Regression tests for the Windows-safe PID liveness helper (#1544)."""

from __future__ import annotations

import sys
import types

from headroom._subprocess import pid_alive


def test_pid_alive_rejects_non_positive() -> None:
    assert pid_alive(0) is False
    assert pid_alive(-1) is False


def test_pid_alive_prefers_psutil_without_signalling(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(pid_exists=lambda pid: True))

    def boom(pid: int, sig: int) -> None:
        raise AssertionError("os.kill must not run when psutil answers")

    monkeypatch.setattr("headroom._subprocess.os.kill", boom)
    assert pid_alive(4321) is True


def test_pid_alive_systemerror_is_not_alive(monkeypatch) -> None:
    """WinError 87 surfaces as SystemError on Windows; it must read as 'not alive', not crash."""
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        types.SimpleNamespace(pid_exists=lambda pid: (_ for _ in ()).throw(RuntimeError())),
    )
    monkeypatch.setattr(
        "headroom._subprocess.os.kill",
        lambda pid, sig: (_ for _ in ()).throw(SystemError("WinError 87")),
    )
    assert pid_alive(4321) is False


def test_pid_alive_only_uses_signal_zero(monkeypatch) -> None:
    """The liveness probe must never send a real (terminating) signal."""
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        types.SimpleNamespace(pid_exists=lambda pid: (_ for _ in ()).throw(RuntimeError())),
    )
    sent: list[int] = []
    monkeypatch.setattr("headroom._subprocess.os.kill", lambda pid, sig: sent.append(sig))
    assert pid_alive(4321) is True
    assert sent == [0]


def test_pid_alive_win32_no_psutil_never_calls_os_kill(monkeypatch) -> None:
    """On Windows without psutil, pid_alive must not call os.kill (it routes through TerminateProcess)."""
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        types.SimpleNamespace(pid_exists=lambda pid: (_ for _ in ()).throw(RuntimeError())),
    )
    monkeypatch.setattr("headroom._subprocess.sys.platform", "win32")

    fake_handle = 42
    opened: list[int] = []

    def fake_open_process(access, inherit, pid):
        opened.append(pid)
        return fake_handle

    closed: list[int] = []

    def fake_close_handle(handle):
        closed.append(handle)

    fake_kernel32 = types.SimpleNamespace(
        OpenProcess=fake_open_process,
        CloseHandle=fake_close_handle,
    )
    fake_ctypes = types.SimpleNamespace(windll=types.SimpleNamespace(kernel32=fake_kernel32))
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)

    def boom(pid: int, sig: int) -> None:
        raise AssertionError("os.kill must not be called on Windows")

    monkeypatch.setattr("headroom._subprocess.os.kill", boom)

    assert pid_alive(4321) is True
    assert opened == [4321]
    assert closed == [fake_handle]


def test_pid_alive_win32_no_psutil_no_ctypes_returns_conservative(monkeypatch) -> None:
    """On Windows without psutil AND ctypes failure, return True (assume alive) rather than crash."""
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        types.SimpleNamespace(pid_exists=lambda pid: (_ for _ in ()).throw(RuntimeError())),
    )
    monkeypatch.setattr("headroom._subprocess.sys.platform", "win32")
    monkeypatch.setitem(
        sys.modules,
        "ctypes",
        types.SimpleNamespace(
            windll=types.SimpleNamespace(
                kernel32=types.SimpleNamespace(
                    OpenProcess=lambda *a: (_ for _ in ()).throw(OSError("no kernel32")),
                )
            )
        ),
    )

    def boom(pid: int, sig: int) -> None:
        raise AssertionError("os.kill must not be called on Windows")

    monkeypatch.setattr("headroom._subprocess.os.kill", boom)

    assert pid_alive(4321) is True


def test_pid_alive_win32_access_denied_returns_alive(monkeypatch) -> None:
    """OpenProcess returning NULL with ERROR_ACCESS_DENIED means the process exists but is protected."""
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        types.SimpleNamespace(pid_exists=lambda pid: (_ for _ in ()).throw(RuntimeError())),
    )
    monkeypatch.setattr("headroom._subprocess.sys.platform", "win32")

    ERROR_ACCESS_DENIED = 5

    fake_kernel32 = types.SimpleNamespace(
        OpenProcess=lambda access, inherit, pid: 0,
        GetLastError=lambda: ERROR_ACCESS_DENIED,
    )
    fake_ctypes = types.SimpleNamespace(windll=types.SimpleNamespace(kernel32=fake_kernel32))
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)

    def boom(pid: int, sig: int) -> None:
        raise AssertionError("os.kill must not be called on Windows")

    monkeypatch.setattr("headroom._subprocess.os.kill", boom)

    assert pid_alive(4321) is True
