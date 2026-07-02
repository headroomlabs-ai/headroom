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
