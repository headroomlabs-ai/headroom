"""Tests for headroom._subprocess.pid_alive — non-destructive PID liveness probe."""

from __future__ import annotations

import os

from headroom._subprocess import pid_alive


class TestPidAlive:
    """pid_alive must never signal/terminate the target process."""

    def test_current_process_is_alive(self):
        assert pid_alive(os.getpid()) is True

    def test_zero_pid_is_not_alive(self):
        assert pid_alive(0) is False

    def test_negative_pid_is_not_alive(self):
        assert pid_alive(-1) is False

    def test_stale_pid_is_not_alive(self):
        assert pid_alive(2**22 + 7) is False

    def test_system_error_treated_as_dead(self, monkeypatch):
        """SystemError from os.kill (Windows WinError 87) → not alive."""
        monkeypatch.setattr(
            "headroom._subprocess.os.kill",
            lambda pid, sig: (_ for _ in ()).throw(SystemError("WinError 87")),
        )
        _real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def _no_psutil(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("no psutil")
            return _real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _no_psutil)
        assert pid_alive(99999) is False

    def test_permission_error_means_alive(self, monkeypatch):
        """PermissionError from os.kill means process exists but we can't signal it."""
        monkeypatch.setattr(
            "headroom._subprocess.os.kill",
            lambda pid, sig: (_ for _ in ()).throw(PermissionError("not allowed")),
        )
        import builtins

        _real_import = builtins.__import__

        def _no_psutil(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("no psutil")
            return _real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _no_psutil)
        assert pid_alive(99999) is True
