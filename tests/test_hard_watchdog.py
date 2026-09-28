"""The hard watchdog must fire under a held GIL -- the case nothing else survives.

The proxy's soft watchdogs (timed thread joins, asyncio timeouts) all need the
GIL to act, so a native call that computes without releasing it freezes the
process beyond their reach (#3178: 61+ seconds of full-process silence at three
times the compression deadline). ``faulthandler.dump_traceback_later``'s timer
is a C thread that never takes the GIL; these tests pin the two properties the
design depends on:

* a process whose GIL is seized past the deadline is dumped and exited, and
* a healthy process re-arms the timer and is never shot.

The GIL seizure is simulated with ``ctypes.PyDLL`` -- unlike ``CDLL``, calls
through it do NOT release the GIL, so ``libc sleep()`` becomes a perfect stand
-in for a native call that holds the interpreter hostage.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from headroom.proxy.hard_watchdog import DEFAULT_SECS, MIN_SECS, _resolve_secs


def test_resolve_secs_defaults_and_clamps(monkeypatch):
    monkeypatch.delenv("HEADROOM_HARD_WATCHDOG_SECS", raising=False)
    assert _resolve_secs() == DEFAULT_SECS

    monkeypatch.setenv("HEADROOM_HARD_WATCHDOG_SECS", "0")
    assert _resolve_secs() == 0.0
    monkeypatch.setenv("HEADROOM_HARD_WATCHDOG_SECS", "-3")
    assert _resolve_secs() == 0.0

    # A deadline near scheduler jitter would shoot healthy processes.
    monkeypatch.setenv("HEADROOM_HARD_WATCHDOG_SECS", "1")
    assert _resolve_secs() == MIN_SECS

    monkeypatch.setenv("HEADROOM_HARD_WATCHDOG_SECS", "not-a-number")
    assert _resolve_secs() == DEFAULT_SECS


_SEIZE_GIL = textwrap.dedent(
    """
    import ctypes, ctypes.util, sys
    from headroom.proxy.hard_watchdog import start_hard_watchdog

    assert start_hard_watchdog()
    libc = ctypes.PyDLL(ctypes.util.find_library("c"))
    print("SEIZING", flush=True)
    libc.sleep(60)  # PyDLL: the GIL is held for the whole call
    print("UNREACHABLE", flush=True)
    """
)

_HEALTHY = textwrap.dedent(
    """
    import time
    from headroom.proxy.hard_watchdog import start_hard_watchdog

    assert start_hard_watchdog()
    time.sleep(12)  # well past the 5s deadline; sleep releases the GIL
    print("SURVIVED", flush=True)
    """
)


@pytest.mark.skipif(sys.platform == "win32", reason="libc lookup is POSIX-only")
def test_seized_gil_is_dumped_and_exited():
    proc = subprocess.run(
        [sys.executable, "-c", _SEIZE_GIL],
        env={"HEADROOM_HARD_WATCHDOG_SECS": "5", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert "UNREACHABLE" not in proc.stdout
    # faulthandler's timeout path exits the process abnormally...
    assert proc.returncode != 0
    # ...after writing every thread's stack to stderr, naming the culprit.
    assert "Thread 0x" in proc.stderr or "Current thread" in proc.stderr
    assert "sleep" in proc.stderr or "SEIZING" in proc.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="keep the pair symmetric")
def test_healthy_process_is_never_shot():
    proc = subprocess.run(
        [sys.executable, "-c", _HEALTHY],
        env={"HEADROOM_HARD_WATCHDOG_SECS": "5", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert proc.returncode == 0, proc.stderr
    assert "SURVIVED" in proc.stdout


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("HEADROOM_HARD_WATCHDOG_SECS", "0")
    from headroom.proxy import hard_watchdog

    monkeypatch.setattr(hard_watchdog, "_started", type(hard_watchdog._started)())
    assert hard_watchdog.start_hard_watchdog() is False
