"""Spawn the tray as a child of ``headroom proxy --tray``."""

from __future__ import annotations

import logging
import signal
import subprocess
import sys
import threading

from .app import QUIT_EXIT_CODE, tray_available

logger = logging.getLogger(__name__)


def request_graceful_shutdown() -> None:
    """Signal the proxy to shut down via its normal SIGINT path.

    No hard-exit deadline is scheduled: uvicorn's request drain, the sequential
    cleanup steps and the savings flush have no single bounded total, so a fixed
    timer could kill the process before persistence completes. A second SIGINT
    (or the OS) remains the operator's escape hatch for a hung shutdown.
    """
    signal.raise_signal(signal.SIGINT)


def start_tray_for_proxy(host: str, port: int) -> subprocess.Popen[bytes] | None:
    """Start the tray; returns None (after logging why) when it can't run. Never raises."""
    ok, reason = tray_available()
    if not ok:
        logger.warning("Tray not started: %s", reason)
        print(f"  Tray not started: {reason}", file=sys.stderr)
        return None
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        child = subprocess.Popen(  # noqa: S603 — fixed argv, our own interpreter
            [
                sys.executable,
                "-m",
                "headroom.tray",
                "--host",
                host,
                "--port",
                str(port),
                "--owns-proxy",
            ],
            stdin=subprocess.PIPE,
            creationflags=flags,
        )
    except Exception as exc:
        logger.warning("Tray not started: %s", exc)
        return None

    def _watch() -> None:
        if child.wait() == QUIT_EXIT_CODE:
            request_graceful_shutdown()

    threading.Thread(target=_watch, name="headroom-tray-watch", daemon=True).start()
    return child
