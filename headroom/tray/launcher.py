"""Spawn the tray as a child of ``headroom proxy --tray``."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading

from .app import QUIT_EXIT_CODE, tray_available

logger = logging.getLogger(__name__)


def start_tray_for_proxy(host: str, port: int) -> subprocess.Popen[bytes] | None:
    """Start the tray; returns None (after logging why) when it can't run. Never raises."""
    ok, reason = tray_available()
    if not ok:
        logger.warning("Tray not started: %s", reason)
        print(f"  Tray not started: {reason}", file=sys.stderr)
        return None
    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
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
        if child.wait() != QUIT_EXIT_CODE:
            return
        # "Quit Headroom" was chosen: shut the proxy down gracefully, then force
        # exit if it hasn't stopped (a stuck shutdown must not strand the user).
        signal.raise_signal(signal.SIGINT)
        threading.Timer(10.0, lambda: os._exit(0)).start()

    threading.Thread(target=_watch, name="headroom-tray-watch", daemon=True).start()
    return child
