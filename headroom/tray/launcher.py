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

_FORCE_EXIT_AFTER_S = 30.0


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
        if child.wait() != QUIT_EXIT_CODE:
            return
        # "Quit Headroom" was chosen: shut the proxy down gracefully, then force
        # exit only as a last resort. The grace period is longer than uvicorn's
        # request-drain timeout so savings are flushed during normal teardown.
        signal.raise_signal(signal.SIGINT)
        timer = threading.Timer(_FORCE_EXIT_AFTER_S, lambda: os._exit(0))
        timer.daemon = True
        timer.start()

    threading.Thread(target=_watch, name="headroom-tray-watch", daemon=True).start()
    return child
