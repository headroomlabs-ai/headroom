"""Last-resort liveness watchdog that survives a held GIL.

The in-process compression watchdogs bound slow work with timed thread joins,
which quietly assumes the GIL keeps changing hands: resuming from a timed
``join`` needs the GIL, so a native call that computes without releasing it
freezes every Python thread at once -- uvicorn, ``/readyz``, logging, and the
very watchdog meant to bound the offender. Field report: 61+ seconds of
full-process silence at three times the compression deadline, TCP backlog
still accepting connections nothing would ever answer (#3178).

``faulthandler.dump_traceback_later`` is the one escape hatch the interpreter
offers: its timer runs on a C thread that never takes the GIL, and on expiry
it writes every thread's stack with signal-safe code and (optionally) hard
exits. A Python heartbeat thread re-arms the timer while the interpreter is
healthy; if the GIL is seized for longer than the deadline, the heartbeat
cannot re-arm and the C timer fires -- naming the culprit stack in stderr
(the proxy log) and ending a process that was already unable to serve
anything. Supervised deployments restart it; unsupervised ones trade an
eternal zombie that hangs every client for a visible crash with a diagnosis.

The deadline is deliberately generous (90s against a 20s compression
deadline and 15s stats timeouts): this must only ever fire when the process
is beyond every softer recovery path.
"""

from __future__ import annotations

import atexit
import faulthandler
import logging
import os
import sys
import threading
import time

logger = logging.getLogger(__name__)

ENV_VAR = "HEADROOM_HARD_WATCHDOG_SECS"
DEFAULT_SECS = 90.0
# Below this the heartbeat interval (a third of the deadline) gets close to
# scheduler jitter and a busy-but-healthy interpreter could be shot.
MIN_SECS = 5.0

_started = threading.Event()


def _resolve_secs() -> float:
    raw = os.environ.get(ENV_VAR, "")
    if not raw.strip():
        return DEFAULT_SECS
    try:
        secs = float(raw)
    except ValueError:
        logger.warning("Ignoring non-numeric %s=%r; using %ss", ENV_VAR, raw, DEFAULT_SECS)
        return DEFAULT_SECS
    if secs <= 0:
        return 0.0
    return max(secs, MIN_SECS)


def _arm(secs: float) -> None:
    # Re-arming replaces the previous timer, so a healthy interpreter never
    # lets it expire. ``file`` is the underlying stderr fd, which supervisors
    # already redirect into the proxy log.
    faulthandler.dump_traceback_later(secs, exit=True, file=sys.stderr)


def _heartbeat(secs: float) -> None:
    while True:
        time.sleep(secs / 3.0)
        _arm(secs)


def start_hard_watchdog() -> bool:
    """Arm the watchdog for this process. Returns True when armed.

    Per-process by design: uvicorn workers each arm their own (the timer and
    the GIL are both process-local). Safe to call more than once.
    """
    if _started.is_set():
        return True
    secs = _resolve_secs()
    if not secs:
        logger.info("Hard watchdog disabled (%s=0)", ENV_VAR)
        return False
    _started.set()
    # Normal interpreter shutdown must not be shot by a timer armed moments
    # earlier; CPython additionally cancels the C thread during finalization.
    atexit.register(faulthandler.cancel_dump_traceback_later)
    # First arm happens here, synchronously: the caller is covered from the
    # moment this returns, even if the very next call seizes the GIL before
    # the heartbeat thread gets scheduled.
    _arm(secs)
    threading.Thread(
        target=_heartbeat, args=(secs,), name="headroom-hard-watchdog", daemon=True
    ).start()
    logger.info(
        "Hard watchdog armed: dump all stacks and exit if the interpreter "
        "makes no progress for %.0fs (%s to tune, 0 to disable)",
        secs,
        ENV_VAR,
    )
    return True
