"""Entry point used when the tray is spawned as a child of ``headroom proxy --tray``."""

from __future__ import annotations

import argparse
import os
import sys
import threading


def _exit_when_stdin_closes() -> None:
    # The proxy owns our stdin pipe; EOF means it died, so don't linger as an orphan.
    try:
        while sys.stdin.buffer.read(1024):
            pass
    except Exception:
        pass
    os._exit(0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m headroom.tray")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--owns-proxy", action="store_true")
    args = parser.parse_args(argv)

    from .app import run_tray, tray_available
    from .core import base_url

    ok, reason = tray_available()
    if not ok:
        print(f"headroom tray unavailable: {reason}", file=sys.stderr)
        return 2
    if args.owns_proxy:
        threading.Thread(target=_exit_when_stdin_closes, daemon=True).start()
    code = run_tray(base_url(args.host, args.port), owns_proxy=args.owns_proxy)
    # Skip interpreter finalization: the stdin reader thread is blocked in a read.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    raise SystemExit(main())
