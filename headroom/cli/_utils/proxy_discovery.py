"""Discover the port of a live Headroom proxy for read-only CLI commands.

``headroom wrap <tool>`` can start the proxy on a port other than the 8787
default — e.g. a copilot-subscription session that must not claim the shared
port bumps to ``port + 1``, or a plain ``--port``-busy fallback. Commands that
only *read* from a running proxy (``dashboard``, ``doctor``, ``inspect``,
``learn``) previously defaulted straight to 8787 with no awareness of where
the proxy actually ended up.

This module scans the per-port wrap-client marker directories that
``headroom.cli.wrap`` already maintains (``paths.proxy_clients_dir(port)``,
one JSON file per live client PID) to find candidate ports, prefers the most
recently started session when several are live, and confirms each candidate
with a ``/health`` probe before trusting it. Kept dependency-light (stdlib +
``headroom.paths``/``headroom.fsutil``/``headroom._subprocess`` + Click only) so
importing it from lightweight read-only commands never pulls in
``headroom.cli.wrap``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import click

from headroom import fsutil
from headroom import paths as _paths
from headroom._subprocess import pid_alive
from headroom.install.health import probe_json

#: Historical default port, used when nothing else can be discovered.
DEFAULT_PORT = 8787

PortOrigin = Literal["explicit", "env", "discovered", "default"]


def _proc_identity(pid: int) -> tuple[str, float] | None:
    """Best-effort ``(source, start_time)`` identity for a PID.

    Mirrors ``headroom.cli.wrap._proc_identity``: used to defeat PID reuse so
    a marker is only trusted while the live PID is the same process that
    wrote it. Returns ``None`` when the start time can't be determined, in
    which case callers fall back to existence-only liveness.
    """
    try:
        import psutil  # type: ignore[import-untyped]  # optional dependency

        return ("psutil", psutil.Process(pid).create_time())
    except Exception:
        pass
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            fields = fh.read().rpartition(b")")[2].split()
        return ("proc", float(fields[19]))
    except (OSError, IndexError, ValueError):
        return None


def _identity_mismatch(src: Any, recorded: Any, pid: int) -> bool:
    """True only if ``pid``'s current identity provably differs from the
    recorded ``(src, recorded)`` identity (i.e. the PID was recycled)."""
    if not isinstance(src, str) or not isinstance(recorded, int | float):
        return False
    ident = _proc_identity(pid)
    if ident is None or ident[0] != src:
        return False
    return abs(ident[1] - float(recorded)) > 1.0


def _marker_pid_reused(marker: Path, pid: int) -> bool:
    """True only if the live ``pid`` is provably a different process than the
    one that wrote ``marker``."""
    try:
        rec = json.loads(fsutil.read_text(marker))
    except (OSError, ValueError):
        return False
    return _identity_mismatch(rec.get("start_src"), rec.get("start_time"), pid)


def live_client_pids(port: int) -> list[int]:
    """Live wrap-client PIDs for ``port``, pruning stale markers as we go."""
    d = _paths.proxy_clients_dir(port)
    if not d.exists():
        return []
    live: list[int] = []
    for marker in d.glob("*.json"):
        try:
            pid = int(marker.stem)
        except ValueError:
            continue
        if not pid_alive(pid) or _marker_pid_reused(marker, pid):
            try:
                marker.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        live.append(pid)
    return live


def _marker_started_at(port: int, pid: int) -> float:
    marker = _paths.proxy_clients_dir(port) / f"{pid}.json"
    try:
        rec = json.loads(fsutil.read_text(marker))
    except (OSError, ValueError):
        return 0.0
    started_at = rec.get("started_at")
    return float(started_at) if isinstance(started_at, int | float) else 0.0


def live_proxy_ports() -> list[tuple[int, float]]:
    """Return ``(port, newest started_at)`` for every port with a live client,
    sorted from most recently started session to oldest.
    """
    clients_dir = _paths.workspace_dir() / "clients"
    if not clients_dir.exists():
        return []
    candidates: list[tuple[int, float]] = []
    try:
        entries = list(clients_dir.iterdir())
    except OSError:
        return []
    for entry in entries:
        if not entry.is_dir():
            continue
        try:
            port = int(entry.name)
        except ValueError:
            continue
        pids = live_client_pids(port)
        if not pids:
            continue
        newest = max(_marker_started_at(port, pid) for pid in pids)
        candidates.append((port, newest))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


def proxy_is_healthy(port: int, *, timeout: float = 1.0) -> bool:
    """Return True when a Headroom proxy answers ``/health`` on ``port``."""
    payload = probe_json(f"http://127.0.0.1:{port}/health", timeout=timeout)
    return payload is not None and payload.get("service") == "headroom-proxy"


def resolve_proxy_port(
    explicit: int | None, *, default: int = DEFAULT_PORT
) -> tuple[int, PortOrigin]:
    """Resolve the port a read-only command (dashboard/doctor/inspect/learn)
    should target.

    Resolution order:

    1. ``explicit`` (an explicit ``--port``) always wins.
    2. ``HEADROOM_PORT`` in the environment always wins over discovery.
    3. Auto-discovery: the most recently started live wrap session with a
       healthy ``/health`` response.
    4. ``default`` (historically 8787) when nothing else applies.

    Returns ``(port, origin)`` so callers can print where the port came from.
    """
    if explicit is not None:
        return explicit, "explicit"

    env_port = os.environ.get("HEADROOM_PORT")
    if env_port:
        try:
            parsed = int(env_port)
        except ValueError:
            raise click.ClickException(
                f"HEADROOM_PORT must be an integer, got {env_port!r}"
            ) from None
        if not 1 <= parsed <= 65535:
            raise click.ClickException(f"HEADROOM_PORT must be between 1 and 65535, got {parsed}")
        return parsed, "env"

    for port, _started_at in live_proxy_ports():
        if proxy_is_healthy(port):
            return port, "discovered"

    return default, "default"
