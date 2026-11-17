"""Unix-domain-socket transport for the Headroom proxy.

``headroom proxy --uds PATH`` serves the same ASGI app over an ``AF_UNIX``
socket instead of a TCP port. Nothing about request handling changes — this is
purely the transport shell.

Why a socket at all, when a loopback port already works: no port to collide
with, nothing listening on the network, and access governed by filesystem
permissions rather than by anything reachable over TCP. That suits container
and systemd deployments, and any client that can dial an ``AF_UNIX`` path.

Note for anyone arriving from GH #1779: this does **not** restore Claude Code's
Remote Control. ``ANTHROPIC_UNIX_SOCKET`` does satisfy that feature's
``api.anthropic.com`` host check, but it is reserved for ``claude ssh``, where
the process on the other end of the socket supplies credentials. Setting it
makes Claude Code classify the session as API-key auth, and Remote Control
separately requires claude.ai subscription auth — so the same variable opens one
gate and closes the other. Verified on Claude Code 2.1.198; see
``docs/content/docs/troubleshooting.mdx``.

Access control is filesystem permissions. A socket inherits no credentials of
its own, so the mode of the directory holding it *is* the security boundary.
A parent this module creates is made ``0700``; a parent that already exists is
never modified, only checked, since something else owns its policy.
"""

from __future__ import annotations

import errno
import os
import socket
import stat
import sys
from pathlib import Path

__all__ = [
    "UDS_SUPPORTED",
    "UdsError",
    "socket_usage_lines",
    "max_uds_path_length",
    "prepare_uds_path",
    "remove_uds_path",
    "require_uds_support",
]

# ``AF_UNIX`` is absent from CPython on Windows even where the OS supports the
# address family, and asyncio has no Windows UDS transport either. Reading the
# constant through getattr keeps this module importable — and type-checkable —
# on Windows, where every call site is already behind UDS_SUPPORTED.
_AF_UNIX: int = getattr(socket, "AF_UNIX", -1)

#: The single capability check the rest of the module keys off.
UDS_SUPPORTED = _AF_UNIX != -1

# ``sockaddr_un.sun_path`` is a fixed-size buffer: 108 bytes on Linux, 104 on
# the BSDs and macOS. Overrunning it fails inside bind() with a bare ENAMETOOLONG
# that says nothing about which path was too long, so check it up front.
_SUN_PATH_MAX_LINUX = 108
_SUN_PATH_MAX_BSD = 104


class UdsError(RuntimeError):
    """A Unix socket path cannot be used for the reason described."""


def max_uds_path_length(platform: str | None = None) -> int:
    """Longest usable socket path, including the trailing NUL, for *platform*."""
    plat = sys.platform if platform is None else platform
    if plat.startswith("linux"):
        return _SUN_PATH_MAX_LINUX
    return _SUN_PATH_MAX_BSD


def require_uds_support(platform: str | None = None) -> None:
    """Raise :class:`UdsError` when this interpreter cannot serve on a socket."""
    plat = sys.platform if platform is None else platform
    if plat == "win32" or not UDS_SUPPORTED:
        raise UdsError(
            "--uds needs Unix domain sockets, which are unavailable on this "
            "platform (Python has no socket.AF_UNIX and asyncio has no Windows "
            "UDS transport). Use --port instead."
        )


def _is_live_socket(path: Path) -> bool:
    """True when something is already accepting connections on *path*.

    A leftover socket file from a crashed proxy looks identical to a live one
    on disk; the only way to tell them apart is to try connecting.
    """
    sock = socket.socket(_AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.5)
        sock.connect(str(path))
    except OSError as exc:
        # ECONNREFUSED: nothing is listening, the inode is stale.
        # ENOENT: it vanished between the stat and the connect.
        if exc.errno in (errno.ECONNREFUSED, errno.ENOENT):
            return False
        # EACCES, ETIMEDOUT, anything else: something is there, or we cannot
        # tell. Either way, refuse to unlink it.
        return True
    else:
        return True
    finally:
        sock.close()


# Docs page carrying the client-compatibility detail the banner has no room for.
UDS_DOCS_URL = "https://headroom-docs.vercel.app/docs/proxy#serving-on-a-unix-socket"


def socket_usage_lines(path: str | os.PathLike[str]) -> tuple[str, ...]:
    """How the startup banner describes a socket bind, for every banner.

    Deliberately names no agent and hands out no environment variables. An
    earlier revision printed an ``ANTHROPIC_UNIX_SOCKET=... claude`` recipe here,
    which is a configuration that does not work: it satisfies Claude Code's
    ``api.anthropic.com`` host check but reclassifies the session as API-key
    auth, and the session then fails to authenticate. Printing it at startup
    turned a known-negative result into first-party guidance. The rule this
    encodes is that the banner states a transport requirement and points at the
    docs; per-client wiring belongs in the docs, where it can be qualified.
    """
    return (
        f"  Socket:        {path}",
        "  Client:        must support HTTP over a Unix socket natively",
        f"  Example:       curl --unix-socket {path} http://localhost/health",
        f"  Details:       {UDS_DOCS_URL}",
    )


def _missing_ancestors(target: Path) -> list[Path]:
    """Directories along *target* that do not exist yet, shallowest first."""
    missing: list[Path] = []
    node = target
    while not node.exists():
        missing.append(node)
        if node.parent == node:  # reached the filesystem root
            break
        node = node.parent
    return list(reversed(missing))


def _require_safe_existing_parent(parent: Path) -> None:
    """Reject a pre-existing parent that lets other users swap the socket.

    Deliberately does not repair the mode. The directory predates this call, so
    something else owns its policy — silently tightening a shared directory
    would lock out whatever put it there.

    Group/world-writable is tolerated when the sticky bit is set, which is the
    ``/tmp`` case: others may create their own entries but cannot unlink or
    rename ours, so the socket cannot be swapped out from under us.
    """
    try:
        mode = parent.stat().st_mode
    except OSError:
        return  # unreadable; bind() will produce the authoritative error

    if not mode & (stat.S_IWGRP | stat.S_IWOTH):
        return
    if mode & stat.S_ISVTX:
        return

    raise UdsError(
        f"{parent} is writable by other users and not sticky, so anyone on this "
        f"host could replace the socket inside it (mode {stat.S_IMODE(mode):04o}). "
        "Point --uds at a directory only you can write, or chmod this one to 0700. "
        "Headroom will not change the permissions of a directory it did not create."
    )


def _prepare_parent_dir(parent: Path) -> None:
    """Create *parent* ``0700`` if absent; otherwise validate without mutating."""
    to_create = _missing_ancestors(parent)
    if not to_create:
        _require_safe_existing_parent(parent)
        return

    parent.mkdir(parents=True, exist_ok=True)
    for created in to_create:
        # mkdir's mode is masked by the process umask, so set it explicitly --
        # but only on the directories this call brought into existence.
        try:
            created.chmod(0o700)
        except OSError:
            pass


def prepare_uds_path(path: str | os.PathLike[str], *, platform: str | None = None) -> Path:
    """Validate *path*, create its parent ``0700``, and clear a stale socket.

    Returns the resolved path, ready to hand to uvicorn's ``uds=``.

    The parent is created ``0700`` when it does not exist. An existing parent is
    left exactly as it is -- see :func:`_require_safe_existing_parent`.

    Raises :class:`UdsError` when the platform has no Unix sockets, the path is
    too long for ``sun_path``, an existing parent directory is writable by other
    users, something is already listening there, or the path exists as anything
    other than a socket. That last case matters: a regular file at the target is
    far more likely to be a typo'd argument pointing at real data than a
    leftover, so it is never removed.
    """
    require_uds_support(platform)

    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()

    limit = max_uds_path_length(platform)
    encoded = len(str(resolved).encode("utf-8")) + 1  # + trailing NUL
    if encoded > limit:
        raise UdsError(
            f"Socket path is {encoded} bytes, over this platform's {limit}-byte "
            f"sun_path limit: {resolved}. Use a shorter path, e.g. under $TMPDIR."
        )

    _prepare_parent_dir(resolved.parent)

    if resolved.exists() or resolved.is_symlink():
        mode = resolved.lstat().st_mode
        if not stat.S_ISSOCK(mode):
            raise UdsError(
                f"Refusing to replace {resolved}: it exists and is not a socket. "
                "Point --uds at a path Headroom owns."
            )
        if _is_live_socket(resolved):
            raise UdsError(
                f"Another process is already listening on {resolved}. Stop it, or "
                "choose a different --uds path."
            )
        resolved.unlink()

    return resolved


def remove_uds_path(path: str | os.PathLike[str]) -> None:
    """Unlink *path* if it is still a socket. Never raises.

    uvicorn removes its own socket on a clean shutdown; this covers the paths
    where it does not get the chance.
    """
    try:
        target = Path(path)
        if target.is_socket():
            target.unlink()
    except OSError:
        pass
