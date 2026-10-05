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

Access control is filesystem permissions, in two layers. The socket itself is
created ``0600`` (connecting needs write permission on it), so only the proxy's
own user can connect even when the parent directory is ``0755`` or a sticky
``/tmp``-style directory. The parent directory guards the socket against being
replaced: a parent this module creates is made ``0700``; a parent that already
exists is never modified, only checked, since something else owns its policy.
"""

from __future__ import annotations

import errno
import os
import signal
import socket
import stat
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

__all__ = [
    "SOCKET_MODE",
    "UDS_SUPPORTED",
    "UdsListener",
    "bind_uds_listener",
    "UdsError",
    "socket_usage_lines",
    "max_uds_path_length",
    "prepare_uds_path",
    "require_uds_support",
    "resolve_uds_path",
    "serving_uds",
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
# Root may unlink anything anyway, so a root-owned parent adds no exposure.
_ROOT_UID = 0

_SUN_PATH_MAX_LINUX = 108
_SUN_PATH_MAX_BSD = 104


#: Owner read and write. Connecting to a Unix socket needs write permission on it,
#: so this mode admits only the proxy's own user, whatever the parent's mode.
SOCKET_MODE = stat.S_IRUSR | stat.S_IWUSR


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

    The parent must be owned by the effective user or by root. A directory's
    owner can unlink or rename any entry in it, sticky bit or not, so another
    user's directory lets that user swap the socket whatever its mode. Root is
    accepted because root can do that to any directory anyway, and shared roots
    such as ``/tmp`` and ``/run`` are root-owned.

    Group/world-writable is tolerated when the sticky bit is set, which is the
    ``/tmp`` case: others may create their own entries but cannot unlink or
    rename ours, so the socket cannot be swapped out from under us.
    """
    try:
        parent_stat = parent.stat()
    except OSError:
        return  # unreadable; bind() will produce the authoritative error

    euid = os.geteuid()
    if parent_stat.st_uid not in (euid, _ROOT_UID):
        raise UdsError(
            f"{parent} is owned by uid {parent_stat.st_uid}, not by you (uid {euid}) or "
            "root, so its owner could replace the socket inside it. Point --uds at a "
            "directory you own."
        )

    mode = parent_stat.st_mode

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


def resolve_uds_path(path: str | os.PathLike[str]) -> Path:
    """The absolute form of a ``--uds`` argument: ``~`` expanded, a relative path anchored at the working directory."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    return resolved


def prepare_uds_path(path: str | os.PathLike[str], *, platform: str | None = None) -> Path:
    """Validate *path*, create its parent ``0700``, and clear a stale socket.

    Returns the resolved path, ready for :func:`bind_uds_listener`.

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

    resolved = resolve_uds_path(path)

    limit = max_uds_path_length(platform)
    encoded = len(str(resolved).encode("utf-8")) + 1  # + trailing NUL
    if encoded > limit:
        raise UdsError(
            f"Socket path is {encoded} bytes, over this platform's {limit}-byte "
            f"sun_path limit: {resolved}. Use a shorter path, e.g. a private "
            "directory under /tmp (on macOS $TMPDIR is itself too deep to fit)."
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


@dataclass
class UdsListener:
    """A bound, not yet listening, stream socket at *path* with mode :data:`SOCKET_MODE`.

    *device* and *inode* identify the socket file this listener created, so
    :meth:`close` can tell it apart from a successor's socket at the same path.
    """

    path: Path
    sock: socket.socket
    device: int
    inode: int

    def close(self) -> None:
        """Close the socket and remove its file if the path still names this socket.

        A successor proxy may have replaced the file once this one stopped
        accepting connections (it then looks stale); that socket is left alone.
        """
        self.sock.close()
        try:
            current = self.path.lstat()
        except FileNotFoundError:
            return
        if (current.st_dev, current.st_ino) == (self.device, self.inode):
            self.path.unlink()


def bind_uds_listener(path: Path) -> UdsListener:
    """Bind a stream socket at *path*, as returned by :func:`prepare_uds_path`.

    The socket's mode is set to :data:`SOCKET_MODE` before it is returned, and it
    is returned without ``listen()`` having been called: uvicorn receives its
    descriptor (``fd=``) and starts listening, so no peer can connect while the
    file still carries the umask-derived mode. Passing the path to uvicorn as
    ``uds=`` instead would leave the socket ``0666``, which in an existing
    ``0755`` or sticky parent lets every local user connect.
    """
    sock = socket.socket(_AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(str(path))
        os.chmod(path, SOCKET_MODE)
        bound = path.lstat()
    except BaseException:
        sock.close()
        raise
    return UdsListener(path=path, sock=sock, device=bound.st_dev, inode=bound.st_ino)


class _Terminated(BaseException):
    """Raised by the SIGTERM handler; a BaseException so ``except Exception`` cannot swallow it."""


def _raise_terminated(signum: int, frame: FrameType | None) -> None:
    raise _Terminated


@contextmanager
def serving_uds(path: Path) -> Iterator[UdsListener]:
    """Bind *path* for the duration of the block and remove the socket file when it exits.

    uvicorn (0.29 and later) finishes a graceful shutdown on SIGTERM and then
    re-raises the signal with the previous handler restored. Under the default
    handler that kills the process before any ``finally`` runs, leaving the
    socket file behind. So on the main thread this installs a SIGTERM handler
    that unwinds the block instead, removes the file, and then re-raises SIGTERM
    under the handler that was in place before, so the process still ends the
    way it would have (exit status 143 under the default handler). Off the main
    thread ``signal.signal`` is unavailable and uvicorn neither captures nor
    re-raises signals, so no handler is installed there.

    With more than one worker, uvicorn's supervisor replaces this handler with its
    own when it starts. On SIGTERM or SIGINT it stops every worker and returns
    without re-raising, so the block exits normally, the file is still removed,
    and the process exits 0, as a multi-worker proxy on a TCP port does. This
    process keeps the listening descriptor open for the whole block, so a worker
    the supervisor restarts inherits the same socket.
    """
    listener = bind_uds_listener(path)
    on_main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, _raise_terminated) if on_main_thread else None
    terminated = False
    try:
        yield listener
    except _Terminated:
        terminated = True
    finally:
        if on_main_thread:
            signal.signal(signal.SIGTERM, previous)
        listener.close()
    if terminated:
        signal.raise_signal(signal.SIGTERM)
