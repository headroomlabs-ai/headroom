"""Unix domain socket listener for the proxy.

``headroom proxy --uds PATH`` serves HTTP on a unix socket instead of a TCP host and port. A socket inside a directory only its owner can enter authenticates every peer by filesystem ownership, so a forwarder never sends credentials to whichever local process happens to hold a free loopback port.

The socket is bound here and handed to uvicorn as a file descriptor rather than passed as uvicorn's ``uds`` argument, because uvicorn's own path:

* chmods the socket to ``0o666`` (or copies the mode of whatever file was at the path), so any local user who can reach the directory can connect;
* lets asyncio silently unlink any socket already at the path, including one a live server is still listening on;
* fails outright in multi-worker mode when the path exists, even when the socket there is stale.

Contract with the caller:

* The parent directory must already exist and must be private to the user running the proxy: owned by the current effective uid and not writable by group or others (``0o700`` is the recommended mode). Whoever can write the directory can unlink the socket and bind their own at the same path, so this is checked, with a one-line error, before anything is bound; a sticky bit does not substitute for it (see :func:`_require_safe_parent`). This module never creates or chmods the directory; the socket's own ``0o600`` mode is the second layer, not the only one. Directories above the parent are not inspected.
* A socket file left at the path by a process that has exited (nothing accepts connections on it) is removed and replaced. A socket that still accepts connections, or any path that is not a socket, is refused with :class:`UnixSocketInUseError`.
* Checking for a stale socket and binding are two steps, so two proxies started on the same path at the same instant can race. Supervising a single proxy per path is the caller's job.
* On clean shutdown the socket file is removed, but only if the path still names the socket this process bound. That includes SIGTERM: uvicorn finishes its graceful shutdown and then re-raises the signal with the previous handler restored, which under the default handler would kill the process before any cleanup ran, so :func:`serving_unix_socket` installs a handler that unwinds the stack instead and re-raises SIGTERM itself once the file is gone. With several workers uvicorn's supervisor handles SIGTERM itself and returns normally, and the file is removed on that path too.
"""

from __future__ import annotations

import errno
import functools
import os
import signal
import socket
import stat
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import FrameType

#: Owner read and write. Connecting to a unix socket needs write permission on it, so this admits only the proxy's own user.
SOCKET_MODE = stat.S_IRUSR | stat.S_IWUSR


class UnixSocketInUseError(RuntimeError):
    """The socket path is held by a live listener or by something that is not a socket."""


class UnixSocketUnusableError(RuntimeError):
    """A unix socket cannot be served here: the platform has none, or the requested path cannot hold one."""


def require_unix_sockets() -> None:
    """Raise :class:`UnixSocketUnusableError` on a platform without ``socket.AF_UNIX``.

    Windows builds of Python define no ``AF_UNIX``, so ``--uds`` would otherwise die with an ``AttributeError`` the first time the socket module is asked for it. Entry points call this before doing any other work.
    """
    if not hasattr(socket, "AF_UNIX"):
        raise UnixSocketUnusableError(
            f"--uds needs unix domain sockets, which {sys.platform} does not provide"
        )


@dataclass
class UnixSocketListener:
    """A bound, not yet listening, unix socket and the identity of the file it created."""

    path: str
    sock: socket.socket
    device: int
    inode: int

    def close(self) -> None:
        """Close the socket and remove its file if the path still names this socket.

        A successor that replaced the stale-looking file after this process stopped accepting is left alone.
        """
        self.sock.close()
        try:
            current = os.lstat(self.path)
        except FileNotFoundError:
            return
        if (current.st_dev, current.st_ino) == (self.device, self.inode):
            os.unlink(self.path)


def _path_fits_sockaddr(length: int) -> bool:
    """Whether the socket module accepts an absolute path of *length* bytes as an AF_UNIX address.

    Connects to a name that does not exist, so a path that fits fails with the OS's own ``ENOENT`` and one that does not is rejected by the socket module before any syscall, with an ``OSError`` that carries no errno.
    """
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.connect("/" + "a" * (length - 1))
    except OSError as exc:
        return exc.errno is not None
    finally:
        probe.close()
    return True


@functools.cache
def max_socket_path_bytes() -> int:
    """The longest path, in bytes, that this platform's ``sockaddr_un`` accepts.

    ``sun_path`` is 104 bytes on macOS and the BSDs and 108 on Linux, and the socket module reserves room for a terminating NUL on some of them. Rather than hard-code that per platform, the limit is measured on the running interpreter, because the socket module's own check is the one a failed bind would hit. Found by doubling to a length that does not fit and bisecting below it.
    """
    fits = 1
    too_long = 2
    while _path_fits_sockaddr(too_long):
        fits, too_long = too_long, too_long * 2
    while too_long - fits > 1:
        middle = (fits + too_long) // 2
        if _path_fits_sockaddr(middle):
            fits = middle
        else:
            too_long = middle
    return fits


def _require_safe_parent(parent: str, socket_path: str) -> None:
    """Raise :class:`UnixSocketUnusableError` unless *parent* is a directory only the current user can change.

    Write permission on a directory is what allows unlinking or renaming the entries in it, whoever owns them, so a directory that another user owns or that group or others can write to lets them delete the socket and put their own in its place, and a client that trusts the path then talks to them. Both are refused. The sticky bit is not accepted as a substitute: it stops other users removing this user's entries, but they could still plant an entry at the path before the proxy binds, and it is not a property of a private directory.
    """
    try:
        info = os.stat(parent)
    except FileNotFoundError:
        raise UnixSocketUnusableError(
            f"directory {parent} for the socket {socket_path} does not exist; create it with mode 700"
        ) from None
    if not stat.S_ISDIR(info.st_mode):
        raise UnixSocketUnusableError(
            f"{parent} holds the socket {socket_path} but is not a directory"
        )
    if info.st_uid != os.geteuid():
        raise UnixSocketUnusableError(
            f"directory {parent} is owned by uid {info.st_uid}, not by the user running the proxy "
            f"(uid {os.geteuid()}); another owner could replace the socket"
        )
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise UnixSocketUnusableError(
            f"directory {parent} is writable by group or others (mode {stat.S_IMODE(info.st_mode):04o}); "
            "they could replace the socket. Run chmod go-w on it or use a private directory"
        )


def checked_unix_socket_path(path: str) -> str:
    """Return the absolute form of *path*, after checking that a socket can be bound there safely.

    ``~`` is expanded and a relative path is resolved against the current directory first, because the limit applies to the path the kernel is handed, not the one the user typed. Raises :class:`UnixSocketUnusableError` when the resolved path is longer than :func:`max_socket_path_bytes`, or when its parent directory is missing, not owned by the current effective user, or writable by group or others. The directory is only inspected, never created or chmodded.
    """
    if path == "":
        raise UnixSocketUnusableError("--uds must be a socket path, not an empty string")
    resolved = os.path.abspath(os.path.expanduser(path))
    length = len(os.fsencode(resolved))
    limit = max_socket_path_bytes()
    if length > limit:
        raise UnixSocketUnusableError(
            f"socket path {resolved} is {length} bytes, longer than the {limit} bytes "
            f"{sys.platform} allows for a unix socket; use a shorter path"
        )
    _require_safe_parent(os.path.dirname(resolved), resolved)
    return resolved


def _remove_stale_socket(path: str) -> None:
    """Remove a socket file at *path* that nothing is listening on.

    Raises :class:`UnixSocketInUseError` when *path* is not a socket or a listener still accepts connections on it.
    """
    try:
        existing = os.lstat(path)
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(existing.st_mode):
        raise UnixSocketInUseError(f"{path} exists and is not a socket; refusing to replace it")
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # Non-blocking so a live listener with a full backlog reports EAGAIN instead of hanging startup.
    probe.setblocking(False)
    try:
        result = probe.connect_ex(path)
    finally:
        probe.close()
    if result == errno.ECONNREFUSED:
        os.unlink(path)
        return
    if result in (0, errno.EAGAIN, errno.EINPROGRESS):
        raise UnixSocketInUseError(f"another process is already listening on {path}")
    raise OSError(result, os.strerror(result), path)


def bind_unix_listener(path: str) -> UnixSocketListener:
    """Bind a stream socket at *path* with mode :data:`SOCKET_MODE`.

    The socket is chmodded before uvicorn calls ``listen()`` on it, so no peer can connect while the file still carries the umask-derived mode. The returned listener's ``path`` is the one resolved by :func:`checked_unix_socket_path`.
    """
    path = checked_unix_socket_path(path)
    _remove_stale_socket(path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(path)
        os.chmod(path, SOCKET_MODE)
        bound = os.lstat(path)
    except BaseException:
        sock.close()
        raise
    return UnixSocketListener(path=path, sock=sock, device=bound.st_dev, inode=bound.st_ino)


class _Terminated(BaseException):
    """Raised from the SIGTERM handler; a BaseException so ``except Exception`` blocks cannot swallow it."""


def _raise_terminated(signum: int, frame: FrameType | None) -> None:
    raise _Terminated


@contextmanager
def serving_unix_socket(path: str) -> Iterator[UnixSocketListener]:
    """Bind *path* for the duration of the block, removing the socket file when it exits.

    On the main thread SIGTERM unwinds the block, the file is removed, and SIGTERM is then re-raised under the handler that was in place before, so the process still ends the way it would have without this listener. ``signal.signal`` only works on the main thread, and off it uvicorn neither captures nor re-raises signals, so no handler is installed there.

    With more than one worker, uvicorn's supervisor replaces this handler with its own when it starts. On SIGTERM or SIGINT it stops every worker and returns without re-raising, so the block exits normally, the file is still removed, and the process exits 0, as a multi-worker proxy on a TCP port does. This process keeps the listening socket open for the whole block, so a worker the supervisor restarts inherits the same socket.
    """
    listener = bind_unix_listener(path)
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
