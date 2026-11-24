"""Abstract base for per-agent MCP registrars.

The MCP protocol itself is universal, and the headroom MCP server
(``headroom mcp serve``) is a single stdio binary that any compliant client
can launch. What differs between agents (Claude Code, Cursor, Codex, ...) is
how each one *learns* that a server exists: each invented its own config
file, format, and registration mechanism.

Subclasses of :class:`MCPRegistrar` own one agent's registration mechanism.
The orchestrator in :mod:`headroom.mcp_registry.install` calls a fleet of
registrars to install headroom across every detected agent.
"""

from __future__ import annotations

import errno
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class RegisterStatus(str, Enum):
    """Outcome of a :meth:`MCPRegistrar.register_server` call."""

    REGISTERED = "registered"
    """Newly written to the agent's MCP config."""

    ALREADY = "already"
    """Already present with a configuration that matches the requested spec."""

    MISMATCH = "mismatch"
    """Already present but with a different configuration; left untouched."""

    FAILED = "failed"
    """Registration was attempted but the agent's tooling rejected it."""

    NOT_DETECTED = "not_detected"
    """The agent does not appear to be installed on this system."""

    NO_SDK = "no_sdk"
    """A required Python dependency is missing (e.g. the ``mcp`` package)."""


@dataclass
class ServerSpec:
    """Universal description of an MCP server to register.

    Each registrar serializes this to its agent's native config format. The
    fields cover what every JSON/TOML schema we've seen requires.
    """

    name: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class RegisterResult:
    """Outcome plus a human-readable detail string."""

    status: RegisterStatus
    detail: str | None = None

    @property
    def ok(self) -> bool:
        """True when the server is registered (newly or already)."""
        return self.status in (RegisterStatus.REGISTERED, RegisterStatus.ALREADY)


class MCPRegistrar(ABC):
    """Per-agent MCP server registrar.

    Each subclass owns exactly one agent's config schema and write path.

    Contract:

    * :meth:`detect` — does this agent appear installed?
    * :meth:`get_server` — read current config; return the spec or ``None``.
    * :meth:`register_server` — idempotent install. If the named server is
      already registered with a different spec, returns
      :attr:`RegisterStatus.MISMATCH` and does **not** overwrite unless
      ``force=True``.
    * :meth:`unregister_server` — remove the named server.
    """

    #: Stable agent identifier ("claude", "cursor", "codex", ...).
    name: str = ""

    #: Human-readable display name ("Claude Code", "Cursor", ...).
    display_name: str = ""

    @abstractmethod
    def detect(self) -> bool:
        """Return True if this agent appears to be installed."""

    @abstractmethod
    def get_server(self, server_name: str) -> ServerSpec | None:
        """Return the registered :class:`ServerSpec`, or ``None`` if absent."""

    @abstractmethod
    def register_server(self, spec: ServerSpec, *, force: bool = False) -> RegisterResult:
        """Idempotently register an MCP server."""

    @abstractmethod
    def unregister_server(self, server_name: str) -> bool:
        """Remove the named server. Returns True on success."""


# ----------------------------------------------------------------------
# Spec helpers shared by the registrars whose entries use the common
# ``{"command", "args", "env"}`` shape (Claude, Antigravity, Codex, Grok).
# OpenCode has its own entry format and keeps its own converters.
# ----------------------------------------------------------------------


def _spec_to_entry(spec: ServerSpec) -> dict[str, Any]:
    entry: dict[str, Any] = {"command": spec.command}
    if spec.args:
        entry["args"] = list(spec.args)
    if spec.env:
        entry["env"] = dict(spec.env)
    return entry


def _entry_to_spec(name: str, entry: dict[str, Any]) -> ServerSpec:
    args_value = entry.get("args", [])
    if isinstance(args_value, list):
        args = tuple(str(x) for x in args_value)
    else:
        args = ()
    env_value = entry.get("env", {})
    env: dict[str, str] = {}
    if isinstance(env_value, dict):
        env = {str(k): str(v) for k, v in env_value.items()}
    return ServerSpec(
        name=name,
        command=str(entry.get("command", "")),
        args=args,
        env=env,
    )


def _specs_equivalent(a: ServerSpec, b: ServerSpec) -> bool:
    """Two specs match when every field is equal."""
    return (
        a.name == b.name
        and a.command == b.command
        and tuple(a.args) == tuple(b.args)
        and dict(a.env) == dict(b.env)
    )


def _diff_specs(existing: ServerSpec, requested: ServerSpec) -> str:
    """Render the difference between two specs for human consumption."""
    parts: list[str] = []
    if existing.command != requested.command:
        parts.append(f"command {existing.command!r} -> {requested.command!r}")
    if tuple(existing.args) != tuple(requested.args):
        parts.append(f"args {list(existing.args)} -> {list(requested.args)}")
    if dict(existing.env) != dict(requested.env):
        parts.append(f"env {dict(existing.env)} -> {dict(requested.env)}")
    if not parts:
        return "spec differs in unidentified field(s)"
    return "; ".join(parts)


_CONFIG_THREAD_LOCKS_GUARD = threading.Lock()
_CONFIG_THREAD_LOCKS: dict[Path, threading.Lock] = {}

_WIN_LOCK_RETRY_ERRNOS = frozenset(
    code
    for name in ("EACCES", "EAGAIN", "EDEADLK", "EDEADLOCK")
    if (code := getattr(errno, name, None)) is not None
)
_WIN_LOCK_RETRY_WINERRORS = frozenset({32, 33})  # ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION


def _is_windows_lock_contention(exc: OSError) -> bool:
    """True when msvcrt.locking failed because another handle holds the lock."""
    return (
        exc.errno in _WIN_LOCK_RETRY_ERRNOS
        or getattr(exc, "winerror", None) in _WIN_LOCK_RETRY_WINERRORS
    )


def _config_thread_lock_for(path: Path) -> threading.Lock:
    key = path.resolve()
    with _CONFIG_THREAD_LOCKS_GUARD:
        lock = _CONFIG_THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _CONFIG_THREAD_LOCKS[key] = lock
        return lock


@contextmanager
def config_file_lock(path: Path) -> Iterator[None]:
    """Serialize read-modify-replace on a shared MCP config across threads and processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with _config_thread_lock_for(lock_path):
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                os.chmod(lock_path, 0o600)
            except OSError:
                pass
            if sys.platform == "win32":
                import msvcrt

                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"0")
                os.lseek(fd, 0, os.SEEK_SET)
                while True:
                    try:
                        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as exc:
                        if not _is_windows_lock_contention(exc):
                            raise
                        time.sleep(0.05)
                try:
                    yield
                finally:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
