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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
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
