"""Antigravity CLI (agy) MCP registrar.

agy 1.1.x reads MCP server configuration from the global, IDE-shared
``~/.gemini/config/mcp_config.json`` (migrated from the legacy
``~/.gemini/antigravity-cli/mcp_config.json``, which 1.1.x no longer reads —
google-antigravity/antigravity-cli#60), using the same JSON shape as Claude
Code's file path:

    {"mcpServers": {"<name>": {"command": ..., "args": ..., "env": {...}}}}

There is no general-purpose CLI for editing this file, so we read/write the
JSON directly.  We do NOT use marker blocks here (unlike codex.py) because the
JSON format does not admit inline comments; instead we operate on the
``mcpServers`` dict directly — adding a key to register and deleting it to
unregister — which is both safe and merge-friendly (preserves other user and
Antigravity-IDE entries untouched).
"""

from __future__ import annotations

import json
import logging
import os
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .base import (
    MCPRegistrar,
    RegisterResult,
    RegisterStatus,
    ServerSpec,
)
from .base import (
    config_file_lock as _config_file_lock,
)

logger = logging.getLogger(__name__)

#: Config file path relative to home, matching agy's own lookup.
#: agy 1.1.x MIGRATED the MCP-server read-path here from the legacy
#: ``.gemini/antigravity-cli/mcp_config.json`` (which 1.1.x no longer reads —
#: google-antigravity/antigravity-cli#60). This global config is SHARED with the
#: Antigravity IDE, so registration MUST stay merge-not-clobber.
_AGY_CONFIG_RELPATH = ".gemini/config/mcp_config.json"

#: agy's app-data directory (relative to home). Independent of the config file:
#: agy writes its per-tool cache to ``<appdata>/mcp/<server>/<tool>.json``
#: REGARDLESS of which config file declared the server (cli.log: appDataDir).
_AGY_APPDATA_RELPATH = ".gemini/antigravity-cli"


class AgyRegistrar(MCPRegistrar):
    """Register MCP servers with the Antigravity CLI (agy)."""

    name = "agy"
    display_name = "Antigravity CLI"

    def __init__(self, *, home_dir: Path | None = None) -> None:
        """Allow ``home_dir`` override for testing (mirrors codex.py seam).

        Pass ``home_dir`` in tests to redirect all file I/O to a tmp path so
        the real ``~/.gemini`` is never touched.
        """
        home = home_dir if home_dir is not None else Path.home()
        self._config_file: Path = home / _AGY_CONFIG_RELPATH
        self._appdata_dir: Path = home / _AGY_APPDATA_RELPATH

    @property
    def config_dir(self) -> Path:
        """Directory holding ``mcp_config.json`` (the read-path config)."""
        return self._config_file.parent

    @property
    def cache_dir(self) -> Path:
        """agy's per-tool cache root: ``<appdata>/mcp``.

        agy writes ``<appdata>/mcp/<server>/<tool>.json`` when it connects a
        server — the presence of that file is the real exposure signal. This is
        under the app-data dir, NOT the (migrated) config dir, so it is exposed
        separately from ``config_dir``. Derived from the single ``home_dir`` seam.
        """
        return self._appdata_dir / "mcp"

    # ------------------------------------------------------------------
    # MCPRegistrar interface
    # ------------------------------------------------------------------

    def detect(self) -> bool:
        """Return True if agy appears to be installed.

        We key on agy's app-data directory (``~/.gemini/antigravity-cli``) — the
        stable install marker — rather than the config dir, because the migrated
        config dir (``~/.gemini/config``) may not exist until the first server is
        written.  A pre-existing config file is also accepted.  We deliberately
        do NOT shell out to ``shutil.which("agy")`` — the registrar is also used
        in test environments where the CLI may not be on PATH.
        """
        return self._appdata_dir.exists() or self._config_file.exists()

    def get_server(self, server_name: str) -> ServerSpec | None:
        """Return the registered ServerSpec for ``server_name``, or ``None``."""
        servers = _read_json(self._config_file).get("mcpServers")
        if not isinstance(servers, dict):
            return None
        entry = servers.get(server_name)
        if not isinstance(entry, dict):
            return None
        return _try_entry_to_spec(server_name, entry)

    def has_configured_servers(self) -> bool:
        """Inspect all entries before launching MCP-unsafe print-mode clients.

        An unreadable configuration must fail preflight rather than be treated
        as empty: this file is shared with the IDE and can hold user servers.
        """
        config = _read_json_for_write(self._config_file)
        servers = config.get("mcpServers", {})
        if not isinstance(servers, dict):
            raise _MalformedConfigError(f"{self._config_file} has invalid mcpServers")
        return bool(servers)

    def register_server(self, spec: ServerSpec, *, force: bool = False) -> RegisterResult:
        """Idempotently register an MCP server.

        Semantics mirror claude.py's file-path path:

        * Already present and matches → ALREADY.
        * Already present, different, no ``force`` → MISMATCH (no clobber).
        * Already present, different, ``force=True`` → overwrite → REGISTERED.
        * Absent → write → REGISTERED.

        In all cases, only ``spec.name`` is touched; all other ``mcpServers``
        entries are preserved (merge-not-clobber). The read-modify-replace
        transaction is serialized across threads and Headroom processes.
        """
        try:
            with _config_file_lock(self._config_file):
                try:
                    config = _read_json_for_write(self._config_file)
                except (_MalformedConfigError, OSError) as exc:
                    return RegisterResult(
                        RegisterStatus.FAILED,
                        f"refusing to overwrite {self._config_file}: {exc}",
                    )
                servers = config.get("mcpServers")
                if servers is None:
                    servers = {}
                elif not isinstance(servers, dict):
                    return RegisterResult(
                        RegisterStatus.FAILED,
                        f"refusing to overwrite {self._config_file}: mcpServers must be an object",
                    )

                raw_existing = servers.get(spec.name)
                if raw_existing is not None:
                    existing = (
                        _try_entry_to_spec(spec.name, raw_existing)
                        if isinstance(raw_existing, dict)
                        else None
                    )
                    if existing is None:
                        if not force:
                            return RegisterResult(
                                RegisterStatus.MISMATCH,
                                "existing entry has non-string or malformed fields",
                            )
                    else:
                        if _specs_equivalent(existing, spec):
                            return RegisterResult(
                                RegisterStatus.ALREADY, "matches current configuration"
                            )
                        if not force:
                            return RegisterResult(
                                RegisterStatus.MISMATCH, _diff_specs(existing, spec)
                            )

                servers[spec.name] = _spec_to_entry(spec)
                config["mcpServers"] = servers
                try:
                    _write_json(self._config_file, config)
                except OSError as exc:
                    return RegisterResult(
                        RegisterStatus.FAILED, f"could not write {self._config_file}: {exc}"
                    )
                return RegisterResult(RegisterStatus.REGISTERED, f"wrote {self._config_file}")
        except OSError as exc:
            return RegisterResult(
                RegisterStatus.FAILED, f"could not lock {self._config_file}: {exc}"
            )

    def unregister_server(self, server_name: str) -> bool:
        """Remove ``server_name`` from the config; preserves all other entries.

        Returns ``True`` on success, ``False`` if the server was absent or the
        file could not be read/written.
        """
        if not self._config_file.exists():
            return False
        try:
            with _config_file_lock(self._config_file):
                if not self._config_file.exists():
                    return False
                try:
                    config = _read_json_for_write(self._config_file)
                except (_MalformedConfigError, OSError) as exc:
                    logger.debug("agy: refusing to rewrite %s: %s", self._config_file, exc)
                    return False
                servers = config.get("mcpServers", {})
                if not isinstance(servers, dict) or server_name not in servers:
                    return False
                del servers[server_name]
                config["mcpServers"] = servers
                try:
                    _write_json(self._config_file, config)
                except OSError as exc:
                    logger.debug("agy: could not write %s: %s", self._config_file, exc)
                    return False
                return True
        except OSError as exc:
            logger.debug("agy: could not lock %s: %s", self._config_file, exc)
            return False

    def unregister_server_if(
        self,
        server_name: str,
        predicate: Callable[[ServerSpec], bool],
        *,
        on_removed: Callable[[ServerSpec], None] | None = None,
    ) -> str:
        """Atomically verify ownership and remove ``server_name`` under one config lock.

        When ``on_removed`` is provided, it runs while the config lock is still
        held right after the removal write succeeds, so ledger cleanup stays
        serialized against concurrent registrations of a replacement entry.
        Returns ``"removed"`` when ``predicate(current)`` matched and the entry
        was deleted, ``"not_matched"`` when absent/unreadable/non-matching, or
        ``"failed"`` if locking or writing failed.
        """
        if not self._config_file.exists():
            return "not_matched"
        # Fast read-only pre-check: avoid opening a writable lock sidecar when
        # no matching entry exists (e.g. a symlink into a read-only dotfiles dir
        # during `wrap agy --no-mcp`). Ownership is re-verified inside the lock
        # below before any mutation.
        pre_current = self.get_server(server_name)
        if pre_current is None or not predicate(pre_current):
            return "not_matched"
        try:
            with _config_file_lock(self._config_file):
                if not self._config_file.exists():
                    return "not_matched"
                try:
                    config = _read_json_for_write(self._config_file)
                except (_MalformedConfigError, OSError) as exc:
                    logger.debug("agy: refusing to rewrite %s: %s", self._config_file, exc)
                    return "not_matched"
                servers = config.get("mcpServers")
                if not isinstance(servers, dict):
                    return "not_matched"
                raw_entry = servers.get(server_name)
                if not isinstance(raw_entry, dict):
                    return "not_matched"
                current = _try_entry_to_spec(server_name, raw_entry)
                if current is None or not predicate(current):
                    return "not_matched"
                del servers[server_name]
                config["mcpServers"] = servers
                try:
                    _write_json(self._config_file, config)
                except OSError as exc:
                    logger.debug("agy: could not write %s: %s", self._config_file, exc)
                    return "failed"
                if on_removed is not None:
                    on_removed(current)
                return "removed"
        except OSError as exc:
            logger.debug("agy: could not lock %s: %s", self._config_file, exc)
            return "failed"


# ----------------------------------------------------------------------
# JSON helpers (private to this module; do NOT import from claude.py)
# ----------------------------------------------------------------------


class _MalformedConfigError(RuntimeError):
    """Raised when an existing config cannot be parsed before a full rewrite."""


def _read_json(path: Path) -> dict[str, Any]:
    """Read JSON file, returning empty dict if absent or unparseable.

    Safe for READ-ONLY callers. Do NOT use before a full-file rewrite: an
    unparseable file returns ``{}`` here, and writing that back destroys the
    user's config. Use :func:`_read_json_for_write` on the write path instead.
    """
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _read_json_for_write(path: Path) -> dict[str, Any]:
    """Read a JSON object ahead of a full-file rewrite.

    Returns ``{}`` only when the file is absent or empty (safe to create fresh).
    When it has content that is not a JSON object, raises
    :class:`_MalformedConfigError` so the caller aborts instead of overwriting an
    unrelated user config — agy's ``mcp_config.json`` is shared with the
    Antigravity IDE and holds the user's own servers alongside ours.
    """
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")  # OSError propagates to the caller
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _MalformedConfigError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _MalformedConfigError(f"{path} does not contain a JSON object")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically write JSON with LF newlines, preserving symlinks and file mode."""
    target = Path(os.path.realpath(path)) if path.is_symlink() else path
    target.parent.mkdir(parents=True, exist_ok=True)
    existing_mode: int | None = None
    try:
        existing_mode = stat.S_IMODE(target.stat().st_mode)
    except OSError:
        existing_mode = None

    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        if existing_mode is not None:
            try:
                os.chmod(tmp_name, existing_mode)
            except OSError:
                pass
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _spec_to_entry(spec: ServerSpec) -> dict[str, Any]:
    entry: dict[str, Any] = {"command": spec.command}
    if spec.args:
        entry["args"] = list(spec.args)
    if spec.env:
        entry["env"] = dict(spec.env)
    return entry


_ALLOWED_ENTRY_KEYS = frozenset({"command", "args", "env"})


def _try_entry_to_spec(name: str, entry: dict[str, Any]) -> ServerSpec | None:
    """Convert a raw JSON MCP entry into a ServerSpec without coercing invalid types.

    Returning ``None`` for non-string ``command``/``args``/``env`` values or
    extra user-added keys (such as ``cwd``) keeps read-only status non-throwing
    while preventing edited entries from matching an install-ledger fingerprint
    and being deleted on unwrap.
    """
    if not set(entry).issubset(_ALLOWED_ENTRY_KEYS):
        return None
    raw_command = entry.get("command")
    if not isinstance(raw_command, str):
        return None
    raw_args = entry.get("args")
    if raw_args is None and "args" not in entry:
        args: tuple[str, ...] = ()
    elif isinstance(raw_args, list) and all(isinstance(x, str) for x in raw_args):
        args = tuple(raw_args)
    else:
        return None
    raw_env = entry.get("env")
    if raw_env is None and "env" not in entry:
        env: dict[str, str] = {}
    elif isinstance(raw_env, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in raw_env.items()
    ):
        env = dict(raw_env)
    else:
        return None
    return ServerSpec(name=name, command=raw_command, args=args, env=env)


def _specs_equivalent(a: ServerSpec, b: ServerSpec) -> bool:
    return (
        a.name == b.name
        and a.command == b.command
        and tuple(a.args) == tuple(b.args)
        and dict(a.env) == dict(b.env)
    )


def _diff_specs(existing: ServerSpec, requested: ServerSpec) -> str:
    parts: list[str] = []
    if existing.command != requested.command:
        parts.append(f"command {existing.command!r} -> {requested.command!r}")
    if tuple(existing.args) != tuple(requested.args):
        parts.append(f"args {list(existing.args)} -> {list(requested.args)}")
    if dict(existing.env) != dict(requested.env):
        # Never echo env values into conflict messages (stdout/logs): user MCP
        # entries often store API keys or tokens in env.
        existing_keys = sorted(existing.env)
        requested_keys = sorted(requested.env)
        parts.append(f"env keys {existing_keys} -> {requested_keys}")
    if not parts:
        return "spec differs in unidentified field(s)"
    return "; ".join(parts)
