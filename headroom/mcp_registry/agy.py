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
from pathlib import Path
from typing import Any

from .base import MCPRegistrar, RegisterResult, RegisterStatus, ServerSpec

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
        entry = _read_json(self._config_file).get("mcpServers", {}).get(server_name)
        if not isinstance(entry, dict):
            return None
        return _entry_to_spec(server_name, entry)

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
        entries are preserved (merge-not-clobber).
        """
        existing = self.get_server(spec.name)

        if existing is not None:
            if _specs_equivalent(existing, spec):
                return RegisterResult(RegisterStatus.ALREADY, "matches current configuration")
            if not force:
                return RegisterResult(RegisterStatus.MISMATCH, _diff_specs(existing, spec))
            # force=True: fall through and overwrite below.

        return self._write_entry(spec)

    def unregister_server(self, server_name: str) -> bool:
        """Remove ``server_name`` from the config; preserves all other entries.

        Returns ``True`` on success, ``False`` if the server was absent or the
        file could not be read/written.
        """
        if not self._config_file.exists():
            return False
        try:
            config = _read_json_for_write(self._config_file)
        except (_MalformedConfigError, OSError) as exc:
            logger.debug("agy: refusing to rewrite %s: %s", self._config_file, exc)
            return False
        servers: dict[str, Any] = config.get("mcpServers", {})
        if server_name not in servers:
            return False
        del servers[server_name]
        config["mcpServers"] = servers
        try:
            _write_json(self._config_file, config)
        except OSError as exc:
            logger.debug("agy: could not write %s: %s", self._config_file, exc)
            return False
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_entry(self, spec: ServerSpec) -> RegisterResult:
        try:
            config = _read_json_for_write(self._config_file)
        except (_MalformedConfigError, OSError) as exc:
            return RegisterResult(
                RegisterStatus.FAILED,
                f"refusing to overwrite {self._config_file}: {exc}",
            )
        servers: dict[str, Any] = config.setdefault("mcpServers", {})
        servers[spec.name] = _spec_to_entry(spec)
        try:
            _write_json(self._config_file, config)
        except OSError as exc:
            return RegisterResult(
                RegisterStatus.FAILED, f"could not write {self._config_file}: {exc}"
            )
        return RegisterResult(RegisterStatus.REGISTERED, f"wrote {self._config_file}")


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _spec_to_entry(spec: ServerSpec) -> dict[str, Any]:
    entry: dict[str, Any] = {"command": spec.command}
    if spec.args:
        entry["args"] = list(spec.args)
    if spec.env:
        entry["env"] = dict(spec.env)
    return entry


def _entry_to_spec(name: str, entry: dict[str, Any]) -> ServerSpec:
    return ServerSpec(
        name=name,
        command=str(entry.get("command", "")),
        args=tuple(entry.get("args", ())),
        env=dict(entry.get("env", {})),
    )


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
        parts.append(f"env {dict(existing.env)} -> {dict(requested.env)}")
    if not parts:
        return "spec differs in unidentified field(s)"
    return "; ".join(parts)
