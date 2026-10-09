"""OpenAI Codex CLI MCP registrar.

Codex stores MCP server config in ``$CODEX_HOME/config.toml`` when
``CODEX_HOME`` is set, otherwise ``~/.codex/config.toml``, as
``[mcp_servers.<name>]`` tables (with optional ``[mcp_servers.<name>.env]``
sub-tables). There is no general-purpose CLI for adding entries, so we
edit the file in place — using marker-delimited blocks so we can
idempotently inject, replace, and remove our entry without disturbing
anything else the user has configured. The Grok registrar reuses the
same marker-block logic.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from headroom import fsutil

from .base import (
    MCPRegistrar,
    RegisterResult,
    RegisterStatus,
    ServerSpec,
    _diff_specs,
    _entry_to_spec,
    _specs_equivalent,
)

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — exercised only on 3.10
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

_MARKER_START = "# --- Headroom MCP server ---"
_MARKER_END = "# --- end Headroom MCP server ---"


def _marker_start(server_name: str) -> str:
    if server_name == "headroom":
        return _MARKER_START
    return f"# --- Headroom MCP server: {server_name} ---"


def _marker_end(server_name: str) -> str:
    if server_name == "headroom":
        return _MARKER_END
    return f"# --- end Headroom MCP server: {server_name} ---"


def _table_path(line: str) -> list[str] | None:
    """Key path of a ``[table]`` or ``[[array]]`` header line, else ``None``.

    Parsed by tomllib so quoted keys (``[mcp_servers."foo#bar"]``) and trailing
    comments are read as TOML reads them. Only a header line starts with ``[``
    and parses on its own; a continuation line of a multi-line array does not.
    """
    if not line.lstrip().startswith("["):
        return None
    try:
        node: Any = tomllib.loads(line)
    except tomllib.TOMLDecodeError:
        return None
    path: list[str] = []
    while isinstance(node, dict) and node:
        key, node = next(iter(node.items()))
        path.append(key)
    return path


def _evict_foreign_tables(content: str, server_name: str, start: str, end: str) -> str:
    """Move tables Headroom does not own out of its marker span.

    Another app's TOML writer appends a new table before the document's
    trailing comment, so when our span is last in the file (the ChatGPT app's
    ``[mcp_servers.node_repl]`` in ``~/.codex/config.toml``) the table lands
    between our markers and deleting the span would delete it too. Every table
    other than ``mcp_servers.<server_name>`` and its subtables is moved, byte
    for byte and in order, to just after the end marker. This is a line-level
    move, so a header-like line inside a multi-line string can split a table
    wrongly; callers must check the final result with ``_only_server_changed``.
    """
    lines = content.splitlines(keepends=True)
    try:
        i = next(n for n, line in enumerate(lines) if line.rstrip("\r\n") == start)
        j = next(n for n in range(i + 1, len(lines)) if lines[n].rstrip("\r\n") == end)
    except StopIteration:
        return content
    kept: list[str] = []
    foreign: list[str] = []
    in_foreign = False
    for line in lines[i + 1 : j]:
        path = _table_path(line)
        if path is not None:
            in_foreign = path[:2] != ["mcp_servers", server_name]
        (foreign if in_foreign else kept).append(line)
    if not foreign:
        return content
    end_line = lines[j] if lines[j].endswith("\n") else lines[j] + "\n"
    return "".join(lines[: i + 1] + kept + [end_line] + foreign + lines[j + 1 :])


def _without_server(data: dict[str, Any], server_name: str) -> dict[str, Any]:
    servers = data.get("mcp_servers")
    if not isinstance(servers, dict):
        return data
    rest = {k: v for k, v in data.items() if k != "mcp_servers"}
    others = {k: v for k, v in servers.items() if k != server_name}
    return {**rest, "mcp_servers": others} if others else rest


def _only_server_changed(old: str, new: str, server_name: str) -> bool:
    """True when ``new`` parses to ``old`` apart from ``mcp_servers.<server_name>``.

    The fail-closed guard for every rewrite of the marker span: a foreign table
    that could not be moved out of the span, or a file that does not parse,
    makes this False and the rewrite is refused instead of written.
    """
    try:
        before, after = tomllib.loads(old), tomllib.loads(new)
    except tomllib.TOMLDecodeError:
        return False
    return _without_server(before, server_name) == _without_server(after, server_name)


class _TomlMarkerRegistrar(MCPRegistrar):
    """Marker-block registrar for ``<config_dir>/config.toml``.

    Codex and Grok both keep ``[mcp_servers.<name>]`` tables in a
    ``config.toml`` and differ only in where the config directory lives.
    """

    def __init__(self, config_dir: Path) -> None:
        self._config_dir = config_dir
        self._config_file = config_dir / "config.toml"

    # ------------------------------------------------------------------
    # MCPRegistrar interface
    # ------------------------------------------------------------------

    def detect(self) -> bool:
        return self._config_dir.is_dir()

    def get_server(self, server_name: str) -> ServerSpec | None:
        data = self._load_toml()
        servers = data.get("mcp_servers", {})
        if not isinstance(servers, dict):
            return None
        entry = servers.get(server_name)
        if not isinstance(entry, dict):
            return None
        return _entry_to_spec(server_name, entry)

    def register_server(self, spec: ServerSpec, *, force: bool = False) -> RegisterResult:
        existing = self.get_server(spec.name)

        if existing is not None and _specs_equivalent(existing, spec):
            return RegisterResult(RegisterStatus.ALREADY, "matches current configuration")

        if existing is not None and not force:
            content = self._read_text()
            if _marker_start(spec.name) not in content:
                # Entry exists but wasn't written by us — refuse to clobber.
                return RegisterResult(
                    RegisterStatus.MISMATCH,
                    "user-managed [mcp_servers."
                    f"{spec.name}] entry outside Headroom markers; "
                    f"{_diff_specs(existing, spec)}",
                )
            return RegisterResult(RegisterStatus.MISMATCH, _diff_specs(existing, spec))

        if existing is not None and force:
            content = self._read_text()
            if _marker_start(spec.name) not in content:
                # Even force=True is only allowed to replace blocks that
                # Headroom owns. Otherwise appending our table would create a
                # duplicate [mcp_servers.<name>] TOML section and may clobber a
                # user-managed integration.
                return RegisterResult(
                    RegisterStatus.MISMATCH,
                    "user-managed [mcp_servers."
                    f"{spec.name}] entry outside Headroom markers; "
                    f"{_diff_specs(existing, spec)}",
                )
            # Drop any prior Headroom block before re-writing.
            self.unregister_server(spec.name)

        # `existing is None` here can also mean the file is present but
        # unparseable, or defines mcp_servers[.<name>] as a non-table.
        # _write_block appends a `[mcp_servers.<name>]` table, so appending into
        # an unparseable file corrupts it further, and appending alongside a
        # non-table entry creates a duplicate `[mcp_servers.<name>]` key that
        # tomllib and the CLI then reject — destroying a previously-valid user config.
        # Refuse rather than clobber, mirroring the claude (#1660) / opencode
        # (#1661) guards.
        if existing is None:
            reason = self._unmergeable_reason(spec.name)
            if reason is not None:
                return RegisterResult(
                    RegisterStatus.FAILED,
                    f"{reason}; refusing to overwrite. Fix or remove the file, then re-run.",
                )

        return self._write_block(spec)

    def unregister_server(self, server_name: str) -> bool:
        # Only removes the marker-block we wrote. User-managed entries
        # outside markers are intentionally preserved.
        if not self._config_file.exists():
            return False
        marker_start = _marker_start(server_name)
        marker_end = _marker_end(server_name)
        original = self._read_text()
        content = _evict_foreign_tables(original, server_name, marker_start, marker_end)
        if marker_start not in content or marker_end not in content:
            return False
        try:
            start = content.index(marker_start)
            end = content.index(marker_end) + len(marker_end)
        except ValueError:
            return False
        before = content[:start].rstrip("\n")
        after = content[end:].lstrip("\n")
        if before and after:
            new_content = before + "\n\n" + after
        else:
            new_content = (before or after).rstrip("\n") + ("\n" if (before or after) else "")
        if not _only_server_changed(original, new_content, server_name):
            logger.warning(
                "Not removing the Headroom block for %s from %s: the file does not parse, "
                "or the block holds entries Headroom could not move out safely.",
                server_name,
                self._config_file,
            )
            return False
        try:
            fsutil.write_text(self._config_file, new_content)
        except OSError:
            return False
        return True

    # ------------------------------------------------------------------
    # File IO
    # ------------------------------------------------------------------

    def _load_toml(self) -> dict[str, Any]:
        if not self._config_file.exists():
            return {}
        try:
            # Read via fsutil (UTF-8 with locale fallback) so a config that a
            # tool wrote in the system locale (e.g. GBK) still parses instead
            # of failing tomllib's UTF-8 requirement. See #733.
            data = tomllib.loads(fsutil.read_text(self._config_file))
        except (tomllib.TOMLDecodeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def _unmergeable_reason(self, name: str) -> str | None:
        """Return why the existing config cannot be safely merged, or ``None``.

        ``_write_block`` appends a ``[mcp_servers.<name>]`` table. That is only
        safe when the file is absent/empty or parses as a TOML table whose
        ``mcp_servers`` (and ``mcp_servers.<name>``) are tables. A present-but-
        unparseable file, or a non-table ``mcp_servers`` / ``mcp_servers.<name>``,
        would be corrupted (unparseable) or made to hold a duplicate key
        (non-table entry) by a blind append.
        """
        if not self._config_file.exists():
            return None
        raw = self._read_text()
        if not raw.strip():
            return None
        try:
            data = tomllib.loads(fsutil.read_text(self._config_file))
        except (tomllib.TOMLDecodeError, OSError) as exc:
            return f"{self._config_file} is not valid TOML ({exc})"
        if not isinstance(data, dict):
            return f"{self._config_file} top-level TOML is not a table"
        servers = data.get("mcp_servers")
        if servers is not None and not isinstance(servers, dict):
            return f"{self._config_file} has a non-table mcp_servers"
        if isinstance(servers, dict):
            entry = servers.get(name)
            if entry is not None and not isinstance(entry, dict):
                return f"{self._config_file} has a non-table mcp_servers.{name}"
        return None

    def _read_text(self) -> str:
        return fsutil.read_text(self._config_file, default="")

    def _write_block(self, spec: ServerSpec) -> RegisterResult:
        block = _render_block(spec)
        try:
            self._config_dir.mkdir(parents=True, exist_ok=True)
            marker_start = _marker_start(spec.name)
            marker_end = _marker_end(spec.name)
            original = self._read_text()
            content = _evict_foreign_tables(original, spec.name, marker_start, marker_end)
            if marker_start in content and marker_end in content:
                start = content.index(marker_start)
                end = content.index(marker_end) + len(marker_end)
                content = (
                    content[:start].rstrip("\n")
                    + ("\n\n" if content[:start].rstrip("\n") else "")
                    + block
                    + "\n"
                    + content[end:].lstrip("\n")
                )
            elif content.strip():
                content = content.rstrip("\n") + "\n\n" + block + "\n"
            else:
                content = block + "\n"
            if not _only_server_changed(original, content, spec.name):
                return RegisterResult(
                    RegisterStatus.FAILED,
                    f"{self._config_file} does not parse, or the Headroom block holds "
                    "entries Headroom could not move out safely; refusing to rewrite it.",
                )
            fsutil.write_text(self._config_file, content)
        except OSError as exc:
            return RegisterResult(
                RegisterStatus.FAILED, f"could not write {self._config_file}: {exc}"
            )
        return RegisterResult(RegisterStatus.REGISTERED, f"wrote to {self._config_file}")


class CodexRegistrar(_TomlMarkerRegistrar):
    """Register MCP servers with the OpenAI Codex CLI."""

    name = "codex"
    display_name = "OpenAI Codex CLI"

    def __init__(self, *, home_dir: Path | None = None) -> None:
        if home_dir is not None:
            config_dir = home_dir / ".codex"
        else:
            from headroom.install.paths import codex_home_dir

            config_dir = codex_home_dir()
        super().__init__(config_dir)


# ----------------------------------------------------------------------
# TOML rendering / parsing helpers (kept module-private)
# ----------------------------------------------------------------------


def _render_block(spec: ServerSpec) -> str:
    """Render a Headroom-marked TOML block for ``spec``."""
    lines: list[str] = [
        _marker_start(spec.name),
        f"[mcp_servers.{spec.name}]",
        f"command = {_toml_str(spec.command)}",
    ]
    if spec.args:
        items = ", ".join(_toml_str(a) for a in spec.args)
        lines.append(f"args = [{items}]")
    if spec.env:
        lines.append("")
        lines.append(f"[mcp_servers.{spec.name}.env]")
        for k, v in spec.env.items():
            lines.append(f"{k} = {_toml_str(v)}")
    lines.append(_marker_end(spec.name))
    return "\n".join(lines)


# TOML basic strings forbid literal control characters other than tab, so a
# value carrying a newline/carriage-return (e.g. a multi-line PEM key in an
# env var) must be escaped or the rendered block is unparseable TOML. Only
# ``\`` and ``"`` were escaped before, so such a value produced invalid TOML
# and the write guard rejected the whole registration with a misleading
# "file does not parse" error.
_TOML_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def _toml_str(s: str) -> str:
    """Render a Python string as a TOML basic string literal."""
    out: list[str] = []
    for ch in s:
        escape = _TOML_ESCAPES.get(ch)
        if escape is not None:
            out.append(escape)
        elif ch < "\x20" or ch == "\x7f":
            # Remaining C0 controls (and DEL) have no short escape; TOML requires
            # the \uXXXX form.
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'
