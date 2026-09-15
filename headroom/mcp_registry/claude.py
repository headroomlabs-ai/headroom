"""Claude Code MCP registrar.

Claude Code 2.x stores MCP server configuration in ``~/.claude.json``,
directly under the home directory. The ``claude`` CLI (``claude mcp
add/remove/list/get``) owns this file; setting ``CLAUDE_CONFIG_DIR``
relocates it to ``$CLAUDE_CONFIG_DIR/.claude.json``. Older Claude Code
releases (and the Claude Desktop app) read ``~/.claude/mcp.json`` instead.
This registrar prefers the CLI for writes when available, and reads the
underlying JSON files directly for compare / ``get_server`` so it is robust
to CLI output format changes.

Two of Claude Code's scopes matter here (see ``claude mcp add --scope``):

``user``
    The top-level ``mcpServers`` map. Loads in **every** Claude Code session
    on the machine, in every directory.
``local``
    ``projects[<cwd>].mcpServers``. Loads only for sessions started in that
    one directory, and takes precedence over a user-scope entry of the same
    name.

Scope is a blast-radius decision, not a preference: a server registered at
user scope that fails to start degrades every session on the machine, not
just the project it was installed for (#2787). Servers whose failure modes
we do not control belong at ``local``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from headroom._subprocess import run

from .base import MCPRegistrar, RegisterResult, RegisterStatus, ServerSpec

logger = logging.getLogger(__name__)

#: Claude Code scope covering every session on the machine.
SCOPE_USER = "user"

#: Claude Code scope covering only sessions started in one directory.
SCOPE_LOCAL = "local"

_VALID_SCOPES = (SCOPE_USER, SCOPE_LOCAL)


class ClaudeConfigMutationError(ValueError):
    """Raised when a Claude config cannot be safely changed."""


class ClaudeRegistrar(MCPRegistrar):
    """Register MCP servers with Claude Code."""

    name = "claude"
    display_name = "Claude Code"

    def __init__(
        self,
        *,
        claude_cli: str | None | object = ...,
        home_dir: Path | None = None,
        config_dir: Path | None = None,
        scope: str = SCOPE_USER,
        project_dir: Path | None = None,
    ) -> None:
        """Allow overrides for testing.

        ``claude_cli`` defaults to :func:`shutil.which` lookup. Pass
        ``None`` to force the file-based fallback path. Pass an explicit
        path to point at a specific binary. ``CLAUDE_CONFIG_DIR`` is honored
        for real user sessions; ``home_dir`` isolates file-based reads and
        writes from the caller's real home directory, unless ``config_dir``
        is passed explicitly. It does not isolate CLI subprocess calls (see
        ``claude_cli``) — pass ``claude_cli=None`` alongside ``home_dir`` to
        keep a test fully off the real ``claude`` binary.

        ``scope`` selects where :meth:`register_server` writes: ``"user"``
        (default, machine-wide) or ``"local"`` (the current project only).
        Reads and removals span both scopes unless told otherwise, so a
        caller that switches scope still finds — and can clean up — an entry
        an earlier release installed elsewhere. ``project_dir`` overrides the
        directory ``local`` scope resolves to (test seam); real callers rely
        on the process working directory, which is what Claude Code itself
        keys sessions by.
        """
        if scope not in _VALID_SCOPES:
            raise ValueError(f"scope must be one of {_VALID_SCOPES}, got {scope!r}")
        self.scope = scope
        self._project_dir_override = project_dir
        home = home_dir if home_dir is not None else Path.home()
        modern_dir = _resolve_claude_config_dir(home, config_dir, honor_env=home_dir is None)
        # Legacy config lives under the real ``.claude`` directory regardless
        # of where the modern config resolved to (CLAUDE_CONFIG_DIR only
        # relocates the modern file, per Claude Code's own behavior).
        self._claude_dir = home / ".claude"
        self._modern_dir = modern_dir
        self._isolated_cli_env = home_dir is not None or config_dir is not None
        self._modern_config = modern_dir / ".claude.json"
        self._legacy_config = self._claude_dir / "mcp.json"
        if claude_cli is ...:
            self._claude_cli = shutil.which("claude")
        else:
            # ``...`` sentinel preserves "not set"; explicit None disables CLI.
            self._claude_cli = claude_cli  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # MCPRegistrar interface
    # ------------------------------------------------------------------

    def detect(self) -> bool:
        if self._claude_cli:
            return True
        return self._claude_dir.is_dir() or self._modern_config.exists()

    def get_server(self, server_name: str, *, scope: str | None = None) -> ServerSpec | None:
        """Return the registered spec, or ``None``.

        With no ``scope``, search project scope first, then user scope — the
        order Claude Code itself resolves them in, so the answer is the entry
        the agent would actually launch. Pass ``scope`` to ask about one
        specific location (e.g. "is there still a user-scope entry to clean
        up?").
        """
        # Read from disk regardless of whether the CLI is present — the file
        # format is stable and easier to compare than CLI output.
        for search_scope in self._scopes(scope):
            for config_path in self._config_paths(search_scope):
                entry = self._read_server_entry(config_path, server_name, search_scope)
                if entry is not None:
                    return entry
        return None

    def validate_configs_for_mutation(self) -> None:
        """Validate every Claude config root before an explicit mutation."""
        seen: set[Path] = set()
        for config_path in (self._modern_config, self._legacy_config):
            if config_path in seen or not config_path.exists():
                continue
            seen.add(config_path)
            try:
                raw = config_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ClaudeConfigMutationError(
                    f"could not read Claude config {config_path}: {exc}"
                ) from exc
            try:
                config = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ClaudeConfigMutationError(
                    f"Claude config {config_path} is not valid JSON; refusing to mutate"
                ) from exc
            if not isinstance(config, dict):
                raise ClaudeConfigMutationError(
                    f"Claude config {config_path} must contain a JSON object"
                )
            if "mcpServers" in config and not isinstance(config["mcpServers"], dict):
                raise ClaudeConfigMutationError(
                    f"Claude config {config_path} has a non-object mcpServers; refusing to mutate"
                )

    def register_server(self, spec: ServerSpec, *, force: bool = False) -> RegisterResult:
        # Compare against the entry in the scope we are about to write, not
        # against any entry anywhere: a leftover user-scope copy must not make
        # a project-scope registration look like a no-op.
        existing = self.get_server(spec.name, scope=self.scope)
        if existing is not None:
            if _specs_equivalent(existing, spec):
                return RegisterResult(RegisterStatus.ALREADY, "matches current configuration")
            if not force:
                return RegisterResult(
                    RegisterStatus.MISMATCH,
                    _diff_specs(existing, spec),
                )
            # force=True: remove first, then write fresh below. Restricted to
            # the scope we are writing — overwriting the project entry must not
            # also delete a machine-wide one the user registered themselves.
            self.unregister_server(spec.name, scope=self.scope)

        if self._claude_cli:
            return self._register_via_cli(spec)
        return self._register_via_file(spec)

    def unregister_server(self, server_name: str, *, scope: str | None = None) -> bool:
        """Remove the named server. Returns True if anything was removed.

        With no ``scope``, remove from user scope *and* the current project's
        scope, so ``unwrap`` cleans up no matter which one installed the entry
        (a pre-#2787 release put Serena at user scope). Only the current
        project is touched — unwrapping one project must not deregister the
        server in every other project it was wrapped into.
        """
        removed = False
        for target_scope in self._scopes(scope):
            if self._claude_cli:
                result = run(
                    [str(self._claude_cli), "mcp", "remove", server_name, "-s", target_scope],
                    capture_output=True,
                    text=True,
                    env=self._claude_cli_env(),
                    cwd=self._cli_cwd(target_scope),
                )
                if result.returncode == 0:
                    removed = True
                else:
                    logger.debug("claude mcp remove failed: %s", result.stderr.strip())
            # Always clean up the files too — the CLI only touches the modern
            # config, so a legacy entry (or one it didn't know about) can remain.
            for config_path in self._config_paths(target_scope):
                removed = self._remove_from_file(config_path, server_name, target_scope) or removed
        return removed

    # ------------------------------------------------------------------
    # CLI-backed implementation
    # ------------------------------------------------------------------

    def _register_via_cli(self, spec: ServerSpec) -> RegisterResult:
        cmd = [str(self._claude_cli), "mcp", "add", spec.name, "-s", self.scope]
        for k, v in spec.env.items():
            cmd += ["-e", f"{k}={v}"]
        cmd += ["--", spec.command, *spec.args]

        result = run(
            cmd,
            capture_output=True,
            text=True,
            env=self._claude_cli_env(),
            cwd=self._cli_cwd(self.scope),
        )
        if result.returncode == 0:
            return RegisterResult(
                RegisterStatus.REGISTERED, f"via `claude mcp add` (scope: {self.scope})"
            )
        # CLI failed — try the file fallback rather than giving up.
        logger.warning("claude mcp add failed: %s", result.stderr.strip())
        file_result = self._register_via_file(spec)
        if file_result.status == RegisterStatus.REGISTERED:
            return RegisterResult(
                RegisterStatus.REGISTERED,
                f"via file fallback after CLI failed: {result.stderr.strip()}",
            )
        return RegisterResult(
            RegisterStatus.FAILED,
            f"CLI: {result.stderr.strip()}; file: {file_result.detail}",
        )

    # ------------------------------------------------------------------
    # File-backed implementation (CLI absent / older clients)
    # ------------------------------------------------------------------

    def _register_via_file(self, spec: ServerSpec) -> RegisterResult:
        # Prefer the modern config path. If only the legacy file exists,
        # write to that to avoid surprising older clients — but project scope
        # only exists in the modern config, so it always goes there.
        target = self._modern_config
        if (
            self.scope == SCOPE_USER
            and not self._modern_config.exists()
            and self._legacy_config.exists()
        ):
            target = self._legacy_config

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            config = _read_json_for_write(target)
            servers = self._servers_map_for_write(config)
            servers[spec.name] = _spec_to_entry(spec)
            _write_json(target, config)
        except _MalformedConfigError as exc:
            # Refuse to overwrite: the file holds unrelated Claude state
            # (projects, oauthAccount, history) that a blind rewrite would wipe.
            return RegisterResult(
                RegisterStatus.FAILED,
                f"{target} exists but is not valid JSON ({exc}); refusing to overwrite. "
                "Fix or remove the file, then re-run.",
            )
        except OSError as exc:
            return RegisterResult(RegisterStatus.FAILED, f"could not write {target}: {exc}")
        return RegisterResult(RegisterStatus.REGISTERED, f"wrote to {target} (scope: {self.scope})")

    def _remove_from_file(self, path: Path, server_name: str, scope: str) -> bool:
        if not path.exists():
            return False
        try:
            config = _read_json(path)
        except OSError:
            return False
        servers = self._servers_map(config, scope)
        if servers is None or server_name not in servers:
            return False
        del servers[server_name]
        try:
            _write_json(path, config)
        except OSError:
            return False
        return True

    def _read_server_entry(self, path: Path, server_name: str, scope: str) -> ServerSpec | None:
        if not path.exists():
            return None
        try:
            config = _read_json(path)
        except OSError:
            return None
        servers = self._servers_map(config, scope)
        if servers is None:
            return None
        entry = servers.get(server_name)
        if not isinstance(entry, dict):
            return None
        return _entry_to_spec(server_name, entry)

    # ----------------------------------------------------------------------
    # Scope helpers
    # ----------------------------------------------------------------------

    def _scopes(self, scope: str | None) -> tuple[str, ...]:
        """Scopes to act on: the one requested, or project-then-user."""
        if scope is not None:
            if scope not in _VALID_SCOPES:
                raise ValueError(f"scope must be one of {_VALID_SCOPES}, got {scope!r}")
            return (scope,)
        return (SCOPE_LOCAL, SCOPE_USER)

    def _config_paths(self, scope: str) -> tuple[Path, ...]:
        """Config files that can hold an entry at ``scope``.

        The legacy ``~/.claude/mcp.json`` has no notion of projects, so it is
        only consulted for user scope.
        """
        if scope == SCOPE_LOCAL:
            return (self._modern_config,)
        return (self._modern_config, self._legacy_config)

    def _project_dir(self) -> Path:
        """Directory project scope applies to (the session's working directory)."""
        if self._project_dir_override is not None:
            return self._project_dir_override
        return Path.cwd()

    def ownership_key(self, server_name: str, *, scope: str | None = None) -> str:
        """Stable ledger key for one Claude scope without exposing config paths."""
        target_scope = scope or self.scope
        if target_scope == SCOPE_USER:
            # Preserve the pre-scope key so existing installs migrate safely.
            return server_name
        project = os.path.normcase(os.path.normpath(str(self._project_dir().resolve())))
        return f"{server_name}:local:{project}"

    def _project_key(self, config: dict[str, Any]) -> str:
        """Key under ``projects`` naming the current directory.

        Claude Code keys this map by the session's working directory as a
        string — with forward slashes even on Windows. Reuse whatever key an
        existing entry already uses when it names the same directory, so we
        never add a second entry Claude Code will not read; otherwise write
        the forward-slash form it would write itself.
        """
        cwd = self._project_dir()
        target = os.path.normcase(os.path.normpath(str(cwd)))
        projects = config.get("projects")
        if isinstance(projects, dict):
            for key in projects:
                if isinstance(key, str) and os.path.normcase(os.path.normpath(key)) == target:
                    return key
        return cwd.as_posix()

    def _servers_map(self, config: dict[str, Any], scope: str) -> dict[str, Any] | None:
        """Return the ``mcpServers`` map for ``scope``, or ``None`` if absent.

        Read-only: the returned dict is the one inside ``config``, so callers
        on the write path can mutate it, but nothing is created here.
        """
        container: Any = config
        if scope == SCOPE_LOCAL:
            projects = config.get("projects")
            if not isinstance(projects, dict):
                return None
            container = projects.get(self._project_key(config))
            if not isinstance(container, dict):
                return None
        servers = container.get("mcpServers")
        return servers if isinstance(servers, dict) else None

    def _servers_map_for_write(self, config: dict[str, Any]) -> dict[str, Any]:
        """Return the ``mcpServers`` map for this registrar's scope, creating it."""
        container: dict[str, Any] = config
        if self.scope == SCOPE_LOCAL:
            projects = config.get("projects")
            if not isinstance(projects, dict):
                config["projects"] = projects = {}
            key = self._project_key(config)
            project = projects.get(key)
            if not isinstance(project, dict):
                projects[key] = project = {}
            container = project
        servers = container.get("mcpServers")
        if not isinstance(servers, dict):
            container["mcpServers"] = servers = {}
        return servers

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _cli_cwd(self, scope: str) -> str | None:
        """Working directory for a CLI call, or ``None`` to inherit ours.

        ``claude mcp add -s local`` keys the entry by the *CLI process's*
        working directory, so a registrar pointed at a different project
        directory has to move the subprocess with it.
        """
        if scope != SCOPE_LOCAL or self._project_dir_override is None:
            return None
        return str(self._project_dir_override)

    def _claude_cli_env(self) -> dict[str, str] | None:
        if not self._isolated_cli_env:
            return None
        env = os.environ.copy()
        # Point the CLI at the directory holding the modern ``.claude.json``
        # (CLAUDE_CONFIG_DIR relocates that file), not the legacy ``.claude`` dir.
        env["CLAUDE_CONFIG_DIR"] = str(self._modern_dir)
        return env


def _resolve_claude_config_dir(
    home: Path,
    config_dir: Path | None,
    *,
    honor_env: bool,
) -> Path:
    """Resolve the directory holding the *modern* ``.claude.json`` config.

    Defaults to ``home`` itself, since Claude Code's modern config lives at
    ``~/.claude.json`` directly under the home directory. ``CLAUDE_CONFIG_DIR``,
    when set, relocates it to ``$CLAUDE_CONFIG_DIR/.claude.json``.
    """
    if config_dir is not None:
        return config_dir
    if honor_env:
        env_dir = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
        if env_dir:
            return Path(env_dir).expanduser()
    return home


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file, returning empty dict if absent or unparseable.

    Safe for READ-ONLY callers. Do NOT use before a full-file rewrite: an
    unparseable existing file returns ``{}`` here, and writing that back would
    destroy the user's other config. Use :func:`_read_json_for_write` on the
    write path instead.
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


class _MalformedConfigError(Exception):
    """The target config file exists but is not a parseable JSON object.

    Raised on the write path so we refuse to clobber a file we can't safely
    merge into, rather than silently overwriting the user's state.
    """


def _read_json_for_write(path: Path) -> dict[str, Any]:
    """Read a JSON object for a subsequent full-file rewrite.

    Returns ``{}`` only when the file is absent or empty (safe to start fresh).
    If the file exists with content but does not parse as a JSON object, raise
    :class:`_MalformedConfigError` so the caller aborts instead of overwriting
    unrelated user config (e.g. ``~/.claude/.claude.json`` holds ``projects``,
    ``oauthAccount``, and session history alongside ``mcpServers``).
    """
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")  # OSError propagates to the caller
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _MalformedConfigError(str(exc)) from exc
    if not isinstance(data, dict):
        raise _MalformedConfigError("top-level JSON is not an object")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
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
