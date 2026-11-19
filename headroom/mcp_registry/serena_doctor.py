"""Diagnose a Serena MCP server whose language server cannot start (#3914).

A Serena tool call that times out says nothing about *why*. On WSL the usual
cause is the toolchain the agent hands to Serena: an interactive terminal
gets fnm/nvm/asdf from ``~/.bashrc``, but anything started without one stops
at the ``case $- in`` guard at the top of that file, so the Windows ``npm``
WSL appends to ``PATH`` wins. Serena then fails to install or start its
TypeScript language server, and the running server keeps that failed state
until it is restarted.

Each check here reports what *actually* resolves (the real path of the binary
that would run), never what is configured, and nothing here treats "the
server is registered" or "the process is up" as healthy: only
:func:`run_health_check`, which makes real symbol calls through Serena's own
``project health-check``, proves the language server answers.

Every check is read-only. Paths are redacted (home directory and Windows user
name) before they are shown, so the output can be pasted into an issue.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePath
from typing import Any

from headroom._subprocess import Popen as popen_text
from headroom._subprocess import kill_process_tree

from .base import ServerSpec

#: Binaries Serena needs: ``uvx`` launches it; ``node``/``npm``/``npx``
#: install and run the TypeScript/JavaScript language servers.
TOOLCHAIN = ("uvx", "node", "npm", "npx")
_NODE_TOOLCHAIN = ("node", "npm")

_WINDOWS_MOUNT = re.compile(r"^/mnt/[a-zA-Z]/")
# A Windows profile dir as WSL (/mnt/c/Users/<name>) or Windows (C:\Users\<name>) spells it.
_WINDOWS_USER_DIR = re.compile(
    r"((?:/mnt/[a-zA-Z]/|\b[a-zA-Z]:[\\/])Users[\\/])[^\\/\s]+", re.IGNORECASE
)

# Non-interactive guards Ubuntu/Debian-style ~/.bashrc files open with.
_CASE_GUARD_START = re.compile(r"^\s*case\s+[\"']?\$-[\"']?\s+in\b")
_CASE_GUARD_END = re.compile(r"^\s*esac\b")
_ONE_LINE_GUARD = re.compile(
    r"^\s*\[\[?\s*(-z\s+[\"']?\$PS1[\"']?|\$-\s*!=\s*\*i\*)\s*\]\]?\s*&&\s*return\b"
)
_TOOLCHAIN_SETUP = (
    ("fnm", re.compile(r"\bfnm\s+env\b")),
    ("nvm", re.compile(r"\bNVM_DIR\b|nvm\.sh\b")),
    ("asdf", re.compile(r"asdf\.(sh|bash)\b")),
    ("mise", re.compile(r"\bmise\s+activate\b")),
    ("volta", re.compile(r"\bVOLTA_HOME\b")),
)

_LOG_LEVEL_ERROR = re.compile(r"^(ERROR|CRITICAL)\b")
_LOG_TAIL_BYTES = 64 * 1024
_MAX_LOG_LINES = 5
_MAX_LINE_CHARS = 300
_MAX_PROBE_LINES = 5
_PROBE_VERDICT = re.compile(r"^\W*Health check (passed|failed)")

#: Default ceiling for the health-check probe. A first run may install the
#: language server, so this is minutes, not seconds.
DEFAULT_PROBE_TIMEOUT_S = 300

Runner = Callable[[Sequence[str], float], "subprocess.CompletedProcess[str]"]


class Severity(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class Finding:
    """One diagnostic outcome; ``hint`` says what to do about it."""

    severity: Severity
    summary: str
    hint: str | None = None


@dataclass(frozen=True)
class ToolPath:
    """Where one toolchain binary resolves on the agent's ``PATH``."""

    name: str
    found: str | None
    real: str | None

    @property
    def is_windows(self) -> bool:
        return any(p is not None and _WINDOWS_MOUNT.match(p) for p in (self.found, self.real))


def redact(text: str, home: PurePath) -> str:
    """Hide the home directory and Windows user names in a path or log line."""
    home_str = str(home).rstrip("/\\")
    if home_str:
        for sep in ("/", "\\"):  # native Windows paths use backslashes
            text = text.replace(home_str + sep, "~" + sep)
    return _WINDOWS_USER_DIR.sub(r"\1<user>", text)


def serena_home(environ: Mapping[str, str], home: Path) -> Path:
    """Serena's per-user directory: ``$SERENA_HOME`` or ``~/.serena``."""
    configured = environ.get("SERENA_HOME", "").strip()
    return Path(configured) if configured else home / ".serena"


def resolve_toolchain(
    path_env: str | None,
    which: Callable[..., str | None] | None = None,
    realpath: Callable[[str], str] = os.path.realpath,
) -> list[ToolPath]:
    """Resolve each toolchain binary the way a child of this process would."""
    which = which or shutil.which
    resolved = []
    for name in TOOLCHAIN:
        found = which(name, path=path_env)
        resolved.append(ToolPath(name, found, realpath(found) if found else None))
    return resolved


def check_toolchain(tools: Sequence[ToolPath], home: Path) -> list[Finding]:
    by_name = {tool.name: tool for tool in tools}
    findings: list[Finding] = []

    if by_name["uvx"].found is None:
        findings.append(
            Finding(
                Severity.FAIL,
                "uvx not found on PATH: the agent cannot launch Serena",
                "Install uv (https://docs.astral.sh/uv/) in this environment, then restart the agent.",
            )
        )

    windows = [tool for tool in tools if tool.is_windows]
    if windows:
        listed = ", ".join(
            f"{tool.name} ({redact(tool.real or tool.found or '', home)})" for tool in windows
        )
        names = ", ".join(tool.name for tool in windows)
        findings.append(
            Finding(
                Severity.FAIL,
                f"Windows programs on PATH would run Serena: {listed}",
                f"Put the Linux {names} ahead of the Windows entries on PATH (or set "
                "appendWindowsPath=false under [interop] in /etc/wsl.conf), then restart the agent.",
            )
        )

    missing_node = [name for name in _NODE_TOOLCHAIN if by_name[name].found is None]
    if missing_node:
        findings.append(
            Finding(
                Severity.WARN,
                f"{' and '.join(missing_node)} not found on PATH: Serena cannot install "
                "the TypeScript/JavaScript language servers",
                "Install Node.js in this environment if the project has TypeScript or JavaScript.",
            )
        )

    if not findings:
        findings.append(Finding(Severity.OK, "uvx, node and npm resolve to native binaries"))
    return findings


def _guard_end_line(lines: Sequence[str]) -> int | None:
    """Index of the last line of the non-interactive ``return`` guard, if any."""
    for index, line in enumerate(lines):
        if _ONE_LINE_GUARD.match(line):
            return index
        if _CASE_GUARD_START.match(line):
            for end in range(index + 1, min(index + 8, len(lines))):
                if _CASE_GUARD_END.match(lines[end]):
                    block = "\n".join(lines[index : end + 1])
                    return end if re.search(r"\breturn\b", block) else None
            return None
    return None


def check_bashrc_guard(bashrc_text: str | None) -> Finding | None:
    """Flag node toolchains that ``~/.bashrc`` sets up only for interactive shells."""
    if bashrc_text is None:
        return None
    lines = bashrc_text.splitlines()
    guard_end = _guard_end_line(lines)
    if guard_end is None:
        return None
    for offset, line in enumerate(lines[guard_end + 1 :], start=guard_end + 2):
        if line.lstrip().startswith("#"):
            continue
        for tool, pattern in _TOOLCHAIN_SETUP:
            if pattern.search(line):
                return Finding(
                    Severity.WARN,
                    f"~/.bashrc sets up {tool} on line {offset}, below its "
                    f"non-interactive guard (line {guard_end + 1})",
                    "A shell that is not interactive (an IDE, `wsl.exe <command>`, many MCP "
                    f"launchers) stops at that guard, so an agent it starts never gets {tool}'s "
                    "node and npm; on WSL the Windows npm on PATH wins instead. Move the "
                    f"{tool} setup above the guard, or start the agent from an interactive "
                    "terminal (`headroom wrap claude`).",
                )
    return None


def check_language_server_launchers(serena_dir: Path, home: Path) -> Finding | None:
    """Flag cached npm launchers Serena cannot execute (broken link or no +x)."""
    static = serena_dir / "language_servers" / "static"
    if not static.is_dir():
        return None
    broken: list[str] = []
    not_executable: list[str] = []
    # Serena 1.x and 2.x both cache npm-installed servers at
    # static/<LanguageServerClass>/<install dir>/node_modules/.bin/.
    for launcher in sorted(static.glob("*/*/node_modules/.bin/*")):
        if not launcher.exists():
            broken.append(redact(str(launcher), home))
        elif not os.access(launcher, os.X_OK):
            not_executable.append(redact(str(launcher), home))
    if not (broken or not_executable):
        return None
    hints = []
    if not_executable:
        hints.append("chmod +x " + " ".join(not_executable))
    if broken:
        hints.append(
            "Broken links mean an interrupted install; delete "
            + ", ".join(sorted({str(Path(p).parents[2]) for p in broken}))
            + " and Serena reinstalls it on the next start."
        )
    return Finding(
        Severity.FAIL,
        f"{len(broken) + len(not_executable)} cached language-server launcher(s) "
        "cannot be executed",
        " ".join(hints),
    )


def latest_mcp_log(serena_dir: Path) -> Path | None:
    logs = [path for path in (serena_dir / "logs").glob("*/mcp_*.txt") if path.is_file()]
    return max(logs, key=lambda path: path.stat().st_mtime, default=None)


def check_mcp_log(log: Path | None, home: Path) -> Finding | None:
    """Surface the error lines of Serena's newest MCP server log."""
    if log is None:
        return None
    with log.open("rb") as handle:
        handle.seek(max(0, log.stat().st_size - _LOG_TAIL_BYTES))
        tail = handle.read().decode("utf-8", errors="replace")
    errors = [line for line in tail.splitlines() if _LOG_LEVEL_ERROR.match(line)]
    if not errors:
        return Finding(Severity.OK, f"no errors in Serena's latest log ({redact(str(log), home)})")
    shown = [redact(line, home)[:_MAX_LINE_CHARS] for line in errors[-_MAX_LOG_LINES:]]
    return Finding(
        Severity.WARN,
        f"Serena's latest log has {len(errors)} error line(s) ({redact(str(log), home)})",
        "\n".join(shown),
    )


def health_check_command(spec: ServerSpec, project: Path) -> list[str]:
    """Serena's ``project health-check``, launched exactly like the MCP server.

    Reuses the registered spec's launcher prefix (``uvx --from serena-agent
    serena``) so the probe runs the same Serena the agent would.
    """
    args = list(spec.args)
    launcher = args[: args.index("start-mcp-server")] if "start-mcp-server" in args else args
    return [spec.command, *launcher, "project", "health-check", str(project)]


def _default_runner(argv: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
    """Run the probe in its own process group and kill the whole tree on exit.

    ``uvx`` runs Serena as a grandchild, and Serena starts language servers
    under it; ``subprocess.run``'s timeout kills only ``uvx`` and would orphan
    the rest. Ctrl+C no longer reaches a separate process group, so an
    interrupt tears the tree down here too.
    """
    group: dict[str, Any] = {}
    if sys.platform == "win32":
        group["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        group["start_new_session"] = True
    # Shared wrapper: decodes Serena's emoji verdict as UTF-8 on Windows consoles too.
    proc = popen_text(  # noqa: S603 - argv is built from Headroom's own Serena spec
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        **group,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except BaseException:  # TimeoutExpired, KeyboardInterrupt, anything: never leak the tree
        kill_process_tree(proc)
        raise
    return subprocess.CompletedProcess(list(argv), proc.returncode, stdout, stderr)


def run_health_check(
    spec: ServerSpec,
    project: Path,
    home: Path,
    *,
    timeout: float = DEFAULT_PROBE_TIMEOUT_S,
    run: Runner | None = None,
) -> Finding:
    """Prove the language server answers by making real symbol calls through it."""
    argv = health_check_command(spec, project)
    try:
        completed = (run or _default_runner)(argv, timeout)
    except FileNotFoundError:
        return Finding(Severity.FAIL, f"health check not run: {spec.command} not found on PATH")
    except subprocess.TimeoutExpired:
        return Finding(
            Severity.FAIL,
            f"health check did not finish within {timeout:g}s",
            "The language server is hanging while it installs or starts; it is not a slow "
            "request. Fix the findings above, then retry.",
        )
    output = redact((completed.stdout or "") + (completed.stderr or ""), home)
    relevant = [
        line.strip()
        for line in output.splitlines()
        if _PROBE_VERDICT.search(line) or line.startswith("Log saved to")
    ]
    detail = "\n".join(relevant[-_MAX_PROBE_LINES:]) or None
    if completed.returncode == 0:
        return Finding(
            Severity.OK, "health check passed: Serena answered real symbol requests", detail
        )
    if detail is None:
        tail = [line for line in output.splitlines() if line.strip()][-_MAX_PROBE_LINES:]
        detail = "\n".join(line[:_MAX_LINE_CHARS] for line in tail) or None
    return Finding(Severity.FAIL, f"health check failed (exit {completed.returncode})", detail)


RECOVERY_STEPS = (
    "A Serena server whose language server failed to start keeps that state until it "
    "restarts. Once the findings above are fixed, restart only Serena and keep the "
    "conversation: in Claude Code run /mcp, select serena and choose Reconnect. Other "
    "agents: restart the agent. A registered or connected server is not proof of health; "
    "`headroom mcp doctor --probe` is."
)
