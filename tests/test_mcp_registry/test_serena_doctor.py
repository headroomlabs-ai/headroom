"""Serena MCP diagnostics for the WSL toolchain timeout in #3914."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from headroom.cli.main import main
from headroom.mcp_registry import build_serena_spec
from headroom.mcp_registry import serena_doctor as sd

HOME = Path("/home/me")
UBUNTU_GUARD = """\
# ~/.bashrc: executed by bash(1) for non-login shells.
case $- in
    *i*) ;;
      *) return;;
esac
HISTCONTROL=ignoreboth
"""
FNM_SETUP = 'eval "$(fnm env --use-on-cd --shell bash)"\n'


def _which(mapping: dict[str, str]):
    return lambda name, path=None: mapping.get(name)


def _resolve(mapping: dict[str, str], real: dict[str, str] | None = None) -> list[sd.ToolPath]:
    real = real or {}
    return sd.resolve_toolchain("ignored", which=_which(mapping), realpath=lambda p: real.get(p, p))


LINUX_TOOLS = {
    "uvx": "/home/me/.local/bin/uvx",
    "node": "/home/me/.local/share/fnm/aliases/default/bin/node",
    "npm": "/home/me/.local/share/fnm/aliases/default/bin/npm",
    "npx": "/home/me/.local/share/fnm/aliases/default/bin/npx",
}


# --- redaction -------------------------------------------------------------


def test_redact_hides_home_and_windows_user_name():
    text = "spawn /mnt/c/Users/Pouya/AppData/Roaming/npm/npm from /home/me/project"
    assert (
        sd.redact(text, HOME) == "spawn /mnt/c/Users/<user>/AppData/Roaming/npm/npm from ~/project"
    )


def test_redact_leaves_other_users_home_prefix_alone():
    assert sd.redact("/home/meg/x", HOME) == "/home/meg/x"


# --- toolchain resolution --------------------------------------------------


def test_native_toolchain_reports_ok():
    findings = sd.check_toolchain(_resolve(LINUX_TOOLS), HOME)
    assert [f.severity for f in findings] == [sd.Severity.OK]


def test_windows_npm_shadowing_linux_node_fails_with_redacted_paths():
    # Arrange: the #3914 setup — Linux node from fnm, npm/npx from Windows NVM.
    tools = {
        **LINUX_TOOLS,
        "npm": "/mnt/c/nvm4w/nodejs/npm",
        "npx": "/mnt/c/Users/Pouya/AppData/Roaming/npm/npx",
    }

    # Act
    findings = sd.check_toolchain(_resolve(tools), HOME)

    # Assert
    [finding] = findings
    assert finding.severity is sd.Severity.FAIL
    assert "npm (/mnt/c/nvm4w/nodejs/npm)" in finding.summary
    assert "npx (/mnt/c/Users/<user>/AppData/Roaming/npm/npx)" in finding.summary
    assert "node (" not in finding.summary
    assert "Linux npm, npx ahead of the Windows entries" in (finding.hint or "")


def test_linux_symlink_into_windows_install_is_caught_by_its_real_path():
    tools = _resolve(LINUX_TOOLS, real={LINUX_TOOLS["npm"]: "/mnt/c/nvm4w/nodejs/npm"})
    findings = sd.check_toolchain(tools, HOME)
    assert findings[0].severity is sd.Severity.FAIL
    assert "npm (/mnt/c/nvm4w/nodejs/npm)" in findings[0].summary


def test_missing_uvx_fails_because_serena_cannot_launch():
    tools = {k: v for k, v in LINUX_TOOLS.items() if k != "uvx"}
    [finding] = sd.check_toolchain(_resolve(tools), HOME)
    assert finding.severity is sd.Severity.FAIL
    assert "cannot launch Serena" in finding.summary


def test_missing_node_and_npm_warns_about_typescript_language_server():
    [finding] = sd.check_toolchain(_resolve({"uvx": LINUX_TOOLS["uvx"]}), HOME)
    assert finding.severity is sd.Severity.WARN
    assert finding.summary.startswith("node and npm not found")


# --- ~/.bashrc non-interactive guard ---------------------------------------


@pytest.mark.parametrize(
    "guard",
    [
        UBUNTU_GUARD,
        '[ -z "$PS1" ] && return\n',
        "[[ $- != *i* ]] && return\n",
    ],
)
def test_toolchain_below_non_interactive_guard_warns(guard: str):
    finding = sd.check_bashrc_guard(guard + FNM_SETUP)
    assert finding is not None
    assert finding.severity is sd.Severity.WARN
    assert "sets up fnm on line" in finding.summary
    assert "Move the fnm setup above the guard" in (finding.hint or "")


def test_finding_names_the_lines_involved():
    finding = sd.check_bashrc_guard(UBUNTU_GUARD + FNM_SETUP)
    assert finding is not None
    assert finding.summary == (
        "~/.bashrc sets up fnm on line 7, below its non-interactive guard (line 5)"
    )


def test_toolchain_above_guard_is_fine():
    assert sd.check_bashrc_guard(FNM_SETUP + UBUNTU_GUARD) is None


def test_commented_out_toolchain_below_guard_is_ignored():
    assert sd.check_bashrc_guard(UBUNTU_GUARD + "# " + FNM_SETUP) is None


def test_bashrc_without_guard_or_file_is_fine():
    assert sd.check_bashrc_guard(FNM_SETUP) is None
    assert sd.check_bashrc_guard(None) is None


@pytest.mark.parametrize(
    ("line", "tool"),
    [
        ('export NVM_DIR="$HOME/.nvm"\n', "nvm"),
        ('. "$HOME/.asdf/asdf.sh"\n', "asdf"),
        ('eval "$(mise activate bash)"\n', "mise"),
        ('export VOLTA_HOME="$HOME/.volta"\n', "volta"),
    ],
)
def test_each_known_node_toolchain_is_recognized(line: str, tool: str):
    finding = sd.check_bashrc_guard(UBUNTU_GUARD + line)
    assert finding is not None and f"sets up {tool}" in finding.summary


# --- cached language-server launchers --------------------------------------


def _launcher(serena_dir: Path, name: str = "typescript-language-server") -> Path:
    path = (
        serena_dir
        / "language_servers"
        / "static"
        / "TypeScriptLanguageServer"
        / "ts-lsp"
        / "node_modules"
        / ".bin"
        / name
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env node\n")
    return path


@pytest.mark.skipif(os.name == "nt", reason="POSIX execute bit")
def test_non_executable_launcher_fails_with_chmod_fix(tmp_path: Path):
    launcher = _launcher(tmp_path / ".serena")
    launcher.chmod(0o644)

    finding = sd.check_language_server_launchers(tmp_path / ".serena", tmp_path)

    assert finding is not None and finding.severity is sd.Severity.FAIL
    assert finding.hint == (
        "chmod +x ~/.serena/language_servers/static/TypeScriptLanguageServer/ts-lsp/node_modules/.bin/typescript-language-server"
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX execute bit")
def test_executable_launcher_passes(tmp_path: Path):
    _launcher(tmp_path / ".serena").chmod(0o755)
    assert sd.check_language_server_launchers(tmp_path / ".serena", tmp_path) is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks")
def test_broken_launcher_link_points_at_the_install_to_delete(tmp_path: Path):
    bin_dir = (
        tmp_path
        / ".serena"
        / "language_servers"
        / "static"
        / "TypeScriptLanguageServer"
        / "ts-lsp"
        / "node_modules"
        / ".bin"
    )
    bin_dir.mkdir(parents=True)
    (bin_dir / "tsserver").symlink_to(bin_dir / "missing")

    finding = sd.check_language_server_launchers(tmp_path / ".serena", tmp_path)

    assert finding is not None
    assert "delete ~/.serena/language_servers/static/TypeScriptLanguageServer/ts-lsp" in (
        finding.hint or ""
    )


def test_no_language_server_cache_is_fine(tmp_path: Path):
    assert sd.check_language_server_launchers(tmp_path / ".serena", tmp_path) is None


def test_serena_home_honors_env_override():
    assert sd.serena_home({"SERENA_HOME": " /data/serena "}, HOME) == Path("/data/serena")
    assert sd.serena_home({}, HOME) == HOME / ".serena"


# --- Serena MCP log --------------------------------------------------------


def _write_log(serena_dir: Path, day: str, name: str, body: str, mtime: float) -> Path:
    log = serena_dir / "logs" / day / name
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(body)
    os.utime(log, (mtime, mtime))
    return log


def test_latest_mcp_log_picks_newest_mcp_server_log(tmp_path: Path):
    _write_log(tmp_path, "2026-11-01", "mcp_old_1.txt", "", 1_000)
    newest = _write_log(tmp_path, "2026-11-02", "mcp_new_2.txt", "", 3_000)
    _write_log(tmp_path, "2026-11-02", "dashboard-viewer_x_3.txt", "", 9_000)
    assert sd.latest_mcp_log(tmp_path) == newest


def test_log_error_lines_are_surfaced_redacted_and_capped(tmp_path: Path):
    body = "INFO  2026 starting\n" + "".join(
        f"ERROR 2026 [main] ls:start:1 - npm failed {i} in {tmp_path}/proj\n" for i in range(8)
    )
    log = _write_log(tmp_path / ".serena", "2026-11-02", "mcp_a_1.txt", body, 1_000)

    finding = sd.check_mcp_log(log, tmp_path)

    assert finding is not None and finding.severity is sd.Severity.WARN
    assert "8 error line(s)" in finding.summary
    shown = (finding.hint or "").splitlines()
    assert len(shown) == 5
    assert shown[-1].endswith("npm failed 7 in ~/proj")
    assert str(tmp_path) not in (finding.hint or "") + finding.summary


def test_clean_log_reports_ok(tmp_path: Path):
    log = _write_log(tmp_path, "2026-11-02", "mcp_a_1.txt", "INFO all good\n", 1_000)
    finding = sd.check_mcp_log(log, tmp_path)
    assert finding is not None and finding.severity is sd.Severity.OK


def test_no_log_yields_no_finding():
    assert sd.check_mcp_log(None, HOME) is None


# --- health-check probe ----------------------------------------------------


def test_probe_launches_the_same_serena_as_the_registered_spec():
    spec = build_serena_spec("claude-code")
    argv = sd.health_check_command(spec, Path("/repo"))
    assert argv == [
        "uvx",
        "--from",
        "serena-agent",
        "serena",
        "project",
        "health-check",
        "/repo",
    ]


def _completed(returncode: int, stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def test_probe_pass_proves_health_by_output():
    stdout = "noise\nLog saved to: /home/me/repo/.serena/logs/health-checks/h.log\n✅ Health check passed - All tools working correctly\n"
    finding = sd.run_health_check(
        build_serena_spec("claude-code"),
        Path("/repo"),
        HOME,
        run=lambda argv, t: _completed(0, stdout),
    )
    assert finding.severity is sd.Severity.OK
    assert finding.hint == (
        "Log saved to: ~/repo/.serena/logs/health-checks/h.log\n"
        "✅ Health check passed - All tools working correctly"
    )


def test_probe_failure_relays_serena_reason():
    stdout = "❌ Health check failed: npm is not installed or isn't in PATH.\n"
    finding = sd.run_health_check(
        build_serena_spec("claude-code"),
        Path("/repo"),
        HOME,
        run=lambda argv, t: _completed(1, stdout),
    )
    assert finding.severity is sd.Severity.FAIL
    assert finding.summary == "health check failed (exit 1)"
    assert "npm is not installed" in (finding.hint or "")


def test_probe_failure_without_verdict_shows_output_tail():
    finding = sd.run_health_check(
        build_serena_spec("claude-code"),
        Path("/repo"),
        HOME,
        run=lambda argv, t: _completed(2, "Traceback\nRuntimeError: boom\n"),
    )
    assert finding.hint == "Traceback\nRuntimeError: boom"


def test_probe_timeout_is_reported_as_startup_hang_not_slow_request():
    def hang(argv, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    finding = sd.run_health_check(
        build_serena_spec("claude-code"), Path("/repo"), HOME, timeout=42, run=hang
    )
    assert finding.severity is sd.Severity.FAIL
    assert finding.summary == "health check did not finish within 42s"
    assert "not a slow request" in (finding.hint or "")


def test_probe_without_uvx_fails_cleanly():
    def missing(argv, timeout):
        raise FileNotFoundError(argv[0])

    finding = sd.run_health_check(
        build_serena_spec("claude-code"), Path("/repo"), HOME, run=missing
    )
    assert finding.severity is sd.Severity.FAIL
    assert "uvx not found" in finding.summary


# --- CLI -------------------------------------------------------------------


def _cli_env(monkeypatch, home: Path, tools: dict[str, str]) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("SERENA_HOME", raising=False)
    monkeypatch.setattr(sd.shutil, "which", _which(tools))


def test_cli_exits_2_and_names_windows_npm(monkeypatch, tmp_path: Path):
    _cli_env(monkeypatch, tmp_path, {**LINUX_TOOLS, "npm": "/mnt/c/nvm4w/nodejs/npm"})

    result = CliRunner().invoke(main, ["mcp", "doctor"])

    assert result.exit_code == 2, result.output
    assert "npm   /mnt/c/nvm4w/nodejs/npm" in result.output
    assert "✗ Windows programs on PATH" in result.output
    assert "/mcp, select serena and choose Reconnect" in result.output


def test_cli_exits_1_on_bashrc_guard_warning(monkeypatch, tmp_path: Path):
    _cli_env(monkeypatch, tmp_path, LINUX_TOOLS)
    (tmp_path / ".bashrc").write_text(UBUNTU_GUARD + FNM_SETUP)

    result = CliRunner().invoke(main, ["mcp", "doctor"])

    assert result.exit_code == 1, result.output
    assert "! ~/.bashrc sets up fnm on line 7" in result.output


def test_cli_exits_0_when_healthy_and_does_not_probe_unasked(monkeypatch, tmp_path: Path):
    _cli_env(monkeypatch, tmp_path, LINUX_TOOLS)
    monkeypatch.setattr(sd, "_default_runner", lambda *a: pytest.fail("probe ran unasked"))

    result = CliRunner().invoke(main, ["mcp", "doctor"])

    assert result.exit_code == 0, result.output
    assert "✓ uvx, node and npm resolve to native binaries" in result.output


def test_cli_probe_runs_health_check_on_project(monkeypatch, tmp_path: Path):
    _cli_env(monkeypatch, tmp_path, LINUX_TOOLS)
    calls: list[tuple[list[str], float]] = []

    def fake_run(argv, timeout):
        calls.append((list(argv), timeout))
        return _completed(0, "✅ Health check passed - All tools working correctly\n")

    monkeypatch.setattr(sd, "_default_runner", fake_run)

    result = CliRunner().invoke(
        main, ["mcp", "doctor", "--probe", "--project", str(tmp_path), "--timeout", "7"]
    )

    assert result.exit_code == 0, result.output
    assert calls == [(sd.health_check_command(build_serena_spec("claude-code"), tmp_path), 7)]
    assert "✓ health check passed" in result.output
