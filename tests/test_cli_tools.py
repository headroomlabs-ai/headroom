from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from headroom import binaries
from headroom.cli import tools as cli_tools
from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class FakeTable:
    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        self.columns: list[str] = []
        self.rows: list[tuple[object, ...]] = []

    def add_column(self, name: str) -> None:
        self.columns.append(name)

    def add_row(self, *values: object) -> None:
        self.rows.append(values)


class FakeConsole:
    instances: list[FakeConsole] = []

    def __init__(self) -> None:
        self.printed: list[object] = []
        FakeConsole.instances.append(self)

    def print(self, value: object) -> None:
        self.printed.append(value)


def install_fake_rich(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeConsole.instances.clear()
    monkeypatch.setitem(sys.modules, "rich.console", SimpleNamespace(Console=FakeConsole))
    monkeypatch.setitem(sys.modules, "rich.table", SimpleNamespace(Table=FakeTable))
    monkeypatch.setitem(
        sys.modules, "rich.markup", SimpleNamespace(escape=lambda value: f"escaped:{value}")
    )


def test_exec_tool_windows_and_posix_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_tools.binaries, "resolve", lambda tool: Path("C:\\bin\\sg.exe"))
    monkeypatch.setattr(cli_tools.sys, "platform", "win32")
    captured: dict[str, object] = {}

    def fake_run(cmd, check=False):  # noqa: ANN001
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(cli_tools.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as excinfo:
        cli_tools._exec_tool("ast-grep", ["--json"])
    assert excinfo.value.code == 7
    assert captured["cmd"] == ["C:\\bin\\sg.exe", "--json"]

    monkeypatch.setattr(cli_tools.sys, "platform", "linux")

    def fake_execv(path: str, cmd: list[str]) -> None:
        raise SystemExit((path, cmd))

    monkeypatch.setattr(cli_tools.os, "execv", fake_execv)
    with pytest.raises(SystemExit) as posix_exit:
        cli_tools._exec_tool("ast-grep", ["--help"])
    assert posix_exit.value.code == (str(Path("C:\\bin\\sg.exe")), ["C:\\bin\\sg.exe", "--help"])


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (binaries.PlatformNotSupported("unsupported"), "error: unsupported"),
        (binaries.OfflineError("offline"), "Hint: run `headroom tools install`"),
        (binaries.Sha256Mismatch("bad sha"), "error: bad sha"),
        (binaries.BinaryFetchError("fetch failed"), "error: fetch failed"),
    ],
)
def test_sg_command_reports_resolution_errors(
    monkeypatch: pytest.MonkeyPatch,
    runner: CliRunner,
    error: Exception,
    expected: str,
) -> None:
    monkeypatch.setattr(cli_tools.binaries, "resolve", lambda tool: (_ for _ in ()).throw(error))
    result = runner.invoke(main, ["sg", "--version"])
    assert result.exit_code == 2
    assert expected in result.output


def test_tools_list_renders_registry(monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    install_fake_rich(monkeypatch)
    monkeypatch.setattr(
        cli_tools.binaries,
        "detect_platform",
        lambda: SimpleNamespace(key=lambda: "windows-x86_64"),
    )
    monkeypatch.setattr(cli_tools.binaries, "cache_dir", lambda: Path("C:\\cache"))
    monkeypatch.setattr(
        cli_tools.binaries,
        "_registry",
        lambda: {
            "tools": {
                "ast-grep": {
                    "version": "1.2.3",
                    "source": "github",
                    "assets": {"windows-x86_64": {}, "linux-x86_64-gnu": {}},
                },
                "python-tool": {"version": "0.1.0", "source": "pypi", "assets": {}},
            }
        },
    )

    result = runner.invoke(main, ["tools", "list"])
    assert result.exit_code == 0
    console = FakeConsole.instances[-1]
    assert console.printed[0] == "[dim]platform:[/dim] windows-x86_64"
    assert console.printed[1] == "[dim]cache:[/dim] C:\\cache"
    table = console.printed[2]
    assert isinstance(table, FakeTable)
    assert ("ast-grep", "1.2.3", "github", "linux-x86_64-gnu, windows-x86_64") in table.rows
    assert ("python-tool", "0.1.0", "pypi", "(pypi)") in table.rows


def test_tools_doctor_json_and_table_modes(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner
) -> None:
    rows = [
        {
            "tool": "ast-grep",
            "state": "cached",
            "version": "1.0",
            "platform": "win",
            "path": "C:\\bin\\sg.exe",
        },
        {
            "tool": "difft",
            "state": "missing",
            "version": "2.0",
            "platform": "win",
            "path": None,
            "detail": "download needed <soon>",
        },
    ]
    monkeypatch.setattr(cli_tools.binaries, "status", lambda: rows)

    json_result = runner.invoke(main, ["tools", "doctor", "--json"])
    assert json_result.exit_code == 1
    assert '"tool": "ast-grep"' in json_result.output
    assert '"state": "missing"' in json_result.output

    install_fake_rich(monkeypatch)
    table_result = runner.invoke(main, ["tools", "doctor"])
    assert table_result.exit_code == 1
    console = FakeConsole.instances[-1]
    table = console.printed[0]
    assert isinstance(table, FakeTable)
    assert ("ast-grep", "[green]cached[/green]", "1.0", "win", "C:\\bin\\sg.exe") in table.rows
    assert ("difft", "[yellow]missing[/yellow]", "2.0", "win", "-") in table.rows
    assert console.printed[1] == "[dim]difft:[/dim] escaped:download needed <soon>"


def test_tools_install_covers_unknown_pypi_force_and_failures(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    cached_path = tmp_path / "cached-tool.exe"
    cached_path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        cli_tools.binaries,
        "_registry",
        lambda: {
            "tools": {
                "known": {"version": "1.0"},
                "pypi_tool": {"version": "2.0"},
                "broken": {"version": "3.0"},
            }
        },
    )
    monkeypatch.setattr(cli_tools.binaries, "_is_pypi_tool", lambda name: name == "pypi_tool")
    monkeypatch.setattr(
        cli_tools.binaries,
        "_path_lookup",
        lambda name: Path("C:\\Python\\Scripts\\pypi_tool.exe") if name == "pypi_tool" else None,
    )
    monkeypatch.setattr(
        cli_tools.binaries,
        "detect_platform",
        lambda: SimpleNamespace(key=lambda: "windows-x86_64"),
    )
    monkeypatch.setattr(cli_tools.binaries, "_cached_path", lambda name, version, plat: cached_path)
    monkeypatch.setattr(
        cli_tools.binaries,
        "resolve",
        lambda name: (
            (_ for _ in ()).throw(binaries.OfflineError("offline"))
            if name == "broken"
            else Path(f"C:\\cache\\{name}.exe")
        ),
    )

    result = runner.invoke(
        main,
        [
            "tools",
            "install",
            "--tool",
            "missing",
            "--tool",
            "pypi_tool",
            "--tool",
            "known",
            "--tool",
            "broken",
            "--force",
        ],
    )

    assert result.exit_code == 1
    assert "unknown tool 'missing'; skipping" in result.output
    assert "pypi_tool: on PATH at C:\\Python\\Scripts\\pypi_tool.exe (pypi wheel)" in result.output
    assert "known: installed" in result.output
    assert "broken: offline" in result.output
    assert not cached_path.exists()


def test_tools_install_reports_missing_pypi_and_cached_unlink_failure(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner
) -> None:
    problem_path = Path("C:\\cache\\locked.exe")
    monkeypatch.setattr(
        cli_tools.binaries,
        "_registry",
        lambda: {"tools": {"pypi_tool": {"version": "1.0"}, "known": {"version": "2.0"}}},
    )
    monkeypatch.setattr(cli_tools.binaries, "_is_pypi_tool", lambda name: name == "pypi_tool")
    monkeypatch.setattr(cli_tools.binaries, "_path_lookup", lambda name: None)
    monkeypatch.setattr(
        cli_tools.binaries,
        "detect_platform",
        lambda: SimpleNamespace(key=lambda: "windows-x86_64"),
    )
    monkeypatch.setattr(
        cli_tools.binaries, "_cached_path", lambda name, version, plat: problem_path
    )
    monkeypatch.setattr(Path, "exists", lambda self: self == problem_path)

    def fake_unlink(self) -> None:
        raise OSError("locked")

    monkeypatch.setattr(Path, "unlink", fake_unlink)
    monkeypatch.setattr(cli_tools.binaries, "resolve", lambda name: Path(f"C:\\cache\\{name}.exe"))

    result = runner.invoke(
        main,
        ["tools", "install", "--tool", "pypi_tool", "--tool", "known", "--force"],
    )

    assert result.exit_code == 1
    assert "pypi_tool: not on PATH" in result.output
    assert "known: failed to remove cached binary: locked" in result.output
