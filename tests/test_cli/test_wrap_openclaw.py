"""Tests for `headroom wrap openclaw` command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_cli
from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def plugin_dir(tmp_path: Path) -> Path:
    """Create a minimal OpenClaw plugin directory fixture."""
    plugin = tmp_path / "plugins" / "openclaw"
    plugin.mkdir(parents=True)
    (plugin / "package.json").write_text('{"name":"headroom-openclaw"}\n')
    (plugin / "openclaw.plugin.json").write_text('{"id":"headroom"}\n')
    hook_shim = plugin / "hook-shim"
    hook_shim.mkdir()
    (hook_shim / "index.js").write_text("export default {};\n")
    return plugin


def _make_successful_run(calls: list[dict]) -> object:
    def run(cmd, **kwargs):  # noqa: ANN001
        calls.append({"cmd": list(cmd), **kwargs})
        return MagicMock(returncode=0, stdout="", stderr="")

    return run


def test_wrap_openclaw_default_installs_from_npm_and_restarts(runner: CliRunner) -> None:
    calls: list[dict] = []

    def which(name: str) -> str | None:
        mapping = {
            "openclaw": "openclaw",
            "npm": "npm",
        }
        return mapping.get(name)

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=_make_successful_run(calls)):
            result = runner.invoke(main, ["wrap", "openclaw"])

    assert result.exit_code == 0, result.output

    cmds = [c["cmd"] for c in calls]
    assert [
        "openclaw",
        "plugins",
        "install",
        "--dangerously-force-unsafe-install",
        "headroom-ai/openclaw",
    ] in cmds
    assert ["openclaw", "config", "validate"] in cmds
    assert ["openclaw", "gateway", "restart"] in cmds
    assert ["openclaw", "plugins", "inspect", "headroom"] in cmds

    config_set_index = next(
        i
        for i, cmd in enumerate(cmds)
        if cmd[:4] == ["openclaw", "config", "set", "plugins.entries.headroom"]
    )
    install_index = next(
        i
        for i, cmd in enumerate(cmds)
        if cmd[:4] == ["openclaw", "plugins", "install", "--dangerously-force-unsafe-install"]
    )
    assert config_set_index < install_index

    # Verify plugin install in npm mode does not set cwd
    install_call = next(
        c
        for c in calls
        if c["cmd"][:4] == ["openclaw", "plugins", "install", "--dangerously-force-unsafe-install"]
    )
    assert install_call["cwd"] is None

    # No local build in npm mode
    assert ["npm", "install"] not in cmds
    assert ["npm", "run", "build"] not in cmds

    # Verify config payload includes enabled + expected defaults
    set_entry = next(
        c
        for c in calls
        if c["cmd"][:4] == ["openclaw", "config", "set", "plugins.entries.headroom"]
    )
    payload = json.loads(set_entry["cmd"][4])
    assert payload["enabled"] is True
    assert payload["config"]["proxyPort"] == 8787
    assert payload["config"]["autoStart"] is True
    assert payload["config"]["startupTimeoutMs"] == 20000
    assert payload["config"]["gatewayProviderIds"] == ["openai-codex"]
    assert payload["config"]["pythonPath"] == wrap_cli.sys.executable


def test_wrap_openclaw_skip_build_and_no_restart(runner: CliRunner, plugin_dir: Path) -> None:
    calls: list[dict] = []

    def which(name: str) -> str | None:
        mapping = {
            "openclaw": "openclaw",
            "npm": "npm",
        }
        return mapping.get(name)

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=_make_successful_run(calls)):
            result = runner.invoke(
                main,
                [
                    "wrap",
                    "openclaw",
                    "--plugin-path",
                    str(plugin_dir),
                    "--skip-build",
                    "--no-restart",
                ],
            )

    assert result.exit_code == 0, result.output
    cmds = [c["cmd"] for c in calls]
    assert ["npm", "install"] not in cmds
    assert ["npm", "run", "build"] not in cmds
    assert ["openclaw", "gateway", "restart"] not in cmds


def test_wrap_openclaw_local_source_mode_builds_and_links(
    runner: CliRunner, plugin_dir: Path
) -> None:
    calls: list[dict] = []

    def which(name: str) -> str | None:
        mapping = {
            "openclaw": "openclaw",
            "npm": "npm",
        }
        return mapping.get(name)

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=_make_successful_run(calls)):
            result = runner.invoke(
                main,
                ["wrap", "openclaw", "--plugin-path", str(plugin_dir)],
            )

    assert result.exit_code == 0, result.output
    cmds = [c["cmd"] for c in calls]
    assert ["npm", "install"] in cmds
    assert ["npm", "run", "build"] in cmds
    assert [
        "openclaw",
        "plugins",
        "install",
        "--dangerously-force-unsafe-install",
        "--link",
        ".",
    ] in cmds


def test_wrap_openclaw_fails_when_openclaw_missing(runner: CliRunner, plugin_dir: Path) -> None:
    def which(name: str) -> str | None:
        return None if name == "openclaw" else "npm"

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        result = runner.invoke(main, ["wrap", "openclaw", "--plugin-path", str(plugin_dir)])

    assert result.exit_code != 0
    assert "'openclaw' not found in PATH" in result.output


def test_wrap_openclaw_fails_when_plugin_path_invalid(runner: CliRunner, tmp_path: Path) -> None:
    invalid = tmp_path / "missing-plugin"

    with patch("headroom.cli.wrap.shutil.which", return_value="openclaw"):
        result = runner.invoke(main, ["wrap", "openclaw", "--plugin-path", str(invalid)])

    assert result.exit_code != 0
    assert "Plugin path not found" in result.output


def test_wrap_openclaw_uses_extension_fallback_on_linked_install_bug(
    runner: CliRunner, plugin_dir: Path
) -> None:
    calls: list[dict] = []

    def which(name: str) -> str | None:
        mapping = {
            "openclaw": "openclaw",
            "npm": "npm",
        }
        return mapping.get(name)

    def run(cmd, **kwargs):  # noqa: ANN001
        calls.append({"cmd": list(cmd), **kwargs})
        if cmd[:3] == ["openclaw", "plugins", "install"]:
            return MagicMock(
                returncode=1,
                stdout="Also not a valid hook pack",
                stderr='Plugin installation blocked despite "--dangerously-force-unsafe-install"',
            )
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=run):
            with patch(
                "headroom.cli.wrap._copy_openclaw_plugin_into_extensions",
                return_value=Path("C:/Users/test/.openclaw/extensions/headroom"),
            ) as copy_fallback:
                result = runner.invoke(main, ["wrap", "openclaw", "--plugin-path", str(plugin_dir)])

    assert result.exit_code == 0, result.output
    copy_fallback.assert_called_once()


def test_wrap_openclaw_continues_when_plugin_already_exists(
    runner: CliRunner,
) -> None:
    calls: list[dict] = []

    def which(name: str) -> str | None:
        mapping = {
            "openclaw": "openclaw",
            "npm": "npm",
        }
        return mapping.get(name)

    def run(cmd, **kwargs):  # noqa: ANN001
        calls.append({"cmd": list(cmd), **kwargs})
        if cmd[:3] == ["openclaw", "plugins", "install"]:
            return MagicMock(
                returncode=1,
                stdout="plugin already exists: C:\\Users\\test\\.openclaw\\extensions\\headroom",
                stderr="",
            )
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=run):
            result = runner.invoke(main, ["wrap", "openclaw", "--no-restart"])

    assert result.exit_code == 0, result.output
    cmds = [c["cmd"] for c in calls]
    assert ["openclaw", "config", "validate"] in cmds
    assert ["openclaw", "plugins", "inspect", "headroom"] in cmds


def test_wrap_openclaw_verbose_prints_install_restart_and_inspect_output(
    runner: CliRunner,
) -> None:
    def which(name: str) -> str | None:
        mapping = {
            "openclaw": "openclaw",
            "npm": "npm",
        }
        return mapping.get(name)

    def run(cmd, **kwargs):  # noqa: ANN001
        if cmd[:3] == ["openclaw", "plugins", "install"]:
            return MagicMock(returncode=0, stdout="install-ok", stderr="")
        if cmd[:3] == ["openclaw", "gateway", "restart"]:
            return MagicMock(returncode=0, stdout="restart-ok", stderr="")
        if cmd[:3] == ["openclaw", "plugins", "inspect"]:
            return MagicMock(returncode=0, stdout="inspect-ok", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=run):
            result = runner.invoke(main, ["wrap", "openclaw", "--verbose"])

    assert result.exit_code == 0, result.output
    assert "install-ok" in result.output
    assert "restart-ok" in result.output
    assert "inspect-ok" in result.output


def test_wrap_openclaw_starts_gateway_when_restart_fails(runner: CliRunner) -> None:
    calls: list[dict] = []

    def which(name: str) -> str | None:
        return {"openclaw": "openclaw", "npm": "npm"}.get(name)

    def run(cmd, **kwargs):  # noqa: ANN001
        calls.append({"cmd": list(cmd), **kwargs})
        if cmd[:3] == ["openclaw", "gateway", "restart"]:
            return MagicMock(returncode=1, stdout="", stderr="gateway not running")
        if cmd[:3] == ["openclaw", "gateway", "start"]:
            return MagicMock(returncode=0, stdout="started-ok", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=run):
            result = runner.invoke(main, ["wrap", "openclaw", "--verbose"])

    assert result.exit_code == 0, result.output
    cmds = [c["cmd"] for c in calls]
    assert ["openclaw", "gateway", "restart"] in cmds
    assert ["openclaw", "gateway", "start"] in cmds
    assert "Gateway started." in result.output
    assert "started-ok" in result.output


def test_wrap_openclaw_accepts_repeatable_gateway_provider_ids(runner: CliRunner) -> None:
    calls: list[dict] = []

    def which(name: str) -> str | None:
        return {"openclaw": "openclaw", "npm": "npm"}.get(name)

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=_make_successful_run(calls)):
            result = runner.invoke(
                main,
                [
                    "wrap",
                    "openclaw",
                    "--gateway-provider-id",
                    "openai-codex",
                    "--gateway-provider-id",
                    "anthropic",
                    "--no-restart",
                ],
            )

    assert result.exit_code == 0, result.output
    set_entry = next(
        c
        for c in calls
        if c["cmd"][:4] == ["openclaw", "config", "set", "plugins.entries.headroom"]
    )
    payload = json.loads(set_entry["cmd"][4])
    assert payload["config"]["gatewayProviderIds"] == ["openai-codex", "anthropic"]


def test_normalize_openclaw_gateway_provider_ids_dedupes_blanks_and_defaults() -> None:
    assert wrap_cli._normalize_openclaw_gateway_provider_ids(
        (" openai-codex ", "", "anthropic", "openai-codex", "  ")
    ) == ["openai-codex", "anthropic"]
    assert wrap_cli._normalize_openclaw_gateway_provider_ids(None) == ["openai-codex"]


def test_read_openclaw_config_value_handles_missing_and_raw_strings() -> None:
    missing = MagicMock(returncode=1, stdout="", stderr="missing")
    raw_string = MagicMock(returncode=0, stdout="plain-text-value\n", stderr="")

    with patch("headroom.cli.wrap.subprocess.run", side_effect=[missing, raw_string]):
        assert wrap_cli._read_openclaw_config_value("openclaw", "plugins.entries.headroom") is None
        assert (
            wrap_cli._read_openclaw_config_value(
                "openclaw", "plugins.entries.headroom.config.pythonPath"
            )
            == "plain-text-value"
        )


def test_build_openclaw_plugin_entry_sets_and_clears_python_path() -> None:
    with_python = wrap_cli._build_openclaw_plugin_entry(
        existing_entry={"config": {"customFlag": True}},
        proxy_port=8787,
        startup_timeout_ms=20000,
        python_path="C:\\Python312\\python.exe",
        no_auto_start=False,
        gateway_provider_ids=("openai-codex",),
        enabled=True,
    )
    assert with_python["config"]["pythonPath"] == "C:\\Python312\\python.exe"

    without_python = wrap_cli._build_openclaw_plugin_entry(
        existing_entry={"config": {"pythonPath": "C:\\Old\\python.exe", "customFlag": True}},
        proxy_port=8787,
        startup_timeout_ms=20000,
        python_path=None,
        no_auto_start=False,
        gateway_provider_ids=("openai-codex",),
        enabled=True,
    )
    assert "pythonPath" not in without_python["config"]
    assert without_python["config"]["customFlag"] is True


def test_build_openclaw_unwrap_entry_preserves_top_level_metadata() -> None:
    entry = wrap_cli._build_openclaw_unwrap_entry(
        {
            "source": "headroom-ai/openclaw",
            "enabled": True,
            "config": {
                "pythonPath": "C:\\Python312\\python.exe",
                "proxyPort": 8787,
                "customFlag": True,
            },
        }
    )

    assert entry["source"] == "headroom-ai/openclaw"
    assert entry["enabled"] is False
    assert entry["config"] == {"customFlag": True}


def test_unwrap_openclaw_stops_proxy_by_default(runner: CliRunner) -> None:
    calls: list[dict] = []

    def which(name: str) -> str | None:
        return "openclaw" if name == "openclaw" else None

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=_make_successful_run(calls)):
            with patch(
                "headroom.cli.wrap._stop_local_proxy_for_unwrap",
                return_value="stopped",
            ) as stop_proxy:
                result = runner.invoke(
                    main,
                    ["unwrap", "openclaw", "--proxy-port", "9999", "--no-restart"],
                )

    assert result.exit_code == 0, result.output
    stop_proxy.assert_called_once_with(9999)
    assert "Stopped local Headroom proxy on port 9999" in result.output


def test_wrap_openclaw_no_auto_start_does_not_default_python_path(
    runner: CliRunner, plugin_dir: Path
) -> None:
    calls: list[dict] = []

    def which(name: str) -> str | None:
        return {"openclaw": "openclaw", "npm": "npm"}.get(name)

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=_make_successful_run(calls)):
            result = runner.invoke(
                main,
                [
                    "wrap",
                    "openclaw",
                    "--plugin-path",
                    str(plugin_dir),
                    "--skip-build",
                    "--no-auto-start",
                    "--no-restart",
                ],
            )

    assert result.exit_code == 0, result.output
    set_entry = next(
        c
        for c in calls
        if c["cmd"][:4] == ["openclaw", "config", "set", "plugins.entries.headroom"]
    )
    payload = json.loads(set_entry["cmd"][4])
    assert payload["config"]["autoStart"] is False
    assert "pythonPath" not in payload["config"]


def test_wrap_openclaw_fails_for_npm_mode_hook_pack_bug_without_local_fallback(
    runner: CliRunner,
) -> None:
    def which(name: str) -> str | None:
        mapping = {
            "openclaw": "openclaw",
            "npm": "npm",
        }
        return mapping.get(name)

    def run(cmd, **kwargs):  # noqa: ANN001
        if cmd[:3] == ["openclaw", "plugins", "install"]:
            return MagicMock(
                returncode=1,
                stdout="Also not a valid hook pack",
                stderr='Blocked despite "--dangerously-force-unsafe-install"',
            )
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=run):
            result = runner.invoke(main, ["wrap", "openclaw"])

    assert result.exit_code != 0
    assert "openclaw plugins install failed" in result.output


def test_wrap_openclaw_copy_mode_uses_path_install(runner: CliRunner, plugin_dir: Path) -> None:
    calls: list[dict] = []

    def which(name: str) -> str | None:
        mapping = {
            "openclaw": "openclaw",
            "npm": "npm",
        }
        return mapping.get(name)

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=_make_successful_run(calls)):
            result = runner.invoke(
                main,
                [
                    "wrap",
                    "openclaw",
                    "--plugin-path",
                    str(plugin_dir),
                    "--copy",
                    "--skip-build",
                    "--no-restart",
                ],
            )

    assert result.exit_code == 0, result.output
    cmds = [c["cmd"] for c in calls]
    assert [
        "openclaw",
        "plugins",
        "install",
        "--dangerously-force-unsafe-install",
        str(plugin_dir),
    ] in cmds


def test_wrap_openclaw_fails_when_npm_missing_for_local_build(
    runner: CliRunner, plugin_dir: Path
) -> None:
    def which(name: str) -> str | None:
        mapping = {
            "openclaw": "openclaw",
            "npm": None,
        }
        return mapping.get(name)

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        result = runner.invoke(main, ["wrap", "openclaw", "--plugin-path", str(plugin_dir)])

    assert result.exit_code != 0
    assert "'npm' not found in PATH" in result.output


def test_wrap_openclaw_fails_when_local_path_missing_manifest_files(
    runner: CliRunner, tmp_path: Path
) -> None:
    plugin = tmp_path / "plugins" / "openclaw"
    plugin.mkdir(parents=True)

    with patch("headroom.cli.wrap.shutil.which", return_value="openclaw"):
        result = runner.invoke(main, ["wrap", "openclaw", "--plugin-path", str(plugin)])
    assert result.exit_code != 0
    assert "missing package.json" in result.output

    (plugin / "package.json").write_text("{}\n")
    with patch("headroom.cli.wrap.shutil.which", return_value="openclaw"):
        result = runner.invoke(main, ["wrap", "openclaw", "--plugin-path", str(plugin)])
    assert result.exit_code != 0
    assert "missing openclaw.plugin.json" in result.output


def test_run_checked_raises_click_exception_on_command_errors() -> None:
    with patch("headroom.cli.wrap.subprocess.run", side_effect=FileNotFoundError()):
        with pytest.raises(Exception, match="command not found"):
            wrap_cli._run_checked(["missing"], action="demo")

    cpe_stderr = wrap_cli.subprocess.CalledProcessError(
        returncode=2,
        cmd=["x"],
        stderr="bad-stderr",
    )
    with patch("headroom.cli.wrap.subprocess.run", side_effect=cpe_stderr):
        with pytest.raises(Exception, match="bad-stderr"):
            wrap_cli._run_checked(["x"], action="demo")

    cpe_stdout = wrap_cli.subprocess.CalledProcessError(
        returncode=3,
        cmd=["x"],
        output="bad-stdout",
        stderr="",
    )
    with patch("headroom.cli.wrap.subprocess.run", side_effect=cpe_stdout):
        with pytest.raises(Exception, match="bad-stdout"):
            wrap_cli._run_checked(["x"], action="demo")


def test_resolve_openclaw_extensions_dir_empty_output_raises() -> None:
    with patch(
        "headroom.cli.wrap._run_checked",
        return_value=MagicMock(stdout="   \n", stderr="", returncode=0),
    ):
        with pytest.raises(Exception, match="Unable to resolve OpenClaw config path"):
            wrap_cli._resolve_openclaw_extensions_dir("openclaw")


def test_copy_openclaw_plugin_into_extensions_handles_missing_and_existing_dist(
    tmp_path: Path,
) -> None:
    plugin = tmp_path / "plugin"
    plugin.mkdir()

    with pytest.raises(Exception, match="Plugin dist folder missing"):
        wrap_cli._copy_openclaw_plugin_into_extensions(plugin_dir=plugin, openclaw_bin="openclaw")

    dist = plugin / "dist"
    dist.mkdir()
    (dist / "index.js").write_text("x\n")
    (plugin / "package.json").write_text("{}\n")
    (plugin / "openclaw.plugin.json").write_text("{}\n")

    with patch("headroom.cli.wrap._resolve_openclaw_extensions_dir", return_value=tmp_path):
        with pytest.raises(Exception, match="Plugin hook-shim folder missing"):
            wrap_cli._copy_openclaw_plugin_into_extensions(
                plugin_dir=plugin, openclaw_bin="openclaw"
            )

    hook_shim = plugin / "hook-shim"
    hook_shim.mkdir()
    (hook_shim / "index.js").write_text("shim\n")

    ext_root = tmp_path / ".openclaw" / "extensions"
    target_headroom = ext_root / "headroom"
    target_dist = target_headroom / "dist"
    target_hook_shim = target_headroom / "hook-shim"
    target_dist.mkdir(parents=True)
    (target_dist / "old.js").write_text("old\n")
    target_hook_shim.mkdir(parents=True)
    (target_hook_shim / "old.js").write_text("old-shim\n")

    with patch("headroom.cli.wrap._resolve_openclaw_extensions_dir", return_value=ext_root):
        out = wrap_cli._copy_openclaw_plugin_into_extensions(
            plugin_dir=plugin, openclaw_bin="openclaw"
        )

    assert out == target_headroom
    assert (target_dist / "index.js").exists()
    assert not (target_dist / "old.js").exists()
    assert (target_hook_shim / "index.js").exists()
    assert not (target_hook_shim / "old.js").exists()


def test_unwrap_openclaw_disables_plugin_and_restores_legacy_slot(runner: CliRunner) -> None:
    calls: list[dict] = []

    def which(name: str) -> str | None:
        return {"openclaw": "openclaw"}.get(name)

    def run(cmd, **kwargs):  # noqa: ANN001
        calls.append({"cmd": list(cmd), **kwargs})
        if cmd[:4] == ["openclaw", "config", "get", "plugins.entries.headroom"]:
            return MagicMock(
                returncode=0,
                stdout=json.dumps(
                    {
                        "enabled": True,
                        "config": {
                            "proxyPort": 8787,
                            "gatewayProviderIds": ["openai-codex"],
                            "customFlag": True,
                        },
                    }
                ),
                stderr="",
            )
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=run):
            result = runner.invoke(main, ["unwrap", "openclaw"])

    assert result.exit_code == 0, result.output
    set_entry = next(
        c
        for c in calls
        if c["cmd"][:4] == ["openclaw", "config", "set", "plugins.entries.headroom"]
    )
    payload = json.loads(set_entry["cmd"][4])
    assert payload == {"enabled": False, "config": {"customFlag": True}}

    set_slot = next(
        c
        for c in calls
        if c["cmd"][:4] == ["openclaw", "config", "set", "plugins.slots.contextEngine"]
    )
    assert json.loads(set_slot["cmd"][4]) == "legacy"
    assert ["openclaw", "gateway", "restart"] in [c["cmd"] for c in calls]


def test_unwrap_openclaw_no_restart_skips_gateway_restart(runner: CliRunner) -> None:
    calls: list[dict] = []

    def which(name: str) -> str | None:
        return {"openclaw": "openclaw"}.get(name)

    def run(cmd, **kwargs):  # noqa: ANN001
        calls.append({"cmd": list(cmd), **kwargs})
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=run):
            result = runner.invoke(main, ["unwrap", "openclaw", "--no-restart"])

    assert result.exit_code == 0, result.output
    assert ["openclaw", "gateway", "restart"] not in [c["cmd"] for c in calls]


def test_unwrap_openclaw_fails_when_openclaw_missing(runner: CliRunner) -> None:
    with patch("headroom.cli.wrap.shutil.which", return_value=None):
        result = runner.invoke(main, ["unwrap", "openclaw"])

    assert result.exit_code != 0
    assert "'openclaw' not found in PATH" in result.output


def test_unwrap_openclaw_verbose_prints_gateway_and_inspect_output(runner: CliRunner) -> None:
    def which(name: str) -> str | None:
        return {"openclaw": "openclaw"}.get(name)

    def run(cmd, **kwargs):  # noqa: ANN001
        if cmd[:4] == ["openclaw", "config", "get", "plugins.entries.headroom"]:
            return MagicMock(
                returncode=0,
                stdout=json.dumps({"enabled": True, "config": {"proxyPort": 8787}}),
                stderr="",
            )
        if cmd[:3] == ["openclaw", "gateway", "restart"]:
            return MagicMock(returncode=0, stdout="gateway-restarted", stderr="")
        if cmd[:3] == ["openclaw", "plugins", "inspect"]:
            return MagicMock(returncode=0, stdout="inspect-disabled", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("headroom.cli.wrap.shutil.which", side_effect=which):
        with patch("headroom.cli.wrap.subprocess.run", side_effect=run):
            result = runner.invoke(main, ["unwrap", "openclaw", "--verbose"])

    assert result.exit_code == 0, result.output
    assert "gateway-restarted" in result.output
    assert "inspect-disabled" in result.output
