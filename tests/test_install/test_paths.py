from __future__ import annotations

from pathlib import Path

import click
import pytest

from headroom.install import paths as install_paths


def test_validate_profile_name_accepts_and_rejects_values() -> None:
    assert install_paths.validate_profile_name("good.profile-1_2") == "good.profile-1_2"

    for value in (".", "..", "bad/name", "bad space", ""):
        with pytest.raises(click.ClickException, match="Invalid profile name"):
            install_paths.validate_profile_name(value)


def test_profile_and_artifact_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("headroom.install.paths._paths.deploy_root", lambda: tmp_path / "deploy")

    assert install_paths.deploy_root() == tmp_path / "deploy"
    assert install_paths.profile_root("demo") == tmp_path / "deploy" / "demo"
    assert install_paths.manifest_path("demo") == tmp_path / "deploy" / "demo" / "manifest.json"
    assert install_paths.log_path("demo") == tmp_path / "deploy" / "demo" / "runner.log"
    assert install_paths.pid_path("demo") == tmp_path / "deploy" / "demo" / "runner.pid"
    assert (
        install_paths.unix_run_script_path("demo")
        == tmp_path / "deploy" / "demo" / "run-headroom.sh"
    )
    assert install_paths.unix_ensure_script_path("demo") == (
        tmp_path / "deploy" / "demo" / "ensure-headroom.sh"
    )
    assert install_paths.windows_run_script_path("demo") == (
        tmp_path / "deploy" / "demo" / "run-headroom.ps1"
    )
    assert install_paths.windows_run_cmd_path("demo") == (
        tmp_path / "deploy" / "demo" / "run-headroom.cmd"
    )
    assert install_paths.windows_ensure_script_path("demo") == (
        tmp_path / "deploy" / "demo" / "ensure-headroom.ps1"
    )
    assert install_paths.windows_ensure_cmd_path("demo") == (
        tmp_path / "deploy" / "demo" / "ensure-headroom.cmd"
    )


def test_env_target_and_config_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("headroom.install.paths.sys.platform", "linux")

    assert install_paths.unix_user_env_targets() == [
        tmp_path / ".bashrc",
        tmp_path / ".zshrc",
        tmp_path / ".profile",
    ]
    assert install_paths.unix_system_env_targets() == [Path("/etc/profile.d/headroom.sh")]

    monkeypatch.setattr("headroom.install.paths.sys.platform", "darwin")
    assert install_paths.unix_system_env_targets() == [
        Path("/etc/profile"),
        Path("/etc/zprofile"),
        Path("/etc/bashrc"),
    ]

    assert install_paths.claude_settings_path() == tmp_path / ".claude" / "settings.json"
    assert install_paths.codebuddy_settings_path() == tmp_path / ".codebuddy" / "settings.json"
    assert install_paths.codex_config_path() == tmp_path / ".codex" / "config.toml"
    assert install_paths.openclaw_config_path() == tmp_path / ".openclaw" / "openclaw.json"
    assert (
        install_paths.opencode_config_path() == tmp_path / ".config" / "opencode" / "opencode.json"
    )
