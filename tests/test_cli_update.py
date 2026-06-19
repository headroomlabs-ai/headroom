"""Tests for the `headroom update` command + install-method detection."""

from __future__ import annotations

import sys

import pytest
from click.testing import CliRunner

from headroom.cli import update as up
from headroom.cli.main import main


@pytest.fixture(autouse=True)
def _clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.delenv("PIPX_HOME", raising=False)
    monkeypatch.delenv("UV_TOOL_DIR", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    # Default: not a checkout / editable / docker / managed install.
    monkeypatch.setattr(up, "_is_source_checkout", lambda: False)
    monkeypatch.setattr(up, "_is_editable_install", lambda: False)
    monkeypatch.setattr(up, "_in_docker", lambda: False)
    monkeypatch.setattr(up, "_is_externally_managed", lambda: False)


# --------------------------------------------------------------------------- #
# detect_install_method
# --------------------------------------------------------------------------- #
def test_detect_checkout(monkeypatch):
    monkeypatch.setattr(up, "_is_source_checkout", lambda: True)
    m = up.detect_install_method()
    assert m.kind == "checkout" and m.can_self_update is False and "git pull" in m.guidance


def test_detect_editable(monkeypatch):
    monkeypatch.setattr(up, "_is_editable_install", lambda: True)
    m = up.detect_install_method()
    assert m.kind == "editable" and m.can_self_update is False


def test_detect_docker(monkeypatch):
    monkeypatch.setattr(up, "_in_docker", lambda: True)
    m = up.detect_install_method()
    assert m.kind == "docker" and m.can_self_update is False


def test_detect_pipx_by_path(monkeypatch):
    monkeypatch.setattr(up.sys, "prefix", "/home/u/.local/pipx/venvs/headroom-ai")
    m = up.detect_install_method()
    assert m.kind == "pipx" and m.argv == ["pipx", "upgrade", "headroom-ai"]


def test_detect_pipx_windows_path(monkeypatch):
    monkeypatch.setattr(up.sys, "prefix", r"C:\\Users\\u\\pipx\\venvs\\headroom-ai")
    m = up.detect_install_method()
    assert m.kind == "pipx"


def test_detect_uv_tool(monkeypatch):
    monkeypatch.setattr(up.sys, "prefix", "/home/u/.local/share/uv/tools/headroom-ai")
    m = up.detect_install_method()
    assert m.kind == "uv-tool" and m.argv == ["uv", "tool", "upgrade", "headroom-ai"]


def test_detect_venv_uses_current_interpreter(monkeypatch):
    monkeypatch.setattr(up, "_in_virtualenv", lambda: True)
    m = up.detect_install_method()
    assert m.kind == "pip"
    assert m.argv[:4] == [sys.executable, "-m", "pip", "install"]
    assert "-U" in m.argv and "headroom-ai" in m.argv


def test_detect_venv_with_extras(monkeypatch):
    monkeypatch.setattr(up, "_in_virtualenv", lambda: True)
    m = up.detect_install_method(extras="all")
    assert "headroom-ai[all]" in m.argv


def test_detect_user_site(monkeypatch):
    monkeypatch.setattr(up, "_in_virtualenv", lambda: False)
    monkeypatch.setattr(up, "_package_location", lambda: "/home/u/.local/site")
    monkeypatch.setattr(up, "_is_user_site_install", lambda loc: True)
    m = up.detect_install_method()
    assert m.kind == "pip-user" and "--user" in m.argv


def test_detect_externally_managed_refuses(monkeypatch):
    monkeypatch.setattr(up, "_in_virtualenv", lambda: False)
    monkeypatch.setattr(up, "_is_user_site_install", lambda loc: False)
    monkeypatch.setattr(up, "_is_externally_managed", lambda: True)
    m = up.detect_install_method()
    assert m.kind == "system" and m.can_self_update is False
    assert "PEP 668" in m.guidance


# --------------------------------------------------------------------------- #
# `headroom update` command
# --------------------------------------------------------------------------- #
def test_update_already_current(monkeypatch):
    monkeypatch.setattr(up, "installed_version", lambda: "0.26.0")
    monkeypatch.setattr(up, "fetch_latest_version", lambda **k: "0.26.0")
    res = CliRunner().invoke(main, ["update"])
    assert res.exit_code == 0
    assert "up to date" in res.output


def test_update_check_reports_command_without_running(monkeypatch):
    monkeypatch.setattr(up, "installed_version", lambda: "0.26.0")
    monkeypatch.setattr(up, "fetch_latest_version", lambda **k: "0.27.0")
    monkeypatch.setattr(up, "_in_virtualenv", lambda: True)

    def _no_run(*a, **k):
        raise AssertionError("subprocess.run must not be called with --check")

    monkeypatch.setattr(up.subprocess, "run", _no_run)
    res = CliRunner().invoke(main, ["update", "--check"])
    assert res.exit_code == 0
    assert "Update available: 0.26.0 → 0.27.0" in res.output
    assert "pip" in res.output and "install" in res.output


def test_update_runs_upgrade_with_yes(monkeypatch):
    calls = {}

    monkeypatch.setattr(up, "installed_version", lambda: "0.26.0")
    monkeypatch.setattr(up, "fetch_latest_version", lambda **k: "0.27.0")
    monkeypatch.setattr(up, "_in_virtualenv", lambda: True)

    class _Result:
        returncode = 0

    def _run(argv, *a, **k):
        calls["argv"] = argv
        return _Result()

    monkeypatch.setattr(up.subprocess, "run", _run)
    res = CliRunner().invoke(main, ["update", "--yes"])
    assert res.exit_code == 0
    assert calls["argv"][:4] == [sys.executable, "-m", "pip", "install"]
    assert "upgraded to 0.27.0" in res.output


def test_update_refuses_in_checkout(monkeypatch):
    monkeypatch.setattr(up, "installed_version", lambda: "0.26.0")
    monkeypatch.setattr(up, "fetch_latest_version", lambda **k: "0.27.0")
    monkeypatch.setattr(up, "_is_source_checkout", lambda: True)

    def _no_run(*a, **k):
        raise AssertionError("must not upgrade a checkout")

    monkeypatch.setattr(up.subprocess, "run", _no_run)
    res = CliRunner().invoke(main, ["update", "--yes"])
    assert res.exit_code == 0
    assert "git pull" in res.output


def test_update_network_failure(monkeypatch):
    monkeypatch.setattr(up, "installed_version", lambda: "0.26.0")
    monkeypatch.setattr(up, "fetch_latest_version", lambda **k: None)
    res = CliRunner().invoke(main, ["update"])
    assert res.exit_code != 0
    assert "Could not reach PyPI" in res.output


def test_update_upgrade_failure_surfaces_command(monkeypatch):
    monkeypatch.setattr(up, "installed_version", lambda: "0.26.0")
    monkeypatch.setattr(up, "fetch_latest_version", lambda **k: "0.27.0")
    monkeypatch.setattr(up, "_in_virtualenv", lambda: True)

    class _Result:
        returncode = 1

    monkeypatch.setattr(up.subprocess, "run", lambda *a, **k: _Result())
    res = CliRunner().invoke(main, ["update", "--yes"])
    assert res.exit_code != 0
    assert "Upgrade failed" in res.output
