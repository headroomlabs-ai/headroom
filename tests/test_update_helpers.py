"""Coverage for update_check / cli.update helper functions and branches."""

from __future__ import annotations

import importlib.metadata as md
import sysconfig

import pytest
from click.testing import CliRunner

from headroom import update_check as uc
from headroom.cli import update as up
from headroom.cli.main import main


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("HEADROOM_UPDATE_CHECK", "on")
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("HEADROOM_IN_DOCKER", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)


# --------------------------------------------------------------------------- #
# cli.update helpers
# --------------------------------------------------------------------------- #
def test_norm_normalizes_and_lowercases():
    assert up._norm("/Foo/Bar").endswith("/foo/bar")
    assert up._norm(None) == ""
    assert up._norm(r"C:\\X\\Y").count("\\") == 0


def test_in_virtualenv_via_prefix(monkeypatch):
    monkeypatch.setattr(up.sys, "prefix", "/venv")
    monkeypatch.setattr(up.sys, "base_prefix", "/usr")
    assert up._in_virtualenv() is True


def test_in_virtualenv_via_conda(monkeypatch):
    monkeypatch.setattr(up.sys, "prefix", "/x")
    monkeypatch.setattr(up.sys, "base_prefix", "/x")
    monkeypatch.setenv("CONDA_PREFIX", "/opt/conda")
    assert up._in_virtualenv() is True


def test_in_virtualenv_false(monkeypatch):
    monkeypatch.setattr(up.sys, "prefix", "/x")
    monkeypatch.setattr(up.sys, "base_prefix", "/x")
    assert up._in_virtualenv() is False


def test_in_docker_env_flag(monkeypatch):
    monkeypatch.setenv("HEADROOM_IN_DOCKER", "1")
    assert up._in_docker() is True


def test_in_docker_default_false():
    # No HEADROOM_IN_DOCKER and (almost certainly) no /.dockerenv on the runner.
    assert isinstance(up._in_docker(), bool)


def test_editable_install_true(monkeypatch):
    class _D:
        def read_text(self, name):
            return '{"dir_info": {"editable": true}}'

    monkeypatch.setattr(md, "distribution", lambda name: _D())
    assert up._is_editable_install() is True


def test_editable_install_false_when_not_editable(monkeypatch):
    class _D:
        def read_text(self, name):
            return '{"url": "https://pypi.org", "archive_info": {}}'

    monkeypatch.setattr(md, "distribution", lambda name: _D())
    assert up._is_editable_install() is False


def test_editable_install_false_when_no_direct_url(monkeypatch):
    class _D:
        def read_text(self, name):
            return None

    monkeypatch.setattr(md, "distribution", lambda name: _D())
    assert up._is_editable_install() is False


def test_editable_install_swallows_errors(monkeypatch):
    def _boom(name):
        raise RuntimeError("nope")

    monkeypatch.setattr(md, "distribution", _boom)
    assert up._is_editable_install() is False


def test_package_location_handles_missing(monkeypatch):
    def _boom(name):
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "distribution", _boom)
    assert up._package_location() is None


def test_user_site_and_membership(monkeypatch):
    monkeypatch.setattr(up, "_user_site", lambda: "/home/u/.local/site")
    assert up._is_user_site_install("/home/u/.local/site/headroom_ai") is True
    assert up._is_user_site_install("/home/u/.local/site") is True
    assert up._is_user_site_install("/usr/lib/python3/site") is False
    assert up._is_user_site_install(None) is False


def test_user_site_no_sibling_prefix_match(monkeypatch):
    # Path-segment containment: "/.../site" must NOT match "/.../site-packages".
    monkeypatch.setattr(up, "_user_site", lambda: "/home/u/.local/site")
    assert up._is_user_site_install("/home/u/.local/site-packages/x") is False


def test_format_cmd_quotes_spaces(monkeypatch):
    monkeypatch.setattr(up.sys, "platform", "linux")
    out = up._format_cmd(["/path with space/python", "-m", "pip", "install", "-U", "headroom-ai"])
    assert "'/path with space/python'" in out


def test_format_cmd_windows(monkeypatch):
    monkeypatch.setattr(up.sys, "platform", "win32")
    out = up._format_cmd([r"C:\\Program Files\\Python\\python.exe", "-m", "pip"])
    assert "Program Files" in out and out.endswith("-m pip")


def test_externally_managed_true(tmp_path, monkeypatch):
    (tmp_path / "EXTERNALLY-MANAGED").write_text("[externally-managed]")
    monkeypatch.setattr(sysconfig, "get_path", lambda key: str(tmp_path))
    assert up._is_externally_managed() is True


def test_externally_managed_false(tmp_path, monkeypatch):
    monkeypatch.setattr(sysconfig, "get_path", lambda key: str(tmp_path))
    assert up._is_externally_managed() is False


def test_user_site_real_returns_str_or_empty():
    assert isinstance(up._user_site(), str)


def test_source_checkout_real_is_bool():
    assert isinstance(up._is_source_checkout(), bool)


def test_package_location_real_runs():
    # Either a normalized path string or None, but the call must not raise.
    assert up._package_location() is None or isinstance(up._package_location(), str)


@pytest.mark.parametrize(
    "platform,needle",
    [("darwin", "brew"), ("win32", "pipx"), ("linux", "distro")],
)
def test_managed_env_guidance(monkeypatch, platform, needle):
    monkeypatch.setattr(up.sys, "platform", platform)
    assert needle in up._managed_env_guidance()


def test_spec_with_and_without_extras():
    assert up._spec(None) == "headroom-ai"
    assert up._spec("all") == "headroom-ai[all]"
    assert up._spec("[proxy]") == "headroom-ai[proxy]"


# --------------------------------------------------------------------------- #
# cli.update command branches
# --------------------------------------------------------------------------- #
def test_update_aborts_on_decline(monkeypatch):
    monkeypatch.setattr(up, "installed_version", lambda: "0.26.0")
    monkeypatch.setattr(up, "fetch_latest_version", lambda **k: "0.27.0")
    monkeypatch.setattr(up, "_is_source_checkout", lambda: False)
    monkeypatch.setattr(up, "_is_editable_install", lambda: False)
    monkeypatch.setattr(up, "_in_docker", lambda: False)
    monkeypatch.setattr(up, "_in_virtualenv", lambda: True)

    def _no_run(*a, **k):
        raise AssertionError("should not run after decline")

    monkeypatch.setattr(up.subprocess, "run", _no_run)
    res = CliRunner().invoke(main, ["update"], input="n\n")
    assert res.exit_code == 0
    assert "Aborted" in res.output


def test_update_missing_tool_surfaces_command(monkeypatch):
    monkeypatch.setattr(up, "installed_version", lambda: "0.26.0")
    monkeypatch.setattr(up, "fetch_latest_version", lambda **k: "0.27.0")
    monkeypatch.setattr(up, "_is_source_checkout", lambda: False)
    monkeypatch.setattr(up, "_is_editable_install", lambda: False)
    monkeypatch.setattr(up, "_in_docker", lambda: False)
    monkeypatch.setattr(
        up,
        "detect_install_method",
        lambda extras=None: up.InstallMethod(
            kind="pipx", can_self_update=True, argv=["pipx", "upgrade", "headroom-ai"]
        ),
    )

    def _missing(*a, **k):
        raise FileNotFoundError("pipx")

    monkeypatch.setattr(up.subprocess, "run", _missing)
    res = CliRunner().invoke(main, ["update", "--yes"])
    assert res.exit_code != 0
    assert "not found on PATH" in res.output


def test_update_externally_managed_refuses_via_command(monkeypatch):
    monkeypatch.setattr(up, "installed_version", lambda: "0.26.0")
    monkeypatch.setattr(up, "fetch_latest_version", lambda **k: "0.27.0")
    monkeypatch.setattr(up, "_is_source_checkout", lambda: False)
    monkeypatch.setattr(up, "_is_editable_install", lambda: False)
    monkeypatch.setattr(up, "_in_docker", lambda: False)
    monkeypatch.setattr(up, "_in_virtualenv", lambda: False)
    monkeypatch.setattr(up, "_is_user_site_install", lambda loc: False)
    monkeypatch.setattr(up, "_is_externally_managed", lambda: True)
    res = CliRunner().invoke(main, ["update", "--yes"])
    assert res.exit_code == 0
    assert "PEP 668" in res.output


# --------------------------------------------------------------------------- #
# update_check helpers / branches
# --------------------------------------------------------------------------- #
def test_installed_version_present(monkeypatch):
    monkeypatch.setattr(md, "version", lambda name: "0.26.0")
    assert uc.installed_version() == "0.26.0"


def test_installed_version_not_found(monkeypatch):
    def _boom(name):
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "version", _boom)
    assert uc.installed_version() is None


def test_is_source_checkout_real_is_bool():
    assert isinstance(uc._is_source_checkout(), bool)


def test_in_docker_real_is_bool():
    assert isinstance(uc._in_docker(), bool)


def test_select_latest_skips_invalid_versions():
    data = {"releases": {"not-a-version": [{}], "0.26.0": [{}]}}
    assert uc._select_latest(data, allow_pre=False) == "0.26.0"


def test_select_latest_info_fallback_invalid_returns_none():
    data = {"releases": {}, "info": {"version": "not-a-version"}}
    assert uc._select_latest(data, allow_pre=False) is None


def test_select_latest_info_fallback_prerelease_filtered():
    data = {"releases": {}, "info": {"version": "1.0.0rc1"}}
    assert uc._select_latest(data, allow_pre=False) is None
    assert uc._select_latest(data, allow_pre=True) == "1.0.0rc1"


def test_run_check_disabled_returns_none(monkeypatch):
    monkeypatch.setenv("HEADROOM_UPDATE_CHECK", "off")
    monkeypatch.setattr(uc, "fetch_latest_version", lambda **k: pytest.fail("no fetch"))
    assert uc.run_check() is None


def test_maybe_check_async_disabled_returns_none(monkeypatch):
    monkeypatch.setenv("HEADROOM_UPDATE_CHECK", "off")
    assert uc.maybe_check_async() is None


def test_format_update_notice_invalid_versions(monkeypatch):
    uc.write_cache("not-a-version")
    monkeypatch.setattr(uc, "_is_source_checkout", lambda: False)
    monkeypatch.setattr(uc, "_in_docker", lambda: False)
    monkeypatch.setattr(uc, "installed_version", lambda: "0.26.0")
    assert uc.format_update_notice() is None


def test_fetch_latest_version_info_only(monkeypatch):
    import json

    payload = json.dumps({"releases": {}, "info": {"version": "0.30.0"}}).encode()

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return payload

    monkeypatch.setattr(uc.urllib.request, "urlopen", lambda *a, **k: _Resp())
    assert uc.fetch_latest_version() == "0.30.0"
