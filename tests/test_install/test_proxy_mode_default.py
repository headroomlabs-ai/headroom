"""`headroom install` must leave mode ownership with the proxy runtime."""

from __future__ import annotations

from headroom.install.models import DeploymentManifest


def _mode_option_default(command) -> object:
    """The declared default of a command's ``--mode`` option."""
    for param in command.params:
        if param.name == "proxy_mode":
            return param.default
    raise AssertionError(f"{command.name} has no --mode/proxy_mode option")


def test_install_apply_omits_mode_by_default() -> None:
    from headroom.cli.install import install_apply

    assert _mode_option_default(install_apply) is None


def test_deploy_omits_mode_by_default() -> None:
    from headroom.cli.install import deploy

    assert _mode_option_default(deploy) is None


def test_manifest_default_leaves_mode_to_runtime() -> None:
    assert DeploymentManifest.__dataclass_fields__["proxy_mode"].default is None


def test_install_entry_points_leave_mode_to_runtime() -> None:
    """An install must not pin a mode through its CLI or supervisor env."""
    from headroom.cli.install import deploy, install_apply

    assert _mode_option_default(install_apply) is None
    assert _mode_option_default(deploy) is None


def test_token_mode_is_still_reachable() -> None:
    """Changing the default must not take the choice away.

    The option carries no restrictive ``type``, and the normalizer still accepts
    token (plus its aliases), so `--mode token` remains available to anyone who
    wants maximum compression and accepts the prefix-cache busts.
    """
    from headroom.cli.install import deploy, install_apply
    from headroom.proxy.proxy_mode_policy import (
        PROXY_MODE_TOKEN,
        normalize_proxy_mode_value,
    )

    for command in (install_apply, deploy):
        param = next(p for p in command.params if p.name == "proxy_mode")
        assert param.type.name == "text", f"{command.name} --mode became restrictive"

    assert normalize_proxy_mode_value("token") == PROXY_MODE_TOKEN
    assert normalize_proxy_mode_value("token_headroom") == PROXY_MODE_TOKEN
