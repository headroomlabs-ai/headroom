"""Regression tests for the packaged ``headroom`` console-script target."""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path

import click
import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

import headroom.cli
from headroom.cli.main import main

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TARGET = "headroom.cli.main:main"
LEGACY_TARGET = "headroom.cli:main"


def _declared_target() -> str:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["project"]["scripts"]["headroom"]


def _entry_point(value: str) -> EntryPoint:
    return EntryPoint(name="headroom", value=value, group="console_scripts")


def test_pyproject_targets_defining_module_not_package_reexport() -> None:
    ep = _entry_point(_declared_target())

    assert ep.value == EXPECTED_TARGET
    assert ep.module == main.callback.__module__
    assert ep.attr == main.name


def test_declared_target_loads_root_click_command() -> None:
    assert _entry_point(_declared_target()).load() is main


def test_declared_target_survives_submodule_shadowing_package_attr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Re-importing ``headroom.cli.main`` rebinds the package attribute ``main``
    # to the submodule, so the legacy package-level target stops being callable.
    monkeypatch.setattr(headroom.cli, "main", sys.modules["headroom.cli.main"])

    assert not isinstance(_entry_point(LEGACY_TARGET).load(), click.Command)
    assert _entry_point(_declared_target()).load() is main


def test_declared_target_runs_in_fresh_interpreter() -> None:
    script = (
        "import sys\n"
        "from importlib.metadata import EntryPoint\n"
        f"ep = EntryPoint(name='headroom', value={_declared_target()!r}, "
        "group='console_scripts')\n"
        "sys.argv = ['headroom', '--version']\n"
        "sys.exit(ep.load()())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "headroom" in result.stdout.lower()


def test_installed_distribution_metadata_matches_pyproject() -> None:
    installed = [
        ep
        for ep in entry_points(group="console_scripts", name="headroom")
        if ep.dist is not None and ep.dist.name == "headroom-ai"
    ]
    if not installed:
        pytest.skip("headroom-ai distribution metadata is not installed")

    assert installed[0].value == _declared_target(), (
        "Installed console-script metadata is stale; reinstall headroom-ai"
    )
