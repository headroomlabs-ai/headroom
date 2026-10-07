"""Installs on Python 3.14 must not need a compiler.

litellm below 1.93 declared requires-python <3.14 (GH #956), and 1.92 to 1.96.0
shipped wheels for only some platforms, so pip fell back to a Rust source build.
1.96.2 is the first release with prebuilt wheels for Python 3.10 to 3.14 on
Linux, macOS and Windows, so litellm is required everywhere from that floor.

watchdog 6.0.0 publishes no macOS wheel for Python 3.14. It only powers the
optional code graph watcher, so it is skipped there and the proxy must start
without it. litellm is still lazily imported, so the core paths keep degrading
gracefully when it is missing from an environment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from packaging.requirements import Requirement
from packaging.version import Version

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
SUPPORTED_PYTHONS = ("3.10", "3.11", "3.12", "3.13", "3.14")
PLATFORMS = ("linux", "darwin", "win32")
LITELLM_FLOOR = Version("1.96.2")


def _requirements(name: str) -> list[Requirement]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    specs = list(data["project"].get("dependencies", []))
    for extra in data["project"].get("optional-dependencies", {}).values():
        specs.extend(extra)
    return [r for spec in specs if (r := Requirement(spec)).name == name]


def _applies(requirement: Requirement, python: str, platform: str) -> bool:
    if requirement.marker is None:
        return True
    return requirement.marker.evaluate(
        {"python_version": python, "sys_platform": platform, "extra": ""}
    )


def test_litellm_is_required_on_every_supported_python() -> None:
    reqs = _requirements("litellm")
    assert reqs, "expected litellm to be declared in pyproject"
    for r in reqs:
        for python in SUPPORTED_PYTHONS:
            for platform in PLATFORMS:
                assert _applies(r, python, platform), f"{r}: skipped on {python}/{platform}"


def test_litellm_floor_has_wheels_for_python_314() -> None:
    for r in _requirements("litellm"):
        assert not r.specifier.contains("1.96.1"), f"{r}: allows litellm without 3.14 wheels"
        assert r.specifier.contains(str(LITELLM_FLOOR)), f"{r}: excludes {LITELLM_FLOOR}"


def test_watchdog_is_skipped_only_where_it_has_no_wheel() -> None:
    reqs = _requirements("watchdog")
    assert reqs, "expected watchdog to be declared in pyproject"
    for r in reqs:
        assert not _applies(r, "3.14", "darwin"), f"{r}: needs a compiler on macOS 3.14"
        assert _applies(r, "3.13", "darwin"), f"{r}: must still install on macOS 3.13"
        assert _applies(r, "3.14", "linux"), f"{r}: must still install on Linux 3.14"
        assert _applies(r, "3.14", "win32"), f"{r}: must still install on Windows 3.14"


@pytest.mark.proxy_dependency_gate
def test_proxy_starts_without_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    from headroom.cli import proxy

    requested: list[str] = []

    def fake_import(name: str) -> object:
        requested.append(name)
        if name == "watchdog":
            raise ImportError("No module named 'watchdog'")
        return object()

    monkeypatch.setattr(proxy, "import_module", fake_import)
    proxy.ensure_proxy_dependencies()
    assert "watchdog" not in requested


def test_code_graph_watcher_skips_itself_without_watchdog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    from headroom.graph.watcher import CodeGraphWatcher

    real_import = builtins.__import__

    def no_watchdog(name: str, *args: object, **kwargs: object) -> object:
        if name == "watchdog" or name.startswith("watchdog."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_watchdog)
    watcher = CodeGraphWatcher(Path.cwd(), cbm_binary="codebase-memory-mcp")
    assert watcher.start() is False


def test_proxy_cost_degrades_without_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    # With litellm absent from the environment, the proxy cost path must return
    # None rather than raise.
    from headroom.proxy import cost

    monkeypatch.setattr(cost, "LITELLM_AVAILABLE", False)
    monkeypatch.setattr(cost, "litellm", None)
    assert cost._get_litellm_module() is None
