from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import Requirement

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
ALL_EXTRA = "all"
HEADROOM_PACKAGE_NAME = "headroom-ai"
MACOS_X86_64_TORCH_GUARD = "sys_platform != 'darwin' or platform_machine != 'x86_64'"
MACOS_X86_64_SYS_PLATFORM = "darwin"
MACOS_X86_64_PLATFORM_MACHINE = "x86_64"
PYPROJECT_FILE = "pyproject.toml"
SYS_PLATFORM_MARKER = "sys_platform"
PLATFORM_MACHINE_MARKER = "platform_machine"
TORCH_PACKAGE_NAME = "torch"
TORCH_TRANSITIVE_PACKAGE_NAMES = frozenset({"sentence-transformers"})
ORJSON_PACKAGE_NAME = "orjson"
PYTHON_SOCKS_PACKAGE_NAME = "python-socks"
PROXY_EXTRA = "proxy"
LEAN_EXTRA = "lean"
UV_LOCK_FILE = "uv.lock"
EXTRA_MARKER = re.compile(r"^extra == '([^']+)'$")


def _selected_dependency_names_for_extra(
    optional_deps: dict[str, list[str]],
    extra_name: str,
    environment: dict[str, str],
    visited: set[str] | None = None,
) -> set[str]:
    selected: set[str] = set()
    visited = visited or set()
    if extra_name in visited:
        return selected
    visited.add(extra_name)

    for dependency in optional_deps[extra_name]:
        requirement = Requirement(dependency)
        if requirement.marker is not None and not requirement.marker.evaluate(environment):
            continue
        if requirement.name == HEADROOM_PACKAGE_NAME:
            for nested_extra in requirement.extras:
                selected.update(
                    _selected_dependency_names_for_extra(
                        optional_deps,
                        nested_extra,
                        environment,
                        visited,
                    )
                )
        else:
            selected.add(requirement.name)

    return selected


def _locked_package(package_name: str) -> dict[str, Any]:
    lock = tomllib.loads((ROOT / UV_LOCK_FILE).read_text(encoding="utf-8"))
    packages: list[dict[str, Any]] = lock["package"]
    for package in packages:
        if package["name"] == package_name:
            return package
    raise AssertionError(f"{package_name} not found in {UV_LOCK_FILE}")


def _locked_dependency_names(package_name: str) -> set[str]:
    package = _locked_package(package_name)
    return {dependency["name"] for dependency in package.get("dependencies", [])}


def _locked_extra_requirements(
    locked_extras: dict[str, list[dict[str, Any]]], extra_name: str
) -> dict[str, set[str]]:
    """Each package in an extra's flattened lock list, with the extras it requests.

    A package can appear once bare and once with extras (`httpx`, `httpx[http2]`);
    both mean one install, so the extras are merged.
    """
    requirements: dict[str, set[str]] = {}
    for dependency in locked_extras[extra_name]:
        requirements.setdefault(dependency["name"], set()).update(dependency.get("extra", []))
    return requirements


def test_all_extra_does_not_require_torch_on_macos_x86_64() -> None:
    """Keep `headroom-ai[all]` resolvable where PyTorch publishes no wheel."""

    pyproject = tomllib.loads((ROOT / PYPROJECT_FILE).read_text(encoding="utf-8"))
    optional_deps = pyproject["project"]["optional-dependencies"]
    macos_x86_64_environment = default_environment()
    macos_x86_64_environment.update(
        {
            SYS_PLATFORM_MARKER: MACOS_X86_64_SYS_PLATFORM,
            PLATFORM_MACHINE_MARKER: MACOS_X86_64_PLATFORM_MACHINE,
        }
    )

    all_requirements = [Requirement(dep) for dep in optional_deps[ALL_EXTRA]]
    all_nested_extras = {
        extra
        for requirement in all_requirements
        if requirement.name == HEADROOM_PACKAGE_NAME
        for extra in requirement.extras
    }
    assert not {"ml", "memory", "evals", "voice"} & all_nested_extras

    torch_deps = [
        dep
        for extra_name in ("ml", "voice")
        for dep in optional_deps[extra_name]
        if dep.startswith(TORCH_PACKAGE_NAME)
    ]
    selected_all_dependency_names = _selected_dependency_names_for_extra(
        optional_deps,
        ALL_EXTRA,
        macos_x86_64_environment,
    )
    locked_torch_transitive_dependency_names = {
        package_name
        for package_name in TORCH_TRANSITIVE_PACKAGE_NAMES
        if TORCH_PACKAGE_NAME in _locked_dependency_names(package_name)
    }

    assert torch_deps
    assert all(MACOS_X86_64_TORCH_GUARD in dep for dep in torch_deps)
    assert locked_torch_transitive_dependency_names
    assert TORCH_PACKAGE_NAME not in selected_all_dependency_names
    assert selected_all_dependency_names.isdisjoint(locked_torch_transitive_dependency_names)

    # Explicit opt-ins still honor the Intel macOS wheel guard independently
    # of whether the default installation profile requests them.
    for extra_name in ("ml", "voice"):
        selected = _selected_dependency_names_for_extra(
            optional_deps, extra_name, macos_x86_64_environment
        )
        assert TORCH_PACKAGE_NAME not in selected
        assert selected.isdisjoint(locked_torch_transitive_dependency_names)

        linux_environment = dict(macos_x86_64_environment, sys_platform="linux")
        assert TORCH_PACKAGE_NAME in _selected_dependency_names_for_extra(
            optional_deps, extra_name, linux_environment
        )


def test_proxy_extra_includes_orjson_for_litellm_backends() -> None:
    """`headroom-ai[all]` must ship orjson for LiteLLM provider backends (GH #2056)."""

    pyproject = tomllib.loads((ROOT / PYPROJECT_FILE).read_text(encoding="utf-8"))
    optional_deps = pyproject["project"]["optional-dependencies"]
    environment = default_environment()

    selected_proxy_dependency_names = _selected_dependency_names_for_extra(
        optional_deps,
        PROXY_EXTRA,
        environment,
    )
    selected_all_dependency_names = _selected_dependency_names_for_extra(
        optional_deps,
        ALL_EXTRA,
        environment,
    )

    assert ORJSON_PACKAGE_NAME in selected_proxy_dependency_names
    assert ORJSON_PACKAGE_NAME in selected_all_dependency_names


def test_proxy_extra_includes_socks_support_for_websocket_proxy_discovery() -> None:
    """A system SOCKS proxy must not disable Codex Responses WebSockets."""

    pyproject = tomllib.loads((ROOT / PYPROJECT_FILE).read_text(encoding="utf-8"))
    optional_deps = pyproject["project"]["optional-dependencies"]
    environment = default_environment()

    selected_proxy_dependency_names = _selected_dependency_names_for_extra(
        optional_deps,
        PROXY_EXTRA,
        environment,
    )

    assert PYTHON_SOCKS_PACKAGE_NAME in selected_proxy_dependency_names


def test_locked_composite_extras_include_their_nested_extras() -> None:
    """`uv sync --extra lean` must install everything `--extra proxy` does.

    uv installs an extra from its own flattened list in uv.lock, and neither
    `uv lock --check` nor a plain `uv lock` re-derives that list when a nested
    extra gains a dependency: python-socks reached `proxy` but not `lean`.
    """

    package = _locked_package(HEADROOM_PACKAGE_NAME)
    locked_extras = package["optional-dependencies"]
    composites = {
        match.group(1): requirement["extras"]
        for requirement in package["metadata"]["requires-dist"]
        if requirement["name"] == HEADROOM_PACKAGE_NAME
        and (match := EXTRA_MARKER.match(requirement.get("marker", "")))
    }
    missing: dict[str, list[str]] = {}
    for composite, nested in composites.items():
        locked = _locked_extra_requirements(locked_extras, composite)
        for nested_extra in nested:
            for name, extras in _locked_extra_requirements(locked_extras, nested_extra).items():
                if name not in locked or not extras <= locked[name]:
                    missing.setdefault(composite, []).append(f"{name} (from {nested_extra})")

    assert LEAN_EXTRA in composites
    assert missing == {}
