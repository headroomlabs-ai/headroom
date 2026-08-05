from __future__ import annotations

from pathlib import Path

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
PROXY_EXTRA = "proxy"
UV_LOCK_FILE = "uv.lock"


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


def _locked_dependency_names(package_name: str) -> set[str]:
    lock = tomllib.loads((ROOT / UV_LOCK_FILE).read_text(encoding="utf-8"))
    for package in lock["package"]:
        if package["name"] == package_name:
            return {dependency["name"] for dependency in package.get("dependencies", [])}
    raise AssertionError(f"{package_name} not found in {UV_LOCK_FILE}")


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

    assert "ml" in optional_deps[ALL_EXTRA][0]
    assert "voice" in optional_deps[ALL_EXTRA][0]

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
