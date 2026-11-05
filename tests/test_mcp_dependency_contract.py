"""Keep the installed MCP SDK compatible with Headroom's live server API."""

from __future__ import annotations

from pathlib import Path

import tomllib
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def _mcp_requirement(dependencies: list[str]) -> Requirement:
    matches = [Requirement(value) for value in dependencies if Requirement(value).name == "mcp"]
    assert len(matches) == 1, "expected exactly one mcp requirement"
    return matches[0]


def test_all_shipping_mcp_extras_require_the_ported_sdk_v2() -> None:
    """The shipped servers use SDK 2 constructor callbacks, not SDK 1 decorators."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    optional = project["optional-dependencies"]

    for extra in ("proxy", "mcp"):
        requirement = _mcp_requirement(optional[extra])
        assert requirement.specifier.contains("2.2.0"), (
            f"the {extra!r} extra must admit the SDK used by the ported servers"
        )
        assert not requirement.specifier.contains("1.30.0"), (
            f"the {extra!r} extra admits SDK 1 without the required constructor callbacks"
        )
        # CVE-2026-59951 and CVE-2026-116684/116685/116686 affect 2.0.0-2.1.x.
        assert not requirement.specifier.contains("2.1.1"), (
            f"the {extra!r} extra admits an MCP SDK 2.x release before the 2.2.0 security fixes"
        )
        assert not requirement.specifier.contains("3.0.0"), (
            f"the {extra!r} extra admits an untested MCP SDK major version (see #2977)"
        )
